"""TriLevelDataset: per-article training examples with three aligned targets.

Returns for each row:
    input_ids         LongTensor [L]      tokenised title+body
    attention_mask    LongTensor [L]
    macro_targets     FloatTensor [8]     multi-hot BCE targets
    industry_targets  FloatTensor [11]    multi-hot BCE targets
    entity_targets    LongTensor [L]      BIO tag ids, -100 on specials
                                          and subword continuations

Entity alignment: locate each Gemma-emitted entity surface string inside
the article via word-boundary regex (first occurrence wins, single-letter
ticker safe), map the char span to tokens through offset_mapping, paint
B-/I- tags. Subword continuations of non-entity words become -100 so CE
only scores first pieces. Empty-entity rows are kept as all-O (they still
train the macro/industry heads).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_prep.filters import (
    ENTITY_TAG_TO_ID,
    INDUSTRY_LABEL_TO_ID,
    MACRO_LABEL_TO_ID,
    load_and_filter_master,
)

IGNORE_INDEX = -100
_VALID_ENTITY_TYPES = {"ORG", "PER", "TICKER", "PRODUCT", "LOC"}


def _build_text(title, text) -> str:
    title = "" if title is None or (isinstance(title, float) and pd.isna(title)) else str(title)
    text = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    return f"{title}\n\n{text}"


def _find_span(haystack_lower: str, needle: str) -> tuple[int, int] | None:
    """Word-boundary, case-insensitive first-match. Returns (start, end) or None."""
    needle = needle.strip()
    if not needle:
        return None
    try:
        pat = re.compile(r"\b" + re.escape(needle.lower()) + r"\b")
    except re.error:
        return None
    m = pat.search(haystack_lower)
    return (m.start(), m.end()) if m else None


class TriLevelDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        max_length: int = 512,
        text_column_title: str = "title",
        text_column_body: str = "text",
    ):
        self.df = load_and_filter_master(Path(csv_path))
        # Graceful fallback if body is missing (e.g. the title-only master).
        if text_column_body not in self.df.columns:
            self.df[text_column_body] = ""
        if text_column_title not in self.df.columns:
            self.df[text_column_title] = ""

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_col = text_column_title
        self.body_col = text_column_body
        self.n_macro = len(MACRO_LABEL_TO_ID)
        self.n_industry = len(INDUSTRY_LABEL_TO_ID)

    def __len__(self) -> int:
        return len(self.df)

    def _encode_entities(
        self,
        text: str,
        entity_items: list,
        offset_mapping: list[tuple[int, int]],
        special_tokens_mask: list[int],
    ) -> torch.Tensor:
        """Paint BIO tags onto the token sequence. See module docstring."""
        L = len(offset_mapping)
        tags = [IGNORE_INDEX] * L
        for i in range(L):
            if special_tokens_mask[i] == 0:
                tags[i] = ENTITY_TAG_TO_ID["O"]

        # Resolve entity char spans (empty list => all-O after continuation fix).
        spans: list[tuple[int, int, str]] = []
        if entity_items:
            text_lower = text.lower()
            for it in entity_items:
                if not isinstance(it, dict):
                    continue
                name, typ = it.get("name"), it.get("type")
                if not name or typ not in _VALID_ENTITY_TYPES:
                    continue
                sp = _find_span(text_lower, str(name))
                if sp is not None:
                    spans.append((sp[0], sp[1], typ))
            spans.sort(key=lambda x: (x[0], -x[1]))
            # Drop overlaps: earlier entity wins.
            non_overlap: list[tuple[int, int, str]] = []
            last_end = -1
            for s, e, t in spans:
                if s >= last_end:
                    non_overlap.append((s, e, t))
                    last_end = e
            spans = non_overlap

        # Single pass: paint BIO on hits, mark non-hit continuations as ignore.
        prev_end = None
        prev_special = True
        for i, (ts, te) in enumerate(offset_mapping):
            is_special = special_tokens_mask[i] == 1
            if is_special:
                prev_end, prev_special = te, True
                continue

            hit = None
            for s, e, t in spans:
                if ts >= s and ts < e:
                    hit = (s, e, t)
                    break
                if te <= s:
                    break

            is_continuation = (
                prev_end is not None and ts == prev_end and not prev_special
            )

            if hit is not None:
                s, e, t = hit
                prev_in_same = False
                if i > 0 and special_tokens_mask[i - 1] == 0:
                    ps, _ = offset_mapping[i - 1]
                    if ps >= s and ps < e:
                        prev_in_same = True
                tags[i] = ENTITY_TAG_TO_ID[("I-" if prev_in_same else "B-") + t]
            elif is_continuation:
                tags[i] = IGNORE_INDEX

            prev_end, prev_special = te, False

        return torch.tensor(tags, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = _build_text(row[self.title_col], row[self.body_col])

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors=None,
        )

        macro_targets = torch.zeros(self.n_macro, dtype=torch.float32)
        for lbl in row["macro_list"]:
            j = MACRO_LABEL_TO_ID.get(str(lbl))
            if j is not None:
                macro_targets[j] = 1.0

        industry_targets = torch.zeros(self.n_industry, dtype=torch.float32)
        for lbl in row["industry_list"]:
            j = INDUSTRY_LABEL_TO_ID.get(str(lbl))
            if j is not None:
                industry_targets[j] = 1.0

        entity_targets = self._encode_entities(
            text=text,
            entity_items=row["entity_list"],
            offset_mapping=enc["offset_mapping"],
            special_tokens_mask=enc["special_tokens_mask"],
        )

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "macro_targets": macro_targets,
            "industry_targets": industry_targets,
            "entity_targets": entity_targets,
        }
