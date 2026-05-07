"""One-shot inference: title+text -> {macro, industry, entity_spans}.

Loads a trained checkpoint, tokenises the article, runs a single forward
pass, and decodes:
  - macro:    sigmoid(logits) >= threshold over the FRED-MD 8 classes
  - industry: sigmoid(logits) >= threshold over the GICS 11 classes
  - entity:   argmax BIO -> spans -> recovered surface strings via
              offset_mapping back to the original text

Returns a JSON-serialisable dict. Intended for downstream consumers
(API server, batch labeller for new corpora, etc.).

Run:
  python -m src.serve.predict --ckpt artifacts/runs/<run>/best.pt \
      --title "Fed hikes rates by 25bps" \
      --text  "The Federal Reserve raised the benchmark rate ..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.active_learning.score import load_model_from_ckpt
from src.data_prep.dataset import _build_text
from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
from src.eval.span_f1 import bio_to_spans

IGNORE_INDEX = -100


@torch.no_grad()
def predict(
    ckpt_path: Path,
    title: str,
    text: str,
    macro_threshold: float = 0.5,
    industry_threshold: float = 0.5,
    max_length: int | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = load_model_from_ckpt(ckpt_path, device)
    max_length = max_length or cfg["max_length"]

    full = _build_text(title, text)
    enc = tok(
        full,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    out = model(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
    )

    macro_probs = torch.sigmoid(out["macro_logits"][0]).cpu().tolist()
    industry_probs = torch.sigmoid(out["industry_logits"][0]).cpu().tolist()
    macro_labels = [
        {"label": MACRO_LABELS[i], "prob": p}
        for i, p in enumerate(macro_probs) if p >= macro_threshold
    ]
    industry_labels = [
        {"label": INDUSTRY_LABELS[i], "prob": p}
        for i, p in enumerate(industry_probs) if p >= industry_threshold
    ]

    # Decode entity BIO -> token spans -> char spans -> surface strings.
    pred_ids = out["entity_logits"][0].argmax(dim=-1).cpu().tolist()
    special = enc["special_tokens_mask"][0].tolist()
    offsets = enc["offset_mapping"][0].tolist()

    tag_strs: list[str] = []
    keep_mask: list[bool] = []
    for i in range(len(pred_ids)):
        if special[i] == 1:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        tag_strs.append(ENTITY_TAGS[pred_ids[i]])

    spans = bio_to_spans(tag_strs)
    # Map each span (in the kept-token space) back to original char offsets.
    kept_to_full = [i for i, k in enumerate(keep_mask) if k]
    entities = []
    for s, e, t in spans:
        full_start = kept_to_full[s]
        full_end = kept_to_full[e - 1]
        char_start = offsets[full_start][0]
        char_end = offsets[full_end][1]
        if char_end <= char_start:
            continue
        surface = full[char_start:char_end]
        entities.append({
            "type": t,
            "name": surface,
            "char_start": char_start,
            "char_end": char_end,
        })

    return {
        "ckpt": str(ckpt_path),
        "macro": macro_labels,
        "industry": industry_labels,
        "entity": entities,
        "macro_probs": dict(zip(MACRO_LABELS, macro_probs)),
        "industry_probs": dict(zip(INDUSTRY_LABELS, industry_probs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--text", default="")
    p.add_argument("--macro-threshold", type=float, default=0.5)
    p.add_argument("--industry-threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="Optional output JSON file")
    args = p.parse_args()
    result = predict(
        ckpt_path=Path(args.ckpt),
        title=args.title,
        text=args.text,
        macro_threshold=args.macro_threshold,
        industry_threshold=args.industry_threshold,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
