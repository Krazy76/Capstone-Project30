"""Shared silver-corpus filtering, parsing, and label-order constants.

Single source of truth so class_weights.py and dataset.py cannot drift.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

# Import canonical label orders directly from the labeller so there is
# exactly one place they are defined.
BASE = Path(r"E:\Coding\Python\Capstone-Project30")
sys.path.insert(0, str(BASE / "data"))
from llm_auto_labeller import MACRO_RULES, INDUSTRY_RULES  # noqa: E402

MACRO_LABELS: list[str] = list(MACRO_RULES.keys())            # FRED-MD 8
INDUSTRY_LABELS: list[str] = list(INDUSTRY_RULES.keys())      # GICS 11

# Entity taxonomy locked from a full scan of the silver set: exactly
# 5 types, no long tail. -> 11 BIO tags including O.
ENTITY_TYPES: list[str] = ["ORG", "PER", "TICKER", "PRODUCT", "LOC"]
ENTITY_TAGS: list[str] = ["O"]
for _t in ENTITY_TYPES:
    ENTITY_TAGS.append(f"B-{_t}")
    ENTITY_TAGS.append(f"I-{_t}")

ENTITY_TAG_TO_ID: dict[str, int] = {t: i for i, t in enumerate(ENTITY_TAGS)}
MACRO_LABEL_TO_ID: dict[str, int] = {l: i for i, l in enumerate(MACRO_LABELS)}
INDUSTRY_LABEL_TO_ID: dict[str, int] = {l: i for i, l in enumerate(INDUSTRY_LABELS)}


def parse_list_cell(cell) -> list:
    """Parse a CSV cell that stores a list as a JSON or Python-literal string."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return v if isinstance(v, list) else []


def is_financial_true(x) -> bool:
    """Truthiness check for the is_financial column (handles bool / str / NaN)."""
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_and_filter_master(
    csv_path: Path,
    label_version: str | None = "v2",
) -> pd.DataFrame:
    """Load silver_dataset_master*.csv and apply the locked filter rule.

    Keeps rows where:
      - is_financial == True
      - at least one of (macro, industry, entity) is non-empty
      - label_version == label_version arg (default "v2")

    Pass label_version=None to include all rows regardless of version
    (e.g. for backwards-compatible runs or when the column doesn't exist).

    Adds parsed list columns: macro_list, industry_list, entity_list.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter by label version when the column exists and a version is requested.
    if label_version is not None and "label_version" in df.columns:
        df = df[df["label_version"] == label_version].reset_index(drop=True)
    elif label_version is not None and "label_version" not in df.columns:
        # Column absent means this is a legacy file (pre-versioning) -- keep all.
        pass

    df = df[df["is_financial"].apply(is_financial_true)].reset_index(drop=True)

    df["macro_list"] = df["macro"].apply(parse_list_cell)
    df["industry_list"] = df["industry"].apply(parse_list_cell)
    df["entity_list"] = df["entity"].apply(parse_list_cell)

    all_empty = (
        df["macro_list"].map(len).eq(0)
        & df["industry_list"].map(len).eq(0)
        & df["entity_list"].map(len).eq(0)
    )
    return df[~all_empty].reset_index(drop=True)
