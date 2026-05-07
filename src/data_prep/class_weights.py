"""Compute per-class pos_weight tensors for the Macro and Industry BCE heads.

pos_weight_c = (N - n_pos_c) / max(n_pos_c, 1). Rebalances the positive
term of each class against the negative term so BCE does not collapse
to all-zeros under heavy imbalance. A soft cap is applied later in
losses.py (not here) so the raw counts stay faithful on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.data_prep.filters import INDUSTRY_LABELS, MACRO_LABELS, load_and_filter_master

BASE = Path(r"E:\Coding\Python\Capstone-Project30")
MASTER_CSV = BASE / "data" / "silver_dataset_master_with_text.csv"
OUT_DIR = BASE / "data" / "class_weights"


def compute() -> dict:
    print(f"Reading {MASTER_CSV}")
    n_raw = len(pd.read_csv(MASTER_CSV, low_memory=False, usecols=["is_financial"]))
    print(f"  raw rows: {n_raw}")
    df = load_and_filter_master(MASTER_CSV)
    N = len(df)
    print(f"  useful training rows:   {N}")

    macro_counts = {lbl: 0 for lbl in MACRO_LABELS}
    industry_counts = {lbl: 0 for lbl in INDUSTRY_LABELS}
    unknown_macro: dict[str, int] = {}
    unknown_industry: dict[str, int] = {}

    for labels in df["macro_list"]:
        for lbl in labels:
            if lbl in macro_counts:
                macro_counts[lbl] += 1
            else:
                unknown_macro[lbl] = unknown_macro.get(lbl, 0) + 1
    for labels in df["industry_list"]:
        for lbl in labels:
            if lbl in industry_counts:
                industry_counts[lbl] += 1
            else:
                unknown_industry[lbl] = unknown_industry.get(lbl, 0) + 1

    def pos_weight(counts: dict[str, int], order: list[str]) -> torch.Tensor:
        return torch.tensor(
            [(N - counts[lbl]) / max(counts[lbl], 1) for lbl in order],
            dtype=torch.float32,
        )

    macro_w = pos_weight(macro_counts, MACRO_LABELS)
    industry_w = pos_weight(industry_counts, INDUSTRY_LABELS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(macro_w, OUT_DIR / "macro_pos_weight.pt")
    torch.save(industry_w, OUT_DIR / "industry_pos_weight.pt")

    stats = {
        "n_total_rows_raw": n_raw,
        "n_useful_rows": N,
        "macro_order": MACRO_LABELS,
        "macro_counts": macro_counts,
        "macro_pos_weight": macro_w.tolist(),
        "industry_order": INDUSTRY_LABELS,
        "industry_counts": industry_counts,
        "industry_pos_weight": industry_w.tolist(),
        "unknown_macro_labels": unknown_macro,
        "unknown_industry_labels": unknown_industry,
    }
    with open(OUT_DIR / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Macro pos_weight ===")
    for lbl, c, w in zip(MACRO_LABELS, macro_counts.values(), macro_w.tolist()):
        print(f"  {lbl:40s}  n_pos={c:5d}  w={w:8.2f}")
    print("\n=== Industry pos_weight ===")
    for lbl, c, w in zip(INDUSTRY_LABELS, industry_counts.values(), industry_w.tolist()):
        print(f"  {lbl:40s}  n_pos={c:5d}  w={w:8.2f}")
    if unknown_macro:
        print(f"\n[warn] unknown macro labels: {unknown_macro}")
    if unknown_industry:
        print(f"[warn] unknown industry labels: {unknown_industry}")
    print(f"\nWrote: {OUT_DIR}")
    return stats


if __name__ == "__main__":
    compute()
