"""Count entity-type support (B- tag occurrences) per split.

Justifies the per-type entity F1 gap from the §6 evaluation: types with
near-zero F1 (LOC, PRODUCT, TICKER) are precisely the types with thin
support in the silver labels.

Counts both at the row level (how many articles mention the type) and
at the span level (how many entity spans of the type total).

Output: artifacts/eval/entity_support.json + console table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data_prep.filters import ENTITY_TYPES, load_and_filter_master
from src.training.splits import load_or_create_splits

DEFAULT_MASTER = Path(
    r"E:\Coding\Python\Capstone-Project30\data\silver_dataset_master_with_text.csv"
)
DEFAULT_OUT = Path(r"E:\Coding\Python\Capstone-Project30\artifacts\eval\entity_support.json")


def _count_split(df_rows, split_indices: list[int]) -> dict:
    span_counts: Counter = Counter()
    row_with_type_counts: Counter = Counter()
    rows_with_any = 0
    n = 0
    for i in split_indices:
        n += 1
        row = df_rows.iloc[i]
        ent = row["entity_list"] if isinstance(row["entity_list"], list) else []
        if not ent:
            continue
        seen_types_in_row = set()
        rows_with_any += 1
        for it in ent:
            if not isinstance(it, dict):
                continue
            t = (it.get("type") or "").strip().upper()
            if t in ENTITY_TYPES:
                span_counts[t] += 1
                seen_types_in_row.add(t)
        for t in seen_types_in_row:
            row_with_type_counts[t] += 1
    return {
        "n_rows": n,
        "rows_with_any_entity": rows_with_any,
        "span_counts": dict(span_counts),
        "row_with_type_counts": dict(row_with_type_counts),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    df = load_and_filter_master(Path(args.master_csv))
    splits = load_or_create_splits(n_rows=len(df))

    out = {}
    for split in ("train", "val", "test"):
        out[split] = _count_split(df, splits[split])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n== Entity-type support (silver labels) ==\n")
    header = f"{'split':<6} {'n_rows':>7} {'rows_w_ent':>11}  " + " ".join(
        f"{t:>9}" for t in ENTITY_TYPES
    )
    print(header)
    print("-" * len(header))
    for split in ("train", "val", "test"):
        s = out[split]
        spans_row = " ".join(f"{s['span_counts'].get(t, 0):>9d}" for t in ENTITY_TYPES)
        print(
            f"{split:<6} {s['n_rows']:>7d} {s['rows_with_any_entity']:>11d}  {spans_row}"
        )

    print(f"\n(rows containing the type, not span counts):")
    print(header)
    for split in ("train", "val", "test"):
        s = out[split]
        rows_row = " ".join(f"{s['row_with_type_counts'].get(t, 0):>9d}" for t in ENTITY_TYPES)
        print(
            f"{split:<6} {s['n_rows']:>7d} {s['rows_with_any_entity']:>11d}  {rows_row}"
        )

    # Per-class summary as ratios (for the report).
    train = out["train"]
    total_spans = sum(train["span_counts"].values()) or 1
    print(f"\n== Train split - share of total {total_spans} spans ==")
    for t in ENTITY_TYPES:
        c = train["span_counts"].get(t, 0)
        print(f"  {t:<8} {c:>6d}  {100*c/total_spans:>5.1f}%")

    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
