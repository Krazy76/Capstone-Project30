"""Pick candidate rows for re-labelling from scored pool.

Strategy: **stratified per-class top-K**.
  For each macro class c, take the K pool rows with highest sigmoid prob
  for c (the model's strongest hunch that Gemma missed). Union across
  classes, dedup, drop already-relabelled ids.

This biases AL toward closing the macro coverage gap (the report's
finding-2 problem), instead of generic uncertainty which would pick
ambiguous rows the model already half-handles.

Output: data/active_learning/round_{N}/candidates.jsonl
  one row: {row_idx, top_class, top_prob, all_probs}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_prep.filters import MACRO_LABELS

DEFAULT_AL_ROOT = Path(r"E:\Coding\Python\Capstone-Project30\data\active_learning")
RELABELLED_IDS_PATH = DEFAULT_AL_ROOT / "relabelled_ids.json"


def load_scores(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_relabelled_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def mine(
    scores_path: Path,
    out_path: Path,
    relabelled_ids_path: Path = RELABELLED_IDS_PATH,
    k_per_class: int = 50,
    min_prob: float = 0.30,
) -> int:
    rows = load_scores(scores_path)
    skip = load_relabelled_ids(relabelled_ids_path)
    rows = [r for r in rows if r["row_idx"] not in skip]
    print(f"[mine] {len(rows)} pool rows after dropping {len(skip)} already-relabelled")

    selected: dict[int, dict] = {}  # row_idx -> meta
    for c, label in enumerate(MACRO_LABELS):
        ranked = sorted(rows, key=lambda r: r["macro_probs"][c], reverse=True)
        kept = 0
        for r in ranked:
            if r["macro_probs"][c] < min_prob:
                break  # remaining are below threshold
            row_idx = r["row_idx"]
            if row_idx in selected:
                # already taken under a stronger class signal
                continue
            selected[row_idx] = {
                "row_idx": row_idx,
                "top_class": label,
                "top_prob": r["macro_probs"][c],
                "all_probs": r["macro_probs"],
            }
            kept += 1
            if kept >= k_per_class:
                break
        print(f"[mine]   {label:20s}  picked {kept}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for meta in selected.values():
            f.write(json.dumps(meta) + "\n")
    print(f"[mine] {len(selected)} candidates -> {out_path}")
    return len(selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--k-per-class", type=int, default=50)
    p.add_argument("--min-prob", type=float, default=0.30)
    p.add_argument("--al-root", default=str(DEFAULT_AL_ROOT))
    args = p.parse_args()

    al_root = Path(args.al_root)
    scores = al_root / f"round_{args.round}" / "scores.jsonl"
    out = al_root / f"round_{args.round}" / "candidates.jsonl"
    mine(scores, out, k_per_class=args.k_per_class, min_prob=args.min_prob)


if __name__ == "__main__":
    main()
