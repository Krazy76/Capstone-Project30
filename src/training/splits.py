"""Deterministic train/val/test split persisted to disk.

Splits are computed once and saved as a JSON of row indices keyed by
split name. Subsequent runs (and active-learning rounds) reload the
exact same indices so the val/test set is frozen and never contaminated
by AL-selected unlabelled samples.

Index space: positions in the FILTERED dataframe returned by
filters.load_and_filter_master, not the raw csv row numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_SPLIT_PATH = Path(
    r"E:\Coding\Python\Capstone-Project30\data\splits\trilevel_splits.json"
)


def make_splits(
    n_rows: int,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    n_test = int(round(n_rows * test_frac))
    n_val = int(round(n_rows * val_frac))
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return {
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in test),
        "seed": seed,
        "n_rows": n_rows,
    }


def load_or_create_splits(
    n_rows: int,
    path: str | Path = DEFAULT_SPLIT_PATH,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[int]]:
    path = Path(path)
    if path.exists():
        splits = json.loads(path.read_text())
        if splits.get("n_rows") == n_rows and splits.get("seed") == seed:
            return splits
        # Corpus changed (AL added rows). Append-only: keep existing val/test,
        # extend train with the new tail. Caller must ensure new rows are
        # appended at the end of the filtered dataframe.
        old_n = int(splits["n_rows"])
        if n_rows > old_n:
            new_train = sorted(set(splits["train"]) | set(range(old_n, n_rows)))
            splits["train"] = new_train
            splits["n_rows"] = n_rows
            path.write_text(json.dumps(splits, indent=2))
            return splits
        # Otherwise force regeneration.
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = make_splits(n_rows, val_frac, test_frac, seed)
    path.write_text(json.dumps(splits, indent=2))
    return splits
