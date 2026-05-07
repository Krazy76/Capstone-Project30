"""Re-label AL candidates via Gemma-on-Ollama and write a NEW csv.

The source master csv is **never modified**. Each AL round reads from
an input csv and writes a new csv with rescued rows updated:

  round 1: reads silver_dataset_master_with_text.csv
           writes data/active_learning/round_1/master_with_text_post.csv
  round 2: reads round_1/master_with_text_post.csv
           writes round_2/master_with_text_post.csv
  ...

This gives a clean lineage: every round's output csv is a complete,
immutable snapshot of the corpus state after that round. Training on
any round = point --master-csv at that round's post.csv.

For each candidate row we:
  1. Read (title, text) from the input csv (filtered index space matches
     the saved splits, so we resolve via load_and_filter_master).
  2. Call llm_auto_labeller.call_ollama with the same SYSTEM_PROMPT and
     temperature 0.1 used at first labelling -> annotator drift = zero.
  3. Validate the response.
  4. If macro_list is now non-empty, write the new macro/industry/entity
     strings to the row in the OUTPUT csv. Industry and entity are only
     overwritten when the new pass returns non-empty (never erase).
  5. Append row_idx to relabelled_ids.json so future rounds skip it.

Output: data/active_learning/round_{N}/relabel_log.jsonl
  one row: {row_idx, before_macro, after_macro, after_industry, after_entity_n, rescued}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Reuse the production labeller verbatim (single source of truth for prompt + temp).
BASE = Path(r"E:\Coding\Python\Capstone-Project30")
sys.path.insert(0, str(BASE / "data"))
from llm_auto_labeller import call_ollama, validate  # noqa: E402

from src.data_prep.filters import load_and_filter_master, parse_list_cell

DEFAULT_MASTER = BASE / "data" / "silver_dataset_master_with_text.csv"
DEFAULT_AL_ROOT = BASE / "data" / "active_learning"
RELABELLED_IDS_PATH = DEFAULT_AL_ROOT / "relabelled_ids.json"


def load_candidates(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_relabelled_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def save_relabelled_ids(path: Path, ids: set[int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids)))


def _filtered_to_master_index(
    master_csv: Path,
    label_version: str | None = "v2",
) -> list[int]:
    """Map filtered-row index -> raw csv row index.

    Mirrors the rule inside load_and_filter_master (is_financial=True AND
    not all-empty AND label_version match) so this index map stays aligned
    with the training-corpus row order.
    """
    raw = pd.read_csv(master_csv, low_memory=False)

    def _is_fin(x):
        if isinstance(x, bool):
            return x
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"true", "1", "yes"}

    # Apply label_version filter first (mirrors load_and_filter_master).
    if label_version is not None and "label_version" in raw.columns:
        version_mask = raw["label_version"] == label_version
    else:
        version_mask = pd.Series([True] * len(raw))

    mask_fin = raw["is_financial"].apply(_is_fin)
    macro_l = raw["macro"].apply(parse_list_cell)
    industry_l = raw["industry"].apply(parse_list_cell)
    entity_l = raw["entity"].apply(parse_list_cell)
    not_empty = (macro_l.map(len) > 0) | (industry_l.map(len) > 0) | (entity_l.map(len) > 0)
    keep_mask = version_mask & mask_fin & not_empty
    return raw.index[keep_mask].tolist()


def relabel(
    candidates_path: Path,
    log_path: Path,
    input_csv: Path = DEFAULT_MASTER,
    output_csv: Path | None = None,
    relabelled_ids_path: Path = RELABELLED_IDS_PATH,
    round_n: int = 0,
    sleep_between: float = 0.1,
) -> dict:
    """Re-label candidates from input_csv, write updated rows to output_csv.

    The input csv is never modified. If output_csv is None, defaults to
    data/active_learning/round_{round_n}/master_with_text_post.csv.
    """
    candidates = load_candidates(candidates_path)
    if not candidates:
        print("[relabel] no candidates")
        return {"n_candidates": 0, "n_rescued": 0, "n_failed": 0, "output_csv": None}

    if output_csv is None:
        output_csv = DEFAULT_AL_ROOT / f"round_{round_n}" / "master_with_text_post.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv, low_memory=False)
    f2m = _filtered_to_master_index(input_csv)
    if len(f2m) != len(load_and_filter_master(input_csv)):
        raise RuntimeError("filtered->master index map size mismatch")

    relabelled = load_relabelled_ids(relabelled_ids_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")

    n_rescued = 0
    n_failed = 0
    t0 = time.time()
    for i, cand in enumerate(candidates, start=1):
        f_idx = cand["row_idx"]
        if f_idx in relabelled:
            continue
        m_idx = f2m[f_idx]
        title = raw.at[m_idx, "title"] if "title" in raw.columns else ""
        text = raw.at[m_idx, "text"] if "text" in raw.columns else ""
        if pd.isna(title):
            title = ""
        if pd.isna(text):
            text = ""

        before_macro = parse_list_cell(raw.at[m_idx, "macro"])

        obj = call_ollama(str(title), str(text))
        v = validate(obj)
        rescued = len(v["macro"]) > 0

        if v.get("is_financial") is False:
            # Gemma now thinks not financial; do not overwrite labels but record.
            n_failed += 1
            log_f.write(json.dumps({
                "row_idx": f_idx, "rescued": False, "reason": "is_financial=False",
            }) + "\n")
            relabelled.add(f_idx)
            continue

        if rescued:
            raw.at[m_idx, "macro"] = json.dumps(v["macro"])
            # Industry/entity: only overwrite if the new call produced something
            # non-empty (do not erase existing labels).
            if v["industry"]:
                raw.at[m_idx, "industry"] = json.dumps(v["industry"])
            if v["entity"]:
                raw.at[m_idx, "entity"] = json.dumps(v["entity"])
            n_rescued += 1

        relabelled.add(f_idx)
        log_f.write(json.dumps({
            "row_idx": f_idx,
            "before_macro": before_macro,
            "after_macro": v["macro"],
            "after_industry": v["industry"],
            "after_entity_n": len(v["entity"]),
            "rescued": rescued,
        }) + "\n")

        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"[relabel] {i}/{len(candidates)}  rescued={n_rescued}  ({elapsed:.0f}s)")
        if sleep_between:
            time.sleep(sleep_between)

    log_f.close()
    save_relabelled_ids(relabelled_ids_path, relabelled)
    raw.to_csv(output_csv, index=False)
    elapsed = time.time() - t0
    print(
        f"[relabel] done. n={len(candidates)} rescued={n_rescued} failed={n_failed} "
        f"time={elapsed:.0f}s -> {output_csv.name}"
    )
    return {
        "n_candidates": len(candidates),
        "n_rescued": n_rescued,
        "n_failed": n_failed,
        "output_csv": str(output_csv),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument(
        "--input-csv",
        default=None,
        help="Source csv to read from (read-only). Defaults to master, or to "
        "round_{N-1}/master_with_text_post.csv if that exists.",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Where to write the rescued snapshot. Defaults to "
        "round_{N}/master_with_text_post.csv.",
    )
    p.add_argument("--al-root", default=str(DEFAULT_AL_ROOT))
    p.add_argument("--sleep", type=float, default=0.1)
    args = p.parse_args()

    al_root = Path(args.al_root)
    cand = al_root / f"round_{args.round}" / "candidates.jsonl"
    log = al_root / f"round_{args.round}" / "relabel_log.jsonl"

    if args.input_csv:
        input_csv = Path(args.input_csv)
    else:
        prev = al_root / f"round_{args.round - 1}" / "master_with_text_post.csv"
        input_csv = prev if prev.exists() else DEFAULT_MASTER
    output_csv = Path(args.output_csv) if args.output_csv else None

    relabel(
        candidates_path=cand,
        log_path=log,
        input_csv=input_csv,
        output_csv=output_csv,
        round_n=args.round,
        sleep_between=args.sleep,
    )


if __name__ == "__main__":
    main()
