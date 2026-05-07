"""Span-level entity F1 for the §6 report.

Token-level F1 (in `src/training/metrics.py`) is fine for in-loop
monitoring but rewards partial credit (scoring "Apple Inc." as half-right
if you tag "Apple" but not "Inc."). The capstone needs the conventional
NER metric, which is **strict span match**: a prediction is correct only
if both the boundary AND the type match the gold span exactly.

We piggyback on seqeval, the standard library for this. If seqeval is not
installed, we fall back to a dependency-free implementation of the same
metric.

Output: artifacts/eval/<ckpt_stem>_span_f1.json
  {
    "split": "val|test",
    "n_examples": int,
    "overall": {precision, recall, f1, support},
    "per_type": {ORG: {...}, PER: {...}, ...}
  }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.active_learning.score import load_model_from_ckpt
from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import ENTITY_TAGS
from src.training.splits import load_or_create_splits

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = _PROJECT_ROOT / "data" / "silver_dataset_master_with_text.csv"
DEFAULT_OUT_DIR = _PROJECT_ROOT / "artifacts" / "eval"
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Span extraction (BIO -> list of (start, end_exclusive, type))
# --------------------------------------------------------------------------
def bio_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    cur_type = None
    cur_start = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
            cur_type = tag[2:]
            cur_start = i
        elif tag.startswith("I-") and cur_type == tag[2:]:
            continue
        else:
            if cur_type is not None:
                spans.append((cur_start, i, cur_type))
                cur_type, cur_start = None, None
            if tag.startswith("I-"):
                # Orphan I- (no matching B-): treat as B- to be lenient.
                cur_type = tag[2:]
                cur_start = i
    if cur_type is not None:
        spans.append((cur_start, len(tags), cur_type))
    return spans


# --------------------------------------------------------------------------
# Fallback metric (used if seqeval unavailable)
# --------------------------------------------------------------------------
def _strict_span_metrics(
    pred_seqs: list[list[str]], gold_seqs: list[list[str]]
) -> dict:
    overall_tp = overall_fp = overall_fn = 0
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn

    for p_tags, g_tags in zip(pred_seqs, gold_seqs):
        p_spans = set(bio_to_spans(p_tags))
        g_spans = set(bio_to_spans(g_tags))
        for s in p_spans & g_spans:
            overall_tp += 1
            per_type_counts[s[2]][0] += 1
        for s in p_spans - g_spans:
            overall_fp += 1
            per_type_counts[s[2]][1] += 1
        for s in g_spans - p_spans:
            overall_fn += 1
            per_type_counts[s[2]][2] += 1

    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    return {
        "overall": _prf(overall_tp, overall_fp, overall_fn),
        "per_type": {t: _prf(*c) for t, c in per_type_counts.items()},
    }


def compute_span_metrics(
    pred_seqs: list[list[str]], gold_seqs: list[list[str]]
) -> dict:
    try:
        from seqeval.metrics import classification_report  # type: ignore
        from seqeval.scheme import IOB2  # type: ignore
        report = classification_report(
            gold_seqs, pred_seqs, scheme=IOB2, mode="strict",
            output_dict=True, zero_division=0,
        )
        overall = {
            "precision": report["micro avg"]["precision"],
            "recall": report["micro avg"]["recall"],
            "f1": report["micro avg"]["f1-score"],
            "support": report["micro avg"]["support"],
        }
        per_type = {}
        for k, v in report.items():
            if k in {"micro avg", "macro avg", "weighted avg"}:
                continue
            per_type[k] = {
                "precision": v["precision"],
                "recall": v["recall"],
                "f1": v["f1-score"],
                "support": v["support"],
            }
        return {"overall": overall, "per_type": per_type, "backend": "seqeval"}
    except ImportError:
        out = _strict_span_metrics(pred_seqs, gold_seqs)
        out["backend"] = "fallback"
        return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
@torch.no_grad()
def run_span_eval(
    ckpt_path: Path,
    split: str = "val",
    master_csv: Path = DEFAULT_MASTER,
    batch_size: int = 16,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict:
    assert split in {"val", "test"}, f"unknown split {split}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = load_model_from_ckpt(ckpt_path, device)

    ds = TriLevelDataset(master_csv, tok, max_length=cfg["max_length"])
    splits = load_or_create_splits(n_rows=len(ds))
    indices = splits[split]
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False)

    pred_seqs: list[list[str]] = []
    gold_seqs: list[list[str]] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am)
        preds = out["entity_logits"].argmax(dim=-1).cpu()
        gold = batch["entity_targets"]
        for b in range(preds.size(0)):
            p_row, g_row = [], []
            for p, g in zip(preds[b].tolist(), gold[b].tolist()):
                if g == IGNORE_INDEX:
                    continue
                p_row.append(ENTITY_TAGS[p])
                g_row.append(ENTITY_TAGS[g])
            pred_seqs.append(p_row)
            gold_seqs.append(g_row)

    metrics = compute_span_metrics(pred_seqs, gold_seqs)
    out = {
        "ckpt": str(ckpt_path),
        "split": split,
        "n_examples": len(indices),
        **metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(ckpt_path).parent.name}_span_f1_{split}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[span_f1] {split} overall F1={metrics['overall']['f1']:.4f}  -> {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    run_span_eval(
        ckpt_path=Path(args.ckpt),
        split=args.split,
        master_csv=Path(args.master_csv),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
