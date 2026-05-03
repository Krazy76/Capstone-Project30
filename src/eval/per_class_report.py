"""Per-class P/R/F1 report for macro and industry heads.

Produces both:
  - JSON dump for downstream consumption / plots
  - Markdown table ready to paste into the §6 report

Output: artifacts/eval/<run_name>_per_class_<split>.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.active_learning.score import load_model_from_ckpt
from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import INDUSTRY_LABELS, MACRO_LABELS
from src.training.metrics import _binary_prf
from src.training.splits import load_or_create_splits

DEFAULT_MASTER = Path(
    r"E:\Coding\Python\Capstone-Project30\data\silver_dataset_master_with_text.csv"
)
DEFAULT_OUT_DIR = Path(r"E:\Coding\Python\Capstone-Project30\artifacts\eval")


def _table_md(label_names: list[str], prec, rec, f1, support) -> str:
    lines = ["| class | precision | recall | F1 | support |", "|---|---:|---:|---:|---:|"]
    for name, p, r, f, s in zip(label_names, prec.tolist(), rec.tolist(), f1.tolist(), support.tolist()):
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {f:.3f} | {int(s)} |")
    micro_p = prec.mean().item()
    micro_r = rec.mean().item()
    micro_f = f1.mean().item()
    lines.append(f"| **macro avg** | {micro_p:.3f} | {micro_r:.3f} | {micro_f:.3f} | {int(support.sum())} |")
    return "\n".join(lines)


def _compute_for_head(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) >= threshold)
    prec, rec, f1, tp, fp, fn = _binary_prf(pred, targets)
    support = tp + fn  # gold positives
    return prec, rec, f1, support


@torch.no_grad()
def run_per_class(
    ckpt_path: Path,
    split: str = "val",
    master_csv: Path = DEFAULT_MASTER,
    threshold: float = 0.5,
    batch_size: int = 16,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict:
    assert split in {"val", "test"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = load_model_from_ckpt(ckpt_path, device)

    ds = TriLevelDataset(master_csv, tok, max_length=cfg["max_length"])
    splits = load_or_create_splits(n_rows=len(ds))
    indices = splits[split]
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False)

    macro_logits, macro_tgts = [], []
    ind_logits, ind_tgts = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am)
        macro_logits.append(out["macro_logits"].float().cpu())
        macro_tgts.append(batch["macro_targets"])
        ind_logits.append(out["industry_logits"].float().cpu())
        ind_tgts.append(batch["industry_targets"])

    m_log = torch.cat(macro_logits)
    m_tgt = torch.cat(macro_tgts)
    i_log = torch.cat(ind_logits)
    i_tgt = torch.cat(ind_tgts)

    m_prec, m_rec, m_f1, m_sup = _compute_for_head(m_log, m_tgt, threshold)
    i_prec, i_rec, i_f1, i_sup = _compute_for_head(i_log, i_tgt, threshold)

    payload = {
        "ckpt": str(ckpt_path),
        "split": split,
        "threshold": threshold,
        "n_examples": len(indices),
        "macro": {
            name: {
                "precision": float(m_prec[k]), "recall": float(m_rec[k]),
                "f1": float(m_f1[k]), "support": int(m_sup[k]),
            }
            for k, name in enumerate(MACRO_LABELS)
        },
        "industry": {
            name: {
                "precision": float(i_prec[k]), "recall": float(i_rec[k]),
                "f1": float(i_f1[k]), "support": int(i_sup[k]),
            }
            for k, name in enumerate(INDUSTRY_LABELS)
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).parent.name
    json_path = out_dir / f"{stem}_per_class_{split}.json"
    md_path = out_dir / f"{stem}_per_class_{split}.md"
    json_path.write_text(json.dumps(payload, indent=2))

    md = [
        f"# Per-class report - {stem} ({split})",
        "",
        f"- threshold: {threshold}",
        f"- n_examples: {len(indices)}",
        "",
        "## Macro",
        _table_md(MACRO_LABELS, m_prec, m_rec, m_f1, m_sup),
        "",
        "## Industry",
        _table_md(INDUSTRY_LABELS, i_prec, i_rec, i_f1, i_sup),
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[per_class] {split}  macro_f1_macro={m_f1.mean():.3f}  "
          f"industry_f1_macro={i_f1.mean():.3f}  -> {md_path.name}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    run_per_class(
        ckpt_path=Path(args.ckpt),
        split=args.split,
        master_csv=Path(args.master_csv),
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
