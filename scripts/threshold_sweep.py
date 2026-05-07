"""Threshold sweep for macro head — F1 vs sigmoid threshold.

Shows model is well-calibrated (or not) and where F1 maximises.
Defaults to RoBERTa post-AL test set.

Output: reports/figures/threshold_sweep.png + json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
from src.model.architecture import TriLevelFinancialModel
from src.model.lora_wrap import wrap_with_lora
from src.training.metrics import multilabel_metrics
from src.training.splits import load_or_create_splits

CKPT = ROOT / "artifacts" / "runs" / "roberta-base_20260505_145132" / "best.pt"
CSV = ROOT / "data" / "active_learning" / "round_2" / "master_with_text_post.csv"
OUT_FIG = ROOT / "reports" / "figures" / "threshold_sweep.png"
OUT_JSON = ROOT / "reports" / "figures" / "threshold_sweep.json"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def collect_logits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = state["config"]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    ds = TriLevelDataset(CSV, tok, max_length=cfg["max_length"])
    splits = load_or_create_splits(n_rows=len(ds))

    model = TriLevelFinancialModel(
        cfg["backbone"],
        num_macro=len(MACRO_LABELS),
        num_industry=len(INDUSTRY_LABELS),
        num_entity_tags=len(ENTITY_TAGS),
    )
    if cfg.get("use_lora", True):
        model = wrap_with_lora(
            model, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16),
            dropout=cfg.get("lora_dropout", 0.05), verbose=False,
        )
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device).eval()

    sub = Subset(ds, splits["test"])
    loader = DataLoader(sub, batch_size=16, shuffle=False)
    macro_lg, macro_tg = [], []
    ind_lg, ind_tg = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        macro_lg.append(out["macro_logits"].float().cpu())
        macro_tg.append(batch["macro_targets"].cpu())
        ind_lg.append(out["industry_logits"].float().cpu())
        ind_tg.append(batch["industry_targets"].cpu())
    return (torch.cat(macro_lg), torch.cat(macro_tg),
            torch.cat(ind_lg), torch.cat(ind_tg))


def main():
    macro_lg, macro_tg, ind_lg, ind_tg = collect_logits()
    thresholds = np.linspace(0.1, 0.9, 33)

    macro_micro, macro_macro = [], []
    ind_micro, ind_macro = [], []
    for t in thresholds:
        m = multilabel_metrics(macro_lg, macro_tg, threshold=float(t))
        i = multilabel_metrics(ind_lg, ind_tg, threshold=float(t))
        macro_micro.append(m["f1_micro"]); macro_macro.append(m["f1_macro"])
        ind_micro.append(i["f1_micro"]);  ind_macro.append(i["f1_macro"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(thresholds, macro_micro, marker="o", label="micro F1", color="#1f77b4")
    axes[0].plot(thresholds, macro_macro, marker="s", label="macro F1", color="#ff7f0e")
    axes[0].axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="default 0.5")
    best_t_macro = thresholds[int(np.argmax(macro_micro))]
    axes[0].axvline(best_t_macro, color="green", linestyle=":", alpha=0.7,
                    label=f"best={best_t_macro:.2f}")
    axes[0].set_title(f"Macro head | best micro F1 = {max(macro_micro):.3f} at t={best_t_macro:.2f}")
    axes[0].set_xlabel("Sigmoid threshold"); axes[0].set_ylabel("F1")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(thresholds, ind_micro, marker="o", label="micro F1", color="#1f77b4")
    axes[1].plot(thresholds, ind_macro, marker="s", label="macro F1", color="#ff7f0e")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="default 0.5")
    best_t_ind = thresholds[int(np.argmax(ind_micro))]
    axes[1].axvline(best_t_ind, color="green", linestyle=":", alpha=0.7,
                    label=f"best={best_t_ind:.2f}")
    axes[1].set_title(f"Industry head | best micro F1 = {max(ind_micro):.3f} at t={best_t_ind:.2f}")
    axes[1].set_xlabel("Sigmoid threshold"); axes[1].set_ylabel("F1")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Threshold sweep on test set (RoBERTa post-AL)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=140, bbox_inches="tight")

    OUT_JSON.write_text(json.dumps({
        "thresholds": thresholds.tolist(),
        "macro_micro_f1": macro_micro,
        "macro_macro_f1": macro_macro,
        "industry_micro_f1": ind_micro,
        "industry_macro_f1": ind_macro,
        "best_threshold_macro": float(best_t_macro),
        "best_macro_micro_f1": float(max(macro_micro)),
        "default_threshold_macro": 0.5,
        "default_macro_micro_f1": float(macro_micro[list(thresholds).index(min(thresholds, key=lambda x: abs(x-0.5)))]),
        "best_threshold_industry": float(best_t_ind),
        "best_industry_micro_f1": float(max(ind_micro)),
    }, indent=2))

    print(f"saved: {OUT_FIG}\nsaved: {OUT_JSON}")
    print(f"\nMacro:    default(0.5) vs best({best_t_macro:.2f}): "
          f"f1_micro {macro_micro[len(macro_micro)//2]:.4f} -> {max(macro_micro):.4f}")
    print(f"Industry: default(0.5) vs best({best_t_ind:.2f}): "
          f"f1_micro {ind_micro[len(ind_micro)//2]:.4f} -> {max(ind_micro):.4f}")


if __name__ == "__main__":
    main()
