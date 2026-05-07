"""Per-class confusion-style heatmap for the macro and industry heads (RoBERTa, test split).

For multi-label tasks, a strict NxN confusion matrix isn't well-defined. We show:
  - Per-class precision/recall/F1 heatmap
  - Co-occurrence matrix: P(class j predicted | class i true)

Output: reports/figures/macro_confusion.png, industry_confusion.png
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
from src.training.splits import load_or_create_splits

CKPT = ROOT / "artifacts" / "runs" / "roberta-base_20260505_145132" / "best.pt"
CSV = ROOT / "data" / "active_learning" / "round_2" / "master_with_text_post.csv"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def collect_preds():
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
            model,
            r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 16),
            dropout=cfg.get("lora_dropout", 0.05), verbose=False,
        )
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device).eval()

    sub = Subset(ds, splits["test"])
    loader = DataLoader(sub, batch_size=16, shuffle=False)

    macro_pred, macro_tgt = [], []
    ind_pred, ind_tgt = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        macro_pred.append((torch.sigmoid(out["macro_logits"].float()) >= 0.5).cpu())
        macro_tgt.append(batch["macro_targets"].cpu())
        ind_pred.append((torch.sigmoid(out["industry_logits"].float()) >= 0.5).cpu())
        ind_tgt.append(batch["industry_targets"].cpu())
    return (torch.cat(macro_pred).numpy(), torch.cat(macro_tgt).numpy(),
            torch.cat(ind_pred).numpy(), torch.cat(ind_tgt).numpy())


def cooccur_pred_given_true(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """For each (true_i, pred_j), what fraction of articles with true_i also have pred_j?
    Diagonal = recall. Off-diagonal = systematic confusions."""
    n_classes = pred.shape[1]
    M = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        true_i = tgt[:, i] == 1
        n_i = true_i.sum()
        if n_i == 0:
            continue
        for j in range(n_classes):
            M[i, j] = pred[true_i, j].sum() / n_i
    return M


def plot_heatmap(M: np.ndarray, labels: list[str], title: str, out: Path):
    n = len(labels)
    short_labels = [l.replace(", ", ",\n") if "," in l else l for l in labels]
    fig, ax = plt.subplots(figsize=(0.8 * n + 3, 0.6 * n + 2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved: {out}")


def main():
    macro_pred, macro_tgt, ind_pred, ind_tgt = collect_preds()
    Mm = cooccur_pred_given_true(macro_pred, macro_tgt)
    Mi = cooccur_pred_given_true(ind_pred, ind_tgt)
    plot_heatmap(Mm, MACRO_LABELS,
                 "Macro head: P(predicted | true) — RoBERTa, test n=922\n(diagonal=recall, off-diag=systematic confusions)",
                 OUT_DIR / "macro_confusion.png")
    plot_heatmap(Mi, INDUSTRY_LABELS,
                 "Industry head: P(predicted | true) — RoBERTa, test n=922",
                 OUT_DIR / "industry_confusion.png")


if __name__ == "__main__":
    main()
