"""Per-head metrics for the Tri-Level training loop.

Macro and Industry are multi-label BCE heads: sigmoid -> threshold ->
per-class precision/recall/F1, then micro and macro averages.

Entity is a token-level CE head: argmax over tag dim, drop ignore_index
(-100) and (optionally) the O class, then micro F1 over remaining tokens.
A proper span-level F1 (seqeval) is left for §6 evaluation; token F1 is
sufficient for in-loop monitoring and early stopping.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _binary_prf(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-9):
    """Per-class precision, recall, F1 for {0,1} tensors of shape [N, C]."""
    tp = (pred & tgt.bool()).sum(dim=0).float()
    fp = (pred & ~tgt.bool()).sum(dim=0).float()
    fn = (~pred & tgt.bool()).sum(dim=0).float()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1, tp, fp, fn


def multilabel_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    """Micro + macro F1 for a multi-label BCE head."""
    pred = (torch.sigmoid(logits) >= threshold)
    prec, rec, f1, tp, fp, fn = _binary_prf(pred, targets)
    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_p = micro_tp / (micro_tp + micro_fp + 1e-9)
    micro_r = micro_tp / (micro_tp + micro_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)
    return {
        "f1_micro": float(micro_f1),
        "f1_macro": float(f1.mean()),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "per_class_f1": f1.tolist(),
    }


def token_entity_metrics(
    logits: torch.Tensor, targets: torch.Tensor, o_index: int = 0
) -> dict:
    """Token-level micro F1 for the entity head, excluding O and -100.

    logits  [B, L, C]
    targets [B, L]   with -100 on ignore positions
    """
    pred = logits.argmax(dim=-1)
    valid = targets != IGNORE_INDEX
    pred = pred[valid]
    tgt = targets[valid]

    # Restrict scoring to non-O ground truth or non-O prediction so the
    # massive O-class doesn't trivially inflate F1.
    mask = (tgt != o_index) | (pred != o_index)
    pred = pred[mask]
    tgt = tgt[mask]

    if pred.numel() == 0:
        return {"f1_micro": 0.0, "n_tokens": 0}

    tp = ((pred == tgt) & (tgt != o_index)).sum().float()
    fp = ((pred != tgt) & (pred != o_index)).sum().float()
    fn = ((pred != tgt) & (tgt != o_index)).sum().float()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "f1_micro": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "n_tokens": int(valid.sum()),
    }
