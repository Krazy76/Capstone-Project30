"""TriLevelLoss: combined BCE + BCE + CE loss across the three heads.

    L = a_macro * BCE(macro,    pos_weight=w_m)
      + a_ind   * BCE(industry, pos_weight=w_i)
      + a_ent   * CE (entity,   ignore_index=-100)

pos_weight tensors are loaded from data/class_weights/*.pt and clamped
at POS_WEIGHT_CAP = 100 so ultra-rare macro classes (e.g. Consumption
n_pos=4, raw w=2303) don't hijack the gradient. Entity CE is unweighted
for the first pass; if O-dominance collapses recall later we add an
explicit O down-weight.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

POS_WEIGHT_CAP = 100.0
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = _PROJECT_ROOT / "data" / "class_weights"


class TriLevelLoss(nn.Module):
    def __init__(
        self,
        weights_dir: str | Path = DEFAULT_WEIGHTS_DIR,
        alpha_macro: float = 1.0,
        alpha_industry: float = 1.0,
        alpha_entity: float = 3.0,
        pos_weight_cap: float = POS_WEIGHT_CAP,
        ignore_index: int = -100,
    ):
        super().__init__()
        weights_dir = Path(weights_dir)
        macro_w = torch.load(weights_dir / "macro_pos_weight.pt", weights_only=True).float()
        industry_w = torch.load(weights_dir / "industry_pos_weight.pt", weights_only=True).float()

        # Buffers so .to(device) moves them with the module.
        self.register_buffer("macro_pos_weight", torch.clamp(macro_w, max=pos_weight_cap))
        self.register_buffer("industry_pos_weight", torch.clamp(industry_w, max=pos_weight_cap))

        self.alpha_macro = alpha_macro
        self.alpha_industry = alpha_industry
        self.alpha_entity = alpha_entity
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs: dict, targets: dict) -> dict:
        """Return {'loss', 'loss_macro', 'loss_industry', 'loss_entity'}."""
        loss_macro = nn.functional.binary_cross_entropy_with_logits(
            outputs["macro_logits"],
            targets["macro_targets"],
            pos_weight=self.macro_pos_weight,
        )
        loss_industry = nn.functional.binary_cross_entropy_with_logits(
            outputs["industry_logits"],
            targets["industry_targets"],
            pos_weight=self.industry_pos_weight,
        )
        entity_logits = outputs["entity_logits"]
        B, L, C = entity_logits.shape
        loss_entity = self.ce(
            entity_logits.reshape(B * L, C),
            targets["entity_targets"].reshape(B * L),
        )
        loss = (
            self.alpha_macro * loss_macro
            + self.alpha_industry * loss_industry
            + self.alpha_entity * loss_entity
        )
        return {
            "loss": loss,
            "loss_macro": loss_macro.detach(),
            "loss_industry": loss_industry.detach(),
            "loss_entity": loss_entity.detach(),
        }
