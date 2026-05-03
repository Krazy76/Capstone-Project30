"""Score the unlabelled-macro pool with a trained checkpoint.

The "pool" for round N is all rows that
  (a) belong to the **train** split (val/test never enter AL), and
  (b) have an empty macro_list (Gemma did not assign any macro label).

For each pool row we run the trained model and emit:
  - sigmoid(macro_logits)   (8-vector of per-class probabilities)
  - macro_entropy           (sum of binary entropies, used as scalar uncertainty)

Output is written as JSONL so it round-trips without pyarrow:
  data/active_learning/round_{N}/scores.jsonl

Run:
  python -m src.active_learning.score \
      --ckpt artifacts/runs/<run>/best.pt \
      --round 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import MACRO_LABELS
from src.model.architecture import TriLevelFinancialModel
from src.model.lora_wrap import wrap_with_lora
from src.training.splits import load_or_create_splits

DEFAULT_MASTER = Path(
    r"E:\Coding\Python\Capstone-Project30\data\silver_dataset_master_with_text.csv"
)
DEFAULT_AL_ROOT = Path(r"E:\Coding\Python\Capstone-Project30\data\active_learning")


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-element H = -p log p - (1-p) log (1-p), summed over the last dim."""
    eps = 1e-9
    h = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
    return h.sum(dim=-1)


def build_pool_indices(ds: TriLevelDataset, split_train_idx: list[int]) -> list[int]:
    """Train rows with empty macro_list are the AL pool."""
    pool = []
    for i in split_train_idx:
        if len(ds.df.iloc[i]["macro_list"]) == 0:
            pool.append(i)
    return pool


def load_model_from_ckpt(ckpt_path: Path, device) -> tuple[torch.nn.Module, dict, AutoTokenizer]:
    """Recreate model + LoRA wrap + load weights, mirroring training config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
    model = TriLevelFinancialModel(
        cfg["backbone"],
        num_macro=len(MACRO_LABELS),
        num_industry=len(INDUSTRY_LABELS),
        num_entity_tags=len(ENTITY_TAGS),
    )
    if cfg.get("use_lora", True):
        model = wrap_with_lora(
            model,
            r=cfg["lora_r"],
            alpha=cfg["lora_alpha"],
            dropout=cfg["lora_dropout"],
            verbose=False,
        )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[score] missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval(), cfg, tok


@torch.no_grad()
def score_pool(
    ckpt_path: Path,
    out_path: Path,
    master_csv: Path = DEFAULT_MASTER,
    batch_size: int = 16,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok = load_model_from_ckpt(ckpt_path, device)

    ds = TriLevelDataset(master_csv, tok, max_length=cfg["max_length"])
    splits = load_or_create_splits(n_rows=len(ds))
    pool_idx = build_pool_indices(ds, splits["train"])
    print(f"[score] pool size = {len(pool_idx)} (train rows with empty macro_list)")

    loader = DataLoader(Subset(ds, pool_idx), batch_size=batch_size, shuffle=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    cursor = 0
    with out_path.open("w") as f:
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=am)
            probs = torch.sigmoid(out["macro_logits"]).float().cpu()
            ent = _binary_entropy(probs).cpu().tolist()
            probs_l = probs.tolist()
            for j in range(probs.size(0)):
                global_idx = pool_idx[cursor + j]
                row = {
                    "row_idx": int(global_idx),
                    "macro_probs": probs_l[j],
                    "macro_entropy": float(ent[j]),
                }
                f.write(json.dumps(row) + "\n")
                n_written += 1
            cursor += probs.size(0)
    print(f"[score] wrote {n_written} rows -> {out_path}")
    return n_written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best.pt")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--al-root", default=str(DEFAULT_AL_ROOT))
    args = p.parse_args()

    out = Path(args.al_root) / f"round_{args.round}" / "scores.jsonl"
    score_pool(
        ckpt_path=Path(args.ckpt),
        out_path=out,
        master_csv=Path(args.master_csv),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
