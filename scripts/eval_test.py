"""Evaluate a saved checkpoint on the test split. Prints micro+macro F1 for
all three heads, and combined F1 to match the training-loop convention.

Run:
    python scripts/eval_test.py --ckpt artifacts/runs/roberta-base_20260505_145132/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src.*` importable when running this file as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
from src.model.architecture import TriLevelFinancialModel
from src.model.lora_wrap import wrap_with_lora
from src.training.metrics import multilabel_metrics, token_entity_metrics
from src.training.splits import load_or_create_splits


@torch.no_grad()
def evaluate_test(ckpt_path: Path, master_csv: Path, split: str = "test", batch_size: int = 16) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]

    tok = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    ds = TriLevelDataset(master_csv, tok, max_length=cfg["max_length"])
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
            r=cfg.get("lora_r", 8),
            alpha=cfg.get("lora_alpha", 16),
            dropout=cfg.get("lora_dropout", 0.05),
            verbose=False,
        )
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device).eval()

    sub = Subset(ds, splits[split])
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False)

    macro_logits, macro_tgts = [], []
    ind_logits, ind_tgts = [], []
    ent_logits, ent_tgts = [], []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        macro_logits.append(out["macro_logits"].float().cpu())
        macro_tgts.append(batch["macro_targets"].cpu())
        ind_logits.append(out["industry_logits"].float().cpu())
        ind_tgts.append(batch["industry_targets"].cpu())
        ent_logits.append(out["entity_logits"].float().cpu())
        ent_tgts.append(batch["entity_targets"].cpu())

    macro = multilabel_metrics(torch.cat(macro_logits), torch.cat(macro_tgts))
    industry = multilabel_metrics(torch.cat(ind_logits), torch.cat(ind_tgts))
    entity = token_entity_metrics(torch.cat(ent_logits), torch.cat(ent_tgts), o_index=0)
    combined = (macro["f1_micro"] + industry["f1_micro"] + entity["f1_micro"]) / 3.0

    return {
        "ckpt": str(ckpt_path),
        "backbone": cfg["backbone"],
        "split": split,
        "n_examples": len(sub),
        "macro": macro,
        "industry": industry,
        "entity": entity,
        "combined_f1": combined,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--master-csv", default="data/active_learning/round_2/master_with_text_post.csv")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    out = evaluate_test(Path(args.ckpt), Path(args.master_csv), args.split)
    print(f"\n=== {out['backbone']}  {out['split']}  n={out['n_examples']} ===")
    for head in ("macro", "industry"):
        m = out[head]
        print(f"  {head:<10} f1_micro={m['f1_micro']:.4f}  f1_macro={m['f1_macro']:.4f}  "
              f"P_macro={m['precision_macro']:.4f}  R_macro={m['recall_macro']:.4f}")
    e = out["entity"]
    print(f"  {'entity':<10} f1_micro={e['f1_micro']:.4f}  precision={e.get('precision',0):.4f}  "
          f"recall={e.get('recall',0):.4f}  n_tokens={e.get('n_tokens',0)}")
    print(f"  combined_f1 = {out['combined_f1']:.4f}\n")

    # Save JSON
    out_path = Path("artifacts/eval") / f"{Path(args.ckpt).parent.name}_{args.split}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": out["ckpt"],
        "backbone": out["backbone"],
        "split": out["split"],
        "n_examples": out["n_examples"],
        "macro_f1_micro": float(out["macro"]["f1_micro"]),
        "macro_f1_macro": float(out["macro"]["f1_macro"]),
        "industry_f1_micro": float(out["industry"]["f1_micro"]),
        "industry_f1_macro": float(out["industry"]["f1_macro"]),
        "entity_f1_micro": float(out["entity"]["f1_micro"]),
        "entity_precision": float(out["entity"].get("precision", 0)),
        "entity_recall": float(out["entity"].get("recall", 0)),
        "combined_f1": float(out["combined_f1"]),
    }, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
