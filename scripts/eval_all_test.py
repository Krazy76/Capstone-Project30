"""Run test-set eval for all 3 transformer checkpoints + span F1 for entity.

Output: reports/figures/test_results.json + console table
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
from src.model.architecture import TriLevelFinancialModel
from src.model.lora_wrap import wrap_with_lora
from src.training.metrics import multilabel_metrics, token_entity_metrics
from src.training.splits import load_or_create_splits
from src.eval.span_f1 import bio_to_spans, compute_span_metrics

CSV = ROOT / "data" / "active_learning" / "round_2" / "master_with_text_post.csv"
OUT_JSON = ROOT / "reports" / "figures" / "test_results.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "RoBERTa-base": ROOT / "artifacts" / "runs" / "roberta-base_20260505_145132" / "best.pt",
}

# auto-detect latest BERT and FinBERT
def latest(prefix: str) -> Path | None:
    runs = sorted(Path(ROOT / "artifacts" / "runs").glob(prefix + "*"))
    return runs[-1] / "best.pt" if runs else None

CKPTS["BERT-base"] = latest("bert-base-uncased_2026")
CKPTS["FinBERT-tone"] = latest("yiyanghkust_finbert-tone_2026")


@torch.no_grad()
def eval_ckpt(ckpt: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
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

    # test loader
    sub = Subset(ds, splits["test"])
    loader = DataLoader(sub, batch_size=16, shuffle=False)

    macro_lg, macro_tg, ind_lg, ind_tg, ent_lg, ent_tg = [], [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        macro_lg.append(out["macro_logits"].float().cpu())
        macro_tg.append(batch["macro_targets"].cpu())
        ind_lg.append(out["industry_logits"].float().cpu())
        ind_tg.append(batch["industry_targets"].cpu())
        ent_lg.append(out["entity_logits"].float().cpu())
        ent_tg.append(batch["entity_targets"].cpu())

    macro_lg, macro_tg = torch.cat(macro_lg), torch.cat(macro_tg)
    ind_lg, ind_tg = torch.cat(ind_lg), torch.cat(ind_tg)
    ent_lg, ent_tg = torch.cat(ent_lg), torch.cat(ent_tg)

    macro_m = multilabel_metrics(macro_lg, macro_tg)
    ind_m = multilabel_metrics(ind_lg, ind_tg)
    ent_m_token = token_entity_metrics(ent_lg, ent_tg, o_index=0)

    # Span-level F1 via seqeval (strict IOB2)
    pred_tags, true_tags = [], []
    pred_ids = ent_lg.argmax(dim=-1)
    for p_seq, t_seq in zip(pred_ids, ent_tg):
        mask = (t_seq != -100)
        p = [ENTITY_TAGS[int(idx)] for idx in p_seq[mask].tolist()]
        t = [ENTITY_TAGS[int(idx)] for idx in t_seq[mask].tolist()]
        pred_tags.append(p)
        true_tags.append(t)
    span_m = compute_span_metrics(pred_tags, true_tags)
    span_overall = span_m.get("overall", {})
    span_per_type = span_m.get("per_type", {})

    return {
        "ckpt": str(ckpt),
        "macro":     {"f1_micro": macro_m["f1_micro"], "f1_macro": macro_m["f1_macro"]},
        "industry":  {"f1_micro": ind_m["f1_micro"],   "f1_macro": ind_m["f1_macro"]},
        "entity_token": {"f1_micro": ent_m_token["f1_micro"]},
        "entity_span":  {
            "f1_micro": span_overall.get("f1", 0),
            "precision": span_overall.get("precision", 0),
            "recall": span_overall.get("recall", 0),
            "support": span_overall.get("support", 0),
            "per_type": {k: v.get("f1", 0) for k, v in span_per_type.items()},
        },
        "combined_f1_token":
            (macro_m["f1_micro"] + ind_m["f1_micro"] + ent_m_token["f1_micro"]) / 3,
    }


def main():
    results = {}
    for name, ckpt in CKPTS.items():
        if ckpt is None or not ckpt.exists():
            print(f"[skip] {name}: no checkpoint")
            continue
        print(f"[eval] {name}: {ckpt.name}")
        results[name] = eval_ckpt(ckpt)

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # console summary
    print(f"\n{'=== TEST results (922 rows, RoBERTa splits) ===':<60}")
    print(f"  {'Backbone':<14} {'Macro':>8} {'Industry':>9} {'Entity_tok':>11} {'Entity_span':>12} {'Combined':>9}")
    for name, r in results.items():
        print(f"  {name:<14} {r['macro']['f1_micro']:>8.4f} {r['industry']['f1_micro']:>9.4f} "
              f"{r['entity_token']['f1_micro']:>11.4f} {r['entity_span']['f1_micro']:>12.4f} "
              f"{r['combined_f1_token']:>9.4f}")
    print(f"\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
