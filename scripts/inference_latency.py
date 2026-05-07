"""Inference latency benchmark for the production checkpoint.

Reports per-article wall-clock for batch=1 and batch=16 on CPU and GPU
(if available). For section 6.1 / 7 deployment recommendation.

Output: reports/figures/inference_latency.json + console table
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
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
OUT_JSON = ROOT / "reports" / "figures" / "inference_latency.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

WARMUP = 5
N_RUNS = 50


def load_model(device):
    state = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = state["config"]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
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
    return model, tok, cfg


@torch.no_grad()
def bench(model, tok, articles: list[str], device, batch_size: int) -> dict:
    # Warmup
    for i in range(WARMUP):
        sample = articles[i % len(articles)]
        enc = tok([sample] * batch_size, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(device)
        _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Time N_RUNS forward passes
    t0 = time.perf_counter()
    for i in range(N_RUNS):
        sample = articles[i % len(articles)]
        enc = tok([sample] * batch_size, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(device)
        _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_articles = N_RUNS * batch_size
    return {
        "device": str(device),
        "batch_size": batch_size,
        "n_passes": N_RUNS,
        "total_articles": total_articles,
        "wall_seconds": elapsed,
        "ms_per_article": (elapsed / total_articles) * 1000,
        "ms_per_batch": (elapsed / N_RUNS) * 1000,
    }


def main():
    # Get a few real articles
    from transformers import AutoTokenizer as AT
    tok = AT.from_pretrained("roberta-base", use_fast=True)
    ds = TriLevelDataset(CSV, tok, max_length=512)
    splits = load_or_create_splits(n_rows=len(ds))
    test_idx = splits["test"][:50]
    raw = [ds.df.iloc[i]["text"] for i in test_idx]
    articles = [a if isinstance(a, str) and a else "n/a" for a in raw]

    results = {}
    for device_str in ["cuda", "cpu"]:
        if device_str == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(device_str)
        model, tok2, cfg = load_model(device)
        for bs in [1, 16]:
            r = bench(model, tok2, articles, device, batch_size=bs)
            key = f"{device_str}_bs{bs}"
            results[key] = r
            print(f"{key:<12} {r['ms_per_article']:>7.2f} ms/article  "
                  f"({r['ms_per_batch']:>7.2f} ms/batch of {bs})")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
