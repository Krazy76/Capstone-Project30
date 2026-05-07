"""Tri-Level training loop (Step 6).

End-to-end training of the shared-encoder + 3-head model with optional
LoRA wrap. Frozen val/test split via splits.load_or_create_splits, AdamW
with linear warmup, mixed-precision autocast on CUDA, per-epoch eval on
all three heads, best-checkpoint save by combined val score, and a JSONL
log of every epoch for the report.

Run (RoBERTa, default):
    python -m src.training.train --backbone roberta-base --epochs 3 --batch 8 --lora

Run (FinBERT):
    python -m src.training.train --backbone yiyanghkust/finbert-tone --epochs 3 --batch 8 --lora

Run (debug, tiny subset):
    python -m src.training.train --backbone roberta-base --epochs 1 --batch 2 --max-train 32 --max-val 16 --no-lora
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_prep.dataset import TriLevelDataset
from src.data_prep.filters import ENTITY_TAGS, INDUSTRY_LABELS, MACRO_LABELS
from src.model.architecture import TriLevelFinancialModel
from src.model.losses import TriLevelLoss
from src.model.lora_wrap import wrap_with_lora
from src.training.metrics import multilabel_metrics, token_entity_metrics
from src.training.splits import load_or_create_splits

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = _PROJECT_ROOT / "data" / "silver_dataset_master_with_text.csv"
DEFAULT_RUN_DIR = _PROJECT_ROOT / "artifacts" / "runs"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    backbone: str = "roberta-base"
    master_csv: str = str(DEFAULT_MASTER)
    run_dir: str = str(DEFAULT_RUN_DIR)
    max_length: int = 512
    batch: int = 8
    eval_batch: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    val_frac: float = 0.10
    test_frac: float = 0.10
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    amp: bool = True
    num_workers: int = 0  # Windows: keep 0 to avoid pickling pain
    patience: int = 2
    max_train: int | None = None  # debug cap
    max_val: int | None = None    # debug cap


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def build_dataset(cfg: TrainConfig):
    tok = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
    ds = TriLevelDataset(cfg.master_csv, tok, max_length=cfg.max_length)
    splits = load_or_create_splits(
        n_rows=len(ds),
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.seed,
    )
    train_idx = splits["train"]
    val_idx = splits["val"]
    if cfg.max_train is not None:
        train_idx = train_idx[: cfg.max_train]
    if cfg.max_val is not None:
        val_idx = val_idx[: cfg.max_val]
    return ds, Subset(ds, train_idx), Subset(ds, val_idx), splits, tok


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    model = TriLevelFinancialModel(
        cfg.backbone,
        num_macro=len(MACRO_LABELS),
        num_industry=len(INDUSTRY_LABELS),
        num_entity_tags=len(ENTITY_TAGS),
    )
    if cfg.use_lora:
        model = wrap_with_lora(
            model,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
    return model


def move_batch(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loss_fn, loader, device, amp: bool) -> dict:
    model.eval()
    macro_logits, macro_tgts = [], []
    ind_logits, ind_tgts = [], []
    ent_logits, ent_tgts = [], []
    losses = {"loss": 0.0, "loss_macro": 0.0, "loss_industry": 0.0, "loss_entity": 0.0}
    n = 0
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_ctx:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            losses_step = loss_fn(out, batch)
        b = batch["input_ids"].size(0)
        n += b
        for k in losses:
            losses[k] += float(losses_step[k].detach()) * b
        macro_logits.append(out["macro_logits"].float().cpu())
        macro_tgts.append(batch["macro_targets"].cpu())
        ind_logits.append(out["industry_logits"].float().cpu())
        ind_tgts.append(batch["industry_targets"].cpu())
        ent_logits.append(out["entity_logits"].float().cpu())
        ent_tgts.append(batch["entity_targets"].cpu())

    macro = multilabel_metrics(torch.cat(macro_logits), torch.cat(macro_tgts))
    industry = multilabel_metrics(torch.cat(ind_logits), torch.cat(ind_tgts))
    entity = token_entity_metrics(torch.cat(ent_logits), torch.cat(ent_tgts), o_index=0)

    for k in losses:
        losses[k] /= max(n, 1)
    combined = (macro["f1_micro"] + industry["f1_micro"] + entity["f1_micro"]) / 3.0
    return {
        "loss": losses,
        "macro": macro,
        "industry": industry,
        "entity": entity,
        "combined_f1": combined,
        "n_examples": n,
    }


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds, train_ds, val_ds, splits, _ = build_dataset(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    loss_fn = TriLevelLoss().to(device)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimiser = torch.optim.AdamW(grouped, lr=cfg.lr)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    # Run dir + logging
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = cfg.backbone.replace("/", "_")
    run_dir = Path(cfg.run_dir) / f"{safe_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (run_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    log_path = run_dir / "log.jsonl"
    print(f"[train] run_dir={run_dir}")
    print(f"[train] device={device} | train={len(train_ds)} val={len(val_ds)}")

    best_score = -1.0
    best_path = run_dir / "best.pt"
    bad_epochs = 0

    autocast_ctx_train = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.amp and device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            optimiser.zero_grad(set_to_none=True)
            with autocast_ctx_train:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                losses = loss_fn(out, batch)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), cfg.grad_clip
            )
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            global_step += 1
            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"  ep{epoch} step {step}/{len(train_loader)}  "
                    f"loss={running / step:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        eval_out = evaluate(model, loss_fn, val_loader, device, cfg.amp)
        epoch_time = time.time() - t0

        log_row = {
            "epoch": epoch,
            "train_loss_avg": running / max(1, len(train_loader)),
            "val_loss": eval_out["loss"]["loss"],
            "val_macro_f1_micro": eval_out["macro"]["f1_micro"],
            "val_industry_f1_micro": eval_out["industry"]["f1_micro"],
            "val_entity_f1_micro": eval_out["entity"]["f1_micro"],
            "val_combined_f1": eval_out["combined_f1"],
            "epoch_seconds": epoch_time,
            "global_step": global_step,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")
        print(
            f"[ep{epoch}] val combined_f1={eval_out['combined_f1']:.4f} "
            f"(M={eval_out['macro']['f1_micro']:.3f} "
            f"I={eval_out['industry']['f1_micro']:.3f} "
            f"E={eval_out['entity']['f1_micro']:.3f}) "
            f"loss={eval_out['loss']['loss']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if eval_out["combined_f1"] > best_score:
            best_score = eval_out["combined_f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val": eval_out,
                },
                best_path,
            )
            print(f"  [best] combined_f1={best_score:.4f} -> {best_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop after {bad_epochs} no-improve epochs")
                break

    print(f"[train] done. best combined_f1={best_score:.4f}  ckpt={best_path}")
    return run_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--master-csv", default=str(DEFAULT_MASTER))
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--lora", dest="use_lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
