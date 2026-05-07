"""Active-learning round driver.

Per round:
  1. Train a fresh model on the current master csv (train.train).
  2. Score the train-pool (rows with empty macro_list).
  3. Mine top-K-per-class candidates.
  4. Re-label candidates via Gemma; merge results in place.
  5. Recompute pos_weight tensors (class balance shifts after rescue).
  6. Compare val combined-F1 vs previous round; check stop conditions.

Stop if any:
  - max_rounds reached
  - n_rescued in this round < min_rescued_per_round
  - val combined-F1 delta < min_delta for `patience_rounds` consecutive rounds
  - candidates list empty

Per-round outputs land in data/active_learning/round_{N}/:
  scores.jsonl, candidates.jsonl, relabel_log.jsonl, summary.json

Run:
  python -m src.active_learning.loop --backbone roberta-base --max-rounds 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.active_learning.mine import mine
from src.active_learning.relabel import relabel
from src.active_learning.score import score_pool
from src.training.train import TrainConfig, train

BASE = Path(r"E:\Coding\Python\Capstone-Project30")
DEFAULT_AL_ROOT = BASE / "data" / "active_learning"
DEFAULT_MASTER = BASE / "data" / "silver_dataset_master_with_text.csv"


@dataclass
class LoopConfig:
    backbone: str = "roberta-base"
    epochs: int = 3
    batch: int = 8
    lr: float = 2e-5
    use_lora: bool = True
    max_rounds: int = 5
    k_per_class: int = 50
    min_prob: float = 0.30
    min_rescued_per_round: int = 20
    min_delta: float = 0.005
    patience_rounds: int = 2
    skip_first_train: bool = False  # set if you already have a baseline ckpt
    baseline_ckpt: str | None = None


def _read_best_metrics(run_dir: Path) -> dict:
    """Pull last-line metrics from log.jsonl."""
    log = run_dir / "log.jsonl"
    last = None
    with log.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def _recompute_class_weights(round_dir: Path):
    """Refresh pos_weight tensors after relabelling and snapshot them.

    Snapshot is written to data/active_learning/round_{N}/class_weights/
    so we can reproduce any past round's loss exactly.
    """
    print("[loop] recomputing class weights ...")
    subprocess.run(
        [sys.executable, "-m", "src.data_prep.class_weights"],
        check=True,
        cwd=str(BASE),
    )
    src = BASE / "data" / "class_weights"
    dst = round_dir / "class_weights"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("macro_pos_weight.pt", "industry_pos_weight.pt", "label_stats.json"):
        p = src / name
        if p.exists():
            (dst / name).write_bytes(p.read_bytes())


def run_loop(cfg: LoopConfig) -> Path:
    al_root = DEFAULT_AL_ROOT
    al_root.mkdir(parents=True, exist_ok=True)
    summary_path = al_root / "loop_summary.jsonl"
    f_summary = summary_path.open("a")

    history: list[float] = []
    last_ckpt: Path | None = Path(cfg.baseline_ckpt) if cfg.baseline_ckpt else None

    # Source csv evolves per round: round 1 reads master, round N reads round_{N-1}/post.
    # Master csv on disk is never modified.
    current_csv: Path = DEFAULT_MASTER

    bad_rounds = 0
    for round_n in range(1, cfg.max_rounds + 1):
        print(f"\n========== AL round {round_n}/{cfg.max_rounds} ==========")
        print(f"[loop] training corpus: {current_csv.name}")
        round_dir = al_root / f"round_{round_n}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1. Train (skipped on round 1 if a baseline ckpt was passed)
        if not (round_n == 1 and cfg.skip_first_train and last_ckpt and last_ckpt.exists()):
            tcfg = TrainConfig(
                backbone=cfg.backbone,
                epochs=cfg.epochs,
                batch=cfg.batch,
                lr=cfg.lr,
                use_lora=cfg.use_lora,
                master_csv=str(current_csv),
            )
            run_dir = train(tcfg)
            last_ckpt = run_dir / "best.pt"
        else:
            run_dir = last_ckpt.parent
            print(f"[loop] reusing baseline ckpt {last_ckpt}")

        last_metrics = _read_best_metrics(run_dir)
        val_combined = last_metrics.get("val_combined_f1", 0.0)
        history.append(val_combined)

        # 2/3/4. Score -> mine -> relabel (read current_csv, write round_N/post.csv)
        scores_path = round_dir / "scores.jsonl"
        cand_path = round_dir / "candidates.jsonl"
        log_path = round_dir / "relabel_log.jsonl"
        post_csv = round_dir / "master_with_text_post.csv"

        score_pool(ckpt_path=last_ckpt, out_path=scores_path, master_csv=current_csv)
        n_cand = mine(scores_path, cand_path, k_per_class=cfg.k_per_class, min_prob=cfg.min_prob)
        if n_cand == 0:
            print("[loop] STOP: no candidates remaining")
            break
        input_csv_for_round = current_csv
        rl = relabel(
            candidates_path=cand_path,
            log_path=log_path,
            input_csv=input_csv_for_round,
            output_csv=post_csv,
            round_n=round_n,
        )
        # Next round trains on this round's rescued snapshot.
        current_csv = post_csv

        # 5. Refresh class weights for the next training run + snapshot.
        _recompute_class_weights(round_dir)

        round_summary = {
            "round": round_n,
            "val_combined_f1_before": val_combined,
            "n_candidates": rl["n_candidates"],
            "n_rescued": rl["n_rescued"],
            "n_failed": rl["n_failed"],
            "ckpt": str(last_ckpt),
            "input_csv": str(input_csv_for_round),
            "output_csv": rl.get("output_csv"),
            "history": history,
        }
        (round_dir / "summary.json").write_text(json.dumps(round_summary, indent=2))
        f_summary.write(json.dumps(round_summary) + "\n")
        f_summary.flush()
        print(f"[loop] round {round_n} done. rescued={rl['n_rescued']}")

        # 6. Stop conditions.
        if rl["n_rescued"] < cfg.min_rescued_per_round:
            print(f"[loop] STOP: rescued {rl['n_rescued']} < {cfg.min_rescued_per_round}")
            break
        if len(history) >= 2:
            delta = history[-1] - history[-2]
            if abs(delta) < cfg.min_delta:
                bad_rounds += 1
                if bad_rounds >= cfg.patience_rounds:
                    print(f"[loop] STOP: F1 plateau ({bad_rounds} rounds, |delta|<{cfg.min_delta})")
                    break
            else:
                bad_rounds = 0

    f_summary.close()
    print(f"[loop] finished. summary -> {summary_path}")
    return summary_path


def parse_args() -> LoopConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora", dest="use_lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--k-per-class", type=int, default=50)
    p.add_argument("--min-prob", type=float, default=0.30)
    p.add_argument("--min-rescued-per-round", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--patience-rounds", type=int, default=2)
    p.add_argument("--skip-first-train", action="store_true")
    p.add_argument("--baseline-ckpt", default=None)
    args = p.parse_args()
    return LoopConfig(**vars(args))


if __name__ == "__main__":
    run_loop(parse_args())
