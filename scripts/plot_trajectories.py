"""F1 trajectory plot across 3 backbones (post-AL training).

Output: reports/figures/f1_trajectory.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
OUT = ROOT / "reports" / "figures" / "f1_trajectory.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Pick the FINAL post-AL run for each backbone (8 epochs, on round_2/post.csv).
backbones = {
    "RoBERTa-base":  "roberta-base_20260505_145132",
    "BERT-base":     None,   # latest bert-base-uncased
    "FinBERT-tone":  None,   # latest yiyanghkust_finbert-tone
}
# Auto-detect latest bert + finbert if not pinned
for name, dirname in list(backbones.items()):
    if dirname is None:
        prefix = "bert-base-uncased_2026" if "BERT" in name else "yiyanghkust_finbert-tone_2026"
        dirs = sorted([d.name for d in RUNS.iterdir() if d.name.startswith(prefix)])
        backbones[name] = dirs[-1] if dirs else None

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
metric_keys = [
    ("val_macro_f1_micro",    "Macro F1"),
    ("val_industry_f1_micro", "Industry F1"),
    ("val_entity_f1_micro",   "Entity F1 (token)"),
    ("val_combined_f1",       "Combined F1"),
]
colors = {"RoBERTa-base": "#1f77b4", "BERT-base": "#d62728", "FinBERT-tone": "#2ca02c"}

for ax, (key, title) in zip(axes, metric_keys):
    for name, dirname in backbones.items():
        if dirname is None:
            continue
        log = RUNS / dirname / "log.jsonl"
        if not log.exists():
            continue
        epochs, vals = [], []
        for line in log.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                epochs.append(d["epoch"])
                vals.append(d.get(key, 0))
        ax.plot(epochs, vals, marker="o", label=name, color=colors.get(name), linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

plt.suptitle("Per-head F1 trajectory across backbones (8 epochs, post-AL corpus)", y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")
