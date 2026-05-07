"""Pre-AL vs post-AL RoBERTa comparison.

Pre-AL run:  artifacts/runs/roberta-base_20260505_124448  (8 ep, master.csv)
Post-AL run: artifacts/runs/roberta-base_20260505_145132  (8 ep, round_2/post.csv)

Output: reports/figures/pre_vs_post_al.png + json table
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
OUT_FIG = ROOT / "reports" / "figures" / "pre_vs_post_al.png"
OUT_JSON = ROOT / "reports" / "figures" / "pre_vs_post_al.json"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

PRE = RUNS / "roberta-base_20260505_124448"   # pre-AL baseline
POST = RUNS / "roberta-base_20260505_145132"  # post-AL final


def best_metrics(run_dir: Path) -> dict:
    log = run_dir / "log.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    best = max(rows, key=lambda r: r["val_combined_f1"])
    return best


pre = best_metrics(PRE)
post = best_metrics(POST)

heads = ["val_macro_f1_micro", "val_industry_f1_micro", "val_entity_f1_micro", "val_combined_f1"]
head_labels = ["Macro F1", "Industry F1", "Entity F1 (token)", "Combined F1"]
pre_vals = [pre[h] for h in heads]
post_vals = [post[h] for h in heads]
deltas = [b - a for a, b in zip(pre_vals, post_vals)]

x = np.arange(len(heads))
w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.5))
b1 = ax.bar(x - w/2, pre_vals, w, label="Pre-AL (master v2 only)", color="#888888")
b2 = ax.bar(x + w/2, post_vals, w, label="Post-AL (round_2/post.csv)", color="#1f77b4")

for bars, vals in [(b1, pre_vals), (b2, post_vals)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)

# Delta annotations
for i, d in enumerate(deltas):
    sign = "+" if d >= 0 else ""
    ax.text(x[i], max(pre_vals[i], post_vals[i]) + 0.05,
            f"Δ {sign}{d:.4f}",
            ha="center", fontsize=9, fontweight="bold",
            color="#2ca02c" if d > 0 else "#d62728")

ax.set_xticks(x); ax.set_xticklabels(head_labels)
ax.set_ylabel("Validation F1")
ax.set_ylim(0, max(post_vals) * 1.25)
ax.set_title("RoBERTa pre-AL vs post-AL (8 epochs each, same val split)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=140, bbox_inches="tight")

OUT_JSON.write_text(json.dumps({
    "pre_AL_run":  PRE.name,
    "post_AL_run": POST.name,
    "pre_AL":  {h: pre[h]  for h in heads},
    "post_AL": {h: post[h] for h in heads},
    "delta":   {h: post[h] - pre[h] for h in heads},
}, indent=2))

print(f"saved: {OUT_FIG}")
print(f"saved: {OUT_JSON}")
print(f"\n{'metric':<25} {'pre-AL':>8} {'post-AL':>8} {'Δ':>8}")
for h, lbl in zip(heads, head_labels):
    print(f"  {lbl:<23} {pre[h]:>8.4f} {post[h]:>8.4f} {post[h]-pre[h]:>+8.4f}")
