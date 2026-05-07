"""Class distribution histograms for v2 corpus (macro + industry).

Output: reports/figures/class_distribution.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep.filters import (
    INDUSTRY_LABELS, MACRO_LABELS, load_and_filter_master,
)

CSV = ROOT / "data" / "active_learning" / "round_2" / "master_with_text_post.csv"
OUT = ROOT / "reports" / "figures" / "class_distribution.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

df = load_and_filter_master(CSV)
print(f"v2 useful rows: {len(df)}")

# Count per class (across whole v2 set)
mc = {l: 0 for l in MACRO_LABELS}
ic = {l: 0 for l in INDUSTRY_LABELS}
for ml in df["macro_list"]:
    for v in ml:
        if v in mc: mc[v] += 1
for il in df["industry_list"]:
    for v in il:
        if v in ic: ic[v] += 1

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
# Macro
labels_m = list(mc.keys()); counts_m = list(mc.values())
order_m = sorted(range(len(labels_m)), key=lambda i: -counts_m[i])
labels_m = [labels_m[i] for i in order_m]; counts_m = [counts_m[i] for i in order_m]

bars = axes[0].barh(labels_m, counts_m, color="#4c72b0")
axes[0].set_title(f"Macro labels (FRED-MD 8) | total spans = {sum(counts_m)}")
axes[0].set_xlabel("Articles labelled with this class (v2 corpus)")
axes[0].invert_yaxis()
for b, c in zip(bars, counts_m):
    axes[0].text(c + max(counts_m)*0.01, b.get_y() + b.get_height()/2,
                 f"{c}", va="center", fontsize=9)

# Industry
labels_i = list(ic.keys()); counts_i = list(ic.values())
order_i = sorted(range(len(labels_i)), key=lambda i: -counts_i[i])
labels_i = [labels_i[i] for i in order_i]; counts_i = [counts_i[i] for i in order_i]

bars = axes[1].barh(labels_i, counts_i, color="#dd8452")
axes[1].set_title(f"Industry labels (GICS 11) | total spans = {sum(counts_i)}")
axes[1].set_xlabel("Articles labelled with this class (v2 corpus)")
axes[1].invert_yaxis()
for b, c in zip(bars, counts_i):
    axes[1].text(c + max(counts_i)*0.01, b.get_y() + b.get_height()/2,
                 f"{c}", va="center", fontsize=9)

plt.suptitle(f"v2 silver-label class distribution (n={len(df)} useful rows)")
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")
