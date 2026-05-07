"""AL rescue per-class bar chart for rounds 1+2.

Output: reports/figures/al_rescue_per_class.png
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AL = ROOT / "data" / "active_learning"
OUT = ROOT / "reports" / "figures" / "al_rescue_per_class.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

MACRO_LABELS = [
    "Output and Income", "Labor Market", "Housing",
    "Consumption, Orders, and Inventories", "Money and Credit",
    "Interest and Exchange Rates", "Prices", "Stock Market",
]

rounds = {1: Counter(), 2: Counter()}
for r in [1, 2]:
    log = AL / f"round_{r}" / "relabel_log.jsonl"
    if not log.exists():
        continue
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("rescued"):
            for cls in rec.get("after_macro", []):
                rounds[r][cls] += 1

# Plot grouped bars
import numpy as np
x = np.arange(len(MACRO_LABELS))
w = 0.35
r1 = [rounds[1].get(c, 0) for c in MACRO_LABELS]
r2 = [rounds[2].get(c, 0) for c in MACRO_LABELS]

fig, ax = plt.subplots(figsize=(11, 4.5))
b1 = ax.bar(x - w/2, r1, w, label=f"Round 1 (rescued={sum(r1)})", color="#1f77b4")
b2 = ax.bar(x + w/2, r2, w, label=f"Round 2 (rescued={sum(r2)})", color="#ff7f0e")

ax.set_xticks(x)
ax.set_xticklabels([c.replace(", ", ",\n") for c in MACRO_LABELS], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Articles rescued")
ax.set_title(f"Active learning rescues per macro class (out of 50 mined per class per round)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h + 0.1, f"{int(h)}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}  | round1={sum(r1)} round2={sum(r2)} total={sum(r1)+sum(r2)}")
