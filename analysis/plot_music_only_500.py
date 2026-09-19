#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(BASE_DIR / "output" / "music_only_500_probabilities.csv")))

mp = np.array([float(r["music_present_prob"]) for r in rows])
vp = np.array([float(r["voice_present_prob"]) for r in rows])
source = [r["source"] for r in rows]

fig, axes = plt.subplots(2, 1, figsize=(9, 8))
bins = np.linspace(0, 1, 41)

colors = {"musdb18": "#d62728", "jamendo_maxcaps": "#2ca02c"}
for ax, vals, title, err_zone in (
    (axes[0], mp, "MUSIC_PRESENT_PROB (music_only 500 clips, expect -> 1)", (0, 0.8)),
    (axes[1], vp, "VOICE_PRESENT_PROB (music_only 500 clips, expect -> 0)", (0.1, 1.0)),
):
    for src in ("musdb18", "jamendo_maxcaps"):
        v = [vals[i] for i in range(len(vals)) if source[i] == src]
        ax.hist(v, bins=bins, alpha=0.55, label=src, color=colors[src], edgecolor="black", linewidth=0.3)
    ax.axvspan(err_zone[0], err_zone[1], color="red", alpha=0.08, label="error-prone zone")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.set_xlim(0, 1)

axes[1].set_xlabel("probability")
fig.tight_layout()
out_path = BASE_DIR / "output" / "music_only_500_distributions.png"
fig.savefig(out_path, dpi=140)
print(f"saved {out_path}")
