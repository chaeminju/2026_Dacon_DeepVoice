#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(BASE_DIR / "output" / "ambient_noise_500_probabilities.csv")))

mp = np.array([float(r["music_present_prob"]) for r in rows])
vp = np.array([float(r["voice_present_prob"]) for r in rows])

fig, axes = plt.subplots(2, 1, figsize=(9, 8))
bins = np.linspace(0, 1, 41)

for ax, vals, title in (
    (axes[0], mp, "MUSIC_PRESENT_PROB (ambient noise 500 clips, expect -> 0)"),
    (axes[1], vp, "VOICE_PRESENT_PROB (ambient noise 500 clips, expect -> 0)"),
):
    ax.hist(vals, bins=bins, color="#7f7f7f", edgecolor="black", linewidth=0.3)
    ax.axvspan(0.5, 1.0, color="red", alpha=0.10, label="severe false-positive zone")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_xlim(0, 1)

axes[1].set_xlabel("probability")
fig.tight_layout()
out_path = BASE_DIR / "output" / "ambient_noise_500_distributions.png"
fig.savefig(out_path, dpi=140)
print(f"saved {out_path}")
