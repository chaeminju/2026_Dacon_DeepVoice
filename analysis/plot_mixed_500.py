#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

mixed_rows = list(csv.DictReader(open(BASE_DIR / "output" / "mixed_500_probabilities.csv")))
voice_rows = list(csv.DictReader(open(BASE_DIR / "output" / "voice_only_500_probabilities.csv")))

mixed_vp = np.array([float(r["voice_present_prob"]) for r in mixed_rows])
mixed_mp = np.array([float(r["music_present_prob"]) for r in mixed_rows])
voice_only_vp = np.array([float(r["voice_present_prob"]) for r in voice_rows])

fig, axes = plt.subplots(2, 1, figsize=(9, 8))
bins = np.linspace(0, 1, 41)

ax = axes[0]
ax.hist(mixed_mp, bins=bins, color="#2ca02c", edgecolor="black", linewidth=0.3)
ax.axvspan(0.0, 0.8, color="red", alpha=0.08, label="under-detection zone")
ax.set_title("MUSIC_PRESENT_PROB on mixed(voice+music) 500 clips -> mostly fine")
ax.set_ylabel("count")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0, 1)

ax = axes[1]
ax.hist(voice_only_vp, bins=bins, alpha=0.5, label="voice_only 500 (reference)", color="#1f77b4", edgecolor="black", linewidth=0.3)
ax.hist(mixed_vp, bins=bins, alpha=0.6, label="mixed(voice+music) 500", color="#d62728", edgecolor="black", linewidth=0.3)
ax.axvspan(0.0, 0.5, color="red", alpha=0.08)
ax.set_title("VOICE_PRESENT_PROB: mixed vs voice_only -> severe suppression when music is present")
ax.set_ylabel("count")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1)
ax.set_xlabel("probability")

fig.tight_layout()
out_path = BASE_DIR / "output" / "mixed_500_distributions.png"
fig.savefig(out_path, dpi=140)
print(f"saved {out_path}")
