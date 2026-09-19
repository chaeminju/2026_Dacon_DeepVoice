#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(BASE_DIR / "output" / "speech_plus_music_500_probabilities.csv")))
voice_only_rows = list(csv.DictReader(open(BASE_DIR / "output" / "voice_only_500_probabilities.csv")))
mixed_singing_rows = list(csv.DictReader(open(BASE_DIR / "output" / "mixed_500_probabilities.csv")))

groups = defaultdict(list)
for r in rows:
    groups[float(r["music_relative_db"])].append(float(r["voice_present_prob"]))

levels = sorted(groups.keys())
data = [groups[l] for l in levels]

voice_only_vp = [float(r["voice_present_prob"]) for r in voice_only_rows]
mixed_singing_vp = [float(r["voice_present_prob"]) for r in mixed_singing_rows]

fig, ax = plt.subplots(figsize=(9, 6))
positions = list(range(len(levels)))
bp = ax.boxplot(data, positions=positions, widths=0.6, showfliers=True,
                 patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#1f77b4")
    patch.set_alpha(0.5)

ax.axhline(np.median(voice_only_vp), color="green", linestyle="--", linewidth=1.5,
            label=f"voice_only median ({np.median(voice_only_vp):.3f}, no music)")
ax.axhline(np.median(mixed_singing_vp), color="red", linestyle="--", linewidth=1.5,
            label=f"MUSDB singing-mixed median ({np.median(mixed_singing_vp):.3f})")

ax.set_xticks(positions)
ax.set_xticklabels([f"{l:+.0f} dB" for l in levels])
ax.set_xlabel("music loudness relative to speech (dB)")
ax.set_ylabel("VOICE_PRESENT_PROB")
ax.set_title("Spoken speech + background music: VOICE_PRESENT_PROB vs music loudness")
ax.set_ylim(0, 1)
ax.legend(loc="lower left", fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
out_path = BASE_DIR / "output" / "speech_plus_music_500_distributions.png"
fig.savefig(out_path, dpi=140)
print(f"saved {out_path}")
