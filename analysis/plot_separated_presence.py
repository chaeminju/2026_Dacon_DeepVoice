#!/usr/bin/env python3
"""HTDemucs 분리 후 stem 기준 VOICE/MUSIC_PRESENT_PROB 분포를 카테고리별로 시각화한다.
summarize_thresholds.py의 presence_distributions.png와 같은 스타일이되,
분리 전(원본 오디오) 대신 분리 후(stem) 값을 그리고 새 LOW/HIGH 임계값을 함께 표시한다."""

from pathlib import Path

import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PLOT_PATH = OUTPUT_DIR / "separated_presence_distributions.png"

# submit/script.py와 동일한 재보정 값
VOICE_LOW, VOICE_HIGH = 0.40, 0.76
MUSIC_LOW, MUSIC_HIGH = 0.35, 0.71

CORE_CATEGORIES = [
    ("voice_only_500", "voice only", "#1f77b4"),
    ("music_only_500", "music only", "#d62728"),
    ("true_silence_500", "true silence", "#7f7f7f"),
    ("ambient_noise_500", "ambient noise", "#9467bd"),
    ("speech_plus_music_500", "voice+music mixed", "#2ca02c"),
]
# 참고용(임계값 산정에는 미포함, 별도 스타일로만 표시)
REFERENCE_CATEGORIES = [
    ("mixed_500", "[ref] musdb song mixture", "#ff7f0e"),
]


def load_column(name, column):
    path = OUTPUT_DIR / f"{name}_separated_probabilities.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.array([float(r[column]) for r in rows], dtype=np.float64)


def draw_thresholds(ax, low, high):
    ax.axvline(low, color="black", linestyle="--", linewidth=1)
    ax.axvline(high, color="black", linestyle="--", linewidth=1)
    ax.axvspan(low, high, color="yellow", alpha=0.12)
    ymax = ax.get_ylim()[1]
    ax.text(low, ymax * 0.95, f"LOW={low}", rotation=90, va="top", ha="right", fontsize=8)
    ax.text(high, ymax * 0.95, f"HIGH={high}", rotation=90, va="top", ha="right", fontsize=8)


def plot():
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    bins = np.linspace(0, 1, 31)

    ax = axes[0]
    for name, label, color in CORE_CATEGORIES:
        values = load_column(name, "voice_present_prob_sep")
        ax.hist(values, bins=bins, alpha=0.5, label=f"{label} (n={len(values)})",
                color=color, edgecolor="black", linewidth=0.3)
    for name, label, color in REFERENCE_CATEGORIES:
        values = load_column(name, "voice_present_prob_sep")
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, linestyle=":",
                label=f"{label} (n={len(values)})", color=color)
    ax.set_title("VOICE_PRESENT_PROB (voice_audio stem, post-separation)")
    ax.set_ylabel("count")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.set_xlim(0, 1)
    draw_thresholds(ax, VOICE_LOW, VOICE_HIGH)

    ax = axes[1]
    for name, label, color in CORE_CATEGORIES:
        values = load_column(name, "music_present_prob_sep")
        ax.hist(values, bins=bins, alpha=0.5, label=f"{label} (n={len(values)})",
                color=color, edgecolor="black", linewidth=0.3)
    for name, label, color in REFERENCE_CATEGORIES:
        values = load_column(name, "music_present_prob_sep")
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, linestyle=":",
                label=f"{label} (n={len(values)})", color=color)
    ax.set_title("MUSIC_PRESENT_PROB (music_audio stem, post-separation)")
    ax.set_ylabel("count")
    ax.set_xlabel("probability")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.set_xlim(0, 1)
    draw_thresholds(ax, MUSIC_LOW, MUSIC_HIGH)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    plot()
