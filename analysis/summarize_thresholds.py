#!/usr/bin/env python3
"""presence_probabilities.csv를 카테고리별로 요약하고,
presence_weight(low, high)의 low/high 재산정안을 계산 + 시각화한다."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent / "output" / "presence_probabilities.csv"
PLOT_PATH = Path(__file__).resolve().parent / "output" / "presence_distributions.png"

CATEGORY_LABELS = {
    "music_only": "음악만",
    "voice_only": "음성만",
    "nothing": "아무것도 없음",
    "mixed": "음성+음악 혼합",
}
# matplotlib 환경에 한글 폰트가 없어 그래프 라벨은 영문으로 표기
PLOT_LABELS = {
    "music_only": "music only",
    "voice_only": "voice only",
    "nothing": "nothing",
    "mixed": "voice+music mixed",
}
CATEGORY_ORDER = ["music_only", "voice_only", "nothing", "mixed"]
PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]


def load_rows():
    by_cat = defaultdict(lambda: {"voice": [], "music": []})
    with CSV_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_cat[row["category"]]["voice"].append(float(row["voice_present_prob"]))
            by_cat[row["category"]]["music"].append(float(row["music_present_prob"]))
    return by_cat


def print_summary(by_cat):
    for channel in ("voice", "music"):
        print(f"\n=== {channel.upper()}_PRESENT_PROB ===")
        header = f"{'category':16s}" + "".join(f"p{p:<6d}" for p in PERCENTILES) + "   n"
        print(header)
        for cat in CATEGORY_ORDER:
            values = np.array(by_cat[cat][channel])
            pct = np.percentile(values, PERCENTILES)
            line = f"{CATEGORY_LABELS[cat]:16s}" + "".join(f"{v:<7.3f}" for v in pct) + f"  {len(values)}"
            print(line)


def recommend_thresholds(by_cat):
    # voice: 존재(voice_only, mixed) vs 부재(music_only, nothing)
    voice_present = np.array(by_cat["voice_only"]["voice"] + by_cat["mixed"]["voice"])
    voice_absent = np.array(by_cat["music_only"]["voice"] + by_cat["nothing"]["voice"])

    # music: 존재(music_only, mixed) vs 부재(voice_only, nothing)
    music_present = np.array(by_cat["music_only"]["music"] + by_cat["mixed"]["music"])
    music_absent = np.array(by_cat["voice_only"]["music"] + by_cat["nothing"]["music"])

    def suggest(present, absent, label):
        # low: 부재 분포의 상위 percentile (이 값 밑이면 거의 확실히 '부재')
        # high: 존재 분포의 하위 percentile (이 값 위면 거의 확실히 '존재')
        low_p95 = np.percentile(absent, 95)
        high_p05 = np.percentile(present, 5)
        low_max = absent.max()
        high_min = present.min()
        overlap = low_p95 >= high_p05
        print(f"\n--- {label} ---")
        print(f"부재 분포: min={absent.min():.3f} p50={np.median(absent):.3f} p95={low_p95:.3f} max={low_max:.3f}")
        print(f"존재 분포: min={high_min:.3f} p50={np.median(present):.3f} p5={high_p05:.3f} max={present.max():.3f}")
        if overlap:
            print(f"⚠ 두 분포가 겹칩니다 (부재 p95={low_p95:.3f} >= 존재 p5={high_p05:.3f}). "
                  f"완전 분리 임계값은 없음 — p95/p5 기준으로 근사값 사용 권장.")
        low = round(float(low_p95), 2)
        high = round(float(high_p05), 2)
        if low >= high:
            # 최소한의 폭 보장
            mid = (low + high) / 2
            low, high = round(mid - 0.05, 2), round(mid + 0.05, 2)
        print(f"권장 LOW={low}, HIGH={high} (기존 코드: LOW=0.15, HIGH=0.55)")
        return low, high

    voice_low, voice_high = suggest(voice_present, voice_absent, "VOICE presence_weight")
    music_low, music_high = suggest(music_present, music_absent, "MUSIC presence_weight")
    return (voice_low, voice_high), (music_low, music_high)


def plot(by_cat):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = {"music_only": "#d62728", "voice_only": "#1f77b4",
              "nothing": "#7f7f7f", "mixed": "#2ca02c"}
    bins = np.linspace(0, 1, 21)

    for ax, channel, title in (
        (axes[0], "voice", "VOICE_PRESENT_PROB distribution"),
        (axes[1], "music", "MUSIC_PRESENT_PROB distribution"),
    ):
        for cat in CATEGORY_ORDER:
            values = by_cat[cat][channel]
            ax.hist(values, bins=bins, alpha=0.5, label=PLOT_LABELS[cat],
                     color=colors[cat], edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.legend(loc="upper center", ncol=4, fontsize=8)
        ax.set_xlim(0, 1)

    axes[1].set_xlabel("probability")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=140)
    print(f"\nSaved plot to {PLOT_PATH}")


def main():
    by_cat = load_rows()
    print_summary(by_cat)
    recommend_thresholds(by_cat)
    plot(by_cat)


if __name__ == "__main__":
    main()
