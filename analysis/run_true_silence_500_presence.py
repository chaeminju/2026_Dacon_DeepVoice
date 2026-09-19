#!/usr/bin/env python3
"""external_data/true_silence_500 500개 클립에 대해 PANNs VOICE/MUSIC_PRESENT_PROB를 계산한다."""

import csv
from pathlib import Path

import numpy as np
import torch

from calibrate_presence_thresholds import (
    load_panns_model,
    predict_presence,
    load_wav_mono,
)

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "external_data" / "true_silence_500" / "manifest.csv"
RESULTS_CSV = BASE_DIR / "output" / "true_silence_500_probabilities.csv"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, voice_indices, music_indices = load_panns_model(device)

    rows = list(csv.DictReader(MANIFEST_PATH.open(encoding="utf-8")))
    print(f"total clips: {len(rows)}")

    results = []
    for i, row in enumerate(rows):
        audio_path = BASE_DIR / row["path"]
        audio = load_wav_mono(audio_path)
        # 완전 무음(all-zero)은 유효성 검사(isfinite)를 통과하지만 size==0은 아님 -> 그대로 진행
        voice_prob, music_prob = predict_presence(model, voice_indices, music_indices, audio)
        result = dict(row)
        result["voice_present_prob"] = round(voice_prob, 6)
        result["music_present_prob"] = round(music_prob, 6)
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)} done")

    fieldnames = list(results[0].keys())
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
