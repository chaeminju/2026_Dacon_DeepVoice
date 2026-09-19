#!/usr/bin/env python3
"""5개 핵심 카테고리 전체에서 DF-Arena fake score를 샘플링해서, presence_weight
LOW/HIGH 선택이 실제 FILE_FAKE_PROB에 미치는 end-to-end 영향을 정확히 시뮬레이션
하기 위한 데이터를 만든다. (ambient_noise/true_silence만 봤던 check_ambient_fake_scores.py
를 voice_only/music_only/speech_plus_music까지 확장)"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
SUBMIT_DIR = REPO_DIR / "submit"
sys.path.insert(0, str(SUBMIT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import script as submit_script
from calibrate_presence_thresholds import load_wav_mono, load_panns_model, predict_presence
from separated_presence_lib import load_htdemucs_model, separate_voice_and_music

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
N_SAMPLE = 120
random.seed(42)

CATEGORIES = [
    "voice_only_500",
    "music_only_500",
    "true_silence_500",
    "ambient_noise_500",
    "speech_plus_music_500",
]


def sample_manifest(name, n):
    manifest_path = BASE_DIR / "external_data" / name / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    random.Random(42).shuffle(rows)
    return rows[:n]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    htdemucs_model = load_htdemucs_model().to(device)
    panns_model, voice_idx, music_idx = load_panns_model(device)
    df_arena_model, fake_label_index = submit_script.load_df_arena_model(device)

    for name in CATEGORIES:
        rows = sample_manifest(name, N_SAMPLE)
        out_path = OUTPUT_DIR / f"{name}_full_pipeline_sample.csv"
        results = []
        for row in rows:
            audio = load_wav_mono(row["path"])
            voice_audio, music_audio = separate_voice_and_music(audio, htdemucs_model, device)
            voice_present, _ = predict_presence(panns_model, voice_idx, music_idx, voice_audio)
            _, music_present = predict_presence(panns_model, voice_idx, music_idx, music_audio)
            voice_fake = submit_script.predict_fake(df_arena_model, fake_label_index, voice_audio, device)
            music_fake = submit_script.predict_fake(df_arena_model, fake_label_index, music_audio, device)
            results.append({
                "category": row["category"],
                "path": row["path"],
                "voice_present_prob_sep": round(voice_present, 6),
                "music_present_prob_sep": round(music_present, 6),
                "voice_fake_prob": round(voice_fake, 6),
                "music_fake_prob": round(music_fake, 6),
            })
        fieldnames = list(results[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[{name}] saved {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
