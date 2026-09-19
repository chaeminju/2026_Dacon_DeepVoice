#!/usr/bin/env python3
"""MUSIC_PRESENCE_LOW/HIGH를 낮추면 ambient_noise/true_silence stem이 더 높은
music_weight를 받게 되는데, 그게 실제로 위험한지 확인하기 위해 DF-Arena-1B가 그
stem들에 실제로 어떤 MUSIC_FAKE_PROB를 매기는지 샘플링해서 측정한다."""

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

import script as submit_script  # submit/script.py
from calibrate_presence_thresholds import load_wav_mono
from separated_presence_lib import load_htdemucs_model, separate_voice_and_music

BASE_DIR = Path(__file__).resolve().parent
N_SAMPLE = 100
random.seed(42)


def sample_manifest(name, n):
    manifest_path = BASE_DIR / "external_data" / name / "manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    random.shuffle(rows)
    return rows[:n]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    htdemucs_model = load_htdemucs_model().to(device)
    df_arena_model, fake_label_index = submit_script.load_df_arena_model(device)

    for name in ["ambient_noise_500", "true_silence_500"]:
        rows = sample_manifest(name, N_SAMPLE)
        music_fakes = []
        voice_fakes = []
        for row in rows:
            audio = load_wav_mono(row["path"])
            voice_audio, music_audio = separate_voice_and_music(audio, htdemucs_model, device)
            music_fake = submit_script.predict_fake(df_arena_model, fake_label_index, music_audio, device)
            voice_fake = submit_script.predict_fake(df_arena_model, fake_label_index, voice_audio, device)
            music_fakes.append(music_fake)
            voice_fakes.append(voice_fake)

        music_fakes = np.array(music_fakes)
        voice_fakes = np.array(voice_fakes)
        print(f"\n=== {name} (n={len(rows)}) ===")
        print(f"MUSIC_FAKE_PROB: mean={music_fakes.mean():.4f} median={np.median(music_fakes):.4f} "
              f"p90={np.percentile(music_fakes,90):.4f} max={music_fakes.max():.4f}")
        print(f"VOICE_FAKE_PROB: mean={voice_fakes.mean():.4f} median={np.median(voice_fakes):.4f} "
              f"p90={np.percentile(voice_fakes,90):.4f} max={voice_fakes.max():.4f}")


if __name__ == "__main__":
    main()
