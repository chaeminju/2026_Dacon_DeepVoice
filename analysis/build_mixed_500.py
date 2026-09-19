#!/usr/bin/env python3
"""'음성+음악 혼합' 500개 = MUSDB18 test+train 150곡의 mixture 트랙(실제 곡 전체 믹스,
보컬+반주 동시 존재)에서 곡당 여러 구간을 랜덤 크롭."""

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
MUSDB_TEST_DIR = Path("/database/database/musdb18_wav/test")
MUSDB_TRAIN_DIR = Path("/database/database/musdb18_wav/train")

OUT_DIR = BASE_DIR / "external_data" / "mixed_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16_000
N_TOTAL = 500
WINDOW_SEC = 8.0


def load_wav_mono(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def random_window(audio, seconds, sr, rng):
    n = int(seconds * sr)
    if audio.size <= n:
        return audio
    start = rng.randint(0, audio.size - n)
    return audio[start:start + n]


def musdb_song_list():
    songs = []
    for d in (MUSDB_TEST_DIR, MUSDB_TRAIN_DIR):
        for p in d.glob("*.stem_track0.wav"):
            songs.append((d, p.name.rsplit(".stem_track", 1)[0]))
    return songs


def main():
    songs = musdb_song_list()
    rng = random.Random(42)
    print(f"total songs: {len(songs)}")

    # 150곡에서 500클립을 뽑기 위해 곡당 평균 ~3.3윈도우 필요
    plan = []
    per_song = N_TOTAL // len(songs)
    remainder = N_TOTAL - per_song * len(songs)
    for i, (d, name) in enumerate(songs):
        count = per_song + (1 if i < remainder else 0)
        for v in range(count):
            plan.append((d, name, v))
    rng.shuffle(plan)

    manifest_rows = []
    saved = 0
    for d, name, variant in plan:
        p = d / f"{name}.stem_track0.wav"
        audio, sr = load_wav_mono(p)
        window = random_window(audio, WINDOW_SEC, sr, rng)

        safe_name = name.replace("/", "_").replace(" ", "_")
        out_path = OUT_DIR / f"{safe_name}_{variant}.wav"
        sf.write(str(out_path), window.astype(np.float32), sr, subtype="PCM_16")
        manifest_rows.append({
            "category": "mixed",
            "source": "musdb18_mixture",
            "name": out_path.name,
            "duration_sec": round(window.size / sr, 3),
            "path": str(out_path.relative_to(BASE_DIR)),
        })
        saved += 1

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Total saved: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
