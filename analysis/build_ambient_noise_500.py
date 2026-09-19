#!/usr/bin/env python3
"""'환경음(아무것도 없음)' 500개 = DNS-Challenge 실제 환경 소음 클립.
음성/음악 성분은 없지만 실제로 들리는 소리(문소리, 팬소리, 엔진음, 타이핑 등)가 있는 데이터."""

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
NOISE_DIR = Path("/database/database/dns_challenge/datasets/noise")

OUT_DIR = BASE_DIR / "external_data" / "ambient_noise_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16_000
MIN_DUR, MAX_DUR = 4.0, 60.0
N_TOTAL = 500


def load_wav_mono(path, target_sr=TARGET_SR):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return audio.astype(np.float32)


def main():
    rng = random.Random(42)
    noise_files = list(NOISE_DIR.glob("*.wav"))
    rng.shuffle(noise_files)

    manifest_rows = []
    saved = 0
    for p in noise_files:
        if saved >= N_TOTAL:
            break
        audio = load_wav_mono(p)
        dur = audio.size / TARGET_SR
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue
        out_path = OUT_DIR / f"dns_{p.stem}.wav"
        sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
        manifest_rows.append({
            "category": "ambient_noise",
            "source": "dns_challenge_noise",
            "name": out_path.name,
            "duration_sec": round(dur, 3),
            "path": str(out_path.relative_to(BASE_DIR)),
            "orig_name": p.name,
        })
        saved += 1

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Total saved: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
