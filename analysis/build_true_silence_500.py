#!/usr/bin/env python3
"""
'아무 소리도 없는' 데이터 500개 = 실제 디지털 무음 + 다양한 레벨의 잡음 바닥(noise floor).

이전 '아무것도 없음' 파일럿(DNS noise: 문소리/팬소리 등 실제 환경음)과는 다르게,
이번엔 정말로 '아무 소리도 잡히지 않는' 상태 — 완전 무음과, 실제 녹음환경에서
있을 수 있는 초저레벨 잡음(magnetic/전자 노이즈 floor)을 합성해서 만든다.

레벨 구성 (500개):
  - 완전 디지털 무음(전부 0)                : 100개
  - -80dB / -70dB / -60dB / -50dB / -40dB  : 각 80개 (가우시안 화이트 노이즈 floor)

길이는 대회 규격과 동일하게 4~60초 사이 랜덤.
"""

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "external_data" / "true_silence_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16_000
MIN_DUR, MAX_DUR = 4.0, 60.0

LEVELS = [
    ("zero", None, 100),
    ("-80dB", -80.0, 80),
    ("-70dB", -70.0, 80),
    ("-60dB", -60.0, 80),
    ("-50dB", -50.0, 80),
    ("-40dB", -40.0, 80),
]


def make_clip(level_db, duration_sec, rng):
    n = int(duration_sec * TARGET_SR)
    if level_db is None:
        audio = np.zeros(n, dtype=np.float32)
    else:
        amplitude = 10.0 ** (level_db / 20.0)
        audio = rng.normal(0.0, amplitude, n).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
    return audio


def main():
    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    manifest_rows = []
    idx = 0
    for level_name, level_db, count in LEVELS:
        level_dir = OUT_DIR / level_name
        level_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            duration = py_rng.uniform(MIN_DUR, MAX_DUR)
            audio = make_clip(level_db, duration, rng)
            out_path = level_dir / f"{level_name}_{i:03d}.wav"
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
            manifest_rows.append({
                "category": "nothing_true_silence",
                "level": level_name,
                "level_db": level_db if level_db is not None else "-inf",
                "name": out_path.name,
                "duration_sec": round(audio.size / TARGET_SR, 3),
                "rms": round(rms, 8),
                "path": str(out_path.relative_to(BASE_DIR)),
            })
            idx += 1

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Total generated: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
