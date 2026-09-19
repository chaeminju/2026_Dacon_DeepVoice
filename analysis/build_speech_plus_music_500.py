#!/usr/bin/env python3
"""
'일반 대화 음성 + 배경 음악' 혼합 500개.

MUSDB18(노래) 기반 '혼합' 데이터와 달리, 이번엔 실제 발화 음성(voice_only_500: Common Voice +
Emilia-YODAS)과 보컬 없는 실제 반주(music_only_500: MUSDB18 accompaniment + JamendoMaxCaps
instrumental)를 합성으로 섞어서 '통화 중 배경에 음악이 깔린' 상황을 재현한다.

음악을 speech 대비 몇 dB로 섞느냐에 따라 5단계로 나눈다 (100개씩):
  -10dB(살짝 깔림) / -5dB / 0dB(동등) / +5dB / +10dB(음악이 더 큼)
"""

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
VOICE_MANIFEST = BASE_DIR / "external_data" / "voice_only_500" / "manifest.csv"
MUSIC_MANIFEST = BASE_DIR / "external_data" / "music_only_500" / "manifest.csv"

OUT_DIR = BASE_DIR / "external_data" / "speech_plus_music_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16_000
SNR_LEVELS_DB = [-10.0, -5.0, 0.0, 5.0, 10.0]  # 양수일수록 음악이 더 큼
N_PER_LEVEL = 100


def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def fit_length(music, target_len, rng):
    if music.size >= target_len:
        start = rng.randint(0, music.size - target_len) if music.size > target_len else 0
        return music[start:start + target_len]
    reps = target_len // music.size + 1
    return np.tile(music, reps)[:target_len]


def mix_at_snr(speech, music, music_relative_db, rng):
    music = fit_length(music, speech.size, rng)
    speech_rms = rms(speech)
    music_rms = rms(music) + 1e-9
    target_music_rms = speech_rms * (10.0 ** (music_relative_db / 20.0))
    music_scaled = music * (target_music_rms / music_rms)
    mixed = speech + music_scaled

    peak = np.abs(mixed).max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def main():
    voice_rows = list(csv.DictReader(VOICE_MANIFEST.open(encoding="utf-8")))
    music_rows = list(csv.DictReader(MUSIC_MANIFEST.open(encoding="utf-8")))

    rng = random.Random(42)
    py_np_rng = np.random.RandomState(42)

    rng.shuffle(voice_rows)
    rng.shuffle(music_rows)

    manifest_rows = []
    idx = 0
    for level_idx, level_db in enumerate(SNR_LEVELS_DB):
        for i in range(N_PER_LEVEL):
            v_row = voice_rows[idx % len(voice_rows)]
            m_row = music_rows[idx % len(music_rows)]

            speech = load_wav(BASE_DIR / v_row["path"])
            music = load_wav(BASE_DIR / m_row["path"])
            mixed = mix_at_snr(speech, music, level_db, py_np_rng)

            level_tag = f"{level_db:+.0f}dB".replace("+", "p").replace("-", "m")
            out_path = OUT_DIR / f"mix_{level_tag}_{i:03d}.wav"
            sf.write(str(out_path), mixed, TARGET_SR, subtype="PCM_16")

            manifest_rows.append({
                "category": "speech_plus_music",
                "music_relative_db": level_db,
                "name": out_path.name,
                "duration_sec": round(mixed.size / TARGET_SR, 3),
                "path": str(out_path.relative_to(BASE_DIR)),
                "voice_source": v_row.get("source", ""),
                "voice_bucket": v_row.get("bucket", ""),
                "voice_language": v_row.get("language", ""),
                "music_source": m_row.get("source", ""),
                "music_bucket": m_row.get("bucket", ""),
            })
            idx += 1

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Total generated: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
