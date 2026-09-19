#!/usr/bin/env python3
"""
기존 카테고리 코퍼스(음성만/음악만/무음/환경음/음성+음악 혼합, 총 약 3000개)를
HTDemucs로 먼저 분리한 뒤, 그 stem(voice_audio/music_audio)에 대해 PANNs presence를
다시 측정한다. 목적: presence 계산 위치를 원본 오디오 -> 분리 후 stem으로 바꾼 새
아키텍처에 맞는 VOICE/MUSIC_PRESENT_PROB 분포를 얻어 LOW/HIGH 임계값을 재산정하기 위함.

voice_audio에서는 voice_present_prob만, music_audio에서는 music_present_prob만 사용한다
(각 stem은 자기 성분에 대한 presence만 대표하도록 아키텍처가 설계되었으므로).
참고용으로 반대쪽 값(voice_audio의 music_prob, music_audio의 voice_prob)도 같이 저장한다.
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch

from calibrate_presence_thresholds import load_panns_model, predict_presence, load_wav_mono
from separated_presence_lib import load_htdemucs_model, separate_voice_and_music

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MANIFESTS = [
    "voice_only_500",
    "music_only_500",
    "true_silence_500",
    "ambient_noise_500",
    "mixed_500",
    "speech_plus_music_500",
]


def process_manifest(name, htdemucs_model, panns_model, voice_idx, music_idx, device):
    manifest_path = BASE_DIR / "external_data" / name / "manifest.csv"
    out_path = OUTPUT_DIR / f"{name}_separated_probabilities.csv"

    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    print(f"[{name}] {len(rows)} clips")

    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            audio = load_wav_mono(row["path"])
            if audio.size == 0 or not np.isfinite(audio).all():
                print(f"  skip (invalid): {row['path']}")
                continue
            voice_audio, music_audio = separate_voice_and_music(audio, htdemucs_model, device)
            v_from_voice, m_from_voice = predict_presence(panns_model, voice_idx, music_idx, voice_audio)
            v_from_music, m_from_music = predict_presence(panns_model, voice_idx, music_idx, music_audio)
        except Exception as exc:
            print(f"  skip (error {exc}): {row['path']}")
            continue

        result = dict(row)
        result["voice_present_prob_sep"] = round(v_from_voice, 6)
        result["music_present_prob_sep"] = round(m_from_music, 6)
        result["voice_stem_music_prob"] = round(m_from_voice, 6)
        result["music_stem_voice_prob"] = round(v_from_music, 6)
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(rows)} done ({elapsed:.1f}s)")

    if not results:
        print(f"  no results for {name}, skipping write")
        return

    fieldnames = list(results[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[{name}] saved {len(results)} rows to {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    htdemucs_model = load_htdemucs_model().to(device)
    panns_model, voice_idx, music_idx = load_panns_model(device)

    for name in MANIFESTS:
        process_manifest(name, htdemucs_model, panns_model, voice_idx, music_idx, device)

    print("\nAll manifests done.")


if __name__ == "__main__":
    main()
