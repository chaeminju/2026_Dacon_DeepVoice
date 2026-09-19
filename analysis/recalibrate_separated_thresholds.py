#!/usr/bin/env python3
"""
run_separated_presence_all.py가 만든 *_separated_probabilities.csv (HTDemucs 분리 후
stem 기준 VOICE/MUSIC_PRESENT_PROB)로 새 아키텍처(presence를 분리 후 stem에서 계산)에
맞는 LOW/HIGH를 재산정한다.

채널별 부재/존재 후보 풀 정의 (기존 원본-오디오 기준 재산정과 동일한 논리, 신호만
분리 후 stem으로 교체):
  VOICE 부재 후보 = music_only + ambient_noise + true_silence 의 voice_present_prob_sep
  VOICE 존재 후보 = voice_only + speech_plus_music(전체 dB 티어) 의 voice_present_prob_sep
  MUSIC 부재 후보 = voice_only + ambient_noise + true_silence 의 music_present_prob_sep
  MUSIC 존재 후보 = music_only + speech_plus_music(전체 dB 티어) 의 music_present_prob_sep

mixed_500(MUSDB18 노래 mixture, 보컬+전체 악기)는 카테고리 정의가 다른(싱잉보컬) 별도
데이터라 임계값 산정에는 포함하지 않고, 진단용 참고치로만 같이 출력한다.
"""

import csv
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]


def load_column(name, column):
    path = OUTPUT_DIR / f"{name}_separated_probabilities.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return np.array([float(r[column]) for r in rows], dtype=np.float64)


def print_percentiles(label, values):
    pcts = np.percentile(values, PERCENTILES)
    formatted = "  ".join(f"p{p}={v:.4f}" for p, v in zip(PERCENTILES, pcts))
    print(f"{label:40s} n={len(values):5d}  {formatted}")


def recommend(absence, presence, channel_name):
    absence_p95 = np.percentile(absence, 95)
    presence_p5 = np.percentile(presence, 5)
    print(f"\n--- {channel_name} presence_weight 재산정 ---")
    print(f"부재 후보: n={len(absence)} p95={absence_p95:.4f} max={absence.max():.4f}")
    print(f"존재 후보: n={len(presence)} p5={presence_p5:.4f} min={presence.min():.4f}")
    if absence_p95 >= presence_p5:
        print(f"⚠ 두 분포가 겹칩니다 (부재 p95={absence_p95:.4f} >= 존재 p5={presence_p5:.4f})")
    low = round(float(absence_p95), 2)
    high = round(float(presence_p5), 2)
    if high <= low:
        high = round(low + 0.05, 2)
    print(f"권장 LOW={low}, HIGH={high}  (부재 p95 / 존재 p5 그대로 반올림)")
    return low, high


def main():
    print("=== VOICE channel (voice_present_prob_sep) ===")
    voice_only = load_column("voice_only_500", "voice_present_prob_sep")
    speech_plus_music_v = load_column("speech_plus_music_500", "voice_present_prob_sep")
    music_only_v = load_column("music_only_500", "voice_present_prob_sep")
    ambient_v = load_column("ambient_noise_500", "voice_present_prob_sep")
    silence_v = load_column("true_silence_500", "voice_present_prob_sep")
    mixed_v = load_column("mixed_500", "voice_present_prob_sep")

    print_percentiles("voice_only", voice_only)
    print_percentiles("speech_plus_music (all tiers)", speech_plus_music_v)
    print_percentiles("music_only", music_only_v)
    print_percentiles("ambient_noise", ambient_v)
    print_percentiles("true_silence", silence_v)
    print_percentiles("[참고] mixed(musdb 노래)", mixed_v)

    voice_absence = np.concatenate([music_only_v, ambient_v, silence_v])
    voice_presence = np.concatenate([voice_only, speech_plus_music_v])
    voice_low, voice_high = recommend(voice_absence, voice_presence, "VOICE")

    print("\n=== MUSIC channel (music_present_prob_sep) ===")
    voice_only_m = load_column("voice_only_500", "music_present_prob_sep")
    speech_plus_music_m = load_column("speech_plus_music_500", "music_present_prob_sep")
    music_only = load_column("music_only_500", "music_present_prob_sep")
    ambient_m = load_column("ambient_noise_500", "music_present_prob_sep")
    silence_m = load_column("true_silence_500", "music_present_prob_sep")
    mixed_m = load_column("mixed_500", "music_present_prob_sep")

    print_percentiles("voice_only", voice_only_m)
    print_percentiles("speech_plus_music (all tiers)", speech_plus_music_m)
    print_percentiles("music_only", music_only)
    print_percentiles("ambient_noise", ambient_m)
    print_percentiles("true_silence", silence_m)
    print_percentiles("[참고] mixed(musdb 노래)", mixed_m)

    music_absence = np.concatenate([voice_only_m, ambient_m, silence_m])
    music_presence = np.concatenate([music_only, speech_plus_music_m])
    music_low, music_high = recommend(music_absence, music_presence, "MUSIC")

    print("\n=== 최종 권장값 ===")
    print(f"VOICE_PRESENCE_LOW = {voice_low}")
    print(f"VOICE_PRESENCE_HIGH = {voice_high}")
    print(f"MUSIC_PRESENCE_LOW = {music_low}")
    print(f"MUSIC_PRESENCE_HIGH = {music_high}")


if __name__ == "__main__":
    main()
