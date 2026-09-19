#!/usr/bin/env python3
"""
음악만(music_only) 외부 오픈소스 데이터 500개 구성.

- 기존(established) 250개 = MUSDB18 (2017) test+train 150곡의 drums+bass+other 합(보컬 제외 반주)
- 최신(recent) 250개    = JamendoMaxCaps (2023~2024 발매곡, Jamendo 실제 아티스트 instrumental 트랙)

모든 클립은 4~60초로 크롭 후 16kHz mono wav로 저장한다.
"""

import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import requests

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
MUSDB_TEST_DIR = Path("/database/database/musdb18_wav/test")
MUSDB_TRAIN_DIR = Path("/database/database/musdb18_wav/train")

OUT_DIR = BASE_DIR / "external_data" / "music_only_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

JAMENDO_CACHE = BASE_DIR / "external_data" / "jamendo_maxcaps"
JAMENDO_CACHE.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16_000
MIN_DUR, MAX_DUR = 4.0, 60.0
N_ESTABLISHED = 250
N_RECENT = 250

JAMENDO_BASE = "https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps/resolve/main/"
# 최근 발매곡 위주로 수집하기 위해 크롤 범위(2008-01-01 ~ 2024-01-02) 끝부분 shard들을 사용
RECENT_SHARD_DATES = [
    "2023-06-01", "2023-06-21", "2023-07-11", "2023-07-31", "2023-08-20",
    "2023-09-09", "2023-09-29", "2023-10-19", "2023-11-08", "2023-11-28",
    "2023-12-18", "2024-01-02",
]


def resample_and_save(audio, sr, out_path):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    sf.write(str(out_path), audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
    return audio.size / TARGET_SR


def load_wav_mono_local(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def random_window(audio, seconds, rng):
    n = int(seconds * TARGET_SR)
    if audio.size <= n:
        return audio
    start = rng.randint(0, audio.size - n)
    return audio[start:start + n]


# -----------------------------------------------------------------------------
# 1) MUSDB18 (established)
# -----------------------------------------------------------------------------
def musdb_song_list():
    songs = []
    for d in (MUSDB_TEST_DIR, MUSDB_TRAIN_DIR):
        for p in d.glob("*.stem_track0.wav"):
            songs.append((d, p.name.rsplit(".stem_track", 1)[0]))
    return songs


def build_musdb(manifest_rows):
    songs = musdb_song_list()
    rng = random.Random(42)
    rng.shuffle(songs)

    # 150곡에서 250클립을 뽑기 위해 일부 곡은 2윈도우 사용
    plan = []
    for i, (d, name) in enumerate(songs):
        plan.append((d, name, 0))
    extra_needed = N_ESTABLISHED - len(plan)
    for i in range(extra_needed):
        d, name = songs[i % len(songs)]
        plan.append((d, name, 1))
    rng.shuffle(plan)
    plan = plan[:N_ESTABLISHED]

    out_dir = OUT_DIR / "musdb18"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for d, name, variant in plan:
        parts = []
        for track_idx in (1, 2, 3):
            p = d / f"{name}.stem_track{track_idx}.wav"
            audio, sr = load_wav_mono_local(p)
            parts.append(audio)
        min_len = min(a.size for a in parts)
        mix = np.sum([a[:min_len] for a in parts], axis=0).astype(np.float32)
        window = random_window(mix, 8.0, rng)

        safe_name = name.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe_name}_{variant}.wav"
        real_dur = resample_and_save(window, TARGET_SR, out_path)
        manifest_rows.append({
            "category": "music_only",
            "bucket": "established",
            "source": "musdb18",
            "name": out_path.name,
            "duration_sec": round(real_dur, 3),
            "path": str(out_path.relative_to(BASE_DIR)),
        })
        saved += 1
    print(f"[musdb18] saved {saved}/{N_ESTABLISHED}")


# -----------------------------------------------------------------------------
# 2) JamendoMaxCaps (recent)
# -----------------------------------------------------------------------------
def fetch_jsonl(date_str):
    cache_path = JAMENDO_CACHE / f"{date_str}.jsonl"
    if not cache_path.exists():
        url = JAMENDO_BASE + f"{date_str}.jsonl"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    lines = cache_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def build_jamendo(manifest_rows):
    rng = random.Random(42)

    candidates = []
    for date_str in RECENT_SHARD_DATES:
        try:
            tracks = fetch_jsonl(date_str)
        except Exception as exc:
            print(f"skip shard {date_str}: {exc}")
            continue
        for t in tracks:
            dur = t.get("duration")
            if dur is None or not (30 <= dur <= 600):
                continue
            if not t.get("audiodownload"):
                continue
            candidates.append(t)
    print(f"jamendo candidates collected: {len(candidates)}")
    rng.shuffle(candidates)

    out_dir = OUT_DIR / "jamendo_maxcaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    attempted = 0
    for t in candidates:
        if saved >= N_RECENT:
            break
        attempted += 1
        try:
            resp = requests.get(t["audiodownload"], timeout=30)
            resp.raise_for_status()
            audio, sr = sf.read(io.BytesIO(resp.content))
        except Exception as exc:
            continue

        if audio.ndim > 1:
            audio_mono = audio.mean(axis=1)
        else:
            audio_mono = audio
        if audio_mono.size / sr < MIN_DUR + 2:
            continue

        # 인트로/아웃트로를 피해 중간 구간에서 8~15초 랜덤 크롭
        crop_len_sec = rng.uniform(8.0, 15.0)
        n = int(crop_len_sec * sr)
        margin = int(sr * min(5.0, max(0.0, (audio_mono.size / sr) * 0.1)))
        lo = margin
        hi = max(lo + 1, audio_mono.size - n - margin)
        if hi <= lo:
            start = 0
        else:
            start = rng.randint(lo, hi)
        window = audio_mono[start:start + n].astype(np.float32)
        if window.size == 0:
            continue

        out_path = out_dir / f"jamendo_{t['id']}.wav"
        real_dur = resample_and_save(window, sr, out_path)
        if not (MIN_DUR <= real_dur <= MAX_DUR):
            out_path.unlink(missing_ok=True)
            continue

        manifest_rows.append({
            "category": "music_only",
            "bucket": "recent",
            "source": "jamendo_maxcaps",
            "name": out_path.name,
            "duration_sec": round(real_dur, 3),
            "path": str(out_path.relative_to(BASE_DIR)),
            "releasedate": t.get("releasedate", ""),
        })
        saved += 1
        if saved % 25 == 0:
            print(f"  jamendo saved {saved}/{N_RECENT} (attempted {attempted})")

    print(f"[jamendo_maxcaps] saved {saved}/{N_RECENT} (attempted {attempted})")


def main():
    manifest_rows = []
    build_musdb(manifest_rows)
    build_jamendo(manifest_rows)

    fieldnames = sorted({k for r in manifest_rows for k in r.keys()})
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nTotal saved: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
