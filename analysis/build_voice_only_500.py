#!/usr/bin/env python3
"""
음성만(voice_only) 외부 오픈소스 데이터 500개 구성.

- 기존(established) 250개 = Common Voice 17.0 (fsicoli mirror, CC0) 한국어 125 + 영어 125
- 최신(recent, 2025)  250개 = Emilia-YODAS 필터링본 한국어 125(seastar105) + 영어 125(MrDragonFox)

모든 클립은 4~60초 범위로 필터링 후 16kHz mono wav로 저장한다.
원본 아카이브(tar/parquet)는 이미 analysis/external_data/ 아래에 받아둔 상태에서 실행한다.
"""

import csv
import io
import random
import tarfile
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyarrow.parquet as pq

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
EXT_DIR = BASE_DIR / "external_data"
CV_DIR = EXT_DIR / "cv17"
EMILIA_DIR = EXT_DIR / "emilia_yodas"

OUT_DIR = BASE_DIR / "external_data" / "voice_only_500"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT_DIR / "manifest.csv"

TARGET_SR = 16_000
MIN_DUR, MAX_DUR = 4.0, 60.0
N_PER_SOURCE = 125


def resample_and_save(audio, sr, out_path):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    sf.write(str(out_path), audio.astype(np.float32), TARGET_SR, subtype="PCM_16")
    return audio.size / TARGET_SR


# -----------------------------------------------------------------------------
# 1) Common Voice 17.0 (established)
# -----------------------------------------------------------------------------
def load_cv_durations(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d[row["clip"]] = int(row["duration[ms]"]) / 1000.0
    return d


def qualifying_cv_clips(tsv_path, dur_map):
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d = dur_map.get(row["path"])
            if d is not None and MIN_DUR <= d <= MAX_DUR:
                rows.append((row["path"], d))
    return rows


def build_common_voice(lang, tar_name, split_tsv, manifest_rows):
    dur_map = load_cv_durations(CV_DIR / f"{lang}_clip_durations.tsv")
    candidates = qualifying_cv_clips(CV_DIR / split_tsv, dur_map)
    rng = random.Random(42)
    rng.shuffle(candidates)
    chosen = candidates[:N_PER_SOURCE]
    chosen_names = {name for name, _ in chosen}

    out_lang_dir = OUT_DIR / "common_voice" / lang
    out_lang_dir.mkdir(parents=True, exist_ok=True)

    tar_path = CV_DIR / tar_name
    saved = 0
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            base = member.name.split("/")[-1]
            if base not in chosen_names:
                continue
            data = tf.extractfile(member).read()
            audio, sr = sf.read(io.BytesIO(data))
            out_path = out_lang_dir / (Path(base).stem + ".wav")
            real_dur = resample_and_save(audio, sr, out_path)
            manifest_rows.append({
                "category": "voice_only",
                "bucket": "established",
                "source": f"common_voice_17_0_{lang}",
                "language": lang,
                "name": out_path.name,
                "duration_sec": round(real_dur, 3),
                "path": str(out_path.relative_to(BASE_DIR)),
            })
            saved += 1
    print(f"[common_voice/{lang}] saved {saved}/{N_PER_SOURCE}")


# -----------------------------------------------------------------------------
# 2) Emilia-YODAS (recent)
# -----------------------------------------------------------------------------
def build_emilia_ko(manifest_rows):
    tar_path = EMILIA_DIR / "ko_shard0.tar"
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
        json_names = [n for n in names if n.endswith(".json")]

        candidates = []
        for n in json_names:
            meta = json.loads(tf.extractfile(n).read())
            if MIN_DUR <= meta["duration"] <= MAX_DUR:
                candidates.append((n[:-5], meta["duration"]))  # strip .json -> uuid stem

        rng = random.Random(42)
        rng.shuffle(candidates)
        chosen = candidates[:N_PER_SOURCE]

        out_dir = OUT_DIR / "emilia_yodas" / "ko"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for stem, expected_dur in chosen:
            data = tf.extractfile(f"{stem}.mp3").read()
            audio, sr = sf.read(io.BytesIO(data))
            out_path = out_dir / f"{stem}.wav"
            real_dur = resample_and_save(audio, sr, out_path)
            manifest_rows.append({
                "category": "voice_only",
                "bucket": "recent",
                "source": "emilia_yodas_ko",
                "language": "ko",
                "name": out_path.name,
                "duration_sec": round(real_dur, 3),
                "path": str(out_path.relative_to(BASE_DIR)),
            })
            saved += 1
    print(f"[emilia_yodas/ko] saved {saved}/{N_PER_SOURCE}")


def build_emilia_en(manifest_rows):
    parquet_path = EMILIA_DIR / "en_shard0.parquet"
    table = pq.read_table(parquet_path, columns=["audio", "duration", "file_id"])
    durations = table.column("duration").to_numpy()
    mask = (durations >= MIN_DUR) & (durations <= MAX_DUR)
    idx = np.nonzero(mask)[0]

    rng = random.Random(42)
    rng.shuffle(idx.tolist())
    idx = list(idx)
    rng.shuffle(idx)
    chosen_idx = idx[:N_PER_SOURCE]

    audio_col = table.column("audio")
    file_id_col = table.column("file_id")

    out_dir = OUT_DIR / "emilia_yodas" / "en"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in chosen_idx:
        audio_struct = audio_col[i].as_py()
        data = audio_struct["bytes"]
        audio, sr = sf.read(io.BytesIO(data))
        file_id = file_id_col[i].as_py()
        out_path = out_dir / f"{file_id}.wav"
        real_dur = resample_and_save(audio, sr, out_path)
        manifest_rows.append({
            "category": "voice_only",
            "bucket": "recent",
            "source": "emilia_yodas_en",
            "language": "en",
            "name": out_path.name,
            "duration_sec": round(real_dur, 3),
            "path": str(out_path.relative_to(BASE_DIR)),
        })
        saved += 1
    print(f"[emilia_yodas/en] saved {saved}/{N_PER_SOURCE}")


def main():
    manifest_rows = []
    build_common_voice("ko", "ko_test_0.tar", "ko_test.tsv", manifest_rows)
    build_common_voice("en", "en_test_0.tar", "en_test.tsv", manifest_rows)
    build_emilia_ko(manifest_rows)
    build_emilia_en(manifest_rows)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nTotal saved: {len(manifest_rows)} -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
