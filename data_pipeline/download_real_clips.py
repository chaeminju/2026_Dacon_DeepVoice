"""manifests/music_real_candidates.csv의 youtube_id로 real곡을 받아 35초 클립만
16kHz mono wav로 저장한다 (SONICS는 저작권상 mp3를 직접 제공하지 않고
youtube_id만 준다 — README에 명시된 정식 사용법).

- 이미 wav가 있으면 스킵 (재실행 시 이어서 진행).
- 실패한 youtube_id는 real_failed.txt에 기록해 재실행 시 다시 시도하지 않음
  (영상 삭제/비공개 등으로 계속 실패하는 항목에 시간 낭비하지 않기 위함).
- split별 목표 개수(--n-train/--n-valid)에 도달하면 남은 후보는 건너뛰고 종료.
- 전체 곡을 받지 않고 --download-sections로 skip_time 근처 35초만 받아 대역폭 절약.
"""

import argparse
import random
import shutil
import tempfile
import threading
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import yt_dlp

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = BASE_DIR / "manifests" / "music_real_candidates.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "music_raw" / "real"
FAILED_LOG = OUT_DIR / "real_failed.txt"

SAMPLE_RATE = 16_000
CLIP_SECONDS = 35
DEFAULT_SKIP = 20  # skip_time이 없거나 이상치일 때 기본 인트로 스킵(초)

_lock = threading.Lock()
_counts = {"train": 0, "valid": 0}
_failed_ids = set()


def load_failed():
    if FAILED_LOG.exists():
        with FAILED_LOG.open(encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def append_failed(youtube_id):
    with _lock:
        with FAILED_LOG.open("a", encoding="utf-8") as f:
            f.write(youtube_id + "\n")
        _failed_ids.add(youtube_id)


def download_one(row, tmpdir):
    start = row.skip_time if pd.notna(row.skip_time) and row.skip_time >= 0 else DEFAULT_SKIP
    start = min(start, max(row.duration - CLIP_SECONDS, 0)) if pd.notna(row.duration) else start
    end = start + CLIP_SECONDS

    tmpl = str(Path(tmpdir) / f"{row.youtube_id}.%(ext)s")
    ydl_opts = {
        "format": "worstaudio/worst",
        "outtmpl": tmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 1,
        "socket_timeout": 20,
        "download_sections": [f"*{start}-{end}"],
        "force_keyframes_at_cuts": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={row.youtube_id}"])

    files = list(Path(tmpdir).glob(f"{row.youtube_id}.*"))
    if not files:
        raise RuntimeError("no output file")
    src = files[0]

    audio, sr = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    # --download-sections가 hlsnative 다운로더에서는 실제로 트리밍되지 않고
    # 전체 곡이 내려오는 경우가 있어(용량은 worst 품질이라 작음), 여기서
    # 직접 skip_time 위치부터 CLIP_SECONDS만큼 잘라낸다.
    start_sample = int(start * SAMPLE_RATE)
    if audio.shape[0] <= start_sample:
        start_sample = 0
    clip = audio[start_sample : start_sample + int(CLIP_SECONDS * SAMPLE_RATE)]
    if clip.shape[0] < SAMPLE_RATE:
        raise RuntimeError(f"clip too short ({clip.shape[0]} samples)")
    audio = clip

    out_path = OUT_DIR / f"{row.filename}.wav"
    sf.write(out_path, audio, SAMPLE_RATE)
    src.unlink(missing_ok=True)


def worker(rows, n_train_target, n_valid_target, idx_lock, state):
    while True:
        with idx_lock:
            if state["i"] >= len(rows):
                return
            row = rows[state["i"]]
            state["i"] += 1

        with _lock:
            if _counts[row.split] >= (n_train_target if row.split == "train" else n_valid_target):
                continue  # 목표 달성한 split은 스킵(그래도 후보 리스트는 순회)

        out_path = OUT_DIR / f"{row.filename}.wav"
        if out_path.exists():
            with _lock:
                _counts[row.split] += 1
            continue
        if row.youtube_id in _failed_ids:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                download_one(row, tmpdir)
                with _lock:
                    _counts[row.split] += 1
                    print(f"[ok] {row.filename} (train={_counts['train']} valid={_counts['valid']})", flush=True)
            except Exception as e:
                print(f"[fail] {row.filename} ({row.youtube_id}): {e}", flush=True)
                append_failed(row.youtube_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=900)
    parser.add_argument("--n-valid", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global _failed_ids
    _failed_ids = load_failed()

    candidates = pd.read_csv(CANDIDATES_CSV)
    rows = list(candidates.itertuples())
    random.Random(42).shuffle(rows)  # 워커 간 split 쏠림 방지

    # 이미 존재하는 파일 개수를 미리 반영
    for row in rows:
        if (OUT_DIR / f"{row.filename}.wav").exists():
            _counts[row.split] += 1
    print(f"already done before start: train={_counts['train']} valid={_counts['valid']}")

    idx_lock = threading.Lock()
    state = {"i": 0}
    threads = [
        threading.Thread(target=worker, args=(rows, args.n_train, args.n_valid, idx_lock, state))
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"done: train={_counts['train']}/{args.n_train} valid={_counts['valid']}/{args.n_valid}")
    print(f"failed total: {len(_failed_ids)} (see {FAILED_LOG})")


if __name__ == "__main__":
    main()
