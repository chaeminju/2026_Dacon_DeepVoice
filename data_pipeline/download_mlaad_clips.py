"""manifests/mlaad_{ko,en}_candidates.csv의 항목을 HuggingFace `mueller91/MLAAD`
에서 내려받아 16kHz mono wav로 저장하고, 기존 ko/en raw manifest와 동일한
스키마(path,lang,label,technique,system_id,speaker_id,source[,text])로
data/processed/{ko,en}_mlaad_manifest.csv를 만든다.

- 이미 wav가 있으면 스킵 (재실행 시 이어서 진행).
- HF 캐시(~/.cache 또는 HF_HOME)에 원본이 남으므로 재다운로드 방지는 huggingface_hub가
  알아서 처리 — 여기서는 로컬 목적지 wav 존재 여부만 체크.
- 실패 항목은 {lang}_mlaad_failed.txt에 기록해 재실행 시 스킵.
"""

import argparse
import threading
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "manifests"
OUT_BASE = BASE_DIR / "data" / "processed" / "mlaad_raw"

REPO_ID = "mueller91/MLAAD"
SAMPLE_RATE = 16_000

_lock = threading.Lock()


def load_failed(path):
    if path.exists():
        return set(l.strip() for l in path.open(encoding="utf-8") if l.strip())
    return set()


def append_failed(path, key):
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(key + "\n")


def download_one(row, out_dir):
    out_path = out_dir / row.model / f"{row.filename}.wav"
    if out_path.exists():
        return out_path, True

    local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=row.path_in_repo)
    audio, sr = librosa.load(local_path, sr=SAMPLE_RATE, mono=True)
    if audio.shape[0] < SAMPLE_RATE // 2:  # 0.5초 미만은 버림
        raise RuntimeError(f"clip too short ({audio.shape[0]} samples)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE)
    return out_path, False


def worker(rows, out_dir, failed_path, failed_set, results, idx_lock, state, pbar):
    while True:
        with idx_lock:
            if state["i"] >= len(rows):
                return
            row = rows[state["i"]]
            state["i"] += 1

        key = row.path_in_repo
        if key in failed_set:
            with _lock:
                pbar.update(1)
            continue
        try:
            out_path, existed = download_one(row, out_dir)
            with _lock:
                results.append({
                    "path": str(out_path.relative_to(BASE_DIR)),
                    "lang": row.lang,
                    "label": "spoof",
                    "technique": "tts",
                    "system_id": row.model,
                    "speaker_id": "-",
                    "source": "mlaad",
                    "text": "",
                })
                pbar.update(1)
        except Exception as e:
            print(f"[fail] {row.path_in_repo}: {e}", flush=True)
            append_failed(failed_path, key)
            with _lock:
                failed_set.add(key)
                pbar.update(1)


def run_lang(lang, workers, has_text_col):
    candidates_csv = MANIFEST_DIR / f"mlaad_{lang}_candidates.csv"
    if not candidates_csv.exists():
        print(f"[skip] {candidates_csv} not found (run sample_mlaad_candidates.py first)")
        return

    out_dir = OUT_BASE / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_path = out_dir / f"{lang}_mlaad_failed.txt"
    failed_set = load_failed(failed_path)

    candidates = pd.read_csv(candidates_csv)
    rows = list(candidates.itertuples())
    print(f"[{lang}] {len(rows)} candidates, {len(failed_set)} previously failed")

    results = []
    idx_lock = threading.Lock()
    state = {"i": 0}
    with tqdm(total=len(rows), desc=f"mlaad-{lang}") as pbar:
        threads = [
            threading.Thread(
                target=worker,
                args=(rows, out_dir, failed_path, failed_set, results, idx_lock, state, pbar),
            )
            for _ in range(workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # 이미 존재해서 이번 실행에서 스킵된 파일도 manifest에 포함시키기 위해
    # 디스크 기준으로 다시 스캔한다 (재실행 시에도 manifest가 항상 최신 상태가 되도록).
    manifest_rows = []
    for row in rows:
        if row.path_in_repo in failed_set:
            continue
        out_path = out_dir / row.model / f"{row.filename}.wav"
        if out_path.exists():
            manifest_rows.append({
                "path": str(out_path.relative_to(BASE_DIR)),
                "lang": lang,
                "label": "spoof",
                "technique": "tts",
                "system_id": row.model,
                "speaker_id": "-",
                "source": "mlaad",
                "text": "",
            })

    out_df = pd.DataFrame(manifest_rows)
    if not has_text_col:
        out_df = out_df.drop(columns=["text"])
    manifest_path = BASE_DIR / "data" / "processed" / f"{lang}_mlaad_manifest.csv"
    out_df.to_csv(manifest_path, index=False)
    print(f"[{lang}] done: {len(out_df)} files -> {manifest_path} (failed={len(failed_set)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=["ko", "en"], choices=["ko", "en"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for lang in args.langs:
        # ko manifest는 기존 ko_*_manifest.csv처럼 text 컬럼 유지, en은 en_raw_manifest.csv와
        # 맞춰 text 컬럼 없이 저장한다.
        run_lang(lang, args.workers, has_text_col=(lang == "ko"))


if __name__ == "__main__":
    main()
