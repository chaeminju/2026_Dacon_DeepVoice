"""manifests/music_fake_candidates.csv에 있는 항목을 part_01.zip에서 추출해
16kHz mono wav로 저장한다 (전체 곡 대신 35초 구간만 잘라서 용량 절약 +
파인튜닝 시 랜덤 5초 크롭에 쓸 여유를 둠).

이미 존재하는 wav는 건너뛰므로 중단 후 재실행하면 이어서 처리된다.
"""

import zipfile
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
ZIP_PATH = BASE_DIR / "data" / "raw" / "sonics_fake" / "fake_songs" / "part_01.zip"
CANDIDATES_CSV = BASE_DIR / "manifests" / "music_fake_candidates.csv"
OUT_DIR = BASE_DIR / "data" / "processed" / "music_raw" / "fake"

SAMPLE_RATE = 16_000
CLIP_SECONDS = 35
CLIP_START_SECONDS = 15  # 인트로/무음 구간을 피하려고 15초 지점부터 자름


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATES_CSV)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names_in_zip = set(zf.namelist())

        n_done, n_skip, n_missing, n_fail = 0, 0, 0, 0
        for row in candidates.itertuples():
            out_path = OUT_DIR / f"{row.filename}.wav"
            if out_path.exists():
                n_skip += 1
                continue

            entry_name = f"fake_songs/{row.filename}.mp3"
            if entry_name not in names_in_zip:
                print(f"[missing in zip] {entry_name}")
                n_missing += 1
                continue

            try:
                with zf.open(entry_name) as fp:
                    audio, sr = librosa.load(fp, sr=SAMPLE_RATE, mono=True)
                start = int(CLIP_START_SECONDS * SAMPLE_RATE)
                end = start + int(CLIP_SECONDS * SAMPLE_RATE)
                if audio.shape[0] <= start:
                    # 곡이 짧으면 처음부터
                    clip = audio[: int(CLIP_SECONDS * SAMPLE_RATE)]
                else:
                    clip = audio[start:end]
                if clip.shape[0] < SAMPLE_RATE:  # 1초 미만이면 스킵
                    print(f"[too short] {row.filename}")
                    n_fail += 1
                    continue
                sf.write(out_path, clip, SAMPLE_RATE)
                n_done += 1
                if n_done % 100 == 0:
                    print(f"...{n_done} extracted")
            except Exception as e:
                print(f"[fail] {row.filename}: {e}")
                n_fail += 1

    print(f"done: extracted={n_done} skipped(existing)={n_skip} missing={n_missing} failed={n_fail}")


if __name__ == "__main__":
    main()
