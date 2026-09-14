"""data/processed/music_raw/{real,fake}에 실제로 존재하는 wav 파일만 기준으로
manifests/music_train_manifest.csv / music_valid_manifest.csv를 만든다.

목표 개수(900+100)에 못 미쳐도(yt-dlp 실패 등) 있는 만큼으로 진행 가능하도록,
디스크에 실제 존재하는 파일만 집계한다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MUSIC_RAW = BASE_DIR / "data" / "processed" / "music_raw"
MANIFEST_DIR = BASE_DIR / "manifests"


def collect(label, candidates_csv):
    candidates = pd.read_csv(candidates_csv)
    audio_dir = MUSIC_RAW / label
    rows = []
    for row in candidates.itertuples():
        wav_path = audio_dir / f"{row.filename}.wav"
        if wav_path.exists():
            rows.append({
                "path": str(wav_path.relative_to(BASE_DIR)),
                "label": label,
                "target": 1 if label == "fake" else 0,
                "split": row.split,
            })
    return pd.DataFrame(rows)


def main():
    fake_df = collect("fake", MANIFEST_DIR / "music_fake_candidates.csv")
    real_df = collect("real", MANIFEST_DIR / "music_real_candidates.csv")

    all_df = pd.concat([fake_df, real_df], ignore_index=True)

    train_df = all_df[all_df["split"] == "train"].sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = all_df[all_df["split"] == "valid"].sample(frac=1, random_state=42).reset_index(drop=True)

    train_path = MANIFEST_DIR / "music_train_manifest.csv"
    valid_path = MANIFEST_DIR / "music_valid_manifest.csv"
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f"train: {len(train_df)} ({train_df['label'].value_counts().to_dict()}) -> {train_path}")
    print(f"valid: {len(valid_df)} ({valid_df['label'].value_counts().to_dict()}) -> {valid_path}")


if __name__ == "__main__":
    main()
