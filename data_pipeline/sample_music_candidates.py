"""SONICS 데이터셋(awsaf49/sonics)에서 SpecTTTra 파인튜닝용 후보를 샘플링한다.

- fake곡: part_01.zip에 이미 들어있는 항목(train/valid)만 대상으로 샘플링
  (32GB 전체를 받을 필요 없이 3.8GB짜리 하나로 충분한 양을 확보).
- real곡: mp3가 없고 youtube_id만 있으므로, yt-dlp 다운로드 성공률을 감안해
  목표보다 넉넉하게(1.3배) 후보를 뽑아둔다.

seed를 고정해 재실행해도 동일한 후보 리스트가 나오므로 중간에 끊겨도 이 스크립트를
다시 돌리면 됨(멱등).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
META_DIR = BASE_DIR / "data" / "raw" / "sonics_meta"
OUT_DIR = BASE_DIR / "manifests"

SEED = 42
FAKE_PART = "fake_songs/part_01.zip"


def sample_fake(n_train, n_valid):
    meta = json.load(open(META_DIR / "metadata.json"))
    fm = meta["file_mapping"]
    fake = pd.read_csv(META_DIR / "fake_songs.csv", low_memory=False)
    fake["part"] = fake["filename"].map(lambda f: fm.get(f"fake_songs/{f}.mp3"))
    pool = fake[fake["part"] == FAKE_PART]

    train_pool = pool[pool["split"] == "train"]
    valid_pool = pool[pool["split"] == "valid"]

    train_sample = train_pool.sample(n=min(n_train, len(train_pool)), random_state=SEED)
    valid_sample = valid_pool.sample(n=min(n_valid, len(valid_pool)), random_state=SEED)

    train_sample = train_sample.assign(split="train")
    valid_sample = valid_sample.assign(split="valid")
    out = pd.concat([train_sample, valid_sample], ignore_index=True)
    out = out[["filename", "split", "source", "label", "duration"]]
    out["label"] = "fake"
    return out


def sample_real(n_train, n_valid, oversample=1.3):
    real = pd.read_csv(META_DIR / "real_songs.csv", low_memory=False)

    def pick(split, n):
        p = real[real["split"] == split]
        n_pick = min(int(n * oversample), len(p))
        return p.sample(n=n_pick, random_state=SEED)

    train_sample = pick("train", n_train).assign(split="train")
    valid_sample = pick("valid", n_valid).assign(split="valid")
    out = pd.concat([train_sample, valid_sample], ignore_index=True)
    out = out[["filename", "split", "youtube_id", "duration", "skip_time"]]
    out["label"] = "real"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=900)
    parser.add_argument("--n-valid", type=int, default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fake_df = sample_fake(args.n_train, args.n_valid)
    fake_path = OUT_DIR / "music_fake_candidates.csv"
    fake_df.to_csv(fake_path, index=False)
    print(f"fake candidates: {len(fake_df)} -> {fake_path}")
    print(fake_df["split"].value_counts())

    real_df = sample_real(args.n_train, args.n_valid)
    real_path = OUT_DIR / "music_real_candidates.csv"
    real_df.to_csv(real_path, index=False)
    print(f"real candidates: {len(real_df)} -> {real_path}")
    print(real_df["split"].value_counts())


if __name__ == "__main__":
    main()
