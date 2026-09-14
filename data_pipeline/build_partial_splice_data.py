"""D_partial_splice(녹취 짜깁기) 자체 제작 스크립트 — 새 데이터셋을 받지 않고
이미 가진 real 음성(carrier)의 일부 구간을 이미 가진 fake(TTS/VC) 음성
조각으로 치환해 "부분 위조" 샘플을 만든다.

실제 PartialSpoof/HAD 데이터셋을 구하기 전까지의 임시 대체재 — 진짜 partial
spoof 데이터가 갖는 미세한 편집 흔적(코덱 불일치, 배경 잡음 불연속 등)까지
재현하진 못하지만, "발화 중간에 위조 구간이 섞이면 탐지되는가"라는 D 케이스의
핵심 질문 자체는 이 방식으로도 검증 가능하다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_pipeline"))
from audio_mix_utils import load_audio, splice_segment, write_wav  # noqa: E402
import case_analysis  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATASET_TAG = "synthetic_partial_splice"

CARRIER_MANIFESTS = ["data/processed/ko_augmented_manifest.csv", "data/processed/en_augmented_manifest.csv"]
SEGMENT_MANIFESTS = ["data/processed/ko_augmented_manifest.csv", "data/processed/en_augmented_manifest.csv"]

MIN_SEGMENT_SEC = 0.5
MAX_SEGMENT_SEC = 2.5


def _load_pool(manifests: list[str], technique_filter) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(BASE_DIR / m) for m in manifests], ignore_index=True)
    return df[df["technique"].isin(technique_filter)].reset_index(drop=True)


def generate(n_samples: int, out_dir: Path, seed: int = 13) -> list[dict]:
    rng = np.random.default_rng(seed)
    carriers = _load_pool(CARRIER_MANIFESTS, ["real"])
    segments = _load_pool(SEGMENT_MANIFESTS, ["tts", "vc"])
    print(f"carrier(real) pool={len(carriers)}, segment(fake) pool={len(segments)}")

    records = []
    for i in range(n_samples):
        carrier_row = carriers.iloc[rng.integers(0, len(carriers))]
        segment_row = segments.iloc[rng.integers(0, len(segments))]

        carrier = load_audio(BASE_DIR / carrier_row["path"])
        segment_full = load_audio(BASE_DIR / segment_row["path"])

        seg_sec = rng.uniform(MIN_SEGMENT_SEC, MAX_SEGMENT_SEC)
        seg_len = int(seg_sec * 16_000)
        if segment_full.size > seg_len:
            start = int(rng.integers(0, segment_full.size - seg_len + 1))
            segment = segment_full[start : start + seg_len]
        else:
            segment = segment_full

        spliced = splice_segment(carrier, segment, rng)
        out_path = out_dir / f"splice_{i:05d}.wav"
        write_wav(out_path, spliced)

        records.append({
            "file_id": f"partial_splice_{i:05d}",
            "path": str(out_path.relative_to(BASE_DIR)),
            "case_types": ["D_partial_splice"],
            "has_voice": True,
            "has_music": False,
            "voice_label": "fake",
            "music_label": None,
            "source_dataset": SOURCE_DATASET_TAG,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_samples} 생성됨")
    return records


def merge_into_case_metadata(new_records: list[dict], train_ratio: float, seed: int,
                              train_path: str, val_path: str) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(new_records))
    n_train = int(len(new_records) * train_ratio)
    train_new = [new_records[i] for i in idx[:n_train]]
    val_new = [new_records[i] for i in idx[n_train:]]

    for path, new_part in [(train_path, train_new), (val_path, val_new)]:
        existing = case_analysis.load_metadata(path) if Path(path).is_file() else []
        # 이 스크립트가 이전에 만든 레코드는 제거하고(재실행 시 중복 방지) 새로 추가
        existing = [r for r in existing if r.get("source_dataset") != SOURCE_DATASET_TAG]
        case_analysis.save_metadata(existing + new_part, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="D_partial_splice 합성 데이터 생성")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--out-dir", default="data/processed/partial_splice")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-metadata", default="manifests/case_metadata_train.jsonl")
    parser.add_argument("--val-metadata", default="manifests/case_metadata_val.jsonl")
    args = parser.parse_args()

    records = generate(args.n_samples, BASE_DIR / args.out_dir, seed=args.seed)
    merge_into_case_metadata(records, args.train_ratio, args.seed, args.train_metadata, args.val_metadata)
    print(f"D_partial_splice: 총 {len(records)}개 생성 -> {args.train_metadata} / {args.val_metadata}에 병합됨")


if __name__ == "__main__":
    main()
