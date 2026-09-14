"""H_hybrid_composed(반주 AI생성 + 보컬 합성 결합) 자체 제작 스크립트.

실제 SingFake/CtrSVDD류 "AI 커버곡" 데이터와는 다르지만, H가 정의하는 상황
자체("AI가 만든 반주 위에 별도로 합성한 보컬을 얹은 곡")는 이미 가진 자산
조합으로 만들 수 있다:
  1. SONICS AI 완전생성곡(G_full_ai_music의 fake 소스)을 HTDemucs로 분리해
     보컬을 버리고 반주(music_audio)만 취한다 -> "AI 반주"
  2. 우리가 이미 가진 TTS/VC fake 음성 클립을 "별도로 합성한 보컬"로 얹는다
  3. 둘을 믹싱

FakeMusicCaps+MLAAD 직접 믹싱(요구사항에 명시된 원래 추천안)과 사실상 같은
아이디어를 지금 가진 SONICS+MLAAD/RVC 자산으로 대체 실행한 것.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_pipeline"))
from audio_mix_utils import load_audio, mix_at_snr, write_wav  # noqa: E402
import case_analysis  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATASET_TAG = "synthetic_hybrid_composed"

VOICE_MANIFESTS = ["data/processed/ko_augmented_manifest.csv", "data/processed/en_augmented_manifest.csv"]
MUSIC_FAKE_MANIFESTS = ["manifests/music_train_manifest.csv", "manifests/music_valid_manifest.csv"]

MIN_SNR_DB = -2.0  # 보컬이 반주보다 살짝 작을 수도 있는 상황 포함
MAX_SNR_DB = 8.0


def generate(n_samples: int, out_dir: Path, device, pipeline_script, seed: int = 23) -> list[dict]:
    rng = np.random.default_rng(seed)
    vocals = pd.concat([pd.read_csv(BASE_DIR / m) for m in VOICE_MANIFESTS], ignore_index=True)
    vocals = vocals[vocals["technique"].isin(["tts", "vc"])].reset_index(drop=True)

    music = pd.concat([pd.read_csv(BASE_DIR / m) for m in MUSIC_FAKE_MANIFESTS], ignore_index=True)
    music = music[music["label"] == "fake"].reset_index(drop=True)  # SONICS AI 완전생성곡만
    print(f"vocal(fake) pool={len(vocals)}, backing-track source(AI music) pool={len(music)}")

    htdemucs = case_analysis.HTDemucsSeparator(device, script_path=pipeline_script)

    records = []
    for i in range(n_samples):
        vocal_row = vocals.iloc[rng.integers(0, len(vocals))]
        music_row = music.iloc[rng.integers(0, len(music))]

        vocal = load_audio(BASE_DIR / vocal_row["path"])
        # HTDemucs로 원곡의 보컬은 버리고 반주(music_audio)만 취한다.
        _, backing_track = htdemucs.separate(BASE_DIR / music_row["path"])

        snr_db = float(rng.uniform(MIN_SNR_DB, MAX_SNR_DB))
        mixed = mix_at_snr(vocal, backing_track, snr_db, rng)

        out_path = out_dir / f"hybrid_{i:05d}.wav"
        write_wav(out_path, mixed)
        records.append({
            "file_id": f"hybrid_composed_{i:05d}",
            "path": str(out_path.relative_to(BASE_DIR)),
            "case_types": ["H_hybrid_composed"],
            "has_voice": True,
            "has_music": True,
            "voice_label": "fake",
            "music_label": "fake",
            "source_dataset": SOURCE_DATASET_TAG,
        })
        if (i + 1) % 25 == 0:
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
        existing = [r for r in existing if r.get("source_dataset") != SOURCE_DATASET_TAG]
        case_analysis.save_metadata(existing + new_part, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="H_hybrid_composed 합성 데이터 생성 (HTDemucs 필요)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--out-dir", default="data/processed/hybrid_composed")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pipeline-script", default=None)
    parser.add_argument("--train-metadata", default="manifests/case_metadata_train.jsonl")
    parser.add_argument("--val-metadata", default="manifests/case_metadata_val.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = generate(args.n_samples, BASE_DIR / args.out_dir, device, args.pipeline_script, seed=args.seed)
    merge_into_case_metadata(records, args.train_ratio, args.seed, args.train_metadata, args.val_metadata)
    print(f"H_hybrid_composed: 총 {len(records)}개 생성 -> {args.train_metadata} / {args.val_metadata}에 병합됨")


if __name__ == "__main__":
    main()
