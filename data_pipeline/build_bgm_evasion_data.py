"""I_bgm_evasion(배경음악 위장/탐지 회피) 자체 제작 스크립트.

요구사항 원문 그대로 "real 음성에 배경음악을 인위적으로 믹싱해 자체 제작"한다.
I 케이스는 fake 판별이 아니라 "배경음악이 섞여도 PANNs/HTDemucs가 음성 존재를
정확히 잡아내는가"를 보는 것이므로, 두 종류의 샘플이 필요하다.

  - has_voice=True 샘플: real 음성 + 배경음악을 실제로 믹싱해서 새 wav를 만듦
  - has_voice=False 샘플: 배경음악만 있고 음성이 없는 경우

버그 수정 이력(iteration 2에서 발견, 2026-09-14): has_voice=False 샘플을
처음엔 music_raw 원곡 파일을 그대로 재사용했는데, PANNs의 "voice" 그룹에는
Speech뿐 아니라 Singing/Choir/Rapping/Humming 등 가창까지 포함돼 있어서
원곡(대부분 보컬 있음)을 "무음성"으로 라벨링한 게 틀린 라벨이었다(실제로는
has_voice=True인 경우가 많음). PANNs 헤드를 이 라벨로 재학습했더니 voice
존재판별 AUC가 0.63->0.33(랜덤보다 나쁨, 역상관)으로 악화된 게 확인됨.
수정: 원곡을 HTDemucs로 분리해 보컬을 제거한 **반주(instrumental) 스텀만**
"무음성" 샘플로 쓴다 — `build_hybrid_composed_data.py`가 반주 추출에 쓰는
것과 같은 패턴, 이미 만든 `HTDemucsSeparator` 재사용.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_pipeline"))
from audio_mix_utils import load_audio, mix_at_snr, write_wav  # noqa: E402
import case_analysis  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
# mixed/music_only를 별도 태그로 관리해야 한쪽만(예: 라벨 버그 수정된
# music_only만) 다시 만들 때 다른 쪽 기존 레코드가 같이 지워지지 않는다.
SOURCE_DATASET_TAG_MIXED = "synthetic_bgm_evasion_mixed"
SOURCE_DATASET_TAG_MUSIC_ONLY = "synthetic_bgm_evasion_musiconly"

VOICE_MANIFESTS = ["data/processed/ko_augmented_manifest.csv", "data/processed/en_augmented_manifest.csv"]
MUSIC_MANIFESTS = ["manifests/music_train_manifest.csv", "manifests/music_valid_manifest.csv"]

MIN_SNR_DB = -6.0  # 배경음악이 음성보다 살짝 큰 극단적 위장 상황까지 포함
MAX_SNR_DB = 12.0  # 배경음악이 거의 안 들릴 정도로 작은 경우까지 포함


def generate_mixed(n_samples: int, out_dir: Path, seed: int = 17) -> list[dict]:
    rng = np.random.default_rng(seed)
    voices = pd.concat([pd.read_csv(BASE_DIR / m) for m in VOICE_MANIFESTS], ignore_index=True)
    voices = voices[voices["technique"] == "real"].reset_index(drop=True)
    music = pd.concat([pd.read_csv(BASE_DIR / m) for m in MUSIC_MANIFESTS], ignore_index=True)
    print(f"voice(real) pool={len(voices)}, music pool={len(music)}")

    records = []
    for i in range(n_samples):
        voice_row = voices.iloc[rng.integers(0, len(voices))]
        music_row = music.iloc[rng.integers(0, len(music))]

        voice = load_audio(BASE_DIR / voice_row["path"])
        bgm = load_audio(BASE_DIR / music_row["path"])
        snr_db = float(rng.uniform(MIN_SNR_DB, MAX_SNR_DB))
        mixed = mix_at_snr(voice, bgm, snr_db, rng)

        out_path = out_dir / f"bgm_mixed_{i:05d}.wav"
        write_wav(out_path, mixed)
        records.append({
            "file_id": f"bgm_evasion_mixed_{i:05d}",
            "path": str(out_path.relative_to(BASE_DIR)),
            "case_types": ["I_bgm_evasion"],
            "has_voice": True,
            "has_music": True,
            "voice_label": None,
            "music_label": None,
            "source_dataset": SOURCE_DATASET_TAG_MIXED,
        })
        if (i + 1) % 50 == 0:
            print(f"  mixed {i+1}/{n_samples}")
    return records


def build_music_only_records(n_samples: int, out_dir: Path, device, pipeline_script, seed: int = 19) -> list[dict]:
    """원곡을 HTDemucs로 분리해 보컬을 제거한 반주(instrumental)만 취해
    has_voice=False negative로 쓴다 — 원곡 그대로 쓰면 가창(보컬)이 섞여
    있어 실제로는 has_voice=True인 경우가 많다(위 모듈 docstring 참고)."""
    music = pd.concat([pd.read_csv(BASE_DIR / m) for m in MUSIC_MANIFESTS], ignore_index=True)
    picked = music.sample(n=min(n_samples, len(music)), random_state=seed)

    htdemucs = case_analysis.HTDemucsSeparator(device, script_path=pipeline_script)

    records = []
    for i, (_, row) in enumerate(picked.iterrows()):
        _, instrumental = htdemucs.separate(BASE_DIR / row["path"])
        out_path = out_dir / f"bgm_instrumental_{i:05d}.wav"
        write_wav(out_path, instrumental)
        records.append({
            "file_id": f"bgm_evasion_musiconly_{i:05d}",
            "path": str(out_path.relative_to(BASE_DIR)),
            "case_types": ["I_bgm_evasion"],
            "has_voice": False,
            "has_music": True,
            "voice_label": None,
            "music_label": None,
            "source_dataset": SOURCE_DATASET_TAG_MUSIC_ONLY,
        })
        if (i + 1) % 50 == 0:
            print(f"  music_only(instrumental) {i+1}/{n_samples}")
    return records


def merge_into_case_metadata(new_records: list[dict], tags_to_replace: set[str], train_ratio: float,
                              seed: int, train_path: str, val_path: str) -> None:
    """`tags_to_replace`에 속하는 source_dataset 태그를 가진 기존 레코드만
    걷어내고 new_records로 교체한다 — mixed/music_only를 독립적으로 재생성할
    수 있도록(한쪽만 다시 만들 때 다른 쪽까지 지워지지 않게)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(new_records))
    n_train = int(len(new_records) * train_ratio)
    train_new = [new_records[i] for i in idx[:n_train]]
    val_new = [new_records[i] for i in idx[n_train:]]

    for path, new_part in [(train_path, train_new), (val_path, val_new)]:
        existing = case_analysis.load_metadata(path) if Path(path).is_file() else []
        existing = [r for r in existing if r.get("source_dataset") not in tags_to_replace]
        case_analysis.save_metadata(existing + new_part, path)


def main():
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="I_bgm_evasion 합성 데이터 생성")
    parser.add_argument("--n-mixed", type=int, default=300, help="voice+bgm 믹싱 샘플 수")
    parser.add_argument("--n-music-only", type=int, default=300, help="bgm만 있는(무음성) 샘플 수")
    parser.add_argument("--out-dir", default="data/processed/bgm_evasion")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pipeline-script", default=None)
    parser.add_argument("--skip-mixed", action="store_true",
                         help="mixed(voice+bgm) 샘플은 이미 있고 music_only만 다시 만들 때")
    parser.add_argument("--train-metadata", default="manifests/case_metadata_train.jsonl")
    parser.add_argument("--val-metadata", default="manifests/case_metadata_val.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = BASE_DIR / args.out_dir
    mixed_records = [] if args.skip_mixed else generate_mixed(args.n_mixed, out_dir, seed=args.seed)
    music_only_records = build_music_only_records(
        args.n_music_only, out_dir, device, args.pipeline_script, seed=args.seed + 1
    )
    all_records = mixed_records + music_only_records
    tags_to_replace = {SOURCE_DATASET_TAG_MUSIC_ONLY} if args.skip_mixed else {
        SOURCE_DATASET_TAG_MIXED, SOURCE_DATASET_TAG_MUSIC_ONLY,
    }

    merge_into_case_metadata(all_records, tags_to_replace, args.train_ratio, args.seed,
                              args.train_metadata, args.val_metadata)
    print(f"I_bgm_evasion: mixed={len(mixed_records)}, music_only={len(music_only_records)} "
          f"-> {args.train_metadata} / {args.val_metadata}에 병합됨")


if __name__ == "__main__":
    main()
