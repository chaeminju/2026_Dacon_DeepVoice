"""HuggingFace `mueller91/MLAAD`(최신 TTS 12~140종으로 만든 다국어 스푸핑
데이터셋, gated)에서 KO/EN fake 후보 목록을 만든다.

- KO: fake/ko/ 아래 12개 모델(XTTS-v2, Bark, Qwen3-TTS, MiniMax-Speech,
  Chatterbox Multilingual, Higgs-Audio-V2, MOSS-TTS, Fish-S2-Pro, VoxCPM2,
  OmniVoice 등) 전체를 후보로 삼는다 (총 4.79GB로 작아서 전체 사용 가능).
- EN: fake/en/ 아래 140여 개 모델 중 최신/대표 엔진만 골라 샘플링한다
  (전체 45GB는 디스크 예산상 과함).

repo_id 파일 목록은 HfApi().list_repo_files로 가져오며, 결과를 캐싱해서
재실행 시 API를 다시 부르지 않는다(파일 목록 자체는 거의 안 바뀜).
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "manifests"
CACHE_PATH = BASE_DIR / "data" / "raw" / "mlaad_file_list_cache.json"

REPO_ID = "mueller91/MLAAD"
SEED = 42

# EN은 모델 수가 너무 많아 최신/대표 엔진만 고정 목록으로 선택
EN_MODELS = [
    "ElevenLabs-v3",
    "ElevenLabs-Turbo-v2.5",
    "ElevenLabs-v2-Multilingual",
    "Chatterbox",
    "Chatterbox-Turbo",
    "tts_models_multilingual_multi-dataset_xtts_v2",
    "minimax_speech-2.6-hd",
    "Qwen3-TTS-12Hz-1.7B-Base",
    "GPT-SoVITS",
    "Higgs-Audio-V2",
    "kokoro",
    "MOSS-TTS-8B",
]


def list_all_files():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(files), encoding="utf-8")
    return files


def build_candidates(files, lang, models=None, per_model_cap=None):
    prefix = f"fake/{lang}/"
    rows = []
    for f in files:
        if not f.startswith(prefix) or not f.endswith(".wav"):
            continue
        rest = f[len(prefix):]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            continue
        model, fname = parts
        if models is not None and model not in models:
            continue
        rows.append({"path_in_repo": f, "lang": lang, "model": model, "filename": Path(fname).stem})

    df = pd.DataFrame(rows)
    if per_model_cap:
        rng = random.Random(SEED)
        kept = []
        for model, group in df.groupby("model"):
            idx = list(group.index)
            rng.shuffle(idx)
            kept.extend(idx[:per_model_cap])
        df = df.loc[kept].reset_index(drop=True)

    # train/valid 90/10 (모델별로 stratify해서 특정 모델이 한쪽에 쏠리지 않게)
    rng = random.Random(SEED + 1)
    splits = []
    for model, group in df.groupby("model"):
        idx = list(group.index)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * 0.1))
        val_idx = set(idx[:n_val])
        splits.extend("valid" if i in val_idx else "train" for i in group.index)
    df = df.assign(split=splits)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en-per-model-cap", type=int, default=None,
                         help="EN 모델당 최대 샘플 수 (기본: 제한 없음, 모델당 ~1000개)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list_all_files()
    print(f"total files in repo listing: {len(files)}")

    ko_df = build_candidates(files, "ko", models=None, per_model_cap=None)
    ko_path = OUT_DIR / "mlaad_ko_candidates.csv"
    ko_df.to_csv(ko_path, index=False)
    print(f"ko candidates: {len(ko_df)} -> {ko_path}")
    print(ko_df["model"].value_counts())

    en_df = build_candidates(files, "en", models=EN_MODELS, per_model_cap=args.en_per_model_cap)
    en_path = OUT_DIR / "mlaad_en_candidates.csv"
    en_df.to_csv(en_path, index=False)
    print(f"en candidates: {len(en_df)} -> {en_path}")
    print(en_df["model"].value_counts())


if __name__ == "__main__":
    main()
