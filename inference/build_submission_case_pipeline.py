"""`submit_final_koen_v2/`(HTDemucs -> PANNs x2 -> DF-Arena(voice)/SpecTTTra
(music) 구조가 이미 맞게 짜여 있는 추론 전용 코드)를 베이스로, voice/music/
presence 세 모델 전부 이번 세션(case_pipeline, iteration 3) 결과로 교체한다.

기존 `build_submission_final.py`(DF-Arena만 교체)와 다른 점:
  - DF-Arena: df_arena_case_pipeline_v1을 fp16으로 변환해서 사용(용량↓,
    check_fp16_conversion.py로 안전성 확인됨: EER 0.060->0.064, 오차 미미)
  - SpecTTTra: spectttra_case_pipeline_v1 그대로 사용
  - PANNs: **이번이 처음으로 파인튜닝 가중치를 넣는 것** — fp16 변환은
    NaN이 나서(check_fp16_conversion.py로 확인) 포기하고 fp32 그대로 사용
  - manifest.json에 각 모델의 출처/버전/검증 지표를 남겨서 "이 제출본에
    정확히 뭐가 들어갔는지" 나중에 추적 가능하게 함
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_SUBMIT = BASE_DIR / "submit_final_koen_v2"

DEFAULT_DF_ARENA_DIR = BASE_DIR / "data/processed/df_arena_case_pipeline_v1_fp16"
DEFAULT_SPECTTTRA_DIR = BASE_DIR / "data/processed/spectttra_case_pipeline_v1"
DEFAULT_PANNS_CKPT = BASE_DIR / "data/processed/panns_case_pipeline_v2/Cnn14_finetuned.pth"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def replace_hf_weights(dst_dir: Path, source_dir: Path, weight_patterns: tuple[str, ...]) -> None:
    for pattern in weight_patterns:
        for f in dst_dir.glob(pattern):
            f.unlink()
    for f in source_dir.iterdir():
        if f.is_file():
            shutil.copy(f, dst_dir / f.name)


def replace_panns_weights(dst_dir: Path, checkpoint_path: Path) -> None:
    """load_panns_model()이 파일명을 하드코딩해서 읽으므로("Cnn14_mAP=0.431.pth")
    이름은 그대로 두고 내용만 우리 체크포인트로 교체한다."""
    target = dst_dir / "Cnn14_mAP=0.431.pth"
    shutil.copy(checkpoint_path, target)


def write_manifest(staging: Path, args: argparse.Namespace) -> None:
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "base_template": str(SOURCE_SUBMIT.relative_to(BASE_DIR)),
        "pipeline_architecture": "HTDemucs(분리, 1회) -> PANNs(voice_audio), PANNs(music_audio) "
                                  "-> DF-Arena(voice_audio)=VOICE_FAKE_PROB, "
                                  "SpecTTTra(music_audio)=MUSIC_FAKE_PROB",
        "models": {
            "htdemucs": {
                "source": "submit_final_koen_v2/model/htdemucs (원본, 이번 세션에서 파인튜닝 안 함)",
                "finetuned": False,
            },
            "df_arena_voice": {
                "source": str(Path(args.df_arena_dir).relative_to(BASE_DIR)),
                "dtype": "fp16",
                "finetuned": True,
                "training_data": "manifests/case_metadata_train.jsonl (voice_label 있는 레코드, "
                                  "A~E + H_hybrid_composed voice 관점)",
                "val_metrics_fp32_source": {
                    "note": "fp16 변환 전 fp32 기준 iteration 3 case_performance_iter3.csv 참고",
                },
                "fp16_validation": "check_fp16_conversion.py --target df_arena (250 샘플): "
                                    "EER 0.0600->0.0640, 최대 절대오차 0.0162 (안전)",
            },
            "spectttra_music": {
                "source": str(Path(args.spectttra_dir).relative_to(BASE_DIR)),
                "dtype": "fp32",
                "finetuned": True,
                "training_data": "manifests/case_metadata_train.jsonl (music_label 있는 레코드, "
                                  "G_full_ai_music + H_hybrid_composed music 관점)",
            },
            "panns_presence": {
                "source": str(Path(args.panns_ckpt).relative_to(BASE_DIR)),
                "dtype": "fp32",
                "finetuned": True,
                "note": "이번 세션에서 처음 파인튜닝 가중치를 제출본에 반영 "
                        "(fc_audioset 분류 헤드만 재학습, 백본 고정)",
                "training_data": "manifests/case_metadata_train.jsonl (I_bgm_evasion 레코드, "
                                  "HTDemucs로 분리한 voice_audio/music_audio 각각)",
                "fp16_validation": "check_fp16_conversion.py --target panns: NaN 발생, "
                                    "fp16 변환 포기하고 fp32 유지",
            },
        },
        "val_performance_iteration3": "case_performance_iter3.csv (val n=2,140) 참고 — "
                                       "9개 케이스 전부 iteration 1(제로샷) 대비 개선, "
                                       "D_partial_splice EER 0.297->0.045, "
                                       "H_hybrid_composed(music) EER 0.286->0.158, "
                                       "I_bgm_evasion voice presence_auc 0.632->0.869",
        "known_limitations": [
            "F_ai_cover_song: 매핑 가능한 데이터가 없어 이번 재학습에 반영되지 않음(제로샷 그대로)",
            "로컬 val 개선이 실제 Dacon 리더보드로 이어지는지는 업로드해서 확인 필요"
            "(과거 v1/v2 EER 사례처럼 val-리더보드 괴리가 있었던 전례 있음)",
        ],
    }
    # model/ 밑이 아니라 staging 최상위에 둔다 — model/은 .gitignore로
    # 통째로 제외되는 대용량 가중치 폴더라, 그 안에 두면 버전 추적용
    # 매니페스트까지 같이 git에서 빠져버린다.
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MANIFEST.json 기록 -> {staging / 'MANIFEST.json'}")


def build_submission(args: argparse.Namespace) -> Path:
    staging = BASE_DIR / args.out_dir_name
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(SOURCE_SUBMIT, staging, ignore=shutil.ignore_patterns("__pycache__"))

    replace_hf_weights(
        staging / "model" / "df_arena_1b", Path(args.df_arena_dir),
        ("model*.safetensors", "pytorch_model*.bin", "config.json"),
    )
    replace_hf_weights(
        staging / "model" / "spectttra", Path(args.spectttra_dir),
        ("pytorch_model*.bin", "config.json"),
    )
    replace_panns_weights(staging / "model" / "panns", Path(args.panns_ckpt))

    write_manifest(staging, args)

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    unzipped_size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    print(f"staging dir: {staging} ({unzipped_size / 1e9:.2f} GB 비압축)")
    print(f"built {out_zip} ({out_zip.stat().st_size / 1e9:.2f} GB 압축)")
    return staging


def main():
    parser = argparse.ArgumentParser(description="case_pipeline(iteration 3) 결과로 제출본 패키징")
    parser.add_argument("--df-arena-dir", default=str(DEFAULT_DF_ARENA_DIR))
    parser.add_argument("--spectttra-dir", default=str(DEFAULT_SPECTTTRA_DIR))
    parser.add_argument("--panns-ckpt", default=str(DEFAULT_PANNS_CKPT))
    parser.add_argument("--out-dir-name", default="submit_case_pipeline_v1")
    parser.add_argument("--out-zip", default=str(BASE_DIR / "submissions" / "submission_case_pipeline_v1.zip"))
    args = parser.parse_args()
    build_submission(args)


if __name__ == "__main__":
    main()
