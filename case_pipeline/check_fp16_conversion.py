"""제출 패키징 전 fp16 변환 안전성 점검.

- PANNs(Cnn14, panns_case_pipeline_v2): 이번 세션에서 처음 파인튜닝 가중치를
  넣는 모델이라 fp16 캐스팅 시 성능(특히 I_bgm_evasion 존재판별)이 유지되는지
  검증셋으로 fp32 vs fp16을 같은 데이터로 나란히 비교한다.
- DF-Arena(df_arena_case_pipeline_v1): 지난 세션에 fp16 변환 전례가 있지만,
  이번 체크포인트(케이스 파이프라인으로 새로 학습된 것)에도 동일하게
  안전한지 재확인한다.

CNN(Cnn14)은 내부에 STFT/logmel 추출, BatchNorm 등 fp16에서 불안정할 수
있는 연산이 섞여 있어 HF Transformer 모델(DF-Arena)보다 blanket fp16
캐스팅 리스크가 크다 — 그래서 PANNs는 별도로 꼭 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_analysis as ca  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# PANNs fp32 vs fp16
# -----------------------------------------------------------------------------


def check_panns_fp16(device, panns_ckpt: str, val_metadata: str, max_samples: int | None = None):
    pipeline = ca._load_submit_pipeline_module()

    panns_model = ca.PANNsPresenceModel(device)
    # 방금 만든(iteration 3) 파인튜닝 체크포인트로 교체
    state = torch.load(panns_ckpt, map_location=device)
    panns_model.raw_model.load_state_dict(state["model"])
    panns_model.raw_model.eval()

    htdemucs_model = ca.HTDemucsSeparator(device)

    records = ca.load_metadata(val_metadata)
    i_records = [r for r in records if "I_bgm_evasion" in (r.get("case_types") or [])]
    if max_samples:
        i_records = i_records[:max_samples]
    print(f"PANNs fp16 점검: I_bgm_evasion val {len(i_records)}개")

    # fp32 pass
    fp32_scores, labels = [], []
    for r in i_records:
        voice_audio, music_audio = htdemucs_model.separate(BASE_DIR / r["path"])
        score = pipeline.predict_presence(panns_model.model, panns_model.voice_indices, voice_audio)
        fp32_scores.append(score)
        labels.append(int(bool(r.get("has_voice"))))

    # fp16 캐스팅 (모델 전체 파라미터+버퍼를 fp16으로 통일 — DF-Arena 로딩 시
    # 쓰는 "대표 dtype으로 강제 통일" 패턴과 동일)
    raw_model = panns_model.raw_model
    raw_model.half()

    fp16_scores = []
    for r in i_records:
        voice_audio, music_audio = htdemucs_model.separate(BASE_DIR / r["path"])
        segments = pipeline.make_panns_segments(voice_audio)
        segments_t = torch.from_numpy(segments).to(device=device, dtype=torch.float16)
        with torch.no_grad():
            out = raw_model(segments_t, None)["clipwise_output"]
        score = float(out[:, panns_model.voice_indices].max().float().cpu())
        fp16_scores.append(score)

    raw_model.float()  # 원복

    m32 = ca.compute_presence_metrics(labels, fp32_scores)
    m16 = ca.compute_presence_metrics(labels, fp16_scores)
    print(f"  fp32: accuracy={m32['presence_accuracy']:.4f} auc={m32['presence_auc']:.4f}")
    print(f"  fp16: accuracy={m16['presence_accuracy']:.4f} auc={m16['presence_auc']:.4f}")
    max_abs_diff = float(np.max(np.abs(np.array(fp32_scores) - np.array(fp16_scores))))
    print(f"  샘플별 점수 최대 절대오차: {max_abs_diff:.4f}")
    return m32, m16


# -----------------------------------------------------------------------------
# DF-Arena fp32 vs fp16 (safetensors 캐스팅 후 재로드해서 비교)
# -----------------------------------------------------------------------------


def check_df_arena_fp16(device, df_arena_ckpt: str, val_metadata: str, max_samples: int | None = None):
    import safetensors.torch as st

    sys.path.insert(0, str(BASE_DIR / "training"))
    import finetune_df_arena as df_arena_ft

    pipeline = ca._load_submit_pipeline_module()
    htdemucs_model = ca.HTDemucsSeparator(device)

    model = df_arena_ft.load_model(device)
    model.load_state_dict(st.load_file(str(Path(df_arena_ckpt) / "model.safetensors")), strict=False)
    model.eval()
    fake_label_index = int(model.config.label2id["spoof"])

    records = ca.load_metadata(val_metadata)
    voice_records = [r for r in records if r.get("voice_label") in ("real", "fake")]
    if max_samples:
        voice_records = voice_records[:max_samples]
    print(f"DF-Arena fp16 점검: voice val {len(voice_records)}개")

    labels, fp32_scores = [], []
    for r in voice_records:
        voice_audio, _ = htdemucs_model.separate(BASE_DIR / r["path"])
        score = pipeline.predict_fake(model, fake_label_index, voice_audio, device)
        fp32_scores.append(score)
        labels.append(1 if r["voice_label"] == "fake" else 0)

    target_dtype = torch.float16
    model = model.to(device=device, dtype=target_dtype)
    fp16_scores = []
    for r in voice_records:
        voice_audio, _ = htdemucs_model.separate(BASE_DIR / r["path"])
        fp16_scores.append(pipeline.predict_fake(model, fake_label_index, voice_audio, device))

    m32 = ca.compute_binary_metrics(labels, fp32_scores)
    m16 = ca.compute_binary_metrics(labels, fp16_scores)
    print(f"  fp32: EER={m32['EER']:.4f} F1={m32['F1']:.4f} Accuracy={m32['Accuracy']:.4f}")
    print(f"  fp16: EER={m16['EER']:.4f} F1={m16['F1']:.4f} Accuracy={m16['Accuracy']:.4f}")
    max_abs_diff = float(np.max(np.abs(np.array(fp32_scores) - np.array(fp16_scores))))
    print(f"  샘플별 점수 최대 절대오차: {max_abs_diff:.4f}")
    return m32, m16


def main():
    import argparse

    parser = argparse.ArgumentParser(description="fp16 변환 전후 성능 비교")
    parser.add_argument("--target", choices=["panns", "df_arena", "both"], default="both")
    parser.add_argument("--val-metadata", default="manifests/case_metadata_val.jsonl")
    parser.add_argument("--panns-ckpt", default="data/processed/panns_case_pipeline_v2/Cnn14_finetuned.pth")
    parser.add_argument("--df-arena-ckpt", default="data/processed/df_arena_case_pipeline_v1")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.target in ("panns", "both"):
        check_panns_fp16(device, args.panns_ckpt, args.val_metadata, args.max_samples)
    if args.target in ("df_arena", "both"):
        check_df_arena_fp16(device, args.df_arena_ckpt, args.val_metadata, args.max_samples)


if __name__ == "__main__":
    main()
