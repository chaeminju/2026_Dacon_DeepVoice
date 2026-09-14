"""케이스별(9종 딥보이스 case_type) 성능 분석 모듈.

배경
----
샘플은 아래 9개 `case_type` 중 하나 이상에 해당할 수 있다 (F, H는 voice/music
라벨을 모두 가짐).

  [voice, 보이스피싱]
    A_impersonation   지인/가족 사칭 (Voice Cloning, Zero-shot TTS/VC)
    B_institution_ars 기관 사칭 ARS (정형화된 TTS)
    C_realtime_vc     실시간 음성변조 (Real-time VC)
    D_partial_splice  녹취 짜깁기 (Partial Spoof)
    E_phone_channel   전화 품질 위장 (코덱/노이즈 후처리, A~D 공통 레이어)
  [music]
    F_ai_cover_song   AI 커버곡/가짜 가창 (voice+music 라벨 모두 존재)
    G_full_ai_music   완전 AI 작곡/생성곡 (music 라벨만 존재)
    H_hybrid_composed 반주 AI생성 + 보컬 합성 (voice+music 라벨 모두 존재)
    I_bgm_evasion     배경음악 위장/탐지 회피 — fake 여부가 아니라
                      PANNs(존재판별)/HTDemucs(분리) 강건성을 보는 케이스라
                      다른 케이스와 별도로 평가한다.

실제 제출 파이프라인(`submit_final_koen_v2/script.py`)과 동일하게, 이 모듈도
"HTDemucs로 voice_audio/music_audio 분리 → PANNs로 존재 확률 → voice_audio는
DF-Arena(VOICE_FAKE_PROB), music_audio는 SpecTTTra(MUSIC_FAKE_PROB)" 구조를
그대로 재사용한다. 기존 학습/추론 스크립트는 전혀 수정하지 않고, 이 모듈에서
`importlib`로 그 스크립트의 함수만 가져다 쓴다 (`load_default_pipeline_models`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# 0. 케이스 타입 정의
# -----------------------------------------------------------------------------

VOICE_CASE_TYPES = [
    "A_impersonation",
    "B_institution_ars",
    "C_realtime_vc",
    "D_partial_splice",
    "E_phone_channel",
]
DUAL_LABEL_CASE_TYPES = ["F_ai_cover_song", "H_hybrid_composed"]  # voice+music 라벨 모두
MUSIC_ONLY_CASE_TYPES = ["G_full_ai_music"]
PRESENCE_ONLY_CASE_TYPES = ["I_bgm_evasion"]  # fake 라벨이 아니라 존재판별 강건성 평가

ALL_CASE_TYPES = (
    VOICE_CASE_TYPES + DUAL_LABEL_CASE_TYPES + MUSIC_ONLY_CASE_TYPES + PRESENCE_ONLY_CASE_TYPES
)

VOICE_LABELS = {"real", "fake"}
MUSIC_LABELS = {"real", "fake"}


# -----------------------------------------------------------------------------
# 1. 메타데이터 스키마 + load/save
# -----------------------------------------------------------------------------
#
# {
#   "file_id": "sample_0001",              # 필수, 고유 식별자
#   "path": "data/processed/.../x.wav",     # 필수, BASE_DIR(프로젝트 루트) 기준 상대경로
#   "case_types": ["D_partial_splice"],     # 필수, ALL_CASE_TYPES 중 1개 이상
#                                           #   (F, H는 두 개 이상의 케이스에 동시 소속 가능,
#                                           #    예: ["F_ai_cover_song"]이면서 has_voice/has_music 둘 다 True)
#   "has_voice": true,
#   "has_music": false,
#   "voice_label": "fake",                 # "real" | "fake" | null(해당 없음 — 예: G, 순수 배경음만)
#   "music_label": null,                   # "real" | "fake" | null(해당 없음 — 예: A~E 순수 음성만)
#   "source_dataset": "PartialSpoof"
# }
#
# - F_ai_cover_song / H_hybrid_composed: has_voice=has_music=true, voice_label과
#   music_label을 모두 채운다 (voice 관점/music 관점을 둘 다 평가하기 위함).
# - I_bgm_evasion: fake 판별 목적이 아니므로 voice_label/music_label은 null이어도
#   되고, 대신 has_voice/has_music(=실제 음성/배경음악이 섞여 있는지)를 "정답
#   존재 여부" 라벨로 사용해 presence_accuracy/presence_auc를 계산한다. 강건성
#   테스트를 제대로 하려면 "voice+bgm 섞임"(has_voice=true)과 "bgm만 존재"
#   (has_voice=false) 샘플을 함께 준비해야 한다(양성/음성 둘 다 있어야 AUC 계산 가능).

REQUIRED_FIELDS = ["file_id", "path", "case_types", "has_voice", "has_music"]
OPTIONAL_FIELDS = ["voice_label", "music_label", "source_dataset"]


def validate_metadata_record(record: dict) -> list[str]:
    """레코드 하나를 점검해 경고 메시지 목록을 반환한다 (문제가 있어도 raise하지
    않음 — I 케이스처럼 라벨이 없는 것도 정상이라 엄격한 검증은 부적절)."""
    warnings = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            warnings.append(f"필수 필드 누락: {field}")

    case_types = record.get("case_types") or []
    if not case_types:
        warnings.append("case_types가 비어 있음")
    unknown = [c for c in case_types if c not in ALL_CASE_TYPES]
    if unknown:
        warnings.append(f"알 수 없는 case_type: {unknown}")

    voice_label = record.get("voice_label")
    if voice_label is not None and voice_label not in VOICE_LABELS:
        warnings.append(f"voice_label 값이 이상함: {voice_label!r}")
    music_label = record.get("music_label")
    if music_label is not None and music_label not in MUSIC_LABELS:
        warnings.append(f"music_label 값이 이상함: {music_label!r}")

    # has_voice/has_music이 둘 다 True인데(=진짜 voice+music 결합 샘플인데) 라벨이
    # 하나라도 비어 있으면 경고. has_voice/has_music 중 하나만 True인 단면적
    # real 재사용(예: H_hybrid_composed에 real 음성만/real 음악만 negative로
    # 얹는 경우)은 정상이라 경고하지 않는다.
    dual = set(case_types) & set(DUAL_LABEL_CASE_TYPES)
    if dual and record.get("has_voice") and record.get("has_music") and (voice_label is None or music_label is None):
        warnings.append(f"{dual}는 voice_label/music_label을 둘 다 채우는 것을 권장함")
    return warnings


def load_metadata(path: str | Path) -> list[dict]:
    """`.jsonl`(한 줄에 레코드 하나, 대용량에 적합)과 `.json`(레코드 배열) 둘 다
    지원한다. 확장자로 자동 판별."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없음: {path}")

    if path.suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}") from e
    else:
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError(f"{path}는 JSON 배열(레코드 리스트) 형식이어야 함")

    n_warnings = 0
    for record in records:
        warnings = validate_metadata_record(record)
        if warnings:
            n_warnings += 1
            if n_warnings <= 10:
                print(f"[warn] {record.get('file_id', '?')}: {'; '.join(warnings)}")
    if n_warnings > 10:
        print(f"[warn] ...외 {n_warnings - 10}건 추가 경고 생략")
    print(f"load_metadata: {len(records)}개 레코드 로드 ({path})")
    return records


def save_metadata(records: Iterable[dict], path: str | Path) -> None:
    """`.jsonl`/`.json` 확장자에 맞춰 저장. 상위 디렉터리는 자동 생성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)

    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"save_metadata: {len(records)}개 레코드 저장 -> {path}")


# -----------------------------------------------------------------------------
# 2. 평가용 데이터셋 / 데이터로더
# -----------------------------------------------------------------------------


class CaseMetadataDataset(Dataset):
    """메타데이터 레코드 리스트를 읽어 평가에 필요한 필드 + 오디오 파일 경로를
    반환한다. 실제 오디오 로딩(HTDemucs 분리 포함)은 evaluate_by_case가 모델
    어댑터를 통해 수행한다 — HTDemucs가 파일 경로 기반 로더(`load_track`)를
    쓰기 때문에, 여기서는 파형을 미리 읽지 않고 경로만 넘겨 중복 로딩을 피한다.
    """

    def __init__(self, records: list[dict], audio_root: str | Path | None = None):
        self.records = records
        self.audio_root = Path(audio_root) if audio_root else BASE_DIR

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "file_id": r["file_id"],
            "audio_path": self.audio_root / r["path"],
            "case_types": list(r.get("case_types") or []),
            "has_voice": bool(r.get("has_voice", False)),
            "has_music": bool(r.get("has_music", False)),
            "voice_label": r.get("voice_label"),
            "music_label": r.get("music_label"),
            "source_dataset": r.get("source_dataset"),
        }


def build_case_dataloader(
    records: list[dict], audio_root: str | Path | None = None, num_workers: int = 0
) -> DataLoader:
    """batch_size=1 고정 (오디오 길이가 샘플마다 달라 배치 스태킹이 불가능한
    기존 학습 스크립트들의 관례와 동일). collate_fn으로 리스트를 dict 하나로
    풀어줘서 evaluate_by_case에서 `for sample in val_dataloader:`로 바로 순회 가능."""
    dataset = CaseMetadataDataset(records, audio_root=audio_root)
    return DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )


# -----------------------------------------------------------------------------
# 3. 지표 계산
# -----------------------------------------------------------------------------


def compute_eer(y_true: list[int], y_score: list[float]) -> float:
    """Equal Error Rate: FPR과 FNR(=1-TPR)이 가장 가까워지는 지점의 평균."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_binary_metrics(
    y_true: list[int], y_score: list[float], threshold: float = 0.5
) -> dict[str, float]:
    """fake(1) / real(0) 이진분류에 대한 EER, F1, Accuracy. 클래스가 하나뿐이면
    (예: 아직 그 케이스의 real 샘플이 없음) 계산 불가하므로 NaN + 경고."""
    if len(set(y_true)) < 2:
        return {"EER": float("nan"), "F1": float("nan"), "Accuracy": float("nan")}

    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return {
        "EER": compute_eer(y_true, y_score),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compute_presence_metrics(
    y_true: list[int], y_score: list[float], threshold: float = 0.5
) -> dict[str, float]:
    """I_bgm_evasion 전용: fake 여부가 아니라 '배경음악이 섞였을 때도
    VOICE_PRESENT_PROB/MUSIC_PRESENT_PROB이 정확한가'를 본다."""
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else float("nan")
    if len(set(y_true)) < 2:
        auc = float("nan")  # 양성/음성 둘 다 있어야 AUC 계산 가능
    else:
        auc = float(roc_auc_score(y_true, y_score))
    return {"presence_accuracy": accuracy, "presence_auc": auc}


def _append_score(store: dict, label: Any, score: Optional[float]) -> None:
    if score is None or label is None:
        return
    if isinstance(label, bool):
        y = int(label)
    elif label == "fake":
        y = 1
    elif label == "real":
        y = 0
    else:
        return
    store["y_true"].append(y)
    store["y_score"].append(float(score))


# -----------------------------------------------------------------------------
# 4. 케이스별 평가 본체
# -----------------------------------------------------------------------------


def evaluate_by_case(
    voice_model, music_model, panns_model, htdemucs_model, val_dataloader, verbose: bool = True
) -> pd.DataFrame:
    """9개 case_type을 아래처럼 나눠서 평가한다.

    - A~E, F/H(voice 관점), F/H(music 관점), G: EER / F1 / Accuracy
      (F, H는 `"F_ai_cover_song__voice"`, `"F_ai_cover_song__music"`처럼
      `__voice`/`__music` 접미사가 붙은 case_type 행으로 분리되어 나온다)
    - I: presence_accuracy / presence_auc (fake 지표는 전부 NaN)

    모델 인자는 아래 프로토콜(덕타이핑)만 만족하면 되고, 실제 구현체는
    `load_default_pipeline_models()`가 기존 제출 스크립트를 감싸서 제공한다:
      - voice_model.predict_fake(audio: np.ndarray) -> float
      - music_model.predict_fake(audio: np.ndarray) -> float
      - panns_model.predict_presence(audio: np.ndarray, kind: "voice"|"music") -> float
      - htdemucs_model.separate(audio_path) -> (voice_audio, music_audio)
    """
    fake_scores: dict[str, dict] = defaultdict(lambda: {"y_true": [], "y_score": []})
    presence_scores: dict[str, dict] = defaultdict(lambda: {"y_true": [], "y_score": []})
    presence_scores_music: dict[str, dict] = defaultdict(lambda: {"y_true": [], "y_score": []})

    iterator = tqdm(val_dataloader, desc="evaluate_by_case") if verbose else val_dataloader
    n_failed = 0
    for sample in iterator:
        case_types = sample["case_types"]
        try:
            voice_audio, music_audio = htdemucs_model.separate(sample["audio_path"])
        except Exception as e:  # 개별 파일 손상 등으로 전체 평가가 죽지 않도록 함
            n_failed += 1
            print(f"[warn] HTDemucs 분리 실패, 샘플 스킵: file_id={sample['file_id']} ({e})")
            continue

        voice_present_prob = panns_model.predict_presence(voice_audio, kind="voice")
        music_present_prob = panns_model.predict_presence(music_audio, kind="music")

        needs_voice_fake = sample["has_voice"] and sample["voice_label"] in VOICE_LABELS
        needs_music_fake = sample["has_music"] and sample["music_label"] in MUSIC_LABELS
        voice_fake_prob = voice_model.predict_fake(voice_audio) if needs_voice_fake else None
        music_fake_prob = music_model.predict_fake(music_audio) if needs_music_fake else None

        for case in case_types:
            if case in VOICE_CASE_TYPES:
                _append_score(fake_scores[case], sample["voice_label"], voice_fake_prob)
            elif case in DUAL_LABEL_CASE_TYPES:
                _append_score(fake_scores[f"{case}__voice"], sample["voice_label"], voice_fake_prob)
                _append_score(fake_scores[f"{case}__music"], sample["music_label"], music_fake_prob)
            elif case in MUSIC_ONLY_CASE_TYPES:
                _append_score(fake_scores[case], sample["music_label"], music_fake_prob)
            elif case in PRESENCE_ONLY_CASE_TYPES:
                _append_score(presence_scores[case], sample["has_voice"], voice_present_prob)
                _append_score(presence_scores_music[case], sample["has_music"], music_present_prob)
            else:
                print(f"[warn] 알 수 없는 case_type 무시: {case} (file_id={sample['file_id']})")

    if n_failed:
        print(f"[warn] 총 {n_failed}개 샘플이 HTDemucs 분리 실패로 평가에서 제외됨")

    rows = []
    for case, store in fake_scores.items():
        metrics = compute_binary_metrics(store["y_true"], store["y_score"])
        rows.append({"case_type": case, "metric_type": "fake_detection",
                      "n_samples": len(store["y_true"]), **metrics})
    for case, store in presence_scores.items():
        metrics = compute_presence_metrics(store["y_true"], store["y_score"])
        rows.append({"case_type": case, "metric_type": "presence_robustness(voice)",
                      "n_samples": len(store["y_true"]), **metrics})
    for case, store in presence_scores_music.items():
        metrics = compute_presence_metrics(store["y_true"], store["y_score"])
        rows.append({"case_type": f"{case}__music_presence", "metric_type": "presence_robustness(music)",
                      "n_samples": len(store["y_true"]), **metrics})

    df = pd.DataFrame(rows)
    for col in ["EER", "F1", "Accuracy", "presence_accuracy", "presence_auc"]:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[["case_type", "metric_type", "n_samples", "EER", "F1", "Accuracy",
              "presence_accuracy", "presence_auc"]]
    df = df.sort_values("case_type").reset_index(drop=True)

    print("\n=== case_type별 성능 ===")
    print(df.to_string(index=False))
    return df


def plot_case_performance(df: pd.DataFrame, out_path: str | Path | None = "case_performance.png") -> None:
    """fake_detection 케이스는 EER/F1/Accuracy 그룹 막대그래프로, presence
    케이스(I)는 별도 서브플롯으로 그린다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fake_df = df[df["metric_type"] == "fake_detection"]
    presence_df = df[df["metric_type"].str.startswith("presence_robustness", na=False)]

    n_plots = int(len(fake_df) > 0) + int(len(presence_df) > 0)
    if n_plots == 0:
        print("[warn] 그릴 데이터가 없음 (plot_case_performance 스킵)")
        return

    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(7 * max(n_plots, 1), 5))
    axes = np.atleast_1d(axes)
    ax_idx = 0

    if len(fake_df) > 0:
        ax = axes[ax_idx]
        ax_idx += 1
        x = np.arange(len(fake_df))
        width = 0.25
        ax.bar(x - width, fake_df["EER"], width, label="EER (lower is better)")
        ax.bar(x, fake_df["F1"], width, label="F1")
        ax.bar(x + width, fake_df["Accuracy"], width, label="Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(fake_df["case_type"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        # 참고: 기본 matplotlib 폰트(DejaVu Sans)가 한글 글리프를 지원하지 않아
        # 라벨은 영문으로 표기함(한글 폰트를 쓰려면 plt.rcParams["font.family"] 설정).
        ax.set_title("Fake detection cases (A-E, F/H, G)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    if len(presence_df) > 0:
        ax = axes[ax_idx]
        x = np.arange(len(presence_df))
        width = 0.35
        ax.bar(x - width / 2, presence_df["presence_accuracy"], width, label="presence_accuracy")
        ax.bar(x + width / 2, presence_df["presence_auc"], width, label="presence_auc")
        ax.set_xticks(x)
        ax.set_xticklabels(presence_df["case_type"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title("I_bgm_evasion: presence-detection robustness")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"plot_case_performance: 저장됨 -> {out_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. 약점 케이스 자동 식별
# -----------------------------------------------------------------------------


def identify_weak_cases(
    case_performance_df: pd.DataFrame,
    metric_col: str,
    threshold_percentile: float = 25,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """`metric_col` 기준 하위 `threshold_percentile`%를 "약점 케이스"로 반환한다.

    - lower_is_better=True (EER 등): 값이 클수록 나쁨 -> 상위(100-p)% 퍼센타일
      이상인 케이스가 "하위 p%"에 해당.
    - lower_is_better=False (F1/Accuracy 등): 값이 작을수록 나쁨 -> 하위 p%
      퍼센타일 이하인 케이스.
    """
    df = case_performance_df.dropna(subset=[metric_col]).copy()
    if df.empty:
        print(f"[warn] {metric_col}에 유효한 값이 있는 케이스가 없음")
        return df

    values = df[metric_col].to_numpy()
    if lower_is_better:
        cutoff = np.percentile(values, 100 - threshold_percentile)
        weak_mask = df[metric_col] >= cutoff
    else:
        cutoff = np.percentile(values, threshold_percentile)
        weak_mask = df[metric_col] <= cutoff

    weak_df = df[weak_mask].sort_values(metric_col, ascending=lower_is_better).reset_index(drop=True)

    direction = "높을수록 나쁨" if lower_is_better else "낮을수록 나쁨"
    print(f"\n=== 약점 케이스 (metric={metric_col}, {direction}, 하위 {threshold_percentile}%, cutoff={cutoff:.4f}) ===")
    print(weak_df[["case_type", metric_col, "n_samples"]].to_string(index=False))
    return weak_df


# -----------------------------------------------------------------------------
# 6. 반복 실험 로그
# -----------------------------------------------------------------------------

EXPERIMENT_LOG_COLUMNS = ["iteration", "case_type", "n_samples", "EER", "F1", "timestamp"]


def append_experiment_log(
    iteration: int, case_performance_df: pd.DataFrame, log_path: str | Path = "experiment_log.csv"
) -> None:
    """"1차 학습 → 증강 → 2차 학습 → 증강 ..." 반복마다 호출해 case_type별
    성능을 CSV에 누적 저장한다. 파일이 없으면 헤더를 새로 쓰고, 있으면 append."""
    log_path = Path(log_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = case_performance_df[["case_type", "n_samples", "EER", "F1"]].copy()
    rows.insert(0, "iteration", iteration)
    rows["timestamp"] = timestamp
    rows = rows[EXPERIMENT_LOG_COLUMNS]

    write_header = not log_path.is_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(log_path, mode="a", header=write_header, index=False)
    print(f"append_experiment_log: iteration={iteration}, {len(rows)}행 -> {log_path}")


# -----------------------------------------------------------------------------
# 7. 기본 모델 어댑터 (기존 제출 스크립트를 건드리지 않고 재사용)
# -----------------------------------------------------------------------------
#
# submit_final_koen_v2/script.py는 HTDemucs -> PANNs -> DF-Arena(voice) ->
# SpecTTTra(music) 파이프라인을 이미 구현/검증해뒀다. 여기서는 그 파일을
# importlib로 "그냥 읽어서" 함수만 재사용한다 (해당 파일은 전혀 수정하지 않음).

_PIPELINE_MODULE_CACHE: dict[str, Any] = {}


def _load_submit_pipeline_module(script_path: str | Path | None = None):
    script_path = Path(script_path) if script_path else (BASE_DIR / "submit_final_koen_v2" / "script.py")
    key = str(script_path)
    if key in _PIPELINE_MODULE_CACHE:
        return _PIPELINE_MODULE_CACHE[key]
    if not script_path.is_file():
        raise FileNotFoundError(
            f"제출 파이프라인 스크립트를 찾을 수 없음: {script_path}. "
            "voice_model/music_model/panns_model/htdemucs_model을 직접 만들어 "
            "evaluate_by_case에 넘기거나 script_path를 지정하세요."
        )
    spec = importlib.util.spec_from_file_location("_case_pipeline_submit_script", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PIPELINE_MODULE_CACHE[key] = module
    return module


class HTDemucsSeparator:
    """`htdemucs_model.separate(audio_path) -> (voice_audio, music_audio)`"""

    def __init__(self, device, script_path: str | Path | None = None):
        self._pipeline = _load_submit_pipeline_module(script_path)
        self._model = self._pipeline.load_htdemucs_model()
        self._device = device

    def separate(self, audio_path):
        return self._pipeline.separate_voice_and_music(Path(audio_path), self._model, self._device)


class PANNsPresenceModel:
    """`panns_model.predict_presence(audio, kind="voice"|"music") -> float`.
    `.voice_indices`/`.music_indices`/`.model`은 retrain_pipeline의 PANNs 분류
    헤드 파인튜닝에서 재사용한다."""

    def __init__(self, device, script_path: str | Path | None = None):
        self._pipeline = _load_submit_pipeline_module(script_path)
        self.model, self.voice_indices, self.music_indices = self._pipeline.load_panns_model(device)

    def predict_presence(self, audio, kind: str):
        indices = self.voice_indices if kind == "voice" else self.music_indices
        return self._pipeline.predict_presence(self.model, indices, audio)


class DFArenaVoiceModel:
    """`voice_model.predict_fake(audio) -> float` (DF-Arena-1B)."""

    def __init__(self, device, script_path: str | Path | None = None):
        self._pipeline = _load_submit_pipeline_module(script_path)
        self._device = device
        self.model, self._fake_label_index = self._pipeline.load_df_arena_model(device)

    def predict_fake(self, audio):
        return self._pipeline.predict_fake(self.model, self._fake_label_index, audio, self._device)


class SpecTTTraMusicModel:
    """`music_model.predict_fake(audio) -> float` (SpecTTTra)."""

    def __init__(self, device, script_path: str | Path | None = None):
        self._pipeline = _load_submit_pipeline_module(script_path)
        self._device = device
        self.model, self.config = self._pipeline.load_spectttra_model(device)

    def predict_fake(self, audio):
        return self._pipeline.predict_fake_music(self.model, self.config, audio, self._device)


def load_default_pipeline_models(device, script_path: str | Path | None = None):
    """evaluate_by_case(voice_model, music_model, panns_model, htdemucs_model, ...)
    순서 그대로 4개 어댑터를 만들어 반환한다."""
    voice_model = DFArenaVoiceModel(device, script_path)
    music_model = SpecTTTraMusicModel(device, script_path)
    panns_model = PANNsPresenceModel(device, script_path)
    htdemucs_model = HTDemucsSeparator(device, script_path)
    return voice_model, music_model, panns_model, htdemucs_model


# -----------------------------------------------------------------------------
# 8. CLI
# -----------------------------------------------------------------------------


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="케이스별 성능 분석 (evaluate_by_case + identify_weak_cases)")
    parser.add_argument("--metadata", required=True, help="load_metadata로 읽을 .json/.jsonl 경로")
    parser.add_argument("--audio-root", default=None, help="path 필드의 기준 디렉터리 (기본: 프로젝트 루트)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pipeline-script", default=None, help="어댑터가 재사용할 script.py 경로 (기본: submit_final_koen_v2/script.py)")
    parser.add_argument("--out-csv", default="case_performance.csv")
    parser.add_argument("--out-plot", default="case_performance.png")
    parser.add_argument("--metric-col", default="EER", help="identify_weak_cases에 쓸 기준 지표")
    parser.add_argument("--percentile", type=float, default=25)
    parser.add_argument("--higher-is-better", action="store_true", help="metric-col이 F1/Accuracy처럼 높을수록 좋은 지표면 지정")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = load_metadata(args.metadata)
    dataloader = build_case_dataloader(records, audio_root=args.audio_root)

    voice_model, music_model, panns_model, htdemucs_model = load_default_pipeline_models(
        device, script_path=args.pipeline_script
    )
    df = evaluate_by_case(voice_model, music_model, panns_model, htdemucs_model, dataloader)
    df.to_csv(args.out_csv, index=False)
    print(f"case_performance -> {args.out_csv}")
    plot_case_performance(df, out_path=args.out_plot)
    identify_weak_cases(df, args.metric_col, args.percentile, lower_is_better=not args.higher_is_better)


if __name__ == "__main__":
    main()
