"""기존 manifest(`manifests_v2/{ko,en}_{train,val}.csv`, `manifests/music_
{train,valid}_manifest.csv`)를 `case_analysis`의 9-케이스 메타데이터 스키마로
변환한다. 새 데이터를 만들지 않고 "지금 가진 데이터를 case_type 기준으로
재분류"만 하는 1단계 스크립트.

매핑 규칙 (전부 아래 상수/함수에 모여 있어 필요하면 쉽게 조정 가능)
-------------------------------------------------------------
voice(ko/en 공통, `technique`/`source`/`augmentation` 컬럼 기준):
  - technique == "real"(bonafide): 공격 유형이 없는 "진짜" 샘플이므로, A~D
    네 케이스 전부의 평가에 negative(real) 클래스로 재사용한다
    (case_types = [A, B, C, D]).
  - technique == "tts": B_institution_ars로 분류. (MLAAD 12종 + VITS 자체생성
    — "정형화된 TTS" 전반을 대표하는 것으로 봄. ARS 전용 데이터셋은 아직 없음.)
  - technique == "vc":
      - source가 "rvc_from_"로 시작 (특정 화자 1명을 목표로 변환) ->
        A_impersonation (특정인 목소리를 노려 클로닝했다는 점이 "지인/가족
        사칭"에 가장 가까움)
      - 그 외(ASVspoof2019 LA의 A05/A06 VC 공격 — 특정 인물을 노리지 않는
        범용 변환 알고리즘) -> C_realtime_vc (완전히 같은 개념은 아니지만
        현재 보유 데이터 중 "실시간 적용 가능한 범용 VC"에 가장 가까운
        후보라 임시로 배정. 실제 실시간 VC 전용 데이터 확보 전까지의 잠정치)
  - augmentation에 "codec"이 포함되면(= codec_roundtrip.py로 전화 채널 열화를
    이미 입힌 샘플) 원래 case_type에 더해 E_phone_channel도 추가한다
    (E는 "A~D에 얹히는 레이어"라 단독이 아니라 다중 라벨로 취급).

music(`manifests/music_{train,valid}_manifest.csv`, label/target 컬럼 기준):
  - label == "fake" (SONICS fake_songs, AI 작곡+AI 가창 완전생성곡) ->
    G_full_ai_music
  - label == "real" (유튜브 실제곡) -> G_full_ai_music의 negative(real) 클래스

⚠️ 아직 매핑 가능한 데이터가 전혀 없는 케이스: D_partial_splice(D, 참고: A/C의
   real 재사용 대상이라 real 샘플은 있지만 D 전용 fake 샘플은 0개),
   F_ai_cover_song, H_hybrid_composed, I_bgm_evasion. 스크립트 실행 후 요약에서
   0으로 나오면 이 케이스들은 별도로 데이터를 확보해야 한다는 뜻이다
   (augmentation_planner.py가 그 다음 단계).

--expand-vc-pool (2단계 평가 이후 추가된 옵션)
-----------------------------------------------
1차 평가에서 A_impersonation/C_realtime_vc의 val fake 표본이 각각 3개/5개뿐인
문제가 드러났는데, 실제로는 **이미 디스크에 훨씬 많은 vc 샘플이 있는데
`manifests_v2` 빌드 때 real:fake 1:1 밸런싱 로직이 MLAAD(12,000개)에 압도돼
거의 다 버려진 것**이었다(`data/processed/{ko,en}_augmented_manifest.csv`에
케이스당 600개씩, 총 1,200개가 이미 존재 — 그중 지금까지 학습/평가에 쓰인
건 86개뿐, 1,114개가 그냥 안 쓰이고 있었음). 새로 다운로드/생성할 필요 없이
그 미사용분을 case_metadata_val.jsonl에 추가로 편입시키는 옵션.
`manifests_v2/{ko,en}_train.csv`에 이미 쓰인 경로는 (모델이 학습 때 본
데이터라) 제외하고, 나머지 vc 행 전부를 source 규칙(A/C 분류) 그대로 적용해
추가한다. real은 새로 늘리지 않고 기존 val의 real(technique=real, 총
440개)을 그대로 재사용한다 — real 전체 풀(2,303개)의 대부분이 이미
train/val에서 소진돼(2,206개 사용, 97개만 미사용) 더 늘릴 여지가 거의 없기
때문.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_analysis  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent

VOICE_ATTACK_CASE_TYPES = ["A_impersonation", "B_institution_ars", "C_realtime_vc", "D_partial_splice"]
# H_hybrid_composed는 voice+music 이중라벨 케이스라 real 음성도 negative(voice_label="real")로
# 재사용한다 — has_music=False라 evaluate_by_case가 음성 관점(__voice)에만 반영하고
# 음악 관점(__music)에는 영향 없음(has_music 게이트로 자동 분리됨).
REAL_VOICE_CASE_TYPES = VOICE_ATTACK_CASE_TYPES + ["H_hybrid_composed"]


def _voice_case_types(row: pd.Series) -> tuple[list[str], str]:
    """(case_types, voice_label) 반환. voice_label은 항상 "real"|"fake".

    E_phone_channel은 real/fake 양쪽 다 codec 증강이 걸려 있으면 추가한다 —
    real 쪽도 포함해야 E 케이스 평가 시 real/fake 두 클래스가 모두 존재해서
    EER을 계산할 수 있다(한쪽 클래스만 있으면 계산 불가, compute_binary_metrics
    참고)."""
    if row["technique"] == "real":
        base, voice_label = list(REAL_VOICE_CASE_TYPES), "real"
    elif row["technique"] == "tts":
        base, voice_label = ["B_institution_ars"], "fake"
    elif row["technique"] == "vc":
        base = ["A_impersonation"] if str(row["source"]).startswith("rvc_from_") else ["C_realtime_vc"]
        voice_label = "fake"
    else:
        raise ValueError(f"알 수 없는 technique: {row['technique']!r} (path={row['path']})")

    augmentation = str(row.get("augmentation", ""))
    if "codec" in augmentation:
        base = base + ["E_phone_channel"]
    return base, voice_label


def convert_voice_manifest(csv_path: str | Path, id_prefix: str) -> list[dict]:
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        case_types, voice_label = _voice_case_types(row)
        records.append({
            "file_id": f"{id_prefix}_{i:06d}",
            "path": row["path"],
            "case_types": case_types,
            "has_voice": True,
            "has_music": False,
            "voice_label": voice_label,
            "music_label": None,
            "source_dataset": row["source"],
        })
    return records


def convert_music_manifest(csv_path: str | Path, id_prefix: str) -> list[dict]:
    """label=="real"(유튜브 실제곡)은 G_full_ai_music뿐 아니라
    H_hybrid_composed의 music 관점 negative로도 재사용한다(has_voice=False라
    evaluate_by_case가 __music 관점에만 반영, H_hybrid_composed 자체 합성
    스크립트가 만드는 fake 쪽과 짝이 맞음)."""
    df = pd.read_csv(csv_path)
    records = []
    for i, row in df.iterrows():
        case_types = ["G_full_ai_music"]
        if row["label"] == "real":
            case_types.append("H_hybrid_composed")
        records.append({
            "file_id": f"{id_prefix}_{i:06d}",
            "path": row["path"],
            "case_types": case_types,
            "has_voice": False,
            "has_music": True,
            "voice_label": None,
            "music_label": row["label"],  # "real" | "fake"
            "source_dataset": "sonics" if row["label"] == "fake" else "youtube_real_songs",
        })
    return records


FULL_VOICE_MANIFESTS = [
    "data/processed/ko_augmented_manifest.csv",
    "data/processed/en_augmented_manifest.csv",
]


def find_unused_vc_rows(already_used_paths: set[str]) -> pd.DataFrame:
    """`FULL_VOICE_MANIFESTS`(ko/en 각 600개, 총 1,200개 vc 샘플)에서
    `already_used_paths`(현재 배포 모델이 학습/평가에 이미 쓴 경로)에 없는,
    즉 한 번도 학습/평가에 안 쓰인 vc 행만 골라 반환한다."""
    full = pd.concat([pd.read_csv(BASE_DIR / p) for p in FULL_VOICE_MANIFESTS], ignore_index=True)
    vc_rows = full[full["technique"] == "vc"]
    unused = vc_rows[~vc_rows["path"].isin(already_used_paths)]
    print(f"find_unused_vc_rows: 전체 vc {len(vc_rows)}개 중 미사용 {len(unused)}개 발견")
    return unused


def expand_val_with_unused_vc(val_records: list[dict], voice_train_paths: list[str | Path]) -> list[dict]:
    """미사용 vc 행 전부를 A_impersonation/C_realtime_vc fake 샘플로 val에
    추가한다 (real은 기존 val의 real을 그대로 재사용하므로 새로 추가하지 않음).
    `voice_train_paths`(모델이 학습에 쓴 경로)뿐 아니라 `val_records`에 이미
    들어있는 경로도 제외해서, 같은 파일이 두 번 편입되는 것을 막는다."""
    already_used = {r["path"] for r in val_records}
    for p in voice_train_paths:
        already_used |= set(pd.read_csv(BASE_DIR / p)["path"])

    unused = find_unused_vc_rows(already_used)
    new_records = []
    for i, row in unused.iterrows():
        case_types, voice_label = _voice_case_types(row)
        new_records.append({
            "file_id": f"expand_vc_{i:06d}",
            "path": row["path"],
            "case_types": case_types,
            "has_voice": True,
            "has_music": False,
            "voice_label": voice_label,
            "music_label": None,
            "source_dataset": row["source"],
        })
    print(f"expand_val_with_unused_vc: val에 {len(new_records)}개 신규 fake 샘플 추가")
    return val_records + new_records


def print_case_type_summary(records: list[dict], title: str) -> None:
    counts = {c: 0 for c in case_analysis.ALL_CASE_TYPES}
    for r in records:
        for c in r["case_types"]:
            counts[c] += 1
    print(f"\n=== {title}: case_type별 샘플 수 (n={len(records)}) ===")
    for c in case_analysis.ALL_CASE_TYPES:
        flag = "  <- 데이터 없음, 별도 확보 필요" if counts[c] == 0 else ""
        print(f"  {c:22s} {counts[c]:6d}{flag}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="기존 manifest -> 9-케이스 메타데이터(.jsonl) 변환")
    parser.add_argument("--voice-train", nargs="+",
                         default=["manifests_v2/ko_train.csv", "manifests_v2/en_train.csv"])
    parser.add_argument("--voice-val", nargs="+",
                         default=["manifests_v2/ko_val.csv", "manifests_v2/en_val.csv"])
    parser.add_argument("--music-train", default="manifests/music_train_manifest.csv")
    parser.add_argument("--music-val", default="manifests/music_valid_manifest.csv")
    parser.add_argument("--out-train", default="manifests/case_metadata_train.jsonl")
    parser.add_argument("--out-val", default="manifests/case_metadata_val.jsonl")
    parser.add_argument(
        "--expand-vc-pool", action="store_true",
        help="A/C의 미사용 vc 샘플(현재 배포 모델 학습에 안 쓰인 것)을 전부 val에 추가",
    )
    args = parser.parse_args()

    train_records = []
    for path in args.voice_train:
        train_records += convert_voice_manifest(BASE_DIR / path, id_prefix=Path(path).stem)
    train_records += convert_music_manifest(BASE_DIR / args.music_train, id_prefix="music_train")

    val_records = []
    for path in args.voice_val:
        val_records += convert_voice_manifest(BASE_DIR / path, id_prefix=Path(path).stem)
    val_records += convert_music_manifest(BASE_DIR / args.music_val, id_prefix="music_val")

    if args.expand_vc_pool:
        val_records = expand_val_with_unused_vc(val_records, args.voice_train)

    case_analysis.save_metadata(train_records, args.out_train)
    case_analysis.save_metadata(val_records, args.out_val)

    print_case_type_summary(train_records, "train")
    print_case_type_summary(val_records, "val")


if __name__ == "__main__":
    main()
