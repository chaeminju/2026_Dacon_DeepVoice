"""약점 케이스(identify_weak_cases 결과)를 근거로 데이터 증강 계획을 세운다.

case_analysis.py의 evaluate_by_case -> identify_weak_cases 결과를 받아,
"각 케이스별로 얼마나 더 모아야 하는가"를 계산하고 추천 데이터셋과 함께
`augmentation_plan.json`으로 저장한다.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_analysis  # noqa: E402  (위 sys.path 조작 이후에 import)

# 케이스별 추천 데이터셋/생성 방법 (요구사항에 명시된 그대로).
# E는 별도 데이터셋이라기보다 "증강 기법"으로 적용한다는 점에 주의.
RECOMMENDED_DATASETS = {
    "A_impersonation": "In-The-Wild, 직접 RVC/YourTTS 생성",
    "B_institution_ars": "ASVspoof2019 LA, MLAAD",
    "C_realtime_vc": "ASVspoof2019 LA(VC), VCC",
    "D_partial_splice": "PartialSpoof, HAD",
    "E_phone_channel": "ASVspoof2021 DF, CFAD (증강 기법으로 적용)",
    "F_ai_cover_song": "SingFake, CtrSVDD",
    "G_full_ai_music": "SONICS, FakeMusicCaps",
    "H_hybrid_composed": "FakeMusicCaps+MLAAD/ASVspoof 직접 믹싱",
    "I_bgm_evasion": "real 음성에 배경음악을 인위적으로 믹싱해 자체 제작",
}


def _base_case_type(case_type_key: str) -> str:
    """evaluate_by_case가 만든 "F_ai_cover_song__voice" 같은 접미사 붙은 키에서
    원래 case_type("F_ai_cover_song")을 복원한다."""
    for suffix in ("__voice", "__music", "__music_presence"):
        if case_type_key.endswith(suffix):
            return case_type_key[: -len(suffix)]
    return case_type_key


def _normalize_weak_cases(weak_cases) -> list[str]:
    """`weak_cases`는 identify_weak_cases가 반환하는 DataFrame이거나, 이미
    추려진 case_type 문자열 리스트일 수 있다. 순서를 보존한 채 중복 제거."""
    if isinstance(weak_cases, pd.DataFrame):
        case_list = weak_cases["case_type"].tolist()
    else:
        case_list = list(weak_cases)
    seen = set()
    ordered = []
    for c in case_list:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def build_augmentation_plan(
    weak_cases,
    current_counts_dict: dict[str, int],
    increase_ratio: float = 0.2,
    out_path: str | Path = "augmentation_plan.json",
) -> dict:
    """약점 케이스별로 `current * (1 + increase_ratio)`를 목표 개수로 잡고
    부족분(need)을 계산한다.

    - weak_cases: identify_weak_cases()의 반환 DataFrame 또는 case_type 문자열 리스트.
    - current_counts_dict: {"D_partial_splice": 1000, ...} 케이스별 현재 보유 샘플 수.
      "F_ai_cover_song__voice"처럼 접미사가 붙은 키가 넘어오면 원래 case_type
      ("F_ai_cover_song")의 카운트로 대체 조회한다.

    반환/저장 형식 예:
      {"D_partial_splice": {"current": 1000, "target": 1200, "need": 200,
                             "recommended_dataset": "PartialSpoof, HAD"}}
    """
    case_list = _normalize_weak_cases(weak_cases)
    plan = {}
    for case in case_list:
        base = _base_case_type(case)
        current = current_counts_dict.get(case, current_counts_dict.get(base, 0))
        target = math.ceil(current * (1 + increase_ratio))
        need = max(0, target - current)
        plan[case] = {
            "current": current,
            "target": target,
            "need": need,
            "recommended_dataset": RECOMMENDED_DATASETS.get(base, "TBD"),
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame.from_dict(plan, orient="index").reset_index(names="case_type")
    print(f"\n=== 데이터 증강 계획 (increase_ratio={increase_ratio}) ===")
    print(table.to_string(index=False))
    print(f"augmentation_plan -> {out_path}")
    return plan


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="약점 케이스 기반 데이터 증강 계획 수립")
    parser.add_argument("--case-performance-csv", required=True, help="evaluate_by_case가 저장한 CSV")
    parser.add_argument("--metric-col", default="EER")
    parser.add_argument("--percentile", type=float, default=25)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--current-counts", required=True, help='{"case_type": count, ...} JSON 파일')
    parser.add_argument("--increase-ratio", type=float, default=0.2)
    parser.add_argument("--out", default="augmentation_plan.json")
    args = parser.parse_args()

    df = pd.read_csv(args.case_performance_csv)
    weak_df = case_analysis.identify_weak_cases(
        df, args.metric_col, args.percentile, lower_is_better=not args.higher_is_better
    )
    current_counts = json.loads(Path(args.current_counts).read_text(encoding="utf-8"))
    build_augmentation_plan(weak_df, current_counts, args.increase_ratio, args.out)


if __name__ == "__main__":
    main()
