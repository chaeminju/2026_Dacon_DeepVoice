# case_pipeline — 케이스별 성능 분석 & 약점 기반 데이터 증강

9개 `case_type`(A~I)별로 검증셋 성능을 쪼개서 보고, 약한 케이스를 자동으로
찾아 증강 계획을 세우고, 증강된 데이터로 voice/music/presence 세 모델을
분리 재학습하는 파이프라인. **기존 학습/추론 스크립트(`training/`,
`submit_final_koen_v2/script.py` 등)는 전혀 수정하지 않고**, 이 폴더의
세 모듈을 추가로 실행해서 통합한다.

- `case_analysis.py` — 메타데이터 스키마(load/save) + `evaluate_by_case` +
  `identify_weak_cases` + `plot_case_performance` + `append_experiment_log`
- `augmentation_planner.py` — `build_augmentation_plan` (약점 케이스 → 목표
  샘플 수 → `augmentation_plan.json`)
- `retrain_pipeline.py` — `retrain_with_augmented_data`(voice/music 분리 학습)
  + `finetune_presence_head`(I_bgm_evasion 전용, PANNs 헤드만) + 반복 로그 기록

## 1. 메타데이터 준비

각 샘플을 아래 형태의 레코드로 만들어 `.jsonl`(권장, 대용량) 또는 `.json`
배열로 저장한다 (`case_analysis.save_metadata`).

```json
{"file_id": "sample_0001", "path": "data/processed/.../x.wav",
 "case_types": ["D_partial_splice"], "has_voice": true, "has_music": false,
 "voice_label": "fake", "music_label": null, "source_dataset": "PartialSpoof"}
```

`case_types`는 `case_analysis.ALL_CASE_TYPES` 중 1개 이상. F/H는
`has_voice=has_music=true`로 voice_label/music_label을 모두 채운다. I는
fake 라벨이 없어도 되지만, presence 강건성 평가를 위해 "voice+bgm 섞임"
(`has_voice=true`)과 "bgm만"(`has_voice=false`) 샘플을 둘 다 준비해야
`presence_auc`를 계산할 수 있다.

## 2. 케이스별 평가 → 약점 식별

```bash
python case_pipeline/case_analysis.py \
  --metadata manifests/case_val_metadata.jsonl \
  --out-csv case_performance.csv --out-plot case_performance.png \
  --metric-col EER --percentile 25
```

내부적으로 `submit_final_koen_v2/script.py`의 HTDemucs/PANNs/DF-Arena/
SpecTTTra 로딩·추론 함수를 `importlib`로 그대로 재사용한다
(`load_default_pipeline_models`). 다른 체크포인트를 쓰려면
`--pipeline-script`로 다른 script.py 경로를 지정하면 된다.

## 3. 증강 계획

```bash
python case_pipeline/augmentation_planner.py \
  --case-performance-csv case_performance.csv --metric-col EER \
  --current-counts current_counts.json --increase-ratio 0.2 \
  --out augmentation_plan.json
```

`current_counts.json`은 `{"case_type": 현재_샘플수, ...}` 형태. 결과
`augmentation_plan.json`의 각 항목에는 `current`/`target`/`need`와 함께
추천 데이터셋(`recommended_dataset`)이 같이 저장된다.

## 4. 재학습

```bash
python case_pipeline/retrain_pipeline.py \
  --train-metadata manifests/case_train_metadata_v2.jsonl \
  --val-metadata manifests/case_val_metadata.jsonl \
  --voice-out-dir data/processed/df_arena_v3 \
  --music-out-dir data/processed/spectttra_v3 \
  --iteration 2 --log-path experiment_log.csv
```

voice_label이 있는 A~E/F/H 샘플만 DF-Arena를, music_label이 있는 F/G/H
샘플만 SpecTTTra를 업데이트한다(같은 배치 루프 안에서 필터링). I_bgm_evasion
데이터는 별도로 PANNs `fc_audioset` 헤드만 재학습한다(`--skip-presence`로
끌 수 있음). `--case-weights '{"D_partial_splice": 2.0}'` 같은 JSON을 주면
`WeightedRandomSampler`로 약점 케이스를 더 자주 샘플링한다.

## 5. 반복

`--val-metadata`를 주면 재학습 직후 `evaluate_by_case`를 다시 돌려
`experiment_log.csv`에 `iteration, case_type, n_samples, EER, F1, timestamp`
행을 누적 저장한다 — "1차 학습 → 평가 → 증강 → 2차 학습 → 평가 → ..." 흐름의
개선 추이를 이 CSV 하나로 추적한다.
