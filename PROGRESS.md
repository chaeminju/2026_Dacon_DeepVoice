# 진행 상황 체크포인트

이 파일은 세션이 끊겨도 어디까지 진행됐는지 바로 파악하기 위한 기록. 각 단계는
디스크에 실제로 남는 산출물 기준으로 표시 — 파일/폴더가 있으면 그 단계는 재실행 불필요.
전체 계획은 `.claude/plans/virtual-cooking-boot.md` 참고.

## M0. 환경 세팅 — 완료
- conda env `deepvoice` (python 3.10): 베이스라인 추론 + VITS + 파인튜닝용
- conda env `rvc` (python 3.12): RVC 학습/변환 전용 (transformers<4.50, numpy<2 요구라 분리)
- `baseline_submit/script.py` 로컬 샘플 3개로 GPU 스모크 테스트 통과

## M1. Real 데이터 확보
- [x] KSS: `data/raw/kss/` (3.6GB, HF `Bingsu/KSS_Dataset`, parquet, 12,854행)
- [x] EN(ASVspoof2019 LA, SpoofCeleb 대신 사용 — 승인 대기 없이 즉시 가능):
      `data/processed/en_raw/` + `data/processed/en_raw_manifest.csv`
      (real 1200 + TTS-fake 600[A01-A04] + VC-fake 600[A05-A06] = 2400개, 16kHz)
- [ ] Zeroth-Korean: **다운로드 진행 중** (`data/raw/zeroth_korean/zeroth_korean.tar.gz`,
      openslr.org에서 10GB 중 진행, 속도 느려짐~0.85MB/s). **중단되면 `curl -C -`로
      이어받기** (서버가 Range 지원하는지는 미확인 — 안 되면 처음부터 재다운로드).
      완료 후 `tar xzf`로 압축 해제 필요.
- [ ] SpoofCeleb: HF gated 승인 대기 중 (참고용, 지금 파이프라인은 안 씀)

## M1 추가
- [x] Zeroth-Korean: `data/raw/zeroth_korean/train_data_01/` (22,720 flac, 105화자,
      화자별 `<spk>_<chapter>.trans.txt` 트랜스크립트 포함)

## M2. Fake 생성
- [x] KO TTS(VITS, mms-tts-kor): **600개 생성 완료** →
      `data/processed/ko_raw/tts/` + `data/processed/ko_tts_manifest.csv`
- [x] RVC 환경/자산 준비 완료: `third_party/rvc/assets/{hubert_base,rmvpe,pretrained_v2}`
- [x] RVC 파이프라인 end-to-end 검증 완료 (KSS 300클립 스모크 테스트로):
      전처리→F0→HuBERT→학습(재개 확인됨)→인덱스→변환 전부 성공.
      **해결한 이슈**: (1) 스크립트 직접 실행 시 `train/` 디렉터리가 sys.path와 충돌 →
      `python -m train.xxx` 모듈 방식으로 실행. (2) `/dev/shm`이 64MB로 매우 작아서
      DataLoader worker(기본 4개)가 Bus error로 죽음 → `third_party/rvc/train/train.py`의
      `num_workers=4`를 `0`으로 패치(파일시스템 공유 전략도 시도했으나 결국 POSIX shm을
      쓰므로 무의미, worker 자체를 없애야 함).
      재사용 스크립트: `fake_gen/rvc_train_speaker.sh <실험명> <소스wav디렉터리> [epoch]`
- [x] KO real manifest: Zeroth 803 + KSS 300 = 1,103개 →
      `data/processed/ko_raw/real/` + `data/processed/ko_real_manifest.csv`
      (`data_pipeline/build_ko_real_manifest.py`, RVC 타깃 화자 201/152/200/185는
      real 세트에서 제외)
- [x] KO RVC 화자변환 학습 — Zeroth-Korean 상위 4화자(201,152,200,185, 각 300클립)
      201/152/200 완료, **185 진행 중(F0 추출 단계)**
- [ ] KO RVC 변환 실행 (`fake_gen/rvc_convert_batch.py`, 화자당 150개씩 총 600개 목표) —
      185 끝나면 바로 실행
- [x] EN RVC 화자변환: 불필요 — ASVspoof2019 LA에 이미 A05/A06(VC 공격)이 라벨링되어
      있어서 그대로 서브샘플링함 (M1에서 완료)

## M3. 증강 — 스크립트 완료, 아직 실행 안 함
- `data_pipeline/rawboost.py`, `data_pipeline/codec_roundtrip.py`, `data_pipeline/augment.py`
  작성 및 개별 테스트 통과 (RawBoost/GSM 코덱 왕복 둘 다 정상 동작 확인)

## M4. Manifest 구성 — 스크립트 완료, 아직 실행 안 함
- `manifests/build_manifests.py`: real:fake 1:1 균형 + 3케이스 총 개수 동일화 +
  train/val 90/10 stratified split 로직 작성 완료

## M5. DF-Arena-1B 파인튜닝 — 스크립트 완료 + 스모크 테스트 통과
- `training/finetune_df_arena.py`: SSL 백본(wav2vec2-xls-r-1b) 고정, conformer
  헤드만 학습(185M/1.15B 파라미터). **20개 샘플 스모크 테스트에서 ~11 it/s로 매우
  빠름 확인** — 케이스당(~2000개, 3epoch) 대략 10분 내외로 예상, 애초 예상(1.5~2.5시간)
  보다 훨씬 빠를 전망.

## M6. 제출 zip 패키징 — KO 완료!
- `inference/build_submission.py`: baseline_submit/ 복제 → df_arena_1b 가중치만 교체
  → zip. (이 대회는 코드 제출 방식 — CSV 아님, 확인됨)
- **해결한 버그**: 원본 `pytorch_model.bin`(4.6GB)을 안 지우고 파인튜닝
  `model.safetensors`(4.5GB)를 옆에 추가만 해서 zip이 9GB로 부풀었던 문제 →
  df_arena_1b 폴더의 기존 가중치 파일을 먼저 삭제하도록 수정
- **`submissions/submission_ko.zip` (4.55GB) 완성, 로컬 샘플 3개로 실제
  script.py 실행까지 성공(5개 컬럼 정상 출력)**. 사용자가 데이콘에 직접 업로드하면 됨.

## KO 파인튜닝 결과 요약
- 데이터: real 1,103(Zeroth 803+KSS 300) + TTS-fake 600(VITS) + VC-fake 600(RVC,
  4화자) = 2,206개(증강 적용), train 1,986 / val 220
- 학습: DF-Arena-1B의 SSL 백본 고정, conformer 헤드만 파인튜닝(185M/1.15B 파라미터)
- **best val accuracy: 97.3%** (epoch 2, best-checkpoint 저장 로직으로 선택됨 —
  epoch 3에서 50%로 붕괴하는 걸 피함)

## EN, KO+EN 파인튜닝 결과
- EN: best val acc 95.45%(epoch2, epoch3에서 51.8%로 붕괴 — best-checkpoint 로직이 걸러줌)
- KO+EN: best val acc 96.36%(epoch1이 최고, 이후 계속 하락)
- 세 케이스 모두 동일 패턴: 후반 epoch에서 가끔 무작위 수준으로 붕괴 → best-checkpoint
  저장 로직 덕분에 실제 제출본은 항상 붕괴 전 최고 성능 시점으로 고정됨

## 사용자 리포트: "제출했더니 파일이 손상되었거나 유효하지 않다" (KO 제출 후)
원인을 추적해 2가지 실제 버그를 발견/수정함:

1. **`baseline_submit/requirements.txt`가 세션 중 0바이트로 손상되어 있었음**
   (원래 `matplotlib==3.9.2` 1줄). 원인 불명이나 발견 즉시 복구.
2. **zip이 4GiB(4,294,967,296B) 표준 ZIP 한도를 254MB 초과**(fp32 모델 4.59GB 포함해
   전체 4.55GB) — Zip64 확장이 필요한데 업로드 검증기가 이를 못 받아들여 "손상/유효하지
   않음"으로 뜬 것으로 추정됨. **fp16으로 가중치를 저장해 모델 파일을 2.3GB로 줄이고
   zip을 전체 1.98GB로 축소**(4GB 한도 여유있게 통과).
   - fp16 변환 중 커스텀 conformer의 `positional_emb` 파라미터 하나가 저장/로드
     과정에서 fp32로 남는 라이브러리 버그성 동작을 발견 → `dtype mismatch`로
     추론이 크래시남. `baseline_submit/script.py`의 `load_df_arena_model()`에
     "로드 직후 전체 모델을 대표 dtype으로 강제 통일"하는 코드를 추가해 해결
     (fp16/fp32 어느 체크포인트든 안전하게 동작하도록 일반화함).
   - `predict_fake()`에도 입력 텐서를 모델 dtype에 맞춰 캐스팅하는 코드 추가.

## 최종 검증 (2026-09-08 13:09 기준) — 3개 zip 전부 통과
완전히 새 임시 디렉터리에 zip 압축 해제 → `pip install -r requirements.txt` →
`script.py` 실행까지 데이콘과 동일한 절차로 재현, 3개 케이스 모두 정상 출력 확인:
- `submissions/submission_ko.zip` (1.98GB) — 원래 KO 결과와 소수점 3~4자리 오차 내 일치
- `submissions/submission_en.zip` (1.98GB)
- `submissions/submission_koen.zip` (1.98GB)

## 다음
사용자가 3개를 데이콘에 업로드해 리더보드 점수 확인 → 가장 높은 데이터 구성 채택.
필요시 6,000~8,000개로 증량 재실행 검토.

## M3~M6: 미착수

## 재개 시 확인할 것
1. `du -sh data/raw/zeroth_korean/*.tar.gz` 로 다운로드 완료 여부(10GB 도달했는지) 확인
2. `ls data/processed/ko_raw/tts/*.wav | wc -l` 로 VITS 생성 개수 확인 (600 목표)
3. RVC 순환 임포트 문제 해결 여부 — `third_party/rvc/train/preprocess.py` 실행해서 확인
4. HF 토큰은 `.env`에 그대로 있음 (재확인 불필요)

---

# M7. SpecTTTra 음악 파인튜닝 (2026-09-10 시작)

`submit_spectttra`는 지금까지 SpecTTTra 체크포인트(awsaf49/sonics-spectttra-gamma-5s)를
제로샷 그대로 썼음 — 이번엔 SpecTTTra가 원래 학습된 SONICS 데이터셋(awsaf49/sonics,
HF)으로 실제 파인튜닝해서 `submit_spectttra_ft`로 제출 예정.

**세션 환경 참고**: 세션이 끊기면 GPU가 다시 안 잡힐 수 있음(`torch.cuda.is_available()`
확인 필수, 이전에 한 번 GPU 없이 시작된 적 있음 — 사용자에게 GPU 재연결 요청 필요).

## 데이터 소스 결정
- **fake곡**: HF `awsaf49/sonics` 데이터셋의 `fake_songs/part_01.zip` 하나만 사용.
  `metadata.json`의 file_mapping 분석 결과 이 zip 하나에 train 4,465개 + valid 78개가
  이미 들어있어 32GB 전체를 받을 필요 없음(part_01만 3.8GB).
- **real곡**: SONICS는 저작권상 mp3를 안 주고 `real_songs.csv`에 `youtube_id`만 제공 →
  yt-dlp로 직접 다운로드 필요. `skip_time`(인트로 길이) 이후 구간에서 30~40초만 잘라
  받아 용량/시간 절약 (`--download-sections`로 필요한 구간만).
- 목표 규모: fake train 900 + valid 100, real train 900 + valid 100 (총 ~2,000, KO
  파인튜닝 때와 비슷한 규모). yt-dlp 실패(영상 삭제 등) 대비 후보를 20~30% 여유있게 뽑음.

## 체크포인트 설계 (중간에 끊겨도 재개 가능하도록)
- [ ] `data/raw/sonics_fake/fake_songs/part_01.zip` — HF 다운로드 진행 중
      (huggingface_hub 자체가 `.incomplete` 파일로 이어받기 지원하므로 스크립트
      재실행만 하면 됨. 로그: `logs/download_part01.log`)
- [ ] `data_pipeline/sample_music_candidates.py` 실행 산출물:
      `manifests/music_fake_candidates.csv`, `manifests/music_real_candidates.csv`
      (샘플링 결과 고정 — seed 고정이라 재실행해도 동일 리스트 나옴)
- [ ] `data_pipeline/extract_fake_clips.py`: part_01.zip에서 wav 추출 →
      `data/processed/music_raw/fake/*.wav`. **이미 존재하는 파일은 건너뜀** —
      중단 후 재실행하면 이어서 처리.
- [ ] `data_pipeline/download_real_clips.py`: yt-dlp로 real곡 다운로드 →
      `data/processed/music_raw/real/*.wav`. **이미 존재하는 파일은 건너뜀**,
      실패 목록은 `data/processed/music_raw/real_failed.txt`에 append로 기록해
      재실행 시 스킵(무한 재시도 방지).
- [ ] `manifests/music_train_manifest.csv`, `manifests/music_valid_manifest.csv` —
      위 두 폴더에 실제 존재하는 파일만 기준으로 빌드(개수가 목표에 못 미쳐도
      있는 만큼 진행 가능하게 설계).
- [ ] `training/finetune_spectttra.py` — 체크포인트별 `best_val_acc` 기준 저장,
      `--resume`으로 기존 out-dir에서 이어 학습 가능하게 구현 예정.
- [ ] `submit_spectttra_ft/` 패키징 (submit_spectttra 복제 + spectttra 가중치만 교체)

## 재개 시 확인할 것 (M7)
1. `torch.cuda.is_available()` 먼저 확인 — False면 사용자에게 GPU 재연결 요청
2. `ls data/raw/sonics_fake/fake_songs/*.zip` 크기가 3.8GB 도달했는지 (`du -sh`)
3. `wc -l data/processed/music_raw/real_failed.txt` 로 real 다운로드 실패 개수 확인
4. `ls data/processed/music_raw/{real,fake}/*.wav | wc -l` 로 각각 목표(900+100) 대비
   진행률 확인
5. `training/finetune_spectttra.py`가 만드는 out-dir에 체크포인트가 있으면 이어서
   학습할지 새로 할지 판단

## M7 완료 (2026-09-10)
- 데이터: fake 900(train)+78(valid) = SONICS `fake_songs/part_01.zip`에서 추출
  (곡당 35초, 15초 지점부터), real 903(train)+103(valid) = SONICS `real_songs.csv`의
  youtube_id로 yt-dlp 다운로드(35초 클립, 실패 115개는 영상 삭제 등 — 정상 범위) →
  총 train 1,803 + valid 181 = 1,984개
- 학습: SpecTTTra 전체(17.17M 파라미터) 파인튜닝 — DF-Arena와 달리 모델이 작아
  백본 고정 없이 전부 학습. lr=3e-5, batch=16, 6epoch, 원본 학습 레시피의
  MixUp+SpecAugment(모델에 내장된 forward(audio,y))를 그대로 재사용.
  RTX 3090에서 6epoch 총 30초 이내(에폭당 ~4~5초, 40+ it/s) — 예상보다 훨씬 빠름.
- **best val accuracy: 99.45%**(epoch 6, 180/181) — DF-Arena 케이스들과 달리 후반
  epoch 붕괴 없이 계속 98.9%→99.45%로 안정적으로 유지됨(원래 zero-shot 체크포인트가
  이미 같은 데이터 분포로 학습됐던 것이라 그런 것으로 추정).
- 산출물: `data/processed/spectttra_finetuned/`(config.json+pytorch_model.bin) →
  `submit_spectttra_ft/`(submit_spectttra 복제 + spectttra 가중치만 교체) →
  **`submissions/submission_spectttra_ft.zip` (2.04GB, 4GB 한도 여유있게 통과)**
- 최종 검증: 완전히 새 임시 디렉터리에 압축 해제 → `script.py` 실행까지 재현,
  로컬 샘플 3개(모두 동일 파일, md5 확인) 정상 출력 확인. 사용자가 데이콘에
  직접 업로드하면 됨.
- 재사용 스크립트(전부 멱등/재개 가능): `data_pipeline/sample_music_candidates.py`,
  `data_pipeline/extract_fake_clips.py`, `data_pipeline/download_real_clips.py`,
  `data_pipeline/build_music_manifest.py`, `training/finetune_spectttra.py`
  (`--resume`으로 기존 out-dir 이어학습 가능), `inference/build_submission_spectttra.py`

---

# M8. MLAAD 최신 TTS 추가 반영 (2026-09-11)

기존 KO/EN/KOEN fake가 VITS(mms-tts-kor)/RVC/ASVspoof2019 LA(2019년) 위주라
최신 TTS 반영이 안 된다는 사용자 지적으로, HF `mueller91/MLAAD`(gated, 계속
업데이트되는 다국어 스푸핑 데이터셋)에서 2023~2025년 TTS 12종을 KO/EN 각각
추가하고 DF-Arena-1B를 재파인튜닝함.

## 디스크 정리 (선행 작업)
88GB 여유 확보를 위해 이미 추출/사용 완료된 원본만 삭제(총 ~20GB):
`zeroth_korean.tar.gz`(9.7G, flac 22,720개로 추출 완료 확인), ASR용
`zeroth.lm.*.arpa.gz`+`morfessor`+`lexicon`(~7.1G, 파이프라인에서 미사용
grep 확인), `sonics_fake/fake_songs/part_01.zip`(3.6G, wav 978개로 추출 완료
확인) → 88GB → 108GB.

## 데이터 소스: MLAAD (`mueller91/MLAAD`, gated, CC-BY-NC-4.0)
- 사용자가 HF에서 직접 접근 승인 완료
- KO: `fake/ko/` 12개 모델(Chatterbox Multilingual, Fish-S2-Pro, Higgs-Audio-V2,
  MOSS-TTS-1.7B/8B, OmniVoice, Qwen3-TTS-12Hz-0.6B/1.7B, VoxCPM2,
  minimax_speech-02-turbo, XTTS-v2, Bark) 전체 12,000개 샘플링(모델당 1,000개)
- EN: `fake/en/` 140여 개 모델 중 최신 12개(ElevenLabs-v3/Turbo-v2.5/
  v2-Multilingual, Chatterbox/-Turbo, XTTS-v2, minimax_speech-2.6-hd,
  Qwen3-TTS-1.7B, GPT-SoVITS, Higgs-Audio-V2, kokoro, MOSS-TTS-8B) 샘플링
  12,000개(모델당 1,000개)
- 22050Hz 등 원본 SR → 16kHz mono로 변환해 저장, 실패 0건(24,000/24,000 성공)
- 재사용 스크립트: `data_pipeline/sample_mlaad_candidates.py`(후보 목록, HF 파일
  목록 캐시), `data_pipeline/download_mlaad_clips.py`(다운로드+manifest 생성,
  재실행 시 이어받기)

## Manifest 재구성
`data/processed/{ko,en}_mlaad_manifest.csv`를 `data_pipeline/augment.py`로 증강 →
기존 `{ko,en}_augmented_manifest.csv`와 함께 `manifests/build_manifests.py`에
입력해 `manifests_v2/{ko,en,koen}_{train,val}.csv` 생성. real:fake 1:1 밸런싱
로직 특성상 real(ko 1,103/en 1,200)이 병목이 되어 총 개수는 기존과 비슷하지만
(ko/en 각 2,206), fake 후보 풀이 12배 이상 커지면서 실제 뽑힌 fake의 ~92%가
MLAAD(최신 TTS)로 채워짐 (구성 확인: ko_train spoof 993개 중 mlaad 912개).

## 재파인튜닝 결과 (v1 → v2, 기존 스크립트/하이퍼파라미터 그대로)
| 케이스 | v1 best val acc | v2 best val acc | 비고 |
|---|---|---|---|
| KO | 97.3%(epoch2) | 94.55%(epoch2) | epoch3 소폭 하락(93.64%) |
| EN | 95.45%(epoch2) | 88.18%(epoch2) | epoch1 50%(랜덤 수준) 이후 회복, epoch3 다시 붕괴(52.27%) |
| KOEN | 96.36%(epoch1) | 86.36%(epoch1) | epoch2 25.91%로 크게 붕괴, epoch3도 회복 못함(50%) |

**Why**: v2 val acc가 전반적으로 낮아진 건 val set 자체도 MLAAD가 지배적이라
(12개 다른 최신 TTS 엔진을 섞은) 훨씬 어려운/다양한 과제가 됐기 때문 — 기존
숫자(97.3% 등)는 단일 VITS/RVC 스타일에 대한 과적합성 고성능이었을 가능성이
높음. best-checkpoint 저장 로직 덕분에 실제 산출물은 항상 붕괴 전 최고
시점으로 고정됨(기존과 동일 패턴).

## 산출물
`data/processed/df_arena_finetuned_v2/{ko,en,koen}/`(fp16 변환, 2.30GB) →
`submissions/submission_{ko,en,koen}_v2.zip`(각 1.98GB, 기존 v1 zip은 보존됨) —
완전히 새 임시 디렉터리에 압축 해제 → `script.py` 실행까지 3개 케이스 모두
정상 출력 확인(5개 컬럼, NaN/dtype 에러 없음).

## 다음
사용자가 v1과 v2 중 어느 쪽을 데이콘에 제출할지 결정 필요 — 로컬 val acc는
v1이 높지만 v1은 좁은 fake 분포에 과적합됐을 가능성이 있어 실제 리더보드
(미공개 최신 생성기법 포함 가능)에서는 v2가 더 잘 일반화할 수도 있음. 실제
업로드해서 비교하는 게 유일한 확실한 검증 방법.

---

# M9. Voice+Music 파인튜닝 통합 제출본 + EER 기반 검증 (2026-09-11)

지금까지 voice 파인튜닝(DF-Arena v1/v2)과 music 파인튜닝(SpecTTTra ft)이 서로
다른 제출 zip에 나뉘어 있었음(`submission_koen_v2.zip`은 voice만 파인튜닝,
music은 DF-Arena 제로샷; `submission_spectttra_ft.zip`은 music만 파인튜닝,
voice는 DF-Arena 제로샷) — 리더보드 산식(0.9*ADS+0.1*CPS, ADS=File 0.5+Voice
0.2+Music 0.3)에서 File EER이 voice/music 중 더 안 좋은 쪽에 끌려가므로, 두
파인튜닝을 하나로 합치는 게 가장 저비용 고효율 개선이라 판단해 즉시 실행.

## 통합 제출본
- `inference/build_submission_final.py`: `submit_spectttra_ft/`(SpecTTTra 파인튜닝
  파이프라인 코드 기준)를 베이스로 df_arena_1b 가중치만 v2(koen, MLAAD 반영)로
  교체.
- 산출물: `submit_final_koen_v2/` → **`submissions/submission_final_koen_v2.zip`
  (2.04GB, 해제 시 2.6GB)** — voice(DF-Arena v2)와 music(SpecTTTra ft) 둘 다
  파인튜닝된 버전이 처음으로 하나의 제출본에 합쳐짐.
- `model/MODEL_INFO.txt`, `model/SHA256SUMS.txt`를 실제 구성(v2 데이터 출처
  전체, SpecTTTra 파인튜닝 정보)에 맞게 재작성 — 대회 규정상 출처 명시 의무 대응.
- 검증: 완전히 새 임시 디렉터리에 압축 해제 → `script.py --device cuda` 로컬
  샘플 3개 정상 실행 확인(5개 확률 컬럼, NaN 없음). 3개 로컬 샘플은 전부 동일
  파일(md5 일치, 기존에 확인된 사실)이라 출력이 동일한 것은 정상.
- 타이밍: 3파일 기준 3.313s/file은 최초 1회성 CUDA 워밍업 비용이 크게 섞인
  수치이고, 스테이지별 정상 상태 비용(HTDemucs 0.337 + PANNs 0.061 + DF-Arena
  0.077 + SpecTTTra 0.012 ≈ 0.49s/file)로 추정하면 1,200개 기준 약 10분 —
  기존에 이미 검증된 `submission_spectttra_ft.zip`과 모델 구성이 동일해
  60분 제한 대비 여유 충분.

## EER 기반 v1 vs v2 vs soup 비교 (그동안 val acc만 봤던 것을 EER로 재검증)
`training/finetune_df_arena.py`의 체크포인트 선택 기준이 지금까지 val
accuracy였는데, 실제 대회 지표는 EER이라 이번에 처음으로 EER을 직접 측정함
(스크립트: 스크래치패드 `eval_df_arena.py`, 재사용하려면 다시 작성 필요 —
세션 한정 저장소라 휘발됨). `manifests_v2/koen_val.csv`(220개, MLAAD 포함 —
v2 학습 때 쓴 val set과 동일해 v2에 유리한 비교라는 한계 있음) 기준:

| 모델 | val acc | EER |
|---|---|---|
| v1 (VITS/RVC만) | 85.00% | 13.18% |
| v2 (MLAAD 반영) | 87.27% | **6.36%** |
| soup (v1+v2 파라미터 평균) | 87.27% | 9.09% |

**Why**: v2가 EER 기준으로도 v1보다 명확히 우수(13.18%→6.36%, 절반 이하)해,
기존에 "v1이 val acc는 더 높지만 좁은 분포 과적합일 수 있다"고 추정만 했던
것이 EER로 확인됨. Model soup(같은 프로즌 backbone에서 갈라져 나온 두 헤드의
파라미터 평균, 별도 학습 없이 즉시 시도 가능한 무료 앙상블 기법)은 이번엔
v1의 약점을 v2에 섞어 오히려 v2 단독보다 악화시켰음 — 채택하지 않음. 최종
결론: **v2를 그대로 voice 모델로 유지, 앙상블/souping 불필요**.

**How to apply**: 다음에 파인튜닝을 다시 돌릴 기회가 있으면
`training/finetune_df_arena.py`의 best-checkpoint 선택 기준을 val acc 대신
EER로 바꾸는 게 실제 리더보드 지표와 더 일치함(현재는 acc 기준이라 최적이 아닐
수 있음, 코드 미변경 상태). 관련: [[project_finetune_experiment_1]].

## 다음
- `submissions/submission_final_koen_v2.zip`이 현재 시점 최선 제출 후보 —
  사용자가 데이콘에 업로드해 실제 리더보드 점수 확인 필요(이게 최종 검증).
  기존 `submission_koen_v2.zip`(voice만), `submission_spectttra_ft.zip`(music만)
  대비 File/Voice/Music EER 전부 개선될 것으로 기대.
- 남은 개선 여지(우선순위 낮음, 미착수): PANNs 존재판별(CPS, 가중치 0.1로
  낮음)은 아직 제로샷 그대로 — 파인튜닝 시도 안 함. EER 기준 체크포인트 선택
  로직 반영한 재학습.

---

# M10. 케이스별 성능 분석 + 약점 기반 데이터 증강 파이프라인 (2026-09-13)

기존 학습(1차)은 9개 case_type(A~I, 딥보이스 보이스피싱 A~E + 음악 F~I)이
섞인 데이터로 진행됐음 — 검증셋을 케이스별로 쪼개 평가하고, 약한 케이스만
골라 추가 데이터를 확보/재학습하는 반복 파이프라인을 신규 추가.
`training/`, `submit_final_koen_v2/script.py` 등 **기존 스크립트는 전혀
수정하지 않고**, 새 폴더 `case_pipeline/`(3개 모듈)를 추가 실행하는 방식으로
통합함. 자세한 사용법은 `case_pipeline/README.md` 참고.

- `case_pipeline/case_analysis.py`: 메타데이터 스키마(`load_metadata`/
  `save_metadata`, `.jsonl`/`.json` 지원) + `evaluate_by_case`(A~E, F/H
  voice·music 관점, G는 EER/F1/Accuracy — I_bgm_evasion은 fake 판별이 아니라
  presence_accuracy/presence_auc로 별도 평가) + `identify_weak_cases`(하위
  N% 자동 식별, lower_is_better로 EER↔F1/Acc 둘 다 처리) + matplotlib 시각화
  + `append_experiment_log`. 내부적으로 `submit_final_koen_v2/script.py`의
  HTDemucs/PANNs/DF-Arena/SpecTTTra 로딩·추론 함수를 importlib로 그대로
  재사용하는 기본 어댑터(`load_default_pipeline_models`) 제공.
- `case_pipeline/augmentation_planner.py`: `build_augmentation_plan`(약점
  케이스별 현재/목표(기본 +20%)/부족분 + 추천 데이터셋을 함께
  `augmentation_plan.json`에 저장).
- `case_pipeline/retrain_pipeline.py`: `retrain_with_augmented_data`(voice
  갈래는 DF-Arena만, music 갈래는 SpecTTTra만 갱신되도록 분리) +
  `finetune_presence_head`(I_bgm_evasion 전용, PANNs `fc_audioset` 헤드만
  재학습하는 별도 루틴) + `build_case_weighted_sampler`(WeightedRandomSampler
  옵션) + `experiment_log.csv` 누적 기록.
- 검증: mock 모델(HTDemucs/PANNs/DF-Arena/SpecTTTra 대역폭 없이 duck-typing만
  맞춘 객체)로 evaluate_by_case → identify_weak_cases → build_augmentation_plan
  → append_experiment_log 전체 흐름 스모크 테스트 통과.

## 다음 (M10)
- 실제 9케이스 메타데이터(`.jsonl`)를 채워 `case_pipeline/case_analysis.py`를
  실제 검증셋에 돌려 케이스별 EER을 확인해야 함(지금까지는 koen 전체 합산
  EER만 봤음 — 이번에 처음으로 케이스 단위 breakdown 가능해짐).
- 약점 케이스가 나오면 `augmentation_planner.py`로 계획을 뽑고, 추천
  데이터셋(RECOMMENDED_DATASETS에 9케이스 전부 매핑됨)에서 실제로 데이터를
  받아와 `retrain_pipeline.py`로 반복.

## M10-1단계 완료: 기존 manifest -> 9케이스 메타데이터 매핑 (2026-09-13)

`case_pipeline/build_case_metadata.py` 추가: `manifests_v2/{ko,en}_{train,val}.csv`
(technique/source/augmentation 컬럼 기준)와 `manifests/music_{train,valid}_
manifest.csv`(label 컬럼 기준)를 case_analysis 스키마로 변환해
`manifests/case_metadata_{train,val}.jsonl`로 저장. 매핑 규칙은 스크립트
상단 docstring에 전부 문서화(조정 필요시 거기만 고치면 됨):
- technique=="real" -> A/B/C/D 네 케이스 모두의 negative(real) 클래스로 재사용
- technique=="tts" -> B_institution_ars
- technique=="vc" & source가 "rvc_from_"로 시작 -> A_impersonation(특정 화자
  1명을 노린 클로닝), 그 외(ASVspoof2019 LA A05/A06) -> C_realtime_vc (잠정
  배정 — 실제 실시간 VC 전용 데이터 확보 전까지의 임시 분류)
- augmentation에 "codec" 포함 -> E_phone_channel 추가(real/fake 둘 다, A~D
  위에 얹히는 다중 라벨)
- music label=="fake"(SONICS) -> G_full_ai_music, label=="real"(유튜브) ->
  G의 negative 클래스

**변환 결과 (train n=5,775 / val n=621)**: A 2,029/223, B 3,894/432,
C 2,021/225, D 1,986/220(전부 real만 — D 전용 fake 0개), E 1,199/126,
G 1,803/181, **F/H/I는 0개 — 아직 매핑 가능한 데이터가 전혀 없어 별도
확보 필요**(F_ai_cover_song, H_hybrid_composed, I_bgm_evasion).

**실제 파이프라인 통합 검증**: mock이 아니라 실제 DF-Arena-1B/SpecTTTra/
PANNs/HTDemucs 4개 모델을 `load_default_pipeline_models()`로 로드해 val
subset(11개 샘플)에 `evaluate_by_case`를 실제로 돌려 정상 동작 확인(모델
로딩 9.0s, 11개 평가 7.2s, GPU=RTX 3090). A/C/D는 표본이 너무 작아
클래스가 한쪽만 나와 EER NaN(정상 — 경고 로직대로 동작), B/E/G는 EER 0.0
산출(표본 4~7개라 수치 자체는 아직 의미 없음, 코드 경로 검증용).

## M10-2단계 완료: val 전체(621개) 케이스별 첫 평가 (2026-09-13)

`case_pipeline/case_analysis.py --metadata manifests/case_metadata_val.jsonl`
실행(RTX 3090, 621개 5분 43초 = 0.55s/file). `case_performance.csv`/
`case_performance.png` 생성됨.

| case_type | n | EER | F1 | Accuracy |
|---|---|---|---|---|
| A_impersonation | 223 | 0.2803 | 0.1026 | 0.8430 |
| B_institution_ars | 432 | 0.0833 | 0.8969 | 0.8935 |
| C_realtime_vc | 225 | 0.0432 | 0.2273 | 0.8489 |
| D_partial_splice | 220 | NaN(계산불가) | - | - |
| E_phone_channel | 126 | 0.1189 | 0.8852 | 0.8889 |
| G_full_ai_music | 181 | 0.0274 | 0.9388 | 0.9503 |

`identify_weak_cases(metric_col="EER", percentile=25)` -> **약점 케이스:
E_phone_channel, A_impersonation**.

**중요한 주의점**: A_impersonation/C_realtime_vc의 val fake 표본이 각각
3개/5개뿐(real 220개에 얹혀 계산됨 — M10-1단계에서 이미 예견됨). A의
F1=0.10(Accuracy는 0.84로 높아 보이지만 fake 표본이 극소수라 F1이 그
불균형을 그대로 드러냄)은 통계적으로 노이즈에 가까움 — "DF-Arena가 A를
못 맞춘다"기보다 "A를 제대로 평가할 표본 자체가 없다"에 가까운 신호로
해석해야 함. D는 fake 표본이 0개라 EER 계산 자체가 불가(NaN, 정상 동작).
즉 지금 시점의 진짜 결론은: **A/C/D/F/H/I 전부 "성능이 나쁘다"가 아니라
"평가할 데이터 자체가 없거나 턱없이 부족하다"** — 데이터 확보가 최우선.

## M10-3단계: 데이터 추가 확보 (2026-09-13, "데이터 더 모으자")

### 3-1. "새 데이터"가 아니라 이미 있던 미사용 데이터 발견 (A/C/E, 다운로드 0)
`data/processed/{ko,en}_augmented_manifest.csv`를 다시 보니 vc(변환) 기술
샘플이 케이스당 600개(총 1,200개)나 이미 있었는데, `manifests_v2` 빌드 때
real:fake 1:1 밸런싱 로직이 MLAAD(12,000개 tts)에 압도돼 **1,200개 중 86개만
학습/평가에 쓰이고 1,114개가 그냥 방치돼 있었음**. `build_case_metadata.py
--expand-vc-pool` 옵션을 추가해 이 미사용분을 전부 val에 편입(real은 이미
있는 걸 재사용, 새로 추가 안 함 — real 풀은 2,206/2,303개가 이미 소진돼
97개밖에 안 남아 늘릴 여지가 거의 없음).

**효과 (val, n 3~5개짜리 노이즈 수치 → 실제 신뢰 가능한 수치로)**:
| case_type | 이전 n / EER | 이후 n / EER |
|---|---|---|
| A_impersonation | 223 / 0.2803 | **777 / 0.0732** |
| C_realtime_vc | 225 / 0.0432(fake 5개뿐, 노이즈) | **785 / 0.0874** |
| E_phone_channel | 126 / 0.1189 | **476 / 0.0948** |

A/C는 "성능이 나쁜 게" 아니라 "표본이 없어서 EER이 요동친 것"이었다는 게
확인됨 — 실제로는 B/G와 비슷한 수준(EER 7~9%대)으로 잘 맞추고 있었음.

### 3-2. D/H/I를 자체 합성으로 신규 제작 (다운로드 0, `data_pipeline/` 3개 스크립트 추가)
PartialSpoof/HAD, SingFake/CtrSVDD, FakeMusicCaps 등 실제 데이터셋을 받기
전 임시 대체재로, 이미 가진 real/fake 음성·음악 자산을 조합해 세 케이스를
직접 만들었다(`data_pipeline/audio_mix_utils.py` 공용 유틸 + 아래 3개):

- `build_partial_splice_data.py` (D): real 음성(carrier) 중간 구간을
  TTS/VC fake 조각으로 치환(크로스페이드 포함). 400개 생성(train 340/val 60).
- `build_bgm_evasion_data.py` (I): real 음성 + real/fake 음악을 SNR
  -6~12dB로 믹싱(mixed, has_voice=True) + 기존 음악 파일 그대로 재사용
  (music_only, has_voice=False, 합성 불필요). 각 300개, 총 600개
  (train 510/val 90).
- `build_hybrid_composed_data.py` (H): SONICS AI 완전생성곡을 HTDemucs로
  분리해 반주만 취하고, 거기에 TTS/VC fake 보컬을 SNR -2~8dB로 새로 믹싱
  (원본 SONICS 보컬은 버림 — "AI 반주 + 별도 합성 보컬"이라는 H의 정의를
  재현). 200개 생성(train 170/val 30). HTDemucs 추론이 필요해 GPU 사용.

세 스크립트 모두 `merge_into_case_metadata()`로 `case_metadata_{train,val}
.jsonl`에 직접 병합(재실행 시 `source_dataset` 태그로 이전 산출물을 걷어내고
새로 채워 넣어 멱등적). **F_ai_cover_song만 여전히 0개** — 우리가 가진
음성 자산이 전부 "말하기(TTS/VC)"라 노래(가창)에 쓸 원천이 없어서 자체
합성이 불가능함(실제 SingFake/CtrSVDD 다운로드가 필요).

### 3-3. 최종(v3) 케이스별 재평가
D/H/I가 반영된 `case_metadata_val.jsonl`(1,915개)로 `case_performance_v3.csv`
재생성 — 결과는 다음 항목 참고.

### 3-4. H_hybrid_composed real 쪽 누락 버그 수정 + 최종(v-final) 재평가
v3에서 H_hybrid_composed의 voice/music 양쪽 다 EER=NaN이 나온 원인을 확인:
`build_hybrid_composed_data.py`가 fake 샘플만 만들고 real 대응 표본을 전혀
안 만들어서 단일 클래스였음. `build_case_metadata.py`의 real 분류 로직에
H_hybrid_composed를 추가(REAL_VOICE_CASE_TYPES, real 음악에도 태깅)해서
기존 real 음성/음악을 H의 negative로 재사용하도록 수정 → 전체 재빌드 후
최종 재평가(`case_performance_final.csv`, val 1,915개).

**최종 케이스별 성능 (val)**:
| case_type | n | EER | F1 | Accuracy |
|---|---|---|---|---|
| A_impersonation | 777 | 0.073 | 0.958 | 0.938 |
| B_institution_ars | 432 | 0.083 | 0.897 | 0.894 |
| C_realtime_vc | 785 | 0.087 | 0.949 | 0.926 |
| **D_partial_splice** | 280 | **0.302** | 0.531 | 0.786 |
| E_phone_channel | 476 | 0.095 | 0.940 | 0.901 |
| G_full_ai_music | 181 | 0.027 | 0.939 | 0.950 |
| H_hybrid_composed(voice) | 250 | 0.052 | 0.638 | 0.864 |
| **H_hybrid_composed(music)** | 133 | **0.296** | 0.500 | 0.850 |
| F_ai_cover_song | - | 데이터 없음(자체 합성 불가) | - | - |

| I_bgm_evasion | n | presence_accuracy | presence_auc |
|---|---|---|---|
| voice 존재판별 | 90 | **0.489(거의 랜덤)** | 0.604 |
| music 존재판별 | 90 | 0.933 | - |

`identify_weak_cases(EER, 하위25%)` -> **D_partial_splice, H_hybrid_composed
(music)**. 여기에 presence_accuracy가 사실상 랜덤 수준인 **I_bgm_evasion(voice)**
도 별도 지표라 자동 필터엔 안 걸렸지만 수동으로 약점에 포함.

**해석**:
- D(짜깁기)는 DF-Arena가 세그먼트 단위 최대값으로 fake를 판정하는 구조상,
  발화 일부만 위조되면 그 구간이 세그먼트에 잘 걸리지 않거나 희석돼 놓치는
  것으로 추정 — "부분위조 탐지"라는 게 실제로 어려운 문제라는 게 확인됨.
- H(music 관점)는 SpecTTTra가 "AI로 만든 반주"를 HTDemucs로 분리·재믹싱한
  뒤에는 원래(G) 만큼 잘 못 잡음 — G(온전한 SONICS 완전생성곡, EER 0.027)와
  비교하면 HTDemucs 분리 자체가 판별에 쓰이는 미세한 생성 흔적을 지우거나
  바꿔버리는 것으로 추정.
- I(voice presence)는 원래 문제의식(배경음악 위장 시 존재판별 강건성)이
  실제로 맞았음을 확인 — 지금 제로샷 PANNs는 bgm이 섞이면 사실상 랜덤
  수준(AUC 0.60은 약한 신호는 있으나 임계값 0.5가 안 맞음).

### 3-5. 증강 계획 (`augmentation_planner.py`)
```
case_type                  current  target  need  recommended_dataset
H_hybrid_composed__music     3059    3671   612   FakeMusicCaps+MLAAD/ASVspoof 직접 믹싱
D_partial_splice             2326    2792   466   PartialSpoof, HAD
I_bgm_evasion                  510     612   102   real 음성에 배경음악을 인위적으로 믹싱해 자체 제작
```
(`augmentation_plan.json` 저장됨)

### 3-6. D/H/I 데이터 3배 증량(D 400→1200, I 600→1000, H 200→500) 후 재평가(v5)
"평가 데이터만 늘리면 되나?"를 검증하기 위해 같은 제로샷 모델로 재평가만
해봄 — **D/H(music)/I 전부 거의 개선 안 됨**(D EER 0.302→0.297, H(music)
0.296→0.286, I 정확도 0.489→0.533). **결론: 지금 배포된(제로샷) 모델은
바뀌지 않았으니 평가 데이터를 늘려봤자 측정 신뢰도만 오를 뿐 실제 탐지
성능은 그대로 — 진짜 개선하려면 이 데이터로 재학습해야 한다.** 이 결론이
다음 단계(재학습)로 이어짐.

## M10-4단계: 재학습(iteration 2) — retrain_pipeline.py 실전 투입 + 버그 3개 발견/수정

`retrain_pipeline.py`를 처음으로 실제 augmented 데이터(train 8,070개)에
돌려봄. 스켈레톤 단계에서는 안 드러났던 실제 버그 3개를 순서대로 발견/수정:

1. **SpecTTTra 타겟 텐서 shape**: `(1,1)`로 넘겼는데 모델의 MixUp 레이어가
   `(batch,)` 1차원을 요구 — `ValueError: Target ndim should be 1. Got 2`로
   music 컴포넌트 학습 시작 직후 죽음. `target = torch.tensor([...])`(1차원)로 수정.
2. **best-checkpoint 저장 로직 누락**: 원래 코드는 마지막 epoch을 무조건
   저장했는데, 실제로 voice(DF-Arena) 학습이 epoch1(val_acc 0.897) →
   epoch2(0.893) → epoch3(0.878)로 붕괴하는 걸 실측(PROGRESS.md M5~M9에
   이미 기록된 known issue였는데 스켈레톤엔 반영 안 돼 있었음). `train_voice
   _component`/`train_music_component`에 매 epoch 검증 + 최고 시점만 저장
   하는 로직 추가(`training/finetune_df_arena.py`와 동일한 패턴). 이후 재실행
   에서 정상 작동 확인(voice epoch2 val_acc 0.9243이 최종 채택, music epoch6
   val_acc 0.9883 채택).
3. **PANNs 래퍼/실제 모델 혼동**: `panns_model.model`이 `panns_inference.
   AudioTagging` 래퍼 객체인데 그 자체를 `named_parameters()`/`state_dict()`
   대상으로 착각 — 학습 직후, 그리고 저장 직후 두 번에 걸�，`AttributeError`로
   두 번 죽음. 실제 학습 가능한 Cnn14는 `panns_model.model.model`로 접근해야
   함 — 두 지점 모두 수정.

GPU가 중간에 한 번 끊겼다가(`nvidia-smi: Failed to initialize NVML`, 이전에
M7에서도 겪은 문제) 재연결 후 재개 — voice/music은 이미 학습·저장이 끝나
있어서(디스크에 체크포인트 존재) 그 두 개는 다시 안 돌리고 PANNs 헤드
재학습 + 최종 평가만 이어서 완료.

### iteration 1(제로샷) vs iteration 2(재학습) — 최종 비교
`experiment_log.csv`에 두 iteration 모두 기록됨.

| case_type | iter1 EER | iter2 EER | 비고 |
|---|---|---|---|
| A_impersonation | 0.073 | **0.058** | 개선 |
| B_institution_ars | 0.083 | **0.060** | 개선 |
| C_realtime_vc | 0.087 | **0.059** | 개선 |
| D_partial_splice | 0.297 | **0.045** | **대폭 개선(약점 해소)** |
| E_phone_channel | 0.095 | **0.089** | 소폭 개선 |
| G_full_ai_music | 0.027 | **0.005** | 개선 |
| H_hybrid_composed(voice) | 0.074 | 0.052 | 개선 |
| H_hybrid_composed(music) | 0.286 | **0.158** | **대폭 개선(약점 크게 해소)** |

| I_bgm_evasion | iter1 | iter2 | 비고 |
|---|---|---|---|
| voice presence_accuracy | 0.533 | **0.420** | **악화** |
| voice presence_auc | 0.632 | **0.332**(0.5 미만 = 역상관) | **명백히 악화** |
| music presence_accuracy | 0.933 | 0.973 | 개선(하지만 아래 원인 때문에 무의미) |

**전반적으로 대성공** — 원래 약점이던 D/H(music)이 재학습으로 실제 해소됨
(다른 케이스들도 대부분 동반 개선). 다만 **I_bgm_evasion의 PANNs 헤드
재학습만 명백히 역효과**가 남.

### I_bgm_evasion이 악화된 원인 (데이터 라벨링 버그, 발견)
`build_bgm_evasion_data.py`의 "bgm만 있고 음성 없음"(has_voice=False)
negative 클래스가 **기존 음악 파일(SONICS AI곡 + 유튜브 실제곡)을 그대로
재사용**한 것이었는데, PANNs의 `component_labels.json`을 보면 "voice"
그룹에 `Singing, Choir, Male/Female/Child singing, Rapping, Humming, Chant`
등 **가창까지 전부 포함**돼 있음. 즉 재사용한 "무음성" 샘플들이 실제로는
노래(보컬)가 들어있는 완전한 곡이라 **PANNs 기준으로는 오히려 has_voice=True
가 맞는 경우가 많아 라벨이 뒤집혀 있었던 것** — 이 잘못된 라벨로 3epoch
헤드를 학습시키니 voice_indices 쪽 캘리브레이션이 왜곡돼 AUC가 0.33(랜덤
0.5보다 나쁨, 즉 역상관)까지 떨어짐. music 쪽 정확도가 오히려 오른 것도
같은 이유(모든 샘플이 사실상 has_music=True라 "항상 True" 방향으로만
치우쳐도 우연히 맞음 — 의미 있는 학습이 아님).

**수정 방향**: "무음성" negative는 원곡을 그대로 쓰지 말고, HTDemucs로
분리한 **반주(instrumental) 스텀만** 쓰도록 `build_bgm_evasion_data.py`의
`build_music_only_records()`를 고쳐야 함(이미 만들어둔 `HTDemucsSeparator`
재사용 가능, `build_hybrid_composed_data.py`가 반주 추출에 쓰는 것과 같은
패턴). 아직 미수정 — 다음 세션 최우선 작업.

## M10-5단계: I_bgm_evasion 라벨 버그 수정 + iteration 3 (2026-09-14)

`data_pipeline/build_bgm_evasion_data.py`의 `build_music_only_records()`를
원곡 그대로 쓰던 것에서 **HTDemucs로 분리한 반주(vocal 제거) 스텀만** 쓰도록
재작성(`build_hybrid_composed_data.py`와 동일 패턴, `HTDemucsSeparator`
재사용). mixed/music_only를 독립적으로 재생성할 수 있도록 source_dataset
태그도 분리(`synthetic_bgm_evasion_mixed`/`_musiconly`) — 기존 레거시 태그
(`synthetic_bgm_evasion`) 850개는 일괄 제거 후 500+500으로 재생성.

voice(DF-Arena)/music(SpecTTTra)는 iteration 2 체크포인트를 그대로 재사용
(안 건드림 — 이미 좋은 상태). PANNs 헤드만 고친 데이터로 다시 재학습 후
val 전체 재평가(iteration 3).

### 최종 3-iteration 비교 (`experiment_log.csv`에 전부 기록됨)
| case_type | iter1(제로샷) | iter2(1차 재학습, I버그있음) | iter3(I버그 수정) |
|---|---|---|---|
| A_impersonation EER | 0.073 | 0.058 | 0.058 (동일, 안 건드림) |
| B_institution_ars EER | 0.083 | 0.060 | 0.060 |
| C_realtime_vc EER | 0.087 | 0.059 | 0.059 |
| D_partial_splice EER | 0.297 | **0.045** | 0.045 |
| E_phone_channel EER | 0.095 | 0.089 | 0.089 |
| G_full_ai_music EER | 0.027 | **0.005** | 0.005 |
| H_hybrid_composed(voice) EER | 0.074 | 0.052 | 0.052 |
| H_hybrid_composed(music) EER | 0.286 | **0.158** | 0.158 |
| I voice presence_accuracy | 0.533 | 0.420(악화) | **0.667** |
| I voice presence_auc | 0.632 | 0.332(역상관) | **0.869** |
| I music presence_accuracy | 0.933 | 0.973(의미없는 상승) | 0.967 |

**iteration 3에서 9개 케이스 전부 iteration 1(제로샷) 대비 개선 확인**
(A~H는 iteration 2에서 이미 개선된 상태 그대로 유지, I는 라벨 버그
수정으로 iteration 1보다도 크게 개선: AUC 0.632→0.869). D_partial_splice와
H_hybrid_composed(music) — 원래 identify_weak_cases가 찍었던 두 약점 —
가 가장 극적으로 개선됨(EER 각각 0.297→0.045, 0.286→0.158).

### 최종 산출물
- voice: `data/processed/df_arena_case_pipeline_v1/` (best epoch2, val_acc 0.9243)
- music: `data/processed/spectttra_case_pipeline_v1/` (best epoch6, val_acc 0.9883)
- presence: `data/processed/panns_case_pipeline_v2/Cnn14_finetuned.pth`
  (iteration 3, 라벨 버그 수정된 데이터로 학습된 버전 — v1은 버려짐)
- `case_performance_iter3.csv`, `experiment_log.csv`(iteration 1~3 전체 이력)

## 다음
1. `submissions/`에 있는 기존 제출본들은 아직 이번 재학습 결과로 안 바뀜 —
   실제 데이콘 제출용으로 채택하려면 `inference/build_submission_final.py`
   패턴으로 voice=df_arena_case_pipeline_v1, music=spectttra_case_pipeline_v1,
   presence=panns_case_pipeline_v2 세 가중치를 넣어 새 zip 패키징 필요
   (사용자 결정 필요 — 로컬 val 개선이 실제 리더보드로 이어지는지는 업로드
   해봐야 확실함, 과거 v1/v2 EER 사례처럼).
2. F_ai_cover_song은 유일하게 자체 합성이 불가능한 케이스(보유 음성 자산이
   전부 "말하기"라 가창 원천이 없음) — SingFake/CtrSVDD 실제 다운로드 여부를
   사용자와 상의 필요(HF 접근 승인이 필요할 수 있음, 이전 MLAAD 때처럼).
3. E_phone_channel은 세 iteration 내내 개선폭이 가장 작음(0.095→0.089) —
   다음 반복에서 볼 만한 후보.
