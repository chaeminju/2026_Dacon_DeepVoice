#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다.

[파이프라인 구조]
원본 오디오
  → HTDemucs로 voice_audio/music_audio 분리 (파일당 1회만 실행)
  → voice_audio → PANNs → VOICE_PRESENT_PROB
  → music_audio → PANNs → MUSIC_PRESENT_PROB
  → voice_audio → AntiDeepfake MMS-1B → VOICE_FAKE_PROB
  → music_audio → SpecTTTra → MUSIC_FAKE_PROB
  → combine_file_fake_score()로 FILE_FAKE_PROB 계산

HTDemucs·PANNs·DF-Arena 1B·SpecTTTra 네 모델은 전체 파일 루프 시작 전
딱 한 번만 로딩하고, 파일별로 한 바퀴만 도는 단일 루프 안에서 재사용한다.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 추론에는 model 폴더에 포함된 로컬 파일만 사용한다.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from tqdm import tqdm


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "antideepfake_mms1b"  # AntiDeepfake MMS-1B (DF-Arena 1B 대체)
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"
SPECTTTRA_DIR = MODEL_DIR / "spectttra"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600  # DF-Arena 1B 세그먼트 길이 (약 4.04s @16kHz)
SILENCE_RMS = 1e-5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


# -----------------------------------------------------------------------------
# 1. 입력 파일 및 제출 양식 확인
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the zero-shot audio deepfake baseline (SpecTTTra variant)."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "파라미터 수가 가장 큰 DF-Arena 1B를 평소 CPU에 두고, 실제로 "
            "호출하는 순간에만 device로 옮겼다가 즉시 CPU로 되돌린다(PANNs/ "
            "HTDemucs/SpecTTTra는 용량이 작아 항상 device에 상주). GPU 메모리가 "
            "부족한 환경(예: GPU RAM 8GB 이하)에서 사용한다. 기본값은 네 모델 "
            "모두 device에 동시 상주(더 빠름)."
        ),
    )
    return parser.parse_args()


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]


# -----------------------------------------------------------------------------
# 2. 오디오 구간 분할
# -----------------------------------------------------------------------------
# segment_samples를 인자로 받아 DF-Arena(4.04s)와 SpecTTTra(체크포인트별 길이,
# 기본 5s)처럼 모델마다 세그먼트 길이가 달라도 동일한 함수를 공유해서 쓴다.

def get_segment_starts(audio_length, segment_samples=SEGMENT_SAMPLES):
    if audio_length <= segment_samples:
        return [0]

    last_start = audio_length - segment_samples
    starts = list(range(0, last_start + 1, segment_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start, segment_samples=SEGMENT_SAMPLES):
    if audio.size < segment_samples:
        repeat_count = segment_samples // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:segment_samples].astype(np.float32)

    end = start + segment_samples
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. PANNs를 이용한 음성·음악 존재 여부 추론
# -----------------------------------------------------------------------------
# predict_presence()는 voice_audio 또는 music_audio 각각을 독립적으로 입력받아
# 단일 확률(voice_prob 또는 music_prob)을 반환한다. 원본 오디오 전체가 아니라
# HTDemucs로 이미 분리된 성분을 입력으로 쓴다.

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )

    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence(model, indices, audio):
    """voice_audio 또는 music_audio 하나를 입력받아 해당 성분의 존재 확률을
    반환한다. voice_audio를 넣으면 voice_indices와 함께 VOICE_PRESENT_PROB를,
    music_audio를 넣으면 music_indices와 함께 MUSIC_PRESENT_PROB를 계산한다."""
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    return float(predictions[:, indices].max())


# -----------------------------------------------------------------------------
# 4. HTDemucs를 이용한 음성·음악 분리
# -----------------------------------------------------------------------------

def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        # PyTorch 2.6부터 바뀐 기본값에 맞춰 기존 체크포인트를 불러온다.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# 5. AntiDeepfake MMS-1B를 이용한 음성 성분 Fake 추론 (기존 로직 유지)
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
    """DF-Arena 1B 대신 AntiDeepfake MMS-1B(fp32)를 로드한다. 함수명/반환 형식은
    기존 파이프라인과 맞추기 위해 유지한다."""
    if str(DF_ARENA_DIR) not in sys.path:
        sys.path.insert(0, str(DF_ARENA_DIR))
    from detector import AntiDeepfakeDetector, FAKE_LABEL_INDEX

    model = AntiDeepfakeDetector(DF_ARENA_DIR).to(device=device).eval()
    return model, FAKE_LABEL_INDEX


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index, audio, device):
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        model_dtype = next(model.parameters()).dtype
        segment_tensor = torch.from_numpy(segment).to(device=device, dtype=model_dtype)

        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        segment_scores.append(float(probabilities[0, fake_label_index]))

    return max(segment_scores)


# -----------------------------------------------------------------------------
# 6. SpecTTTra를 이용한 음악 성분 Fake 추론 (신규)
# -----------------------------------------------------------------------------
# awsaf49/sonics-spectttra 계열 체크포인트(예: sonics-spectttra-gamma-5s)를
# model/spectttra/ 폴더에 로컬로 저장해두고 오프라인으로 로드한다. 구성:
#   model/spectttra/config.json          - 체크포인트 설정
#   model/spectttra/pytorch_model.bin    - 체크포인트 가중치
#   model/spectttra/sonics/              - 추론에 필요한 sonics 패키지 소스 벤더링
#                                           (pip install git+https://github.com/
#                                           awsaf49/sonics.git 로 설치되는 것과 동일한
#                                           패키지를 오프라인 환경을 위해 로컬 복사)

def load_spectttra_model(device):
    """DF-Arena 1B 로딩 함수와 동일한 패턴: model/spectttra/ 하위의 로컬 파일만
    사용해 오프라인으로 로드한다. HFAudioClassifier.from_pretrained()는 인자로
    받은 경로가 로컬 디렉터리면 허브에 접속하지 않고 그 안의 config.json/
    pytorch_model.bin을 바로 읽어들인다."""
    if str(SPECTTTRA_DIR) not in sys.path:
        sys.path.insert(0, str(SPECTTTRA_DIR))
    from sonics import HFAudioClassifier

    model = HFAudioClassifier.from_pretrained(str(SPECTTTRA_DIR))
    model = model.to(device=device).eval()

    # DF-Arena와 입력 형식(샘플레이트/세그먼트 길이/정규화 방식)이 다를 수 있으므로
    # 체크포인트의 config.json에서 직접 읽어와 preprocess_for_spectttra()에 넘긴다.
    # (체크포인트를 alpha-5s/beta-5s/120s 계열 등으로 교체해도 코드 수정 없이 동작)
    spectttra_config = {
        "segment_samples": int(model.config.audio.max_len),
        "sample_rate": int(model.config.audio.sample_rate),
        "normalize": bool(model.config.audio.normalize),
    }
    return model, spectttra_config


def preprocess_for_spectttra(audio, spectttra_config):
    """SpecTTTra 입력 형식에 맞춰 music_audio를 세그먼트 단위로 전처리한다.
    DF-Arena(16kHz, 4.04s, 정규화 없음)와 달리 SpecTTTra는 체크포인트마다
    샘플레이트·세그먼트 길이·정규화 방식이 다를 수 있어 별도로 처리한다."""
    segment_samples = spectttra_config["segment_samples"]
    sample_rate = spectttra_config["sample_rate"]

    if sample_rate != AUDIO_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=sample_rate,
            res_type="soxr_hq",
        ).astype(np.float32)

    segments = []
    for start in get_segment_starts(audio.size, segment_samples):
        segment = extract_segment(audio, start, segment_samples)
        if spectttra_config["normalize"]:
            # sonics 학습 코드(AudioDataset)의 normalize="std"와 동일한 방식:
            # 표준편차로 나눠 스케일만 맞추고(평균은 빼지 않음) 학습 시 입력 분포를 재현한다.
            std = float(np.std(segment))
            segment = segment / max(std, 1e-6)
        segments.append(segment.astype(np.float32))
    return segments


def predict_fake_music(model, spectttra_config, audio, device):
    """music_audio에 대해 SpecTTTra로 spoof(fake) 확률을 계산한다. 무음 구간은
    DF-Arena의 predict_fake()와 동일하게 RMS 기준으로 0.0을 반환한다."""
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segments = preprocess_for_spectttra(audio, spectttra_config)
    model_dtype = next(model.parameters()).dtype

    segment_scores = []
    for segment in segments:
        segment_tensor = torch.from_numpy(segment)[None].to(
            device=device, dtype=model_dtype
        )
        with torch.inference_mode():
            logits = model(segment_tensor)
            # SpecTTTra는 num_classes=1인 이진 분류기라 softmax가 아닌 sigmoid로
            # spoof(target=1) 확률을 구한다.
            probabilities = torch.sigmoid(logits.float())
        segment_scores.append(float(probabilities[0, 0]))

    return max(segment_scores)


# -----------------------------------------------------------------------------
# 7. 파일 단위 점수 계산 및 제출 파일 저장
# -----------------------------------------------------------------------------

def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    voice_score = voice_present * voice_fake
    music_score = music_present * music_fake
    return max(voice_score, music_score)


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def validate_submission(output_path, sample_submission_path):
    """생성된 submission.csv의 컬럼 순서와 ID 개수(및 ID 집합)가
    sample_submission.csv와 정확히 일치하는지 검증한다."""
    with sample_submission_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = reader.fieldnames
        expected_ids = [row["ID"].strip() for row in reader]

    with output_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = reader.fieldnames
        actual_ids = [row["ID"].strip() for row in reader]

    if actual_columns != expected_columns:
        raise ValueError(
            "컬럼 순서가 sample_submission.csv와 다릅니다: "
            f"expected={expected_columns}, actual={actual_columns}"
        )
    if len(actual_ids) != len(expected_ids):
        raise ValueError(
            "ID 개수가 sample_submission.csv와 다릅니다: "
            f"expected={len(expected_ids)}, actual={len(actual_ids)}"
        )
    if set(actual_ids) != set(expected_ids):
        raise ValueError("ID 구성이 sample_submission.csv와 일치하지 않습니다")

    print(
        f"[Validation] OK - columns match ({len(actual_columns)}), "
        f"{len(actual_ids)} IDs match sample_submission.csv"
    )


# -----------------------------------------------------------------------------
# 8. 모델 로딩(전체 루프 시작 전 1회) 및 GPU/CPU 이동 헬퍼
# -----------------------------------------------------------------------------
# 네 모델의 GPU 메모리 사용량을 합쳐도(PANNs Cnn14 ~300MB, HTDemucs ~80MB,
# SpecTTTra 17M 파라미터 ~70MB, DF-Arena 1B fp16 기준 ~2.3GB) 수 GB 수준이라
# 8GB 이상 GPU라면 네 모델을 동시에 device에 상주시켜도 60분/1,200개 처리에
# 무리가 없다. 다만 파라미터 수 기준으로 DF-Arena 1B가 지배적이므로,
# --low-memory 옵션을 켜면 DF-Arena 1B만 평소 CPU에 두고 실제로 호출하는
# 순간에만 device로 옮겼다가 바로 CPU로 되돌린다(가장 작은 PANNs/SpecTTTra는
# 굳이 옮길 필요가 없어 항상 device에 상주). HTDemucs는 원래 apply_model()
# 호출 시점에만 device를 사용하는 구조라 그대로 둔다.
# 주의: panns_inference.AudioTagging은 생성 시점의 device 문자열에 입력 텐서
# 이동 로직이 고정되어 있어(내부 self.device), 로딩 후 임의로 .model만 다른
# device로 옮기면 입력/가중치 device가 어긋나 에러가 난다 — 그래서 PANNs는
# 토글 대상에서 제외했다.

def move_model(model, device):
    return model.to(device)


def main():
    args = parse_arguments()
    device = select_device(args.device)
    cpu_device = torch.device("cpu")
    pipeline_start = time.perf_counter()

    # 1. 테스트 파일을 제출 양식의 ID 순서에 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)
    num_files = max(len(audio_files), 1)

    # 2. 네 모델(HTDemucs, PANNs, DF-Arena 1B, SpecTTTra)을 루프 시작 전
    #    딱 한 번만 로딩한다. 루프 안에서는 재사용만 하고 반복 로딩하지 않는다.
    print("Loading models: HTDemucs, PANNs, DF-Arena 1B, SpecTTTra ...")
    htdemucs_model = load_htdemucs_model()
    # PANNs·SpecTTTra는 용량이 작아 --low-memory에서도 device에 항상 상주시킨다
    # (DF-Arena 1B만 토글 대상, 위 8번 섹션 설명 참고).
    panns_model, voice_indices, music_indices = load_panns_model(device)
    df_arena_model, fake_label_index = load_df_arena_model(
        cpu_device if args.low_memory else device
    )
    spectttra_model, spectttra_config = load_spectttra_model(device)

    timings = {
        "HTDemucs (separation)": 0.0,
        "PANNs (voice presence)": 0.0,
        "PANNs (music presence)": 0.0,
        "DF-Arena (voice fake)": 0.0,
        "SpecTTTra (music fake)": 0.0,
    }

    # 3~7. 파일별로: HTDemucs 분리(1회) → PANNs×2(존재 판별) →
    #      DF-Arena(voice)/SpecTTTra(music)(fake 판별) → 파일 단위 점수 결합
    for index, audio_path in enumerate(
        tqdm(audio_files, desc="Pipeline (HTDemucs+PANNs+DF-Arena+SpecTTTra)")
    ):
        # 3-1. HTDemucs로 음성·음악을 분리한다 (파일당 1회, 이후 4개 계산에서 재사용).
        t0 = time.perf_counter()
        voice_audio, music_audio = separate_voice_and_music(
            audio_path, htdemucs_model, device
        )
        timings["HTDemucs (separation)"] += time.perf_counter() - t0

        # 3-2. voice_audio/music_audio 각각을 PANNs에 입력해 존재 확률을 계산한다.
        t0 = time.perf_counter()
        voice_present = predict_presence(panns_model, voice_indices, voice_audio)
        timings["PANNs (voice presence)"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        music_present = predict_presence(panns_model, music_indices, music_audio)
        timings["PANNs (music presence)"] += time.perf_counter() - t0

        # 3-3. voice_audio → DF-Arena 1B (기존 로직 유지) → VOICE_FAKE_PROB
        if args.low_memory:
            move_model(df_arena_model, device)
        t0 = time.perf_counter()
        voice_fake = predict_fake(df_arena_model, fake_label_index, voice_audio, device)
        timings["DF-Arena (voice fake)"] += time.perf_counter() - t0
        if args.low_memory:
            move_model(df_arena_model, cpu_device)

        # 3-4. music_audio → SpecTTTra (신규) → MUSIC_FAKE_PROB
        t0 = time.perf_counter()
        music_fake = predict_fake_music(
            spectttra_model, spectttra_config, music_audio, device
        )
        timings["SpecTTTra (music fake)"] += time.perf_counter() - t0

        # 3-5. 파일 단위 FILE_FAKE_PROB 계산 (기존 combine_file_fake_score 유지)
        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present
        )

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    del htdemucs_model, panns_model, df_arena_model, spectttra_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 8. 5개 예측값을 제출 파일로 저장하고, 형식이 sample_submission.csv와
    #    일치하는지 검증한다.
    save_submission(args.output, column_names, submission_rows)
    validate_submission(args.output, args.sample_submission)

    total_elapsed = time.perf_counter() - pipeline_start
    print("\n[Timing] 단계별 처리 시간 (60분/1,200개 제한 대비 병목 확인용)")
    for stage_name, elapsed in timings.items():
        print(f"  - {stage_name}: {elapsed:.2f}s total, {elapsed / num_files:.3f}s/file")
    print(
        f"[Timing] Total pipeline: {total_elapsed:.2f}s for {num_files} files "
        f"({total_elapsed / num_files:.3f}s/file)"
    )
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
