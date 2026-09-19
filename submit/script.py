#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다."""

import argparse
import csv
import json
import os
import shutil
import sys
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
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
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
        description="Run the zero-shot audio deepfake baseline."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
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

def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. PANNs를 이용한 음성·음악 존재 여부 추론
# -----------------------------------------------------------------------------

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


def predict_presence(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    return voice_probability, music_probability


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
# 5. DF-Arena 1B를 이용한 성분별 Fake 추론
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)

    # 파인튜닝 체크포인트가 fp16(F16)으로 저장되어 있어, 나머지 파이프라인이 쓰는
    # fp32 입력(segment_tensor)과 dtype을 맞추기 위해 명시적으로 float32로 캐스팅한다.
    model = model.to(device=device, dtype=torch.float32).eval()
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index, audio, device):
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment_tensor = torch.from_numpy(segment).to(device)

        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        segment_scores.append(float(probabilities[0, fake_label_index]))

    return max(segment_scores)


# -----------------------------------------------------------------------------
# 6. 파일 단위 점수 계산 및 제출 파일 저장
# -----------------------------------------------------------------------------
#
# VOICE_PRESENT_PROB / MUSIC_PRESENT_PROB는 원본(분리 전) 오디오가 아니라 HTDemucs로
# 분리된 stem에서 각각 계산한다 (voice_audio -> voice_present, music_audio ->
# music_present). presence 판정과 fake 판정이 같은 신호를 보게 되어 리더보드 검증 결과
# 원본 오디오 기준보다 소폭 성능이 개선됨을 확인했다.
#
# 아래 LOW/HIGH는 기존 3,000개 카테고리 코퍼스(음성만/음악만/무음/환경음/음성+음악
# 혼합, 각 500개)를 HTDemucs로 먼저 분리한 뒤 그 stem으로 PANNs를 다시 돌려 재산정한
# 값이다 (analysis/run_separated_presence_all.py, analysis/recalibrate_separated_
# thresholds.py). 근거:
#   - VOICE 부재 후보(music_only+ambient_noise+true_silence, n=1500)의 p95=0.395,
#     존재 후보(voice_only+speech_plus_music, n=1000)의 p5=0.757 -> LOW=0.40, HIGH=0.76
#   - MUSIC은 처음에 부재 후보(voice_only+ambient_noise+true_silence, n=1500)의
#     p95=0.818을 그대로 LOW로 썼더니 실제 music_only의 10~15%(p10=0.798, p25=0.857)가
#     LOW 밑으로 깔려 통째로 weight=0 처리되는 문제가 있었다. 원인은 ambient_noise
#     카테고리 하나가 p95=0.907로 비정상적으로 높아 부재 후보 전체를 끌어올린 것
#     (voice_only/true_silence만 보면 p95=0.119, max=0.696에 불과). ambient_noise를
#     제외한 깨끗한 부재 후보로 다시 잡고, 실제 fake score까지 반영한 end-to-end
#     시뮬레이션으로 검증해 LOW=0.35, HIGH=0.71(존재 후보 p5)로 재조정했다.
#     이 값에서 music_only의 평균 presence_weight는 0.789->0.971로 회복되지만, 그
#     대가로 ambient_noise의 평균 weight도 0.134->0.285로 다소 늘어난다(노이즈를
#     music으로 오인하는 잔여 리스크가 커지는 대신, 진짜 음악을 놓치는 위험이 줄어듦).
#   - 잔여 리스크: ambient_noise(DNS 환경음)는 분리 후 일부가 voice/music
#     presence_weight를 상당히 받는 오탐이 남아있음(분리 시 노이즈 입력에서 나오는
#     아티팩트로 추정). 원본(분리 전) 방식에서도 이미 알려져 있던 환경음-음악 혼동의
#     연장선.
#   - 별도로 확인된 더 큰 문제: DF-Arena-1B 자체가 완전히 real인 음성/음악에도 상당한
#     확률로 fake를 예측하는 경향이 있다(원본/분리 여부 무관하게 재현됨). presence
#     게이트만으로는 이 근본 원인을 해결할 수 없고, DF-Arena 쪽 재보정이 필요하다.
#
# HIGH를 없애고 LOW만 남긴 이유: LOW~HIGH 선형 램프(기존 방식)는 경계 위 값을 전부
# 1.0으로 밀어올리는데, 4개 핵심 카테고리(voice_only/music_only/true_silence/
# speech_plus_music)의 raw presence 확률은 이미 0.84~0.88대로 충분히 높아서, 굳이
# 1.0까지 밀어올리면 DF-Arena의 fake 과확신을 한 번 더 증폭시킬 뿐이었다. 실제
# fake score까지 반영한 end-to-end 시뮬레이션(n=120/카테고리)으로 세 방식을 비교한
# 결과, "LOW 밑은 0, 위는 raw 값 그대로"가 모든 카테고리에서 기존 램프 방식과 같거나
# 더 낮은(=더 안전한) FILE_FAKE_PROB를 냈다:
#   전체 평균 FILE_FAKE_PROB - RAW(게이트 없음) 0.355 / 램프(LOW~HIGH) 0.385 / LOW-only 0.337
#   무음:  RAW 0.035 -> LOW-only 0.000 (여전히 유효 - 잔여 노이즈 완전 차단)
#   음성만/음악만/혼합: 램프 방식이 오히려 RAW보다 나빴던 것을 LOW-only가 RAW 수준으로 되돌림
# 즉 LOW(바닥 차단)는 유지할 가치가 있지만 HIGH(천장 증폭)는 제거하는 것이 더 낫다.
VOICE_PRESENCE_LOW = 0.40
MUSIC_PRESENCE_LOW = 0.35


def presence_weight(value, low):
    return value if value >= low else 0.0


def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    voice_score = presence_weight(voice_present, VOICE_PRESENCE_LOW) * voice_fake
    music_score = presence_weight(music_present, MUSIC_PRESENCE_LOW) * music_fake
    return max(voice_score, music_score)


def run_pipeline_for_all_files(audio_files, submission_rows, device):
    panns_model, voice_indices, music_indices = load_panns_model(device)
    htdemucs_model = load_htdemucs_model()
    df_arena_model, fake_label_index = load_df_arena_model(device)

    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):
        voice_audio, music_audio = separate_voice_and_music(
            audio_path, htdemucs_model, device
        )

        voice_present, _ = predict_presence(
            panns_model, voice_indices, music_indices, voice_audio
        )
        _, music_present = predict_presence(
            panns_model, voice_indices, music_indices, music_audio
        )

        voice_fake = predict_fake(
            df_arena_model, fake_label_index, voice_audio, device
        )
        music_fake = predict_fake(
            df_arena_model, fake_label_index, music_audio, device
        )

        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present
        )

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    del panns_model, htdemucs_model, df_arena_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return submission_rows


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)

    # 1. 테스트 파일을 제출 양식의 ID 순서에 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)

    # 2. 음성·음악을 분리한 뒤, 분리된 stem 기준으로 존재 확률과 Fake 확률을 계산한다.
    submission_rows = run_pipeline_for_all_files(audio_files, submission_rows, device)

    # 3. 5개 예측값을 제출 파일로 저장한다.
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
