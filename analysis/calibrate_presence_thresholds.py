#!/usr/bin/env python3
"""
PANNs VOICE_PRESENT_PROB / MUSIC_PRESENT_PROB의 실제 분포를 4가지 카테고리
(음악만 / 음성만 / 아무것도 없음 / 음성+음악 혼합)에 대해 측정하고,
change_combine/script.py의 presence_weight(low, high)를 재산정하기 위한
근거 데이터를 만든다.

PANNs 전처리(get_segment_starts/extract_segment/make_panns_segments/predict_presence)는
baseline_submit/script.py와 동일한 로직을 그대로 사용한다 (제출 코드와 동일 조건 보장).

데이터 소스 (모두 실제 데이터, 합성/생성 없음):
  - 음악만       : MUSDB18 test의 drums+bass+other 스템 합(보컬 제외 반주)
  - 음성만       : LibriSpeech test-clean(순수 발화) + MUSDB18 vocals 스템(아카펠라)
  - 아무것도 없음 : DNS-Challenge 환경 소음 클립
  - 음성+음악 혼합 : MUSDB18 test의 mixture 트랙(실제 곡 믹스)
"""

import csv
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

random.seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent.parent
PANNS_DIR = REPO_DIR / "baseline_submit" / "model" / "panns"

MUSDB_TEST_DIR = Path("/database/database/musdb18_wav/test")
LIBRISPEECH_DIR = Path("/database/database/LibriSpeech/test-clean_wav_16k")
NOISE_DIR = Path("/database/database/dns_challenge/datasets/noise")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "presence_probabilities.csv"

AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600

N_SONGS = 25          # MUSDB 곡 수 (음악만/혼합 각각 이 수만큼 클립 생성)
N_SPEECH = 30          # LibriSpeech 발화 클립 수
N_VOCAL_STEMS = 20     # MUSDB 아카펠라 보컬 클립 수 (음성만에 합산)
N_NOISE = 50           # DNS 소음 클립 수


# -----------------------------------------------------------------------------
# baseline_submit/script.py와 동일한 PANNs 전처리 (demucs 미설치 환경이라 직접 재사용)
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


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment, orig_sr=AUDIO_SAMPLE_RATE, target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def prepare_panns_labels():
    import shutil
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device,
    )
    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def predict_presence(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    return voice_probability, music_probability


# -----------------------------------------------------------------------------
# 카테고리별 클립 로더
# -----------------------------------------------------------------------------
def load_wav_mono(path, target_sr=AUDIO_SAMPLE_RATE):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return audio.astype(np.float32)


def musdb_song_names():
    names = sorted({
        p.name.rsplit(".stem_track", 1)[0]
        for p in MUSDB_TEST_DIR.glob("*.stem_track0.wav")
    })
    return names


def random_window(audio, seconds, rng):
    n = int(seconds * AUDIO_SAMPLE_RATE)
    if audio.size <= n:
        return audio
    start = rng.randint(0, audio.size - n)
    return audio[start:start + n]


def build_clip_list():
    rng = random.Random(42)
    clips = []  # (category, source, path_or_desc, loader)

    song_names = musdb_song_names()
    rng.shuffle(song_names)

    # 1) 음악만: drums(1) + bass(2) + other(3) 합 (보컬 제외 반주), 곡당 8초 랜덤 구간
    for name in song_names[:N_SONGS]:
        def loader(name=name):
            parts = []
            for track_idx in (1, 2, 3):
                p = MUSDB_TEST_DIR / f"{name}.stem_track{track_idx}.wav"
                parts.append(load_wav_mono(p))
            min_len = min(a.size for a in parts)
            mix = np.sum([a[:min_len] for a in parts], axis=0).astype(np.float32)
            return random_window(mix, 8.0, rng)
        clips.append(("music_only", "musdb_accompaniment", name, loader))

    # 2) 음성+음악 혼합: mixture(0) 트랙, 곡당 8초 랜덤 구간 (음악만과 겹치지 않는 곡 사용)
    for name in song_names[N_SONGS:N_SONGS * 2]:
        def loader(name=name):
            p = MUSDB_TEST_DIR / f"{name}.stem_track0.wav"
            return random_window(load_wav_mono(p), 8.0, rng)
        clips.append(("mixed", "musdb_mixture", name, loader))

    # 3) 음성만 (a): LibriSpeech 순수 발화, 파일 그대로 사용
    speech_files = list(LIBRISPEECH_DIR.rglob("*.wav"))
    rng.shuffle(speech_files)
    for p in speech_files[:N_SPEECH]:
        def loader(p=p):
            return load_wav_mono(p)
        clips.append(("voice_only", "librispeech_speech", p.name, loader))

    # 3) 음성만 (b): MUSDB 아카펠라 보컬 스템(track4), 8초 랜덤 구간
    # (MUSDB test는 50곡뿐이라 music_only에 쓴 곡을 재사용 — 다른 스템이므로 문제 없음)
    vocal_song_names = song_names[:N_VOCAL_STEMS]
    for name in vocal_song_names:
        def loader(name=name):
            p = MUSDB_TEST_DIR / f"{name}.stem_track4.wav"
            return random_window(load_wav_mono(p), 8.0, rng)
        clips.append(("voice_only", "musdb_vocals", name, loader))

    # 4) 아무것도 없음: DNS 환경 소음 클립, 파일 그대로 사용(약 10초)
    noise_files = list(NOISE_DIR.glob("*.wav"))
    rng.shuffle(noise_files)
    for p in noise_files[:N_NOISE]:
        def loader(p=p):
            return load_wav_mono(p)
        clips.append(("nothing", "dns_noise", p.name, loader))

    return clips


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, voice_indices, music_indices = load_panns_model(device)

    clips = build_clip_list()
    print(f"total clips: {len(clips)}")

    rows = []
    for category, source, name, loader in clips:
        try:
            audio = loader()
            if audio.size == 0 or not np.isfinite(audio).all():
                print(f"skip (invalid audio): {category}/{source}/{name}")
                continue
            voice_prob, music_prob = predict_presence(model, voice_indices, music_indices, audio)
        except Exception as exc:
            print(f"skip (error {exc}): {category}/{source}/{name}")
            continue
        rows.append({
            "category": category,
            "source": source,
            "name": name,
            "duration_sec": round(audio.size / AUDIO_SAMPLE_RATE, 3),
            "voice_present_prob": round(voice_prob, 6),
            "music_present_prob": round(music_prob, 6),
        })
        print(f"{category:10s} {source:22s} {name[:40]:40s} "
              f"voice={voice_prob:.4f} music={music_prob:.4f}")

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
