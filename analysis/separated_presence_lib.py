#!/usr/bin/env python3
"""
HTDemucs로 먼저 분리한 뒤 그 stem(voice_audio/music_audio)에 대해 PANNs presence를
측정하기 위한 공용 유틸리티. submit/script.py의 separate_voice_and_music과 정규화·추론
로직을 동일하게 유지하되, load_track(ffmpeg/torchaudio 디코딩) 대신 이미 로드된 numpy
배열(analysis 스크립트들이 쓰는 load_wav_mono 결과)을 입력으로 받는다.
분석 대상 데이터가 전부 WAV이므로 디코딩 경로 차이로 인한 수치 차이는 없다.
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_DIR = Path(__file__).resolve().parent.parent
HTDEMUCS_DIR = REPO_DIR / "baseline_submit" / "model" / "htdemucs"

AUDIO_SAMPLE_RATE = 16_000


def load_htdemucs_model():
    from demucs.pretrained import get_model

    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.eval()


def _to_model_input(mono_audio_16k, model):
    waveform = torch.from_numpy(mono_audio_16k).float()
    waveform = torchaudio.functional.resample(
        waveform, AUDIO_SAMPLE_RATE, model.samplerate
    )
    waveform = waveform.unsqueeze(0).repeat(model.audio_channels, 1)
    return waveform


def separate_voice_and_music(mono_audio_16k, model, device):
    """submit/script.py:separate_voice_and_music과 동일한 정규화/분리/합산 로직.
    입력만 파일 경로 대신 16kHz mono numpy 배열."""
    from demucs.apply import apply_model

    waveform = _to_model_input(mono_audio_16k, model)
    mean = waveform.mean()
    std = waveform.std()

    if float(std) < 1e-8:
        silence = np.zeros_like(mono_audio_16k, dtype=np.float32)
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
