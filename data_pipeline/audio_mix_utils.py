"""D_partial_splice / H_hybrid_composed / I_bgm_evasion 자체 제작용 공용 오디오
유틸(로드/길이맞춤/SNR 믹싱/저장). 새 데이터셋을 받지 않고 기존 real/fake
음성·음악 자산을 조합해 새 케이스 데이터를 합성할 때 세 스크립트가 공통으로
쓴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, orig_sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """x를 정확히 n 샘플로 맞춘다 (짧으면 반복, 길면 임의 구간 크롭)."""
    if x.size < n:
        reps = n // x.size + 1
        x = np.tile(x, reps)
    if x.size > n:
        start = int(rng.integers(0, x.size - n + 1))
        x = x[start : start + n]
    return x.astype(np.float32)


def mix_at_snr(foreground: np.ndarray, background: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """foreground(주로 음성) 대비 background(배경음)를 지정한 SNR(dB)로 섞는다.
    background가 foreground보다 짧으면 반복, 길면 임의 구간을 크롭해 길이를
    맞춘다."""
    background = fit_length(background, foreground.size, rng)
    fg_rms, bg_rms = rms(foreground), rms(background)
    target_bg_rms = fg_rms / (10 ** (snr_db / 20))
    scale = target_bg_rms / bg_rms
    mixed = foreground + background * scale
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32)


def splice_segment(carrier: np.ndarray, segment: np.ndarray, rng: np.random.Generator, crossfade_samples: int = 160) -> np.ndarray:
    """carrier(주로 real 음성) 중간의 임의 구간을 segment(주로 fake 음성)로
    치환한다. 경계에서 짧게 크로스페이드해 클릭음을 줄인다."""
    out = carrier.copy()
    seg_len = min(segment.size, max(1, carrier.size // 3))
    segment = segment[:seg_len]
    if carrier.size <= seg_len:
        return fit_length(segment, carrier.size, rng)

    start = int(rng.integers(0, carrier.size - seg_len + 1))
    end = start + seg_len
    cf = min(crossfade_samples, seg_len // 4 if seg_len >= 4 else 0)

    out[start:end] = segment
    if cf > 0:
        fade = np.linspace(0, 1, cf, dtype=np.float32)
        out[start : start + cf] = carrier[start : start + cf] * (1 - fade) + segment[:cf] * fade
        out[end - cf : end] = segment[-cf:] * (1 - fade) + carrier[end - cf : end] * fade
    return out


def write_wav(path: str | Path, wav: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)
