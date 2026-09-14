"""RawBoost 증강 (ASVspoof2021 LA 베이스라인에서 널리 쓰이는 3종 알고리즘의
축약 구현): 선형/비선형 컨볼루션 노이즈(LnL), 충격성 신호독립 잡음(ISD),
정상 신호독립 잡음(SSI). 원 논문: Tak et al., "RawBoost: A Raw Data Boosting
and Augmentation Method for Anti-Spoofing", ICASSP 2022.
"""

import numpy as np


def _normalize(x, eps=1e-9):
    peak = np.max(np.abs(x)) + eps
    return x / peak


def lnl_convolutive_noise(
    x, n_bands=5, min_f=20, max_f=8000, min_bw=100, max_bw=1000,
    min_coeff=10, max_coeff=100, min_g=0, max_g=0, sr=16000, rng=None,
):
    rng = rng or np.random
    y = np.zeros_like(x)
    for _ in range(n_bands):
        fc = rng.uniform(min_f, max_f)
        bw = rng.uniform(min_bw, max_bw)
        order = rng.randint(min_coeff, max_coeff + 1)
        low = max(1, fc - bw / 2)
        high = min(sr / 2 - 1, fc + bw / 2)
        if high <= low:
            continue
        from scipy.signal import firwin, lfilter

        taps = firwin(order, [low, high], pass_zero=False, fs=sr)
        gain_db = rng.uniform(min_g, max_g)
        y += lfilter(taps, [1.0], x) * (10 ** (gain_db / 20))
    return _normalize(x + y)


def isd_additive_noise(x, p=10, g=2, rng=None):
    rng = rng or np.random
    n = len(x)
    n_impulses = int(n * p / 100 / 10)
    idx = rng.choice(n, size=max(1, n_impulses), replace=False)
    noise = np.zeros(n)
    noise[idx] = rng.uniform(-1, 1, size=idx.size) * g
    return _normalize(x + noise)


def ssi_additive_noise(x, snr_db_range=(10, 40), rng=None):
    rng = rng or np.random
    snr_db = rng.uniform(*snr_db_range)
    signal_power = np.mean(x ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=len(x))
    return _normalize(x + noise)


def apply_rawboost(x, sr=16000, seed=None):
    """세 알고리즘 중 1~2개를 무작위로 조합 적용."""
    rng = np.random.RandomState(seed)
    algos = []
    if rng.rand() < 0.6:
        algos.append(lambda a: lnl_convolutive_noise(a, sr=sr, rng=rng))
    if rng.rand() < 0.5:
        algos.append(lambda a: isd_additive_noise(a, rng=rng))
    if rng.rand() < 0.5:
        algos.append(lambda a: ssi_additive_noise(a, rng=rng))
    if not algos:
        algos.append(lambda a: ssi_additive_noise(a, rng=rng))

    y = x.astype(np.float64)
    for algo in algos:
        y = algo(y)
    return y.astype(np.float32)
