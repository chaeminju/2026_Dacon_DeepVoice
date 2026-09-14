"""ffmpeg의 GSM 코덱으로 왕복 인코딩해 전화채널 대역 열화를 시뮬레이션한다.
(AMR-NB는 이 환경 ffmpeg 빌드에 인코더가 없어 GSM-FR 사용 — libgsm 인/디코더 확인됨)
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def codec_roundtrip(wav, sr, target_sr=8000):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_wav = tmp / "in.wav"
        gsm_path = tmp / "roundtrip.gsm"
        out_wav = tmp / "out.wav"

        sf.write(in_wav, wav, sr)

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(in_wav),
                "-ar", str(target_sr), "-ac", "1",
                "-c:a", "gsm",
                str(gsm_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(gsm_path),
                "-ar", str(sr), "-ac", "1",
                str(out_wav),
            ],
            check=True,
        )
        out, _ = sf.read(out_wav, dtype="float32")
        return out


def apply_codec_roundtrip_if(wav, sr, probability, rng):
    if rng.random() < probability:
        try:
            return codec_roundtrip(wav, sr)
        except subprocess.CalledProcessError:
            return wav
    return wav
