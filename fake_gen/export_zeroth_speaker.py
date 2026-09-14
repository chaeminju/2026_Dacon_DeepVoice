"""Zeroth-Korean에서 특정 화자의 flac을 wav로 내보낸다 (RVC 학습/변환 소스용)."""

import argparse
from pathlib import Path

import soundfile as sf
import librosa

ROOT = Path("data/raw/zeroth_korean/train_data_01")


def export_speaker(speaker_id, out_dir, limit=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ROOT.rglob(f"{speaker_id}_*.flac"))
    if limit:
        files = files[:limit]
    for f in files:
        wav, sr = librosa.load(f, sr=None, mono=True)
        sf.write(out_dir / f"{f.stem}.wav", wav, sr)
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    n = export_speaker(args.speaker_id, args.out_dir, args.limit)
    print(f"exported {n} files for speaker {args.speaker_id} -> {args.out_dir}")
