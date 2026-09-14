"""입력 manifest(real/TTS-fake/VC-fake 등)의 모든 소스에 동일한 확률로
RawBoost + 코덱 왕복을 적용한다. 특정 소스만 증강되면 탐지기가 증강 여부로
라벨을 유추할 수 있으므로, label/technique와 무관하게 같은 확률을 적용한다.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from rawboost import apply_rawboost
from codec_roundtrip import apply_codec_roundtrip_if

TARGET_SR = 16000


def process_row(row, out_dir, rawboost_p, codec_p, rng, seed):
    wav, sr = sf.read(row["path"], dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    applied = []
    if rng.random() < rawboost_p:
        wav = apply_rawboost(wav, sr=sr, seed=seed)
        applied.append("rawboost")

    before = wav
    wav = apply_codec_roundtrip_if(wav, sr, codec_p, rng)
    if wav is not before:
        applied.append("codec")

    src_name = Path(row["path"]).stem
    out_path = out_dir / f"{src_name}_aug.wav"
    sf.write(out_path, wav.astype(np.float32), sr)

    new_row = dict(row)
    new_row["path"] = str(out_path)
    new_row["augmentation"] = "+".join(applied) if applied else "none"
    return new_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--rawboost-p", type=float, default=0.5)
    parser.add_argument("--codec-p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    fieldnames = None
    for manifest_path in args.manifests:
        with open(manifest_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            all_rows.extend(list(reader))

    rng = random.Random(args.seed)
    out_rows = []
    for i, row in enumerate(tqdm(all_rows, desc="augment")):
        out_rows.append(
            process_row(row, out_dir, args.rawboost_p, args.codec_p, rng, args.seed + i)
        )

    out_fieldnames = list(fieldnames) + (
        ["augmentation"] if "augmentation" not in fieldnames else []
    )
    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"augmented {len(out_rows)} files -> {args.out_manifest}")


if __name__ == "__main__":
    main()
