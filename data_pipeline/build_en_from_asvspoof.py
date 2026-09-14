"""ASVspoof2019 LA train split에서 EN 케이스용 real/TTS-fake/VC-fake를 서브샘플링해
wav 파일 + manifest로 추출한다.

A01-A04 = TTS 계열, A05-A06 = VC 계열 (ASVspoof2019 논문 Table 1 기준, 문헌에서
잘 알려진 매핑). bonafide는 key==0.
"""

import argparse
import collections
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

TTS_SYSTEMS = ["A01", "A02", "A03", "A04"]
VC_SYSTEMS = ["A05", "A06"]

OUT_DIR = Path("data/processed/en_raw")
MANIFEST_PATH = Path("data/processed/en_raw_manifest.csv")


def stratified_sample(indices_by_key, n_total, seed):
    rng = random.Random(seed)
    keys = list(indices_by_key.keys())
    per_key = n_total // len(keys)
    remainder = n_total - per_key * len(keys)
    selected = []
    for i, key in enumerate(keys):
        pool = list(indices_by_key[key])
        rng.shuffle(pool)
        take = per_key + (1 if i < remainder else 0)
        selected.extend(pool[:take])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real", type=int, default=1200)
    parser.add_argument("--n-tts", type=int, default=600)
    parser.add_argument("--n-vc", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bonafide").mkdir(exist_ok=True)
    (OUT_DIR / "tts").mkdir(exist_ok=True)
    (OUT_DIR / "vc").mkdir(exist_ok=True)

    print("loading dataset (train split)...")
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split="train")

    by_speaker_bonafide = collections.defaultdict(list)
    by_system = collections.defaultdict(list)
    for idx, (key, system_id, speaker_id) in enumerate(
        zip(ds["key"], ds["system_id"], ds["speaker_id"])
    ):
        if key == 0:
            by_speaker_bonafide[speaker_id].append(idx)
        else:
            by_system[system_id].append(idx)

    real_indices = stratified_sample(by_speaker_bonafide, args.n_real, args.seed)
    tts_pool = {s: by_system[s] for s in TTS_SYSTEMS}
    vc_pool = {s: by_system[s] for s in VC_SYSTEMS}
    tts_indices = stratified_sample(tts_pool, args.n_tts, args.seed + 1)
    vc_indices = stratified_sample(vc_pool, args.n_vc, args.seed + 2)

    print(f"selected: real={len(real_indices)} tts={len(tts_indices)} vc={len(vc_indices)}")

    rows = []

    def export(indices, technique, subdir):
        for idx in indices:
            ex = ds[idx]
            audio = ex["audio"].get_all_samples()
            wav = audio.data.numpy()
            if wav.ndim > 1:
                wav = wav.mean(axis=0)
            sr = audio.sample_rate
            file_id = ex["audio_file_name"]
            out_path = OUT_DIR / subdir / f"{file_id}.wav"
            sf.write(out_path, wav.astype(np.float32), sr)
            rows.append(
                {
                    "path": str(out_path),
                    "lang": "en",
                    "label": "bonafide" if technique == "real" else "spoof",
                    "technique": technique,
                    "system_id": ex["system_id"],
                    "speaker_id": ex["speaker_id"],
                    "source": "asvspoof2019la",
                }
            )

    export(real_indices, "real", "bonafide")
    export(tts_indices, "tts", "tts")
    export(vc_indices, "vc", "vc")

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "lang", "label", "technique", "system_id", "speaker_id", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} files, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
