"""Zeroth-Korean(메인) + KSS(보조)에서 KO real 데이터를 서브샘플링해
16kHz wav + manifest로 추출한다.
"""

import argparse
import csv
import random
from pathlib import Path

import librosa
import soundfile as sf

ZEROTH_ROOT = Path("data/raw/zeroth_korean/train_data_01")
OUT_DIR = Path("data/processed/ko_raw/real")
MANIFEST_PATH = Path("data/processed/ko_real_manifest.csv")


def load_zeroth_transcripts():
    """speaker_id -> {utt_id: text}"""
    transcripts = {}
    for trans_file in ZEROTH_ROOT.rglob("*.trans.txt"):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            transcripts[utt_id] = text
    return transcripts


def stratified_zeroth_sample(n_total, seed, exclude_speakers):
    import collections

    by_speaker = collections.defaultdict(list)
    for flac in ZEROTH_ROOT.rglob("*.flac"):
        speaker_id = flac.stem.split("_")[0]
        if speaker_id in exclude_speakers:
            continue
        by_speaker[speaker_id].append(flac)

    rng = random.Random(seed)
    speakers = list(by_speaker.keys())
    per_speaker = max(1, n_total // len(speakers))
    selected = []
    for spk in speakers:
        pool = by_speaker[spk]
        rng.shuffle(pool)
        selected.extend(pool[:per_speaker])
    rng.shuffle(selected)
    return selected[:n_total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-zeroth", type=int, default=900)
    parser.add_argument("--n-kss", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exclude-speakers",
        default="201,152,200,185",
        help="RVC 타깃으로 쓴 화자는 real 세트에서 제외해 fake 학습과 real 라벨이 겹치지 않게 함",
    )
    args = parser.parse_args()

    exclude = set(args.exclude_speakers.split(","))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading zeroth transcripts...")
    transcripts = load_zeroth_transcripts()

    zeroth_files = stratified_zeroth_sample(args.n_zeroth, args.seed, exclude)
    print(f"selected {len(zeroth_files)} zeroth files (excluding speakers {exclude})")

    rows = []
    for f in zeroth_files:
        wav, sr = librosa.load(f, sr=16000, mono=True)
        out_path = OUT_DIR / f"zeroth_{f.stem}.wav"
        sf.write(out_path, wav, sr)
        speaker_id = f.stem.split("_")[0]
        rows.append(
            {
                "path": str(out_path),
                "lang": "ko",
                "label": "bonafide",
                "technique": "real",
                "system_id": "-",
                "speaker_id": f"zeroth_{speaker_id}",
                "source": "zeroth_korean",
                "text": transcripts.get(f.stem, ""),
            }
        )

    print("loading KSS...")
    from datasets import load_dataset

    ds = load_dataset("data/raw/kss", split="train")
    rng = random.Random(args.seed + 1)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for i in indices[: args.n_kss]:
        ex = ds[i]
        audio = ex["audio"].get_all_samples()
        wav = audio.data.numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        wav16k = librosa.resample(wav, orig_sr=audio.sample_rate, target_sr=16000)
        out_path = OUT_DIR / f"kss_{i:05d}.wav"
        sf.write(out_path, wav16k, 16000)
        rows.append(
            {
                "path": str(out_path),
                "lang": "ko",
                "label": "bonafide",
                "technique": "real",
                "system_id": "-",
                "speaker_id": "kss_speaker",
                "source": "kss",
                "text": ex["original_script"],
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "lang", "label", "technique", "system_id",
                "speaker_id", "source", "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} files, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
