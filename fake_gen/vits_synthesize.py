"""facebook/mms-tts-kor(사전학습 VITS, zero-shot)로 한국어 transcript를 합성해
KO TTS-fake 데이터를 만든다.
"""

import argparse
import csv
import random
from pathlib import Path

import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
from tqdm import tqdm


def load_kss_transcripts(kss_dir):
    from datasets import load_dataset

    ds = load_dataset(kss_dir, split="train")
    return [(f"kss_{i:05d}", ex["original_script"]) for i, ex in enumerate(ds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kss-dir", default="data/raw/kss")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--out-dir", default="data/processed/ko_raw/tts")
    parser.add_argument("--manifest", default="data/processed/ko_tts_manifest.csv")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading transcripts...")
    transcripts = load_kss_transcripts(args.kss_dir)
    rng = random.Random(args.seed)
    rng.shuffle(transcripts)
    selected = transcripts[: args.n_samples]

    print(f"loading VITS model (facebook/mms-tts-kor)...")
    model = VitsModel.from_pretrained("facebook/mms-tts-kor").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    sr = model.config.sampling_rate

    rows = []
    for utt_id, text in tqdm(selected, desc="VITS synthesize"):
        inputs = tok(text, return_tensors="pt").to("cuda")
        if inputs["input_ids"].shape[1] == 0:
            continue
        with torch.no_grad():
            waveform = model(**inputs).waveform[0].cpu().numpy()
        out_path = out_dir / f"{utt_id}.wav"
        sf.write(out_path, waveform, sr)
        rows.append(
            {
                "path": str(out_path),
                "lang": "ko",
                "label": "spoof",
                "technique": "tts",
                "system_id": "mms-tts-kor",
                "speaker_id": "vits_default",
                "source": "kss_transcript",
                "text": text,
            }
        )

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "lang", "label", "technique", "system_id",
                "speaker_id", "source", "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} files, manifest at {args.manifest}")


if __name__ == "__main__":
    main()
