"""학습된 RVC 모델로 다른 화자의 실음성을 타깃 화자 목소리로 변환한다
(KO VC-fake 생성). infer/cli.py의 디렉터리 배치 모드를 화자(실험)당 1회 호출."""

import argparse
import csv
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

RVC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "rvc"


def convert_batch(model_pth, input_dir, output_dir):
    subprocess.run(
        [
            "python", "-m", "infer.cli",
            "--model", str(model_pth),
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--pitch", "0",
            "--f0-method", "rmvpe",
            "--index-rate", "0.75",
            "--overwrite",
        ],
        cwd=RVC_DIR,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", required=True, help="RVC 실험명 목록 (타깃 화자들)")
    parser.add_argument("--source-manifest", required=True, help="변환 소스로 쓸 real manifest csv")
    parser.add_argument("--n-per-speaker", type=int, default=150)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.source_manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    rows_out = []
    idx = 0
    for exp in args.experiments:
        model_pth = RVC_DIR / "assets" / "weights" / f"{exp}.pth"
        if not model_pth.exists():
            raise FileNotFoundError(model_pth)

        picked = rows[idx: idx + args.n_per_speaker]
        idx += args.n_per_speaker

        with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
            tmp_in = Path(tmp_in)
            tmp_out = Path(tmp_out)
            src_by_stem = {}
            for row in picked:
                src_path = Path(row["path"])
                dst = tmp_in / src_path.name
                shutil.copy(src_path, dst)
                src_by_stem[src_path.stem] = row

            convert_batch(model_pth, tmp_in, tmp_out)

            n_ok = 0
            for out_file in tmp_out.glob("*.wav"):
                stem = out_file.stem
                row = src_by_stem.get(stem)
                if row is None:
                    continue
                final_path = out_dir / f"{exp}_from_{stem}.wav"
                shutil.move(str(out_file), final_path)
                rows_out.append(
                    {
                        "path": str(final_path),
                        "lang": "ko",
                        "label": "spoof",
                        "technique": "vc",
                        "system_id": f"rvc_{exp}",
                        "speaker_id": exp,
                        "source": f"rvc_from_{row.get('speaker_id', '')}",
                        "text": row.get("text", ""),
                    }
                )
                n_ok += 1
        print(f"{exp}: converted {n_ok}/{len(picked)} files")

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path", "lang", "label", "technique", "system_id",
                "speaker_id", "source", "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"wrote {len(rows_out)} files, manifest at {args.out_manifest}")


if __name__ == "__main__":
    main()
