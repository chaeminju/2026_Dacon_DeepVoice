"""submit_spectttra/ (기존 KO+EN DF-Arena + zero-shot SpecTTTra 제출본)를 복제해
spectttra 가중치만 파인튜닝된 것으로 교체하고 제출용 zip으로 묶는다.
build_submission.py와 동일한 패턴, 대상 서브모델만 spectttra로 바꾼 버전."""

import argparse
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_SUBMIT = BASE_DIR / "submit_spectttra"


def build_submission(finetuned_dir, out_dir_name, out_zip):
    finetuned_dir = Path(finetuned_dir)
    staging = BASE_DIR / out_dir_name
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(SOURCE_SUBMIT, staging)

    spectttra_dst = staging / "model" / "spectttra"
    # 원본 가중치(pytorch_model.bin)만 지우고 sonics/ 패키지 코드는 그대로 유지.
    for pattern in ("pytorch_model*.bin", "model*.safetensors"):
        for f in spectttra_dst.glob(pattern):
            f.unlink()

    for f in finetuned_dir.iterdir():
        if f.is_file():
            shutil.copy(f, spectttra_dst / f.name)

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    print(f"built {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB)")
    return staging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument("--out-dir-name", default="submit_spectttra_ft")
    parser.add_argument("--out-zip", default=str(BASE_DIR / "submissions" / "submission_spectttra_ft.zip"))
    args = parser.parse_args()
    build_submission(args.finetuned_dir, args.out_dir_name, args.out_zip)


if __name__ == "__main__":
    main()
