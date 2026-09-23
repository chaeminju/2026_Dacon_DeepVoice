#!/usr/bin/env python3
"""submit/ 폴더(model/, script.py, requirements.txt)를 대회 제출 규격에 맞춰
submit.zip으로 압축한다 (최상위 폴더 없이 세 항목이 zip 루트에 오도록)."""

import argparse
import os
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ZIP = BASE_DIR / "submit.zip"
MAX_ZIP_SIZE_GB = 10

REQUIRED_ENTRIES = ["model", "script.py", "requirements.txt"]


def check_required_entries():
    missing = [name for name in REQUIRED_ENTRIES if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"제출에 필요한 항목이 없습니다: {missing}")


def build_submit_zip(output_zip=DEFAULT_OUTPUT_ZIP):
    check_required_entries()

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(BASE_DIR / "script.py", arcname="script.py")
        zip_file.write(BASE_DIR / "requirements.txt", arcname="requirements.txt")

        # model/ 하위(예: htdemucs, panns)는 baseline_submit/model에 대한 심볼릭
        # 링크일 수 있으므로 followlinks=True로 실제 경로를 따라간다. Path.rglob은
        # 심볼릭 링크된 디렉터리를 내려가지 않아 사용하지 않는다.
        model_dir = (BASE_DIR / "model").resolve()
        file_paths = []
        for root, _dirs, file_names in os.walk(model_dir, followlinks=True):
            for file_name in file_names:
                file_paths.append(Path(root) / file_name)

        for file_path in sorted(file_paths):
            arcname = Path("model") / file_path.relative_to(model_dir)
            zip_file.write(file_path, arcname=str(arcname))

    return output_zip


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help="생성할 zip 파일 경로 (기본값: submit/submit.zip)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_zip = args.output if args.output.is_absolute() else BASE_DIR / args.output
    zip_path = build_submit_zip(output_zip)
    size_gb = zip_path.stat().st_size / (1024 ** 3)
    print(f"Created {zip_path} ({size_gb:.2f} GB)")
    if size_gb > MAX_ZIP_SIZE_GB:
        print(f"경고: 제출 파일 용량이 {MAX_ZIP_SIZE_GB}GB를 초과합니다.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
