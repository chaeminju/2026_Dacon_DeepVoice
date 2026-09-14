"""baseline_submit/ 폴더를 복제해 df_arena_1b 가중치만 파인튜닝된 것으로
교체하고, 제출용 zip으로 묶는다. (이 대회는 코드 제출 방식 — CSV가 아니라
script.py + model/ + requirements.txt 전체를 zip으로 올리면 데이콘 서버가
직접 실행해 채점한다.)
"""

import argparse
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BASELINE_SUBMIT = BASE_DIR / "baseline_submit"


def build_submission(case_name, finetuned_dir, out_zip):
    finetuned_dir = Path(finetuned_dir)
    staging = BASE_DIR / f"submit_{case_name}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(BASELINE_SUBMIT, staging)

    df_arena_dst = staging / "model" / "df_arena_1b"
    # 원본 가중치 파일(pytorch_model.bin 등)을 먼저 지워야 한다 — 안 지우면
    # 파인튜닝 가중치(safetensors)가 옆에 추가만 되어 두 배로 커지고, from_pretrained가
    # 어느 걸 로드할지도 불분명해진다.
    for pattern in ("pytorch_model*.bin", "model*.safetensors", "*.safetensors.index.json", "pytorch_model.bin.index.json"):
        for f in df_arena_dst.glob(pattern):
            f.unlink()

    # 파인튜닝 산출물(config.json + 가중치 파일들)로 교체하되, 나머지 커스텀
    # 코드 파일(backbone.py, conformer.py 등)은 원본 그대로 유지한다.
    for f in finetuned_dir.iterdir():
        if f.is_file():
            shutil.copy(f, df_arena_dst / f.name)

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
    parser.add_argument("--case", required=True, choices=["ko", "en", "koen"])
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument("--out-zip", required=True)
    args = parser.parse_args()
    build_submission(args.case, args.finetuned_dir, args.out_zip)


if __name__ == "__main__":
    main()
