"""submit_spectttra_ft/ (zero-shot DF-Arena + 파인튜닝 SpecTTTra)를 베이스로,
DF-Arena 1B 가중치까지 파인튜닝본(v2, MLAAD 반영)으로 교체해 보이스·뮤직 두
경로 모두 파인튜닝된 최종 제출본을 만든다.

build_submission.py / build_submission_spectttra.py와 동일한 패턴, 이번엔
df_arena_1b + spectttra 둘 다 교체."""

import argparse
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_SUBMIT = BASE_DIR / "submit_spectttra_ft"


def replace_weights(dst_dir, finetuned_dir, weight_patterns):
    finetuned_dir = Path(finetuned_dir)
    for pattern in weight_patterns:
        for f in dst_dir.glob(pattern):
            f.unlink()
    for f in finetuned_dir.iterdir():
        if f.is_file():
            shutil.copy(f, dst_dir / f.name)


def build_submission(df_arena_finetuned_dir, out_dir_name, out_zip):
    staging = BASE_DIR / out_dir_name
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(SOURCE_SUBMIT, staging, ignore=shutil.ignore_patterns("__pycache__"))

    # DF-Arena 1B: 원본(zero-shot) 가중치만 지우고 코드/설정은 유지, v2 파인튜닝
    # 가중치(config.json + model.safetensors, fp16)로 교체.
    df_arena_dst = staging / "model" / "df_arena_1b"
    replace_weights(
        df_arena_dst,
        df_arena_finetuned_dir,
        ("model*.safetensors", "pytorch_model*.bin", "config.json"),
    )

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
    parser.add_argument(
        "--df-arena-finetuned-dir",
        default=str(BASE_DIR / "data" / "processed" / "df_arena_finetuned_v2" / "koen"),
    )
    parser.add_argument("--out-dir-name", default="submit_final_koen_v2")
    parser.add_argument(
        "--out-zip", default=str(BASE_DIR / "submissions" / "submission_final_koen_v2.zip")
    )
    args = parser.parse_args()
    build_submission(args.df_arena_finetuned_dir, args.out_dir_name, args.out_zip)


if __name__ == "__main__":
    main()
