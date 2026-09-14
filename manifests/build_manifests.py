"""증강된 KO/EN raw manifest들을 모아 3케이스(ko/en/koen) manifest를 만든다.
- 케이스별 real:fake = 1:1로 다운샘플링
- 3케이스의 총 개수를 동일하게 맞춤 (가장 작은 케이스 기준)
- train/val = 90/10, label로 stratify
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def balance_real_fake(rows, seed):
    rng = random.Random(seed)
    real = [r for r in rows if r["label"] == "bonafide"]
    fake = [r for r in rows if r["label"] == "spoof"]
    rng.shuffle(real)
    rng.shuffle(fake)
    n = min(len(real), len(fake))
    return real[:n] + fake[:n]


def split_train_val(rows, val_ratio, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_case(name, train_rows, val_rows, out_dir, fieldnames):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
        path = out_dir / f"{name}_{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{name}/{split_name}: {len(split_rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ko-manifests", nargs="+", required=True)
    parser.add_argument("--en-manifests", nargs="+", required=True)
    parser.add_argument("--out-dir", default="manifests")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    ko_rows = []
    for m in args.ko_manifests:
        ko_rows.extend(read_manifest(m))
    en_rows = []
    for m in args.en_manifests:
        en_rows.extend(read_manifest(m))

    fieldnames = ko_rows[0].keys() if ko_rows else en_rows[0].keys()
    fieldnames = list(dict.fromkeys(list(fieldnames) + list(en_rows[0].keys() if en_rows else [])))

    ko_balanced = balance_real_fake(ko_rows, args.seed)
    en_balanced = balance_real_fake(en_rows, args.seed + 1)

    print(f"ko balanced: {len(ko_balanced)} (real={len(ko_balanced)//2} fake={len(ko_balanced)//2})")
    print(f"en balanced: {len(en_balanced)} (real={len(en_balanced)//2} fake={len(en_balanced)//2})")

    # 3케이스 총 개수를 동일하게: 가장 작은 쪽 기준으로 짝수로 맞춤
    target_n = min(len(ko_balanced), len(en_balanced))
    if target_n % 2 == 1:
        target_n -= 1

    rng = random.Random(args.seed + 2)

    def resample_balanced(rows, n):
        real = [r for r in rows if r["label"] == "bonafide"]
        fake = [r for r in rows if r["label"] == "spoof"]
        rng.shuffle(real)
        rng.shuffle(fake)
        half = n // 2
        return real[:half] + fake[:half]

    ko_final = resample_balanced(ko_balanced, target_n)
    en_final = resample_balanced(en_balanced, target_n)

    koen_final = resample_balanced(ko_balanced, target_n // 2) + resample_balanced(
        en_balanced, target_n // 2
    )

    print(f"target size per case: {target_n} (koen: {len(koen_final)})")

    for name, rows in [("ko", ko_final), ("en", en_final), ("koen", koen_final)]:
        train_rows, val_rows = split_train_val(rows, args.val_ratio, args.seed + 3)
        write_case(name, train_rows, val_rows, args.out_dir, fieldnames)


if __name__ == "__main__":
    main()
