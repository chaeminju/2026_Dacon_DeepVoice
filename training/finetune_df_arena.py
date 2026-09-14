"""DF-Arena-1B를 우리 manifest(케이스별 KO/EN/KO+EN)로 파인튜닝한다.

SSL 백본(wav2vec2-xls-r-1b, ~1B 파라미터)은 고정하고 conformer 분류 헤드만
학습한다 — 소규모 데이터셋(수천 개)에서 1B 전체를 흔들면 과적합/불안정 위험이
크고, 우리가 비교하려는 변수는 "데이터 구성"이지 "얼마나 강하게 파인튜닝했는가"가
아니므로 가벼운 헤드 튜닝이 합리적인 기본값이다.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DF_ARENA_SRC = BASE_DIR / "baseline_submit" / "model" / "df_arena_1b"

SEGMENT_SAMPLES = 64_600
SAMPLE_RATE = 16_000


def load_model(device):
    if str(BASE_DIR / "baseline_submit" / "model") not in sys.path:
        sys.path.insert(0, str(BASE_DIR / "baseline_submit" / "model"))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    import os

    prev = Path.cwd()
    import os as _os

    _os.chdir(DF_ARENA_SRC)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_SRC), local_files_only=True, low_cpu_mem_usage=True,
        )
    finally:
        _os.chdir(prev)
    model = model.to(device)

    for p in model.backbone.ssl_model.parameters():
        p.requires_grad = False
    model.backbone.ssl_model.eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,}")
    return model


class SpoofDataset(Dataset):
    def __init__(self, rows, label2id):
        self.rows = rows
        self.label2id = label2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav, sr = sf.read(row["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        if wav.size < SEGMENT_SAMPLES:
            reps = SEGMENT_SAMPLES // wav.size + 1
            wav = np.tile(wav, reps)
        start = 0
        if wav.size > SEGMENT_SAMPLES:
            start = random.randint(0, wav.size - SEGMENT_SAMPLES)
        segment = wav[start:start + SEGMENT_SAMPLES].astype(np.float32)

        label = self.label2id[row["label"]]
        return torch.from_numpy(segment), label


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    label2id = model.config.label2id  # {"bonafide": 1, "spoof": 0}

    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest) if args.val_manifest else []
    print(f"train: {len(train_rows)} rows | val: {len(val_rows)} rows")

    train_ds = SpoofDataset(train_rows, label2id)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    accum_steps = args.accum_steps
    model.train()
    model.backbone.ssl_model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        running_loss = 0.0
        n_correct = 0
        n_total = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, (segment, label) in enumerate(pbar):
            segment = segment[0].to(device)
            label = torch.tensor([label.item()], device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_values=segment)["logits"]
                loss = criterion(logits, label) / accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            n_correct += int(logits.argmax(-1).item() == label.item())
            n_total += 1
            if n_total % 50 == 0:
                pbar.set_postfix(loss=running_loss / n_total, acc=n_correct / n_total)

        print(f"epoch {epoch+1}: loss={running_loss/max(1,n_total):.4f} acc={n_correct/max(1,n_total):.4f}")

        if val_rows:
            val_acc = evaluate(model, val_rows, label2id, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(out_dir)
                print(f"  -> new best (val_acc={val_acc:.4f}), saved to {out_dir}")
        else:
            model.save_pretrained(out_dir)

    if val_rows and best_val_acc < 0:
        model.save_pretrained(out_dir)
    print(f"done. best_val_acc={best_val_acc if val_rows else 'n/a'} -> {out_dir}")


@torch.no_grad()
def evaluate(model, rows, label2id, device):
    model.eval()
    ds = SpoofDataset(rows, label2id)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    n_correct = 0
    for segment, label in loader:
        segment = segment[0].to(device)
        logits = model(input_values=segment)["logits"]
        pred = logits.argmax(-1).item()
        n_correct += int(pred == label.item())
    acc = n_correct / max(1, len(rows))
    print(f"  val acc: {acc:.4f} ({n_correct}/{len(rows)})")
    model.train()
    model.backbone.ssl_model.eval()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum-steps", type=int, default=8)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
