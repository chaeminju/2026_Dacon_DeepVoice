"""SpecTTTra(sonics-spectttra-gamma-5s)를 우리 음악 manifest(SONICS 재활용:
real=유튜브에서 받은 진짜곡, fake=SONICS fake_songs part_01)로 파인튜닝한다.

DF-Arena-1B 파인튜닝(SSL 백본 고정 + conformer 헤드만 학습)과 달리 SpecTTTra는
전체가 17M 파라미터로 작아서 백본까지 전부 파인튜닝한다. 대신 원본 학습
config(lr=8e-4, batch=256, 50epoch, 대규모 처음부터 학습용)를 그대로 쓰면 이미
수렴된 가중치가 깨질 위험이 크므로, 훨씬 작은 LR과 적은 epoch로 가볍게 얹는다.

모델 자체에 내장된 MixUp/SpecAugment(model.forward(audio, y)가 self.training일 때
자동 적용)를 그대로 재사용해 원본 학습 레시피와 최대한 일치시킨다.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
SPECTTTRA_SRC = BASE_DIR / "submit_spectttra" / "model" / "spectttra"

SAMPLE_RATE = 16_000


def load_model(device, checkpoint_dir=None):
    if str(SPECTTTRA_SRC) not in sys.path:
        sys.path.insert(0, str(SPECTTTRA_SRC))
    from sonics.models.hf_model import HFAudioClassifier

    src = checkpoint_dir or SPECTTTRA_SRC
    model = HFAudioClassifier.from_pretrained(str(src))
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"SpecTTTra params: {n_total:,} (전체 파인튜닝, 백본 고정 안 함)")
    return model


def build_datasets(train_manifest, val_manifest, model_cfg):
    if str(SPECTTTRA_SRC) not in sys.path:
        sys.path.insert(0, str(SPECTTTRA_SRC))
    from sonics.utils.dataset import AudioDataset
    import pandas as pd

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest) if val_manifest else None

    train_ds = AudioDataset(
        filepaths=[str(BASE_DIR / p) for p in train_df["path"]],
        labels=train_df["target"].tolist(),
        num_classes=1,
        normalize="std",
        max_len=model_cfg.audio.max_len,
        random_sampling=True,
        train=True,
    )
    val_ds = None
    if val_df is not None and len(val_df) > 0:
        val_ds = AudioDataset(
            filepaths=[str(BASE_DIR / p) for p in val_df["path"]],
            labels=val_df["target"].tolist(),
            num_classes=1,
            normalize="std",
            max_len=model_cfg.audio.max_len,
            random_sampling=False,
            train=False,
        )
    return train_ds, val_ds


def bce_with_label_smoothing(logits, target, smoothing):
    target = target * (1 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def evaluate(model, val_ds, device, batch_size=16):
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    n_correct, n_total = 0, 0
    for batch in loader:
        audio = batch["audio"].to(device)
        target = batch["target"].to(device)
        logits = model(audio).squeeze(-1)
        pred = (torch.sigmoid(logits) > 0.5).float()
        n_correct += int((pred == target).sum().item())
        n_total += target.numel()
    acc = n_correct / max(1, n_total)
    print(f"  val acc: {acc:.4f} ({n_correct}/{n_total})")
    return acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    resume_from = args.resume if args.resume and Path(args.resume).exists() else None
    if resume_from:
        print(f"resuming from checkpoint: {resume_from}")
    model = load_model(device, checkpoint_dir=resume_from)

    train_ds, val_ds = build_datasets(args.train_manifest, args.val_manifest, model.config)
    print(f"train: {len(train_ds)} | val: {len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    label_smoothing = 0.02

    for epoch in range(args.epochs):
        model.train()
        running_loss, n_correct, n_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            audio = batch["audio"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, mixed_target = model(audio, target)
                logits = logits.squeeze(-1)
                loss = bce_with_label_smoothing(logits, mixed_target, label_smoothing)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                n_correct += int((pred == target).sum().item())
                n_total += target.numel()
            running_loss += loss.item() * target.numel()
            if (step + 1) % 20 == 0:
                pbar.set_postfix(loss=running_loss / n_total, acc=n_correct / n_total)

        print(f"epoch {epoch+1}: loss={running_loss/max(1,n_total):.4f} acc={n_correct/max(1,n_total):.4f}")

        if val_ds is not None:
            val_acc = evaluate(model, val_ds, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(out_dir)
                print(f"  -> new best (val_acc={val_acc:.4f}), saved to {out_dir}")
        else:
            model.save_pretrained(out_dir)

    if val_ds is not None and best_val_acc < 0:
        model.save_pretrained(out_dir)
    print(f"done. best_val_acc={best_val_acc if val_ds is not None else 'n/a'} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--resume", default=None,
        help="이전에 저장된 out-dir 체크포인트에서 이어서 학습(끊겼을 때 재개용)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
