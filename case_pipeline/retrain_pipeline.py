"""증강된 데이터로 voice/music/presence 세 갈래를 각각 재학습하는 파이프라인.

기존 `training/finetune_df_arena.py`(DF-Arena, voice), `training/
finetune_spectttra.py`(SpecTTTra, music)는 전혀 수정하지 않는다 — 이 파일은
그 두 모듈의 Dataset/설정값만 재사용하고, 학습 루프 자체는 "케이스별로 어느
모델만 업데이트할지"를 분리하기 위해 여기서 새로 구성한다(요구사항 5 골격).

세 갈래:
  - voice 갈래 (A~E, F/H의 voice_label): DF-Arena만 업데이트
  - music 갈래 (F/G/H의 music_label): SpecTTTra만 업데이트
  - presence 갈래 (I_bgm_evasion): PANNs 분류 헤드만 업데이트 (HTDemucs 자체
    파인튜닝은 비용이 커 기본 범위에서 제외 — 요구사항의 "최소한 PANNs 분류
    헤드 재학습"에 해당)
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR / "training"))

import case_analysis  # noqa: E402
import finetune_df_arena as df_arena_ft  # noqa: E402  (기존 스크립트, 수정 없이 재사용)
import finetune_spectttra as spectttra_ft  # noqa: E402

SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000

VOICE_TRAINABLE_CASES = set(case_analysis.VOICE_CASE_TYPES) | set(case_analysis.DUAL_LABEL_CASE_TYPES)
MUSIC_TRAINABLE_CASES = set(case_analysis.MUSIC_ONLY_CASE_TYPES) | set(case_analysis.DUAL_LABEL_CASE_TYPES)
PRESENCE_TRAINABLE_CASES = set(case_analysis.PRESENCE_ONLY_CASE_TYPES)


# -----------------------------------------------------------------------------
# 1. 증강 데이터셋 (voice/music 공용 — raw waveform + 메타데이터만 반환하고,
#    실제 세그먼트 크롭/정규화는 각 컴포넌트 학습 루프에서 모델에 맞게 처리)
# -----------------------------------------------------------------------------


class AugmentedCaseDataset(Dataset):
    def __init__(self, records: list[dict], audio_root: str | Path | None = None):
        self.records = records
        self.audio_root = Path(audio_root) if audio_root else BASE_DIR

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        path = self.audio_root / r["path"]
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        return {
            "file_id": r["file_id"],
            "audio": wav.astype(np.float32),
            "case_types": list(r.get("case_types") or []),
            "has_voice": bool(r.get("has_voice", False)),
            "has_music": bool(r.get("has_music", False)),
            "voice_label": r.get("voice_label"),
            "music_label": r.get("music_label"),
        }


def build_case_weighted_sampler(
    records: list[dict], case_weight_overrides: Optional[dict[str, float]] = None
) -> WeightedRandomSampler:
    """case_type별 샘플링 가중치. 기본은 `1/케이스별개수`(희귀 케이스를 자주
    뽑도록)에 `case_weight_overrides`(예: 약점 케이스는 2.0 등)를 곱한다.
    한 샘플이 여러 case_type에 속하면 그 평균 가중치를 사용."""
    case_counts = Counter()
    for r in records:
        for c in (r.get("case_types") or ["unknown"]):
            case_counts[c] += 1

    overrides = case_weight_overrides or {}
    weights = []
    for r in records:
        cases = r.get("case_types") or ["unknown"]
        per_case = [(1.0 / max(case_counts[c], 1)) * overrides.get(c, 1.0) for c in cases]
        weights.append(sum(per_case) / len(per_case))

    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(weights_t), replacement=True)


def build_augmented_dataloader(
    records: list[dict],
    audio_root: str | Path | None = None,
    sampler: Optional[WeightedRandomSampler] = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = AugmentedCaseDataset(records, audio_root=audio_root)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )


@torch.no_grad()
def _eval_voice_accuracy(voice_model, val_records: list[dict], device, audio_root=None) -> float:
    """PROGRESS.md에 기록된 대로 이 파인튜닝 설정(SSL 백본 고정 + conformer
    헤드만 학습)은 후반 epoch에서 가끔 정확도가 랜덤 수준으로 붕괴하는 경향이
    있다 — 매 epoch마다 이걸로 검증해서 "붕괴 전 최고 시점"만 저장해야 한다."""
    label2id = voice_model.config.label2id
    was_training = voice_model.training
    voice_model.eval()
    audio_root = Path(audio_root) if audio_root else BASE_DIR

    n_correct, n_total = 0, 0
    for r in val_records:
        if r.get("voice_label") not in ("real", "fake"):
            continue
        if not (set(r.get("case_types") or []) & VOICE_TRAINABLE_CASES):
            continue
        wav = load_audio_for_eval(audio_root / r["path"])
        segment = _crop_or_pad(wav, df_arena_ft.SEGMENT_SAMPLES)
        segment_t = torch.from_numpy(segment).to(device)
        label_key = "bonafide" if r["voice_label"] == "real" else "spoof"
        logits = voice_model(input_values=segment_t)["logits"]
        n_correct += int(logits.argmax(-1).item() == label2id[label_key])
        n_total += 1

    if was_training:
        voice_model.train()
        if hasattr(voice_model, "backbone") and hasattr(voice_model.backbone, "ssl_model"):
            voice_model.backbone.ssl_model.eval()
    return n_correct / max(1, n_total)


@torch.no_grad()
def _eval_music_accuracy(music_model, val_records: list[dict], device, audio_root=None) -> float:
    was_training = music_model.training
    music_model.eval()
    max_len = music_model.config.audio.max_len
    audio_root = Path(audio_root) if audio_root else BASE_DIR

    n_correct, n_total = 0, 0
    for r in val_records:
        if r.get("music_label") not in ("real", "fake"):
            continue
        if not (set(r.get("case_types") or []) & MUSIC_TRAINABLE_CASES):
            continue
        wav = load_audio_for_eval(audio_root / r["path"])
        segment = _crop_or_pad(wav, max_len)
        audio_t = torch.from_numpy(segment).unsqueeze(0).to(device)
        logits = music_model(audio_t)  # eval 모드에서는 target 없이 호출(MixUp 비활성)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = int(torch.sigmoid(logits.squeeze()).item() > 0.5)
        n_correct += int(pred == (1 if r["music_label"] == "fake" else 0))
        n_total += 1

    if was_training:
        music_model.train()
    return n_correct / max(1, n_total)


def load_audio_for_eval(path) -> np.ndarray:
    import librosa

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    return wav.astype(np.float32)


def _crop_or_pad(wav: np.ndarray, n_samples: int) -> np.ndarray:
    if wav.size < n_samples:
        reps = n_samples // wav.size + 1
        wav = np.tile(wav, reps)
    if wav.size > n_samples:
        start = random.randint(0, wav.size - n_samples)
        wav = wav[start : start + n_samples]
    return wav.astype(np.float32)


# -----------------------------------------------------------------------------
# 2. voice 갈래 학습 (DF-Arena만 업데이트) — A~E, F/H(voice_label)
# -----------------------------------------------------------------------------


def train_voice_component(
    voice_model,
    augmented_dataloader: DataLoader,
    device,
    epochs: int = 3,
    lr: float = 1e-4,
    accum_steps: int = 8,
    val_records: list[dict] | None = None,
    out_dir: str | Path | None = None,
    audio_root=None,
) -> float:
    """`training/finetune_df_arena.py`의 학습 루프(세그먼트 크롭 + SSL 백본
    고정 + conformer 헤드만 학습)와 같은 관례를 따르되, music 라벨만 있는
    샘플(G 등)은 자동으로 건너뛰어 DF-Arena 파라미터만 갱신되도록 한다.

    `val_records`+`out_dir`을 주면 매 epoch마다 검증 정확도를 재고, 그
    이전보다 좋을 때만 `out_dir`에 저장한다 — 이 설정이 후반 epoch에서 가끔
    랜덤 수준으로 붕괴하는 경향이 있어서(PROGRESS.md 기록) 붕괴 전 최고
    시점을 놓치지 않기 위함(기존 `finetune_df_arena.py`와 동일한 안전장치).
    반환값은 관측된 최고 검증 정확도(val_records 없으면 -1.0)."""
    label2id = voice_model.config.label2id  # {"bonafide": 1, "spoof": 0}
    trainable_params = [p for p in voice_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    voice_model.train()
    if hasattr(voice_model, "backbone") and hasattr(voice_model.backbone, "ssl_model"):
        voice_model.backbone.ssl_model.eval()  # 백본 고정 (기존 파인튜닝 스크립트와 동일)

    best_val_acc = -1.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        step, running_loss, n_correct, n_total = 0, 0.0, 0, 0
        pbar = tqdm(augmented_dataloader, desc=f"[voice] epoch {epoch+1}/{epochs}")
        for sample in pbar:
            if sample["voice_label"] not in ("real", "fake"):
                continue
            if not (set(sample["case_types"]) & VOICE_TRAINABLE_CASES):
                continue

            segment = _crop_or_pad(sample["audio"], df_arena_ft.SEGMENT_SAMPLES)
            segment_t = torch.from_numpy(segment).to(device)
            label_key = "bonafide" if sample["voice_label"] == "real" else "spoof"
            label = torch.tensor([label2id[label_key]], device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = voice_model(input_values=segment_t)["logits"]
                loss = criterion(logits, label) / accum_steps
            scaler.scale(loss).backward()
            step += 1
            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            n_correct += int(logits.argmax(-1).item() == label.item())
            n_total += 1
            if n_total % 50 == 0:
                pbar.set_postfix(loss=running_loss / n_total, acc=n_correct / n_total)
        print(f"[voice] epoch {epoch+1}: n={n_total} loss={running_loss/max(1,n_total):.4f} acc={n_correct/max(1,n_total):.4f}")

        if val_records:
            val_acc = _eval_voice_accuracy(voice_model, val_records, device, audio_root=audio_root)
            print(f"[voice] epoch {epoch+1}: val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if out_dir:
                    voice_model.save_pretrained(out_dir)
                    print(f"  -> new best (val_acc={val_acc:.4f}), saved to {out_dir}")
        elif out_dir and epoch == epochs - 1:
            voice_model.save_pretrained(out_dir)  # 검증셋이 없으면 마지막 epoch을 그대로 저장

    return best_val_acc


# -----------------------------------------------------------------------------
# 3. music 갈래 학습 (SpecTTTra만 업데이트) — G, F/H(music_label)
# -----------------------------------------------------------------------------


def train_music_component(
    music_model,
    augmented_dataloader: DataLoader,
    device,
    epochs: int = 3,
    lr: float = 3e-5,
    val_records: list[dict] | None = None,
    out_dir: str | Path | None = None,
    audio_root=None,
) -> float:
    """`training/finetune_spectttra.py`와 동일하게 모델에 내장된 MixUp을 통해
    `model(audio, target)`을 호출한다(백본까지 전체 파인튜닝, 17M 파라미터라
    가능). `val_records`+`out_dir`을 주면 voice와 동일하게 매 epoch 검증 후
    최고 성능 시점만 저장한다."""
    max_len = music_model.config.audio.max_len
    optimizer = torch.optim.AdamW(music_model.parameters(), lr=lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    music_model.train()
    best_val_acc = -1.0
    for epoch in range(epochs):
        running_loss, n_correct, n_total = 0.0, 0, 0
        pbar = tqdm(augmented_dataloader, desc=f"[music] epoch {epoch+1}/{epochs}")
        for sample in pbar:
            if sample["music_label"] not in ("real", "fake"):
                continue
            if not (set(sample["case_types"]) & MUSIC_TRAINABLE_CASES):
                continue

            segment = _crop_or_pad(sample["audio"], max_len)
            audio_t = torch.from_numpy(segment).unsqueeze(0).to(device)
            # sonics의 MixUp/SpecAugment 레이어가 target.ndim==1(배치 차원만)을
            # 요구한다 — (batch,) 형태로 맞춰야 함, (batch,1)이면 에러남.
            target = torch.tensor(
                [1.0 if sample["music_label"] == "fake" else 0.0], device=device
            )

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, mixed_target = music_model(audio_t, target)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), mixed_target
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(music_model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                n_correct += int((pred == target).sum().item())
                n_total += target.numel()
            running_loss += loss.item() * target.numel()
            if n_total and n_total % 50 == 0:
                pbar.set_postfix(loss=running_loss / n_total, acc=n_correct / n_total)
        print(f"[music] epoch {epoch+1}: n={n_total} loss={running_loss/max(1,n_total):.4f} acc={n_correct/max(1,n_total):.4f}")

        if val_records:
            val_acc = _eval_music_accuracy(music_model, val_records, device, audio_root=audio_root)
            print(f"[music] epoch {epoch+1}: val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if out_dir:
                    music_model.save_pretrained(out_dir)
                    print(f"  -> new best (val_acc={val_acc:.4f}), saved to {out_dir}")
        elif out_dir and epoch == epochs - 1:
            music_model.save_pretrained(out_dir)

    return best_val_acc


# -----------------------------------------------------------------------------
# 4. presence 갈래 학습 (PANNs 분류 헤드만 업데이트) — I_bgm_evasion 전용 루틴
# -----------------------------------------------------------------------------


def _resample_for_panns(audio_16k: np.ndarray) -> torch.Tensor:
    audio_32k = librosa.resample(audio_16k, orig_sr=SAMPLE_RATE, target_sr=PANNS_SAMPLE_RATE)
    return torch.from_numpy(audio_32k.astype(np.float32)).unsqueeze(0)


def finetune_presence_head(
    panns_model: "case_analysis.PANNsPresenceModel",
    records: list[dict],
    htdemucs_model=None,
    device=None,
    epochs: int = 3,
    lr: float = 1e-4,
    audio_root: str | Path | None = None,
) -> None:
    """I_bgm_evasion 데이터로 PANNs(Cnn14)의 `fc_audioset` 분류 헤드만 재학습한다.
    HTDemucs 자체 파인튜닝은 비용이 커 기본 범위에서 다루지 않음(요구사항의
    "최소한 PANNs 분류 헤드 재학습"에 해당) — `htdemucs_model`을 넘기면 실제
    추론과 동일하게 분리된 voice_audio/music_audio로 학습해 분포를 맞추고,
    넘기지 않으면 원본 오디오를 그대로 사용한다(간이 버전).
    """
    # panns_model.model은 panns_inference.AudioTagging 래퍼 객체이고, 실제
    # 학습 가능한 nn.Module(Cnn14, fc_audioset 포함)은 그 안의 .model 속성.
    model = panns_model.raw_model
    device = device or next(model.parameters()).device

    for name, p in model.named_parameters():
        p.requires_grad = "fc_audioset" in name
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("fc_audioset 레이어를 찾지 못함 — PANNs 모델 구조를 확인하세요")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    bce = nn.BCELoss()  # Cnn14.forward가 이미 sigmoid를 적용한 clipwise_output을 반환함

    i_records = [r for r in records if PRESENCE_TRAINABLE_CASES & set(r.get("case_types") or [])]
    if not i_records:
        print("[warn] I_bgm_evasion 레코드가 없어 finetune_presence_head를 스킵함")
        return
    print(f"[presence] I_bgm_evasion {len(i_records)}개 샘플로 PANNs 헤드 재학습")

    audio_root = Path(audio_root) if audio_root else BASE_DIR
    model.train()
    for epoch in range(epochs):
        running_loss, n = 0.0, 0
        for r in tqdm(i_records, desc=f"[presence] epoch {epoch+1}/{epochs}"):
            path = audio_root / r["path"]
            if htdemucs_model is not None:
                voice_audio, music_audio = htdemucs_model.separate(path)
            else:
                wav, sr = sf.read(str(path), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != SAMPLE_RATE:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
                voice_audio = music_audio = wav.astype(np.float32)

            optimizer.zero_grad()
            loss_total = 0.0
            for audio, indices, target_val in (
                (voice_audio, panns_model.voice_indices, float(r.get("has_voice", False))),
                (music_audio, panns_model.music_indices, float(r.get("has_music", False))),
            ):
                audio_t = _resample_for_panns(audio).to(device)
                out = model(audio_t, None)["clipwise_output"]  # [1, 527]
                pooled = out[:, indices].max(dim=1).values  # predict_presence와 동일한 max-pool
                target_t = torch.tensor([target_val], device=device)
                loss = bce(pooled, target_t)
                loss.backward()
                loss_total += loss.item()
            optimizer.step()
            running_loss += loss_total
            n += 1
        print(f"[presence] epoch {epoch+1}: n={n} loss={running_loss/max(1,n):.4f}")
    model.eval()


# -----------------------------------------------------------------------------
# 5. 오케스트레이터
# -----------------------------------------------------------------------------


def retrain_with_augmented_data(
    voice_model,
    music_model,
    augmented_dataloader: DataLoader,
    device=None,
    voice_epochs: int = 3,
    music_epochs: int = 3,
    voice_lr: float = 1e-4,
    music_lr: float = 3e-5,
    accum_steps: int = 8,
    voice_out_dir: str | Path | None = None,
    music_out_dir: str | Path | None = None,
    val_records: list[dict] | None = None,
    audio_root=None,
) -> dict:
    """voice 갈래(DF-Arena)와 music 갈래(SpecTTTra)를 같은 `augmented_dataloader`
    에서 각각 필터링해 분리 학습한다(한쪽 모델 업데이트가 다른 쪽에 영향 없음).
    I_bgm_evasion 데이터는 여기서 다루지 않고 `finetune_presence_head`를 별도로
    호출해야 한다(요구사항: "별도 루틴으로 분리").

    `val_records`(case_metadata 스키마의 val 레코드 리스트)를 주면 매 epoch
    검증 정확도로 최고 시점만 저장한다 — 이 파인튜닝 설정은 후반 epoch에
    가끔 랜덤 수준으로 붕괴하는 경향이 실측됨(PROGRESS.md), 안 주면 마지막
    epoch을 그대로 저장(붕괴 위험 있음, 짧은 스모크 테스트용).
    """
    device = device or next(voice_model.parameters()).device

    print("=== [1/2] voice 컴포넌트 재학습 (DF-Arena, A~E + F/H voice_label) ===")
    best_voice_acc = train_voice_component(
        voice_model, augmented_dataloader, device, epochs=voice_epochs, lr=voice_lr,
        accum_steps=accum_steps, val_records=val_records, out_dir=voice_out_dir, audio_root=audio_root,
    )

    print("=== [2/2] music 컴포넌트 재학습 (SpecTTTra, G + F/H music_label) ===")
    best_music_acc = train_music_component(
        music_model, augmented_dataloader, device, epochs=music_epochs, lr=music_lr,
        val_records=val_records, out_dir=music_out_dir, audio_root=audio_root,
    )

    return {
        "voice_model": voice_model, "music_model": music_model,
        "best_voice_val_acc": best_voice_acc, "best_music_val_acc": best_music_acc,
    }


# -----------------------------------------------------------------------------
# 6. CLI — "평가 -> 증강계획 -> 재학습 -> 평가" 반복의 한 iteration을 실행하는 예시
# -----------------------------------------------------------------------------


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="증강 데이터로 voice/music/presence 재학습")
    parser.add_argument("--train-metadata", required=True, help="load_metadata로 읽을 증강 후 학습셋")
    parser.add_argument("--val-metadata", default=None, help="재평가용(선택) — 주면 재학습 후 evaluate_by_case + 로그 기록")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pipeline-script", default=None)
    parser.add_argument("--voice-epochs", type=int, default=3)
    parser.add_argument("--music-epochs", type=int, default=3)
    parser.add_argument("--voice-lr", type=float, default=1e-4)
    parser.add_argument("--music-lr", type=float, default=3e-5)
    parser.add_argument("--voice-out-dir", default=None)
    parser.add_argument("--music-out-dir", default=None)
    parser.add_argument("--case-weights", default=None, help='{"D_partial_splice": 2.0, ...} JSON (WeightedRandomSampler 가중치)')
    parser.add_argument("--skip-presence", action="store_true", help="I_bgm_evasion PANNs 헤드 재학습 건너뛰기")
    parser.add_argument("--panns-out-path", default=None,
                         help="재학습된 PANNs(Cnn14) state_dict를 저장할 .pth 경로 "
                              "(panns_inference.AudioTagging(checkpoint_path=...)로 다시 로드 가능한 형식)")
    parser.add_argument("--iteration", type=int, default=1, help="experiment_log.csv에 기록할 반복 번호")
    parser.add_argument("--log-path", default="experiment_log.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    records = case_analysis.load_metadata(args.train_metadata)

    case_weights = json.loads(Path(args.case_weights).read_text(encoding="utf-8")) if args.case_weights else None
    sampler = build_case_weighted_sampler(records, case_weights) if case_weights else None
    dataloader = build_augmented_dataloader(records, audio_root=args.audio_root, sampler=sampler)

    val_records = case_analysis.load_metadata(args.val_metadata) if args.val_metadata else None

    print("모델 로딩 중 (DF-Arena, SpecTTTra, PANNs) ...")
    voice_model = df_arena_ft.load_model(device)
    music_model = spectttra_ft.load_model(device)
    panns_model = case_analysis.PANNsPresenceModel(device, script_path=args.pipeline_script)

    retrain_with_augmented_data(
        voice_model, music_model, dataloader, device=device,
        voice_epochs=args.voice_epochs, music_epochs=args.music_epochs,
        voice_lr=args.voice_lr, music_lr=args.music_lr,
        voice_out_dir=args.voice_out_dir, music_out_dir=args.music_out_dir,
        val_records=val_records, audio_root=args.audio_root,
    )

    htdemucs_model = case_analysis.HTDemucsSeparator(device, script_path=args.pipeline_script)

    if not args.skip_presence:
        finetune_presence_head(panns_model, records, htdemucs_model=htdemucs_model, device=device, audio_root=args.audio_root)
        if args.panns_out_path:
            Path(args.panns_out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": panns_model.raw_model.state_dict()}, args.panns_out_path)
            print(f"panns_model 저장 -> {args.panns_out_path}")

    if val_records is not None:
        val_dataloader = case_analysis.build_case_dataloader(val_records, audio_root=args.audio_root)

        # voice_model/music_model은 이제 방금 학습한 순수 nn.Module이므로,
        # case_analysis가 재사용하는 것과 동일한 추론 함수(predict_fake류)를
        # 학습된 가중치로 감싸 어댑터로 만든다.
        pipeline = case_analysis._load_submit_pipeline_module(args.pipeline_script)
        fake_label_index = int(voice_model.config.label2id["spoof"])
        spectttra_config = {
            "segment_samples": int(music_model.config.audio.max_len),
            "sample_rate": int(music_model.config.audio.sample_rate),
            "normalize": bool(music_model.config.audio.normalize),
        }

        class _TrainedVoiceAdapter:
            def predict_fake(self, audio):
                return pipeline.predict_fake(voice_model, fake_label_index, audio, device)

        class _TrainedMusicAdapter:
            def predict_fake(self, audio):
                return pipeline.predict_fake_music(music_model, spectttra_config, audio, device)

        df = case_analysis.evaluate_by_case(
            _TrainedVoiceAdapter(), _TrainedMusicAdapter(), panns_model, htdemucs_model, val_dataloader
        )
        case_analysis.append_experiment_log(args.iteration, df, log_path=args.log_path)


if __name__ == "__main__":
    main()
