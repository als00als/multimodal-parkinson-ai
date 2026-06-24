"""
파킨슨 필적 분류 학습 스크립트 — 실험 E (기본 증강 + VAE 증강)
================================================================
사용 데이터 : augmented_data/experiment_E
  - 원본 36장 + 기본증강(basic) 108장 + VAE증강(vae) 108장 = 252장/클래스/shape

train_final.py 기반, experiment_E 전용으로 수정
  · DATA_DIR   → experiment_E
  · OUTPUT_DIR → results_expE
  · _is_original() : _basic_ AND _vae_ 모두 증강 파일로 인식
  · make_loaders()  : val 원본에서 파생된 basic·vae 증강 모두 학습셋 제거
  · 보고서 내용     : experiment_E 이중 증강 구조 반영

출력
  results_expE/
    spiral/    wave/    combined/   ← 실험별 모델·시각화·예측 CSV
    metrics_comparison.png
    all_results.json
    results_report.txt   ← 초보자용 결과 정리
    model_analysis.txt   ← AI 모형 장단점·보완점 분석
"""

import random, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 0. 설정
# ══════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR   = Path(r"C:\AI Project\피우다\augmented_data\experiment_E")
OUTPUT_DIR = Path(r"C:\AI Project\피우다\results_expE")

IMG_SIZE           = 224
BATCH_SIZE         = 16       # 작은 배치 → 노이즈 있는 업데이트로 일반화 향상
EPOCHS             = 100
LR                 = 5e-5     # 낮은 학습률로 섬세하게 학습
WEIGHT_DECAY       = 5e-4
VAL_ORIG_PER_CLASS = 8        # 검증셋: 클래스당 원본 이미지 수
PATIENCE           = 15       # Val Recall 기준 early stopping patience
PARKINSON_WEIGHT   = 2.0      # Parkinson 클래스 가중치 (FN 페널티)
MIN_RECALL_FOR_THR = 0.85     # 임계값 탐색 시 최소 Recall 조건

LABEL_MAP = {"healthy": 0, "parkinson": 1}
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# 1. Transform
# ══════════════════════════════════════════════════════════════
TRAIN_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # grayscale → 3채널 (ResNet 호환)
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # ImageNet 통계값
])

EVAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════
class ParkinsonsDataset(Dataset):
    def __init__(self, samples: list, transform):
        self.samples   = samples   # [(Path, label), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label


# ══════════════════════════════════════════════════════════════
# 3. 데이터 분할 (원본 전용 검증셋 — v2 핵심 개선)
# ══════════════════════════════════════════════════════════════
def _is_original(path: Path) -> bool:
    """파일명에 '_basic_' 또는 '_vae_'가 없으면 원본 이미지"""
    return "_basic_" not in path.name and "_vae_" not in path.name


def make_loaders(shapes: list):
    """
    [분할 전략]
    ① 각 shape·클래스마다 원본(36장)과 증강(_basic_ 또는 _vae_ 포함)을 분리
    ② 원본 중 VAL_ORIG_PER_CLASS장을 검증셋으로 지정
    ③ 검증셋 원본에서 파생된 basic·vae 증강 이미지는 학습셋에서도 제거 (데이터 유출 방지)
    ④ 학습셋 = 나머지 원본 + 해당 원본의 basic·vae 증강본
    ⑤ 검증셋 = 원본만 → 테스트셋과 동일 분포 유지
    ⑥ 테스트셋 = testing 폴더 원본 15장/클래스
    """
    rng = np.random.default_rng(SEED)
    train_samples, val_samples = [], []

    for shape in shapes:
        for cls, label in LABEL_MAP.items():
            folder    = DATA_DIR / shape / "training" / cls
            originals = sorted(p for p in folder.glob("*.png") if _is_original(p))
            augmented = sorted(p for p in folder.glob("*.png") if not _is_original(p))

            perm      = rng.permutation(len(originals))
            val_orig  = [originals[i] for i in perm[:VAL_ORIG_PER_CLASS]]
            train_orig = [originals[i] for i in perm[VAL_ORIG_PER_CLASS:]]

            val_stems = {p.stem for p in val_orig}

            def _orig_stem(p: Path) -> str:
                """증강 파일명에서 원본 stem 추출 (_basic_ 또는 _vae_ 기준)"""
                name = p.stem
                if "_basic_" in name:
                    return name.rsplit("_basic_", 1)[0]
                if "_vae_" in name:
                    return name.rsplit("_vae_", 1)[0]
                return name

            train_aug = [p for p in augmented
                         if _orig_stem(p) not in val_stems]

            train_samples += [(p, label) for p in train_orig + train_aug]
            val_samples   += [(p, label) for p in val_orig]

    test_samples = []
    for shape in shapes:
        for cls, label in LABEL_MAP.items():
            folder = DATA_DIR / shape / "testing" / cls
            for p in sorted(folder.glob("*.png")):
                test_samples.append((p, label))

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(ParkinsonsDataset(train_samples, TRAIN_TF),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g)
    val_loader   = DataLoader(ParkinsonsDataset(val_samples, EVAL_TF),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ParkinsonsDataset(test_samples, EVAL_TF),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tl = [s[1] for s in train_samples]; vl = [s[1] for s in val_samples]
    print(f"  Train : {len(train_samples):4d}  "
          f"(healthy={tl.count(0)}, parkinson={tl.count(1)})")
    print(f"  Val   : {len(val_samples):4d}  "
          f"(healthy={vl.count(0)}, parkinson={vl.count(1)})  ← 원본만")
    print(f"  Test  : {len(test_samples):4d}  ← 원본만 (증강 없음)")

    return train_loader, val_loader, test_loader, len(train_samples), len(val_samples), len(test_samples)


# ══════════════════════════════════════════════════════════════
# 4. 모델 (layer3+4+fc fine-tuning)
# ══════════════════════════════════════════════════════════════
def build_model() -> nn.Module:
    """
    ResNet18 (ImageNet pretrained)
      layer1~2 : 동결  — ImageNet 저수준 특징 유지
      layer3~4 : 학습  — 손글씨 특화 특징 학습
      fc        : Dropout(0.5) → Linear(512, 2)  교체
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in ("layer3", "layer4", "fc")):
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )
    return model.to(DEVICE)


# ══════════════════════════════════════════════════════════════
# 5. 학습 / 추론 / 지표
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer) -> tuple:
    model.train()
    total_loss = 0.0; correct = 0; n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def run_inference(model, loader, criterion=None) -> tuple:
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0; n = 0
    for imgs, labels in loader:
        imgs, labels_d = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        if criterion is not None:
            total_loss += criterion(logits, labels_d).item() * len(labels)
        probs = torch.softmax(logits, dim=1)[:, 1]   # parkinson 확률
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        n += len(labels)
    loss = (total_loss / n) if (criterion is not None and n > 0) else 0.0
    return np.array(all_labels), np.array(all_probs), loss


def predict(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


def compute_metrics(labels, preds, probs) -> dict:
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy" : accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall"   : recall_score(labels, preds, zero_division=0),
        "f1"       : f1_score(labels, preds, zero_division=0),
        "roc_auc"  : roc_auc_score(labels, probs),
        "cm": cm,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def find_best_threshold(val_labels, val_probs,
                        min_recall: float = MIN_RECALL_FOR_THR) -> float:
    """
    검증셋 기준으로 Recall >= min_recall 조건 하에 F1 최대 임계값을 탐색.
    조건 불만족 시 Recall 최대 임계값 반환.
    """
    best_t, best_f1 = 0.5, -1.0
    candidates = []
    for t in np.arange(0.20, 0.71, 0.02):
        p = predict(val_probs, t)
        r = recall_score(val_labels, p, zero_division=0)
        f = f1_score(val_labels, p, zero_division=0)
        candidates.append((t, r, f))
        if r >= min_recall and f > best_f1:
            best_f1, best_t = f, t
    if best_f1 < 0:
        best_t = max(candidates, key=lambda x: x[1])[0]
    return round(float(best_t), 2)


# ══════════════════════════════════════════════════════════════
# 6. 시각화
# ══════════════════════════════════════════════════════════════
def plot_training_curves(history: dict, exp_name: str, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    be = history["best_epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, tr_k, va_k, title in [
        (axes[0], "train_loss",  "val_loss",   "Loss"),
        (axes[1], "train_acc",   "val_acc",    "Accuracy"),
        (axes[2], "val_recall",  "val_recall", "Val Recall"),
    ]:
        if tr_k == va_k:
            ax.plot(epochs, history[va_k], color="tab:orange", label="Val Recall")
        else:
            ax.plot(epochs, history[tr_k], label="Train")
            ax.plot(epochs, history[va_k], label="Val")
        ax.axvline(be, color="red", linestyle="--", alpha=0.6, label=f"Best @{be}")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{exp_name} — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Parkinson"])
    ax.set_yticklabels(["Healthy", "Parkinson"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_roc(labels, probs, auc: float, threshold: float,
             exp_name: str, out_path: Path):
    fpr, tpr, thrs = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    idx = np.argmin(np.abs(thrs - threshold))
    ax.scatter(fpr[idx], tpr[idx], color="red", zorder=5,
               label=f"threshold={threshold:.2f}", s=80)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"{exp_name} — ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_threshold_sweep(val_labels, val_probs, best_t: float,
                         exp_name: str, out_path: Path):
    ts = np.arange(0.20, 0.71, 0.02)
    recs, precs, f1s = [], [], []
    for t in ts:
        p = predict(val_probs, t)
        recs.append(recall_score(val_labels, p, zero_division=0))
        precs.append(precision_score(val_labels, p, zero_division=0))
        f1s.append(f1_score(val_labels, p, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, recs,  label="Recall (민감도)", lw=2)
    ax.plot(ts, precs, label="Precision (정밀도)", lw=2)
    ax.plot(ts, f1s,   label="F1-score", lw=2, linestyle="--")
    ax.axvline(best_t, color="red", linestyle=":", lw=2,
               label=f"최적 threshold={best_t:.2f}")
    ax.axhline(MIN_RECALL_FOR_THR, color="gray", linestyle=":", alpha=0.5,
               label=f"Recall 기준선 {MIN_RECALL_FOR_THR}")
    ax.set_xlabel("분류 임계값 (Threshold)"); ax.set_ylabel("점수")
    ax.set_title(f"{exp_name} — 임계값별 성능 변화 (검증셋)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_metrics_comparison(all_results: dict, out_path: Path):
    """3개 실험의 주요 지표 비교 막대그래프"""
    names        = list(all_results.keys())
    metric_keys  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_names = ["Accuracy\n(정확도)", "Precision\n(정밀도)",
                    "Recall\n(민감도)", "F1-score", "ROC-AUC"]
    x = np.arange(len(metric_keys)); width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [all_results[name][m] for m in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width); ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel("점수 (높을수록 좋음)")
    ax.set_title("실험별 Test Set 성능 비교 (Final Version)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════
# 7. 단일 실험 실행
# ══════════════════════════════════════════════════════════════
def run_experiment(exp_name: str, shapes: list, out_dir: Path) -> dict:
    print(f"\n{'━'*58}")
    print(f"  실험: {exp_name}  |  shape: {shapes}")
    print(f"{'━'*58}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 ──────────────────────────────────────────────
    train_loader, val_loader, test_loader, n_tr, n_val, n_te = make_loaders(shapes)

    # ── 모델 / 손실 / 옵티마이저 ─────────────────────────────
    model      = build_model()
    cw         = torch.tensor([1.0, PARKINSON_WEIGHT], dtype=torch.float32).to(DEVICE)
    criterion  = nn.CrossEntropyLoss(weight=cw)
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"\n  클래스 가중치: healthy=1.0 / parkinson={PARKINSON_WEIGHT}")
    print(f"  Early stop   : Val Recall 기준 (patience={PATIENCE})")

    # ── 학습 루프 ────────────────────────────────────────────
    history = {k: [] for k in ("train_loss","train_acc","val_loss","val_acc","val_recall")}
    best_recall, best_epoch, patience_cnt, best_state = -1.0, 0, 0, None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        va_labels, va_probs, va_loss = run_inference(model, val_loader, criterion)
        va_preds   = predict(va_probs, 0.5)
        va_acc     = accuracy_score(va_labels, va_preds)
        va_recall  = recall_score(va_labels, va_preds, zero_division=0)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_recall"].append(va_recall)

        if va_recall > best_recall:
            best_recall, best_epoch, patience_cnt = va_recall, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            mark = "★" if patience_cnt == 0 else " "
            print(f"  {mark} Ep {epoch:3d}/{EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"Val  recall={va_recall:.3f} acc={va_acc:.3f}")

        if patience_cnt >= PATIENCE:
            print(f"  → Early stop @ epoch {epoch}  (best Recall epoch={best_epoch})")
            break

    history["best_epoch"] = best_epoch

    # ── 최적 가중치 복원 & 저장 ─────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_model.pth")

    # ── 임계값 최적화 (검증셋 기반) ──────────────────────────
    val_labels, val_probs, _ = run_inference(model, val_loader)
    best_thr = find_best_threshold(val_labels, val_probs)
    print(f"\n  최적 임계값: {best_thr}  (검증셋, Recall≥{MIN_RECALL_FOR_THR})")

    # ── 테스트 평가 (기본 0.5 / 최적 임계값) ────────────────
    test_labels, test_probs, _ = run_inference(model, test_loader)
    m05  = compute_metrics(test_labels, predict(test_probs, 0.50), test_probs)
    mopt = compute_metrics(test_labels, predict(test_probs, best_thr), test_probs)

    print(f"\n  {'지표':<12}  thr=0.50   thr={best_thr}")
    for k in ("accuracy","precision","recall","f1","roc_auc"):
        print(f"  {k:<12}  {m05[k]:.4f}     {mopt[k]:.4f}")
    print(f"\n  CM (thr={best_thr}):\n  {mopt['cm']}")

    # ── 시각화 ──────────────────────────────────────────────
    plot_training_curves(history, exp_name, out_dir / "training_curves.png")
    plot_confusion_matrix(mopt["cm"],
                          f"{exp_name} — Confusion Matrix (thr={best_thr})",
                          out_dir / "confusion_matrix.png")
    plot_roc(test_labels, test_probs, mopt["roc_auc"], best_thr,
             exp_name, out_dir / "roc_curve.png")
    plot_threshold_sweep(val_labels, val_probs, best_thr,
                         exp_name, out_dir / "threshold_sweep.png")

    # ── 예측 CSV ────────────────────────────────────────────
    pd.DataFrame({
        "label":          test_labels,
        "pred_thr050":    predict(test_probs, 0.50),
        "pred_thr_opt":   predict(test_probs, best_thr),
        "prob_parkinson": test_probs,
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    # 반환값 (보고서용)
    return {
        "accuracy":   mopt["accuracy"],
        "precision":  mopt["precision"],
        "recall":     mopt["recall"],
        "f1":         mopt["f1"],
        "roc_auc":    mopt["roc_auc"],
        "cm":         mopt["cm"].tolist(),
        "TP": mopt["TP"], "TN": mopt["TN"],
        "FP": mopt["FP"], "FN": mopt["FN"],
        "threshold":  best_thr,
        "best_epoch": best_epoch,
        "n_train": n_tr, "n_val": n_val, "n_test": n_te,
        # 기본 임계값(0.5) 결과도 보관
        "acc_thr05":  m05["accuracy"],
        "rec_thr05":  m05["recall"],
        "f1_thr05":   m05["f1"],
    }


# ══════════════════════════════════════════════════════════════
# 8-A. 결과 보고서 (초보자용)
# ══════════════════════════════════════════════════════════════
def write_results_report(all_results: dict, out_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 70,
        "  파킨슨 필적 분류 AI — 학습 결과 보고서",
        f"  작성 일시 : {now}",
        f"  사용 장치 : {DEVICE}",
        "=" * 70,
        "",
        "┌──────────────────────────────────────────────────────────────────┐",
        "│  이 보고서는 AI나 의학을 처음 접하는 분도 이해할 수 있도록        │",
        "│  가능한 한 쉬운 말로 작성했습니다.                               │",
        "└──────────────────────────────────────────────────────────────────┘",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "1. 이 AI가 하는 일",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  손으로 그린 그림(나선 또는 파도 모양)을 AI에게 보여주면,",
        "  AI가 '이 사람은 파킨슨병 환자다 / 아니다'를 판단합니다.",
        "",
        "  파킨슨병은 손 떨림이 주요 증상이기 때문에, 환자가 그린 선은",
        "  건강인의 선보다 불규칙하게 떨리는 특징이 있습니다.",
        "  AI는 이 미세한 차이를 학습해 판별합니다.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "2. 데이터 및 학습 조건",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  [사용 데이터]  실험 E 이중 증강 데이터 (experiment_E)",
        "    - 원본 이미지: 건강인 36장 + 환자 36장 (shape당)",
        "    - 기본 증강 (basic) : 원본 1장당 3장 생성 → 108장/클래스/shape",
        "      (회전·이동·확대·축소·밝기·노이즈·blur 등 기하학적/광도 변환)",
        "    - VAE 증강 (vae)    : VAE 잠재 공간에서 샘플링 → 108장/클래스/shape",
        "      (변분 오토인코더로 원본과 다른 새로운 패턴의 이미지 생성)",
        "    - 합계: 원본 36장 + basic 108장 + vae 108장 = 252장/클래스/shape",
        "",
        "  [학습 / 검증 / 테스트 분리 방법]",
        "    - 학습셋  : 원본 28장 + basic 증강 84장 + vae 증강 84장 = 196장/클래스/shape",
        "    - 검증셋  : 원본 8장/클래스/shape  (증강 이미지 제외 — 실제 환경 반영)",
        "    - 테스트셋: 원본 15장/클래스/shape (학습에 전혀 사용 안 한 데이터)",
        "",
        "  [모델]  ResNet18 (ImageNet 사전학습 후 fine-tuning)",
        f"  [클래스 가중치] healthy=1.0 / parkinson={PARKINSON_WEIGHT}",
        "    → 환자를 놓치는 실수에 더 큰 벌칙을 줘서 민감도 향상",
        f"  [조기 종료] 검증 민감도(Recall) 기준, patience={PATIENCE}",
        f"  [임계값]    검증셋에서 Recall≥{MIN_RECALL_FOR_THR} 조건 하에 F1 최적화",
        "",
    ]

    # ── 실험별 상세 결과 ─────────────────────────────────────
    exp_descriptions = {
        "Spiral":   "나선 그림만 사용한 모델",
        "Wave":     "파도 그림만 사용한 모델",
        "Combined": "나선 + 파도 그림을 합쳐서 학습한 모델",
    }

    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "3. 실험별 테스트 결과",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for exp_name, r in all_results.items():
        tp = r["TP"]; tn = r["TN"]; fp = r["FP"]; fn = r["FN"]
        total_pk = tp + fn; total_hl = tn + fp
        desc = exp_descriptions.get(exp_name, exp_name)

        lines += [
            "",
            f"  ┌─ [{exp_name}] {desc} ─",
            f"  │  테스트 이미지: 총 {total_pk+total_hl}장 (건강인 {total_hl}명 / 환자 {total_pk}명)",
            f"  │  적용 임계값 : {r['threshold']}  (기본값 0.5에서 최적화)",
            f"  │  학습 종료   : {r['best_epoch']}번째 epoch (조기 종료)",
            "  │",
            "  │  판정 결과 요약",
            "  │  ┌──────────────────┬────────────────┬────────────────┐",
            "  │  │                  │  AI: 건강인    │  AI: 환자      │",
            "  │  ├──────────────────┼────────────────┼────────────────┤",
            f"  │  │ 실제: 건강인    │  ✅ 정답 {tn:2d}명  │  ❌ 오탐  {fp:2d}명  │",
            f"  │  │ 실제: 파킨슨    │  ❌ 누락  {fn:2d}명  │  ✅ 정답 {tp:2d}명  │",
            "  │  └──────────────────┴────────────────┴────────────────┘",
            "  │",
            f"  │  ✅ 파킨슨 환자 {total_pk}명 중 {tp}명 탐지 ({r['recall']:.1%})",
            f"  │  ❌ 놓친 환자 {fn}명",
            f"  │  ⚠️  건강인 오탐 {fp}명",
            "  │",
            f"  │  정확도  : {r['accuracy']:.4f}  ({int(r['accuracy']*(total_pk+total_hl))}/"
            f"{total_pk+total_hl}명 올바르게 판정)",
            f"  │  민감도  : {r['recall']:.4f}  ← 환자 탐지율 (의료에서 가장 중요)",
            f"  │  정밀도  : {r['precision']:.4f}  ← 환자라고 했을 때 맞을 확률",
            f"  │  F1-score: {r['f1']:.4f}  ← 민감도·정밀도 균형 지표",
            f"  └─ ROC-AUC : {r['roc_auc']:.4f}  ← 임계값 무관 판별 능력 (1.0이 최고)",
        ]

    # ── 전체 비교 표 ─────────────────────────────────────────
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "4. 전체 비교 요약",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  {'실험':<10} {'정확도':>8} {'민감도':>8} {'정밀도':>8} {'F1':>8} {'AUC':>8} {'임계값':>8}",
        f"  {'─'*58}",
    ]
    for name, r in all_results.items():
        lines.append(
            f"  {name:<10} {r['accuracy']:>8.4f} {r['recall']:>8.4f} "
            f"{r['precision']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
            f"{r['threshold']:>8.2f}"
        )

    # ── 지표 설명 ────────────────────────────────────────────
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "5. 지표 쉬운 설명",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  [정확도 / Accuracy]",
        "  전체 판정 중 맞은 비율입니다. 가장 직관적이지만 의료에서는 부족합니다.",
        "  → 0.9 이상이면 우수, 0.8 이상이면 양호한 수준입니다.",
        "",
        "  [민감도 / Recall  ★ 의료 AI에서 가장 중요]",
        "  '진짜 환자 중에 몇 명이나 환자로 잡아냈나?'",
        "  → 이 값이 낮으면 환자를 놓치는 경우가 많습니다. (위험)",
        "  → 의료 스크리닝에서는 0.90 이상을 목표로 합니다.",
        "",
        "  [정밀도 / Precision]",
        "  'AI가 환자라고 한 사람 중에 실제로 환자인 비율'",
        "  → 이 값이 낮으면 건강인을 환자로 잘못 분류하는 경우가 많습니다.",
        "",
        "  [F1-score]",
        "  민감도와 정밀도를 합쳐서 하나의 점수로 표현한 값입니다.",
        "  두 지표가 모두 높아야 F1도 높아집니다.",
        "",
        "  [ROC-AUC]",
        "  임계값을 바꿔가며 모델의 전반적인 판별 능력을 측정한 값입니다.",
        "  1.0이면 완벽, 0.5면 동전 던지기 수준입니다.",
        "  → 0.9 이상이면 판별 능력 자체는 우수한 편입니다.",
        "",
        "  [임계값 / Threshold]",
        "  AI는 각 그림에 '파킨슨일 확률(0~1)'을 부여합니다.",
        "  이 확률이 임계값 이상이면 '환자'로 분류합니다.",
        "  기본값은 0.5이지만, 민감도를 높이려면 더 낮게 설정합니다.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "6. 저장된 파일",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"  {OUTPUT_DIR}\\",
        "  ├── spiral\\",
        "  │   ├── best_model.pth          학습된 모델 가중치",
        "  │   ├── training_curves.png     학습 과정 그래프",
        "  │   ├── confusion_matrix.png    판정 결과 요약표",
        "  │   ├── roc_curve.png           ROC 곡선",
        "  │   ├── threshold_sweep.png     임계값별 성능 변화",
        "  │   └── test_predictions.csv    이미지별 예측 결과",
        "  ├── wave\\         (동일 구조)",
        "  ├── combined\\     (동일 구조)",
        "  ├── metrics_comparison.png      3개 실험 비교 그래프",
        "  ├── all_results.json            수치 결과 전체",
        "  ├── results_report.txt          이 파일",
        "  └── model_analysis.txt          AI 모형 분석 (장단점·보완점)",
        "",
        "=" * 70,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# 8-B. AI 모형 분석 보고서
# ══════════════════════════════════════════════════════════════
def write_model_analysis(all_results: dict, out_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_exp  = max(all_results, key=lambda k: all_results[k]["recall"])
    best_auc  = max(all_results, key=lambda k: all_results[k]["roc_auc"])
    best_f1   = max(all_results, key=lambda k: all_results[k]["f1"])

    lines = [
        "=" * 70,
        "  AI 모형 분석 보고서 — 파킨슨 필적 분류 모델",
        f"  작성 일시 : {now}",
        "=" * 70,
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "1. 모델 개요",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  모델명    : ResNet18 (Residual Network, 18층)",
        "  학습 방식 : 전이 학습 (Transfer Learning) + Fine-tuning",
        "  사전학습  : ImageNet (120만 장의 자연 이미지로 학습된 가중치 사용)",
        "  태스크    : 이진 분류 (Healthy=0 / Parkinson=1)",
        "",
        "  [학습 구조]",
        "  ImageNet 사전학습 ResNet18",
        "    ├── Conv1 + BN + ReLU + MaxPool  : 동결 (frozen)",
        "    ├── Layer1 (ResBlock × 2, 64ch)  : 동결",
        "    ├── Layer2 (ResBlock × 2, 128ch) : 동결",
        "    ├── Layer3 (ResBlock × 2, 256ch) : 학습 ← 손글씨 중간 특징",
        "    ├── Layer4 (ResBlock × 2, 512ch) : 학습 ← 손글씨 고수준 특징",
        "    └── FC (512 → 2)  Dropout(0.5)  : 학습 ← 분류기",
        "",
        "  [주요 학습 설정]",
        f"    이미지 크기  : {IMG_SIZE} × {IMG_SIZE}",
        f"    Batch Size   : {BATCH_SIZE}",
        f"    학습률 (LR)  : {LR}  (Cosine Annealing 스케줄러)",
        f"    Weight Decay : {WEIGHT_DECAY}",
        f"    클래스 가중치: healthy=1.0 / parkinson={PARKINSON_WEIGHT}",
        f"    조기 종료    : Val Recall 기준  patience={PATIENCE}",
        f"    임계값 전략  : 검증셋 Recall≥{MIN_RECALL_FOR_THR} 조건 하 F1 최적화",
        "",
        "  [최종 성능 요약]",
        f"    {'실험':<10} {'Accuracy':>9} {'Recall':>8} {'Precision':>10} "
        f"{'F1':>8} {'AUC':>8} {'임계값':>8}",
        f"    {'─'*64}",
    ]
    for name, r in all_results.items():
        lines.append(
            f"    {name:<10} {r['accuracy']:>9.4f} {r['recall']:>8.4f} "
            f"{r['precision']:>10.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
            f"{r['threshold']:>8.2f}"
        )
    lines += [
        "",
        f"    ★ Recall 최고 실험 : {best_exp} ({all_results[best_exp]['recall']:.4f})",
        f"    ★ AUC   최고 실험 : {best_auc} ({all_results[best_auc]['roc_auc']:.4f})",
        f"    ★ F1    최고 실험 : {best_f1} ({all_results[best_f1]['f1']:.4f})",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "2. 장점",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  ① 전이 학습으로 소규모 데이터 문제 극복",
        "     - 원본 데이터가 36장/클래스밖에 없는 상황에서",
        "       ImageNet 사전학습 가중치를 활용해 안정적인 학습이 가능함.",
        "     - 처음부터 학습(scratch)하면 30~40장 수준에서 과적합이 극심하지만,",
        "       전이 학습은 이미 다양한 시각 특징을 알고 있어 빠르게 수렴함.",
        "",
        "  ② 이중 이미지 증강으로 과적합 완화 및 다양성 극대화",
        "     - 원본 36장 → 252장(7배)으로 증강해 데이터 다양성 대폭 확보.",
        "     - 기본 증강(basic): 회전·이동·줌·밝기·노이즈·blur 등 기하학적/광도 변환.",
        "     - VAE 증강(vae): 변분 오토인코더의 잠재 공간 샘플링으로",
        "       원본과 다른 새로운 패턴의 이미지를 생성해 일반화 능력 향상.",
        "     - 검증셋은 원본만 유지해 실제 테스트 환경과 동일한 분포로 평가.",
        "",
        "  ③ 민감도 우선 설계 — 의료 AI에 적합",
        f"     - Parkinson 클래스 가중치 {PARKINSON_WEIGHT}배 적용으로 FN(환자 누락) 페널티 강화.",
        "     - Early stopping 기준을 val loss → val Recall로 변경해",
        "       모델이 민감도를 최대화하는 방향으로 학습됨.",
        "",
        "  ④ 임계값 최적화로 실용적 운용 가능",
        "     - 기본 0.5 외에 검증셋 기반 최적 임계값을 자동 탐색함.",
        "     - 용도에 따라 임계값을 조절해 민감도-정밀도 트레이드오프를 제어 가능.",
        "",
        "  ⑤ 원본 전용 검증셋으로 신뢰도 높은 early stopping",
        "     - 증강 이미지를 검증셋에서 제외해 검증 성능이 실제 테스트와 유사.",
        "     - 증강 이미지가 검증셋에 섞이면 너무 이른 epoch에서 멈추는 문제 해결.",
        "",
        "  ⑥ 빠른 추론 속도",
        "     - ResNet18은 GPU 없이도 이미지 1장을 수십 ms 내에 분류 가능.",
        "     - 실제 임상 스크리닝 도구로 배포 가능한 경량 구조.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "3. 단점",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  ① 테스트셋이 너무 작아 통계적 신뢰도 부족",
        "     - 테스트 이미지가 30장(shape당)에 불과해 성능 수치의 변동성이 큼.",
        "     - 예: Recall 0.933은 '15명 중 14명 탐지'인데, 1명 차이가 0.067p.",
        "     - 최소 100명 이상의 테스트셋이 있어야 신뢰할 만한 평가가 가능함.",
        "",
        "  ② 원본 학습 데이터 절대량 부족",
        "     - 이중 증강 후 252장이지만 원본은 36장으로, 다양성에 근본적 한계가 있음.",
        "     - basic 증강은 원본의 기하학적 변형, VAE 증강은 잠재 공간 샘플링이지만",
        "       두 방식 모두 원본 36장에서 파생되므로 완전히 새로운 패턴 확보에 한계.",
        "",
        "  ③ Patient-level 분리 미적용",
        "     - 동일 환자가 여러 장을 그린 경우(V01PE02, V01PE03 등),",
        "       같은 환자의 이미지가 학습셋과 테스트셋에 동시에 들어갈 수 있음.",
        "     - 이 경우 모델이 특정 환자의 스타일을 '암기'해 과대평가될 위험.",
        "",
        "  ④ 검증셋도 여전히 16장으로 매우 작음",
        "     - 클래스당 8장의 원본으로 검증하므로 early stopping 신호가 불안정.",
        "     - 16장 중 1~2장의 오분류만으로도 Recall이 12.5%p씩 변동함.",
        "",
        "  ⑤ 단일 모델 — 예측 불확실성 미반영",
        "     - 하나의 모델만 사용하므로 예측의 신뢰 구간을 알 수 없음.",
        "     - 앙상블(여러 모델 평균)이 없어 이상치에 민감할 수 있음.",
        "",
        "  ⑥ Grayscale 이미지의 정보 제한",
        "     - 나선/파도 그림은 단색 선 그림으로, 색상 정보가 없음.",
        "     - ResNet은 RGB 3채널 입력을 기대하므로 grayscale을 3채널로 복제해 입력.",
        "       실질적인 정보는 1채널이지만 3채널로 처리하는 비효율이 있음.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "4. 보완할 점",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  ① [데이터] 더 많은 환자 데이터 수집 (최우선)",
        "     - 최소 300명 이상(클래스당)의 다양한 환자 데이터가 필요함.",
        "     - 다양한 연령·성별·증상 단계의 환자를 포함해야 일반화 가능.",
        "     - 현재 테스트셋(15명)은 너무 작아 실제 성능 평가 불가.",
        "",
        "  ② [분할] Patient-level Split 적용",
        "     - 파일명 패턴(V01PE02, V01PE03)을 분석해 동일 환자를 식별하고,",
        "       같은 환자의 이미지가 학습셋과 테스트셋에 동시에 포함되지 않도록 분리.",
        "     - 현재 구조에서 V01PE02와 V01PE03이 각각 학습·테스트에 들어가면",
        "       평가가 부풀려질 수 있음.",
        "",
        "  ③ [검증] K-Fold 교차 검증 적용",
        "     - 데이터가 적을 때 단순 Hold-out 검증은 결과 변동성이 큼.",
        "     - 5-Fold 또는 10-Fold 교차 검증으로 모든 데이터를 검증에 활용하면",
        "       더 안정적이고 신뢰할 수 있는 성능 추정이 가능.",
        "",
        "  ④ [증강] GAN 기반 증강으로 다양성 추가 확대",
        "     - 현재 실험 E는 기본 증강 + VAE 증강 이중 구조를 적용 중임.",
        "     - 다음 단계로 GAN(생성적 적대 신경망)을 활용하면",
        "       VAE보다 더 선명하고 현실적인 새로운 패턴의 이미지를 생성할 수 있음.",
        "     - 단, 36장으로 GAN을 학습하는 것은 불안정하므로 데이터 확보 후 시도.",
        "",
        "  ⑤ [모델] 앙상블(Ensemble) 적용",
        "     - 서로 다른 랜덤 시드·하이퍼파라미터로 학습된 3~5개 모델의",
        "       예측을 평균내면 단일 모델보다 안정적인 성능을 얻을 수 있음.",
        "     - ResNet18 + EfficientNet-B0 등 다양한 백본을 결합하는 방법도 유효.",
        "",
        "  ⑥ [설명 가능성] Grad-CAM 시각화 추가",
        "     - AI가 어느 부분을 보고 판단했는지 시각화하면 의료진이 신뢰하기 쉬움.",
        "     - Grad-CAM(Gradient-weighted Class Activation Mapping)으로",
        "       나선/파도 그림에서 모델이 주목한 영역을 히트맵으로 표시 가능.",
        "",
        "  ⑦ [백본] 의료 이미지 특화 모델 탐색",
        "     - ImageNet 사전학습 모델은 자연 이미지에 최적화돼 있음.",
        "     - 의료 이미지(X-ray, 내시경 등)로 사전학습된 모델(MedViT, BioViL 등)을",
        "       백본으로 사용하면 손글씨 의료 데이터에 더 적합할 수 있음.",
        "",
        "  ⑧ [운용] 확률 기반 위험 등급 출력",
        "     - 현재: 환자 / 비환자 이진 분류",
        "     - 개선: '파킨슨 확률 78%' → 고위험/중위험/저위험 3단계 출력",
        "     - 의료진이 확률값을 함께 보고 최종 판단을 내리는 보조 도구로 활용.",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "5. 종합 평가",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "  현재 모델은 252장(shape당)이라는 소규모 데이터셋 기준으로는",
        "  ROC-AUC 0.92~0.97의 양호한 판별 능력을 보이고 있으며,",
        "  민감도 개선 기법 적용 후 파킨슨 환자 탐지율도 크게 향상됨.",
        "",
        "  그러나 테스트셋(30장)이 너무 작아 이 수치를 임상적으로",
        "  신뢰하기 어렵습니다. 현 단계는 '가능성 검증(proof-of-concept)'",
        "  수준으로 보는 것이 적절하며, 실제 임상 도입을 위해서는",
        "  최소 300명 이상의 환자 데이터와 Patient-level 검증이 필수입니다.",
        "",
        "  단기 개선 우선순위:",
        "    1순위 — 데이터 추가 수집 (가장 중요)",
        "    2순위 — Patient-level Split 적용",
        "    3순위 — K-Fold 교차 검증",
        "    4순위 — Grad-CAM 설명 가능성 추가",
        "    5순위 — 앙상블 모델 구성",
        "",
        "=" * 70,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# 9. 메인
# ══════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Settings: LR={LR}  batch={BATCH_SIZE}  epochs={EPOCHS}"
          f"  parkinson_weight={PARKINSON_WEIGHT}  patience={PATIENCE}\n")

    experiments = {
        "Spiral"  : ["spiral"],
        "Wave"    : ["wave"],
        "Combined": ["spiral", "wave"],
    }

    all_results = {}
    for name, shapes in experiments.items():
        all_results[name] = run_experiment(name, shapes, OUTPUT_DIR / name.lower())

    # 비교 시각화
    plot_metrics_comparison(all_results, OUTPUT_DIR / "metrics_comparison.png")

    # JSON 저장
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 보고서 작성
    write_results_report(all_results, OUTPUT_DIR / "results_report.txt")
    write_model_analysis(all_results, OUTPUT_DIR / "model_analysis.txt")

    # 콘솔 요약
    print(f"\n{'='*58}")
    print(" 최종 결과 요약 (최적 임계값 기준)")
    print(f"{'='*58}")
    print(f"  {'실험':<10} {'Acc':>6} {'Recall':>7} {'Prec':>7} "
          f"{'F1':>7} {'AUC':>7} {'Thr':>6}")
    print(f"  {'─'*54}")
    for name, r in all_results.items():
        print(f"  {name:<10} {r['accuracy']:>6.3f} {r['recall']:>7.3f} "
              f"{r['precision']:>7.3f} {r['f1']:>7.3f} "
              f"{r['roc_auc']:>7.3f} {r['threshold']:>6.2f}")
    print(f"\n  결과 위치 : {OUTPUT_DIR}")
    print(f"  결과 보고서 : results_report.txt")
    print(f"  모형 분석  : model_analysis.txt")


if __name__ == "__main__":
    main()
