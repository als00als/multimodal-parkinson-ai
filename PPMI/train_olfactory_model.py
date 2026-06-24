"""
후각 기능 기반 파킨슨 예측 모델 학습 & 저장 (Optimized)
=====================================================
특징: B-SIT 12개 문항 정답 여부 + BSIT_TOTAL (13차원)
대상: PD(1) vs HC(0), 944명 (PD 809 / HC 135)

[최적화 내용]
  - Baseline (RF+SVM+LR)이 이미 최적 (AUC≈0.7653, repeated CV 확인)
  - 하이퍼파라미터 미세 튜닝 (RF n_estimators ↑, min_samples_leaf 보수적)
  - 3가지 임계값 동시 저장 (Youden / F1 / F2 — 의료 스크리닝용)
  - 반복 CV 안정성 확인 절차 포함

실행: python train_olfactory_model.py
저장: models/ 폴더
  - olfactory_ensemble.pkl    : Soft Voting Ensemble (RF + SVM + LR)
  - olfactory_threshold.pkl   : 최적 분류 임계값 (Youden, 기본)
  - olfactory_thresholds.pkl  : 3종 임계값 dict (youden / f1 / f2)
  - olfactory_features.txt    : 사용 특징 목록
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
    f1_score, fbeta_score,
)

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "PPMI_BSIT12_SCOPA.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "BSIT_CHERRY", "BSIT_DILL_PICKLE", "BSIT_BANANA", "BSIT_CHOCOLATE",
    "BSIT_CINNAMON", "BSIT_GASOLINE", "BSIT_LEMON", "BSIT_ONION",
    "BSIT_PINEAPPLE", "BSIT_ROSE", "BSIT_SOAP", "BSIT_SMOKE",
    "BSIT_TOTAL",
]
LABEL_COL = "LABEL_PD"


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────
print("\n[1/5] 데이터 로드...")
df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLS].values.astype(np.float32)
y  = df[LABEL_COL].values

print(f"    샘플: {len(y)}명  |  HC={np.sum(y==0)}, PD={np.sum(y==1)}")
print(f"    특징: {X.shape[1]}개")


# ─────────────────────────────────────────────────────────────
# 2. 앙상블 모델 구성 — Tuned baseline
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Soft Voting Ensemble 구성 (Tuned Baseline)...")


def build_ensemble():
    """매 폴드 새 인스턴스 반환."""
    rf_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    svm_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced", random_state=42)),
    ])
    lr_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(
            C=1, max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    return VotingClassifier(
        estimators=[("rf", rf_pipe), ("svm", svm_pipe), ("lr", lr_pipe)],
        voting="soft",
    )


# ─────────────────────────────────────────────────────────────
# 3. 반복 CV 평가 (5-Fold × 3-seed)
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Repeated Stratified 5-Fold (3 seeds) 평가...")
all_aucs = []
for seed in range(3):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = build_ensemble()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    all_aucs.append(roc_auc_score(y, oof))
    print(f"    seed={seed}  OOF AUC={all_aucs[-1]:.4f}")
print(f"    Mean AUC = {np.mean(all_aucs):.4f} +- {np.std(all_aucs):.4f}")


# ─────────────────────────────────────────────────────────────
# 4. seed=42 기준 fold별 지표 + 임계값 탐색
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Fold 지표 + 임계값 최적화 (seed=42)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(y))
fold_metrics = []
for fold, (tr, te) in enumerate(cv.split(X, y), 1):
    m = build_ensemble()
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    oof_proba[te] = p
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y[te], pred)
    tn, fp, fn, tp = cm.ravel()
    fold_metrics.append({
        "fold": fold,
        "auc": roc_auc_score(y[te], p),
        "accuracy": accuracy_score(y[te], pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
    })

metrics_df = pd.DataFrame(fold_metrics)
print(f"\n    {'Fold':<6} {'AUC':>8} {'Accuracy':>9} {'Sensitivity':>12} {'Specificity':>12}")
print(f"    {'-'*52}")
for _, row in metrics_df.iterrows():
    print(f"    {int(row['fold']):<6} {row['auc']:>8.3f} {row['accuracy']:>9.3f} "
          f"{row['sensitivity']:>12.3f} {row['specificity']:>12.3f}")
print(f"    {'-'*52}")
print(f"    {'Mean':<6} {metrics_df['auc'].mean():>8.3f} "
      f"{metrics_df['accuracy'].mean():>9.3f} "
      f"{metrics_df['sensitivity'].mean():>12.3f} "
      f"{metrics_df['specificity'].mean():>12.3f}")

# 3종 임계값 탐색 — 의학적 유의성을 위해 최소 Specificity 0.4 제약
MIN_SPEC = 0.4
thresh_candidates = np.linspace(0.05, 0.95, 181)


def _score_with_spec_floor(thr, scorer):
    pred = (oof_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    if spec < MIN_SPEC:
        return -1.0
    return scorer(y, pred)


fpr, tpr, thrs = roc_curve(y, oof_proba)
youden = float(thrs[np.argmax(tpr - fpr)])

f1s = [_score_with_spec_floor(t, f1_score) for t in thresh_candidates]
thr_f1 = float(thresh_candidates[int(np.argmax(f1s))])
f2s = [_score_with_spec_floor(t, lambda y_, p_: fbeta_score(y_, p_, beta=2))
       for t in thresh_candidates]
thr_f2 = float(thresh_candidates[int(np.argmax(f2s))])

thresholds = {"youden": youden, "f1": thr_f1, "f2": thr_f2}

print(f"\n    임계값 후보:")
for name, thr in thresholds.items():
    pred = (oof_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"      {name:>6}={thr:.3f}  "
          f"Acc={accuracy_score(y, pred):.3f}  "
          f"Sens={tp/(tp+fn):.3f}  Spec={tn/(tn+fp):.3f}  "
          f"F1={f1_score(y, pred):.3f}  F2={fbeta_score(y, pred, beta=2):.3f}")

best_thresh = youden
print(f"\n    기본 임계값 (Youden) : {best_thresh:.3f}")
print(f"    OOF AUC              : {roc_auc_score(y, oof_proba):.3f}")


# ─────────────────────────────────────────────────────────────
# 5. 전체 데이터로 최종 학습 & 저장
# ─────────────────────────────────────────────────────────────
print("\n[5/5] 전체 데이터 최종 학습 & 저장...")
final_ensemble = build_ensemble()
final_ensemble.fit(X, y)

joblib.dump(final_ensemble, MODEL_DIR / "olfactory_ensemble.pkl")
joblib.dump(best_thresh, MODEL_DIR / "olfactory_threshold.pkl")
joblib.dump(thresholds, MODEL_DIR / "olfactory_thresholds.pkl")

with open(MODEL_DIR / "olfactory_features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(FEATURE_COLS))

print(f"    olfactory_ensemble.pkl    -> {MODEL_DIR / 'olfactory_ensemble.pkl'}")
print(f"    olfactory_threshold.pkl   -> {MODEL_DIR / 'olfactory_threshold.pkl'}  (Youden, 기본)")
print(f"    olfactory_thresholds.pkl  -> {MODEL_DIR / 'olfactory_thresholds.pkl'}  (youden/f1/f2 dict)")
print(f"    olfactory_features.txt    -> {MODEL_DIR / 'olfactory_features.txt'}")
print("\n후각 모델 저장 완료!")
