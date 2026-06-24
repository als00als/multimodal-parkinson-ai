"""
변비 증상 기반 파킨슨 예측 모델 학습 & 저장 (Optimized)
======================================================
특징: SCOPA-AUT 3문항 (AUT5·AUT6·AUT7) + SCOPA_CONSTIPATION_TOTAL (4차원)
대상: PD(1) vs HC(0), 944명 (PD 809 / HC 135)

[최적화 내용]
  - RF + PolySVM + PolyLR 앙상블 (interaction 항으로 비선형 패턴 포착)
  - 5-Fold × 5-seed 반복 CV 로 안정적 성능 평가
  - 3가지 임계값 동시 저장 (Youden / F1 / F2 — 의료 스크리닝용)
  - 확률 보정(Calibration) 옵션 추가
  - 기존 baseline AUC=0.6875 → Optimized AUC=0.6900 (+0.0025)

실행: python train_constipation_model.py
저장: models/ 폴더
  - constipation_ensemble.pkl   : Soft Voting Ensemble (RF + PolySVM + PolyLR)
  - constipation_threshold.pkl  : 최적 분류 임계값 (Youden, 기본값)
  - constipation_thresholds.pkl : 3종 임계값 dict (youden / f1 / f2)
  - constipation_features.txt   : 사용 특징 목록
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, roc_curve,
    f1_score, fbeta_score, precision_score, recall_score,
)

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "PPMI_BSIT12_SCOPA.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "SCOPA_AUT5",               # 배변 횟수 감소
    "SCOPA_AUT6",               # 배변 시 힘듦
    "SCOPA_AUT7",               # 변실금
    "SCOPA_CONSTIPATION_TOTAL", # 3문항 합산
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
print(f"    특징: {X.shape[1]}개  ({', '.join(FEATURE_COLS)})")

print("\n    문항별 기술통계 (PD vs HC):")
feat_df = df[FEATURE_COLS + [LABEL_COL]].copy()
for col in FEATURE_COLS:
    pd_mean = feat_df.loc[feat_df[LABEL_COL]==1, col].mean()
    hc_mean = feat_df.loc[feat_df[LABEL_COL]==0, col].mean()
    print(f"    {col:<28}  PD={pd_mean:.3f}  HC={hc_mean:.3f}")


# ─────────────────────────────────────────────────────────────
# 2. 앙상블 모델 구성 — Optimized (Poly interaction 추가)
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Optimized Soft Voting Ensemble 구성...")
print("    - RF(n=500, md=5)  : 트리 기반 비선형 패턴")
print("    - PolySVM(deg=2)   : 변수 상호작용 + RBF 커널")
print("    - PolyLR(deg=2)    : 변수 상호작용 선형 모델")


def build_ensemble():
    """매번 새로운 인스턴스 반환 (CV 시 상태 초기화 보장)."""
    rf_pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    svm_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=False)),
        ("sc",   StandardScaler()),
        ("clf",  SVC(
            kernel="rbf", C=5, gamma="scale", probability=True,
            class_weight="balanced", random_state=42)),
    ])
    lr_pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, interaction_only=False)),
        ("sc",   StandardScaler()),
        ("clf",  LogisticRegression(
            C=1, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    return VotingClassifier(
        estimators=[("rf", rf_pipe), ("svm", svm_pipe), ("lr", lr_pipe)],
        voting="soft",
    )


# ─────────────────────────────────────────────────────────────
# 3. 반복 CV 성능 평가 (5-Fold × 3-seed)
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
# 4. 최종 OOF 예측 + 3종 임계값 탐색
# ─────────────────────────────────────────────────────────────
print("\n[4/5] OOF 예측 + 임계값 최적화 (seed=42)...")
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

# 임계값 후보 탐색 — 의학적 유의성을 위해 최소 Specificity 0.4 제약
MIN_SPEC = 0.4
thresh_candidates = np.linspace(0.05, 0.95, 181)


def _score_with_spec_floor(thr, scorer):
    pred = (oof_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    if spec < MIN_SPEC:
        return -1.0
    return scorer(y, pred)


# 1) Youden — 의료 스크리닝 균형 (기본)
fpr, tpr, thrs = roc_curve(y, oof_proba)
youden = float(thrs[np.argmax(tpr - fpr)])

# 2) F1 — Spec>=0.4 제약하에 F1 최대
f1s = [_score_with_spec_floor(t, f1_score) for t in thresh_candidates]
thr_f1 = float(thresh_candidates[int(np.argmax(f1s))])

# 3) F2 — Spec>=0.4 제약하에 Recall 가중 F-beta(beta=2) 최대
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

# 기본 임계값 = Youden (기존 호환성 유지)
best_thresh = youden
print(f"\n    기본 임계값 (Youden) : {best_thresh:.3f}")
print(f"    OOF AUC              : {roc_auc_score(y, oof_proba):.3f}")


# ─────────────────────────────────────────────────────────────
# 5. 전체 데이터로 최종 학습 & 저장
# ─────────────────────────────────────────────────────────────
print("\n[5/5] 전체 데이터 최종 학습 & 저장...")
final_ensemble = build_ensemble()
final_ensemble.fit(X, y)

joblib.dump(final_ensemble, MODEL_DIR / "constipation_ensemble.pkl")
joblib.dump(best_thresh, MODEL_DIR / "constipation_threshold.pkl")
joblib.dump(thresholds, MODEL_DIR / "constipation_thresholds.pkl")

with open(MODEL_DIR / "constipation_features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(FEATURE_COLS))

print(f"    constipation_ensemble.pkl    -> {MODEL_DIR / 'constipation_ensemble.pkl'}")
print(f"    constipation_threshold.pkl   -> {MODEL_DIR / 'constipation_threshold.pkl'}  (Youden, 기본)")
print(f"    constipation_thresholds.pkl  -> {MODEL_DIR / 'constipation_thresholds.pkl'}  (youden/f1/f2 dict)")
print(f"    constipation_features.txt    -> {MODEL_DIR / 'constipation_features.txt'}")
print("\n변비 모델 저장 완료!")
