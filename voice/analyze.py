import os, librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

hc_dir = r"C:\Project_AI\파킨슨 전구기 예측 서비스\음성\data\HC_AH"
pd_dir = r"C:\Project_AI\파킨슨 전구기 예측 서비스\음성\data\PD_AH"

def make_feat_names():
    names = []
    for i in range(13): names.append(f"mfcc{i}_mean")
    for i in range(13): names.append(f"mfcc{i}_std")
    for i in range(13): names.append(f"dmfcc{i}_mean")
    for i in range(13): names.append(f"d2mfcc{i}_mean")
    names += ["f0_mean","f0_std","f0_range","f0_iqr","voiced_ratio","jitter"]
    names += ["rms_mean","rms_std","shimmer"]
    names += ["centroid_mean","centroid_std","bandwidth_mean","rolloff_mean","zcr_mean","zcr_std"]
    for i in range(4): names.append(f"contrast{i}")
    for i in range(12): names.append(f"chroma{i}")
    names += ["mel_mean","mel_std"]
    return names

def extract_features(filepath, sr_target=8000):
    y, sr = librosa.load(filepath, sr=sr_target, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    if len(y) < sr * 0.5:
        return None
    feats = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats.extend(np.mean(d1, axis=1).tolist())
    feats.extend(np.mean(d2, axis=1).tolist())
    f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=350, sr=sr)
    f0_v = f0[voiced_flag & ~np.isnan(f0)]
    if len(f0_v) > 2:
        feats += [float(np.mean(f0_v)), float(np.std(f0_v)), float(np.ptp(f0_v)),
                  float(np.percentile(f0_v,75)-np.percentile(f0_v,25)),
                  float(np.sum(voiced_flag)/len(voiced_flag)),
                  float(np.mean(np.abs(np.diff(f0_v)))/(np.mean(f0_v)+1e-9))]
    else:
        feats += [0.0]*6
    rms = librosa.feature.rms(y=y)[0]
    feats += [float(np.mean(rms)), float(np.std(rms)),
              float(np.mean(np.abs(np.diff(rms)))/(np.mean(rms)+1e-9))]
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats += [float(np.mean(sc)), float(np.std(sc)), float(np.mean(sbw)),
              float(np.mean(sr_)), float(np.mean(zcr)), float(np.std(zcr))]
    sct = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3, fmin=100)
    feats.extend(np.mean(sct, axis=1).tolist())
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1).tolist())
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats += [float(np.mean(mel_db)), float(np.std(mel_db))]
    return np.array(feats, dtype=np.float32)

# 데이터 로드
X_list, y_list = [], []
for fname in sorted(os.listdir(hc_dir)):
    if not fname.endswith('.wav'): continue
    f = extract_features(os.path.join(hc_dir, fname))
    if f is not None: X_list.append(f); y_list.append(0)
for fname in sorted(os.listdir(pd_dir)):
    if not fname.endswith('.wav'): continue
    f = extract_features(os.path.join(pd_dir, fname))
    if f is not None: X_list.append(f); y_list.append(1)

X = np.nan_to_num(np.array(X_list), nan=0.0, posinf=0.0, neginf=0.0)
y = np.array(y_list)
feat_names = make_feat_names()
print(f"[Data] HC={np.sum(y==0)}, PD={np.sum(y==1)}, Features={X.shape[1]}")

# 1. 특징 중요도
print("="*60)
print("  TOP-20 Feature Importance (Random Forest)")
print("="*60)
sc_all = StandardScaler()
X_sc = sc_all.fit_transform(X)
rf_full = RandomForestClassifier(n_estimators=500, random_state=42)
rf_full.fit(X_sc, y)
imp = rf_full.feature_importances_
top20_idx = np.argsort(imp)[::-1][:20]
for rank, idx in enumerate(top20_idx, 1):
    print(f"  {rank:2d}. {feat_names[idx]:<22}  {imp[idx]:.4f}")

# 2. 특징 수별 성능
print("\n" + "="*60)
print("  Feature Count vs LOOCV Performance")
print("="*60)
loo = LeaveOneOut()
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_k, best_k_auc = 30, 0

for k in [10, 20, 30, 40, 85]:
    top_k_idx = np.argsort(imp)[::-1][:k]
    Xk = X[:, top_k_idx]
    pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42))])
    auc_cv = cross_val_score(pipe, Xk, y, cv=cv5, scoring='roc_auc')
    preds = []
    for tr, te in loo.split(Xk, y):
        pipe.fit(Xk[tr], y[tr])
        preds.append(pipe.predict(Xk[te])[0])
    preds = np.array(preds)
    loo_acc = np.mean(preds==y)
    print(f"  Top-{k:2d} | 5CV AUC={auc_cv.mean():.3f}+-{auc_cv.std():.3f} | LOOCV Acc={loo_acc:.3f}")
    if auc_cv.mean() > best_k_auc:
        best_k_auc = auc_cv.mean()
        best_k = k

print(f"  => 최적 특징 수: Top-{best_k}")

# 3. GridSearch
print("\n" + "="*60)
print("  GridSearch Hyperparameter Tuning (5-Fold CV, AUC)")
print("="*60)
top_k_idx = np.argsort(imp)[::-1][:best_k]
Xopt = X[:, top_k_idx]

param_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 4, 5, 6],
    'clf__min_samples_leaf': [1, 2, 3]
}
gs_rf = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))]),
    param_rf, cv=cv5, scoring='roc_auc', n_jobs=-1)
gs_rf.fit(Xopt, y)
print(f"  RF  Best: {gs_rf.best_params_}  AUC={gs_rf.best_score_:.3f}")

param_svm = {'clf__C': [0.01, 0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}
gs_svm = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('clf', SVC(kernel='rbf', probability=True, random_state=42))]),
    param_svm, cv=cv5, scoring='roc_auc', n_jobs=-1)
gs_svm.fit(Xopt, y)
print(f"  SVM Best: {gs_svm.best_params_}  AUC={gs_svm.best_score_:.3f}")

param_lr = {'clf__C': [0.001, 0.01, 0.1, 1]}
gs_lr = GridSearchCV(
    Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=500, random_state=42))]),
    param_lr, cv=cv5, scoring='roc_auc', n_jobs=-1)
gs_lr.fit(Xopt, y)
print(f"  LR  Best: {gs_lr.best_params_}  AUC={gs_lr.best_score_:.3f}")

# 4. 앙상블 LOOCV 최종 평가
print("\n" + "="*60)
print("  Soft Voting Ensemble - LOOCV Final Evaluation")
print("="*60)
best_rf_m  = gs_rf.best_estimator_
best_svm_m = gs_svm.best_estimator_
best_lr_m  = gs_lr.best_estimator_

ensemble = VotingClassifier(
    estimators=[('rf', best_rf_m), ('svm', best_svm_m), ('lr', best_lr_m)],
    voting='soft')

preds_ens, proba_ens, trues = [], [], []
for tr, te in loo.split(Xopt, y):
    ensemble.fit(Xopt[tr], y[tr])
    preds_ens.append(ensemble.predict(Xopt[te])[0])
    proba_ens.append(ensemble.predict_proba(Xopt[te])[0][1])
    trues.append(y[te][0])

preds_ens = np.array(preds_ens)
trues     = np.array(trues)
proba_ens = np.array(proba_ens)
cm = confusion_matrix(trues, preds_ens)
auc_ens = roc_auc_score(trues, proba_ens)
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
specificity = cm[0,0] / (cm[0,0] + cm[0,1])

print(f"  Accuracy   : {np.mean(preds_ens==trues):.3f}  ({np.sum(preds_ens==trues)}/{len(trues)})")
print(f"  AUC        : {auc_ens:.3f}")
print(f"  Sensitivity: {sensitivity:.3f}  (PD 탐지율)")
print(f"  Specificity: {specificity:.3f}  (정상 탐지율)")
print(f"\n  Confusion Matrix:")
print(f"            Pred HC  Pred PD")
print(f"  True HC :  {cm[0,0]:5d}    {cm[0,1]:5d}")
print(f"  True PD :  {cm[1,0]:5d}    {cm[1,1]:5d}")
print()
print(classification_report(trues, preds_ens, target_names=['HC(Normal)','PD(Parkinson)']))
