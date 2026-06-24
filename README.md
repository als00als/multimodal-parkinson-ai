# 🧠 Multimodal Parkinson's Disease Early Detection AI

> **보건소 기반 파킨슨병 전구기 고위험군 스크리닝 AI 서비스**  
> 음성 · 필적 · 후각 · 변비 4개 모달리티를 융합하여 증상 발현 전 조기 탐지

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)

---

## 📌 프로젝트 개요

파킨슨병은 운동 증상이 나타나기 수 년 전부터 **전구기(prodromal stage)** 비운동성 이상 징후가 발생합니다. 그러나 기존 임상 스크리닝은 비용·접근성 문제로 조기 탐지에 한계가 있습니다.

이 프로젝트는 **보건소 키오스크에서 비전문가도 5분 내에 자가 스크리닝**할 수 있는 멀티모달 AI 시스템을 구현합니다.

| 모달리티 | 측정 방식 | 핵심 지표 |
|---|---|---|
| 🖊️ **필적** | 태블릿 나선형·파도형 그림 | 운동 조절 능력 저하 |
| 🎙️ **음성** | 마이크 모음 발성 녹음 | 성대 진전·발성 불안정 |
| 👃 **후각** | B-SIT 12문항 선택 | 후각 기능 저하 |
| 🩺 **변비** | SCOPA-AUT 3문항 선택 | 자율신경 이상 |

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│    Canvas 그리기 │ 마이크 녹음 │ 설문 UI            │
└────────┬────────────────┬───────────────┬───────────┘
         │                │               │
         ▼                ▼               ▼
  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
  │ DualResNet18│  │ Voice        │  │ PPMI-based   │
  │ (Spiral +   │  │ Ensemble     │  │ Ensemble     │
  │  Wave CNN)  │  │ RF+SVM+LR    │  │ (후각·변비)  │
  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
         │                │                  │
         ▼                ▼                  ▼
  ┌─────────────────────────────────────────────────┐
  │         Soft Voting Fusion (종합 위험도)         │
  └─────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────┐
  │         XAI 설명 (SHAP · Grad-CAM · Waterfall)  │
  └─────────────────────────────────────────────────┘
```

---

## 🤖 모델 상세

### 1. 필적 분류 — DualResNet18

- **아키텍처**: ResNet18 기반 이중 입력 멀티모달 CNN
  - Spiral 브랜치 + Wave 브랜치를 각각 처리 후 Feature Concat → FC 분류
  - ImageNet 사전학습 가중치 전이학습(Fine-tuning)
- **데이터 증강 전략 (Experiment E)**:
  - 원본 36장 → 기본 증강 108장 + **VAE 생성 증강 108장** = 252장/클래스
  - VAE로 분포 보존 합성 데이터를 생성하여 소규모 의료 데이터 한계 극복
- **학습 설정**: `lr=5e-5`, `batch=16`, `epoch=100`, Stratified K-Fold
- **XAI**: Grad-CAM 히트맵으로 분류 근거 시각화

### 2. 음성 분류 — Soft Voting Ensemble

- **특징 추출** (`librosa`): 총 85차원
  - MFCC 13개 (mean·std·Δ·Δ²) / F0 (mean·std·range·IQR·voiced ratio·jitter)
  - RMS Shimmer / Spectral Centroid·Bandwidth·Rolloff / ZCR / Chroma 12개 / Mel
- **모델**: RandomForest + SVC(RBF) + LogisticRegression → Soft Voting
- **XAI**: SHAP으로 상위 8개 음향 특징 기여도 시각화

### 3. 후각 / 변비 분류 — PPMI 기반 Ensemble

- **데이터**: [PPMI(Parkinson's Progression Markers Initiative)](https://www.ppmi-info.org/) 공개 임상 데이터
- **후각**: B-SIT 12문항 정답 여부 + 총점 → VotingClassifier
- **변비**: SCOPA-AUT 5·6·7번 문항 + 총점 → VotingClassifier
- **임계값 전략**: Youden Index / F1 / F-beta(β=2) 3종 제공
  - `F2` 전략 선택 시 FN(위험군 미탐지) 최소화 — 의료 스크리닝에 적합
- **XAI**: SHAP 문항별 기여도 분석

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| **딥러닝** | PyTorch, torchvision (ResNet18, Transfer Learning) |
| **머신러닝** | scikit-learn (RF, SVM, LR, VotingClassifier, GridSearchCV) |
| **음성 처리** | librosa (MFCC, F0/pyin, Spectral features) |
| **XAI** | SHAP, Grad-CAM, Waterfall Chart |
| **데이터** | PPMI 공개 임상 DB, VAE 합성 데이터 증강 |
| **프론트엔드** | Streamlit, streamlit-drawable-canvas |
| **기타** | numpy, pandas, PIL, joblib |

---

## 📁 프로젝트 구조

```
multimodal-parkinson-ai/
├── streamlit/
│   ├── app.py                    # 메인 Streamlit 앱 (키오스크 UI)
│   ├── app_kiosk_restructured.py # 리팩토링 버전
│   ├── voice_predictor.py        # 음성 예측 모듈
│   ├── spiral_predictor.py       # 필적 예측 모듈
│   ├── ppmi_predictor.py         # 후각·변비 예측 모듈
│   └── xai_utils.py              # XAI 시각화 유틸리티
├── voice/
│   ├── analyze.py                # 음성 특징 추출 & 앙상블 학습
│   ├── save_voice_model.py       # 모델 저장
│   └── data/                     # HC / PD 음성 데이터
├── handwriting/
│   ├── train_expE.py             # DualResNet18 학습 (Experiment E)
│   ├── predict_drawings.py       # 추론 스크립트
│   └── data/                     # 나선형·파도형 필적 데이터
└── PPMI/                         # PPMI 임상 데이터 & 모델
```

---

## 🚀 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
cd streamlit
streamlit run app.py
```

---

## 📊 주요 구현 포인트

- **소규모 의료 데이터 문제 해결**: VAE 기반 합성 데이터 증강으로 클래스당 36장 → 252장 확보
- **멀티모달 융합**: 4개 이질적 모달리티를 Soft Voting으로 후기 융합(Late Fusion)
- **설명 가능한 AI(XAI)**: SHAP·Grad-CAM으로 임상 현장 신뢰성 확보
- **의료 리스크 최적화**: F-beta(β=2) 임계값으로 위험군 미탐지(FN) 최소화
- **실사용 고려**: 보건소 키오스크 시나리오 기반 UI/UX 설계

---

## 📄 관련 문서

- [제안 보고서](./23017018%20김수민_제안%20보고서.docx)
- [발표 보고서](./23017018%20김수민%20인공지능%20발표%20보고서.pptx)

---

## 📬 Contact

**김수민** · maristella0625@gmail.com
