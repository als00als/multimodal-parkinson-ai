"""
PPMI 기반 후각 / 변비 예측기
==============================
사용:
    from ppmi_predictor import OlfactoryPredictor, ConstipationPredictor

    olf = OlfactoryPredictor()
    is_pd, prob = olf.predict(bsit_answers)   # bsit_answers: dict or list[int] 12개 + total

    con = ConstipationPredictor()
    is_pd, prob = con.predict(scopa_answers)  # scopa_answers: dict {aut5, aut6, aut7}

[모델 파일]
    PPMI/models/ 안의
      - {olfactory,constipation}_ensemble.pkl   : 학습된 VotingClassifier
      - {olfactory,constipation}_threshold.pkl  : 기본 임계값 (Youden index)
      - {olfactory,constipation}_thresholds.pkl : (선택) 3종 임계값 dict
                                                  {"youden", "f1", "f2"}
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "PPMI" / "models"

BSIT_COLS = [
    "BSIT_CHERRY", "BSIT_DILL_PICKLE", "BSIT_BANANA", "BSIT_CHOCOLATE",
    "BSIT_CINNAMON", "BSIT_GASOLINE", "BSIT_LEMON", "BSIT_ONION",
    "BSIT_PINEAPPLE", "BSIT_ROSE", "BSIT_SOAP", "BSIT_SMOKE",
    "BSIT_TOTAL",
]

SCOPA_COLS = [
    "SCOPA_AUT5",
    "SCOPA_AUT6",
    "SCOPA_AUT7",
    "SCOPA_CONSTIPATION_TOTAL",
]


def _load_thresholds(name: str, default: float):
    """{name}_thresholds.pkl (dict) 또는 {name}_threshold.pkl (float) 로드."""
    multi_path = MODEL_DIR / f"{name}_thresholds.pkl"
    single_path = MODEL_DIR / f"{name}_threshold.pkl"
    multi = joblib.load(multi_path) if multi_path.exists() else None
    single = joblib.load(single_path) if single_path.exists() else default
    return float(single), multi


class OlfactoryPredictor:
    def __init__(self, strategy: str = "youden"):
        """
        strategy: "youden" | "f1" | "f2"
          - youden : ROC 균형 (기본)
          - f1     : F1 최대 (Spec≥0.4 제약)
          - f2     : Recall 가중 F-beta(β=2) 최대 (FN 위험 ↓, Spec≥0.4 제약)
        """
        model_path = MODEL_DIR / "olfactory_ensemble.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"후각 모델 파일이 없습니다: {model_path}\n"
                "PPMI/train_olfactory_model.py 를 먼저 실행하세요."
            )
        self.model = joblib.load(model_path)
        default_t, multi = _load_thresholds("olfactory", 0.566)
        if multi is not None and strategy in multi:
            self.threshold = float(multi[strategy])
        else:
            self.threshold = default_t
        self.strategy = strategy

    def predict(self, item_scores: list[int]) -> tuple[bool, float]:
        """
        item_scores: 12개 문항 정답 여부(0/1) 리스트 (CHERRY~SMOKE 순서)
        반환: (is_pd: bool, prob: float)
        """
        scores = list(item_scores)
        total  = sum(scores)
        feat   = np.array(scores + [total], dtype=np.float32).reshape(1, -1)
        prob   = float(self.model.predict_proba(feat)[0, 1])
        return prob >= self.threshold, prob


class ConstipationPredictor:
    def __init__(self, strategy: str = "youden"):
        model_path = MODEL_DIR / "constipation_ensemble.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"변비 모델 파일이 없습니다: {model_path}\n"
                "PPMI/train_constipation_model.py 를 먼저 실행하세요."
            )
        self.model = joblib.load(model_path)
        default_t, multi = _load_thresholds("constipation", 0.731)
        if multi is not None and strategy in multi:
            self.threshold = float(multi[strategy])
        else:
            self.threshold = default_t
        self.strategy = strategy

    def predict(self, aut5: int, aut6: int, aut7: int) -> tuple[bool, float]:
        """
        aut5: 배변 횟수 감소 (0~3)
        aut6: 배변 시 힘듦   (0~3)
        aut7: 변실금          (0~3)
        반환: (is_pd: bool, prob: float)
        """
        total = aut5 + aut6 + aut7
        feat  = np.array([aut5, aut6, aut7, total], dtype=np.float32).reshape(1, -1)
        prob  = float(self.model.predict_proba(feat)[0, 1])
        return prob >= self.threshold, prob
