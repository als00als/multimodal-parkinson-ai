"""
파킨슨 음성 예측 모듈 (voice_predictor.py)
============================================
VoicePredictor 클래스를 통해 WAV 파일을 HC / PD 로 분류합니다.
Streamlit app.py 에서 import 하여 사용합니다.

모델 위치: ../음성/voice_model/
  - voice_ensemble.pkl    : 학습된 VotingClassifier
  - top_feature_idx.npy   : 상위 N개 특징 인덱스 (Optimized: 8개)
  - voice_threshold.pkl   : 학습 시 자동 결정된 Youden 최적 임계값
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import joblib
from pathlib import Path
from typing import Optional, Tuple

# 모델 저장 위치 (streamlit/ → 음성/voice_model/)
_STREAMLIT_DIR = Path(__file__).parent
MODEL_DIR      = _STREAMLIT_DIR.parent / "음성" / "voice_model"

# Fallback 임계값 (저장된 voice_threshold.pkl 이 없을 때 사용)
DEFAULT_PD_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════
# 음향 특징 추출 (85차원)
# ══════════════════════════════════════════════════════════════
def extract_features(
    filepath: str | Path,
    sr_target: int = 8000,
) -> Optional[np.ndarray]:
    """
    WAV 파일에서 85차원 음향 특징 벡터를 추출합니다.
    실패 시 None 반환.

    특징 구성 (85차원)
    ------------------
    MFCC 13 × (mean+std+delta_mean+delta2_mean) = 52
    F0 관련 (f0_mean, f0_std, f0_range, f0_iqr, voiced_ratio, jitter) = 6
    RMS 에너지 / Shimmer (rms_mean, rms_std, shimmer_approx)          = 3
    스펙트럼 (centroid×2, bandwidth, rolloff, zcr×2)                  = 6
    Spectral Contrast (n_bands=3)                                      = 4
    Chroma                                                             = 12
    Mel-spectrogram 통계 (mean, std)                                   = 2
    """
    try:
        y, sr = librosa.load(str(filepath), sr=sr_target, mono=True)
        y, _  = librosa.effects.trim(y, top_db=20)
        if len(y) < sr * 0.5:      # 0.5초 미만 → 유효하지 않음
            return None

        feats: list[float] = []

        # ── MFCC ─────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats.extend(np.mean(mfcc, axis=1).tolist())                           # 13
        feats.extend(np.std(mfcc, axis=1).tolist())                            # 13
        feats.extend(np.mean(librosa.feature.delta(mfcc), axis=1).tolist())   # 13
        feats.extend(np.mean(librosa.feature.delta(mfcc, order=2), axis=1).tolist())  # 13

        # ── 기본 주파수 F0 ────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=350, sr=sr)
        f0_v = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_v) > 2:
            feats += [
                float(np.mean(f0_v)),                                          # f0 평균
                float(np.std(f0_v)),                                           # f0 변동성 (↑ PD)
                float(np.ptp(f0_v)),                                           # f0 범위 (↓ PD)
                float(np.percentile(f0_v, 75) - np.percentile(f0_v, 25)),     # IQR
                float(np.sum(voiced_flag) / len(voiced_flag)),                 # 유성음 비율
                float(np.mean(np.abs(np.diff(f0_v))) / (np.mean(f0_v) + 1e-9)),  # Jitter 근사
            ]
        else:
            feats += [0.0] * 6

        # ── RMS 에너지 / Shimmer 근사 ─────────────────────────
        rms = librosa.feature.rms(y=y)[0]
        feats += [
            float(np.mean(rms)),
            float(np.std(rms)),
            float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9)),     # Shimmer 근사
        ]

        # ── 스펙트럼 특징 ─────────────────────────────────────
        feats += [
            float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])),
            float(np.std(librosa.feature.spectral_centroid(y=y, sr=sr)[0])),
            float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])),
            float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])),
            float(np.mean(librosa.feature.zero_crossing_rate(y)[0])),
            float(np.std(librosa.feature.zero_crossing_rate(y)[0])),
        ]

        # ── Spectral Contrast (8kHz 안전: n_bands=3) ─────────
        sct = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=3, fmin=100)
        feats.extend(np.mean(sct, axis=1).tolist())                            # 4

        # ── Chroma ───────────────────────────────────────────
        feats.extend(
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).tolist()  # 12
        )

        # ── Mel-spectrogram 통계 ─────────────────────────────
        mel_db = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=16), ref=np.max
        )
        feats += [float(np.mean(mel_db)), float(np.std(mel_db))]               # 2

        return np.array(feats, dtype=np.float32)

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# 예측기 클래스
# ══════════════════════════════════════════════════════════════
class VoicePredictor:
    """
    학습된 Soft Voting Ensemble 모델로 음성 파일을 분류합니다.

    사용 예시
    ---------
    predictor = VoicePredictor()
    is_pd, prob = predictor.predict("path/to/audio.wav")
    # is_pd  : True=파킨슨, False=정상, None=분석 실패
    # prob   : 파킨슨 확률 0.0~1.0
    """

    def __init__(self) -> None:
        model_path  = MODEL_DIR / "voice_ensemble.pkl"
        idx_path    = MODEL_DIR / "top_feature_idx.npy"
        thresh_path = MODEL_DIR / "voice_threshold.pkl"

        if not model_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"음성 모델 파일을 찾을 수 없습니다.\n"
                f"경로: {MODEL_DIR}\n"
                "먼저 save_voice_model.py 를 실행하세요."
            )

        self.model     = joblib.load(model_path)
        self.top_idx   = np.load(idx_path)
        self.threshold = (
            float(joblib.load(thresh_path)) if thresh_path.exists()
            else DEFAULT_PD_THRESHOLD
        )

    def predict(
        self,
        wav_path: str | Path,
    ) -> Tuple[Optional[bool], Optional[float]]:
        """
        Parameters
        ----------
        wav_path : str | Path
            분석할 WAV 파일 경로

        Returns
        -------
        (is_pd, probability)
            is_pd       : True=파킨슨 의심, False=정상, None=실패
            probability : 파킨슨 확률 (0.0 ~ 1.0), None=실패
        """
        feats = extract_features(wav_path)
        if feats is None:
            return None, None

        feats_clean    = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats_selected = feats_clean[self.top_idx].reshape(1, -1)
        prob           = float(self.model.predict_proba(feats_selected)[0][1])

        return prob >= self.threshold, prob

    @property
    def is_available(self) -> bool:
        """모델 파일이 존재하면 True."""
        return (
            (MODEL_DIR / "voice_ensemble.pkl").exists()
            and (MODEL_DIR / "top_feature_idx.npy").exists()
        )
