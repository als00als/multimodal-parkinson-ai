"""
파킨슨 전구기 예측 서비스 — Streamlit 앱
=========================================
평가 지표
  1. 필적(나선 Spiral + 파도 Wave) — DualResNet18 (이중 입력 멀티모달)
  2. 음성(Voice)       — Soft Voting Ensemble (RF + SVM + LR)
  3. 후각(Olfactory)   — Soft Voting Ensemble (BSIT 12문항)
  4. 변비(Constipation)— Soft Voting Ensemble (SCOPA-AUT 3문항)

흐름
  [그림] 캔버스에 그리기 → 저장
  [음성] 마이크 녹음     → 자동 저장 → 즉시 예측
  [후각] 12문항 선택     → 응답 저장
  [변비] 3문항 선택      → 응답 저장
  [종합] 버튼 클릭 → 필적·후각·변비 일괄 산출 → 결과 + XAI 표시
"""

import io
import sys
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# voice_predictor / ppmi_predictor / spiral_predictor 는 같은 폴더에 위치
sys.path.insert(0, str(Path(__file__).parent))
from voice_predictor import VoicePredictor, extract_features
from spiral_predictor import SpiralPredictor
from ppmi_predictor import OlfactoryPredictor, ConstipationPredictor
from xai_utils import (
    waterfall_chart,
    explain_voice_shap,
    explain_olfactory_shap,
    explain_constipation_shap,
)

# ══════════════════════════════════════════════════════════════
# 경로 & 모델 설정
# ══════════════════════════════════════════════════════════════
BASE_DIR   = Path(r"C:\Project_AI\파킨슨 전구기 예측 서비스")
SAVE_DIR   = BASE_DIR / "streamlit" / "user_drawings"
SPIRAL_DIR = SAVE_DIR / "Spiral"
VOICE_DIR  = BASE_DIR / "streamlit" / "user_voice"

EXAMPLE_PATHS = {
    "Spiral": (BASE_DIR / "streamlit" / "dev_drawings" / "dev_V55HE14.png"),
}


# ══════════════════════════════════════════════════════════════
# 그림 모델 — Spiral 단일 입력
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_spiral_predictor():
    """SpiralPredictor 를 캐시하여 반환. 모델 없으면 None."""
    try:
        return SpiralPredictor()
    except FileNotFoundError:
        return None


def predict_spiral(spiral_image: Image.Image) -> tuple[bool, float]:
    """Spiral PIL Image → (is_parkinson: bool, prob: float)"""
    predictor = load_spiral_predictor()
    if predictor is None:
        return False, 0.0
    return predictor.predict(spiral_image)


# ══════════════════════════════════════════════════════════════
# 음성 모델 함수
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_voice_predictor():
    """VoicePredictor 를 캐시하여 반환. 모델 없으면 None."""
    try:
        return VoicePredictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_olfactory_predictor():
    try:
        return OlfactoryPredictor()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_constipation_predictor():
    try:
        return ConstipationPredictor()
    except FileNotFoundError:
        return None


# ══════════════════════════════════════════════════════════════
# UI 공통 헬퍼
# ══════════════════════════════════════════════════════════════
def risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "고위험"
    if prob >= 0.50:
        return "중위험"
    return "저위험"


def verdict_html(label: str, prob: float, is_parkinson: bool) -> str:
    bg = "#E53E3E" if is_parkinson else "#38A169"
    return (
        f'<div style="background:{bg};color:white;padding:16px 10px;'
        f'border-radius:12px;text-align:center;margin-top:8px">'
        f'<span style="font-size:1.5em;font-weight:bold">{label}</span><br>'
        f'<span style="font-size:0.95em;opacity:0.9">파킨슨 확률 {prob:.1%}</span>'
        f'</div>'
    )


def voice_result_html(label: str, prob: float, is_parkinson: bool) -> str:
    bg = "#E53E3E" if is_parkinson else "#38A169"
    return (
        f'<div style="background:{bg};color:white;padding:18px 14px;'
        f'border-radius:14px;text-align:center;margin-top:10px">'
        f'<div style="font-size:1.6em;font-weight:bold">{label}</div>'
        f'<div style="font-size:1.05em;opacity:0.92;margin-top:4px">'
        f'파킨슨 확률 {prob:.1%}</div>'
        f'<div style="font-size:0.88em;opacity:0.80;margin-top:2px">'
        f'위험도: {risk_level(prob)}</div>'
        f'</div>'
    )


def save_wav_bytes(audio_bytes: bytes, save_path: Path) -> bool:
    """
    브라우저에서 받은 오디오 바이트를 WAV 파일로 저장합니다.
    soundfile 으로 포맷 변환 후 저장 (WebM / WAV 모두 지원).
    """
    import soundfile as sf

    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        buf = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(buf)
        sf.write(str(save_path), data, samplerate)
        return True
    except Exception:
        # fallback: 바이트 그대로 저장
        try:
            with open(save_path, "wb") as f:
                f.write(audio_bytes)
            return True
        except Exception:
            return False


def audio_peak_level(audio_bytes: bytes) -> float:
    """
    오디오 바이트의 최대 진폭(0.0~1.0)을 반환합니다.
    무음(마이크 미입력) 녹음을 사전에 걸러내기 위해 사용.
    디코딩 실패 시 -1.0 을 반환합니다.
    """
    import soundfile as sf

    try:
        data, _ = sf.read(io.BytesIO(audio_bytes))
        arr = np.asarray(data, dtype=np.float64)
        return float(np.max(np.abs(arr))) if arr.size else 0.0
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="파킨슨 전구기 예측",
    page_icon="🧠",
    layout="wide",
)



# ══════════════════════════════════════════════════════════════
# 키오스크형 화면 구성 헬퍼
# ══════════════════════════════════════════════════════════════
def inject_kiosk_css():
    """큰 화면 키오스크 환경에 맞춘 Streamlit 스타일."""
    st.markdown(
        """
        <style>
        :root {
            --kiosk-primary: #2457F5;
            --kiosk-primary-dark: #183EA8;
            --kiosk-bg: #F4F7FB;
            --kiosk-card: #FFFFFF;
            --kiosk-text: #172033;
            --kiosk-muted: #697386;
            --kiosk-soft: #EAF0FF;
            --kiosk-border: #DDE5F2;
            --kiosk-success: #1F9D55;
            --kiosk-warning: #F59E0B;
            --kiosk-danger: #DC2626;
        }

        .stApp {
            background: linear-gradient(180deg, #F7FAFF 0%, #EEF4FF 48%, #F7FAFC 100%);
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }

        header[data-testid="stHeader"] { background: rgba(255,255,255,0); }
        #MainMenu, footer { visibility: hidden; }

        h1, h2, h3, h4, p, label, span, div {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
        }

        h1 { letter-spacing: -0.04em; }
        h2, h3 { letter-spacing: -0.035em; color: var(--kiosk-text); }

        .kiosk-hero {
            padding: 34px 38px;
            border-radius: 30px;
            background: radial-gradient(circle at 6% 0%, #E7EEFF 0%, #FFFFFF 42%, #FFFFFF 100%);
            border: 1px solid rgba(36, 87, 245, 0.12);
            box-shadow: 0 18px 45px rgba(31, 58, 147, 0.10);
            margin-bottom: 20px;
        }
        .hero-eyebrow {
            color: var(--kiosk-primary);
            font-weight: 900;
            font-size: 1.02rem;
            margin-bottom: 8px;
        }
        .hero-title {
            color: var(--kiosk-text);
            font-size: 3.15rem;
            line-height: 1.12;
            font-weight: 900;
            letter-spacing: -0.06em;
            margin: 0 0 10px 0;
        }
        .hero-subtitle {
            color: var(--kiosk-muted);
            font-size: 1.18rem;
            line-height: 1.65;
            max-width: 860px;
            margin: 0;
        }

        .flow-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin: 18px 0 30px 0;
        }
        .flow-card {
            min-height: 112px;
            padding: 17px 16px;
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--kiosk-border);
            border-radius: 22px;
            box-shadow: 0 10px 24px rgba(17, 24, 39, 0.055);
        }
        .flow-no {
            color: var(--kiosk-primary);
            font-size: 0.82rem;
            font-weight: 900;
            margin-bottom: 8px;
        }
        .flow-title {
            color: var(--kiosk-text);
            font-size: 1.02rem;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .flow-desc {
            color: var(--kiosk-muted);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .step-wrap {
            display: flex;
            align-items: center;
            gap: 16px;
            margin: 42px 0 18px 0;
        }
        .step-badge {
            flex: 0 0 auto;
            width: 56px;
            height: 56px;
            border-radius: 18px;
            display: flex;
            justify-content: center;
            align-items: center;
            background: var(--kiosk-primary);
            color: white;
            font-size: 1.04rem;
            font-weight: 900;
            box-shadow: 0 10px 22px rgba(36, 87, 245, 0.23);
        }
        .step-title {
            color: var(--kiosk-text);
            font-size: 1.72rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 4px;
        }
        .step-desc {
            color: var(--kiosk-muted);
            font-size: 1.03rem;
            line-height: 1.45;
        }

        .info-panel, .mini-panel, .result-panel {
            padding: 22px 24px;
            border-radius: 24px;
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--kiosk-border);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .mini-panel h4, .info-panel h4 {
            margin: 0 0 8px 0;
            color: var(--kiosk-text);
            font-size: 1.18rem;
            font-weight: 900;
        }
        .mini-panel p, .info-panel p {
            margin: 0;
            color: var(--kiosk-muted);
            line-height: 1.55;
            font-size: 0.98rem;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin: 10px 0 18px 0;
        }
        .status-card {
            padding: 18px 18px;
            border-radius: 22px;
            background: #FFFFFF;
            border: 1px solid var(--kiosk-border);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        }
        .status-title {
            color: var(--kiosk-muted);
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .status-value {
            color: var(--kiosk-text);
            font-size: 1.18rem;
            font-weight: 900;
            margin-bottom: 4px;
        }
        .status-help {
            color: var(--kiosk-muted);
            font-size: 0.84rem;
        }
        .status-ready { border-color: rgba(31, 157, 85, 0.35); background: #F0FFF6; }
        .status-wait { border-color: rgba(245, 158, 11, 0.30); background: #FFFBEB; }

        .kiosk-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 7px 12px;
            font-weight: 900;
            font-size: 0.86rem;
            margin-bottom: 10px;
        }
        .chip-ok { background: #E8F8EF; color: #137A3D; }
        .chip-wait { background: #FFF4D6; color: #92400E; }

        .stButton > button {
            min-height: 3.35rem;
            border-radius: 18px;
            font-size: 1.05rem;
            font-weight: 900;
            border: 0;
        }
        .stButton > button[kind="primary"] {
            background: var(--kiosk-primary);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            transition: 0.12s ease-in-out;
        }

        div[data-testid="stTextInput"] label {
            color: var(--kiosk-text) !important;
            font-weight: 800 !important;
        }
        div[data-testid="stTextInput"] input {
            min-height: 3.1rem;
            border-radius: 14px;
            font-size: 1.04rem;
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border: 1px solid #D6DDEA !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10) !important;
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #7B8598 !important;
            opacity: 1 !important;
        }
        div[data-testid="stTextInput"] > div {
            background: transparent !important;
        }

        div[data-testid="stAudioInput"] {
            background: #FFFFFF !important;
            border: 1px solid #D6DDEA !important;
            border-radius: 20px !important;
            padding: 14px 16px !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10) !important;
        }
        div[data-testid="stAudioInput"] label {
            color: var(--kiosk-text) !important;
            font-weight: 800 !important;
        }
        div[data-testid="stAudioInput"] button {
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border: 1px solid #D6DDEA !important;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08) !important;
        }
        div[data-testid="stAudioInput"] audio {
            background: #FFFFFF !important;
            border-radius: 14px !important;
            width: 100%;
        }

        div[role="radiogroup"] label {
            padding: 8px 10px;
            border-radius: 12px;
        }
        .stProgress > div > div > div > div {
            border-radius: 999px;
        }

        /* ──────────────────────────────────────────────
           입력/녹음 위젯 가독성 보정
           - Streamlit 기본 다크 테마가 섞여도 이름/전화번호/녹음 영역은 흰색 카드로 고정
           - 테두리 대신 부드러운 그림자로 구분
        ────────────────────────────────────────────── */
        div[data-testid="stTextInput"],
        div[data-testid="stAudioInput"] {
            color: var(--kiosk-text) !important;
        }

        div[data-testid="stTextInput"] label,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stAudioInput"] label,
        div[data-testid="stAudioInput"] label p {
            color: var(--kiosk-text) !important;
            font-weight: 900 !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"],
        div[data-testid="stTextInput"] div[data-baseweb="base-input"],
        div[data-testid="stTextInput"] input {
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
            border: 0 !important;
            border-radius: 14px !important;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.16),
                        0 0 0 1px rgba(214, 221, 234, 0.95) !important;
        }

        div[data-testid="stTextInput"] input {
            min-height: 3.25rem !important;
            font-size: 1.08rem !important;
            font-weight: 800 !important;
            caret-color: var(--kiosk-primary) !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #8A94A8 !important;
            -webkit-text-fill-color: #8A94A8 !important;
            opacity: 1 !important;
        }

        div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
        div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
            box-shadow: 0 12px 30px rgba(36, 87, 245, 0.18),
                        0 0 0 2px rgba(36, 87, 245, 0.35) !important;
        }

        div[data-testid="stAudioInput"] {
            background: #FFFFFF !important;
            border: 0 !important;
            border-radius: 22px !important;
            padding: 18px 18px !important;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16),
                        0 0 0 1px rgba(214, 221, 234, 0.95) !important;
        }

        div[data-testid="stAudioInput"] > div,
        div[data-testid="stAudioInput"] div,
        div[data-testid="stAudioInput"] section,
        div[data-testid="stAudioInput"] [data-testid="InputInstructions"],
        div[data-testid="stAudioInput"] [data-testid="stMarkdownContainer"] {
            background: transparent !important;
            color: var(--kiosk-text) !important;
        }

        div[data-testid="stAudioInput"] button,
        div[data-testid="stAudioInput"] [role="button"] {
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border: 0 !important;
            border-radius: 999px !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.16),
                        0 0 0 1px rgba(214, 221, 234, 0.95) !important;
        }

        div[data-testid="stAudioInput"] svg,
        div[data-testid="stAudioInput"] path {
            color: var(--kiosk-text) !important;
            fill: currentColor !important;
        }

        div[data-testid="stAudioInput"] span,
        div[data-testid="stAudioInput"] p {
            color: var(--kiosk-text) !important;
        }

        div[data-testid="stAudioInput"] audio {
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.10) !important;
            color-scheme: light !important;
        }

        /*
           오디오 입력 완료 후 오른쪽에 보이던 검은 박스는
           Streamlit 버튼이 아니라 브라우저 기본 audio 컨트롤의 시간 표시(예: 00:02)입니다.
           키오스크 화면에서는 불필요하고 버튼처럼 오해될 수 있어 숨기고,
           나머지 재생 컨트롤은 흰색 카드 톤으로 고정합니다.
        */
        html, body, .stApp,
        div[data-testid="stAudioInput"],
        div[data-testid="stAudioInput"] audio {
            color-scheme: light !important;
        }

        div[data-testid="stAudioInput"] audio::-webkit-media-controls-enclosure,
        div[data-testid="stAudioInput"] audio::-webkit-media-controls-panel {
            background-color: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border-radius: 16px !important;
        }

        div[data-testid="stAudioInput"] audio::-webkit-media-controls-play-button,
        div[data-testid="stAudioInput"] audio::-webkit-media-controls-mute-button {
            background-color: #FFFFFF !important;
            border-radius: 999px !important;
        }

        div[data-testid="stAudioInput"] audio::-webkit-media-controls-current-time-display,
        div[data-testid="stAudioInput"] audio::-webkit-media-controls-time-remaining-display {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            color: transparent !important;
            background: transparent !important;
            text-shadow: none !important;
        }

        div[data-testid="stAudioInput"] audio::-webkit-media-controls-timeline,
        div[data-testid="stAudioInput"] audio::-webkit-media-controls-volume-slider {
            background-color: transparent !important;
            accent-color: var(--kiosk-primary) !important;
        }

        /* 다크 테마가 섞일 때 흐려지는 체크박스/라디오/탭 글자 보정 */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p,
        div[role="radiogroup"] label,
        div[role="radiogroup"] label p,
        button[data-baseweb="tab"],
        button[data-baseweb="tab"] p {
            color: var(--kiosk-text) !important;
            opacity: 1 !important;
        }

        div[role="radiogroup"] label span,
        div[data-testid="stCheckbox"] label span {
            color: var(--kiosk-text) !important;
        }

        div[data-testid="stAlert"] div,
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {
            color: var(--kiosk-text) !important;
        }



        .voice-complete-card {
            background: #FFFFFF;
            border-radius: 24px;
            padding: 28px 30px;
            border: 1px solid rgba(214, 221, 234, 0.95);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
            margin-bottom: 14px;
        }
        .voice-complete-title {
            color: #137A3D;
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            margin-bottom: 8px;
        }
        .voice-complete-desc {
            color: var(--kiosk-text);
            font-size: 1.02rem;
            line-height: 1.55;
            margin-bottom: 10px;
        }
        .voice-file-name {
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: #F1F5F9;
            color: #475569;
            font-size: 0.88rem;
            font-weight: 800;
        }

        /* ──────────────────────────────────────────────
           글자 색상 가독성 보정
           - 문항 라벨과 Streamlit 기본 텍스트가 다크 테마 영향을 받아 흰색으로 보이는 문제 방지
           - 버튼/뱃지처럼 의도적으로 흰색이어야 하는 요소는 아래에서 다시 지정
        ────────────────────────────────────────────── */
        .question-label {
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
            font-size: 1.02rem;
            font-weight: 900;
            line-height: 1.35;
            margin: 12px 0 8px 0;
            opacity: 1 !important;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container h5,
        .block-container h6,
        .block-container p,
        .block-container li,
        .block-container label,
        .block-container label p,
        .block-container small,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaptionContainer"] *,
        div[data-testid="stMetric"],
        div[data-testid="stMetric"] *,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }

        div[role="radiogroup"],
        div[role="radiogroup"] *,
        div[data-baseweb="radio"],
        div[data-baseweb="radio"] * {
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
            opacity: 1 !important;
        }

        button[data-baseweb="tab"],
        button[data-baseweb="tab"] *,
        div[data-testid="stTabs"] *,
        div[data-testid="stTabs"] p {
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
            opacity: 1 !important;
        }

        /* 의도적으로 흰색 글자를 써야 하는 요소 복구 */
        .step-badge,
        .step-badge *,
        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[data-testid="baseButton-primary"] * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        @media (max-width: 980px) {
            .flow-grid, .status-grid { grid-template-columns: repeat(2, 1fr); }
            .hero-title { font-size: 2.35rem; }
        }
        

        /* ──────────────────────────────────────────────
           요청 수정 1) 음성 다시 녹음하기 버튼 전용 색상
        ────────────────────────────────────────────── */
        .voice-reset-button-marker + div[data-testid="stButton"] button,
        .voice-reset-button-marker ~ div[data-testid="stButton"] button {
            background: var(--kiosk-primary) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: 0 !important;
            box-shadow: 0 12px 26px rgba(36, 87, 245, 0.25) !important;
        }
        .voice-reset-button-marker + div[data-testid="stButton"] button:hover,
        .voice-reset-button-marker ~ div[data-testid="stButton"] button:hover {
            background: #7DD3FC !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: 0 !important;
        }

        /* ──────────────────────────────────────────────
           요청 수정 2) 상세 수치 expander / dataframe 밝은 배경 고정
        ────────────────────────────────────────────── */
        details[data-testid="stExpander"],
        div[data-testid="stExpander"] {
            background: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            border: 1px solid var(--kiosk-border) !important;
            border-radius: 18px !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08) !important;
        }
        details[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary,
        details[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary p,
        details[data-testid="stExpander"] div,
        details[data-testid="stExpander"] p,
        details[data-testid="stExpander"] span {
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] div,
        div[data-testid="stDataFrame"] span,
        div[data-testid="stDataFrame"] p {
            background-color: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
        }
        div[data-testid="stDataFrame"] [class*="ag-root"],
        div[data-testid="stDataFrame"] [class*="ag-header"],
        div[data-testid="stDataFrame"] [class*="ag-row"],
        div[data-testid="stDataFrame"] [class*="ag-cell"] {
            background-color: #FFFFFF !important;
            color: var(--kiosk-text) !important;
            -webkit-text-fill-color: var(--kiosk-text) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="kiosk-hero">
            <div class="hero-eyebrow">지역사회 기반 비침습 선별 서비스</div>
            <div class="hero-title">파킨슨 전구기<br>위험도 예측 키오스크</div>
            <p class="hero-subtitle">
                필적, 음성, 후각, 변비 문항을 순서대로 진행하면 AI가 위험도를 종합합니다.
                이 서비스는 확진이 아니라 조기 선별과 의료기관 방문 권유를 돕는 보조 도구입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_overview():
    steps = [
        ("01", "동의·정보", "안내 확인 후 이름과 연락처 입력"),
        ("02", "필적", "나선 그림을 따라 그리고 저장"),
        ("03", "음성", "'아~' 발성을 녹음하고 자동 분석"),
        ("04", "문항", "후각 12문항과 변비 3문항 응답"),
        ("05", "결과", "종합 위험도와 XAI 설명 확인"),
    ]
    html = '<div class="flow-grid">'
    for no, title, desc in steps:
        html += (
            f'<div class="flow-card">'
            f'<div class="flow-no">STEP {no}</div>'
            f'<div class="flow-title">{title}</div>'
            f'<div class="flow-desc">{desc}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_step_header(no: str, title: str, desc: str):
    st.markdown(
        f"""
        <div class="step-wrap">
            <div class="step-badge">{no}</div>
            <div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_grid(items):
    html = '<div class="status-grid">'
    for title, value, helper, ready in items:
        cls = "status-ready" if ready else "status-wait"
        html += (
            f'<div class="status-card {cls}">'
            f'<div class="status-title">{title}</div>'
            f'<div class="status-value">{value}</div>'
            f'<div class="status-help">{helper}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_panel(title: str, body: str):
    st.markdown(
        f"""
        <div class="info-panel">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rerun_app():
    """Streamlit 버전에 따라 화면을 즉시 다시 그립니다."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════
def main():
    inject_kiosk_css()

    # ── Session state 초기화 ──────────────────────────────────
    defaults = {
        "spiral_saved"        : False,
        "results"             : None,   # Spiral 필적 결과
        "voice_saved"         : False,
        "voice_path"          : None,   # 저장된 음성 경로
        "voice_audio_sig"     : None,   # 마지막으로 처리한 녹음 바이트 해시(중복 처리 방지)
        "voice_input_version"  : 0,      # 녹음 위젯을 초기화하기 위한 버전 키
        "voice_result"        : None,   # {"is_pd": bool, "prob": float, "fname": str}
        "olfactory_result"    : None,   # {"is_pd": bool, "prob": float, "scores": list}
        "constipation_result" : None,   # {"is_pd": bool, "prob": float, "aut5": int, ...}
        "olf_scores"          : None,   # 후각 12문항 응답(0/1) 리스트
        "olf_total"           : None,   # 후각 총점
        "scopa_vals"          : None,   # (aut5, aut6, aut7, total) 변비 응답
        "spiral_gradcam"      : None,   # Grad-CAM 오버레이 PIL Image
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── 헤더 / 진행 흐름 ──────────────────────────────────────
    render_hero()
    render_flow_overview()

    # ══════════════════════════════════════════════════════════
    # 01. 안내 및 동의
    # ══════════════════════════════════════════════════════════
    render_step_header("01", "서비스 안내 및 본인 확인", "검사 전 안내를 확인하고 최소한의 이용 정보를 입력합니다.")

    with st.expander("서비스 안내와 주의사항 자세히 보기", expanded=True):
        st.info(
            """
• 이 서비스는 **파킨슨병 확진**을 위한 것이 아닙니다.  
• 본 앱은 **위험도 선별**과 **조기 검사 권유**를 지원하는 보조 도구입니다.  
• 제공된 개인정보(이름, 전화번호)는 서비스 이용 기록 관리와 결과 확인을 위해서만 사용됩니다.  
• 결과는 AI 예측 값이며, **임상 진단이 아니며 오차가 있을 수 있습니다**.  
• 결과가 높게 나올 경우 반드시 **전문 의료기관을 방문하여 추가 진료**를 받으세요.
            """
        )

    consent_given = st.checkbox(
        "위 안내 내용을 확인하였으며, 본 서비스가 확진 목적이 아니라 위험도 선별 목적임을 이해합니다.",
        key="consent_given",
    )
    if not consent_given:
        st.warning("서비스를 이용하려면 안내 및 동의 문구를 확인하고 체크박스를 선택해 주세요.")
        st.stop()

    user_col, guide_col = st.columns([1.15, 0.85])
    with user_col:
        col_n, col_p = st.columns(2)
        user_name_input   = col_n.text_input("이름", placeholder="예: 김수민")
        user_number_input = col_p.text_input("전화번호", placeholder="예: 010-0000-0000")
        if user_name_input:
            st.success(f"안녕하세요, {user_name_input}님. 아래 순서대로 검사를 진행해 주세요.")
    with guide_col:
        render_panel(
            "키오스크 사용 팁",
            "화면의 STEP 순서대로 진행하세요. 저장 또는 녹음 완료 표시가 뜨면 다음 검사로 이동하면 됩니다.",
        )

    # ══════════════════════════════════════════════════════════
    # 02. 필적 검사
    # ══════════════════════════════════════════════════════════
    render_step_header("02", "필적 검사", "예시 나선을 보고 오른쪽 그림판에 최대한 비슷하게 그린 뒤 저장합니다.")

    sp_ex, sp_draw = st.columns([1, 1.05], gap="large")

    with sp_ex:
        st.markdown('<span class="kiosk-chip chip-wait">예시 확인</span>', unsafe_allow_html=True)
        render_panel("따라 그릴 나선", "나선의 시작점, 회전 간격, 선 흔들림을 자연스럽게 그려 주세요.")
        p_spiral = EXAMPLE_PATHS["Spiral"]
        if p_spiral.exists():
            st.image(Image.open(p_spiral), width=500)
        else:
            st.info("나선 예시 이미지를 찾을 수 없습니다.")

    with sp_draw:
        st.markdown(
            '<span class="kiosk-chip chip-ok">저장 완료</span>' if st.session_state.spiral_saved else '<span class="kiosk-chip chip-wait">그림 대기</span>',
            unsafe_allow_html=True,
        )
        render_panel("나선 그리기", "회색 영역 안에서 마우스 또는 터치로 한 번에 그려 주세요. 펜 굵기는 검사 일관성을 위해 고정되어 있습니다.")
        stroke_width_spiral = 4
        stroke_color_spiral = "#4E4D51"

        canvas_result_spiral = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width_spiral,
            stroke_color=stroke_color_spiral,
            background_color="#DCDCDC",
            height=500, width=500,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="spiral_canvas",
        )

        if canvas_result_spiral.image_data is not None:
            spiral_drawn = Image.fromarray(
                canvas_result_spiral.image_data.astype(np.uint8)
            ).convert("RGB")

            if st.button("나선 그림 저장하기", key="btn_save_spiral", type="primary", use_container_width=True):
                SPIRAL_DIR.mkdir(parents=True, exist_ok=True)
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = user_name_input if user_name_input else "unknown"
                spiral_drawn.save(SPIRAL_DIR / f"{name}_{ts}.png")
                st.session_state.spiral_saved = True
                st.session_state.results = None
                st.success(f"나선 그림 저장 완료: {SPIRAL_DIR / f'{name}_{ts}.png'}")

        if st.session_state.spiral_saved:
            st.success("나선 그림 저장 완료 ✅")

    # ══════════════════════════════════════════════════════════
    # 03. 음성 검사
    # ══════════════════════════════════════════════════════════
    render_step_header("03", "음성 검사", "마이크 버튼을 누르고 '아~' 소리를 3초 이상 일정하게 녹음합니다.")

    voice_predictor = load_voice_predictor()
    voice_guide_col, voice_input_col = st.columns([0.95, 1.05], gap="large")

    with voice_guide_col:
        render_panel(
            "녹음 방법",
            "마이크 권한을 허용한 뒤 조용한 환경에서 '아~'를 일정하게 발성하세요. 정지하면 자동 저장 및 분석됩니다.",
        )
        st.markdown(
            """
            - 녹음 시간: **3초 이상**
            - 권장 환경: 주변 소음이 적은 곳
            - 무음으로 판단되면 입력 장치와 브라우저 마이크 권한을 확인
            """
        )

    with voice_input_col:
        if voice_predictor is None:
            st.warning(
                "음성 모델이 준비되지 않았습니다.  \n"
                "`save_voice_model.py` 를 먼저 실행하여 모델을 학습시켜 주세요."
            )
        else:
            voice_done = st.session_state.voice_saved and st.session_state.voice_result is not None

            if voice_done:
                st.markdown('<span class="kiosk-chip chip-ok">녹음 완료</span>', unsafe_allow_html=True)
                saved_name = st.session_state.voice_result.get("fname", "저장된 음성 파일")
                st.markdown(
                    f"""
                    <div class="voice-complete-card">
                        <div class="voice-complete-title">✅ 음성 녹음이 저장되었습니다.</div>
                        <div class="voice-complete-desc">
                            녹음 파일은 정상적으로 저장되었고, 음성 분석도 완료되었습니다.<br>
                            결과 확률과 SHAP 설명은 아래 <b>종합 예측</b> 단계에서 함께 확인하세요.
                        </div>
                        <div class="voice-file-name">파일명: {saved_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="voice-reset-button-marker"></div>', unsafe_allow_html=True)
                if st.button("음성 다시 녹음하기", key="btn_reset_voice", use_container_width=True):
                    st.session_state.voice_saved = False
                    st.session_state.voice_path = None
                    st.session_state.voice_audio_sig = None
                    st.session_state.voice_result = None
                    st.session_state.voice_input_version += 1
                    rerun_app()

            else:
                st.markdown('<span class="kiosk-chip chip-wait">녹음 대기</span>', unsafe_allow_html=True)
                audio_file = st.audio_input(
                    "마이크 버튼을 눌러 '아~' 소리를 녹음하세요",
                    key=f"voice_mic_{st.session_state.voice_input_version}",
                )

                if audio_file is not None:
                    audio_bytes: bytes = audio_file.getvalue()
                    sig = hashlib.md5(audio_bytes).hexdigest()

                    if sig != st.session_state.voice_audio_sig:
                        st.session_state.voice_audio_sig = sig

                        peak = audio_peak_level(audio_bytes)
                        if 0.0 <= peak < 0.01:
                            st.error(
                                "🔇 녹음된 소리가 거의 없습니다 (무음).  \n"
                                "마이크 권한은 허용됐지만 **실제 입력 신호가 0**입니다.  \n\n"
                                "**확인해 주세요:**  \n"
                                "1. Windows 설정 → 시스템 → 소리 → **입력 장치**가 사용 중인 마이크로 선택돼 있는지  \n"
                                "2. 입력 장치 **볼륨이 0이거나 음소거**가 아닌지  \n"
                                "3. 브라우저 주소창의 🎙️ 아이콘 → **올바른 마이크 장치** 선택  \n"
                                "4. 다른 앱(줌·팀즈 등)이 마이크를 점유하고 있지 않은지  \n"
                                "5. 노트북 내장 마이크가 **하드웨어 음소거** 상태가 아닌지"
                            )
                            st.session_state.voice_saved  = False
                            st.session_state.voice_result = None
                        else:
                            VOICE_DIR.mkdir(parents=True, exist_ok=True)
                            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                            name = user_name_input if user_name_input else "unknown"
                            save_path = VOICE_DIR / f"{name}_{ts}.wav"

                            if save_wav_bytes(audio_bytes, save_path):
                                st.session_state.voice_saved = True
                                st.session_state.voice_path  = save_path

                                with st.spinner("음성 분석 중..."):
                                    is_pd, prob = voice_predictor.predict(save_path)

                                if prob is None:
                                    st.warning(
                                        "음성 분석에 실패했습니다.  \n"
                                        "녹음 시간이 너무 짧거나 음성이 인식되지 않았습니다.  \n"
                                        "다시 녹음해 주세요."
                                    )
                                    st.session_state.voice_saved = False
                                    st.session_state.voice_result = None
                                else:
                                    st.session_state.voice_result = {
                                        "is_pd": is_pd,
                                        "prob" : prob,
                                        "fname": save_path.name,
                                    }
                                    # 같은 실행 흐름에서 이미 출력된 st.audio_input UI를 제거하기 위해 즉시 재실행합니다.
                                    # 다음 화면에서는 검은 시간 표시가 있는 기본 녹음 컨트롤 대신 녹음 완료 카드가 표시됩니다.
                                    rerun_app()
                            else:
                                st.error("음성 파일 저장에 실패했습니다. 다시 시도해주세요.")

                if st.session_state.voice_saved and st.session_state.voice_result is not None:
                    st.success("음성 저장 완료 ✅ — 종합 예측에서 분석 결과를 확인하세요.")

    # ══════════════════════════════════════════════════════════
    # 04. 문항 검사
    # ══════════════════════════════════════════════════════════
    render_step_header("04", "후각 및 변비 문항", "선택지만 누르면 되는 문항형 검사입니다. 모르는 후각 항목은 '못 맡겠다'를 선택합니다.")

    olf_predictor = load_olfactory_predictor()
    con_predictor = load_constipation_predictor()

    q_tab_olf, q_tab_con = st.tabs(["👃 후각 기능 평가", "🚽 변비 증상 평가"])

    with q_tab_olf:
        render_panel(
            "B-SIT 12문항",
            "각 향기에 대해 평소 해당 냄새를 맡을 수 있는지 선택하세요. 응답은 저장되며 종합 예측 버튼을 누를 때 결과가 산출됩니다.",
        )
        if olf_predictor is None:
            st.warning(
                "후각 모델이 준비되지 않았습니다.  \n"
                "`PPMI/train_olfactory_model.py` 를 먼저 실행하세요."
            )
        else:
            BSIT_ITEMS = [
                ("BSIT_CHERRY",       "🍒 체리 (Cherry)"),
                ("BSIT_DILL_PICKLE",  "🥒 딜 피클 (Dill Pickle)"),
                ("BSIT_BANANA",       "🍌 바나나 (Banana)"),
                ("BSIT_CHOCOLATE",    "🍫 초콜릿 (Chocolate)"),
                ("BSIT_CINNAMON",     "🌿 계피 (Cinnamon)"),
                ("BSIT_GASOLINE",     "⛽ 휘발유 (Gasoline)"),
                ("BSIT_LEMON",        "🍋 레몬 (Lemon)"),
                ("BSIT_ONION",        "🧅 양파 (Onion)"),
                ("BSIT_PINEAPPLE",    "🍍 파인애플 (Pineapple)"),
                ("BSIT_ROSE",         "🌹 장미 (Rose)"),
                ("BSIT_SOAP",         "🧼 비누 (Soap)"),
                ("BSIT_SMOKE",        "🔥 연기 (Smoke)"),
            ]

            olf_scores = []
            olf_col1, olf_col2 = st.columns(2, gap="large")
            for i, (key, label) in enumerate(BSIT_ITEMS):
                col = olf_col1 if i % 2 == 0 else olf_col2
                with col:
                    st.markdown(
                        f'<div class="question-label">{label}</div>',
                        unsafe_allow_html=True,
                    )
                    ans = st.radio(
                        "후각 문항 응답 선택",
                        options=["잘 맡을 수 있다", "못 맡겠다"],
                        index=0,
                        horizontal=True,
                        key=f"bsit_{key}",
                        label_visibility="collapsed",
                    )
                    olf_scores.append(1 if ans == "잘 맡을 수 있다" else 0)

            bsit_total = sum(olf_scores)
            st.progress(bsit_total / 12)
            st.caption(f"현재 후각 점수: **{bsit_total} / 12**  (맡을 수 있다고 선택한 항목 수)")

            st.session_state.olf_scores = olf_scores
            st.session_state.olf_total  = bsit_total

    with q_tab_con:
        render_panel(
            "SCOPA-AUT 3문항",
            "지난 한 달 동안 경험한 배변 관련 증상을 선택하세요. 응답은 저장되며 종합 예측 버튼을 누를 때 결과가 산출됩니다.",
        )
        if con_predictor is None:
            st.warning(
                "변비 모델이 준비되지 않았습니다.  \n"
                "`PPMI/train_constipation_model.py` 를 먼저 실행하세요."
            )
        else:
            SCOPA_OPTIONS = ["전혀 없다 (0)", "가끔 있다 (1)", "자주 있다 (2)", "항상 그렇다 (3)"]

            sc1, sc2, sc3 = st.columns(3, gap="large")
            with sc1:
                st.markdown(
                    '<div class="question-label">🟡 변비로 문제가 있었나요?</div>',
                    unsafe_allow_html=True,
                )
                aut5_ans = st.radio(
                    "변비 문항 1 응답 선택",
                    options=SCOPA_OPTIONS,
                    index=0,
                    key="scopa_aut5",
                    label_visibility="collapsed",
                )
            with sc2:
                st.markdown(
                    '<div class="question-label">🟠 배변 시 심하게 힘을 주어야 했나요?</div>',
                    unsafe_allow_html=True,
                )
                aut6_ans = st.radio(
                    "변비 문항 2 응답 선택",
                    options=SCOPA_OPTIONS,
                    index=0,
                    key="scopa_aut6",
                    label_visibility="collapsed",
                )
            with sc3:
                st.markdown(
                    '<div class="question-label">🔴 본인 의지와 무관하게 변이 나온 적이 있나요? (변실금)</div>',
                    unsafe_allow_html=True,
                )
                aut7_ans = st.radio(
                    "변비 문항 3 응답 선택",
                    options=SCOPA_OPTIONS,
                    index=0,
                    key="scopa_aut7",
                    label_visibility="collapsed",
                )

            aut5 = SCOPA_OPTIONS.index(aut5_ans)
            aut6 = SCOPA_OPTIONS.index(aut6_ans)
            aut7 = SCOPA_OPTIONS.index(aut7_ans)
            scopa_total = aut5 + aut6 + aut7
            st.progress(scopa_total / 9)
            st.caption(f"현재 변비 총점: **{scopa_total} / 9**")

            st.session_state.scopa_vals = (aut5, aut6, aut7, scopa_total)

    # ══════════════════════════════════════════════════════════
    # 05. 종합 예측
    # ══════════════════════════════════════════════════════════
    render_step_header("05", "파킨슨 위험도 종합 예측", "준비 상태를 확인한 뒤 버튼을 누르면 필적·음성·후각·변비 결과를 한 번에 산출합니다.")

    spiral_ok = st.session_state.spiral_saved
    voice_ok  = st.session_state.voice_result is not None
    olf_ready = olf_predictor is not None and st.session_state.olf_scores is not None
    con_ready = con_predictor is not None and st.session_state.scopa_vals is not None

    render_status_grid([
        ("나선(Spiral)", "저장 완료 ✅" if spiral_ok else "미저장", "그림을 저장해 주세요" if not spiral_ok else "준비됨", spiral_ok),
        ("음성(Voice)", "녹음 완료 ✅" if voice_ok else "미녹음", "음성을 녹음해 주세요" if not voice_ok else "준비됨", voice_ok),
        ("후각(Olfactory)", "응답 완료 ✅" if olf_ready else "미완료", "문항을 작성해 주세요" if not olf_ready else "준비됨", olf_ready),
        ("변비(Constipation)", "응답 완료 ✅" if con_ready else "미완료", "문항을 작성해 주세요" if not con_ready else "준비됨", con_ready),
    ])

    if not spiral_ok:
        st.warning("나선(Spiral) 그림을 먼저 저장해주세요. 필적 검사가 최소 필수 입력입니다.")

    predict_clicked = st.button(
        "🧠 파킨슨 위험도 종합 예측하기",
        type="primary",
        use_container_width=True,
        disabled=not spiral_ok,
    )

    if predict_clicked:
        spiral_files = sorted(SPIRAL_DIR.glob("*.png"))

        if not spiral_files:
            st.error("저장된 Spiral 이미지를 찾을 수 없습니다.")
        else:
            with st.spinner("AI 모델이 분석 중입니다..."):
                try:
                    # 1) 필적 (Spiral)
                    latest_spiral = Image.open(spiral_files[-1])
                    is_pd_spiral, spiral_prob = predict_spiral(latest_spiral)
                    st.session_state.results = {
                        "spiral"      : spiral_prob,
                        "spiral_file" : spiral_files[-1].name,
                    }

                    # 2) 후각 (B-SIT 12문항 응답 기반)
                    if olf_ready:
                        olf_scores_now = st.session_state.olf_scores
                        is_pd_o, prob_o = olf_predictor.predict(olf_scores_now)
                        st.session_state.olfactory_result = {
                            "is_pd"  : is_pd_o,
                            "prob"   : prob_o,
                            "scores" : olf_scores_now,
                            "total"  : st.session_state.olf_total,
                        }

                    # 3) 변비 (SCOPA-AUT 3문항 응답 기반)
                    if con_ready:
                        a5, a6, a7, c_total = st.session_state.scopa_vals
                        is_pd_c, prob_c = con_predictor.predict(a5, a6, a7)
                        st.session_state.constipation_result = {
                            "is_pd": is_pd_c,
                            "prob" : prob_c,
                            "aut5" : a5,
                            "aut6" : a6,
                            "aut7" : a7,
                            "total": c_total,
                        }
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {e}")

    # ══════════════════════════════════════════════════════════
    # 예측 결과 표시
    # ══════════════════════════════════════════════════════════
    voice_ok        = st.session_state.voice_result is not None
    olfactory_ok    = st.session_state.olfactory_result is not None
    constipation_ok = st.session_state.constipation_result is not None

    if st.session_state.results:
        r  = st.session_state.results
        spiral_prob = r["spiral"]

        spiral_predictor = load_spiral_predictor()
        spiral_thr = spiral_predictor.threshold if spiral_predictor else 0.5
        spiral_pos = spiral_prob >= spiral_thr

        st.markdown("---")
        render_step_header("결과", "예측 결과", "종합 위험도와 검사 항목별 세부 수치를 확인합니다.")

        # ── 최종 종합 판정 (가용 지표 모두 포함) ─────────────
        extra_probs  = []
        extra_labels = []
        if voice_ok:
            extra_probs.append(st.session_state.voice_result["prob"])
            extra_labels.append("음성")
        if olfactory_ok:
            extra_probs.append(st.session_state.olfactory_result["prob"])
            extra_labels.append("후각")
        if constipation_ok:
            extra_probs.append(st.session_state.constipation_result["prob"])
            extra_labels.append("변비")

        if extra_probs:
            all_probs     = [spiral_prob] + extra_probs
            combined_prob = sum(all_probs) / len(all_probs)
            combined_pos  = combined_prob >= 0.55

            modality_str = "필적 + " + " + ".join(extra_labels)
            detail_parts = (
                [f"필적 {spiral_prob:.1%}"]
                + [f"{lbl} {p:.1%}" for lbl, p in zip(extra_labels, extra_probs)]
            )

            st.markdown(f"### 🏆 최종 종합 판정 ({modality_str})")

            fv1, fv2 = st.columns([0.9, 2.1], gap="large")
            with fv1:
                st.progress(float(combined_prob))
                st.metric("종합 파킨슨 확률", f"{combined_prob:.1%}")
            with fv2:
                combined_label = "파킨슨 의심" if combined_pos else "정상"
                st.markdown(
                    f'<div class="result-panel" style="background:{"#FEF2F2" if combined_pos else "#F0FFF6"};border-color:{"#FCA5A5" if combined_pos else "#A7F3D0"};">'
                    f'<div style="font-size:2.2em;font-weight:900;color:{"#B91C1C" if combined_pos else "#047857"};">{combined_label}</div>'
                    f'<div style="font-size:1.04em;color:#374151;margin-top:8px;">{" + ".join(detail_parts)}</div>'
                    f'<div style="font-size:0.96em;color:#6B7280;margin-top:6px;">위험도: {risk_level(combined_prob)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            modality_prob_map = {"✍️ 필적": spiral_prob}
            label_map = {"음성": "🎤 음성", "후각": "👃 후각", "변비": "🚽 변비"}
            for lbl, p in zip(extra_labels, extra_probs):
                modality_prob_map[label_map.get(lbl, lbl)] = p
            st.plotly_chart(
                waterfall_chart(modality_prob_map),
                use_container_width=True,
            )
            st.caption("기준선보다 높은 모달리티가 종합 위험도를 끌어올린 주요 신호입니다.")
        else:
            st.info("음성·후각·변비 결과가 없어서 현재는 필적 결과만 산출되었습니다.")

        # ── 상세 수치 테이블 ──────────────────────────────────
        with st.expander("상세 수치 보기", expanded=True):
            import pandas as pd
            rows = [
                {
                    "모델"      : "✍️ 필적 (Spiral)",
                    "파킨슨 확률": f"{spiral_prob:.1%}",
                    "임계값"    : round(spiral_thr, 3),
                    "판정"      : "파킨슨 의심" if spiral_pos else "정상",
                    "위험도"    : risk_level(spiral_prob),
                },
            ]
            if voice_ok:
                vp = st.session_state.voice_result["prob"]
                rows.append({
                    "모델"      : "🎤 음성",
                    "파킨슨 확률": f"{vp:.1%}",
                    "임계값"    : 0.50,
                    "판정"      : "파킨슨 의심" if vp >= 0.50 else "정상",
                    "위험도"    : risk_level(vp),
                })
            if olfactory_ok:
                op = st.session_state.olfactory_result["prob"]
                ot = olf_predictor.threshold if olf_predictor else 0.566
                rows.append({
                    "모델"      : "👃 후각",
                    "파킨슨 확률": f"{op:.1%}",
                    "임계값"    : round(ot, 3),
                    "판정"      : "파킨슨 의심" if op >= ot else "정상",
                    "위험도"    : risk_level(op),
                })
            if constipation_ok:
                cp = st.session_state.constipation_result["prob"]
                ct = con_predictor.threshold if con_predictor else 0.731
                rows.append({
                    "모델"      : "🚽 변비",
                    "파킨슨 확률": f"{cp:.1%}",
                    "임계값"    : round(ct, 3),
                    "판정"      : "파킨슨 의심" if cp >= ct else "정상",
                    "위험도"    : risk_level(cp),
                })
            if extra_probs:
                combined = sum([spiral_prob] + extra_probs) / (1 + len(extra_probs))
                rows.append({
                    "모델"      : "🏆 최종 종합",
                    "파킨슨 확률": f"{combined:.1%}",
                    "임계값"    : 0.55,
                    "판정"      : "파킨슨 의심" if combined >= 0.55 else "정상",
                    "위험도"    : risk_level(combined),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── XAI 설명 (모달리티별) ─────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔬 모달리티별 XAI 설명")
        st.caption("각 검사 항목이 파킨슨 위험도 판정에 어떻게 기여했는지를 설명합니다.")

        # 나선 XAI 영역은 항상 표시합니다.
        # 기존 코드는 spiral_gradcam 값이 None이면 expander 자체가 보이지 않았습니다.
        # 현재 파일에는 Grad-CAM 이미지를 생성해서 spiral_gradcam에 넣는 코드가 연결되어 있지 않으므로,
        # 생성 결과가 없을 때도 사용자에게 원인과 대체 화면을 보여줍니다.
        with st.expander("🔍 나선(Spiral) XAI / Grad-CAM", expanded=True):
            if st.session_state.spiral_gradcam is not None:
                st.image(
                    st.session_state.spiral_gradcam,
                    caption="빨간 영역: 파킨슨 위험 판정에 기여한 부분",
                    use_container_width=True,
                )
            else:
                st.warning(
                    "나선 Grad-CAM 이미지가 아직 생성되지 않았습니다. "
                    "현재 코드에는 Grad-CAM 생성 함수를 호출해 `st.session_state.spiral_gradcam`에 저장하는 부분이 연결되어 있지 않습니다."
                )
                try:
                    latest_spiral_for_xai = Image.open(SPIRAL_DIR / r["spiral_file"])
                    st.image(
                        latest_spiral_for_xai,
                        caption="현재 분석에 사용된 나선 입력 이미지입니다. Grad-CAM 생성 함수가 연결되면 이 위치에 XAI 히트맵이 표시됩니다.",
                        use_container_width=True,
                    )
                except Exception:
                    st.info("분석에 사용된 나선 이미지를 불러오지 못했습니다.")

        if voice_ok and st.session_state.get("voice_path"):
            with st.expander("🎤 음성 — 특징 기여도 (SHAP)"):
                _vp = load_voice_predictor()
                if _vp is not None:
                    with st.spinner("SHAP 계산 중 (약 10~20초)..."):
                        try:
                            raw_feats = extract_features(st.session_state.voice_path)
                            if raw_feats is not None:
                                raw_feats = np.nan_to_num(
                                    raw_feats, nan=0.0, posinf=0.0, neginf=0.0
                                )
                                shap_fig = explain_voice_shap(
                                    _vp.model, _vp.top_idx, raw_feats
                                )
                                st.plotly_chart(shap_fig, use_container_width=True)
                                st.caption(
                                    "양수(빨강): 해당 특징이 파킨슨 위험도를 높임  |  "
                                    "음수(초록): 위험도를 낮춤"
                                )
                        except Exception as e:
                            st.warning(f"SHAP 계산 실패: {e}")

        if olfactory_ok:
            _or = st.session_state.olfactory_result
            with st.expander("👃 후각 — SHAP 특징 기여도"):
                _olf_p = load_olfactory_predictor()
                if _olf_p is not None:
                    with st.spinner("SHAP 계산 중 (약 10~20초)..."):
                        try:
                            shap_fig_o = explain_olfactory_shap(_olf_p.model, _or["scores"])
                            st.plotly_chart(shap_fig_o, use_container_width=True)
                            st.caption(
                                "양수(빨강): 해당 항목이 파킨슨 위험도를 높임  |  "
                                "음수(초록): 위험도를 낮춤"
                            )
                        except Exception as e:
                            st.warning(f"SHAP 계산 실패: {e}")

        if constipation_ok:
            _cr = st.session_state.constipation_result
            with st.expander("🚽 변비 — SHAP 특징 기여도"):
                _con_p = load_constipation_predictor()
                if _con_p is not None:
                    with st.spinner("SHAP 계산 중 (약 10~20초)..."):
                        try:
                            shap_fig_c = explain_constipation_shap(
                                _con_p.model, _cr["aut5"], _cr["aut6"], _cr["aut7"]
                            )
                            st.plotly_chart(shap_fig_c, use_container_width=True)
                            st.caption(
                                "양수(빨강): 해당 항목이 파킨슨 위험도를 높임  |  "
                                "음수(초록): 위험도를 낮춤"
                            )
                        except Exception as e:
                            st.warning(f"SHAP 계산 실패: {e}")

        st.info(
            "⚠️ 이 결과는 AI 스크리닝 보조 도구입니다.  \n"
            "임상 진단은 반드시 **전문 의료진**에게 받으세요."
        )


if __name__ == "__main__":
    main()
