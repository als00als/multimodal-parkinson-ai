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
# 메인
# ══════════════════════════════════════════════════════════════
def main():
    # ── 헤더 ──────────────────────────────────────────────────
    st.title("🧠 파킨슨 전구기 예측 서비스")
    st.write("4가지 전구기 평가 지표를 통해 사용자의 파킨슨 위험도를 예측합니다.")

    st.subheader("🔔 서비스 안내 및 동의")
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
        st.warning(
            "서비스를 이용하려면 위 안내 및 동의 문구를 확인하고 체크박스를 선택해 주세요."
        )
        st.stop()

    # ── 사용자 정보 ───────────────────────────────────────────
    col_n, col_p = st.columns(2)
    user_name_input   = col_n.text_input("이름을 입력하세요:")
    user_number_input = col_p.text_input("전화번호를 입력하세요:")
    if user_name_input:
        st.success(f"안녕하세요, {user_name_input}님!")

    st.divider()

    # ── 필적 검사 안내 ────────────────────────────────────────
    st.subheader("✏️ 필적 검사")

    # ── Session state 초기화 ──────────────────────────────────
    defaults = {
        "spiral_saved"        : False,
        "results"             : None,   # Spiral 필적 결과
        "voice_saved"         : False,
        "voice_path"          : None,   # 저장된 음성 경로
        "voice_audio_sig"     : None,   # 마지막으로 처리한 녹음 바이트 해시(중복 처리 방지)
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

    # ══════════════════════════════════════════════════════════
    # 나선(Spiral) — 좌: 예시 이미지 / 우: 그리기 캔버스
    # ══════════════════════════════════════════════════════════
    st.subheader("🖼️ 나선 그림 그리기")

    sp_ex, sp_draw = st.columns(2)

    with sp_ex:
        st.markdown("# 아래 그림을 참고해주세요.")
        p_spiral = EXAMPLE_PATHS["Spiral"]
        if p_spiral.exists():
            st.image(Image.open(p_spiral), width = 500)
        else:
            st.info("나선 예시 이미지를 찾을 수 없습니다.")
    
    with sp_draw:    
        st.markdown("# 아래 그림판에 나선을 그려주세요.")
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

            if st.button("나선 그림 저장하기", key="btn_save_spiral"):
                SPIRAL_DIR.mkdir(parents=True, exist_ok=True)
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = user_name_input if user_name_input else "unknown"
                spiral_drawn.save(SPIRAL_DIR / f"{name}_{ts}.png")
                st.session_state.spiral_saved = True
                st.session_state.results = None
                st.success(f"나선 그림 저장 완료: {SPIRAL_DIR / f'{name}_{ts}.png'}")

        if st.session_state.spiral_saved:
            st.success("나선 그림 저장 완료 ✅")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 음성 녹음 & 예측
    # ══════════════════════════════════════════════════════════
    st.subheader("🎙️ 음성 녹음")

    voice_predictor = load_voice_predictor()
    if voice_predictor is None:
        st.warning(
            "음성 모델이 준비되지 않았습니다.  \n"
            "`save_voice_model.py` 를 먼저 실행하여 모델을 학습시켜 주세요."
        )
    else:
        # 안내 메시지
        st.info(
            "📌 **녹음 방법**  \n"
            "1. 아래 🎙️ 버튼을 눌러 녹음을 시작하세요.  \n"
            "2. **'아~'** 소리를 **3초 이상** 일정하게 내주세요.  \n"
            "3. 정지 버튼을 누르면 자동으로 저장 및 분석됩니다.  \n"
            "※ 브라우저가 마이크 권한을 요청하면 **허용**을 눌러주세요."
        )

        # ── 녹음 컴포넌트 (Streamlit 내장 마이크 위젯) ─────────
        audio_file = st.audio_input(
            "🎙️ 마이크 버튼을 눌러 '아~' 소리를 녹음하세요",
            key="voice_mic",
        )

        # ── 녹음 완료 시: 저장 → 예측 (새 녹음일 때만 1회 처리) ─
        if audio_file is not None:
            audio_bytes: bytes = audio_file.getvalue()
            sig = hashlib.md5(audio_bytes).hexdigest()

            # 동일 녹음이 재실행마다 재처리되지 않도록 해시로 중복 차단
            if sig != st.session_state.voice_audio_sig:
                st.session_state.voice_audio_sig = sig

                # 무음 녹음 사전 차단 (마이크 입력이 실제로 들어왔는지 확인)
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

                        # 즉시 예측
                        with st.spinner("음성 분석 중..."):
                            is_pd, prob = voice_predictor.predict(save_path)

                        if prob is None:
                            st.warning(
                                "음성 분석에 실패했습니다.  \n"
                                "녹음 시간이 너무 짧거나 음성이 인식되지 않았습니다.  \n"
                                "다시 녹음해 주세요."
                            )
                            st.session_state.voice_result = None
                        else:
                            st.session_state.voice_result = {
                                "is_pd": is_pd,
                                "prob" : prob,
                                "fname": save_path.name,
                            }
                    else:
                        st.error("음성 파일 저장에 실패했습니다. 다시 시도해주세요.")

        # ── 음성 예측 알림 (즉시 상세 결과 숨김) ─────────────────
        # 녹음 직후에는 상세한 확률/SHAP를 바로 표시하지 않습니다.
        # 종합 예측을 실행할 때 모든 모달리티 결과와 함께 상세 분석을 확인하세요.
        if st.session_state.voice_saved:
            st.success("음성 저장 완료 ✅ — 종합 예측에서 분석 결과를 확인하세요.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 후각 기능 평가 (B-SIT 12문항)
    # ══════════════════════════════════════════════════════════
    st.subheader("👃 후각 기능 평가")
    st.info(
        "📌 **검사 방법**  \n"
        "아래 12가지 향기 각각에 대해 **평소에 해당 향기를 맡을 수 있는지** 선택해 주세요.  \n"
        "잘 모르겠으면 '못 맡겠다'를 선택하세요."
    )

    olf_predictor = load_olfactory_predictor()
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
        olf_col1, olf_col2 = st.columns(2)
        for i, (key, label) in enumerate(BSIT_ITEMS):
            col = olf_col1 if i % 2 == 0 else olf_col2
            with col:
                ans = st.radio(
                    label,
                    options=["잘 맡을 수 있다", "못 맡겠다"],
                    index=0,
                    horizontal=True,
                    key=f"bsit_{key}",
                )
                olf_scores.append(1 if ans == "잘 맡을 수 있다" else 0)

        bsit_total = sum(olf_scores)
        st.caption(f"현재 점수: **{bsit_total} / 12**  (맞힌 항목 수)")

        # 응답만 저장 — 결과는 '파킨슨 위험도 종합 예측'에서 한 번에 산출
        st.session_state.olf_scores = olf_scores
        st.session_state.olf_total  = bsit_total

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 변비 증상 평가 (SCOPA-AUT 3문항)
    # ══════════════════════════════════════════════════════════
    st.subheader("🚽 변비 증상 평가")
    st.info(
        "📌 **지난 한 달** 동안 경험한 배변 관련 증상을 선택해 주세요."
    )

    con_predictor = load_constipation_predictor()
    if con_predictor is None:
        st.warning(
            "변비 모델이 준비되지 않았습니다.  \n"
            "`PPMI/train_constipation_model.py` 를 먼저 실행하세요."
        )
    else:
        SCOPA_OPTIONS = ["전혀 없다 (0)", "가끔 있다 (1)", "자주 있다 (2)", "항상 그렇다 (3)"]

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            aut5_ans = st.radio(
                "🟡 변비로 문제가 있었나요?",
                options=SCOPA_OPTIONS,
                index=0,
                key="scopa_aut5",
            )
        with sc2:
            aut6_ans = st.radio(
                "🟠 배변 시 심하게 힘을 주어야 했나요?",
                options=SCOPA_OPTIONS,
                index=0,
                key="scopa_aut6",
            )
        with sc3:
            aut7_ans = st.radio(
                "🔴 본인 의지와 무관하게 변이 나온 적이 있나요? (변실금)",
                options=SCOPA_OPTIONS,
                index=0,
                key="scopa_aut7",
            )

        aut5 = SCOPA_OPTIONS.index(aut5_ans)
        aut6 = SCOPA_OPTIONS.index(aut6_ans)
        aut7 = SCOPA_OPTIONS.index(aut7_ans)
        scopa_total = aut5 + aut6 + aut7
        st.caption(f"현재 변비 총점: **{scopa_total} / 9**")

        # 응답만 저장 — 결과는 '파킨슨 위험도 종합 예측'에서 한 번에 산출
        st.session_state.scopa_vals = (aut5, aut6, aut7, scopa_total)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # 종합 예측 섹션 (필적 + 음성 + 후각 + 변비 한 번에 산출)
    # ══════════════════════════════════════════════════════════
    st.subheader("🔍 파킨슨 위험도 종합 예측")

    spiral_ok = st.session_state.spiral_saved
    voice_ok  = st.session_state.voice_result is not None
    olf_ready = olf_predictor is not None and st.session_state.olf_scores is not None
    con_ready = con_predictor is not None and st.session_state.scopa_vals is not None

    # 준비 상태 카드
    st_c1, st_c2, st_c3, st_c4 = st.columns(4)
    st_c1.metric(
        label="나선(Spiral)", value="저장 완료 ✅" if spiral_ok else "미저장",
        delta="준비됨" if spiral_ok else "그림을 저장해 주세요",
        delta_color="normal" if spiral_ok else "inverse",
    )
    st_c2.metric(
        label="음성(Voice)", value="녹음 완료 ✅" if voice_ok else "미녹음",
        delta="준비됨" if voice_ok else "음성을 녹음해 주세요",
        delta_color="normal" if voice_ok else "off",
    )
    st_c3.metric(
        label="후각(Olfactory)", value="응답 완료 ✅" if olf_ready else "미완료",
        delta="준비됨" if olf_ready else "문항을 작성해 주세요",
        delta_color="normal" if olf_ready else "off",
    )
    st_c4.metric(
        label="변비(Constipation)", value="응답 완료 ✅" if con_ready else "미완료",
        delta="준비됨" if con_ready else "문항을 작성해 주세요",
        delta_color="normal" if con_ready else "off",
    )

    if not spiral_ok:
        st.warning("나선(Spiral) 그림을 먼저 저장해주세요.")

    st.caption("버튼을 누르면 필적·음성·후각·변비 결과를 한 번에 산출합니다.")
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
    # 결과 표시 직전에 최신 상태로 재평가 (예측 버튼 처리 직후 반영)
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
        st.subheader("📊 예측 결과")

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

            st.markdown("---")
            st.markdown(f"### 🏆 최종 종합 판정 ({modality_str})")

            fv1, fv2 = st.columns([1, 2])
            with fv1:
                st.progress(float(combined_prob))
                st.metric("종합 파킨슨 확률", f"{combined_prob:.1%}")
            with fv2:
                combined_label = "파킨슨 의심" if combined_pos else "정상"
                st.markdown(
                    f'<div style="background:{"#E53E3E" if combined_pos else "#38A169"};'
                    f'color:white;padding:22px 16px;border-radius:16px;text-align:center">'
                    f'<div style="font-size:2em;font-weight:bold">{combined_label}</div>'
                    f'<div style="font-size:1.05em;opacity:0.9;margin-top:6px">'
                    f'{" + ".join(detail_parts)}</div>'
                    f'<div style="font-size:0.9em;opacity:0.8;margin-top:4px">'
                    f'위험도: {risk_level(combined_prob)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 모달리티 기여도 분해 waterfall (우선순위 1)
            modality_prob_map = {"✍️ 필적": spiral_prob}
            label_map = {"음성": "🎤 음성", "후각": "👃 후각", "변비": "🚽 변비"}
            for lbl, p in zip(extra_labels, extra_probs):
                modality_prob_map[label_map.get(lbl, lbl)] = p
            st.markdown("")
            st.plotly_chart(
                waterfall_chart(modality_prob_map),
                use_container_width=True,
            )
            st.caption(
                "기준선보다 높은 모달리티(빨강)가 종합 위험도를 끌어올린 주요 신호입니다."
            )

        # ── 상세 수치 테이블 ──────────────────────────────────
        st.markdown("")
        with st.expander("상세 수치 보기"):
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

        # ── Grad-CAM 히트맵 (나선) ────────────────────────
        if st.session_state.spiral_gradcam is not None:
            with st.expander("🔍 나선(Spiral) Grad-CAM"):
                st.image(
                    st.session_state.spiral_gradcam,
                    caption="빨간 영역: 파킨슨 위험 판정에 기여한 부분",
                    use_container_width=True,
                )

        # 음성 SHAP
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

        # 후각 SHAP 설명
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

        # 변비 SHAP 설명
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

        # ── 면책 조항 ─────────────────────────────────────────
        st.info(
            "⚠️ 이 결과는 AI 스크리닝 보조 도구입니다.  \n"
            "임상 진단은 반드시 **전문 의료진**에게 받으세요."
        )


if __name__ == "__main__":
    main()
