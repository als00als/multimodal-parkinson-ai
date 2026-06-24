"""
XAI 유틸리티 모듈
=================
우선순위 1  종합 위험도 기여도 분해  waterfall_chart()
우선순위 2  후각·변비 문항별 기여    explain_olfactory_items(), explain_constipation_items()
우선순위 3  Grad-CAM 히트맵         compute_gradcam()
우선순위 4  SHAP 특징 기여도        explain_voice_shap(), explain_olfactory_shap(), explain_constipation_shap()
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

# ── 경로 ─────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent.parent
_PPMI_CSV  = _BASE_DIR / "PPMI" / "PPMI_BSIT12_SCOPA.csv"
_VOICE_NPZ = _BASE_DIR / "음성" / "voice_features.npz"

# ── 음성: top-8 특징 인덱스 → 임상 이름 매핑 ─────────────────
# top_feature_idx = [45, 13, 59, 14, 22, 25, 78, 83]
# 인덱스 계산 기준 (extract_features 순서)
#   0-12  MFCC mean, 13-25 MFCC std, 26-38 MFCC Δ mean, 39-51 MFCC Δ² mean
#   52-57 F0(mean,std,range,iqr,voiced_ratio,jitter), 58-60 RMS(mean,std,shimmer)
#   61-66 Spectral, 67-70 Spectral Contrast, 71-82 Chroma, 83-84 Mel(mean,std)
_VOICE_NAMES_KO = [
    "조음 가속 변화 (MFCC Δ²₇)",   # 45
    "음색 변동성 1 (MFCC σ₁)",     # 13
    "음량 변동성 (RMS σ)",          # 59
    "음색 변동성 2 (MFCC σ₂)",     # 14
    "음색 변동성 3 (MFCC σ₁₀)",    # 22
    "음색 변동성 4 (MFCC σ₁₃)",    # 25
    "음높이 분포 (Chroma₈)",        # 78
    "스펙트럼 에너지 (Mel μ)",      # 83
]

# ── 후각: B-SIT 12문항 ──────────────────────────────────────
_BSIT_ITEMS_KO = [
    "🍒 체리", "🥒 딜 피클", "🍌 바나나", "🍫 초콜릿",
    "🌿 계피", "⛽ 휘발유", "🍋 레몬", "🧅 양파",
    "🍍 파인애플", "🌹 장미", "🧼 비누", "🔥 연기",
]
_BSIT_COLS = [
    "BSIT_CHERRY", "BSIT_DILL_PICKLE", "BSIT_BANANA", "BSIT_CHOCOLATE",
    "BSIT_CINNAMON", "BSIT_GASOLINE", "BSIT_LEMON", "BSIT_ONION",
    "BSIT_PINEAPPLE", "BSIT_ROSE", "BSIT_SOAP", "BSIT_SMOKE", "BSIT_TOTAL",
]
_BSIT_NAMES_KO = [
    "체리", "딜피클", "바나나", "초콜릿", "계피",
    "휘발유", "레몬", "양파", "파인애플", "장미", "비누", "연기", "총점",
]

# ── 변비: SCOPA-AUT 3문항 ────────────────────────────────────
_SCOPA_COLS      = ["SCOPA_AUT5", "SCOPA_AUT6", "SCOPA_AUT7", "SCOPA_CONSTIPATION_TOTAL"]
_SCOPA_NAMES_KO  = ["배변횟수감소(AUT5)", "배변시힘듦(AUT6)", "변실금(AUT7)", "변비총점"]
_SCOPA_LABELS    = [
    ("배변 횟수 감소", "주 3회 미만"),
    ("배변 시 힘듦",   "힘을 주어야 하는 경우"),
    ("변실금",         "의지와 무관한 배변"),
]
_SCOPA_SEVERITY  = ["없음 (0)", "가끔 (1)", "자주 (2)", "항상 (3)"]

# ── SHAP background 캐시 ─────────────────────────────────────
_olf_bg:   Optional[np.ndarray] = None
_con_bg:   Optional[np.ndarray] = None
_voice_bg: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════
# 우선순위 3: Grad-CAM (ResNet18)
# ══════════════════════════════════════════════════════════════
class _GradCAM:
    """ResNet18 마지막 합성곱 블록에 대한 Grad-CAM 구현."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._save_acts)
        self._bwd = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self._acts = out.detach().clone()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach().clone()

    def __call__(self, tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.eval()
        with torch.enable_grad():
            t = tensor.clone().requires_grad_(True)
            logits = self.model(t)
            prob = float(torch.softmax(logits, dim=1)[0, 1])
            self.model.zero_grad()
            logits[0, 1].backward()

        if self._grads is None or self._acts is None:
            h = int(self._acts.shape[2]) if self._acts is not None else 7
            return prob, np.ones((h, h), dtype=np.float32)

        weights = self._grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._acts).sum(dim=1)[0]              # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        return prob, cam

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def compute_gradcam(
    model: nn.Module,
    pil_image: Image.Image,
    eval_tf,
    device: torch.device,
) -> tuple[float, Image.Image]:
    """
    PIL 이미지 → (파킨슨 확률, Grad-CAM 오버레이 PIL Image)
    모델 재학습 없이 추론 시 hook만으로 동작.
    """
    gray   = pil_image.convert("L")
    tensor = eval_tf(gray).unsqueeze(0).to(device)

    cam_obj  = _GradCAM(model, model.layer4[-1])
    prob, hm = cam_obj(tensor)
    cam_obj.remove()

    overlay = _overlay_heatmap(pil_image.convert("RGB"), hm)
    return prob, overlay


# ══════════════════════════════════════════════════════════════
# 우선순위 3-B: Grad-CAM (DualResNet18 — 나선·파도 이중 브랜치)
# ══════════════════════════════════════════════════════════════
class _DualGradCAM:
    """DualResNet18 의 두 브랜치(layer4) 각각에 대한 Grad-CAM 구현."""

    def __init__(self, model: nn.Module,
                 spiral_layer: nn.Module, wave_layer: nn.Module) -> None:
        self.model = model
        self._s_act = self._s_grad = None
        self._w_act = self._w_grad = None
        self._handles = [
            spiral_layer.register_forward_hook(self._save_s_act),
            spiral_layer.register_full_backward_hook(self._save_s_grad),
            wave_layer.register_forward_hook(self._save_w_act),
            wave_layer.register_full_backward_hook(self._save_w_grad),
        ]

    def _save_s_act(self, m, i, o):   self._s_act  = o.detach().clone()
    def _save_s_grad(self, m, gi, go): self._s_grad = go[0].detach().clone()
    def _save_w_act(self, m, i, o):   self._w_act  = o.detach().clone()
    def _save_w_grad(self, m, gi, go): self._w_grad = go[0].detach().clone()

    @staticmethod
    def _cam(acts: Optional[torch.Tensor],
             grads: Optional[torch.Tensor]) -> np.ndarray:
        if acts is None or grads is None:
            return np.ones((7, 7), dtype=np.float32)
        weights = grads.mean(dim=[2, 3], keepdim=True)      # (1, C, 1, 1)
        cam = torch.relu((weights * acts).sum(dim=1)[0])    # (H, W)
        cam = cam.cpu().numpy()
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        return cam

    def __call__(self, s_tensor: torch.Tensor, w_tensor: torch.Tensor):
        self.model.eval()
        with torch.enable_grad():
            s = s_tensor.clone().requires_grad_(True)
            w = w_tensor.clone().requires_grad_(True)
            logits = self.model(s, w)
            prob = float(torch.softmax(logits, dim=1)[0, 1])
            self.model.zero_grad()
            logits[0, 1].backward()
        return prob, self._cam(self._s_act, self._s_grad), \
                     self._cam(self._w_act, self._w_grad)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def compute_dual_gradcam(
    model: nn.Module,
    spiral_pil: Image.Image,
    wave_pil: Image.Image,
    eval_tf,
    device: torch.device,
) -> tuple[float, Image.Image, Image.Image]:
    """
    나선·파도 PIL 이미지 → (파킨슨 확률,
                            나선 Grad-CAM 오버레이, 파도 Grad-CAM 오버레이)
    DualResNet18 의 spiral_branch / wave_branch 마지막 합성곱 블록에서 추출.
    """
    s_t = eval_tf(spiral_pil.convert("L")).unsqueeze(0).to(device)
    w_t = eval_tf(wave_pil.convert("L")).unsqueeze(0).to(device)

    cam_obj = _DualGradCAM(
        model,
        model.spiral_branch.layer4[-1],
        model.wave_branch.layer4[-1],
    )
    prob, s_cam, w_cam = cam_obj(s_t, w_t)
    cam_obj.remove()

    s_overlay = _overlay_heatmap(spiral_pil.convert("RGB"), s_cam)
    w_overlay = _overlay_heatmap(wave_pil.convert("RGB"), w_cam)
    return prob, s_overlay, w_overlay


def _overlay_heatmap(
    base: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    import matplotlib.cm as cm

    w, h = base.size
    hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L").resize(
        (w, h), Image.LANCZOS
    )
    hm_rgb = (cm.get_cmap("jet")(np.array(hm_pil) / 255.0)[:, :, :3] * 255).astype(
        np.uint8
    )
    return Image.blend(base, Image.fromarray(hm_rgb, mode="RGB"), alpha=alpha)


# ══════════════════════════════════════════════════════════════
# 우선순위 1: 종합 위험도 기여도 분해 (waterfall)
# ══════════════════════════════════════════════════════════════
def waterfall_chart(modality_probs: dict[str, float]):
    """
    각 모달리티가 종합 위험도를 기준 대비 얼마나 끌어올렸는지 시각화.
    반환: plotly Figure
    """
    import plotly.graph_objects as go

    labels = list(modality_probs.keys())
    probs  = list(modality_probs.values())
    mean_p = sum(probs) / len(probs)

    devs   = [p - mean_p for p in probs]
    pairs  = sorted(zip(devs, labels), key=lambda x: x[0])
    devs_s, labels_s = zip(*pairs)

    colors = ["#E53E3E" if d > 0 else "#38A169" for d in devs_s]
    texts  = [f"{d:+.1%}" for d in devs_s]

    fig = go.Figure(
        go.Bar(
            x=list(devs_s),
            y=list(labels_s),
            orientation="h",
            marker_color=colors,
            text=texts,
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#888", line_width=1.5, line_dash="dot")
    fig.update_layout(
        title=dict(
            text=f"모달리티별 기여도 분해  (기준선 {mean_p:.1%})",
            font=dict(size=13),
        ),
        xaxis_title="기준선 대비 기여 (±%p)",
        xaxis_tickformat=".0%",
        height=140 + 44 * len(labels),
        margin=dict(l=140, r=90, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 우선순위 2: 후각·변비 문항별 기여 (규칙 기반, 추가 학습 불필요)
# ══════════════════════════════════════════════════════════════
def explain_olfactory_items(scores: list[int]) -> dict:
    """
    B-SIT 12문항 응답에서 감점 항목을 추출한다.
    반환: {"failed": list[str], "passed": list[str], "n_failed": int, "n_total": int}
    """
    failed = [_BSIT_ITEMS_KO[i] for i, s in enumerate(scores) if s == 0]
    passed = [_BSIT_ITEMS_KO[i] for i, s in enumerate(scores) if s == 1]
    return {"failed": failed, "passed": passed, "n_failed": len(failed), "n_total": 12}


def explain_constipation_items(aut5: int, aut6: int, aut7: int) -> list[dict]:
    """
    SCOPA-AUT 3문항 각각의 점수와 위험 플래그를 반환한다.
    반환: list of {"name", "desc", "score", "severity", "flagged"}
    """
    return [
        {
            "name"    : name,
            "desc"    : desc,
            "score"   : score,
            "severity": _SCOPA_SEVERITY[score],
            "flagged" : score >= 2,
        }
        for (name, desc), score in zip(_SCOPA_LABELS, [aut5, aut6, aut7])
    ]


# ══════════════════════════════════════════════════════════════
# 우선순위 4: SHAP 기반 특징 기여도
# ══════════════════════════════════════════════════════════════
def _olf_background() -> np.ndarray:
    global _olf_bg
    if _olf_bg is None:
        import pandas as pd
        df  = pd.read_csv(_PPMI_CSV)
        arr = df[_BSIT_COLS].dropna().values.astype(np.float32)
        idx = np.random.default_rng(42).choice(len(arr), size=min(100, len(arr)), replace=False)
        _olf_bg = arr[idx]
    return _olf_bg


def _con_background() -> np.ndarray:
    global _con_bg
    if _con_bg is None:
        import pandas as pd
        df  = pd.read_csv(_PPMI_CSV)
        arr = df[_SCOPA_COLS].dropna().values.astype(np.float32)
        idx = np.random.default_rng(42).choice(len(arr), size=min(100, len(arr)), replace=False)
        _con_bg = arr[idx]
    return _con_bg


def _voice_background(top_idx: np.ndarray) -> np.ndarray:
    global _voice_bg
    if _voice_bg is None:
        data = np.load(_VOICE_NPZ)
        _voice_bg = data["X"][:, top_idx].astype(np.float32)
    return _voice_bg


def _shap_bar(
    shap_vals: np.ndarray,
    feature_names: list[str],
    base_value: float,
    title: str,
):
    """SHAP 값을 plotly 수평 막대 차트로 반환 (크기 순 정렬)."""
    import plotly.graph_objects as go

    pairs   = sorted(zip(shap_vals, feature_names), key=lambda x: abs(x[0]))
    sv_s, fn_s = zip(*pairs) if pairs else ([], [])

    colors = ["#E53E3E" if v > 0 else "#38A169" for v in sv_s]
    fig = go.Figure(
        go.Bar(
            x=list(sv_s),
            y=list(fn_s),
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in sv_s],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#888", line_width=1.5, line_dash="dot")
    fig.update_layout(
        title=dict(text=f"{title}  (기준값 {base_value:.3f})", font=dict(size=13)),
        xaxis_title="SHAP 기여도  (양수: 파킨슨 위험↑ · 음수: 위험↓)",
        height=160 + 32 * len(feature_names),
        margin=dict(l=200, r=90, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    return fig


def _run_shap(model, background: np.ndarray, input_vec: np.ndarray):
    """KernelExplainer 실행 → (shap_vals for PD class, base_value) 반환."""
    import shap
    explainer = shap.KernelExplainer(model.predict_proba, background)
    sv = explainer.shap_values(input_vec.reshape(1, -1), nsamples=200, silent=True)
    ev = explainer.expected_value

    # sv: list[ndarray] (per-class) 또는 ndarray (단일 출력)
    if isinstance(sv, list):
        vals = np.array(sv[1]).flatten()
        base = float(np.array(ev).flatten()[1] if np.array(ev).ndim > 0 and len(np.array(ev)) > 1 else np.array(ev).flat[0])
    else:
        vals = np.array(sv).flatten()
        ev_arr = np.array(ev).flatten()
        base = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])

    return vals, base


def explain_voice_shap(model, top_idx: np.ndarray, full_feat_vec: np.ndarray):
    """85차원 음성 특징 벡터에서 top-8 선택 후 SHAP 차트 반환."""
    bg  = _voice_background(top_idx)
    vec = full_feat_vec[top_idx]
    sv, base = _run_shap(model, bg, vec)
    return _shap_bar(sv, _VOICE_NAMES_KO, base, "음성 특징 기여도 (SHAP)")


def explain_olfactory_shap(model, scores: list[int]):
    """B-SIT 12문항 + 총점 → SHAP 차트 반환."""
    total = sum(scores)
    vec   = np.array(scores + [total], dtype=np.float32)
    sv, base = _run_shap(model, _olf_background(), vec)
    return _shap_bar(sv, _BSIT_NAMES_KO, base, "후각 문항 기여도 (SHAP)")


def explain_constipation_shap(model, aut5: int, aut6: int, aut7: int):
    """SCOPA-AUT 3문항 + 총점 → SHAP 차트 반환."""
    total = aut5 + aut6 + aut7
    vec   = np.array([aut5, aut6, aut7, total], dtype=np.float32)
    sv, base = _run_shap(model, _con_background(), vec)
    return _shap_bar(sv, _SCOPA_NAMES_KO, base, "변비 문항 기여도 (SHAP)")
