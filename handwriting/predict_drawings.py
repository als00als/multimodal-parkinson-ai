"""
파킨슨 필적 예측 스크립트
============================
user_drawings 하위 폴더별로 담당 모델이 예측합니다.

  user_drawings/Spiral/ 폴더 -> Spiral 모델  (임계값 0.54, Recall 93.3%)
  user_drawings/Wave/   폴더 -> Wave   모델  (임계값 0.60, Recall 100%)

실행 방법:
  python predict_drawings.py               # Spiral + Wave 모두 예측 + 종합 결과
  python predict_drawings.py --model Spiral  # Spiral 폴더만 예측
  python predict_drawings.py --model Wave    # Wave 폴더만 예측
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# Windows 콘솔 한글 출력을 위한 UTF-8 설정
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ══════════════════════════════════════════════════════════════
# 경로 & 설정
# ══════════════════════════════════════════════════════════════
BASE_DIR = Path(r"C:\Project_AI\파킨슨 전구기 예측 서비스")
MODEL_DIR = BASE_DIR / "필적" / "results_expE"

SPIRAL_IMAGE_DIR = BASE_DIR / "streamlit" / "user_drawings" / "Spiral"
WAVE_IMAGE_DIR   = BASE_DIR / "streamlit" / "user_drawings" / "Wave"

# 모델별 설정: 전용 이미지 폴더, 모델 경로, 최적 임계값
MODEL_CONFIGS: Dict[str, dict] = {
    "Spiral": {
        "path"      : MODEL_DIR / "spiral" / "best_model.pth",
        "threshold" : 0.54,
        "image_dir" : SPIRAL_IMAGE_DIR,
        "desc"      : "나선 그림 전용  (Recall 93.3%)",
    },
    "Wave": {
        "path"      : MODEL_DIR / "wave" / "best_model.pth",
        "threshold" : 0.60,
        "image_dir" : WAVE_IMAGE_DIR,
        "desc"      : "파도 그림 전용  (Recall 100%)",
    },
}

ENSEMBLE_THRESHOLD = 0.58   # Spiral + Wave 평균에 적용하는 종합 임계값

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 시 EVAL_TF와 동일
EVAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # grayscale -> 3채널 (ResNet 호환)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════
# 모델 빌드 & 로드
# ══════════════════════════════════════════════════════════════
def build_model() -> nn.Module:
    """train_expE.py 의 build_model() 과 동일한 구조"""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )
    return model


def load_model(model_path: Path) -> nn.Module:
    model = build_model()
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════
def get_image_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg"))
    )


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path) -> float:
    """이미지 1장에 대한 파킨슨 확률(0.0 ~ 1.0)을 반환합니다."""
    img = Image.open(image_path).convert("L")
    tensor = EVAL_TF(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    return torch.softmax(logits, dim=1)[0, 1].item()


def classify(prob: float, threshold: float) -> str:
    return "[파킨슨 의심]" if prob >= threshold else "[정상]"


def risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "고위험"
    elif prob >= 0.50:
        return "중위험"
    else:
        return "저위험"


# ══════════════════════════════════════════════════════════════
# 폴더 단위 예측
# ══════════════════════════════════════════════════════════════
def predict_folder(model_name: str) -> Dict[str, float]:
    """
    model_name 에 해당하는 전용 폴더의 이미지를 해당 모델로 예측합니다.
    반환값: { 파일명: 파킨슨 확률 }
    """
    cfg   = MODEL_CONFIGS[model_name]
    thr   = cfg["threshold"]
    imgs  = get_image_files(cfg["image_dir"])

    # 폴더가 비어있는 경우
    if not imgs:
        print(f"\n[{model_name}] 이미지가 없습니다: {cfg['image_dir']}")
        return {}

    # 모델 파일 존재 확인
    if not cfg["path"].exists():
        print(f"\n[오류] {model_name} 모델 파일이 없습니다: {cfg['path']}")
        print("  -> train_expE.py 를 먼저 실행해 모델을 학습시키세요.")
        return {}

    model = load_model(cfg["path"])
    results: Dict[str, float] = {}

    print(f"\n── {model_name} 모델  ({cfg['desc']}, 임계값: {thr}) " +
          "─" * max(0, 38 - len(model_name) - len(cfg["desc"])))
    print(f"  이미지 폴더: {cfg['image_dir']}")
    print(f"  {'파일명':<36} {'파킨슨 확률':>10}   {'판정':>14}")
    print("  " + "─" * 63)

    for img_path in imgs:
        prob  = predict_image(model, img_path)
        label = classify(prob, thr)
        results[img_path.name] = prob
        print(f"  {img_path.name:<36} {prob:>9.1%}    {label}")

    return results


# ══════════════════════════════════════════════════════════════
# 메인 예측 실행
# ══════════════════════════════════════════════════════════════
def run_all(model_filter: Optional[str] = None):
    targets = [model_filter] if model_filter else list(MODEL_CONFIGS.keys())

    # ── 헤더 출력 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  파킨슨 필적 분류 예측 결과")
    print(f"  장치: {DEVICE}")
    for name in targets:
        d = MODEL_CONFIGS[name]["image_dir"]
        n = len(get_image_files(d))
        print(f"  [{name}] {d}  ({n}장)")
    print("=" * 70)

    # ── 모델별 예측 ──────────────────────────────────────────
    #   Spiral 폴더 -> Spiral 모델
    #   Wave   폴더 -> Wave   모델
    all_results: Dict[str, Dict[str, float]] = {}
    for name in targets:
        all_results[name] = predict_folder(name)

    # ── 종합 결과 (Spiral + Wave 둘 다 실행한 경우) ──────────
    spiral_res = all_results.get("Spiral", {})
    wave_res   = all_results.get("Wave",   {})

    if spiral_res and wave_res:
        spiral_avg = sum(spiral_res.values()) / len(spiral_res)
        wave_avg   = sum(wave_res.values())   / len(wave_res)
        total_avg  = (sum(spiral_res.values()) + sum(wave_res.values())) / \
                     (len(spiral_res) + len(wave_res))

        print(f"\n── 종합 결과  (Spiral + Wave 전체 평균, 임계값: {ENSEMBLE_THRESHOLD}) "
              + "─" * 10)
        print(f"  {'항목':<20} {'평균 확률':>10}   {'위험도':>6}   {'판정':>14}")
        print("  " + "─" * 55)
        print(f"  {'Spiral 평균':<20} {spiral_avg:>9.1%}   {risk_level(spiral_avg):>6}   "
              f"{classify(spiral_avg, MODEL_CONFIGS['Spiral']['threshold']):>14}")
        print(f"  {'Wave 평균':<20} {wave_avg:>9.1%}   {risk_level(wave_avg):>6}   "
              f"{classify(wave_avg, MODEL_CONFIGS['Wave']['threshold']):>14}")
        print("  " + "─" * 55)
        print(f"  {'전체 평균 (최종)':<20} {total_avg:>9.1%}   {risk_level(total_avg):>6}   "
              f"{classify(total_avg, ENSEMBLE_THRESHOLD):>14}")

    elif spiral_res and "Wave" not in targets:
        pass   # Wave 모델만 제외한 단독 실행 -> 종합 없음
    elif wave_res and "Spiral" not in targets:
        pass   # Spiral 모델만 제외한 단독 실행 -> 종합 없음

    print("\n" + "=" * 70)
    print("  ※ 이 결과는 AI 스크리닝 보조 도구입니다.")
    print("    임상 진단은 반드시 전문 의료진에게 받으세요.")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="파킨슨 필적 예측 — Spiral/Wave 폴더별 전담 모델로 분류합니다."
    )
    parser.add_argument(
        "--model",
        choices=["Spiral", "Wave"],
        default=None,
        help=(
            "사용할 모델 (기본값: Spiral + Wave 모두)\n"
            "  Spiral -> user_drawings/Spiral/ 폴더 이미지 예측\n"
            "  Wave   -> user_drawings/Wave/   폴더 이미지 예측"
        ),
    )
    args = parser.parse_args()
    run_all(args.model)


if __name__ == "__main__":
    main()
