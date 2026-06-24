"""
파킨슨 필적(Spiral) 단일 입력 예측기
====================================
사용:
    from spiral_predictor import SpiralPredictor

    predictor = SpiralPredictor()
    is_pd, prob = predictor.predict(spiral_image)  # PIL Image → (bool, float)

[모델 파일]
    필적/results_optimized/spiral/best_model.pth
    필적/results_optimized/spiral/best_threshold.pkl
"""

import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "필적" / "results_optimized" / "spiral"

EVAL_TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SpiralResNet18(nn.Module):
    """Spiral 이미지 단일 입력 ResNet18 분류기."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # layer3, layer4만 학습 (layer1, layer2 동결)
        for name, p in self.backbone.named_parameters():
            if not any(name.startswith(x) for x in ("layer3", "layer4")):
                p.requires_grad = False
        # fc 교체
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SpiralPredictor:
    """Spiral 단일 입력 예측기."""

    def __init__(self):
        model_path = MODEL_DIR / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Spiral 모델 파일이 없습니다: {model_path}\n"
                "필적/train_optimized.py 를 먼저 실행해 모델을 학습시키세요."
            )

        # 임계값 로드
        threshold_path = MODEL_DIR / "best_threshold.pkl"
        if threshold_path.exists():
            try:
                self.threshold = float(joblib.load(threshold_path))
            except Exception:
                self.threshold = 0.2
        else:
            # results.json에서 로드 시도
            results_path = MODEL_DIR.parent / "all_results.json"
            if results_path.exists():
                try:
                    data = json.loads(results_path.read_text(encoding="utf-8"))
                    self.threshold = float(data.get("Spiral", {}).get("threshold", 0.2))
                except Exception:
                    self.threshold = 0.2
            else:
                self.threshold = 0.2

        # 모델 로드
        self.model = SpiralResNet18().to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)

        # 가중치 키를 backbone. 접두사로 변환 (학습 시 저장 포맷 호환)
        new_state = {}
        for k, v in state.items():
            if not k.startswith("backbone."):
                new_state[f"backbone.{k}"] = v
            else:
                new_state[k] = v

        self.model.load_state_dict(new_state)
        self.model.eval()

    def predict(self, spiral_image: Image.Image) -> tuple[bool, float]:
        """
        Spiral 이미지 → 파킨슨 여부 및 확률

        Args:
            spiral_image: PIL Image (RGB 또는 Grayscale)

        Returns:
            (is_parkinson: bool, probability: float)
        """
        with torch.no_grad():
            img_t = EVAL_TF(spiral_image.convert("L")).unsqueeze(0).to(DEVICE)
            logits = self.model(img_t)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        return prob >= self.threshold, prob
