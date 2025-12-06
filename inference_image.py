import os
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch
from PIL import Image
from torchvision import transforms

# Detector 클래스 import
from inference import AutoEncoderDetector, PaDiMDetector, PatchCoreDetector


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


# -------------------------------
# 1. 이미지 하나 → 모델별 anomaly score 시각화
# -------------------------------
@torch.no_grad()
def visualize_single_image(img_path, ae, padim, patchcore):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    # AE score
    ae_score = ae.score_batch(x)[0]
    ae_pred = "ANOMALY" if ae_score > ae.threshold else "NORMAL"

    # PaDiM score
    padim_score = padim.score_batch(x)[0]
    padim_pred = "ANOMALY" if padim_score > padim.threshold else "NORMAL"

    # PatchCore score
    patch_score = patchcore.score_batch(x)[0]
    patch_pred = "ANOMALY" if patch_score > patchcore.threshold else "NORMAL"

    # ---- 시각화 ----
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.axis("off")

    text = (
        f"AE: {ae_pred} (score={ae_score:.4f})\n"
        f"PaDiM: {padim_pred} (score={padim_score:.4f})\n"
        f"PatchCore: {patch_pred} (score={patch_score:.4f})"
    )
    plt.title(text, fontsize=12)
    plt.show()


# -------------------------------
# 2. 여러 이미지(랜덤 10장) → grid 시각화
# -------------------------------
@torch.no_grad()
def visualize_multiple_images(folder, ae, padim, patchcore, count=10):
    folder = Path(folder)
    img_list = list(folder.rglob("*.jpg"))
    img_list = random.sample(img_list, min(count, len(img_list)))

    cols = 5
    rows = (len(img_list) + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 4))

    for i, img_path in enumerate(img_list):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        # AE
        ae_score = ae.score_batch(x)[0]
        ae_pred = "Anomaly" if ae_score > ae.threshold else "Normal"

        # PaDiM
        padim_score = padim.score_batch(x)[0]
        padim_pred = "Anomaly" if padim_score > padim.threshold else "Normal"
        # PatchCore
        patch_score = patchcore.score_batch(x)[0]
        patch_pred = "Anomaly" if patch_score > patchcore.threshold else "Normal"

        # ---------------- 시각화 ----------------
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"{img_path.name}\n"
            f"AE:{ae_pred}  P:{padim_pred}  PC:{patch_pred}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


# -------------------------------
# 실행 예시
# -------------------------------
if __name__ == "__main__":
    print("Loading detectors...")
    ae = AutoEncoderDetector()
    padim = PaDiMDetector()
    patchcore = PatchCoreDetector()
    image_list = glob(os.path.join("test_images", "*.jpg"))
    # 단일 이미지 테스트
    # for test_img in image_list:
    #     visualize_single_image(test_img, ae, padim, patchcore)

    visualize_multiple_images("test_samples", ae, padim, patchcore, count=15)
