import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

# inference.py 에 세 클래스가 정의되어 있다고 가정
from inference import AutoEncoderDetector, PaDiMDetector, PatchCoreDetector


# ----------------- 설정 -----------------
NORMAL_DIR = "normal"
ANOMALY_DIR = "sampling"
TEST_IMAGES_DIR = "test_samples"

IMG_SIZE = (512, 512)       # inference 때 사용하던 사이즈 맞춰주기
BATCH_SIZE = 4             # HW 여유에 따라 조절
NUM_WORKERS = 0
PIN_MEMORY = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("eval_results")
OUT_DIR.mkdir(exist_ok=True)


# ----------------- 데이터셋 정의 -----------------
class LabeledImageFolder(Dataset):
    """
    root_dir 아래의 모든 .jpg를 읽어서 label을 부여하는 Dataset
    label: 0(normal) 또는 1(anomaly)
    """
    def __init__(self, root_dir, label, transform=None):
        self.root = Path(root_dir)
        self.label = label
        self.transform = transform

        self.files = [p for p in self.root.rglob("*.jpg") if p.is_file()]
        if not self.files:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label, str(img_path)


class UnlabeledImageFolder(Dataset):
    """test_images/ 용 – label 없이 경로만 반환"""
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.files = [p for p in self.root.rglob("*.jpg") if p.is_file()]
        if not self.files:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(img_path)


def build_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    normal_ds = LabeledImageFolder(NORMAL_DIR, label=0, transform=transform)
    anom_ds = LabeledImageFolder(ANOMALY_DIR, label=1, transform=transform)

    full_ds = torch.utils.data.ConcatDataset([normal_ds, anom_ds])
    full_loader = DataLoader(
        full_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = None
    if os.path.isdir(TEST_IMAGES_DIR):
        test_ds = UnlabeledImageFolder(TEST_IMAGES_DIR, transform=transform)
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    return full_loader, test_loader


# ----------------- 공통 유틸 함수 -----------------
def find_best_threshold(y_true, y_scores, num_steps=500):
    """F1이 최대가 되는 threshold 탐색 (모든 모델 공통)"""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    t_min, t_max = y_scores.min(), y_scores.max()
    thresholds = np.linspace(t_min, t_max, num_steps)

    best_thr = thresholds[0]
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in thresholds:
        y_pred = (y_scores > thr).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_prec = prec
            best_rec = rec

    return best_thr, best_f1, best_prec, best_rec


def evaluate_detector(detector, loader, name="Model"):
    """
    detector.score_batch(imgs) 를 이용해 전체 데이터에 대한
    y_true, y_scores, 추론시간을 측정
    """
    detector_name = name
    all_scores = []
    all_labels = []
    all_paths = []

    total_time = 0.0
    total_imgs = 0

    print(f"\n==> Evaluating {detector_name} ...")

    with torch.no_grad():
        for imgs, labels, paths in loader:
            start = time.perf_counter()
            scores = detector.score_batch(imgs)   # np.ndarray 또는 list 반환 가정
            end = time.perf_counter()

            infer_time = end - start
            total_time += infer_time
            total_imgs += len(imgs)

            all_scores.append(np.asarray(scores))
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

    y_scores = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    # 성능 지표
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    best_thr, best_f1, best_prec, best_rec = find_best_threshold(y_true, y_scores)

    # Confusion matrix at best threshold
    y_pred = (y_scores > best_thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    avg_time = total_time / max(1, total_imgs)

    print(f"[{detector_name}] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    print(f"[{detector_name}] Best F1={best_f1:.4f}, thr={best_thr:.6f}")
    print(f"[{detector_name}] Precision={best_prec:.4f}, Recall={best_rec:.4f}")
    print(f"[{detector_name}] TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"[{detector_name}] Avg inference time per image: {avg_time*1000:.3f} ms")

    results = {
        "model": detector_name,
        "AUROC": auroc,
        "AUPRC": auprc,
        "BestF1": best_f1,
        "Precision": best_prec,
        "Recall": best_rec,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "AvgTimePerImage": avg_time,
        "BestThreshold": best_thr,
    }

    return results, (y_true, y_scores, y_pred, all_paths)


def run_on_test_images(detector, loader, name="Model"):
    """레이블 없는 test_images 20장에 대해 score + 예측을 CSV로 저장 (옵션)"""
    if loader is None:
        return

    all_paths = []
    all_scores = []

    with torch.no_grad():
        for imgs, paths in loader:
            scores = detector.score_batch(imgs)
            all_scores.extend(list(scores))
            all_paths.extend(list(paths))

    df = pd.DataFrame({
        "path": all_paths,
        "score": all_scores,
    })
    out_csv = OUT_DIR / f"test_images_scores_{name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{name}] Saved test_images scores to {out_csv}")


# ----------------- 메인 -----------------
def main():
    print("DEVICE:", DEVICE)

    full_loader, test_loader = build_dataloaders()

    # 1. Detector 로드 (inference.py 안에서 내부적으로 모델/threshold를 로드한다고 가정)
    ae = AutoEncoderDetector()
    padim = PaDiMDetector()
    patchcore = PatchCoreDetector()

    # 2. 전체 normal + anomaly 데이터로 성능 평가
    results = []

    r_ae, _ = evaluate_detector(ae, full_loader, name="AutoEncoder")
    results.append(r_ae)

    r_padim, _ = evaluate_detector(padim, full_loader, name="PaDiM")
    results.append(r_padim)

    r_pc, _ = evaluate_detector(patchcore, full_loader, name="PatchCore")
    results.append(r_pc)

    # 3. 결과를 테이블(표)로 저장
    df = pd.DataFrame(results)
    table_path = OUT_DIR / "model_performance_summary.csv"
    df.to_csv(table_path, index=False)
    print(f"\nSaved performance table to {table_path}")
    print(df)

    # 4. 메트릭 시각화 (바 차트)
    # 4-1. AUROC / AUPRC / BestF1
    plt.figure(figsize=(8, 5))
    x = np.arange(len(results))
    width = 0.25

    aurocs = [r["AUROC"] for r in results]
    auprcs = [r["AUPRC"] for r in results]
    f1s = [r["BestF1"] for r in results]
    labels = [r["model"] for r in results]

    plt.bar(x - width, aurocs, width, label="AUROC")
    plt.bar(x, auprcs, width, label="AUPRC")
    plt.bar(x + width, f1s, width, label="Best F1")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    perf_fig_path = OUT_DIR / "performance_comparison.png"
    plt.tight_layout()
    plt.savefig(perf_fig_path, dpi=150)
    plt.close()
    print(f"Saved performance plot to {perf_fig_path}")

    # 4-2. Precision / Recall
    plt.figure(figsize=(8, 5))
    precs = [r["Precision"] for r in results]
    recs = [r["Recall"] for r in results]

    plt.bar(x - width/2, precs, width, label="Precision")
    plt.bar(x + width/2, recs, width, label="Recall")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Precision / Recall at Best F1")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    pr_fig_path = OUT_DIR / "precision_recall_comparison.png"
    plt.tight_layout()
    plt.savefig(pr_fig_path, dpi=150)
    plt.close()
    print(f"Saved precision/recall plot to {pr_fig_path}")

    # 5. 추론 시간 시각화 (초/이미지)
    plt.figure(figsize=(6, 4))
    times = [r["AvgTimePerImage"] for r in results]  # seconds per image

    plt.bar(labels, times)
    plt.ylabel("Seconds per image")
    plt.title("Average Inference Time per Image")
    plt.grid(axis="y", alpha=0.3)

    time_fig_path = OUT_DIR / "inference_time_comparison.png"
    plt.tight_layout()
    plt.savefig(time_fig_path, dpi=150)
    plt.close()
    print(f"Saved inference time plot to {time_fig_path}")

    # 6. 옵션: test_images/ 20장에 대한 score CSV 저장
    if test_loader is not None:
        run_on_test_images(ae, test_loader, name="AutoEncoder")
        run_on_test_images(padim, test_loader, name="PaDiM")
        run_on_test_images(patchcore, test_loader, name="PatchCore")


if __name__ == "__main__":
    main()
