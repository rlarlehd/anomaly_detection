import os
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------- 설정 -----------------
Path("checkpoints").mkdir(exist_ok=True)
MODEL_PATH = "checkpoints/autoencoder_256.pth"
NORMAL_DIR = "normal"
ANOMALY_DIR = "sampling"

# 512x512 -> 256x256로 내려서 메모리 절감 (필요하면 다시 512로 올려도 됨)
IMG_SIZE = (256, 256)

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataloader 설정도 DEVICE에 맞게
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PIN_MEMORY = True if DEVICE == "cuda" else False


class SimpleImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.files = []
        print(f"[SimpleImageFolder] DEVICE = {DEVICE}")
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".jpg":
                self.files.append(p)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(img_path)


class Cnn_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # IMG_SIZE가 (256, 256)일 때 기준: 256 -> 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),# H/16
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # H*2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # H*4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # H*8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # H*16 (원래 해상도)
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # 필요하면 Normalize 추가
        # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

    normal_dataset = SimpleImageFolder(NORMAL_DIR, transform=transform)

    # normal 데이터 안에서 train / test_normal 분리
    n_total = len(normal_dataset)
    n_train = int(n_total * 0.8)
    n_test_normal = n_total - n_train
    train_normal, test_normal = random_split(
        normal_dataset,
        lengths=[n_train, n_test_normal],
        generator=torch.Generator().manual_seed(42),
    )

    anomaly_dataset = SimpleImageFolder(ANOMALY_DIR, transform=transform)

    train_loader = DataLoader(
        train_normal,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_normal_loader = DataLoader(
        test_normal,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, test_normal_loader, test_anomaly_loader


def train_autoencoder(model, train_loader):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # AMP 사용 (메모리 절감 + 속도↑, 특히 cuda일 때)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{epoch:03d}/{EPOCHS}] Train", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                recon = model(imgs)
                loss = F.mse_loss(recon, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"[{epoch:03d}/{EPOCHS}] Train Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved AE model to {MODEL_PATH}")
    return model


@torch.no_grad()
def compute_errors(model, dataloader):
    """
    각 이미지에 대해 reconstruction error(mean MSE per image)를 계산
    """
    model.eval()
    model.to(DEVICE)

    all_scores = []
    all_paths = []

    pbar = tqdm(dataloader, desc="Computing reconstruction errors")
    for imgs, paths in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            recon = model(imgs)
            err = F.mse_loss(recon, imgs, reduction="none")  # [B, C, H, W]

        # 채널/공간 평균 -> 이미지당 스칼라
        err = err.view(err.size(0), -1).mean(dim=1)  # [B]

        all_scores.extend(err.cpu().numpy().tolist())
        all_paths.extend(paths)

    return np.array(all_scores), all_paths


def main():
    print("start, DEVICE =", DEVICE)
    train_loader, test_normal_loader, test_anomaly_loader = get_dataloaders()

    print("모델생성")
    model = Cnn_autoencoder()

    if os.path.exists(MODEL_PATH):
        print(f"Found saved model: {MODEL_PATH}, loading...")
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print("학습시작")
        print("No saved model, training from scratch...")
        model = train_autoencoder(model, train_loader)

    print("정상 데이터에 대한 재구성 오류 계산 중")
    scores_normal, paths_normal = compute_errors(model, test_normal_loader)

    print("이상 데이터에 대한 재구성 오류 계산 중")
    scores_anom, paths_anom = compute_errors(model, test_anomaly_loader)

    # normal=0, anomaly=1
    y_true = np.concatenate([
        np.zeros_like(scores_normal),
        np.ones_like(scores_anom),
    ])
    y_scores = np.concatenate([scores_normal, scores_anom])

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    print("결과")
    print(f"AUROC : {auroc:.4f}")
    print(f"AUPRC : {auprc:.4f}")

    mu = scores_normal.mean()
    sigma = scores_normal.std()
    thr = mu + 3 * sigma
    print(f"\nNormal error mean: {mu:.6f}, std: {sigma:.6f}, example threshold: {thr:.6f}")

    thresholds = np.linspace(y_scores.min(), y_scores.max(), 500)

    best_thr = thresholds[0]
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for thr_cand in thresholds:
        y_pred = (y_scores > thr_cand).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr_cand
            best_prec = prec
            best_rec = rec

    print("\n[Best-F1 threshold search]")
    print(f"Best threshold: {best_thr:.6f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Precision at best F1: {best_prec:.4f}")
    print(f"Recall at best F1: {best_rec:.4f}")

    # 최적 threshold로 최종 confusion matrix
    y_pred = (y_scores > best_thr).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


if __name__ == "__main__":
    main()
