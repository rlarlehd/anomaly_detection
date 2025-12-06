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
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
PADIM_STAT_PATH = "checkpoints/padim_stats_256.pt"
Path("checkpoints").mkdir(exist_ok=True)
NORMAL_DIR = "normal" 
ANOMALY_DIR = "sampling"
MAX_FEATS = 5000
IMG_SIZE = (512, 512)
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


class SimpleImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.files = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() == '.jpg':
                self.files.append(p)
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


def get_dataloaders():
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # Normalize
    ])

    normal_dataset = SimpleImageFolder(NORMAL_DIR, transform=transform)
    n_total = len(normal_dataset)
    n_train = int(n_total * 0.8)
    n_test_normal = n_total - n_train

    train_normal, test_normal = random_split(
        normal_dataset,
        [n_train, n_test_normal],
        generator=torch.Generator().manual_seed(SEED),
    )

    anomaly_dataset = SimpleImageFolder(ANOMALY_DIR, transform=transform)

    train_loader = DataLoader(train_normal, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_normal_loader = DataLoader(test_normal, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4, pin_memory=True)
    test_anom_loader = DataLoader(anomaly_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_normal_loader, test_anom_loader


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.layers = {}
        
        def hook_fn(name):
            def fn(_, __, output):
                self.layers[name] = output
            return fn

        self.backbone.layer1.register_forward_hook(hook_fn("layer1"))
        self.backbone.layer2.register_forward_hook(hook_fn("layer2"))
        self.backbone.layer3.register_forward_hook(hook_fn("layer3"))

    def forward(self, x):
        _ = self.backbone(x)  # forward만 해서 hooks 채우기

        f1 = self.layers["layer1"]  # [B, 64, H/4,  W/4 ]
        f2 = self.layers["layer2"]  # [B,128, H/8,  W/8 ]
        f3 = self.layers["layer3"]  # [B,256, H/16, W/16]

        # 가장 큰 해상도(f1)에 맞게 upsample 후 concat
        B, C1, H1, W1 = f1.shape
        f2_up = F.interpolate(f2, size=(H1, W1), mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=(H1, W1), mode="bilinear", align_corners=False)

        feat = torch.cat([f1, f2_up, f3_up], dim=1)  # [B, C, H, W]
        return feat


@torch.no_grad()
def collect_features(dataloader, extractor, max_feats=MAX_FEATS):
    extractor.eval()
    all_feats = None

    for imgs, _ in tqdm(dataloader, desc="Collecting features"):
        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)  # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu()  # [B*H*W, C] on CPU

        if all_feats is None:
            all_feats = feats
        else:
            # 일단 붙이고
            all_feats = torch.cat([all_feats, feats], dim=0)

        # 너무 커지면 바로 줄여버리기
        if all_feats.size(0) > max_feats:
            idx = torch.randperm(all_feats.size(0))[:max_feats]
            all_feats = all_feats[idx]

    print("Using features:", all_feats.shape)
    return all_feats


def fit_gaussian(features: torch.Tensor):
    # features: [N, C]
    mean = features.mean(dim=0)               # [C]
    # torch.cov는 (C,C) 반환 (input: [C, N])
    cov = torch.cov(features.T)               # [C, C]

    # 수치 안정성을 위해 diagonal에 eps 추가
    eps = 1e-6
    cov = cov + eps * torch.eye(cov.size(0))

    cov_inv = torch.linalg.inv(cov)
    return mean, cov_inv


@torch.no_grad()
def mahalanobis_scores(dataloader, extractor, mean, cov_inv):
    extractor.eval()
    mean = mean.to(DEVICE)
    cov_inv = cov_inv.to(DEVICE)

    img_scores = []

    for imgs, _ in tqdm(dataloader, desc="Mahalanobis scores"):
        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)           # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        diff = feats - mean  # [N, C]
        # (x-μ)^T Σ^-1 (x-μ)
        # -> (diff @ cov_inv) * diff, then sum over C
        m = diff @ cov_inv   # [N, C]
        m = (m * diff).sum(dim=1)  # [N]

        # 각 이미지별로 다시 reshape해서 max/mean 등으로 요약
        m = m.reshape(B, H, W)   # [B, H, W]
        # 여기서는 max를 이미지 anomaly score로 사용
        score_per_img = m.view(B, -1).max(dim=1)[0]  # [B]
        img_scores.extend(score_per_img.cpu().numpy().tolist())

    return np.array(img_scores)


def main():
    try:
        print("Device:", DEVICE)
        train_loader, test_normal_loader, test_anom_loader = get_dataloaders()

        print("==> Building feature extractor (ResNet18)...")
        extractor = ResNetFeatureExtractor().to(DEVICE)

        # 1) 먼저 기존 stats 파일이 있는지 확인
        if os.path.exists(PADIM_STAT_PATH):
            print(f"Loading PaDiM stats from {PADIM_STAT_PATH}")
            stats = torch.load(PADIM_STAT_PATH, map_location="cpu")
            mean = stats["mean"]
            cov_inv = stats["cov_inv"]
        else:
            # 2) 없으면 그때만 feature 수집 + gaussian fitting
            print("==> Collecting normal features for Gaussian fitting...")
            normal_features = collect_features(train_loader, extractor)
            print("Collected features:", normal_features.shape)

            print("==> Fitting Gaussian (mean, covariance^-1)...")
            mean, cov_inv = fit_gaussian(normal_features)
            print("Feature dim:", mean.shape[0])

            torch.save({"mean": mean.cpu(), "cov_inv": cov_inv.cpu()}, PADIM_STAT_PATH)
            print(f"Saved PaDiM stats to {PADIM_STAT_PATH}")

        print("==> Computing scores for normal test images...")
        scores_normal = mahalanobis_scores(test_normal_loader, extractor, mean, cov_inv)

        print("==> Computing scores for anomaly test images...")
        scores_anom = mahalanobis_scores(test_anom_loader, extractor, mean, cov_inv)

        # 이하 동일
        y_true = np.concatenate([
            np.zeros_like(scores_normal),
            np.ones_like(scores_anom),
        ])
        y_scores = np.concatenate([scores_normal, scores_anom])

        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        print("\n==> PaDiM minimal results")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        mu = scores_normal.mean()
        sigma = scores_normal.std()
        thr = mu + 3 * sigma
        print(f"\nNormal mean: {mu:.6f}, std: {sigma:.6f}, thr (mu+3σ): {thr:.6f}")

        prec, rec, thr_list = precision_recall_curve(y_true, y_scores)

        thr_list = np.append(thr_list, thr_list[-1])

        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1)
        best_thr = thr_list[best_idx]

        print(f"Best F1 threshold: {best_thr:.6f}")
        print(f"Best F1: {f1[best_idx]:.4f}")
        print(f"Precision at best F1: {prec[best_idx]:.4f}")
        print(f"Recall at best F1: {rec[best_idx]:.4f}")

        y_pred = (y_scores > best_thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
