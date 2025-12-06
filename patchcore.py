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

PATCHCORE_MEM_PATH = "checkpoints/patchcore_mem_256.pt"
Path("checkpoints").mkdir(exist_ok=True)
NORMAL_DIR = "normal" 
ANOMALY_DIR = "sampling"

IMG_SIZE = (512, 512)
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
MAX_PATCHES = 10000
N_CHUNK = 512
M_CHUNK = 512
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
    ])

    normal_dataset = SimpleImageFolder(NORMAL_DIR, transform=transform)
    n_total = len(normal_dataset)
    # 일단 train/test 비율 크게 안 써도 됨, 그냥 8:2
    n_train = int(n_total * 0.8)
    n_test_normal = n_total - n_train

    train_normal, test_normal = random_split(
        normal_dataset,
        [n_train, n_test_normal],
        generator=torch.Generator().manual_seed(SEED),
    )

    anomaly_dataset = SimpleImageFolder(ANOMALY_DIR, transform=transform)

    train_loader = DataLoader(train_normal, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=False)
    test_normal_loader = DataLoader(test_normal, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=0, pin_memory=False)
    test_anom_loader = DataLoader(anomaly_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=False)

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
        _ = self.backbone(x)

        f1 = self.layers["layer1"]  # [B, 64, H/4,  W/4 ]
        f2 = self.layers["layer2"]  # [B,128, H/8,  W/8 ]
        f3 = self.layers["layer3"]  # [B,256, H/16, W/16]

        B, C1, H1, W1 = f1.shape
        f2_up = F.interpolate(f2, size=(H1, W1), mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=(H1, W1), mode="bilinear", align_corners=False)

        feat = torch.cat([f1, f2_up, f3_up], dim=1)  # [B, C, H, W]
        return feat

@torch.no_grad()
def build_memory_bank(dataloader, extractor, max_patches=MAX_PATCHES):
    extractor.eval()
    collected = []
    total = 0

    print("==> Collecting normal features for memory bank (online, capped)...")

    for i, (imgs, _) in enumerate(tqdm(dataloader, desc="build_memory_bank")):
        if total >= max_patches:
            break  # 이미 꽉 찼으면 더 안 모음

        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)                 # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        feats = feats.cpu()

        n = feats.size(0)
        remaining = max_patches - total

        if n <= remaining:
            collected.append(feats)
            total += n
        else:
            # 남은 칸만큼만 잘라서 사용
            collected.append(feats[:remaining])
            total += remaining

        if i % 10 == 0:
            print(f"  batch {i}, collected patches: {total}/{max_patches}")

    if not collected:
        raise RuntimeError("No features collected for memory bank.")

    memory_bank = torch.cat(collected, dim=0)   # [<=max_patches, C]
    print(f"Final memory bank size: {memory_bank.shape[0]} patches, dim={memory_bank.shape[1]}")

    return memory_bank.to(DEVICE)



@torch.no_grad()
def compute_patchcore_scores(dataloader, extractor, memory_bank,
                             n_chunk=N_CHUNK, m_chunk=M_CHUNK):
    extractor.eval()
    mb = memory_bank.to(DEVICE)        # [M, C]
    M, C = mb.shape

    # 미리 memory_bank norm^2 계산
    mb_norm2 = (mb ** 2).sum(dim=1)    # [M]

    img_scores = []

    for batch_idx, (imgs, _) in enumerate(tqdm(dataloader, desc="compute_patchcore_scores")):
        imgs = imgs.to(DEVICE)
        feats = extractor(imgs)        # [B, C, H, W]
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
        N = feats.size(0)

        # 각 패치별 최소 거리^2 저장
        min_dist2 = torch.full((N,), float("inf"), device=DEVICE)

        # N 방향 chunk
        for n_start in range(0, N, n_chunk):
            n_end = min(n_start + n_chunk, N)
            x = feats[n_start:n_end]                     # [n_chunk, C]
            x_norm2 = (x ** 2).sum(dim=1, keepdim=True)  # [n_chunk, 1]

            local_min = torch.full((x.size(0),), float("inf"), device=DEVICE)

            # M 방향 chunk
            for m_start in range(0, M, m_chunk):
                m_end = min(m_start + m_chunk, M)
                y = mb[m_start:m_end]                    # [m_chunk, C]
                y_norm2 = mb_norm2[m_start:m_end]        # [m_chunk]

                # dist^2 = ||x||^2 + ||y||^2 - 2 x·y
                xy = x @ y.T                             # [n_chunk, m_chunk]
                dist2 = x_norm2 + y_norm2.unsqueeze(0) - 2 * xy  # [n_chunk, m_chunk]

                local_min = torch.minimum(local_min, dist2.min(dim=1)[0])

            min_dist2[n_start:n_end] = torch.minimum(min_dist2[n_start:n_end],
                                                     local_min)

        # 이미지별 score (max distance)
        min_dist2 = min_dist2.reshape(B, H, W)
        score_per_img = min_dist2.view(B, -1).max(dim=1)[0]  # [B]

        img_scores.extend(score_per_img.cpu().numpy().tolist())

        if batch_idx % 10 == 0:
            print(f"  scored batch {batch_idx}, total scores: {len(img_scores)}")

    return np.array(img_scores)

def main():
    print("Device:", DEVICE)
    train_loader, test_normal_loader, test_anom_loader = get_dataloaders()

    print("==> Building feature extractor (ResNet18)...")
    extractor = ResNetFeatureExtractor().to(DEVICE)

    if os.path.exists(PATCHCORE_MEM_PATH):
        print(f"Loading memory bank from {PATCHCORE_MEM_PATH}")
        memory_bank = torch.load(PATCHCORE_MEM_PATH, map_location=DEVICE)
    else:
        print("==> Building memory bank from normal training features...")
        memory_bank = build_memory_bank(train_loader, extractor, max_patches=MAX_PATCHES)
        torch.save(memory_bank.cpu(), PATCHCORE_MEM_PATH)
        print(f"Saved PatchCore memory bank to {PATCHCORE_MEM_PATH}")

    print("==> Computing scores for normal test images...")
    scores_normal = compute_patchcore_scores(test_normal_loader, extractor, memory_bank)

    print("==> Computing scores for anomaly test images...")
    scores_anom = compute_patchcore_scores(test_anom_loader, extractor, memory_bank)

    # normal=0, anomaly=1
    y_true = np.concatenate([
        np.zeros_like(scores_normal),
        np.ones_like(scores_anom),
    ])
    y_scores = np.concatenate([scores_normal, scores_anom])

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    print("\n==> PatchCore minimal results")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # 정상 분포 기준 threshold 예시
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


if __name__ == "__main__":
    main()
