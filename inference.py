import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

NORMAL_DIR = "normal"
ANOMALY_DIR = "sampling"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

AE_MODEL_PATH = CKPT_DIR / "autoencoder_256.pth"
AE_RESULT_PATH = CKPT_DIR / "autoencoder_result.json"

PADIM_STAT_PATH = CKPT_DIR / "padim_stats_256.pt"
PADIM_RESULT_PATH = CKPT_DIR / "padim_result.json"

PATCHCORE_MEM_PATH = CKPT_DIR / "patchcore_mem_256.pt"
PATCHCORE_RESULT_PATH = CKPT_DIR / "patchcore_result.json"


class SimpleImageFolder(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.label = label

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


def get_eval_dataloader():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    normal_ds = SimpleImageFolder(NORMAL_DIR, 0, transform)
    anom_ds = SimpleImageFolder(ANOMALY_DIR, 1, transform)

    dataset = torch.utils.data.ConcatDataset([normal_ds, anom_ds])
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if DEVICE == "cuda" else 0,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    return loader


def evaluate_from_scores(y_true, y_scores, thr):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    y_pred = (y_scores > thr).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return auroc, auprc, tp, tn, fp, fn


def find_best_threshold(y_true, y_scores, num_steps: int = 500):
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


class CnnAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class AutoEncoderDetector:
    def __init__(self):
        self.model = CnnAutoEncoder().to(DEVICE)
        print("[AE] Loading weights from", AE_MODEL_PATH)
        self.model.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        if AE_RESULT_PATH.exists():
            print("[AE] Loading threshold from", AE_RESULT_PATH)
            cfg = json.load(open(AE_RESULT_PATH, "r"))
            self.threshold = float(cfg["best_threshold"])
        else:
            print("[AE] Threshold file not found. Will compute later.")
            self.threshold = None

    @torch.no_grad()
    def score_batch(self, imgs):
        imgs = imgs.to(DEVICE, non_blocking=True)
        recon = self.model(imgs)
        err = F.mse_loss(recon, imgs, reduction="none")  
        err = err.view(err.size(0), -1).mean(dim=1)     
        return err.cpu().numpy()

    def run_on_loader(self, loader):
        all_scores, all_labels, all_paths = [], [], []
        for imgs, labels, paths in tqdm(loader, desc="[AE] Scoring"):
            scores = self.score_batch(imgs)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        return all_labels, all_scores, all_paths


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
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
        f1 = self.layers["layer1"]
        f2 = self.layers["layer2"]
        f3 = self.layers["layer3"]

        B, C1, H1, W1 = f1.shape
        f2_up = F.interpolate(f2, size=(H1, W1), mode="bilinear", align_corners=False)
        f3_up = F.interpolate(f3, size=(H1, W1), mode="bilinear", align_corners=False)
        feat = torch.cat([f1, f2_up, f3_up], dim=1)  
        return feat


class PaDiMDetector:
    def __init__(self):
        self.extractor = ResNetFeatureExtractor().to(DEVICE)
        self.extractor.eval()

        print("[PaDiM] Loading stats from", PADIM_STAT_PATH)
        stats = torch.load(PADIM_STAT_PATH, map_location="cpu")
        self.mean = stats["mean"].to(DEVICE)    
        self.cov_inv = stats["cov_inv"].to(DEVICE) 

        if PADIM_RESULT_PATH.exists():
            print("[PaDiM] Loading threshold from", PADIM_RESULT_PATH)
            cfg = json.load(open(PADIM_RESULT_PATH, "r"))
            self.threshold = float(cfg["best_threshold"])
        else:
            print("[PaDiM] Threshold file not found. Will compute later.")
            self.threshold = None

    @torch.no_grad()
    def score_batch(self, imgs):
        imgs = imgs.to(DEVICE, non_blocking=True)
        feat = self.extractor(imgs)            
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(-1, C) 

        mean = self.mean
        cov_inv = self.cov_inv

        diff = feat - mean 
        m = diff @ cov_inv
        m = (m * diff).sum(dim=1)  
        m = m.reshape(B, H, W)  
        score_per_img = m.view(B, -1).mean(dim=1)  
        return score_per_img.cpu().numpy()

    def run_on_loader(self, loader):
        all_scores, all_labels, all_paths = [], [], []
        for imgs, labels, paths in tqdm(loader, desc="[PaDiM] Scoring"):
            scores = self.score_batch(imgs)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        return all_labels, all_scores, all_paths


class PatchCoreDetector:
    def __init__(self):
        self.extractor = ResNetFeatureExtractor().to(DEVICE)
        self.extractor.eval()

        print("[PatchCore] Loading memory bank from", PATCHCORE_MEM_PATH)
        mem_obj = torch.load(PATCHCORE_MEM_PATH, map_location="cpu")
        if isinstance(mem_obj, dict) and "memory" in mem_obj:
            self.memory = mem_obj["memory"].to(DEVICE)  # [N_mem, C]
        else:
            self.memory = mem_obj.to(DEVICE)

        if PATCHCORE_RESULT_PATH.exists():
            print("[PatchCore] Loading threshold from", PATCHCORE_RESULT_PATH)
            cfg = json.load(open(PATCHCORE_RESULT_PATH, "r"))
            self.threshold = float(cfg["best_threshold"])
        else:
            print("[PatchCore] Threshold file not found. Will compute later.")
            self.threshold = None

    @torch.no_grad()
    def score_batch(self, imgs, k=5):
        imgs = imgs.to(DEVICE, non_blocking=True)
        feat = self.extractor(imgs) 
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, -1, C) 

        scores = []
        for b in range(B):
            f = feat[b] 
            dist = torch.cdist(f.unsqueeze(0), self.memory.unsqueeze(0)).squeeze(0) 
            topk_vals, _ = torch.topk(dist, k, dim=1, largest=False) 
            patch_score = topk_vals.mean(dim=1) 

            p = patch_score.numel()
            kk = max(1, int(p * 0.01))
            top_patches, _ = torch.topk(patch_score, kk, largest=True)
            img_score = top_patches.mean()
            scores.append(img_score.item())

        return np.array(scores)

    def run_on_loader(self, loader):
        all_scores, all_labels, all_paths = [], [], []
        for imgs, labels, paths in tqdm(loader, desc="[PatchCore] Scoring"):
            scores = self.score_batch(imgs)
            all_scores.append(scores)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        return all_labels, all_scores, all_paths


def main():
    loader = get_eval_dataloader()

    # 1) AutoEncoder
    print("\n========== AutoEncoder ==========")
    ae = AutoEncoderDetector()
    y_true, y_scores, paths = ae.run_on_loader(loader)

    if ae.threshold is None:
        thr, bf1, bp, br = find_best_threshold(y_true, y_scores)
        ae.threshold = thr
        json.dump({"best_threshold": float(thr)}, open(AE_RESULT_PATH, "w"), indent=2)
        print(f"[AE] Best F1 thr={thr:.6f}, F1={bf1:.4f}, P={bp:.4f}, R={br:.4f}")

    auroc, auprc, tp, tn, fp, fn = evaluate_from_scores(y_true, y_scores, ae.threshold)
    print(f"[AE] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    print(f"[AE] thr={ae.threshold:.6f}  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 2) PaDiM
    print("\n========== PaDiM ==========")
    padim = PaDiMDetector()
    y_true, y_scores, paths = padim.run_on_loader(loader)

    if padim.threshold is None:
        thr, bf1, bp, br = find_best_threshold(y_true, y_scores)
        padim.threshold = thr
        json.dump({"best_threshold": float(thr)}, open(PADIM_RESULT_PATH, "w"), indent=2)
        print(f"[PaDiM] Best F1 thr={thr:.6f}, F1={bf1:.4f}, P={bp:.4f}, R={br:.4f}")

    auroc, auprc, tp, tn, fp, fn = evaluate_from_scores(y_true, y_scores, padim.threshold)
    print(f"[PaDiM] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    print(f"[PaDiM] thr={padim.threshold:.6f}  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 3) PatchCore
    print("\n========== PatchCore ==========")
    patch = PatchCoreDetector()
    y_true, y_scores, paths = patch.run_on_loader(loader)

    if patch.threshold is None:
        thr, bf1, bp, br = find_best_threshold(y_true, y_scores)
        patch.threshold = thr
        json.dump({"best_threshold": float(thr)}, open(PATCHCORE_RESULT_PATH, "w"), indent=2)
        print(f"[PatchCore] Best F1 thr={thr:.6f}, F1={bf1:.4f}, P={bp:.4f}, R={br:.4f}")

    auroc, auprc, tp, tn, fp, fn = evaluate_from_scores(y_true, y_scores, patch.threshold)
    print(f"[PatchCore] AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    print(f"[PatchCore] thr={patch.threshold:.6f}  TP={tp}, TN={tn}, FP={fp}, FN={fn}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    main()
