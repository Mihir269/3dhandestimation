"""
Attempt 6 — Final Iteration: Combined Dataset Training

Takes the final model (RHD + FreiHAND + Roboflow) and trains 10 more epochs
on a MIXED dataset sampled as: 40% RHD, 40% FreiHAND, 20% Roboflow.

Since Roboflow is tiny (299) relative to RHD (41k) and FreiHAND (32k), we
oversample Roboflow and subsample the larger sets so each epoch sees a balanced
mix at the requested proportions.

Per-epoch sample budget:   Roboflow * 5 = 1495
  -> RHD:      598  (40%)
  -> FreiHAND: 598  (40%)
  -> Roboflow: 299  (20%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from model_handsegnet_cpm_fast import HandSegNetCPMFast

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial weights — Attempt 6 final model (RHD + FreiHAND + Roboflow sequential)
INITIAL_WEIGHTS = os.path.join(
    os.path.dirname(__file__),
    "phase2_roboflow_20260313_220707",
    "final_model.pth"
)

# Data paths
RHD_COLOR_DIR   = "C:/Users/gunem/studying-2dto3d/alldata/RHD_data/RHD_published_v2/training/color"
RHD_MASK_DIR    = "C:/Users/gunem/studying-2dto3d/alldata/RHD_data/RHD_published_v2/training/mask"

FREIHAND_RGB_DIR  = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/training/rgb"
FREIHAND_MASK_DIR = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/training/mask"

ROBOFLOW_IMAGES_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/images"
ROBOFLOW_LABELS_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/labels"

# Proportions
RHD_RATIO      = 0.40
FREIHAND_RATIO = 0.40
ROBOFLOW_RATIO = 0.20

# Training hyperparameters
BATCH_SIZE    = 6
NUM_EPOCHS    = 10
LEARNING_RATE = 5e-5       # Low LR — the model already knows all three domains
WEIGHT_DECAY  = 1e-5

TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), f"final_combined_{TIMESTAMP}")

# Common image size — all datasets resized to this before batching
TARGET_SIZE = (224, 224)


# ═══════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════

class RHDDataset(Dataset):
    """RHD segmentation — labels 2-33 = hand. Resized to TARGET_SIZE."""

    def __init__(self, color_dir, mask_dir):
        self.color_dir = color_dir
        self.mask_dir  = mask_dir
        self.files = sorted(os.listdir(color_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.color_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        seg = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = ((seg >= 2) & (seg <= 33)).astype(np.float32)

        return torch.tensor(img).permute(2, 0, 1), torch.tensor(mask).unsqueeze(0)


class FreiHandDataset(Dataset):
    """FreiHAND training — mask JPGs thresholded at 128. Resized to TARGET_SIZE."""

    def __init__(self, rgb_dir, mask_dir):
        self.rgb_dir  = rgb_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(mask_dir))   # only images with masks

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        stem = os.path.splitext(name)[0]

        img = cv2.imread(os.path.join(self.rgb_dir, stem + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        seg = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        seg = cv2.resize(seg, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = (seg > 128).astype(np.float32)

        return torch.tensor(img).permute(2, 0, 1), torch.tensor(mask).unsqueeze(0)


class RoboflowDataset(Dataset):
    """Roboflow — YOLO polygon annotations -> binary masks. Resized to TARGET_SIZE."""

    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img = cv2.imread(os.path.join(self.images_dir, img_name))
        h, w = img.shape[:2]

        # Build mask at original resolution first (polygon coords are normalised)
        mask = np.zeros((h, w), dtype=np.float32)
        label_path = os.path.join(self.labels_dir, Path(img_name).stem + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    coords = [float(x) for x in parts[1:]]
                    pts = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            pts.append([int(coords[i] * w), int(coords[i+1] * h)])
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

        # Resize to common size
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        return torch.tensor(img).permute(2, 0, 1), torch.tensor(mask).unsqueeze(0)


# ═══════════════════════════════════════════════════════════
#  BUILD MIXED DATASET FOR ONE EPOCH
# ═══════════════════════════════════════════════════════════

def build_epoch_dataset(rhd_ds, freihand_ds, roboflow_ds):
    """
    Creates a single dataset for one epoch by randomly sampling:
      40% from RHD, 40% from FreiHAND, 20% from Roboflow (all of it).
    """
    n_roboflow = len(roboflow_ds)
    total      = int(n_roboflow / ROBOFLOW_RATIO)     # 1495
    n_rhd      = int(total * RHD_RATIO)                # 598
    n_freihand = total - n_roboflow - n_rhd            # 598

    rhd_indices      = random.sample(range(len(rhd_ds)),      n_rhd)
    freihand_indices = random.sample(range(len(freihand_ds)), n_freihand)
    roboflow_indices = list(range(n_roboflow))                # use all

    rhd_subset      = Subset(rhd_ds,      rhd_indices)
    freihand_subset = Subset(freihand_ds, freihand_indices)
    roboflow_subset = Subset(roboflow_ds, roboflow_indices)

    combined = ConcatDataset([rhd_subset, freihand_subset, roboflow_subset])
    return combined, n_rhd, n_freihand, n_roboflow


# ═══════════════════════════════════════════════════════════
#  LOSS & METRICS
# ═══════════════════════════════════════════════════════════

def compute_loss(outputs, targets):
    criterion = nn.BCEWithLogitsLoss()
    weights = [0.3, 0.3, 0.4]
    loss = 0.0
    for i, out in enumerate(outputs):
        if out.shape[2:] != targets.shape[2:]:
            out = F.interpolate(out, size=targets.shape[2:],
                                mode="bilinear", align_corners=False)
        loss += weights[i] * criterion(out, targets)
    return loss


def quick_iou(pred_logits, targets):
    with torch.no_grad():
        if pred_logits.shape[2:] != targets.shape[2:]:
            pred_logits = F.interpolate(pred_logits, size=targets.shape[2:],
                                        mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        inter = (pred * targets).sum()
        union = pred.sum() + targets.sum() - inter
        return (inter / (union + 1e-6)).item()


# ═══════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_iou  = 0.0

    pbar = tqdm(loader, desc=f"  [Combined] Epoch {epoch}")
    for imgs, masks in pbar:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        s1, s2, s3 = model(imgs)
        loss = compute_loss([s1, s2, s3], masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou = quick_iou(s3, masks)
        running_loss += loss.item()
        running_iou  += iou
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

    n = len(loader)
    return running_loss / n, running_iou / n


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="  [Combined] Val"):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            s1, s2, s3 = model(imgs)
            loss = compute_loss([s1, s2, s3], masks)
            running_loss += loss.item()
            running_iou  += quick_iou(s3, masks)

    n = len(loader)
    return running_loss / n, running_iou / n


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ATTEMPT 6 — Final Iteration: Combined 40/40/20 Training")
    print("  Initial weights: RHD+FreiHAND+Roboflow (sequential)")
    print("  Training mix   : 40% RHD  |  40% FreiHAND  |  20% Roboflow")
    print("=" * 70)
    print(f"  Device       : {DEVICE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output dir   : {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load model
    print("Loading initial weights...")
    model = HandSegNetCPMFast().to(DEVICE)
    if not os.path.exists(INITIAL_WEIGHTS):
        raise FileNotFoundError(f"Weights not found: {INITIAL_WEIGHTS}")
    model.load_state_dict(torch.load(INITIAL_WEIGHTS, map_location=DEVICE))
    print(f"  Loaded: {INITIAL_WEIGHTS}\n")

    # Load all three datasets
    print("Loading datasets...")
    rhd_ds      = RHDDataset(RHD_COLOR_DIR, RHD_MASK_DIR)
    freihand_ds = FreiHandDataset(FREIHAND_RGB_DIR, FREIHAND_MASK_DIR)
    roboflow_ds = RoboflowDataset(ROBOFLOW_IMAGES_DIR, ROBOFLOW_LABELS_DIR)
    print(f"  RHD      : {len(rhd_ds)} images")
    print(f"  FreiHAND : {len(freihand_ds)} images")
    print(f"  Roboflow : {len(roboflow_ds)} images\n")

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop — reshuffle the combined set each epoch
    best_val_loss = float("inf")
    history = []

    print("Starting combined training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Build fresh mixed dataset for this epoch
        combined_ds, n_r, n_f, n_rb = build_epoch_dataset(rhd_ds, freihand_ds, roboflow_ds)

        # 80/20 split
        n_train = int(0.8 * len(combined_ds))
        n_val   = len(combined_ds) - n_train
        train_sub, val_sub = torch.utils.data.random_split(
            combined_ds, [n_train, n_val]
        )

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=False)

        if epoch == 1:
            print(f"  Per-epoch mix: RHD={n_r} | FreiHAND={n_f} | Roboflow={n_rb} "
                  f"| Total={n_r+n_f+n_rb} | Train={n_train} | Val={n_val}\n")

        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss, val_iou     = validate(model, val_loader, DEVICE)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch}/{NUM_EPOCHS} -- "
              f"Train loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val loss: {val_loss:.4f}  IoU: {val_iou:.4f}")

        history.append((epoch, train_loss, train_iou, val_loss, val_iou))

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best val loss: {val_loss:.4f}")

    # Save final
    final_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    # Report
    report_path = os.path.join(OUTPUT_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ATTEMPT 6 — Final Iteration: Combined 40/40/20 Training\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Initial weights  : {INITIAL_WEIGHTS}\n")
        f.write(f"RHD images       : {len(rhd_ds)}  (sampled {n_r} per epoch = 40%)\n")
        f.write(f"FreiHAND images  : {len(freihand_ds)}  (sampled {n_f} per epoch = 40%)\n")
        f.write(f"Roboflow images  : {len(roboflow_ds)}  (all {n_rb} per epoch = 20%)\n")
        f.write(f"Total per epoch  : {n_r + n_f + n_rb}\n")
        f.write(f"Batch size       : {BATCH_SIZE}\n")
        f.write(f"Learning rate    : {LEARNING_RATE}\n")
        f.write(f"Epochs           : {NUM_EPOCHS}\n")
        f.write(f"Device           : {DEVICE}\n\n")

        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train IoU':>12} "
                f"{'Val Loss':>12} {'Val IoU':>12}\n")
        f.write("-" * 60 + "\n")
        for ep, tl, ti, vl, vi in history:
            f.write(f"{ep:>6d} {tl:>12.6f} {ti:>12.4f} {vl:>12.6f} {vi:>12.4f}\n")

        f.write(f"\nBest val loss : {best_val_loss:.6f}\n")
        f.write(f"Best model    : {os.path.join(OUTPUT_DIR, 'best_model.pth')}\n")
        f.write(f"Final model   : {final_path}\n")

    print(f"\n{'='*70}")
    print(f"  COMBINED TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best model  : {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
    print(f"  Final model : {final_path}")
    print(f"  Report      : {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
