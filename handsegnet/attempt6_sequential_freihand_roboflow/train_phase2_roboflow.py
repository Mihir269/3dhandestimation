"""
Attempt 6 — Phase 2: Fine-tune the FreiHAND model on Roboflow data.

Loads the model_after_freihand.pth (RHD + FreiHAND) and trains 12 epochs
on the Roboflow dataset to produce the final model (RHD + FreiHAND + Roboflow).

Same settings as the original train_sequential.py Phase 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from model_handsegnet_cpm_fast import HandSegNetCPMFast

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FreiHAND fine-tuned model (Phase 1 output)
FREIHAND_MODEL = os.path.join(
    os.path.dirname(__file__),
    "trained_models_20260313_155930",
    "model_after_freihand.pth"
)

# Roboflow training data
ROBOFLOW_IMAGES_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/images"
ROBOFLOW_LABELS_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/labels"

# Training hyperparameters — same as original Phase 2
BATCH_SIZE    = 6
NUM_EPOCHS    = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5

# Output
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), f"phase2_roboflow_{TIMESTAMP}")


# ═══════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════

class RoboflowDataset(Dataset):
    """Roboflow dataset with YOLO polygon annotations -> binary masks."""

    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"  Roboflow: found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot load {img_path}")
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mask = np.zeros((h, w), dtype=np.float32)
        label_name = Path(img_name).stem + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

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
                    points = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            points.append([int(coords[i] * w), int(coords[i+1] * h)])
                    if len(points) >= 3:
                        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1.0)

        img_t  = torch.tensor(img).permute(2, 0, 1)
        mask_t = torch.tensor(mask).unsqueeze(0)
        return img_t, mask_t


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

    pbar = tqdm(loader, desc=f"  [Roboflow] Epoch {epoch}")
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
        for imgs, masks in tqdm(loader, desc="  [Roboflow] Val"):
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
    print("  ATTEMPT 6 — Phase 2: Roboflow Fine-tuning")
    print("  FreiHAND model (RHD+FreiHAND) --> + Roboflow")
    print("=" * 70)
    print(f"  Device       : {DEVICE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output dir   : {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load FreiHAND model
    print("Loading FreiHAND fine-tuned model (RHD + FreiHAND)...")
    model = HandSegNetCPMFast().to(DEVICE)

    if not os.path.exists(FREIHAND_MODEL):
        raise FileNotFoundError(f"FreiHAND model not found: {FREIHAND_MODEL}")

    model.load_state_dict(torch.load(FREIHAND_MODEL, map_location=DEVICE))
    print(f"  Loaded: {FREIHAND_MODEL}\n")

    # Load Roboflow dataset
    print("Loading Roboflow dataset...")
    dataset = RoboflowDataset(ROBOFLOW_IMAGES_DIR, ROBOFLOW_LABELS_DIR)

    # 80/20 split
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"  Train: {n_train}  |  Val: {n_val}\n")

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("Starting training...\n")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss, val_iou     = validate(model, val_loader, DEVICE)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch}/{NUM_EPOCHS} -- "
              f"Train loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val loss: {val_loss:.4f}  IoU: {val_iou:.4f}")

        history.append((epoch, train_loss, train_iou, val_loss, val_iou))

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best val loss: {val_loss:.4f}")

    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    # Training report
    report_path = os.path.join(OUTPUT_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ATTEMPT 6 — Phase 2: Roboflow Fine-tuning Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Initial weights  : {FREIHAND_MODEL}\n")
        f.write(f"Roboflow images  : {len(dataset)}\n")
        f.write(f"Train / Val      : {n_train} / {n_val}\n")
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
        f.write(f"Best model    : {best_path}\n")
        f.write(f"Final model   : {final_path}\n")

    print(f"\n{'='*70}")
    print(f"  PHASE 2 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best model  : {best_path}")
    print(f"  Final model : {final_path}  (RHD + FreiHAND + Roboflow)")
    print(f"  Report      : {report_path}")
    print(f"  Output dir  : {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
