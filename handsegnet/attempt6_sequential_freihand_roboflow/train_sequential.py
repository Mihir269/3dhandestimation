"""
Attempt 6 — Sequential Fine-tuning Training Script

Strategy:
  1. Load initial weights from Attempt 3 (trained on RHD with CPM architecture)
  2. Phase 1: Fine-tune on FreiHAND training data for 10-15 epochs
  3. Phase 2: Fine-tune on Roboflow data for 10-15 epochs

Result: A single model that has learned from RHD + FreiHAND + Roboflow
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

# Attempt 3 pre-trained weights (trained on RHD)
ATTEMPT3_WEIGHTS = "C:/Users/gunem/studying-2dto3d/handsegnet/attempt3_usingcpmandrhddata/handsegnet_cpm_rhd.pth"

# FreiHAND training data
FREIHAND_RGB_DIR  = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/training/rgb"
FREIHAND_MASK_DIR = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/training/mask"

# Roboflow training data
ROBOFLOW_IMAGES_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/images"
ROBOFLOW_LABELS_DIR = "C:/Users/gunem/studying-2dto3d/alldata/roboflow_data/train/labels"

# Training hyperparameters
BATCH_SIZE = 8

PHASE1_EPOCHS = 12          # FreiHAND fine-tuning
PHASE1_LR     = 5e-5        # Lower LR to preserve RHD knowledge

PHASE2_EPOCHS = 12          # Roboflow fine-tuning
PHASE2_LR     = 1e-4        # Slightly higher for smaller dataset

WEIGHT_DECAY = 1e-5

# Output
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), f"trained_models_{TIMESTAMP}")


# ═══════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════

class FreiHandTrainDataset(Dataset):
    """
    FreiHAND training dataset.
    Masks are saved as JPG so they have compression artifacts.
    Threshold at 128 to get binary hand mask.
    Only uses the first 32560 images (those that have masks).
    """

    def __init__(self, rgb_dir, mask_dir):
        self.rgb_dir  = rgb_dir
        self.mask_dir = mask_dir

        # Only use images that have corresponding masks
        mask_files = sorted(os.listdir(mask_dir))
        self.files = mask_files
        print(f"  FreiHAND: found {len(self.files)} images with masks")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        stem = os.path.splitext(name)[0]

        # RGB image
        img_path = os.path.join(self.rgb_dir, stem + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Cannot load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Mask (JPG with artifacts — threshold at 128)
        mask_path = os.path.join(self.mask_dir, name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot load {mask_path}")
        mask = (mask > 128).astype(np.float32)

        img_t  = torch.tensor(img).permute(2, 0, 1)     # (3, H, W)
        mask_t = torch.tensor(mask).unsqueeze(0)          # (1, H, W)

        return img_t, mask_t


class RoboflowDataset(Dataset):
    """
    Roboflow dataset with YOLO polygon annotations.
    Converts polygon labels to binary segmentation masks.
    """

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

        # Polygon mask from label file
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
                    if len(parts) < 7:     # class + at least 3 xy pairs
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
#  LOSS
# ═══════════════════════════════════════════════════════════

def compute_loss(outputs, targets):
    """Multi-stage BCE loss (stage 3 weighted highest)."""
    criterion = nn.BCEWithLogitsLoss()
    weights = [0.3, 0.3, 0.4]
    loss = 0.0

    for i, out in enumerate(outputs):
        if out.shape[2:] != targets.shape[2:]:
            out = F.interpolate(out, size=targets.shape[2:],
                                mode="bilinear", align_corners=False)
        loss += weights[i] * criterion(out, targets)

    return loss


# ═══════════════════════════════════════════════════════════
#  METRICS (computed on-the-fly for monitoring)
# ═══════════════════════════════════════════════════════════

def quick_iou(pred_logits, targets):
    """Quick batch IoU for monitoring (uses stage-3 output)."""
    with torch.no_grad():
        if pred_logits.shape[2:] != targets.shape[2:]:
            pred_logits = F.interpolate(pred_logits, size=targets.shape[2:],
                                        mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        inter = (pred * targets).sum()
        union = pred.sum() + targets.sum() - inter
        return (inter / (union + 1e-6)).item()


# ═══════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, epoch, phase_name):
    model.train()
    running_loss = 0.0
    running_iou  = 0.0

    pbar = tqdm(loader, desc=f"  [{phase_name}] Epoch {epoch}")
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


def validate(model, loader, device, phase_name):
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=f"  [{phase_name}] Val"):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            s1, s2, s3 = model(imgs)
            loss = compute_loss([s1, s2, s3], masks)

            running_loss += loss.item()
            running_iou  += quick_iou(s3, masks)

    n = len(loader)
    return running_loss / n, running_iou / n


# ═══════════════════════════════════════════════════════════
#  PHASE RUNNER
# ═══════════════════════════════════════════════════════════

def run_phase(model, train_dataset, num_epochs, lr, phase_name, save_dir):
    """Run a training phase: split data, train, save checkpoints."""

    os.makedirs(save_dir, exist_ok=True)

    # 80/20 train-val split
    n_train = int(0.8 * len(train_dataset))
    n_val   = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"\n  Train split : {n_train}")
    print(f"  Val split   : {n_val}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs       : {num_epochs}\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, DEVICE, epoch, phase_name
        )
        val_loss, val_iou = validate(model, val_loader, DEVICE, phase_name)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch}/{num_epochs} — "
              f"Train loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val loss: {val_loss:.4f}  IoU: {val_iou:.4f}")

        history.append((epoch, train_loss, train_iou, val_loss, val_iou))

        # Save checkpoint every epoch
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"epoch_{epoch:03d}.pth"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best val loss: {val_loss:.4f} — saved to {best_path}")

    return history


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ATTEMPT 6 — Sequential Fine-tuning")
    print("  Attempt3 (RHD) --> FreiHAND --> Roboflow")
    print("=" * 70)
    print(f"  Device       : {DEVICE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Output dir   : {OUTPUT_DIR}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load Attempt 3 weights ──────────────────────
    print("STEP 1: Loading Attempt 3 weights (RHD-trained)")
    print("-" * 70)
    model = HandSegNetCPMFast().to(DEVICE)

    if not os.path.exists(ATTEMPT3_WEIGHTS):
        raise FileNotFoundError(f"Attempt3 weights not found: {ATTEMPT3_WEIGHTS}")

    model.load_state_dict(torch.load(ATTEMPT3_WEIGHTS, map_location=DEVICE))
    print(f"  Loaded: {ATTEMPT3_WEIGHTS}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}\n")

    # ── Step 2: Phase 1 – Fine-tune on FreiHAND ────────────
    print("STEP 2 / PHASE 1: Fine-tuning on FreiHAND data")
    print("-" * 70)
    print("  Loading FreiHAND dataset...")
    freihand_dataset = FreiHandTrainDataset(FREIHAND_RGB_DIR, FREIHAND_MASK_DIR)

    phase1_dir = os.path.join(OUTPUT_DIR, "phase1_freihand")
    phase1_history = run_phase(
        model, freihand_dataset,
        num_epochs=PHASE1_EPOCHS, lr=PHASE1_LR,
        phase_name="FreiHAND", save_dir=phase1_dir
    )

    # Save model after Phase 1
    after_freihand_path = os.path.join(OUTPUT_DIR, "model_after_freihand.pth")
    torch.save(model.state_dict(), after_freihand_path)
    print(f"\n  Phase 1 complete. Model saved: {after_freihand_path}\n")

    # ── Step 3: Phase 2 – Fine-tune on Roboflow ────────────
    print("STEP 3 / PHASE 2: Fine-tuning on Roboflow data")
    print("-" * 70)
    print("  Loading Roboflow dataset...")
    roboflow_dataset = RoboflowDataset(ROBOFLOW_IMAGES_DIR, ROBOFLOW_LABELS_DIR)

    phase2_dir = os.path.join(OUTPUT_DIR, "phase2_roboflow")
    phase2_history = run_phase(
        model, roboflow_dataset,
        num_epochs=PHASE2_EPOCHS, lr=PHASE2_LR,
        phase_name="Roboflow", save_dir=phase2_dir
    )

    # ── Save final model ────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n  Phase 2 complete. Final model saved: {final_path}\n")

    # ── Save training report ────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ATTEMPT 6 — Sequential Fine-tuning Training Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("STRATEGY:\n")
        f.write("  1. Load Attempt 3 weights (trained on RHD)\n")
        f.write(f"  2. Phase 1: Fine-tune on FreiHAND ({PHASE1_EPOCHS} epochs, lr={PHASE1_LR})\n")
        f.write(f"  3. Phase 2: Fine-tune on Roboflow ({PHASE2_EPOCHS} epochs, lr={PHASE2_LR})\n\n")

        f.write(f"Initial weights : {ATTEMPT3_WEIGHTS}\n")
        f.write(f"FreiHAND images : {len(freihand_dataset)}\n")
        f.write(f"Roboflow images : {len(roboflow_dataset)}\n")
        f.write(f"Batch size      : {BATCH_SIZE}\n")
        f.write(f"Device          : {DEVICE}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PHASE 1 — FreiHAND Fine-tuning\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train IoU':>12} {'Val Loss':>12} {'Val IoU':>12}\n")
        for ep, tl, ti, vl, vi in phase1_history:
            f.write(f"{ep:>6d} {tl:>12.6f} {ti:>12.4f} {vl:>12.6f} {vi:>12.4f}\n")

        f.write(f"\nBest val loss: {min(h[3] for h in phase1_history):.6f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PHASE 2 — Roboflow Fine-tuning\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train IoU':>12} {'Val Loss':>12} {'Val IoU':>12}\n")
        for ep, tl, ti, vl, vi in phase2_history:
            f.write(f"{ep:>6d} {tl:>12.6f} {ti:>12.4f} {vl:>12.6f} {vi:>12.4f}\n")

        f.write(f"\nBest val loss: {min(h[3] for h in phase2_history):.6f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  After Phase 1 : {after_freihand_path}\n")
        f.write(f"  Final model   : {final_path}\n")

    # ── Done ────────────────────────────────────────────────
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model after FreiHAND : {after_freihand_path}")
    print(f"  Final model (all 3)  : {final_path}")
    print(f"  Training report      : {report_path}")
    print(f"  Output directory     : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
