"""
Attempt 5: Fine-tune HandSegNetCPMFast on FreiHand dataset
Using pretrained weights from attempt4 transfer learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Import model and dataset
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast
from handsegnet.attempt1_owncnnarchitecture.dataset.datasetloader import FreiHandSegDataset

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_ROOT = REPO_ROOT / "alldata" / "freihand_data"

# Pretrained model from attempt4
PRETRAINED_MODEL = REPO_ROOT / "handsegnet" / "attempt4_usingroboflow+rhd+cpm" / "trained_models" / "transfer_learning_20260309_014131" / "final_model.pth"

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 15 # 10-15 range
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-5

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = SCRIPT_DIR / "trained_models" / f"freihand_finetune_{TIMESTAMP}"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================================
# LOSS FUNCTION
# ============================================================================
def compute_loss(outputs, targets):
    """
    Compute combined segmentation loss for CPM multi-stage output

    Args:
        outputs: Model outputs (3 stages) - tuple of 3 tensors
        targets: Ground truth masks (B, 1, H, W)

    Returns:
        Total loss
    """
    criterion = nn.BCEWithLogitsLoss()

    loss = 0
    for i, output in enumerate(outputs):
        # Interpolate output to match target size
        if output.shape != targets.shape:
            output = torch.nn.functional.interpolate(
                output, size=targets.shape[2:],
                mode='bilinear', align_corners=False
            )

        # Weight stages (later stages more important)
        weight = 0.2 if i == 0 else (0.3 if i == 1 else 0.5)
        loss += weight * criterion(output, targets)

    return loss


# ============================================================================
# TRAINING EPOCH
# ============================================================================
def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass - returns 3 stages
        s1, s2, s3 = model(images)

        # Compute loss
        loss = compute_loss((s1, s2, s3), masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ============================================================================
# VALIDATION
# ============================================================================
def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            s1, s2, s3 = model(images)

            # Compute loss
            loss = compute_loss((s1, s2, s3), masks)

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ============================================================================
# COMPUTE METRICS
# ============================================================================
def compute_metrics(model, dataloader, device):
    """Compute IoU and Dice metrics"""
    model.eval()
    
    total_iou = 0
    total_dice = 0
    count = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing metrics"):
            images = images.to(device)
            masks = masks.to(device)

            # Get final stage output
            _, _, output = model(images)
            
            # Interpolate to mask size
            output = torch.nn.functional.interpolate(
                output, size=masks.shape[2:],
                mode='bilinear', align_corners=False
            )
            
            # Binary prediction
            pred = (torch.sigmoid(output) > 0.5).float()

            for p, m in zip(pred, masks):
                p = p.view(-1)
                m = m.view(-1)
                
                intersection = (p * m).sum()
                union = p.sum() + m.sum() - intersection
                
                iou = intersection / (union + 1e-6)
                dice = (2 * intersection) / (p.sum() + m.sum() + 1e-6)
                
                total_iou += iou.item()
                total_dice += dice.item()
                count += 1

    return total_iou / count, total_dice / count


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train():
    """Main training function"""

    print("=" * 60)
    print("ATTEMPT 5: FREIHAND FINE-TUNING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Pretrained model: {PRETRAINED_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    print("Loading FreiHand dataset...")
    dataset = FreiHandSegDataset(str(DATA_ROOT), split="training")
    print(f"Total samples: {len(dataset)}")

    # Split into train and validation (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}\n")

    # -------------------------------------------------------------------------
    # Load model with pretrained weights
    # -------------------------------------------------------------------------
    print("Loading pretrained model...")
    model = HandSegNetCPMFast().to(DEVICE)

    if PRETRAINED_MODEL.exists():
        state_dict = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained weights from {PRETRAINED_MODEL.name}")
    else:
        print(f"⚠️ Pretrained model not found at {PRETRAINED_MODEL}")
        print("   Training from scratch...")

    # -------------------------------------------------------------------------
    # Setup optimizer and scheduler
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print(f"\nOptimizer: AdamW (lr={LEARNING_RATE})")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"Epochs: {NUM_EPOCHS}\n")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print("Starting training...\n")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print("=" * 60)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.6f}")

        # Learning rate step
        scheduler.step()

        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch+1:03d}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = OUTPUT_DIR / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved (val_loss: {val_loss:.6f})")

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model for final metrics
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE))
    iou, dice = compute_metrics(model, val_loader, DEVICE)
    print(f"\nBest Model Metrics (on validation set):")
    print(f"  IoU:  {iou:.4f}")
    print(f"  Dice: {dice:.4f}")

    # -------------------------------------------------------------------------
    # Save final model
    # -------------------------------------------------------------------------
    final_model_path = OUTPUT_DIR / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)

    # -------------------------------------------------------------------------
    # Save training statistics
    # -------------------------------------------------------------------------
    stats_file = OUTPUT_DIR / "training_stats.txt"
    with open(stats_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ATTEMPT 5: FREIHAND FINE-TUNING STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Pretrained model: {PRETRAINED_MODEL}\n")
        f.write(f"Dataset: FreiHand\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n\n")

        f.write("FINAL RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.6f}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Best model IoU: {iou:.4f}\n")
        f.write(f"Best model Dice: {dice:.4f}\n\n")

        f.write("EPOCH-BY-EPOCH RESULTS:\n")
        f.write("-" * 60 + "\n")
        f.write("Epoch | Train Loss | Val Loss\n")
        f.write("-" * 60 + "\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1:5d} | {tl:10.6f} | {vl:10.6f}\n")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Statistics: {stats_file}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
