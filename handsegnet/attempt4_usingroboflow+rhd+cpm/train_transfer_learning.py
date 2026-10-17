import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from dataset_roboflow import RoboflowSegmentationDataset
from model_handsegnet_cpm_fast import HandSegNetCPMFast

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "roboflow_data/train"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

# Pre-trained model path
PRETRAINED_MODEL = "handsegnet_cpm_rhd.pth"

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Output directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"trained_models/transfer_learning_{TIMESTAMP}"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def compute_loss(outputs, targets):
    """
    Compute combined segmentation loss

    Args:
        outputs: Model outputs (3 stages) - list of 3 tensors
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
        weight = 0.3 if i == 0 else (0.3 if i == 1 else 0.4)
        loss += weight * criterion(output, targets)

    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = compute_loss(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
            outputs = model(images)

            # Compute loss
            loss = compute_loss(outputs, masks)

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_transfer_learning():
    """Train model with transfer learning from pre-trained weights"""

    print("="*60)
    print("HAND SEGMENTATION TRAINING - TRANSFER LEARNING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Pre-trained model: {PRETRAINED_MODEL}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    # Load dataset
    print("Loading Roboflow dataset...")
    dataset = RoboflowSegmentationDataset(IMAGES_DIR, LABELS_DIR)
    print(f"Dataset size: {len(dataset)}")

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}\n")

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = HandSegNetCPMFast().to(DEVICE)

    if os.path.exists(PRETRAINED_MODEL):
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=DEVICE))
        print(f"Loaded pre-trained weights from {PRETRAINED_MODEL}")
    else:
        print(f"Warning: Pre-trained model not found at {PRETRAINED_MODEL}")

    model.to(DEVICE)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print("\nStarting training...\n")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved: {best_model_path}")

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Save training statistics
    stats_file = os.path.join(OUTPUT_DIR, "training_stats.txt")
    with open(stats_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("TRANSFER LEARNING TRAINING STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Pre-trained model: {PRETRAINED_MODEL}\n")
        f.write(f"Training dataset size: {len(train_dataset)}\n")
        f.write(f"Validation dataset size: {len(val_dataset)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Number of epochs: {NUM_EPOCHS}\n\n")

        f.write("FINAL RESULTS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.6f}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n\n")

        f.write("EPOCH-BY-EPOCH RESULTS:\n")
        f.write("-"*60 + "\n")
        f.write("Epoch | Train Loss | Val Loss\n")
        f.write("-"*60 + "\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1:5d} | {tl:10.6f} | {vl:10.6f}\n")

    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Statistics saved: {stats_file}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60)


if __name__ == "__main__":
    train_transfer_learning()
