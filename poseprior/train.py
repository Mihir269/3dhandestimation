"""
Training Script for PosePrior Network
Trains the 2D to 3D hand pose lifting network from Zimmermann & Brox ICCV 2017.

Usage:
    cd poseprior
    python train.py --dataset rhd --epochs 100
    python train.py --dataset freihand --epochs 100
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Tensorboard not available, skipping logging")
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from models.pose_prior import PosePriorNetwork, PosePriorSimple, create_scoremaps
from datasets.hand_dataset import RHDDataset, FreiHANDDataset


class PosePriorLoss(nn.Module):
    """
    Combined loss for PosePrior network.
    Based on losses from the paper:
    - L_c: Canonical coordinates loss
    - L_r: Rotation matrix loss
    - L_rel: Relative coordinates loss
    """

    def __init__(self, lambda_canonical=1.0, lambda_rotation=0.1, lambda_rel=1.0):
        super(PosePriorLoss, self).__init__()
        self.lambda_canonical = lambda_canonical
        self.lambda_rotation = lambda_rotation
        self.lambda_rel = lambda_rel

    def forward(self, pred_rel, pred_canonical, pred_rotation,
                gt_3d, visibility=None):
        """
        Compute combined loss.

        Args:
            pred_rel: (B, 21, 3) predicted relative coordinates
            pred_canonical: (B, 21, 3) predicted canonical coordinates
            pred_rotation: (B, 3, 3) predicted rotation matrix
            gt_3d: (B, 21, 3) ground truth 3D coordinates (relative, normalized)
            visibility: (B, 21) visibility mask

        Returns:
            total_loss, loss_dict
        """
        batch_size = pred_rel.size(0)

        # Apply visibility mask if provided
        if visibility is not None:
            vis_mask = visibility.unsqueeze(2).expand_as(pred_rel)
        else:
            vis_mask = torch.ones_like(pred_rel)

        # Relative coordinates loss (primary loss)
        loss_rel = torch.sum(vis_mask * (pred_rel - gt_3d) ** 2) / (torch.sum(vis_mask) + 1e-8)

        # For canonical loss, we need to compute GT canonical coords
        # We use SVD to find the best rotation that aligns pred to GT
        # Then canonical_gt = gt_3d @ R
        # For simplicity, we enforce consistency: canonical @ R^T should = pred_rel
        pred_rel_from_canonical = torch.bmm(pred_canonical, pred_rotation.transpose(1, 2))
        loss_consistency = torch.mean((pred_rel_from_canonical - pred_rel) ** 2)

        # Rotation orthogonality loss (R @ R^T = I)
        I = torch.eye(3, device=pred_rotation.device).unsqueeze(0).expand(batch_size, -1, -1)
        RRT = torch.bmm(pred_rotation, pred_rotation.transpose(1, 2))
        loss_ortho = torch.mean((RRT - I) ** 2)

        # Total loss
        total_loss = (self.lambda_rel * loss_rel +
                      self.lambda_canonical * loss_consistency +
                      self.lambda_rotation * loss_ortho)

        loss_dict = {
            'total': total_loss.item(),
            'rel': loss_rel.item(),
            'consistency': loss_consistency.item(),
            'ortho': loss_ortho.item()
        }

        return total_loss, loss_dict


def compute_metrics(pred, gt, visibility=None):
    """
    Compute evaluation metrics.

    Args:
        pred: (B, 21, 3) predictions
        gt: (B, 21, 3) ground truth
        visibility: (B, 21) visibility mask

    Returns:
        dict with metrics
    """
    # Per-joint position error
    errors = torch.norm(pred - gt, dim=2)  # (B, 21)

    if visibility is not None:
        # Masked mean
        errors = errors * visibility
        mpjpe = torch.sum(errors) / (torch.sum(visibility) + 1e-8)
    else:
        mpjpe = torch.mean(errors)

    # Convert to mm (assuming normalized coords, multiply by typical hand size ~200mm)
    mpjpe_mm = mpjpe.item() * 200

    # PCK at different thresholds (in normalized space)
    pck_thresholds = [0.05, 0.1, 0.15, 0.2]  # Corresponds to ~10, 20, 30, 40mm
    pck_values = {}

    for thresh in pck_thresholds:
        if visibility is not None:
            correct = ((errors < thresh).float() * visibility).sum()
            total = visibility.sum()
        else:
            correct = (errors < thresh).float().sum()
            total = errors.numel()
        pck_values[f'pck_{int(thresh*200)}mm'] = (correct / (total + 1e-8)).item() * 100

    return {
        'mpjpe': mpjpe.item(),
        'mpjpe_mm': mpjpe_mm,
        **pck_values
    }


def train_epoch(model, dataloader, criterion, optimizer, device, use_scoremaps=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_metrics = []

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        keypoints_2d = batch['keypoints_2d'].to(device)
        keypoints_3d = batch['keypoints_3d'].to(device)
        hand_side = batch['hand_side'].to(device)
        visibility = batch['visibility'].to(device)

        optimizer.zero_grad()

        if use_scoremaps:
            # Create score maps from 2D keypoints
            keypoints_2d_pixel = batch['keypoints_2d_pixel'].to(device)
            scoremaps = create_scoremaps(keypoints_2d_pixel, image_size=320, scoremap_size=32)
            pred_rel, pred_canonical, pred_rotation = model(scoremaps, hand_side)
        else:
            pred_rel, pred_canonical, pred_rotation = model(keypoints_2d, hand_side)

        loss, loss_dict = criterion(pred_rel, pred_canonical, pred_rotation,
                                    keypoints_3d, visibility)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(pred_rel, keypoints_3d, visibility)
            all_metrics.append(metrics)

        pbar.set_postfix({'loss': f"{loss.item():.4f}",
                          'mpjpe': f"{metrics['mpjpe_mm']:.2f}mm"})

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    return avg_loss, avg_metrics


def validate(model, dataloader, criterion, device, use_scoremaps=False):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            keypoints_2d = batch['keypoints_2d'].to(device)
            keypoints_3d = batch['keypoints_3d'].to(device)
            hand_side = batch['hand_side'].to(device)
            visibility = batch['visibility'].to(device)

            if use_scoremaps:
                keypoints_2d_pixel = batch['keypoints_2d_pixel'].to(device)
                scoremaps = create_scoremaps(keypoints_2d_pixel, image_size=320, scoremap_size=32)
                pred_rel, pred_canonical, pred_rotation = model(scoremaps, hand_side)
            else:
                pred_rel, pred_canonical, pred_rotation = model(keypoints_2d, hand_side)

            loss, _ = criterion(pred_rel, pred_canonical, pred_rotation,
                               keypoints_3d, visibility)
            total_loss += loss.item()

            metrics = compute_metrics(pred_rel, keypoints_3d, visibility)
            all_metrics.append(metrics)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train PosePrior Network')
    parser.add_argument('--dataset', type=str, default='both', choices=['rhd', 'freihand', 'both'],
                        help='Dataset to use (default: both)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset (auto-detected if not specified)')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'full'],
                        help='Model variant: simple (MLP) or full (with score maps)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs per dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine which datasets to train on
    if args.dataset == 'both':
        datasets_to_train = ['rhd', 'freihand']
    else:
        datasets_to_train = [args.dataset]

    # Train on each dataset
    for dataset_name in datasets_to_train:
        print("\n" + "=" * 60)
        print(f"TRAINING ON {dataset_name.upper()} DATASET")
        print("=" * 60)

        # Set data directory
        if dataset_name == 'rhd':
            data_dir = '../alldata/RHD_data/RHD_published_v2'
        else:
            data_dir = '../alldata/freihand_data'

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f"{dataset_name}_{args.model}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config = vars(args).copy()
        config['dataset'] = dataset_name
        config['data_dir'] = data_dir
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Dataset: {dataset_name}")
        print(f"Model: {args.model}")
        print(f"Data dir: {data_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Epochs: {args.epochs}")
        print("=" * 60)

        # Create data loaders
        print("\nLoading datasets...")
        if dataset_name == 'rhd':
            train_dataset = RHDDataset(data_dir, split='training')
            val_dataset = RHDDataset(data_dir, split='evaluation')
        else:
            train_dataset = FreiHANDDataset(data_dir, split='training')
            val_dataset = FreiHANDDataset(data_dir, split='training')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        # Create model
        print("\nCreating model...")
        use_scoremaps = (args.model == 'full')

        if args.model == 'simple':
            model = PosePriorSimple(num_keypoints=21)
        else:
            model = PosePriorNetwork(num_keypoints=21, scoremap_size=32)

        model = model.to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = PosePriorLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                          patience=5, verbose=True)

        # Resume from checkpoint
        start_epoch = 0
        best_val_loss = float('inf')

        if args.resume:
            print(f"\nResuming from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Tensorboard
        if HAS_TENSORBOARD:
            writer = SummaryWriter(output_dir / 'logs')
        else:
            writer = None

        # Training loop
        print("\nStarting training...")
        print("-" * 60)

        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, use_scoremaps
            )

            # Validate
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device, use_scoremaps
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Log metrics
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('MPJPE/train', train_metrics['mpjpe_mm'], epoch)
                writer.add_scalar('MPJPE/val', val_metrics['mpjpe_mm'], epoch)

            print(f"Train Loss: {train_loss:.4f} | MPJPE: {train_metrics['mpjpe_mm']:.2f}mm")
            print(f"Val Loss: {val_loss:.4f} | MPJPE: {val_metrics['mpjpe_mm']:.2f}mm")

            # Save checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'config': config
            }

            # Save latest
            torch.save(ckpt, output_dir / 'latest.pth')

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, output_dir / 'best.pth')
                print(f"  -> New best model saved!")

        if writer:
            writer.close()

        print("\n" + "=" * 60)
        print(f"Training on {dataset_name.upper()} complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {output_dir}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
