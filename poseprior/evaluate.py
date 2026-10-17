"""
Evaluation Script for PosePrior Network
Tests the trained 2D to 3D hand pose lifting network on test dataset.

Usage:
    cd poseprior
    python evaluate.py --checkpoint checkpoints/best.pth --dataset rhd
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from models.pose_prior import PosePriorNetwork, PosePriorSimple, create_scoremaps
from datasets.hand_dataset import RHDDataset, FreiHANDDataset


LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]


def procrustes_align(pred, gt):
    """
    Procrustes alignment of predictions to ground truth.

    Args:
        pred: (N, 21, 3) predictions
        gt: (N, 21, 3) ground truth

    Returns:
        aligned: (N, 21, 3) aligned predictions
    """
    aligned = []
    for p, g in zip(pred, gt):
        # Center
        p_centered = p - p.mean(axis=0)
        g_centered = g - g.mean(axis=0)

        # Scale
        p_scale = np.sqrt(np.sum(p_centered ** 2))
        g_scale = np.sqrt(np.sum(g_centered ** 2))

        if p_scale < 1e-8:
            aligned.append(p)
            continue

        p_norm = p_centered / p_scale
        g_norm = g_centered / g_scale

        # Rotation (SVD)
        H = p_norm.T @ g_norm
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply
        p_aligned = p_norm @ R * g_scale + g.mean(axis=0)
        aligned.append(p_aligned)

    return np.array(aligned)


def compute_per_keypoint_error(pred, gt, visibility=None):
    """
    Compute error for each of the 21 keypoints.

    Args:
        pred: (N, 21, 3) predictions
        gt: (N, 21, 3) ground truth
        visibility: (N, 21) visibility mask

    Returns:
        per_keypoint_stats: dict with stats for each keypoint
    """
    # Per-joint errors
    errors = np.linalg.norm(pred - gt, axis=2)  # (N, 21)

    # Apply visibility mask
    if visibility is not None:
        errors = errors * visibility

    per_keypoint_stats = []
    for i, name in enumerate(LANDMARK_NAMES):
        if visibility is not None:
            valid_errors = errors[:, i][visibility[:, i] > 0]
        else:
            valid_errors = errors[:, i]

        if len(valid_errors) > 0:
            stats = {
                'id': i,
                'name': name,
                'mean': float(np.mean(valid_errors)),
                'std': float(np.std(valid_errors)),
                'min': float(np.min(valid_errors)),
                'max': float(np.max(valid_errors)),
                'median': float(np.median(valid_errors))
            }
        else:
            stats = {
                'id': i,
                'name': name,
                'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0
            }
        per_keypoint_stats.append(stats)

    return per_keypoint_stats


def compute_pck_auc(errors, thresholds=None):
    """
    Compute PCK at various thresholds and AUC.

    Args:
        errors: (N, 21) per-keypoint errors

    Returns:
        pck_values: dict with PCK at each threshold
        auc: area under PCK curve
    """
    if thresholds is None:
        thresholds = np.linspace(0, 0.5, 51)  # 0 to 0.5 normalized

    pck_values = {}
    pck_curve = []

    for thresh in thresholds:
        pck = np.mean(errors < thresh) * 100
        pck_curve.append(pck)
        if thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            pck_values[f'pck_{int(thresh*200)}mm'] = pck

    auc = np.trapz(pck_curve, thresholds) / (thresholds[-1] - thresholds[0])

    return pck_values, auc, pck_curve


def evaluate(model, dataloader, device, use_scoremaps=False, scale_factor=200.0):
    """
    Evaluate model on dataset.

    Args:
        model: PosePrior model
        dataloader: Test data loader
        device: torch device
        use_scoremaps: Whether to use score maps
        scale_factor: Scale factor to convert normalized coords to mm

    Returns:
        results: dict with all metrics
    """
    model.eval()

    all_pred = []
    all_gt = []
    all_vis = []
    all_palm_lengths = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            keypoints_2d = batch['keypoints_2d'].to(device)
            keypoints_3d = batch['keypoints_3d'].numpy()
            hand_side = batch['hand_side'].to(device)
            visibility = batch['visibility'].numpy()
            palm_length = batch['palm_length'].numpy()

            if use_scoremaps:
                keypoints_2d_pixel = batch['keypoints_2d_pixel'].to(device)
                scoremaps = create_scoremaps(keypoints_2d_pixel, image_size=320, scoremap_size=32)
                pred_rel, _, _ = model(scoremaps, hand_side)
            else:
                pred_rel, _, _ = model(keypoints_2d, hand_side)

            all_pred.append(pred_rel.cpu().numpy())
            all_gt.append(keypoints_3d)
            all_vis.append(visibility)
            all_palm_lengths.append(palm_length)

    # Concatenate
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    all_vis = np.concatenate(all_vis, axis=0)
    all_palm_lengths = np.concatenate(all_palm_lengths, axis=0)

    print(f"\nTotal samples: {len(all_pred)}")

    # Procrustes alignment
    print("Applying Procrustes alignment...")
    all_pred_aligned = procrustes_align(all_pred, all_gt)

    # Compute errors
    errors_raw = np.linalg.norm(all_pred - all_gt, axis=2)
    errors_aligned = np.linalg.norm(all_pred_aligned - all_gt, axis=2)

    # Scale to mm
    errors_raw_mm = errors_raw * scale_factor
    errors_aligned_mm = errors_aligned * scale_factor

    # MPJPE
    mpjpe_raw = np.mean(errors_raw_mm)
    mpjpe_aligned = np.mean(errors_aligned_mm)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  MPJPE (raw): {mpjpe_raw:.2f} mm")
    print(f"  MPJPE (aligned): {mpjpe_aligned:.2f} mm")

    # PCK and AUC
    pck_values_raw, auc_raw, _ = compute_pck_auc(errors_raw)
    pck_values_aligned, auc_aligned, _ = compute_pck_auc(errors_aligned)

    print(f"\n  AUC (raw): {auc_raw:.2f}")
    print(f"  AUC (aligned): {auc_aligned:.2f}")

    print(f"\n  PCK (aligned):")
    for k, v in sorted(pck_values_aligned.items()):
        print(f"    {k}: {v:.2f}%")

    # Per-keypoint analysis
    print("\n" + "=" * 70)
    print("PER-KEYPOINT ERROR ANALYSIS (aligned, mm)")
    print("=" * 70)
    print(f"{'ID':<4} {'Landmark':<20} {'Mean':<10} {'Std':<10} {'Median':<10}")
    print("-" * 70)

    per_keypoint_stats = compute_per_keypoint_error(
        all_pred_aligned * scale_factor,
        all_gt * scale_factor,
        all_vis
    )

    for stats in per_keypoint_stats:
        print(f"{stats['id']:<4} {stats['name']:<20} {stats['mean']:<10.2f} "
              f"{stats['std']:<10.2f} {stats['median']:<10.2f}")

    # Finger-wise analysis
    print("\n" + "=" * 70)
    print("FINGER-WISE ERROR ANALYSIS (aligned, mm)")
    print("=" * 70)

    finger_groups = {
        'Wrist': [0],
        'Thumb': [1, 2, 3, 4],
        'Index': [5, 6, 7, 8],
        'Middle': [9, 10, 11, 12],
        'Ring': [13, 14, 15, 16],
        'Pinky': [17, 18, 19, 20]
    }

    finger_stats = {}
    for finger, indices in finger_groups.items():
        finger_errors = errors_aligned_mm[:, indices].flatten()
        mean_err = np.mean(finger_errors)
        std_err = np.std(finger_errors)
        print(f"  {finger:<10}: {mean_err:.2f} mm (+/- {std_err:.2f})")
        finger_stats[finger] = {'mean_mm': mean_err, 'std_mm': std_err}

    # Compile results
    results = {
        'num_samples': len(all_pred),
        'overall': {
            'mpjpe_raw_mm': float(mpjpe_raw),
            'mpjpe_aligned_mm': float(mpjpe_aligned),
            'auc_raw': float(auc_raw),
            'auc_aligned': float(auc_aligned)
        },
        'pck_raw': {k: float(v) for k, v in pck_values_raw.items()},
        'pck_aligned': {k: float(v) for k, v in pck_values_aligned.items()},
        'per_keypoint': per_keypoint_stats,
        'finger_wise': {k: {kk: float(vv) for kk, vv in v.items()}
                        for k, v in finger_stats.items()}
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PosePrior Network')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='rhd', choices=['rhd', 'freihand'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    # Set data directory (relative to parent folder)
    if args.data_dir is None:
        if args.dataset == 'rhd':
            args.data_dir = '../alldata/RHD_data/RHD_published_v2'
        else:
            args.data_dir = '../alldata/freihand_data'

    # Create model
    model_type = config.get('model', 'simple')
    use_scoremaps = (model_type == 'full')

    if model_type == 'simple':
        model = PosePriorSimple(num_keypoints=21)
    else:
        model = PosePriorNetwork(num_keypoints=21, scoremap_size=32)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model type: {model_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test dataset
    print(f"\nLoading test dataset: {args.dataset}")
    if args.dataset == 'rhd':
        test_dataset = RHDDataset(args.data_dir, split='evaluation')
    else:
        # FreiHAND - use evaluation set if available, else use training subset
        try:
            test_dataset = FreiHANDDataset(args.data_dir, split='evaluation')
        except:
            print("Using training subset for evaluation")
            test_dataset = FreiHANDDataset(args.data_dir, split='training')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Starting Evaluation")
    print("=" * 70)

    results = evaluate(model, test_loader, device, use_scoremaps)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f'evaluation_{args.dataset}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
