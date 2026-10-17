import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from handsegnet.attempt2_usingrhddata.dataset_rhd import RHDSegmentationDataset
from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast
from tqdm import tqdm
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "data/RHD_published_v2/evaluation"
MODEL_PATH = "handsegnet_cpm_rhd.pth"
OUTPUT_DIR = "eval_results_cpm"

BATCH_SIZE = 4

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)


def compute_metrics(pred, gt):
    """
    Compute segmentation metrics

    Args:
        pred: Predicted segmentation mask (binary, values 0 or 1)
        gt: Ground truth segmentation mask (binary, values 0 or 1)

    Returns:
        pixel_acc, iou, dice, precision, recall, f1
    """
    pred = pred.view(-1)
    gt = gt.view(-1)

    # IoU
    intersection = torch.sum(pred * gt)
    union = torch.sum(pred) + torch.sum(gt) - intersection
    iou = intersection / (union + 1e-6)

    # Dice Score
    dice = (2 * intersection) / (torch.sum(pred) + torch.sum(gt) + 1e-6)

    # Pixel Accuracy
    pixel_acc = torch.mean((pred == gt).float())

    # Precision
    tp = torch.sum(pred * gt)
    fp = torch.sum(pred * (1 - gt))
    precision = tp / (tp + fp + 1e-6)

    # Recall
    fn = torch.sum((1 - pred) * gt)
    recall = tp / (tp + fn + 1e-6)

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)

    return {
        "pixel_acc": pixel_acc.item(),
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }


def get_bounding_box(mask):
    """
    Extract bounding box from binary mask

    Args:
        mask: Binary mask (numpy array)

    Returns:
        (x, y, w, h) bounding box or None if no hand detected
    """
    mask = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 100:  # Minimum area threshold
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def evaluate():
    """Main evaluation function"""

    print("Loading dataset...")
    dataset = RHDSegmentationDataset(DATA_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )

    print("Loading model...")
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Evaluating on {len(dataset)} images...")

    # Metrics storage
    all_metrics = {
        "pixel_acc": [],
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    bbox_stats = {
        "detected": 0,
        "not_detected": 0
    }

    # Get original image paths for visualization
    rgb_dir = os.path.join(DATA_ROOT, "color")
    image_files = sorted(os.listdir(rgb_dir))

    image_idx = 0

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(loader, desc="Evaluating")):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass - get stage 3 output (final output)
            _, _, outputs = model(imgs)

            # Interpolate to match mask size
            outputs = F.interpolate(
                outputs,
                size=masks.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # Convert to predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Process each image in batch
            for img_idx, (pred, gt) in enumerate(zip(preds, masks)):
                # Get original image for visualization
                current_image_idx = image_idx + img_idx
                if current_image_idx < len(image_files):
                    img_name = image_files[current_image_idx]
                    img_path = os.path.join(rgb_dir, img_name)
                    original_img = cv2.imread(img_path)

                    # Denormalize prediction for visualization
                    pred_mask = torch.sigmoid(outputs[img_idx]).squeeze().cpu().numpy()

                    # Compute metrics
                    metrics = compute_metrics(pred, gt)
                    for key in metrics:
                        all_metrics[key].append(metrics[key])

                    # Extract bounding box
                    bbox = get_bounding_box(pred_mask)

                    # Create visualization
                    vis_img = original_img.copy()

                    # Draw segmentation mask overlay
                    pred_mask_3ch = np.zeros_like(original_img)
                    pred_mask_uint8 = (pred_mask > 0.5).astype(np.uint8)
                    pred_mask_3ch[pred_mask_uint8 == 1] = [0, 255, 0]  # Green

                    # Blend mask with original image
                    vis_img = cv2.addWeighted(vis_img, 0.7, pred_mask_3ch, 0.3, 0)

                    # Draw bounding box if detected
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(vis_img, f"Hand Detected", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        bbox_stats["detected"] += 1
                    else:
                        cv2.putText(vis_img, "No Hand Detected", (50, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        bbox_stats["not_detected"] += 1

                    # Add metrics text
                    y_offset = 80
                    cv2.putText(vis_img, f"IoU: {metrics['iou']:.4f}", (50, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(vis_img, f"Dice: {metrics['dice']:.4f}", (50, y_offset + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(vis_img, f"Acc: {metrics['pixel_acc']:.4f}", (50, y_offset + 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Save visualization
                    output_path = os.path.join(OUTPUT_DIR, "visualizations", f"result_{current_image_idx:05d}_{img_name}")
                    cv2.imwrite(output_path, vis_img)

                    # Save predicted mask
                    mask_output_path = os.path.join(OUTPUT_DIR, "masks", f"mask_{current_image_idx:05d}_{img_name.replace('png', 'png')}")
                    pred_mask_uint8_colored = (pred_mask > 0.5).astype(np.uint8) * 255
                    cv2.imwrite(mask_output_path, pred_mask_uint8_colored)

            image_idx += len(imgs)

    # Calculate and print statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS - HandSegNet CPM on RHD Dataset")
    print("="*60)

    mean_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {key: np.std(values) for key, values in all_metrics.items()}

    print(f"\nTotal images evaluated: {len(all_metrics['iou'])}\n")

    print("SEGMENTATION METRICS:")
    print("-" * 60)
    print(f"Pixel Accuracy: {mean_metrics['pixel_acc']:.4f} ± {std_metrics['pixel_acc']:.4f}")
    print(f"IoU (Intersection over Union): {mean_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"Dice Coefficient: {mean_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1 Score: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

    print("\nBOUNDING BOX DETECTION:")
    print("-" * 60)
    total_images = bbox_stats["detected"] + bbox_stats["not_detected"]
    detection_rate = 100 * bbox_stats["detected"] / total_images if total_images > 0 else 0
    print(f"Hands Detected: {bbox_stats['detected']}/{total_images} ({detection_rate:.2f}%)")
    print(f"Hands Not Detected: {bbox_stats['not_detected']}/{total_images}")

    print("\nOUTPUT FILES:")
    print("-" * 60)
    print(f"Visualizations saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'visualizations'))}")
    print(f"Predicted masks saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'masks'))}")

    # Save detailed results to file
    results_file = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS - HandSegNet CPM on RHD Dataset\n")
        f.write("="*60 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Dataset: {DATA_ROOT}\n")
        f.write(f"Total images evaluated: {len(all_metrics['iou'])}\n\n")

        f.write("SEGMENTATION METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Pixel Accuracy: {mean_metrics['pixel_acc']:.4f} ± {std_metrics['pixel_acc']:.4f}\n")
        f.write(f"IoU (Intersection over Union): {mean_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}\n")
        f.write(f"Dice Coefficient: {mean_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}\n")
        f.write(f"Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}\n")
        f.write(f"Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}\n\n")

        f.write("BOUNDING BOX DETECTION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Hands Detected: {bbox_stats['detected']}/{total_images} ({detection_rate:.2f}%)\n")
        f.write(f"Hands Not Detected: {bbox_stats['not_detected']}/{total_images}\n\n")

        f.write("METRIC RANGES:\n")
        f.write("-" * 60 + "\n")
        for metric in ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]:
            values = all_metrics[metric]
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Min: {np.min(values):.4f}\n")
            f.write(f"  Max: {np.max(values):.4f}\n")
            f.write(f"  Mean: {mean_metrics[metric]:.4f}\n")
            f.write(f"  Std: {std_metrics[metric]:.4f}\n\n")

    print(f"Detailed results saved to: {os.path.abspath(results_file)}\n")
    print("="*60)
    print("Evaluation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    evaluate()
