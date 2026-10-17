"""
Evaluate Attempt4 model (HandSegNetCPMFast trained with transfer learning on Roboflow+RHD)
on FreiHAND evaluation data and RHD evaluation data.

Calculates: Pixel Accuracy, IoU, Dice, Precision, Recall, F1
Generates: Visualizations with bounding boxes, predicted masks
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

# ─── Add attempt4 to path so we can import the model ───
ATTEMPT4_DIR = os.path.join(os.path.dirname(__file__), "..", "attempt4_usingroboflow+rhd+cpm")
sys.path.insert(0, ATTEMPT4_DIR)
from model_handsegnet_cpm_fast import HandSegNetCPMFast

# ─── Configuration ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    ATTEMPT4_DIR,
    "trained_models", "transfer_learning_20260309_014131", "final_model.pth"
)

FREIHAND_RGB = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/evaluation/evaluation/rgb"
FREIHAND_SEG = "C:/Users/gunem/studying-2dto3d/alldata/freihand_data/evaluation/evaluation/segmap"

RHD_COLOR = "C:/Users/gunem/studying-2dto3d/alldata/RHD_data/RHD_published_v2/evaluation/color"
RHD_MASK  = "C:/Users/gunem/studying-2dto3d/alldata/RHD_data/RHD_published_v2/evaluation/mask"

BATCH_SIZE = 4
MASK_THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════

class FreiHandEvalDataset(Dataset):
    """FreiHAND evaluation dataset – non-zero segmap pixels = hand"""

    def __init__(self, rgb_dir, seg_dir):
        self.rgb_dir = rgb_dir
        self.seg_dir = seg_dir
        self.files = sorted(os.listdir(rgb_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        stem = os.path.splitext(name)[0]

        img = cv2.imread(os.path.join(self.rgb_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        seg = cv2.imread(os.path.join(self.seg_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = (seg > 0).astype(np.float32)            # any non-zero = hand

        img_t  = torch.tensor(img).permute(2, 0, 1)
        mask_t = torch.tensor(mask).unsqueeze(0)
        return img_t, mask_t, name


class RHDEvalDataset(Dataset):
    """RHD evaluation dataset – labels 2-33 = hand"""

    def __init__(self, color_dir, mask_dir):
        self.color_dir = color_dir
        self.mask_dir  = mask_dir
        self.files = sorted(os.listdir(color_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.color_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        seg = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = ((seg >= 2) & (seg <= 33)).astype(np.float32)

        img_t  = torch.tensor(img).permute(2, 0, 1)
        mask_t = torch.tensor(mask).unsqueeze(0)
        return img_t, mask_t, name


# ═══════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════

def compute_metrics(pred, gt):
    pred = pred.view(-1)
    gt   = gt.view(-1)

    tp = torch.sum(pred * gt)
    fp = torch.sum(pred * (1 - gt))
    fn = torch.sum((1 - pred) * gt)

    intersection = tp
    union = tp + fp + fn

    iou       = (intersection / (union + 1e-6)).item()
    dice      = (2 * intersection / (2 * tp + fp + fn + 1e-6)).item()
    pixel_acc = ((pred == gt).float().mean()).item()
    precision = (tp / (tp + fp + 1e-6)).item()
    recall    = (tp / (tp + fn + 1e-6)).item()
    f1        = (2 * precision * recall / (precision + recall + 1e-6))

    return {
        "pixel_acc": pixel_acc,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ═══════════════════════════════════════════════════════════
#  Bounding box
# ═══════════════════════════════════════════════════════════

def get_bounding_box(mask_np):
    m = (mask_np > MASK_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None
    return cv2.boundingRect(largest)


# ═══════════════════════════════════════════════════════════
#  Evaluation loop (generic)
# ═══════════════════════════════════════════════════════════

def evaluate_dataset(model, dataset, dataset_name, output_root):
    """Run evaluation on a dataset, save visuals + masks + metrics."""

    vis_dir  = os.path.join(output_root, "visualizations")
    mask_dir = os.path.join(output_root, "masks")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_metrics = {k: [] for k in ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]}
    bbox_detected = 0
    bbox_missed   = 0
    global_idx    = 0

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc=f"Eval {dataset_name}"):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            _, _, out = model(imgs)

            out = F.interpolate(out, size=masks.shape[2:], mode='bilinear', align_corners=False)
            preds = (torch.sigmoid(out) > MASK_THRESHOLD).float()

            for i in range(imgs.shape[0]):
                # metrics
                m = compute_metrics(preds[i], masks[i])
                for k in all_metrics:
                    all_metrics[k].append(m[k])

                # mask as numpy
                pred_np = torch.sigmoid(out[i]).squeeze().cpu().numpy()

                # read original (BGR) for visualisation
                name = names[i]
                if dataset_name == "FreiHAND":
                    orig = cv2.imread(os.path.join(FREIHAND_RGB, name))
                else:
                    orig = cv2.imread(os.path.join(RHD_COLOR, name))

                vis = orig.copy()

                # green overlay
                overlay = np.zeros_like(orig)
                overlay[(pred_np > MASK_THRESHOLD)] = [0, 255, 0]
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

                # bounding box
                bbox = get_bounding_box(pred_np)
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(vis, "Hand", (x, y-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    bbox_detected += 1
                else:
                    cv2.putText(vis, "No Hand", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    bbox_missed += 1

                # per-image metrics overlay
                cv2.putText(vis, f"IoU:{m['iou']:.3f} Dice:{m['dice']:.3f}",
                            (10, orig.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imwrite(os.path.join(vis_dir, f"{global_idx:05d}_{name}"), vis)
                cv2.imwrite(os.path.join(mask_dir, f"{global_idx:05d}_{os.path.splitext(name)[0]}.png"),
                            (pred_np > MASK_THRESHOLD).astype(np.uint8) * 255)
                global_idx += 1

    # ── aggregate ──
    mean = {k: np.mean(v) for k, v in all_metrics.items()}
    std  = {k: np.std(v)  for k, v in all_metrics.items()}
    mn   = {k: np.min(v)  for k, v in all_metrics.items()}
    mx   = {k: np.max(v)  for k, v in all_metrics.items()}

    total = bbox_detected + bbox_missed
    det_rate = 100 * bbox_detected / total if total else 0

    # ── print ──
    print(f"\n{'='*60}")
    print(f"RESULTS — Attempt4 Model on {dataset_name}")
    print(f"{'='*60}")
    print(f"Images evaluated : {len(all_metrics['iou'])}")
    print(f"{'─'*60}")
    for k in ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]:
        print(f"{k:>12s}: {mean[k]:.4f} +/- {std[k]:.4f}  (min {mn[k]:.4f}  max {mx[k]:.4f})")
    print(f"{'─'*60}")
    print(f"BBox detected : {bbox_detected}/{total} ({det_rate:.2f}%)")
    print(f"{'='*60}\n")

    # ── save report ──
    report_path = os.path.join(output_root, "evaluation_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Attempt4 Model Evaluation — {dataset_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model : {MODEL_PATH}\n")
        f.write(f"Images: {len(all_metrics['iou'])}\n\n")
        f.write(f"SEGMENTATION METRICS:\n{'─'*60}\n")
        for k in ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]:
            f.write(f"{k:>12s}: {mean[k]:.4f} +/- {std[k]:.4f}  "
                    f"(min {mn[k]:.4f}  max {mx[k]:.4f})\n")
        f.write(f"\nBOUNDING BOX DETECTION:\n{'─'*60}\n")
        f.write(f"Detected    : {bbox_detected}/{total} ({det_rate:.2f}%)\n")
        f.write(f"Not detected: {bbox_missed}/{total}\n")

    print(f"Report saved: {os.path.abspath(report_path)}")
    return mean


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Loading Attempt4 model …")
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out  = os.path.join(os.path.dirname(__file__), f"attempt4_eval_{timestamp}")

    # ── FreiHAND ──
    print("─" * 60)
    print("Evaluating on FreiHAND evaluation set …")
    freihand_ds  = FreiHandEvalDataset(FREIHAND_RGB, FREIHAND_SEG)
    freihand_out = os.path.join(base_out, "freihand")
    freihand_metrics = evaluate_dataset(model, freihand_ds, "FreiHAND", freihand_out)

    # ── RHD ──
    print("─" * 60)
    print("Evaluating on RHD evaluation set …")
    rhd_ds  = RHDEvalDataset(RHD_COLOR, RHD_MASK)
    rhd_out = os.path.join(base_out, "rhd")
    rhd_metrics = evaluate_dataset(model, rhd_ds, "RHD", rhd_out)

    # ── comparison summary ──
    summary_path = os.path.join(base_out, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"ATTEMPT4 MODEL — CROSS-DATASET COMPARISON\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"{'Metric':<14} {'FreiHAND':>10} {'RHD':>10}\n")
        f.write(f"{'─'*34}\n")
        for k in ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]:
            f.write(f"{k:<14} {freihand_metrics[k]:>10.4f} {rhd_metrics[k]:>10.4f}\n")

    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory : {os.path.abspath(base_out)}")
    print(f"Comparison report: {os.path.abspath(summary_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
