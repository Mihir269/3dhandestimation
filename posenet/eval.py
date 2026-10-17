import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader

from models.posenet import PoseNet
from datasets.posenet_dataset import PoseNetDataset


# ---------------- CONFIG ----------------
DATA_ROOT = "cropped_dataset"
MODEL_PATH = "posenet_freihand.pth"

BATCH_SIZE = 32
NUM_WORKERS = 0
VISUALIZE = False  # set True to see predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- KEYPOINT EXTRACTION ----------------
def heatmaps_to_coords(heatmaps):
    """
    heatmaps: [B, 21, H, W]
    returns: [B, 21, 2]
    """
    B, J, H, W = heatmaps.shape
    heatmaps = heatmaps.reshape(B, J, -1)

    idx = heatmaps.argmax(-1)

    ys = idx // W
    xs = idx % W

    coords = torch.stack([xs, ys], dim=-1).float()
    return coords


# ---------------- VISUALIZATION ----------------
def visualize_sample(img_tensor, gt, pred):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x, y) in gt:
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    for (x, y) in pred:
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow("PoseNet Eval", img)
    cv2.waitKey(0)


# ---------------- MAIN ----------------
def main():
    print("Device:", DEVICE)

    # ---------- dataset ----------
    dataset = PoseNetDataset(DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"✅ Eval samples: {len(dataset)}")

    # ---------- model ----------
    model = PoseNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ PoseNet loaded")

    total_error = 0.0
    total_points = 0

    with torch.no_grad():
        for imgs, gt_heatmaps in loader:
            imgs = imgs.to(DEVICE)
            gt_heatmaps = gt_heatmaps.to(DEVICE)

            preds = model(imgs)
            preds = F.interpolate(
                preds,
                size=gt_heatmaps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            pred_coords = heatmaps_to_coords(preds)
            gt_coords = heatmaps_to_coords(gt_heatmaps)

            error = torch.norm(pred_coords - gt_coords, dim=-1).sum()

            total_error += error.item()
            total_points += pred_coords.numel() / 2

            # optional visualization
            if VISUALIZE:
                visualize_sample(imgs[0], gt_coords[0], pred_coords[0])

    mean_pixel_error = total_error / total_points

    print("\n🎯 Mean Pixel Error:", mean_pixel_error)


if __name__ == "__main__":
    main()