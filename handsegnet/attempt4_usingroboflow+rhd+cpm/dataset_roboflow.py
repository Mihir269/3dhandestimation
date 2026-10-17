import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class RoboflowSegmentationDataset(Dataset):
    """
    Dataset class for Roboflow segmentation data
    Converts polygon annotations to segmentation masks
    """

    def __init__(self, images_dir, labels_dir):
        """
        Args:
            images_dir: Path to directory containing images
            labels_dir: Path to directory containing label files
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(images_dir)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get image and segmentation mask

        Returns:
            img: Image tensor (3, H, W)
            mask: Segmentation mask tensor (1, H, W)
        """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        h, w = img.shape[:2]

        # Convert image to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Create empty mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Load annotations
        label_name = Path(img_name).stem + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        if os.path.exists(label_path):
            mask = self._load_polygon_mask(label_path, h, w)

        # Convert to tensors
        img_tensor = torch.tensor(img).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        return img_tensor, mask_tensor

    def _load_polygon_mask(self, label_path, h, w):
        """
        Load polygon annotation and convert to segmentation mask

        Args:
            label_path: Path to label file
            h: Image height
            w: Image width

        Returns:
            mask: Segmentation mask (h, w) with values in [0, 1]
        """
        mask = np.zeros((h, w), dtype=np.float32)

        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()

            if not line:
                return mask

            # Parse YOLO format: class_id x1 y1 x2 y2 ... (normalized coordinates)
            parts = line.split()

            if len(parts) < 3:  # At least class_id and one point
                return mask

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Check if we have valid polygon (at least 3 points)
            if len(coords) < 6:  # Need at least 3 points (x,y pairs)
                return mask

            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    points.append([x, y])

            if len(points) >= 3:
                # Draw filled polygon on mask
                points_array = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points_array], 1.0)

        except Exception as e:
            print(f"Error loading annotation from {label_path}: {e}")
            return mask

        return mask
