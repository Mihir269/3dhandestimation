"""
PosePrior Network - 3D Hand Pose Estimation from 2D Keypoints
Based on: "Learning to Estimate 3D Hand Pose from Single RGB Images" (Zimmermann & Brox, ICCV 2017)

This network lifts 2D keypoints (from MediaPipe) to 3D coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PosePriorNetwork(nn.Module):
    """
    PosePrior Network from Zimmermann & Brox ICCV 2017.

    Takes 2D keypoint score maps and predicts:
    1. Canonical 3D coordinates (hand in canonical orientation)
    2. Viewpoint rotation (to transform back to camera frame)

    Final output: 3D coordinates relative to root (wrist)
    """

    def __init__(self, num_keypoints=21, scoremap_size=32):
        super(PosePriorNetwork, self).__init__()

        self.num_keypoints = num_keypoints
        self.scoremap_size = scoremap_size

        # Convolutional layers for processing score maps
        # Architecture from paper: 6 conv layers
        self.conv1 = nn.Conv2d(num_keypoints, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Calculate flattened size after convolutions
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        self.flat_size = 256 * 4 * 4

        # Fully connected layers (shared)
        self.fc1 = nn.Linear(self.flat_size + 1, 1024)  # +1 for hand side
        self.fc2 = nn.Linear(1024, 512)

        # Canonical coordinates branch (21 keypoints * 3 coords)
        self.fc_canonical = nn.Linear(512, num_keypoints * 3)

        # Viewpoint branch (axis-angle representation: 3 values)
        self.fc_viewpoint = nn.Linear(512, 3)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, scoremaps, hand_side):
        """
        Forward pass.

        Args:
            scoremaps: (B, 21, 32, 32) - 2D keypoint score maps
            hand_side: (B, 1) - 0 for left, 1 for right

        Returns:
            coords_rel: (B, 21, 3) - 3D coordinates relative to wrist
            coords_canonical: (B, 21, 3) - canonical coordinates
            rotation: (B, 3, 3) - rotation matrix
        """
        batch_size = scoremaps.size(0)

        # Convolutional feature extraction
        x = F.relu(self.conv1(scoremaps))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 32 -> 16

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 16 -> 8

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)  # 8 -> 4

        x = x.view(batch_size, -1)

        # Concatenate hand side info
        x = torch.cat([x, hand_side], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Canonical coordinates
        coords_canonical = self.fc_canonical(x)
        coords_canonical = coords_canonical.view(batch_size, self.num_keypoints, 3)

        # Viewpoint (axis-angle to rotation matrix)
        axis_angle = self.fc_viewpoint(x)
        rotation = self.axis_angle_to_rotation_matrix(axis_angle)

        # Transform canonical to relative coordinates
        # w_rel = w_canonical @ R^T
        coords_rel = torch.bmm(coords_canonical, rotation.transpose(1, 2))

        return coords_rel, coords_canonical, rotation

    def axis_angle_to_rotation_matrix(self, axis_angle):
        """
        Convert axis-angle representation to rotation matrix using Rodrigues formula.

        Args:
            axis_angle: (B, 3) - rotation vector (angle * axis)

        Returns:
            R: (B, 3, 3) - rotation matrix
        """
        batch_size = axis_angle.size(0)

        # Compute angle
        angle = torch.norm(axis_angle, dim=1, keepdim=True)
        angle = angle.clamp(min=1e-8)

        # Normalize to get axis
        axis = axis_angle / angle

        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        # where K is the skew-symmetric matrix of axis

        cos_angle = torch.cos(angle).unsqueeze(2)
        sin_angle = torch.sin(angle).unsqueeze(2)

        # Skew-symmetric matrix
        K = torch.zeros(batch_size, 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        # Identity matrix
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Rotation matrix
        R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)

        return R


class PosePriorSimple(nn.Module):
    """
    Simplified PosePrior that directly maps 2D keypoints to 3D.
    For cases where we have precise 2D coordinates from MediaPipe.
    """

    def __init__(self, num_keypoints=21):
        super(PosePriorSimple, self).__init__()

        self.num_keypoints = num_keypoints
        input_size = num_keypoints * 2  # 21 keypoints * 2 (x, y)

        # MLP architecture
        self.fc1 = nn.Linear(input_size + 1, 512)  # +1 for hand side
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

        # Canonical coordinates
        self.fc_canonical = nn.Linear(256, num_keypoints * 3)

        # Viewpoint
        self.fc_viewpoint = nn.Linear(256, 3)

        self.dropout = nn.Dropout(0.3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, keypoints_2d, hand_side):
        """
        Args:
            keypoints_2d: (B, 21, 2) - normalized 2D keypoints
            hand_side: (B, 1) - 0 for left, 1 for right
        """
        batch_size = keypoints_2d.size(0)

        # Flatten keypoints
        x = keypoints_2d.view(batch_size, -1)
        x = torch.cat([x, hand_side], dim=1)

        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))

        # Canonical coordinates
        coords_canonical = self.fc_canonical(x)
        coords_canonical = coords_canonical.view(batch_size, self.num_keypoints, 3)

        # Viewpoint
        axis_angle = self.fc_viewpoint(x)
        rotation = self._axis_angle_to_rotation_matrix(axis_angle)

        # Transform to relative coordinates
        coords_rel = torch.bmm(coords_canonical, rotation.transpose(1, 2))

        return coords_rel, coords_canonical, rotation

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        batch_size = axis_angle.size(0)
        angle = torch.norm(axis_angle, dim=1, keepdim=True).clamp(min=1e-8)
        axis = axis_angle / angle

        cos_angle = torch.cos(angle).unsqueeze(2)
        sin_angle = torch.sin(angle).unsqueeze(2)

        K = torch.zeros(batch_size, 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(batch_size, -1, -1)
        R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)

        return R


def create_scoremaps(keypoints_2d, image_size=320, scoremap_size=32, sigma=2.0):
    """
    Create Gaussian score maps from 2D keypoints.

    Args:
        keypoints_2d: (B, 21, 2) - 2D keypoints in pixel coordinates
        image_size: Original image size
        scoremap_size: Output score map size
        sigma: Gaussian sigma

    Returns:
        scoremaps: (B, 21, scoremap_size, scoremap_size)
    """
    batch_size, num_keypoints, _ = keypoints_2d.shape
    device = keypoints_2d.device

    # Scale keypoints to scoremap size
    scale = scoremap_size / image_size
    kp_scaled = keypoints_2d * scale

    # Create coordinate grids
    x = torch.arange(scoremap_size, device=device).float()
    y = torch.arange(scoremap_size, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Expand for batch and keypoints
    xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_keypoints, -1, -1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, num_keypoints, -1, -1)

    # Keypoint centers
    kp_x = kp_scaled[:, :, 0].unsqueeze(2).unsqueeze(3)
    kp_y = kp_scaled[:, :, 1].unsqueeze(2).unsqueeze(3)

    # Gaussian score maps
    scoremaps = torch.exp(-((xx - kp_x)**2 + (yy - kp_y)**2) / (2 * sigma**2))

    return scoremaps
