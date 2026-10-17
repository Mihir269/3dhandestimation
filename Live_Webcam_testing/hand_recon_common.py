"""
hand_recon_common.py
====================
Shared machinery for live-webcam and offline-video hand reconstruction with the
v12 angle corrector.

Pipeline per frame:
    RGB frame
      -> MediaPipe HandLandmarker  -> 21 world landmarks (metres)
      -> compute_angles_v12        -> 21 articulation angles + 4 fan angles
      -> AngleCorrectorV8          -> 21 CORRECTED angles
      -> forward_kinematics_v12    -> 21 reconstructed keypoints (mm)

The reconstruction uses AVERAGE bone lengths and an AVERAGE palm template, with
the wrist pinned at the origin, so only the joint angles vary between frames.
Drop in a real `hand_calibration.json` to replace both.

NOTE: no temporal filtering here. Frame-to-frame jitter is exactly the raw
measurement noise the Kalman filter is meant to absorb later.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anatomical_angles_v12 import (
    ARTICULATION_ANGLES, FAN_ANGLE_NAMES, BONES, FINGER_CHAINS,
    compute_angles_v12, forward_kinematics_v12, palm_frame,
)

# ---------------------------------------------------------------------------
# Average hand model
# ---------------------------------------------------------------------------
# Derived from a canonical flat adult right hand built from published segment
# tables (hand length 181 mm, index->pinky knuckle span 70.5 mm). Round-trips
# through compute_angles_v12 / forward_kinematics_v12 at 3e-4 mm, so the
# constants are guaranteed self-consistent with the angle module.
DEFAULT_BONE_MM = {
    (0, 1): 35.6, (1, 2): 46.0, (2, 3): 32.0, (3, 4): 24.0,
    (0, 5): 94.0, (5, 6): 40.0, (6, 7): 23.0, (7, 8): 18.0,
    (0, 9): 92.1, (9, 10): 44.0, (10, 11): 26.0, (11, 12): 19.0,
    (0, 13): 90.6, (13, 14): 41.0, (14, 15): 25.0, (15, 16): 18.0,
    (0, 17): 88.1, (17, 18): 33.0, (18, 19): 18.0, (19, 20): 17.0,
}
DEFAULT_FAN_DEG = {
    'Thumb_Index_Fan': 31.3, 'Index_Middle_Fan': 17.4,
    'Middle_Ring_Fan': 13.9, 'Ring_Pinky_Fan': 14.0,
}

FINGER_COLOR = {'Thumb': '#FF6B6B', 'Index': '#4ECDC4', 'Middle': '#45B7D1',
                'Ring': '#96CEB4', 'Pinky': '#FFEAA7'}
FINGER_OF = {j: f for f, ch in FINGER_CHAINS.items() for j in ch}
FINGER_OF[0] = 'Middle'


def _hex2bgr(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


FINGER_BGR = {f: _hex2bgr(c) for f, c in FINGER_COLOR.items()}


def load_calibration(path):
    """Read hand_calibration.json -> (bone_lengths_mm, fan_angles_deg)."""
    with open(path) as f:
        c = json.load(f)
    bl = {tuple(int(x) for x in k.split('-')): v
          for k, v in c['bone_lengths_mm'].items()}
    return bl, c['palm_template']['fan_angles_deg']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, x, A_hat):
        return self.lin(torch.einsum('ij,bjd->bid', A_hat, x))


class AngleCorrectorV8(nn.Module):
    def __init__(self, ang_dim, node_in, n_ang, A_hat, finger_onehot):
        super().__init__()
        self.register_buffer('A_hat', A_hat)
        self.register_buffer('finger', finger_onehot)
        self.g1, self.g2, self.g3 = (GCNLayer(node_in, 64), GCNLayer(64, 64),
                                     GCNLayer(64, 64))
        self.ang_proj = nn.Sequential(
            nn.Linear(ang_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.graph_proj = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_ang))

    def forward(self, ang_feat, mp_deg, nodes):
        x = torch.cat([nodes, self.finger.unsqueeze(0).expand(nodes.size(0), -1, -1)], -1)
        h = F.relu(self.g1(x, self.A_hat))
        h = F.relu(self.g2(h, self.A_hat))
        h = F.relu(self.g3(h, self.A_hat))
        g = torch.cat([h.mean(1), h.max(1).values], -1)
        z = torch.cat([self.ang_proj(ang_feat), self.graph_proj(g)], 1)
        delta = self.head(z)
        return mp_deg + delta, delta


WRIST_I, MID_MCP, MID_PIP = 0, 9, 10
MIN_BONE_LEN = 5e-4          # metres


class AngleCorrector:
    """Wraps the checkpoint: MediaPipe landmarks -> corrected angles."""

    def __init__(self, ckpt_path, device=None):
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        self.angle_names = list(ck['angle_names'])
        self.n_ang = len(self.angle_names)
        self.mean = np.asarray(ck['ang_mean'], np.float32)
        self.std = np.asarray(ck['ang_std'], np.float32)
        self.lo = np.asarray(ck['ang_lo'], np.float32)
        self.hi = np.asarray(ck['ang_hi'], np.float32)
        self.hand = ck.get('config', {}).get('hand', 'right')

        A_hat = ck['A_hat'] if torch.is_tensor(ck['A_hat']) else torch.tensor(ck['A_hat'])
        fo = ck['finger_onehot']
        fo = fo if torch.is_tensor(fo) else torch.tensor(fo)
        self.model = AngleCorrectorV8(int(ck['ang_feat_dim']), int(ck['node_in']),
                                      self.n_ang, A_hat.float(), fo.float())
        self.model.load_state_dict(ck['state_dict'])
        self.model.to(self.device).eval()

        if self.angle_names != list(ARTICULATION_ANGLES):
            print('WARNING: checkpoint angle order differs from the module; '
                  'using the checkpoint order.')

    def _encode(self, deg):
        a = np.asarray(deg, np.float32).reshape(1, self.n_ang).copy()
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(self.mean, np.where(bad)[1])
        a = np.clip(a, -180.0, 360.0)
        r = np.radians(a)
        return np.concatenate([(a - self.mean) / self.std,
                               np.sin(r), np.cos(r)], 1).astype(np.float32)

    @staticmethod
    def _norm_landmarks(kp):
        kp = np.nan_to_num(np.asarray(kp, np.float32), nan=0., posinf=0., neginf=0.)
        kp = kp - kp[WRIST_I]
        s = float(np.linalg.norm(kp[MID_PIP] - kp[MID_MCP]))
        if not np.isfinite(s) or s < MIN_BONE_LEN:
            return np.zeros((21, 3), np.float32)
        return np.nan_to_num(kp / s, nan=0., posinf=0., neginf=0.).astype(np.float32)

    def __call__(self, kps_world):
        """(21,3) MediaPipe world landmarks -> (mp_angles, corrected_angles, fan)."""
        d = compute_angles_v12(kps_world, hand=self.hand)
        mp = np.array([d[n] for n in self.angle_names], np.float32)
        fan = {n: d[n] for n in FAN_ANGLE_NAMES}

        af = torch.from_numpy(self._encode(mp)).to(self.device)
        nd = torch.from_numpy(self._norm_landmarks(kps_world)).unsqueeze(0).to(self.device)
        mt = torch.from_numpy(mp.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(af, mt, nd)
        return mp, pred.cpu().numpy()[0], fan


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
class HandTracker:
    def __init__(self, model_path, running_mode='IMAGE', num_hands=1, conf=0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self.mp = mp
        mode = getattr(mp_vision.RunningMode, running_mode)
        self.landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mode, num_hands=num_hands,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=conf))
        self.mode = running_mode

    def detect(self, rgb, timestamp_ms=None):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = (self.landmarker.detect(img) if self.mode == 'IMAGE'
               else self.landmarker.detect_for_video(img, int(timestamp_ms)))
        if not res.hand_landmarks:
            return None
        world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[0]], np.float32)
        px = np.array([[l.x, l.y] for l in res.hand_landmarks[0]], np.float32)
        label = res.handedness[0][0].category_name.lower() if res.handedness else 'right'
        return dict(world=world, px=px, hand=label)


# MediaPipe world landmarks use IMAGE axes: +x right, +y DOWN, +z toward the
# camera. The renderers (and ordinary intuition) want +y UP, so a hand held
# fingers-up otherwise reconstructs fingers-down. Rotating 180 deg about x fixes
# it. det = +1, i.e. a PROPER ROTATION -- negating y alone would be a reflection
# and would silently turn a right hand into a left one. Angles are invariant
# under rotation, so this changes the view only, never the model input.
MP_TO_YUP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def to_view_frame(kps):
    """MediaPipe's y-down world frame -> conventional y-up frame."""
    return np.asarray(kps, np.float32) @ MP_TO_YUP.T


def to_right_hand(world, label, target='right'):
    """The model was trained on right hands. Mirror a left hand across the palm's
    lateral axis so its angles land in the same convention. Returns (kps, mirrored)."""
    if label.startswith(target[0]):
        return world, False
    w = world.copy()
    w[:, 0] *= -1.0
    return w, True


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
def reconstruct(angles_deg, angle_names, fan_deg, bone_mm, palm=None):
    """angles -> (21,3) keypoints in mm with the wrist at the origin."""
    a = {n: float(v) for n, v in zip(angle_names, angles_deg)}
    for n in ARTICULATION_ANGLES:
        a.setdefault(n, 0.0)
    return forward_kinematics_v12(a, bone_mm, palm=palm, fan_angles=fan_deg)


# ---------------------------------------------------------------------------
# Fast 3D -> 2D renderer (OpenCV; used by the live script)
# ---------------------------------------------------------------------------
class Renderer3D:
    """Orthographic projection with yaw/pitch, drawn with cv2 primitives.
    Far faster than matplotlib, which is why the live script uses it."""

    def __init__(self, size=520, scale=1.7, yaw=25.0, pitch=-20.0):
        self.size, self.scale = size, scale
        self.yaw, self.pitch = yaw, pitch

    def _R(self):
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        return Rx @ Ry

    def project(self, pts_mm):
        q = np.asarray(pts_mm, float) @ self._R().T
        c = self.size / 2
        u = c + q[:, 0] * self.scale
        v = c - q[:, 1] * self.scale + self.size * 0.22
        return np.stack([u, v], 1), q[:, 2]

    def draw(self, skeletons, labels=None, title=None):
        import cv2
        img = np.full((self.size, self.size, 3), 24, np.uint8)
        for gx in range(0, self.size, 52):
            cv2.line(img, (gx, 0), (gx, self.size), (38, 38, 38), 1)
            cv2.line(img, (0, gx), (self.size, gx), (38, 38, 38), 1)

        for si, (pts, style) in enumerate(skeletons):
            uv, depth = self.project(pts)
            order = np.argsort([-(depth[a] + depth[b]) / 2 for a, b in BONES])
            for bi in order:
                a, b = BONES[bi]
                col = (FINGER_BGR[FINGER_OF[b]] if style == 'color' else (110, 110, 110))
                th = 4 if style == 'color' else 2
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         col, th, cv2.LINE_AA)
            if style == 'color':
                for j in range(21):
                    cv2.circle(img, tuple(uv[j].astype(int)), 4,
                               FINGER_BGR[FINGER_OF[j]], -1, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[0].astype(int)), 7, (255, 255, 255), 2, cv2.LINE_AA)

        if title:
            cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (230, 230, 230), 1, cv2.LINE_AA)
        if labels:
            for i, (txt, col) in enumerate(labels):
                cv2.putText(img, txt, (12, self.size - 14 - 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return img


def draw_landmarks_2d(frame, px, color=(90, 230, 200)):
    import cv2
    h, w = frame.shape[:2]
    p = (px * np.array([w, h])).astype(int)
    for a, b in BONES:
        cv2.line(frame, tuple(p[a]), tuple(p[b]), FINGER_BGR[FINGER_OF[b]], 2, cv2.LINE_AA)
    for j in range(21):
        cv2.circle(frame, tuple(p[j]), 3, color, -1, cv2.LINE_AA)
    return frame