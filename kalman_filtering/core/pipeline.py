"""
pipeline.py
===========
Everything the runners share: MediaPipe, the v12 corrector, per-hand
filters, rendering, recording, and the per-frame CSV. The runners differ ONLY
in `mode`, so a benchmark difference can only come from the method:

    mode      block-1 input          filter            3D output
    ------    --------------------   ---------------   -------------------------
    raw       MediaPipe angles       none              MediaPipe world landmarks
    kf_mp     MediaPipe angles       EKF (2 blocks)    FK of filtered angles
    ekf       v12-CORRECTED angles   EKF (2 blocks)    FK of filtered angles
    angles    corrected (default)    angle-only KF     FK of filtered angles
              or MediaPipe           (angle_kf.py)     (display only)

kf_mp and ekf use the identical filter, tuning and calibration; the only
difference is whether the corrector sits in front of block 1. See Flow.md.

Coordinates: all maths runs in MediaPipe's world frame (metres -> mm), left
hands mirrored into right chirality (the checkpoint is right-handed). The y-up
view rotation is applied only when drawing.

CSV (one row per frame, every mode):
    frame, time_s, detected, hand, mirrored,
    mp_<angle>  x21        raw MediaPipe angles
    out_<angle> x21        this mode's output angles
    kpmp_<j>_<x|y|z> x21   MediaPipe keypoints, wrist-centred, mm
    kpout_<j>_<x|y|z> x21  this mode's 3D output, wrist-centred, mm
  filter modes add: corr_<angle> (when the corrector runs), std_<angle>, nis1, nis2, resid_mm,
                    scale, coasting, block2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

CORE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE)                      # kalman_filtering/
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import hand_angles_v2 as HA  # noqa: E402
from angle_kf import AngleKF  # noqa: E402
from hand_ekf import (EKFConfig, HandEKF, MediaPipeSkeletonEstimator,  # noqa: E402
                      PersonalizedHandModel, load_angle_sigma_csv)

ANGLES = HA.ARTICULATION_ANGLES
BONES = HA.BONES
DEFAULT_CKPT = os.path.join(ROOT, "models", "angle_corrector_v12.pth")
DEFAULT_MP_MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")
DEFAULT_CALIB = {"right": os.path.join(ROOT, "calib", "hand_calibration_ekf_right.json"),
                 "left": os.path.join(ROOT, "calib", "hand_calibration_ekf_left.json")}
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

MODE_TITLES = {"raw": "RAW MediaPipe (no filter)",
               "kf_mp": "KF on MediaPipe angles + 3D",
               "ekf": "v12 corrector + EKF",
               "angles": "Angle-only jerk-min KF"}

FINGER_COLOR = {"Thumb": "#FF6B6B", "Index": "#4ECDC4", "Middle": "#45B7D1",
                "Ring": "#96CEB4", "Pinky": "#FFEAA7"}
FINGER_OF = {j: f for f, ch in HA.FINGER_CHAINS.items() for j in ch}
FINGER_OF[0] = "Middle"
FINGER_BGR = {f: tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))
              for f, c in FINGER_COLOR.items()}

# MediaPipe world axes are image axes (+y DOWN, +z toward camera). Rotating
# 180 deg about x gives y-up for drawing. det=+1: a rotation, not a reflection.
MP_TO_YUP = np.diag([1.0, -1.0, -1.0])
MIRROR_X = np.array([-1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# MediaPipe
# --------------------------------------------------------------------------
class HandTracker:
    def __init__(self, model_path, num_hands=1, conf=0.3):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if not os.path.exists(model_path):
            import urllib.request
            print(f"downloading HandLandmarker -> {model_path}")
            urllib.request.urlretrieve(MODEL_URL, model_path)
        self.mp = mp
        self.landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO, num_hands=num_hands,
                min_hand_detection_confidence=conf,
                min_hand_presence_confidence=conf,
                min_tracking_confidence=conf))
        self._last_ts = -1

    def detect(self, rgb, timestamp_ms):
        ts = max(int(timestamp_ms), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(img, ts)
        if not res.hand_world_landmarks:
            return None
        world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], float)
        px = np.array([[p.x, p.y] for p in res.hand_landmarks[0]], float)
        label, score = "right", 0.0
        if res.handedness:
            label = res.handedness[0][0].category_name.lower()
            score = float(res.handedness[0][0].score)
        return dict(world=world, px=px, hand=label, score=score)


class Packet:
    """One captured frame and its MediaPipe result."""
    __slots__ = ("index", "frame", "det", "t_s", "det_ms")

    def __init__(self, index, frame, det, t_s, det_ms):
        self.index, self.frame, self.det, self.t_s, self.det_ms = index, frame, det, t_s, det_ms


class DetectionStream:
    """Capture + MediaPipe, optionally on a background thread.

    Sequentially, every frame waits for MediaPipe (~36 ms) and then for the
    corrector, filter and drawing (~15 ms). Threaded, MediaPipe works on frame
    k+1 while the main loop handles frame k, so the frame rate is set by the
    slower stage instead of the sum. MediaPipe and OpenCV release the GIL
    while they compute, so a thread is enough (no process needed).

    * Video file: a small blocking queue, so no frame is ever dropped and the
      output is identical to the sequential run (t = index / fps).
    * Webcam: a queue of ONE that keeps only the newest result, so latency
      never builds up. t is the capture time, so dt is the real time between
      the frames the filter sees, including any frame that was skipped.
    """

    def __init__(self, cap, tracker, *, is_cam, mirror, src_fps, max_frames=0,
                 threaded=True):
        import queue
        import threading
        self.cap, self.tracker = cap, tracker
        self.is_cam, self.mirror, self.src_fps = is_cam, mirror, src_fps
        self.max_frames = max_frames
        self.threaded = threaded
        self.dropped = 0
        self._i = 0
        self._t0 = None
        self._done = False
        if threaded:
            self._q = queue.Queue(maxsize=1 if is_cam else 4)
            self._stop = threading.Event()
            self._th = threading.Thread(target=self._worker, name="mediapipe", daemon=True)
            self._th.start()

    def _produce(self):
        """Read and detect one frame. None at the end of the source."""
        import cv2
        if self.max_frames and self._i >= self.max_frames:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        if self.mirror:
            frame = cv2.flip(frame, 1)
        t_s = (now - self._t0) if self.is_cam else self._i / self.src_fps
        t1 = time.perf_counter()
        det = self.tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t_s * 1000.0)
        pkt = Packet(self._i, frame, det, t_s, 1e3 * (time.perf_counter() - t1))
        self._i += 1
        return pkt

    def _worker(self):
        import queue
        try:
            while not self._stop.is_set():
                pkt = self._produce()
                if pkt is None:
                    break
                if self.is_cam:                       # keep only the newest
                    try:
                        self._q.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    self._q.put(pkt)
                else:                                 # never drop a video frame
                    while not self._stop.is_set():
                        try:
                            self._q.put(pkt, timeout=0.1)
                            break
                        except queue.Full:
                            continue
        except Exception as exc:                      # surface it in the main thread
            self._error = exc
        finally:
            self._done = True

    def get(self):
        """Next packet, or None when the source has ended."""
        if not self.threaded:
            return self._produce()
        import queue
        while True:
            try:
                return self._q.get(timeout=0.05)
            except queue.Empty:
                if self._done and self._q.empty():
                    err = getattr(self, "_error", None)
                    if err is not None:
                        raise err
                    return None

    def close(self):
        if self.threaded:
            self._stop.set()
            self._th.join(timeout=2.0)


# --------------------------------------------------------------------------
# v12 corrector
# --------------------------------------------------------------------------
def build_corrector_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GCNLayer(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.lin = nn.Linear(din, dout)

        def forward(self, x, A_hat):
            return self.lin(torch.einsum("ij,bjd->bid", A_hat, x))

    class AngleCorrectorV8(nn.Module):
        def __init__(self, ang_dim, node_in, n_ang, A_hat, finger_onehot):
            super().__init__()
            self.register_buffer("A_hat", A_hat)
            self.register_buffer("finger", finger_onehot)
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

    return AngleCorrectorV8


class AngleCorrector:
    """(21,3) MediaPipe world landmarks, RIGHT chirality, MediaPipe axes, in
    metres -> 21 corrected angles. The GCN branch sees raw coordinates (it is
    not rotation-invariant), so it is fed MediaPipe's own frame -- the frame
    the training NPZ was cached in -- not the y-up view frame."""

    def __init__(self, ckpt_path, device=None):
        import torch
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        names = list(ck["angle_names"])
        if names != ANGLES:
            raise ValueError("checkpoint angle_names differ from hand_angles_v2."
                             f"ARTICULATION_ANGLES:\n{names}\n{ANGLES}")
        if list(ck.get("fan_angle_names", HA.FAN_ANGLE_NAMES)) != HA.FAN_ANGLE_NAMES:
            raise ValueError("checkpoint fan_angle_names differ from hand_angles_v2")
        self.hand = ck.get("config", {}).get("hand", "right")
        self.mean = np.asarray(ck["ang_mean"], np.float32)
        self.std = np.asarray(ck["ang_std"], np.float32)
        A_hat = torch.as_tensor(ck["A_hat"]).float()
        fo = torch.as_tensor(ck["finger_onehot"]).float()
        Model = build_corrector_class()
        self.model = Model(int(ck["ang_feat_dim"]), int(ck["node_in"]), len(names), A_hat, fo)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()

    def _encode(self, deg):
        a = np.asarray(deg, np.float32).reshape(1, -1).copy()
        bad = ~np.isfinite(a)
        if bad.any():
            a[bad] = np.take(self.mean, np.where(bad)[1])
        a = np.clip(a, -180.0, 360.0)
        r = np.radians(a)
        return np.concatenate([(a - self.mean) / self.std, np.sin(r), np.cos(r)], 1)

    @staticmethod
    def _nodes(kp):
        kp = np.nan_to_num(np.asarray(kp, np.float32))
        kp = kp - kp[0]
        s = float(np.linalg.norm(kp[10] - kp[9]))          # Middle MCP->PIP. NOT 11.
        if not np.isfinite(s) or s < 5e-4:
            return np.zeros((21, 3), np.float32)
        return (kp / s).astype(np.float32)

    def __call__(self, kps_world_m, mp_angles):
        t = self.torch
        af = t.from_numpy(self._encode(mp_angles).astype(np.float32)).to(self.device)
        nd = t.from_numpy(self._nodes(kps_world_m)).unsqueeze(0).to(self.device)
        mt = t.from_numpy(np.asarray(mp_angles, np.float32).reshape(1, -1)).to(self.device)
        with t.no_grad():
            pred, _ = self.model(af, mt, nd)
        return pred.cpu().numpy()[0].astype(float)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Renderer3D:
    """Orthographic cv2 skeleton renderer. Input: mm, y-up, wrist-centred."""

    def __init__(self, size=480, scale=1.6, yaw=25.0, pitch=-20.0):
        self.size, self.scale, self.yaw, self.pitch = size, scale, yaw, pitch

    def _R(self, yaw):
        y, p = np.radians(yaw), np.radians(self.pitch)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        return Rx @ Ry

    def draw(self, skeletons, title="", lines=(), yaw_offset=0.0):
        import cv2
        S = self.size
        if getattr(self, "_bg", None) is None or self._bg.shape[0] != S:
            self._bg = np.full((S, S, 3), 24, np.uint8)
            for g in range(0, S, 48):
                cv2.line(self._bg, (g, 0), (g, S), (38, 38, 38), 1)
                cv2.line(self._bg, (0, g), (S, g), (38, 38, 38), 1)
        img = self._bg.copy()
        R = self._R(self.yaw + yaw_offset)
        for pts, style in skeletons:
            if pts is None:
                continue
            q = np.asarray(pts, float) @ R.T
            uv = np.stack([S / 2 + q[:, 0] * self.scale,
                           S / 2 - q[:, 1] * self.scale + S * 0.22], 1).astype(int)
            order = np.argsort([-(q[a, 2] + q[b, 2]) for a, b in BONES])
            for bi in order:
                a, b = BONES[bi]
                col = FINGER_BGR[FINGER_OF[b]] if style == "color" else (110, 110, 110)
                cv2.line(img, tuple(uv[a]), tuple(uv[b]), col,
                         4 if style == "color" else 2, cv2.LINE_AA)
            if style == "color":
                for j in range(21):
                    cv2.circle(img, tuple(uv[j]), 4, FINGER_BGR[FINGER_OF[j]], -1, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[0]), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (230, 230, 230), 1, cv2.LINE_AA)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (10, S - 12 - 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, col, 1, cv2.LINE_AA)
        return img


def draw_landmarks_2d(frame, px):
    import cv2
    h, w = frame.shape[:2]
    p = (np.asarray(px) * [w, h]).astype(int)
    for a, b in BONES:
        cv2.line(frame, tuple(p[a]), tuple(p[b]), FINGER_BGR[FINGER_OF[b]], 2, cv2.LINE_AA)
    for j in range(21):
        cv2.circle(frame, tuple(p[j]), 3, (90, 230, 200), -1, cv2.LINE_AA)


def to_view(kps_mm):
    return None if kps_mm is None else np.asarray(kps_mm, float) @ MP_TO_YUP.T


def resolve_hand(label, image_mirrored, override="auto"):
    """-> (geometric, physical) handedness, each 'right' or 'left'.

    geometric: the chirality of the landmarks as MediaPipe returned them. Decides
               whether x is mirrored before the angles and the corrector.
    physical : the hand actually held up. Decides which calibration and filter.
    MediaPipe's label follows the image as given (checked on an unmirrored
    clip: right hand -> "right"), so flipping the image for a selfie view
    swaps label and physical hand. `override` names the PHYSICAL hand."""
    flip = {"right": "left", "left": "right"}
    if override != "auto":
        physical = override
        geometric = flip[physical] if image_mirrored else physical
    else:
        geometric = "left" if str(label).lower().startswith("l") else "right"
        physical = flip[geometric] if image_mirrored else geometric
    return geometric, physical


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def uses_corrector(mode, args):
    return mode == "ekf" or (mode == "angles" and args.angle_source == "corrected")


def csv_header(mode, has_corr):
    cols = ["frame", "time_s", "detected", "hand", "mirrored"]
    cols += [f"mp_{n}" for n in ANGLES] + [f"out_{n}" for n in ANGLES]
    cols += [f"kpmp_{j}_{a}" for j in range(21) for a in "xyz"]
    cols += [f"kpout_{j}_{a}" for j in range(21) for a in "xyz"]
    if mode != "raw":
        if has_corr:
            cols += [f"corr_{n}" for n in ANGLES]
        cols += [f"std_{n}" for n in ANGLES]
        cols += ["nis1", "nis2", "resid_mm", "scale", "coasting", "block2"]
    return cols


def _vals(v, n):
    return [""] * n if v is None else [f"{float(x):.4f}" for x in np.ravel(v)]


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser(mode):
    ap = argparse.ArgumentParser(description=MODE_TITLES[mode],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="webcam index (0, 1, ...) or a video file path")
    ap.add_argument("--mp-model", default=DEFAULT_MP_MODEL)
    ap.add_argument("--out-dir", default=os.path.join(OUTPUT_DIR, mode))
    ap.add_argument("--tag", default=None, help="filename stem for outputs "
                    "(default: <source>_<timestamp>)")
    ap.add_argument("--no-video", action="store_true", help="do not write the mp4")
    ap.add_argument("--no-csv", action="store_true", help="do not write the csv")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: no window (video files only)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-thread", action="store_true",
                    help="run MediaPipe in the main loop (the old, slower behaviour)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--panel", type=int, default=480)
    mirror = ap.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="mirror", action="store_true",
                        help="flip the image horizontally (default ON for webcam)")
    mirror.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.set_defaults(mirror=None)
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto",
                    help="the PHYSICAL hand you hold up (overrides MediaPipe's "
                         "label). auto = trust MediaPipe, corrected for mirroring")
    ap.add_argument("--unity", nargs="?", const="127.0.0.1:5065", default=None,
                    metavar="HOST:PORT", help="stream the filtered hand to Unity over UDP "
                    "(default 127.0.0.1:5065); see unity/README.md")
    if mode == "raw":
        return ap

    g = ap.add_argument_group("filter")
    d = EKFConfig()
    g.add_argument("--calib-right", default=DEFAULT_CALIB["right"],
                   help="calibration (angle-only mode: used for drawing only)")
    g.add_argument("--calib-left", default=DEFAULT_CALIB["left"])
    g.add_argument("--jerk-psd", type=float, default=d.jerk_psd,
                   help="deg^2/s^5; smaller = smoother but laggier")
    g.add_argument("--sigma-angle", type=float, default=d.sigma_angle,
                   help="block-1 angle noise, deg")
    g.add_argument("--angle-noise-csv", default=None,
                   help="per-angle sigma from test_per_angle_improvement.csv "
                        "(overrides --sigma-angle)")
    g.add_argument("--r-scale-angles", type=float, default=d.r_scale_angles)
    g.add_argument("--max-coast", type=int, default=d.max_coast_frames,
                   help="frames of prediction-only before a track reset")
    g.add_argument("--coast-decay", type=float, default=d.coast_decay,
                   help="velocity/acceleration multiplier per coasting frame (1 = keep extrapolating)")
    adapt = g.add_mutually_exclusive_group()
    adapt.add_argument("--adaptive", dest="adaptive", action="store_true",
                       help="smooth harder when the fingers are slow and less when "
                            "they move fast (per angle)")
    adapt.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    g.set_defaults(adaptive=d.adaptive)
    g.add_argument("--adapt-ref", type=float, default=d.adapt_ref_speed,
                   help="deg/s at which adaptive smoothing is neutral; lower = "
                        "treats more motion as 'fast'")
    rob = g.add_mutually_exclusive_group()
    rob.add_argument("--robust", dest="robust", action="store_true",
                     help="down-weight measurements that jump further from the filter's "
                          "prediction than --robust-k sigmas")
    rob.add_argument("--no-robust", dest="robust", action="store_false")
    g.set_defaults(robust=d.robust)
    g.add_argument("--robust-k", type=float, default=d.robust_k,
                   help="sigmas of innovation before an angle is down-weighted")
    g.add_argument("--robust-k-kp", type=float, default=d.robust_k_kp,
                   help="same for a block-2 keypoint")
    g.add_argument("--robust-power", type=float, default=d.robust_power,
                   help="R is inflated by (d/k)^power; 1 = Huber")
    g.add_argument("--root-tau", type=float, default=d.root_tau_s,
                   help="seconds of smoothing on the drawn hand's overall "
                        "orientation (display only; 0 = raw per-frame fit). "
                        "Raise it if the whole hand shakes while the fingers look fine")
    if mode == "angles":
        g.add_argument("--angle-source", choices=["corrected", "mediapipe"],
                       default="corrected",
                       help="what the filter smooths: v12-corrected or raw MediaPipe angles")
        return ap

    g.add_argument("--sigma-xy", type=float, default=d.sigma_xy, help="mm")
    g.add_argument("--sigma-z", type=float, default=d.sigma_z, help="mm")
    g.add_argument("--r-scale-kps", type=float, default=d.r_scale_kps)
    g.add_argument("--kp-model", choices=["calib", "mediapipe"], default="calib",
                   help="skeleton block 2 compares MediaPipe keypoints against: "
                        "your calibration, or one estimated from MediaPipe's own "
                        "first frames (smaller residual floor, see README)")
    g.add_argument("--kp-model-frames", type=int, default=15)
    g.add_argument("--no-block1", action="store_true", help="ablation")
    g.add_argument("--no-block2", action="store_true", help="ablation (also key 'k')")
    g.add_argument("--no-align-noise", action="store_true",
                   help="ablation: plain diagonal R2 (not recommended)")
    return ap


def make_ekf_config(args, mode):
    d = EKFConfig()
    sigma = args.sigma_angle
    if args.angle_noise_csv:
        sigma = load_angle_sigma_csv(args.angle_noise_csv,
                                     "Corr" if uses_corrector(mode, args) else "MP")
    opt = lambda name, default: getattr(args, name, default)   # noqa: E731
    return EKFConfig(jerk_psd=args.jerk_psd, sigma_angle=sigma,
                     sigma_xy=opt("sigma_xy", d.sigma_xy), sigma_z=opt("sigma_z", d.sigma_z),
                     r_scale_angles=args.r_scale_angles,
                     r_scale_kps=opt("r_scale_kps", d.r_scale_kps),
                     use_block1=not opt("no_block1", False),
                     use_block2=mode != "angles" and not opt("no_block2", False),
                     align_noise=not opt("no_align_noise", False),
                     adaptive=args.adaptive, adapt_ref_speed=args.adapt_ref,
                     root_tau_s=args.root_tau,
                     robust=args.robust, robust_k=args.robust_k,
                     robust_k_kp=args.robust_k_kp, robust_power=args.robust_power,
                     max_coast_frames=args.max_coast, coast_decay=args.coast_decay)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run(mode, argv=None):
    import cv2

    ap = build_parser(mode)
    args = ap.parse_args(argv)
    is_cam = args.source.isdigit()
    mirror = is_cam if args.mirror is None else args.mirror
    if args.no_display and is_cam:
        ap.error("--no-display needs a video file; a webcam run has no end")

    print(f"=== {MODE_TITLES[mode]} ===")
    print("[hand_ekf] using hand_angles_v2.compute_angles "
          f"({HA.N_ANGLES} articulation angles, v12)")

    tracker = HandTracker(args.mp_model)
    corrector = None
    if mode != "raw" and uses_corrector(mode, args):
        corrector = AngleCorrector(DEFAULT_CKPT)
        print(f"corrector: {os.path.basename(DEFAULT_CKPT)} on {corrector.device}, "
              f"trained on {corrector.hand} hands")

    filters, cfg, display_models, estimators = {}, None, {}, {}
    if mode != "raw":
        cfg = make_ekf_config(args, mode)
        models = {}
        for h, path in (("right", args.calib_right), ("left", args.calib_left)):
            if path and os.path.exists(path):
                models[h] = PersonalizedHandModel.from_calibration(path)
                print(f"calibration {h}: {os.path.basename(path)} "
                      f"(hand length {models[h].hand_length_mm:.1f} mm)")
        if not models:
            raise SystemExit("no calibration file found; pass --calib-right")
        kp_model = getattr(args, "kp_model", "calib")
        for h in ("right", "left"):
            display_models[h] = models.get(h) or next(iter(models.values()))
            if mode == "angles":
                filters[h] = AngleKF(display_models[h], cfg)
            else:
                filters[h] = HandEKF(display_models[h], cfg)
            if kp_model == "mediapipe":
                estimators[h] = MediaPipeSkeletonEstimator(args.kp_model_frames)
        sigma_txt = 'per-angle csv' if args.angle_noise_csv else cfg.sigma_angle
        if mode == "angles":
            print(f"angle-only KF on {args.angle_source} angles: jerk_psd={cfg.jerk_psd:g}  "
                  f"sigma_angle={sigma_txt}  r_scale={cfg.r_scale_angles}  "
                  "(21 independent 3-state filters; keypoints used for drawing only)")
            print("NIS target: nis1 ~ 21. Far below = R too large, far above = R too small.")
        else:
            print(f"block-2 skeleton: {kp_model}"
                  + (f" (estimated from the first {args.kp_model_frames} detections;"
                     " calibration until then and for display)"
                     if kp_model == "mediapipe" else ""))
            print(f"EKF: jerk_psd={cfg.jerk_psd:g}  sigma_angle={sigma_txt}  "
                  f"sigma_xy={cfg.sigma_xy}  sigma_z={cfg.sigma_z}  "
                  f"r_scale={cfg.r_scale_angles}/{cfg.r_scale_kps}  "
                  f"block1={cfg.use_block1} block2={cfg.use_block2}  "
                  + (f"robust k={cfg.robust_k}/{cfg.robust_k_kp} p={cfg.robust_power}"
                     if cfg.robust else "robust off"))
            print("NIS targets: nis1 ~ 21, nis2 ~ 45. Far below = R too large, "
                  "far above = R too small.")

    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not is_cam and not (1.0 < src_fps < 1000.0):
        src_fps = 30.0
    stream = DetectionStream(cap, tracker, is_cam=is_cam, mirror=mirror, src_fps=src_fps,
                             max_frames=args.max_frames, threaded=not args.no_thread)
    print(f"MediaPipe: {'main loop' if args.no_thread else 'background thread'}")

    stem = "webcam" if is_cam else os.path.splitext(os.path.basename(args.source))[0]
    tag = args.tag or f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, tag + ".mp4")
    csv_path = os.path.join(args.out_dir, tag + ".csv")

    csv_fh = writer_csv = None
    if not args.no_csv:
        csv_fh = open(csv_path, "w", newline="")
        writer_csv = csv.writer(csv_fh)
        writer_csv.writerow(csv_header(mode, corrector is not None))

    rend = Renderer3D(size=args.panel)
    video = None
    rec_buf, WARMUP = [], 40               # webcam: measure real fps before opening writer
    show_ghost, paused = False, False
    fps_disp, t_prev = 0.0, time.perf_counter()
    last_t_s = None                        # capture time of the previous packet
    last_hand, last_mirrored, last_uv = "right", False, None
    unity = None
    if getattr(args, "unity", None):
        from unity_stream import UnityStream
        unity = UnityStream(args.unity)
        print(f"streaming the hand to Unity at {unity.addr[0]}:{unity.addr[1]} (UDP)")
    frame_i, n_det = 0, 0
    last_step = {}
    t_start = time.perf_counter()

    try:
        while True:
            if paused and not args.no_display:
                k = cv2.waitKey(30) & 0xFF
                if k == ord(" "):
                    paused = False
                    last_t_s = None                # do not predict across the pause
                elif k == ord("q"):
                    break
                continue

            pkt = stream.get()
            if pkt is None:
                break
            frame, det, t_s = pkt.frame, pkt.det, pkt.t_s
            if is_cam:
                dt = 1.0 / 30.0 if last_t_s is None else max(t_s - last_t_s, 1e-3)
            else:
                dt = 1.0 / src_fps
            last_t_s = t_s
            now = time.perf_counter()
            fps_disp = 0.9 * fps_disp + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            # while coasting keep the last frame's chirality, or a left hand
            # would be drawn un-mirrored (a visible flip) during every dropout
            hand, mirrored = last_hand, last_mirrored
            kps_m = mp_ang = corr_ang = out_ang = None
            kpmp_c = kpout_c = None
            if det is not None:
                n_det += 1
                geometric, hand = resolve_hand(det["hand"], mirror, args.hand)
                last_hand = hand                     # physical: picks calibration + filter
                mirrored = geometric == "left"       # geometric: picks the x flip
                last_mirrored = mirrored
                last_uv = det["px"][0]
                kps_m = det["world"] * (MIRROR_X if mirrored else 1.0)   # right chirality
                mp_ang = HA.compute_angles(kps_m, hand="right")
                kpmp_c = (kps_m - kps_m[0]) * 1000.0
                if corrector is not None:
                    corr_ang = corrector(kps_m, mp_ang)

            ekf = filters.get("left" if last_hand.startswith("l") else "right")
            if ekf is not None:
                # a filter left idle while the other hand was tracked is stale
                if ekf.initialised and frame_i - last_step.get(id(ekf), frame_i) \
                        > ekf.cfg.max_coast_frames:
                    ekf.reset()
                last_step[id(ekf)] = frame_i
            if mode == "raw":
                out_ang, kpout_c = mp_ang, kpmp_c
            else:
                hkey = "left" if last_hand.startswith("l") else "right"
                if kps_m is not None and hkey in estimators:
                    est_model = estimators[hkey].add(kps_m * 1000.0)
                    if est_model is not None:
                        ekf.set_measurement_model(est_model)
                        print(f"block-2 skeleton for {hkey} hand estimated from MediaPipe")
                z1 = corr_ang if corrector is not None else mp_ang
                est = ekf.step(z1, None if kps_m is None else kps_m * 1000.0, dt)
                if est is not None:
                    out_ang = est
                    kpout_c = ekf.skeleton(oriented=True, model=display_models[hkey])

            # ---------------- csv ----------------
            if writer_csv is not None:
                row = [frame_i, f"{t_s:.4f}", int(det is not None),
                       hand if det is not None else "", int(mirrored)]
                row += _vals(mp_ang, 21) + _vals(out_ang, 21)
                row += _vals(kpmp_c, 63) + _vals(kpout_c, 63)
                if mode != "raw":
                    if corrector is not None:
                        row += _vals(corr_ang, 21)
                    row += _vals(ekf.angle_std if out_ang is not None else None, 21)
                    row += [f"{v:.3f}" if np.isfinite(v) else "" for v in
                            (ekf.last_nis1, ekf.last_nis2, ekf.last_resid_mm)]
                    row += [f"{ekf.scale:.4f}" if ekf.scale else "",
                            int(ekf.initialised and det is None), int(ekf.cfg.use_block2)]
                writer_csv.writerow(row)

            # ---------------- unity ----------------
            if unity is not None:
                shown = None if kpout_c is None else (kpout_c * MIRROR_X if mirrored else kpout_c)
                unity.send(t_s, hand, det is not None, shown, out_ang, last_uv)

            # ---------------- draw ----------------
            if det is not None:
                draw_landmarks_2d(frame, det["px"])

            def display(k):          # back to the physical hand's chirality, y-up
                if k is None:
                    return None
                return to_view(k * MIRROR_X if mirrored else k)

            skels = []
            if show_ghost and mode != "raw":
                skels.append((display(kpmp_c), "grey"))
            skels.append((display(kpout_c), "color"))

            info = [(f"hand {hand}{' (mirrored->right)' if mirrored else ''}  "
                     f"{'detected' if det is not None else 'NO DETECTION'}",
                     (200, 200, 200) if det is not None else (90, 90, 230))]
            if mode != "raw" and ekf is not None and ekf.initialised:
                std = float(ekf.angle_std.mean())
                if mode == "angles":
                    info.insert(0, (f"nis1 {ekf.last_nis1:6.1f}  (target 21)", (170, 220, 170)))
                    info.insert(0, (f"mean sigma {std:4.1f} deg  angle-only"
                                    f"{'  COASTING' if det is None else ''}", (170, 220, 170)))
                else:
                    info.insert(0, (f"nis1 {ekf.last_nis1:6.1f}  nis2 {ekf.last_nis2:6.1f}  "
                                    f"resid {ekf.last_resid_mm:5.1f}mm", (170, 220, 170)))
                    info.insert(0, (f"mean sigma {std:4.1f} deg  block2 "
                                    f"{'ON' if ekf.cfg.use_block2 else 'OFF'}"
                                    f"{'  COASTING' if det is None else ''}", (170, 220, 170)))
            if corr_ang is not None:
                info.insert(0, (f"mean |correction| "
                                f"{np.mean(np.abs(HA.wrap_deg(corr_ang - mp_ang))):.1f} deg",
                                (200, 200, 200)))
            title = MODE_TITLES[mode]
            front = rend.draw(skels, title + "  (front)", info)
            side = rend.draw(skels, "side view", yaw_offset=65.0)

            cv2.putText(frame, f"{fps_disp:4.1f} fps", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            P = args.panel
            cam = cv2.resize(frame, (int(frame.shape[1] * P / frame.shape[0]), P))
            combo = np.hstack([cam, front, side])

            # ---------------- record ----------------
            if not args.no_video:
                if video is None:
                    if is_cam and len(rec_buf) < WARMUP:
                        rec_buf.append(combo)
                    else:
                        rate = float(np.clip(fps_disp, 5, 60)) if is_cam else src_fps
                        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                rate, (combo.shape[1], combo.shape[0]))
                        for b in rec_buf:
                            video.write(b)
                        rec_buf.clear()
                        video.write(combo)
                else:
                    video.write(combo)

            frame_i += 1
            if not is_cam and frame_i % 100 == 0:
                print(f"  frame {frame_i}  detected {n_det}")

            if args.no_display:
                continue
            cv2.imshow(f"hand filtering - {mode}", combo)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                paused = True
            elif k == ord("m"):
                show_ghost = not show_ghost
            elif k == ord("a"):
                rend.yaw -= 8
            elif k == ord("d"):
                rend.yaw += 8
            elif k == ord("k") and cfg is not None and mode != "angles":
                cfg.use_block2 = not cfg.use_block2        # shared by both filters
                print(f"block 2 {'ON' if cfg.use_block2 else 'OFF'}")
            elif k == ord("r") and filters:
                for f in filters.values():
                    f.reset()
                print("filters reset")
    finally:
        if video is None and rec_buf and not args.no_video:
            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    float(np.clip(fps_disp, 5, 60)),
                                    (rec_buf[0].shape[1], rec_buf[0].shape[0]))
            for b in rec_buf:
                video.write(b)
        if video is not None:
            video.release()
            print(f"wrote {video_path}")
        if csv_fh is not None:
            csv_fh.close()
            print(f"wrote {csv_path}")
        stream.close()
        if unity is not None:
            unity.close()
            print(f"sent {unity.sent} frames to Unity")
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"hand detected in {n_det}/{frame_i} frames"
              f" ({100.0 * n_det / max(frame_i, 1):.1f}%)"
              + (f", {stream.dropped} stale camera frames skipped" if stream.dropped else ""))
        el = time.perf_counter() - t_start
        print(f"processed {frame_i} frames in {el:.1f} s ({frame_i / max(el, 1e-9):.1f} fps)")
    return csv_path if not args.no_csv else None
