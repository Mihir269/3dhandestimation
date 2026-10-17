#!/usr/bin/env python3
"""
test_mediapipe_interhand.py
===========================
Evaluate MediaPipe Hands joint-angle predictions against InterHand2.6M
ground truth on a small subset of the dataset.

Pipeline
--------
1. Load InterHand2.6M annotations (camera, 3D joints, data) and the image
   folder you have on disk -- only images that exist locally are used, so
   you can work with a tiny subset.
2. For every available image, run MediaPipe Hands.
3. Compute joint angles from MediaPipe's 21 landmarks AND from the GT 3D
   joints (after remapping joint indices to a common convention).
4. Aggregate error metrics (MAE, RMSE, per-angle and overall) into CSVs.
5. Save side-by-side visualizations (RGB + MP overlay + GT overlay) for
   N random samples, plus a per-sample bar chart of MP vs GT angles.

Why angles? They are scale-invariant, so the unit mismatch between MP
(meters, wrist-relative) and InterHand2.6M (mm, world / camera frame)
cancels out -- no Procrustes alignment needed.

Run
---
    python test_mediapipe_interhand.py \
        --root  /path/to/InterHand2.6M_5fps_batch1 \
        --split test \
        --out   ./eval_out \
        --num-vis 8 \
        --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# JOINT INDEX CONSTANTS
# =============================================================================
# MediaPipe ordering -- 21 landmarks per hand.
MP_WRIST = 0
MP_THUMB_CMC, MP_THUMB_MCP, MP_THUMB_IP, MP_THUMB_TIP = 1, 2, 3, 4
MP_INDEX_MCP, MP_INDEX_PIP, MP_INDEX_DIP, MP_INDEX_TIP = 5, 6, 7, 8
MP_MIDDLE_MCP, MP_MIDDLE_PIP, MP_MIDDLE_DIP, MP_MIDDLE_TIP = 9, 10, 11, 12
MP_RING_MCP, MP_RING_PIP, MP_RING_DIP, MP_RING_TIP = 13, 14, 15, 16
MP_PINKY_MCP, MP_PINKY_PIP, MP_PINKY_DIP, MP_PINKY_TIP = 17, 18, 19, 20

# InterHand2.6M ordering, single hand -- 21 keypoints, indices 0..20.
# Per finger the order is tip -> DIP -> PIP -> MCP, then wrist last.
#   0  Thumb_4 (tip)   1  Thumb_3   2  Thumb_2    3  Thumb_1 (CMC/MCP)
#   4  Index_4 (tip)   5  Index_3   6  Index_2    7  Index_1 (MCP)
#   8  Middle_4        9  Middle_3 10  Middle_2  11  Middle_1
#  12  Ring_4         13  Ring_3   14  Ring_2    15  Ring_1
#  16  Pinky_4        17  Pinky_3  18  Pinky_2   19  Pinky_1
#  20  Wrist
# In two-hand annotations the right hand uses 0-20 and the left 21-41.
IH_WRIST = 20

# Mapping from each MediaPipe joint to its InterHand2.6M counterpart (single hand).
MP_TO_IH = {
    MP_WRIST: 20,
    MP_THUMB_CMC: 3,  MP_THUMB_MCP: 2,  MP_THUMB_IP: 1,  MP_THUMB_TIP: 0,
    MP_INDEX_MCP: 7,  MP_INDEX_PIP: 6,  MP_INDEX_DIP: 5, MP_INDEX_TIP: 4,
    MP_MIDDLE_MCP: 11, MP_MIDDLE_PIP: 10, MP_MIDDLE_DIP: 9, MP_MIDDLE_TIP: 8,
    MP_RING_MCP: 15, MP_RING_PIP: 14, MP_RING_DIP: 13, MP_RING_TIP: 12,
    MP_PINKY_MCP: 19, MP_PINKY_PIP: 18, MP_PINKY_DIP: 17, MP_PINKY_TIP: 16,
}

# Joint angles to evaluate. Each entry is (name, [p1, p2, p3]) using MediaPipe
# indices; the angle is at p2, formed by vectors p2->p1 and p2->p3, in degrees.
# These are flexion angles only; MCP abduction is left as an extension exercise.
ANGLE_DEFS = [
    ("Thumb_MCP",  [MP_THUMB_CMC,  MP_THUMB_MCP,  MP_THUMB_IP]),
    ("Thumb_IP",   [MP_THUMB_MCP,  MP_THUMB_IP,   MP_THUMB_TIP]),
    ("Index_MCP",  [MP_WRIST,      MP_INDEX_MCP,  MP_INDEX_PIP]),
    ("Index_PIP",  [MP_INDEX_MCP,  MP_INDEX_PIP,  MP_INDEX_DIP]),
    ("Index_DIP",  [MP_INDEX_PIP,  MP_INDEX_DIP,  MP_INDEX_TIP]),
    ("Middle_MCP", [MP_WRIST,      MP_MIDDLE_MCP, MP_MIDDLE_PIP]),
    ("Middle_PIP", [MP_MIDDLE_MCP, MP_MIDDLE_PIP, MP_MIDDLE_DIP]),
    ("Middle_DIP", [MP_MIDDLE_PIP, MP_MIDDLE_DIP, MP_MIDDLE_TIP]),
    ("Ring_MCP",   [MP_WRIST,      MP_RING_MCP,   MP_RING_PIP]),
    ("Ring_PIP",   [MP_RING_MCP,   MP_RING_PIP,   MP_RING_DIP]),
    ("Ring_DIP",   [MP_RING_PIP,   MP_RING_DIP,   MP_RING_TIP]),
    ("Pinky_MCP",  [MP_WRIST,      MP_PINKY_MCP,  MP_PINKY_PIP]),
    ("Pinky_PIP",  [MP_PINKY_MCP,  MP_PINKY_PIP,  MP_PINKY_DIP]),
    ("Pinky_DIP",  [MP_PINKY_PIP,  MP_PINKY_DIP,  MP_PINKY_TIP]),
]
ANGLE_NAMES = [a[0] for a in ANGLE_DEFS]

# Skeleton edges in MediaPipe convention -- used for drawing.
MP_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm
]


# =============================================================================
# GEOMETRY
# =============================================================================
def angle_deg(p1, p2, p3) -> float:
    """Angle at p2 between vectors p2->p1 and p2->p3, in degrees."""
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def angles_from_mp(landmarks_21x3: np.ndarray) -> dict[str, float]:
    """Compute all angles from a (21,3) array in MediaPipe order."""
    out = {}
    for name, (i, j, k) in ANGLE_DEFS:
        out[name] = angle_deg(landmarks_21x3[i], landmarks_21x3[j], landmarks_21x3[k])
    return out


def angles_from_ih(joints_21x3_ih: np.ndarray) -> dict[str, float]:
    """Compute the same angles from a (21,3) array in InterHand2.6M order."""
    out = {}
    for name, (i, j, k) in ANGLE_DEFS:
        ii, jj, kk = MP_TO_IH[i], MP_TO_IH[j], MP_TO_IH[k]
        out[name] = angle_deg(joints_21x3_ih[ii], joints_21x3_ih[jj], joints_21x3_ih[kk])
    return out


def project_cam_to_img(joint_cam_mm: np.ndarray, focal: np.ndarray, princpt: np.ndarray) -> np.ndarray:
    """Pinhole projection. joint_cam_mm: (N,3). Returns (N,2) in pixels."""
    x = joint_cam_mm[:, 0] / joint_cam_mm[:, 2] * focal[0] + princpt[0]
    y = joint_cam_mm[:, 1] / joint_cam_mm[:, 2] * focal[1] + princpt[1]
    return np.stack([x, y], axis=1)


# =============================================================================
# MEDIAPIPE WRAPPER
# =============================================================================
class MediaPipeHands:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.3):
        self._mp = mp.solutions.hands
        self.hands = self._mp.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        """Returns list of detections, each with handedness, image- and world-landmarks."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return []
        h, w = image_bgr.shape[:2]
        out = []
        for idx, lm in enumerate(res.multi_hand_landmarks):
            wlm = res.multi_hand_world_landmarks[idx]
            handedness = res.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            score = res.multi_handedness[idx].classification[0].score
            img_xyz = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark])     # px
            world = np.array([[p.x, p.y, p.z] for p in wlm.landmark])              # m, wrist-relative
            out.append({
                "handedness": handedness,
                "handedness_score": float(score),
                "image_landmarks": img_xyz,
                "world_landmarks": world,
            })
        return out

    def close(self):
        self.hands.close()


# =============================================================================
# INTERHAND2.6M LOADER
# =============================================================================
class InterHand26MLoader:
    """
    Iterates over InterHand2.6M images that physically exist in the images folder.
    Expected layout (whichever fps you downloaded -- 5fps recommended for size):

        <root>/
            annotations/<split>/InterHand2.6M_<...>_<split>_camera.json
            annotations/<split>/InterHand2.6M_<...>_<split>_joint_3d.json
            annotations/<split>/InterHand2.6M_<...>_<split>_data.json
            images/<split>/Capture<X>/<seq_name>/cam<NNNNNN>/image*.jpg
    """

    def __init__(self, root: str | Path, split: str = "test"):
        self.root = Path(root)
        self.split = split
        ann_dir = self.root / "annotations" / split
        if not ann_dir.exists():
            raise FileNotFoundError(f"Annotations dir not found: {ann_dir}")

        cam_path = self._find_one(ann_dir, f"*{split}_camera.json")
        j3d_path = self._find_one(ann_dir, f"*{split}_joint_3d.json")
        data_path = self._find_one(ann_dir, f"*{split}_data.json")

        print(f"[loader] reading {cam_path.name}")
        with open(cam_path) as f:
            self.cameras = json.load(f)
        print(f"[loader] reading {j3d_path.name}")
        with open(j3d_path) as f:
            self.joints_3d = json.load(f)
        print(f"[loader] reading {data_path.name}")
        with open(data_path) as f:
            self.data = json.load(f)

        self.image_dir = self.root / "images" / split
        self.ann_by_img = {a["image_id"]: a for a in self.data["annotations"]}

    @staticmethod
    def _find_one(folder: Path, pattern: str) -> Path:
        matches = list(folder.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {folder}")
        return matches[0]

    def index_existing(self) -> list[dict]:
        """Return metadata for every annotated image whose file is on disk."""
        present = []
        for img in tqdm(self.data["images"], desc="indexing"):
            path = self.image_dir / img["file_name"]
            if path.exists():
                present.append({"img": img, "path": path})
        return present

    def sample(self, image_meta: dict, image_path: Path) -> dict | None:
        """Build a sample dict with GT joints in camera frame for one image."""
        cap_id = str(image_meta["capture"])
        frame_idx = str(image_meta["frame_idx"])
        cam_id = str(image_meta["camera"]) if "camera" in image_meta else str(image_meta["cam"])

        if cap_id not in self.joints_3d or frame_idx not in self.joints_3d[cap_id]:
            return None
        jd = self.joints_3d[cap_id][frame_idx]
        world = np.asarray(jd["world_coord"], dtype=np.float32)        # (42,3) mm
        valid = np.asarray(jd["joint_valid"], dtype=np.float32).flatten()

        cams = self.cameras[cap_id]
        try:
            campos = np.asarray(cams["campos"][cam_id], dtype=np.float32)
            camrot = np.asarray(cams["camrot"][cam_id], dtype=np.float32)
            focal = np.asarray(cams["focal"][cam_id], dtype=np.float32)
            princpt = np.asarray(cams["princpt"][cam_id], dtype=np.float32)
        except KeyError:
            return None

        # World -> camera frame
        joint_cam = (camrot @ (world - campos[None]).T).T  # (42,3) mm

        ann = self.ann_by_img.get(image_meta["id"], {})
        hand_type = ann.get("hand_type", "right")

        return {
            "image_path": str(image_path),
            "image_meta": image_meta,
            "joint_cam": joint_cam,           # (42,3) mm in camera frame
            "joint_valid": valid,             # (42,)
            "focal": focal,
            "princpt": princpt,
            "hand_type": hand_type,           # 'right' / 'left' / 'interacting'
        }


# =============================================================================
# HAND-MATCHING (MediaPipe detections <-> GT hands)
# =============================================================================
def split_gt_hands(sample: dict) -> dict[str, dict]:
    """Return {'right': {...}, 'left': {...}} where each has joints (21,3),
    valid (21,), present (bool). InterHand annotation is right-then-left."""
    jc = sample["joint_cam"]
    jv = sample["joint_valid"]
    out = {}
    for name, sl in (("right", slice(0, 21)), ("left", slice(21, 42))):
        joints = jc[sl]
        valid = jv[sl]
        # A hand counts as present if at least the wrist + a couple of finger
        # tips are valid -- otherwise InterHand may not actually annotate it.
        present = bool(valid[IH_WRIST] > 0.5 and valid.sum() >= 15)
        out[name] = {"joints": joints, "valid": valid, "present": present}
    return out


def match_detections_to_gt(detections: list[dict],
                           gt_hands: dict[str, dict],
                           sample: dict) -> list[tuple[dict, str]]:
    """
    Pair each MediaPipe detection with 'right' or 'left' GT hand by 2D
    proximity in image space. MediaPipe handedness can be flipped relative to
    InterHand because MP assumes a mirrored selfie image, so we don't trust it.
    """
    focal = sample["focal"]; princpt = sample["princpt"]
    # GT 2D centroid for each present hand
    centers = {}
    for side, hand in gt_hands.items():
        if not hand["present"]:
            continue
        v = hand["valid"] > 0.5
        if v.sum() < 5:
            continue
        proj = project_cam_to_img(hand["joints"][v], focal, princpt)
        centers[side] = proj.mean(axis=0)

    pairings = []
    used_sides = set()
    # Greedy: assign each detection to the closest unused GT hand
    for det in detections:
        det_center = det["image_landmarks"][:, :2].mean(axis=0)
        best_side, best_d = None, np.inf
        for side, c in centers.items():
            if side in used_sides:
                continue
            d = float(np.linalg.norm(det_center - c))
            if d < best_d:
                best_d, best_side = d, side
        if best_side is not None:
            pairings.append((det, best_side))
            used_sides.add(best_side)
    return pairings


# =============================================================================
# METRICS
# =============================================================================
def per_sample_errors(mp_angles: dict, gt_angles: dict) -> dict[str, float]:
    return {name: float(mp_angles[name] - gt_angles[name]) for name in ANGLE_NAMES}


def aggregate(records: list[dict]) -> pd.DataFrame:
    """Compute MAE / RMSE / std / median absolute error per angle, plus overall."""
    df = pd.DataFrame(records)
    rows = []
    for name in ANGLE_NAMES:
        errs = df[name].dropna().to_numpy()
        if len(errs) == 0:
            rows.append({"angle": name, "n": 0, "mae": np.nan, "rmse": np.nan,
                         "median_abs_err": np.nan, "std_err": np.nan})
            continue
        rows.append({
            "angle": name,
            "n": int(len(errs)),
            "mae": float(np.mean(np.abs(errs))),
            "rmse": float(np.sqrt(np.mean(errs ** 2))),
            "median_abs_err": float(np.median(np.abs(errs))),
            "std_err": float(np.std(errs)),
        })
    summary = pd.DataFrame(rows)
    # Overall row
    all_errs = df[ANGLE_NAMES].to_numpy().flatten()
    all_errs = all_errs[~np.isnan(all_errs)]
    overall = {
        "angle": "ALL",
        "n": int(len(all_errs)),
        "mae": float(np.mean(np.abs(all_errs))) if len(all_errs) else np.nan,
        "rmse": float(np.sqrt(np.mean(all_errs ** 2))) if len(all_errs) else np.nan,
        "median_abs_err": float(np.median(np.abs(all_errs))) if len(all_errs) else np.nan,
        "std_err": float(np.std(all_errs)) if len(all_errs) else np.nan,
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return summary


# =============================================================================
# VISUALIZATION
# =============================================================================
def draw_skeleton(ax, pts2d: np.ndarray, color: str, label: str | None = None,
                  edges=MP_EDGES, lw: float = 1.5, ms: float = 18):
    for i, j in edges:
        ax.plot([pts2d[i, 0], pts2d[j, 0]], [pts2d[i, 1], pts2d[j, 1]],
                color=color, linewidth=lw, alpha=0.85)
    ax.scatter(pts2d[:, 0], pts2d[:, 1], s=ms, c=color, edgecolor="black",
               linewidth=0.4, label=label, zorder=3)


def ih_to_mp_order(joints_ih_21x3: np.ndarray) -> np.ndarray:
    """Reorder a (21,3) IH array into MediaPipe order so MP_EDGES draws correctly."""
    out = np.zeros_like(joints_ih_21x3)
    for mp_idx, ih_idx in MP_TO_IH.items():
        out[mp_idx] = joints_ih_21x3[ih_idx]
    return out


def make_visual(sample: dict, det: dict, side: str,
                gt_joints_ih: np.ndarray, gt_valid: np.ndarray,
                mp_angles: dict, gt_angles: dict,
                out_path: Path):
    img = cv2.cvtColor(cv2.imread(sample["image_path"]), cv2.COLOR_BGR2RGB)
    focal, princpt = sample["focal"], sample["princpt"]

    gt_proj = project_cam_to_img(gt_joints_ih, focal, princpt)  # (21,2) in IH order
    gt_proj_mp_order = ih_to_mp_order(
        np.column_stack([gt_proj, np.zeros(len(gt_proj))])
    )[:, :2]

    mp_2d = det["image_landmarks"][:, :2]

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img); ax0.set_title("Image")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(img)
    draw_skeleton(ax1, mp_2d, color="lime", label="MediaPipe")
    ax1.set_title(f"MediaPipe ({det['handedness']})"); ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(img)
    draw_skeleton(ax2, gt_proj_mp_order, color="red", label=f"GT ({side})")
    ax2.set_title("Ground Truth"); ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    x = np.arange(len(ANGLE_NAMES))
    width = 0.4
    mp_vals = [mp_angles[n] for n in ANGLE_NAMES]
    gt_vals = [gt_angles[n] for n in ANGLE_NAMES]
    ax3.bar(x - width / 2, gt_vals, width, label="GT", color="red", alpha=0.7)
    ax3.bar(x + width / 2, mp_vals, width, label="MP", color="lime", alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax3.set_xticks(x); ax3.set_xticklabels(ANGLE_NAMES, rotation=70, fontsize=8)
    ax3.set_ylabel("angle (deg)"); ax3.set_title("Joint angles: GT vs MP")
    ax3.legend(); ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(Path(sample["image_path"]).name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_summary(summary_df: pd.DataFrame, out_path: Path):
    df = summary_df[summary_df["angle"] != "ALL"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["mae"], 0.4, label="MAE")
    ax.bar(x + 0.2, df["rmse"], 0.4, label="RMSE")
    ax.set_xticks(x); ax.set_xticklabels(df["angle"], rotation=60)
    ax.set_ylabel("error (degrees)"); ax.set_title("MediaPipe vs InterHand2.6M -- per-angle error")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to InterHand2.6M dataset root")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default="./eval_out", help="Output directory")
    ap.add_argument("--max-samples", type=int, default=500,
                    help="Cap on images to process (after filtering for existence)")
    ap.add_argument("--num-vis", type=int, default=8, help="Number of qualitative figures to save")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.3, help="MP detection confidence threshold")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out); (out_dir / "vis").mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    loader = InterHand26MLoader(args.root, split=args.split)
    present = loader.index_existing()
    if not present:
        print(f"No images found under {loader.image_dir}. "
              f"Download at least one Capture folder and try again.", file=sys.stderr)
        sys.exit(1)
    print(f"[loader] {len(present)} annotated images present on disk")

    if len(present) > args.max_samples:
        present = random.sample(present, args.max_samples)
        print(f"[loader] sampled {len(present)} for evaluation")

    # ---- Run inference + collect errors ----
    mp_runner = MediaPipeHands(min_detection_confidence=args.min_conf)
    error_records: list[dict] = []
    raw_records: list[dict] = []  # for optional CSV of raw angles
    detections_for_vis: list[tuple] = []

    for entry in tqdm(present, desc="evaluating"):
        sample = loader.sample(entry["img"], entry["path"])
        if sample is None:
            continue

        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        detections = mp_runner.predict(img)
        if not detections:
            continue

        gt_hands = split_gt_hands(sample)
        if not any(h["present"] for h in gt_hands.values()):
            continue

        pairings = match_detections_to_gt(detections, gt_hands, sample)
        for det, side in pairings:
            gt_hand = gt_hands[side]
            if not gt_hand["present"]:
                continue
            mp_a = angles_from_mp(det["world_landmarks"])
            gt_a = angles_from_ih(gt_hand["joints"])
            errs = per_sample_errors(mp_a, gt_a)

            rec = {
                "image": Path(sample["image_path"]).name,
                "capture": sample["image_meta"]["capture"],
                "seq": sample["image_meta"].get("seq_name", ""),
                "cam": sample["image_meta"].get("camera", sample["image_meta"].get("cam", "")),
                "frame_idx": sample["image_meta"]["frame_idx"],
                "side": side,
                "mp_handedness": det["handedness"],
                **errs,
            }
            error_records.append(rec)
            raw_records.append({
                **{k: rec[k] for k in ("image", "side")},
                **{f"{k}_mp": v for k, v in mp_a.items()},
                **{f"{k}_gt": v for k, v in gt_a.items()},
            })
            detections_for_vis.append((sample, det, side, gt_hand, mp_a, gt_a))

    mp_runner.close()

    if not error_records:
        print("No matched detections produced -- nothing to evaluate.", file=sys.stderr)
        sys.exit(2)

    # ---- Save tables ----
    err_df = pd.DataFrame(error_records)
    err_df.to_csv(out_dir / "per_sample_errors.csv", index=False)
    pd.DataFrame(raw_records).to_csv(out_dir / "per_sample_angles.csv", index=False)

    summary = aggregate(error_records)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary (degrees) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    plot_error_summary(summary, out_dir / "error_summary.png")

    # ---- Save N qualitative visualizations ----
    n_vis = min(args.num_vis, len(detections_for_vis))
    picks = random.sample(detections_for_vis, n_vis)
    for i, (sample, det, side, gt_hand, mp_a, gt_a) in enumerate(picks):
        fname = out_dir / "vis" / f"sample_{i:02d}.png"
        try:
            make_visual(sample, det, side,
                        gt_hand["joints"], gt_hand["valid"],
                        mp_a, gt_a, fname)
        except Exception as e:
            print(f"[vis] skipped sample {i}: {e}")

    print(f"\nWrote results to {out_dir.resolve()}")
    print("  - summary.csv             per-angle MAE/RMSE/etc.")
    print("  - per_sample_errors.csv   one row per matched (image, hand)")
    print("  - per_sample_angles.csv   raw GT/MP angles (for further analysis)")
    print("  - error_summary.png       per-angle MAE/RMSE bar chart")
    print(f"  - vis/sample_*.png        {n_vis} qualitative figures")


if __name__ == "__main__":
    main()
