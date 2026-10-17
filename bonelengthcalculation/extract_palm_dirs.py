#!/usr/bin/env python3
"""
extract_palm_dirs.py — recover the five wrist->knuckle directions from the same
flat-hand photos the bone-length calibration used.

    python extract_palm_dirs.py \
        --input ./images/filledsquares --model hand_landmarker.task \
        --hand Right --square-mm 20 --out calib_filled_tilt35/palm_dirs.json

Then:
    python calib_adapter.py --primary calib_filled_tilt35 --fallback calib_filled \
        --hand Right --palm-dirs calib_filled_tilt35/palm_dirs.json \
        --out calib/hand_calibration_ekf_right.json --report

Why this exists
---------------
`PersonalizedHandModel` needs the metacarpal fan -- where each knuckle sits
relative to the wrist -- to place the MCP joints. Bone lengths alone give the
DISTANCE wrist->knuckle but not the DIRECTION, and the calibration script
discards the direction: it computes `paper` (landmarks warped into millimetre
paper coordinates), measures lengths from it, and throws the coordinates away.

Without measured directions the model falls back to a nominal anatomical fan,
which puts every knuckle in the wrong place and biases the whole keypoint
residual in the EKF's second measurement block.

This does NOT modify calibrate_bones_filled_squares.py. It imports and reuses
its detection, marker-clustering and homography code, so the geometry is
identical by construction rather than by reimplementation.

⚠️ Run it with the SAME --hand, --square-mm and marker settings you used for the
bone lengths, or the two halves of the calibration will disagree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

MCP_IDX = (1, 5, 9, 13, 17)          # thumb CMC, then index/middle/ring/pinky MCP
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def load_calib_module(path: str):
    """Import calibrate_bones_filled_squares.py from an explicit path.
    Safe: the script guards main() behind __name__ == '__main__'."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit(f"calibration script not found: {path}")
    spec = importlib.util.spec_from_file_location("_calib_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_calib_mod"] = mod
    spec.loader.exec_module(mod)
    for fn in ("detect_markers", "select_marker_cluster",
               "solve_plane_homography", "HandDetector",
               "iter_frames", "gather_inputs", "add_marker_args"):
        if not hasattr(mod, fn):
            sys.exit(f"{path} has no {fn}() — wrong script version?")
    return mod


def build_cfg(mod, args) -> argparse.Namespace:
    """Reconstruct the marker-detection config with the script's own defaults,
    so thresholds match the run that produced the bone lengths."""
    ap = argparse.ArgumentParser()
    mod.add_marker_args(ap)
    cfg = ap.parse_args([])
    cfg.square_mm = args.square_mm
    cfg.size_tol = args.size_tol
    cfg.min_markers = max(3, args.min_markers)     # homography needs >= 3
    cfg.no_refine = args.no_refine
    cfg.max_reproj_px = args.max_reproj_px
    cfg.method = "homography"
    cfg.hand = args.hand
    return cfg


def frame_palm_dirs(bgr, detector, cfg, mod) -> Optional[dict]:
    """One frame -> wrist-relative MCP offsets in paper millimetres."""
    squares = mod.select_marker_cluster(
        mod.detect_markers(bgr, cfg), tol=cfg.size_tol,
        prefer=getattr(mod, "MARKER_CLUSTER_PREFERENCE", "modal"))
    if len(squares) < cfg.min_markers:
        return None

    hand = detector.detect(bgr)
    if hand is None:
        return None
    if cfg.hand and hand["hand"].lower() != cfg.hand.lower():
        return None

    lm = hand["px"]
    h, w = bgr.shape[:2]
    if (lm[:, 0].min() < 2 or lm[:, 1].min() < 2
            or lm[:, 0].max() > w - 3 or lm[:, 1].max() > h - 3):
        return None

    H, reproj = mod.solve_plane_homography(squares, cfg.square_mm,
                                           refine=not cfg.no_refine)
    if H is None or not np.isfinite(reproj) or reproj > cfg.max_reproj_px:
        return None

    paper = cv2.perspectiveTransform(
        lm.reshape(1, -1, 2).astype(np.float32), np.linalg.inv(H)).reshape(-1, 2)

    # wrist-relative, in the paper plane. z = 0: the hand is pressed flat, so the
    # palm arch is DEFINED as zero by this measurement. That is a real limitation
    # of the flat-hand method, not a bug -- see PROJECT_CONTEXT sec.9.
    off = paper[list(MCP_IDX)] - paper[0]
    return {"mcp_paper_mm": off, "reproj": float(reproj),
            "hand": hand["hand"], "n_markers": len(squares)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--script", default="calibrate_bones_filled_squares.py",
                    help="path to the calibration script to reuse")
    ap.add_argument("--model", default=None, help="hand_landmarker.task")
    ap.add_argument("--hand", default=None, choices=[None, "Left", "Right"])
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--size-tol", type=float, default=0.18)
    ap.add_argument("--max-reproj-px", type=float, default=6.0)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--frame-step", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out", default="palm_dirs.json")
    args = ap.parse_args()

    mod = load_calib_module(args.script)
    cfg = build_cfg(mod, args)
    files = mod.gather_inputs(args.input)
    if not files:
        sys.exit("no input images/videos found")
    print(f"[palm_dirs] {len(files)} input file(s); square = {args.square_mm:.2f} mm")

    det = mod.HandDetector(args.model, args.min_conf)
    print(f"MediaPipe backend: {det.backend}")

    offs: List[np.ndarray] = []
    n_seen = 0
    for f in files:
        for idx, frame in mod.iter_frames(f, args.frame_step, args.max_frames):
            n_seen += 1
            r = frame_palm_dirs(frame, det, cfg, mod)
            if r is not None:
                offs.append(r["mcp_paper_mm"])

    if not offs:
        sys.exit("no usable frames — check --hand, --square-mm and marker settings")

    A = np.stack(offs)                                   # (N,5,2)
    med = np.median(A, axis=0)
    mad = np.median(np.abs(A - med), axis=0) * 1.4826

    P = np.column_stack([med, np.zeros(5)])              # (5,3), z = 0
    L = np.linalg.norm(P, axis=1)
    U = P / (L[:, None] + 1e-12)

    # Fan angles about the palm normal, measured from the middle metacarpal.
    # These are the near-constant quantities the v12 angle system calls
    # *_Fan and stores as calibration rather than as filter state.
    ang = np.degrees(np.arctan2(med[:, 0], med[:, 1]))
    fan = ang - ang[2]

    print(f"\nused {len(offs)} of {n_seen} frames")
    print(f"{'finger':<8}{'dx_mm':>9}{'dy_mm':>9}{'len_mm':>9}"
          f"{'fan_deg':>9}{'MAD_mm':>9}")
    for i, f in enumerate(FINGERS):
        print(f"{f:<8}{med[i,0]:9.2f}{med[i,1]:9.2f}{L[i]:9.2f}"
              f"{fan[i]:9.1f}{np.linalg.norm(mad[i]):9.2f}")

    out = {
        "schema": "palm_dirs/v1",
        "hand": args.hand or "unspecified",
        "n_frames_used": len(offs),
        "n_frames_seen": n_seen,
        "mcp_paper_mm": med.tolist(),
        "palm_dirs": U.tolist(),
        "metacarpal_mm": L.tolist(),
        "fan_deg": fan.tolist(),
        "mad_mm": mad.tolist(),
        "note": "z = 0 by construction: the hand is pressed flat, so palm arch "
                "and thumb opposition are defined as zero by this measurement.",
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")

    worst = int(np.argmax(np.linalg.norm(mad, axis=1)))
    print(f"least repeatable: {FINGERS[worst]} "
          f"(MAD {np.linalg.norm(mad[worst]):.2f} mm)")
    if np.linalg.norm(mad[worst]) > 3.0:
        print("  > 3 mm spread suggests the hand shifted between shots; "
              "consider dropping outlier photos.")


if __name__ == "__main__":
    main()
