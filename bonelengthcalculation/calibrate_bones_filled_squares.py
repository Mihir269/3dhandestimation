#!/usr/bin/env python3
"""
calibrate_bones_filled_squares.py
=================================
Subject-specific hand bone-length calibration from photos/videos of a flat hand
resting on a sheet that carries FILLED COLOUR SQUARES of known size (default 20 mm).

Pipeline
--------
  1. Detect the filled squares (HSV colour threshold -> contour -> quad -> sub-pixel
     edge-line refinement).
  2. Build an image -> paper-plane mapping:
       * >=3 squares  -> full perspective homography (square corners, unknown layout
                         solved jointly; squares assumed same size + same orientation)
       * <3 squares   -> linear px/mm scale field (or a constant scale)
  3. Run MediaPipe Hands, map the 21 landmarks into paper-plane millimetres.
  4. Measure the 20 bone lengths, reject out-of-plane (foreshortened) bones.
  5. Aggregate over all frames: mean / median / std / variance / CV / CI, with
     MAD-based outlier rejection.
  6. Optional parallax correction: the back of the hand sits ~20 mm ABOVE the paper,
     so it is magnified by D/(D-h).  Shots from DIFFERENT CAMERA HEIGHTS let us solve
     for the hand-plane offset h and the true lengths jointly.

Usage
-----
  python calibrate_bones_filled_squares.py --input ./shots --out ./calib_filled \
         --square-mm 20 --model hand_landmarker.task --plots --parallax

  --input accepts a directory (images + videos), or repeated file paths.

Outputs (in --out)
------------------
  per_frame_bone_lengths_mm.csv   one row per (frame, bone)
  bone_length_stats.csv           per-bone mean/median/std/var/CV/CI + outliers
  frame_report.csv                per-frame detection diagnostics
  bone_lengths.json               final lengths in mm and metres (FK-ready)
  summary.json                    run configuration + global diagnostics
  debug/*.jpg                     annotated overlays (with --debug)
  *.png                           distribution + stability plots (with --plots)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# 1 · Hand model constants  (21-keypoint MediaPipe / FreiHAND order — identical)
# --------------------------------------------------------------------------------------

BONES = [
    (0, 1, "Thumb_Wrist_CMC", "Thumb"),
    (1, 2, "Thumb_CMC_MCP", "Thumb"),
    (2, 3, "Thumb_MCP_IP", "Thumb"),
    (3, 4, "Thumb_IP_TIP", "Thumb"),
    (0, 5, "Index_Wrist_MCP", "Index"),
    (5, 6, "Index_MCP_PIP", "Index"),
    (6, 7, "Index_PIP_DIP", "Index"),
    (7, 8, "Index_DIP_TIP", "Index"),
    (0, 9, "Middle_Wrist_MCP", "Middle"),
    (9, 10, "Middle_MCP_PIP", "Middle"),
    (10, 11, "Middle_PIP_DIP", "Middle"),
    (11, 12, "Middle_DIP_TIP", "Middle"),
    (0, 13, "Ring_Wrist_MCP", "Ring"),
    (13, 14, "Ring_MCP_PIP", "Ring"),
    (14, 15, "Ring_PIP_DIP", "Ring"),
    (15, 16, "Ring_DIP_TIP", "Ring"),
    (0, 17, "Pinky_Wrist_MCP", "Pinky"),
    (17, 18, "Pinky_MCP_PIP", "Pinky"),
    (18, 19, "Pinky_PIP_DIP", "Pinky"),
    (19, 20, "Pinky_DIP_TIP", "Pinky"),
]
BONE_NAMES = [b[2] for b in BONES]
MIDDLE_MCP_PIP_BONE_IDX = 9          # (9,10) — normalisation reference bone

FINGER_COLORS = {
    "Thumb": "#FF6B6B", "Index": "#4ECDC4", "Middle": "#45B7D1",
    "Ring": "#96CEB4", "Pinky": "#FFEAA7",
}

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


# --------------------------------------------------------------------------------------
# 2 · Planar-quad geometry helpers
# --------------------------------------------------------------------------------------

def order_quad(pts) -> np.ndarray:
    """Return the 4 corners sorted by angle about their centroid (consistent winding)."""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    c = p.mean(0)
    return p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))].astype(np.float32)


def quad_metrics(q: np.ndarray) -> dict:
    e = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    d1 = float(np.linalg.norm(q[2] - q[0]))
    d2 = float(np.linalg.norm(q[3] - q[1]))
    return {
        "side": float(np.mean(e)),
        "side_ratio": float(min(e) / max(max(e), 1e-9)),
        "diag_ratio": float(min(d1, d2) / max(max(d1, d2), 1e-9)),
    }


def refine_quad_edges(pts: np.ndarray, quad: np.ndarray, band=0.14, trim=0.12):
    """
    Sub-pixel corner refinement: fit a total-least-squares line to the support points
    near each of the 4 edges (ignoring the corner regions), then intersect neighbours.
    `pts` are the marker's support points (contour points or component pixels).
    """
    q = order_quad(quad)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 5:
            return None
        u = d / L
        n = np.array([-u[1], u[0]], np.float32)
        rel = pts - a
        perp = np.abs(rel @ n)
        t = (rel @ u) / L
        sel = (perp < band * L) & (t > trim) & (t < 1.0 - trim)
        P = pts[sel]
        if len(P) < 12:
            return None
        m = P.mean(0)
        C = np.cov((P - m).T)
        if not np.all(np.isfinite(C)):
            return None
        _, evec = np.linalg.eigh(C)
        lines.append((m, evec[:, -1]))          # (point on line, direction)

    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i], lines[(i + 1) % 4]
        A = np.array([d1, -d2], np.float64).T
        if abs(np.linalg.det(A)) < 1e-8:
            return None
        t = np.linalg.solve(A, (p2 - p1).astype(np.float64))
        corners.append(p1 + t[0] * d1)
    out = order_quad(np.array(corners, np.float32))
    if not np.all(np.isfinite(out)):
        return None
    return out


# ======================================================================================
# ==== MARKER DETECTION — FILLED COLOUR SQUARES ========================================
# ======================================================================================

MARKER_KIND = "filled_colour_square"
DEFAULT_SQUARE_MM = 20.0


def add_marker_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("filled-square detection")
    g.add_argument("--hue-lo", type=int, default=105, help="OpenCV hue low (purple≈105)")
    g.add_argument("--hue-hi", type=int, default=165, help="OpenCV hue high (purple≈165)")
    g.add_argument("--sat-min", type=int, default=45)
    g.add_argument("--val-min", type=int, default=20)
    g.add_argument("--val-max", type=int, default=235)
    g.add_argument("--fill-min", type=float, default=0.75,
                   help="min contour_area / quad_area — enforces a SOLID square")
    g.add_argument("--min-area-frac", type=float, default=0.0002)
    g.add_argument("--max-area-frac", type=float, default=0.05)
    g.add_argument("--side-ratio-min", type=float, default=0.70)
    g.add_argument("--diag-ratio-min", type=float, default=0.75)


def detect_markers(bgr: np.ndarray, cfg) -> list:
    """Detect solid colour squares. Returns [{corners, center, side_px, quality}]."""
    h, w = bgr.shape[:2]
    area_img = float(h * w)
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,
                       np.array([cfg.hue_lo, cfg.sat_min, cfg.val_min], np.uint8),
                       np.array([cfg.hue_hi, 255, cfg.val_max], np.uint8))
    if cv2.countNonZero(mask) < 0.0003 * area_img:      # fallback: any saturated cool hue
        mask = cv2.inRange(hsv, np.array([90, 35, 15], np.uint8),
                           np.array([175, 255, 245], np.uint8))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.min_area_frac * area_img or a > cfg.max_area_frac * area_img:
            continue
        peri = cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, 0.035 * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
        qa = cv2.contourArea(order_quad(quad))
        if qa <= 0 or a / qa < cfg.fill_min:
            continue

        pts = c.reshape(-1, 2).astype(np.float32)
        ref = refine_quad_edges(pts, quad)
        q = ref if ref is not None else order_quad(quad)

        # reject squares clipped by the image border (their side is not measurable)
        if (q[:, 0].min() < 4 or q[:, 1].min() < 4 or
                q[:, 0].max() > w - 5 or q[:, 1].max() > h - 5):
            continue

        m = quad_metrics(q)
        if m["side_ratio"] < cfg.side_ratio_min or m["diag_ratio"] < cfg.diag_ratio_min:
            continue
        out.append({"corners": q, "center": q.mean(0), "side_px": m["side"],
                    "quality": float(min(m["side_ratio"], m["diag_ratio"]))})
    return out


MARKER_CLUSTER_PREFERENCE = "modal"   # all coloured blobs are markers -> take the mode

# ======================================================================================
# ==== END MARKER DETECTION ============================================================
# ======================================================================================


def select_marker_cluster(sq: list, tol=0.18, min_count=2, prefer="modal") -> list:
    """Keep the largest self-consistent group of equal-sized squares."""
    if len(sq) <= 1:
        return sq
    sides = np.array([s["side_px"] for s in sq], float)
    cands = []
    for ref in sides:
        keep = np.abs(sides - ref) / ref <= tol
        cands.append((int(keep.sum()), float(ref), keep))
    if prefer == "largest":
        valid = [c for c in cands if c[0] >= min_count] or cands
        best = max(valid, key=lambda t: t[1])
    else:
        med = float(np.median(sides))
        best = max(cands, key=lambda t: (t[0], -abs(t[1] - med)))
    return [s for s, k in zip(sq, best[2]) if k]


# --------------------------------------------------------------------------------------
# 3 · Image -> paper-plane mapping
# --------------------------------------------------------------------------------------

def _consistent_corner_order(squares: list) -> list:
    """
    Roll each square's corners so index 0 is the same physical corner across all squares.
    Squares are assumed drawn with a common orientation (they are, on grid paper).
    """
    angs = []
    for s in squares:
        q = s["corners"]
        c = q.mean(0)
        angs.append(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))
    a4 = np.concatenate([(a % (np.pi / 2)) for a in angs]) * 4.0
    phi = math.atan2(float(np.sin(a4).mean()), float(np.cos(a4).mean())) / 4.0

    rolled = []
    for s, a in zip(squares, angs):
        dev = np.abs(np.angle(np.exp(1j * (a - phi))))
        k = int(np.argmin(dev))
        rolled.append(np.roll(s["corners"], -k, axis=0))
    return rolled, phi


def solve_plane_homography(squares: list, square_mm: float, refine=True):
    """
    Joint solve for the plane homography H (paper mm -> image px) and the unknown
    in-plane translation of every square (square 0 pinned at the origin).
    Returns (H, rms_reproj_px) or (None, nan).
    """
    n = len(squares)
    if n < 3:
        return None, float("nan")
    corners, phi = _consistent_corner_order(squares)
    img_pts = np.concatenate(corners, axis=0).astype(np.float64)

    half = square_mm / 2.0
    tmpl = np.array([[half, half], [-half, half], [-half, -half], [half, -half]], np.float64)

    sides = np.array([s["side_px"] for s in squares], float)
    s_px_per_mm = float(np.median(sides)) / square_mm
    rot = phi - math.pi / 4.0
    R = np.array([[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]])
    c0 = squares[0]["center"].astype(np.float64)
    t0 = np.array([((R @ (s["center"].astype(np.float64) - c0)) / s_px_per_mm)
                   for s in squares])

    paper_pts = np.concatenate([t0[i] + tmpl for i in range(n)], axis=0)
    H0, _ = cv2.findHomography(paper_pts.astype(np.float32),
                               img_pts.astype(np.float32), method=0)
    if H0 is None or not np.all(np.isfinite(H0)):
        return None, float("nan")
    H0 = H0 / H0[2, 2]

    def project(H, P):
        Q = np.c_[P, np.ones(len(P))] @ H.T
        w = np.where(np.abs(Q[:, 2]) < 1e-12, 1e-12, Q[:, 2])
        return Q[:, :2] / w[:, None]

    def rms(H, T):
        P = np.concatenate([T[i] + tmpl for i in range(n)], axis=0)
        return float(np.sqrt(((project(H, P) - img_pts) ** 2).sum(1).mean()))

    H, T = H0, t0.copy()
    if refine:
        try:
            from scipy.optimize import least_squares

            def unpack(x):
                Hh = np.append(x[:8], 1.0).reshape(3, 3)
                Tt = np.vstack([np.zeros(2), x[8:].reshape(n - 1, 2)]) + T[0]
                return Hh, Tt

            def resid(x):
                Hh, Tt = unpack(x)
                P = np.concatenate([Tt[i] + tmpl for i in range(n)], axis=0)
                return (project(Hh, P) - img_pts).ravel()

            x0 = np.concatenate([H0.ravel()[:8], (t0[1:] - t0[0]).ravel()])
            sol = least_squares(resid, x0, method="lm", max_nfev=400)
            Hr, Tr = unpack(sol.x)
            if np.all(np.isfinite(Hr)) and rms(Hr, Tr) < rms(H0, t0):
                H, T = Hr / Hr[2, 2], Tr
        except Exception:
            pass
    return H, rms(H, T)


def fit_scale_field(squares: list, square_mm: float):
    """px-per-mm as a function of image position (linear model when enough squares)."""
    c = np.array([s["center"] for s in squares], float)
    s = np.array([s["side_px"] for s in squares], float) / square_mm
    med = float(np.median(s))
    if len(squares) >= 5:
        A = np.c_[np.ones(len(s)), c]
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        return ("linear", coef, med)
    return ("const", None, med)


def eval_scale(model, p, guard=0.35) -> float:
    kind, coef, med = model
    if kind == "const":
        return med
    v = float(coef[0] + coef[1] * p[0] + coef[2] * p[1])
    if not np.isfinite(v) or v <= 0 or abs(v - med) / med > guard:
        return med
    return v


# --------------------------------------------------------------------------------------
# 4 · Optional: printed-grid pitch field (dense perspective shape, squares set the scale)
# --------------------------------------------------------------------------------------

def grid_period_at(gray: np.ndarray, cx: float, cy: float, N=384, min_snr=25.0):
    """Dominant periodic pitch (px) of the printed graph paper in a patch, sub-bin refined."""
    h, w = gray.shape
    x0, y0 = int(cx - N // 2), int(cy - N // 2)
    if x0 < 0 or y0 < 0 or x0 + N > w or y0 + N > h:
        return None
    p = gray[y0:y0 + N, x0:x0 + N].astype(np.float32)
    p = p - cv2.GaussianBlur(p, (0, 0), 8)
    win = np.outer(np.hanning(N), np.hanning(N)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2(p * win)))
    c = N // 2
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.hypot(xx - c, yy - c)
    F = np.where((r < 8) | (r > N // 2 - 3), 0.0, F)
    j, i = np.unravel_index(int(np.argmax(F)), F.shape)
    snr = float(F.max() / (F.mean() + 1e-9))
    if snr < min_snr or not (1 <= i < N - 1 and 1 <= j < N - 1):
        return None
    # parabolic sub-bin refinement in x and y
    def sub(a, b, cc):
        d = a - 2 * b + cc
        return 0.0 if abs(d) < 1e-9 else 0.5 * (a - cc) / d
    di = sub(F[j, i - 1], F[j, i], F[j, i + 1])
    dj = sub(F[j - 1, i], F[j, i], F[j + 1, i])
    R = math.hypot((i + di) - c, (j + dj) - c)
    if R < 1e-6:
        return None
    return N / R, snr


def build_grid_field(bgr: np.ndarray, step=300, N=256, min_snr=40.0):
    """Sample grid pitch on a lattice; return (points Nx2, pitch_px N) or (None, None)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    P, V = [], []
    for cy in range(N // 2, h - N // 2, step):
        for cx in range(N // 2, w - N // 2, step):
            r = grid_period_at(g, cx, cy, N=N, min_snr=min_snr)
            if r is not None:
                P.append((cx, cy)); V.append(r[0])
    if len(P) < 6:
        return None, None
    P = np.array(P, float); V = np.array(V, float)
    # keep the densest pitch cluster (off-paper patches and harmonic locks scatter)
    best, bkeep = -1, None
    for v in V:
        k = np.abs(V - v) / v < 0.08
        if k.sum() > best:
            best, bkeep = int(k.sum()), k
    if best < 6:
        return None, None
    return P[bkeep], V[bkeep]


def fit_grid_shape(P, V):
    """Linear model of relative pitch (normalised to 1 at the median)."""
    v = V / np.median(V)
    A = np.c_[np.ones(len(v)), P]
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = float(np.std(v - A @ coef))
    return coef, resid


# --------------------------------------------------------------------------------------
# 5 · MediaPipe wrapper (lazy import; Tasks API preferred, legacy fallback)
# --------------------------------------------------------------------------------------

class HandDetector:
    def __init__(self, model_path=None, min_conf=0.3):
        import mediapipe as mp
        self.mp = mp
        self.backend = None
        if model_path and Path(model_path).exists():
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision as mpv
            opts = mpv.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mpv.RunningMode.IMAGE, num_hands=1,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf)
            self.det = mpv.HandLandmarker.create_from_options(opts)
            self.backend = "tasks"
        else:
            self.det = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1, model_complexity=1,
                min_detection_confidence=min_conf)
            self.backend = "legacy"

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        if self.backend == "tasks":
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            r = self.det.detect(img)
            if not r.hand_landmarks:
                return None
            px = np.array([[p.x * w, p.y * h] for p in r.hand_landmarks[0]], np.float32)
            world = (np.array([[p.x, p.y, p.z] for p in r.hand_world_landmarks[0]], np.float32)
                     if r.hand_world_landmarks else None)
            cat = r.handedness[0][0]
            return {"px": px, "world": world, "hand": cat.category_name,
                    "score": float(cat.score)}
        r = self.det.process(rgb)
        if not r.multi_hand_landmarks:
            return None
        px = np.array([[p.x * w, p.y * h] for p in r.multi_hand_landmarks[0].landmark],
                      np.float32)
        world = None
        if getattr(r, "multi_hand_world_landmarks", None):
            world = np.array([[p.x, p.y, p.z]
                              for p in r.multi_hand_world_landmarks[0].landmark], np.float32)
        hand, score = "Unknown", float("nan")
        if r.multi_handedness:
            cl = r.multi_handedness[0].classification[0]
            hand, score = cl.label, float(cl.score)
        return {"px": px, "world": world, "hand": hand, "score": score}


# --------------------------------------------------------------------------------------
# 6 · Per-frame processing
# --------------------------------------------------------------------------------------

def out_of_plane_deg(world, i, j):
    """Angle of the bone away from the image plane, from MediaPipe world landmarks."""
    if world is None:
        return 0.0
    d = world[j] - world[i]
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return 90.0
    return float(np.degrees(np.arcsin(np.clip(abs(d[2]) / n, 0, 1))))


def process_frame(bgr, detector, cfg, source, frame_idx, dbg_dir=None):
    h, w = bgr.shape[:2]
    rec = {"source": source, "frame": frame_idx, "width": w, "height": h,
           "n_markers": 0, "scale_px_per_mm": np.nan, "map_method": "",
           "reproj_rms_px": np.nan, "hand": "", "hand_score": np.nan,
           "extrapolation": np.nan, "status": "ok"}

    squares = select_marker_cluster(detect_markers(bgr, cfg),
                                    tol=cfg.size_tol, prefer=MARKER_CLUSTER_PREFERENCE)
    rec["n_markers"] = len(squares)
    if len(squares) < cfg.min_markers:
        rec["status"] = "too_few_markers"
        return rec, None

    sides = np.array([s["side_px"] for s in squares], float)
    rec["scale_px_per_mm"] = float(np.median(sides)) / cfg.square_mm
    rec["marker_side_cv_pct"] = float(100 * sides.std() / sides.mean()) if len(sides) > 1 else 0.0

    hand = detector.detect(bgr)
    if hand is None:
        rec["status"] = "no_hand"
        return rec, None
    rec["hand"] = hand["hand"]
    rec["hand_score"] = hand["score"]
    if cfg.hand and hand["hand"].lower() != cfg.hand.lower():
        rec["status"] = "wrong_handedness"
        return rec, None
    lm = hand["px"]
    if lm[:, 0].min() < 2 or lm[:, 1].min() < 2 or lm[:, 0].max() > w - 3 or lm[:, 1].max() > h - 3:
        rec["status"] = "hand_clipped"
        return rec, None

    mc = np.array([s["center"] for s in squares], float)
    spread = float(np.median(np.linalg.norm(mc - mc.mean(0), axis=1))) if len(mc) > 1 else np.nan
    rec["extrapolation"] = (float(np.linalg.norm(lm.mean(0) - mc.mean(0)) / spread)
                            if spread and spread > 1 else np.nan)

    H, reproj = (None, float("nan"))
    if cfg.method in ("auto", "homography") and len(squares) >= 3:
        H, reproj = solve_plane_homography(squares, cfg.square_mm, refine=not cfg.no_refine)
    rec["reproj_rms_px"] = reproj

    lengths, tilts = [], []
    if H is not None and np.isfinite(reproj) and reproj <= cfg.max_reproj_px:
        rec["map_method"] = "homography"
        paper = cv2.perspectiveTransform(
            lm.reshape(1, -1, 2).astype(np.float32), np.linalg.inv(H)).reshape(-1, 2)
        for i, j, _, _ in BONES:
            lengths.append(float(np.linalg.norm(paper[j] - paper[i])))
            tilts.append(out_of_plane_deg(hand["world"], i, j))
    else:
        rec["map_method"] = "scale_field"
        model = fit_scale_field(squares, cfg.square_mm)
        grid_coef = None
        if cfg.grid_field and len(squares) < 5:
            # The printed graph paper gives a DENSE perspective shape; the squares fix
            # the absolute multiplier.  Useful when only 1-2 squares are unoccluded.
            P, V = build_grid_field(bgr)
            if P is not None:
                coef, resid = fit_grid_shape(P, V)
                if resid < 0.06:
                    grid_coef = coef
                    rec["map_method"] = "scale_field+grid_shape"
                    rec["grid_shape_resid"] = float(resid)
        s_ref = eval_scale(model, mc.mean(0))
        r_ref = (float(grid_coef[0] + grid_coef[1:] @ mc.mean(0))
                 if grid_coef is not None else 1.0)
        for i, j, _, _ in BONES:
            mid = (lm[i] + lm[j]) / 2.0
            if grid_coef is not None and abs(r_ref) > 1e-6:
                r = float(grid_coef[0] + grid_coef[1:] @ mid)
                s = s_ref * r / r_ref
                if not np.isfinite(s) or s <= 0 or abs(s - s_ref) / s_ref > 0.35:
                    s = eval_scale(model, mid)
            else:
                s = eval_scale(model, mid)
            lengths.append(float(np.linalg.norm(lm[j] - lm[i]) / s))
            tilts.append(out_of_plane_deg(hand["world"], i, j))

    if dbg_dir is not None:
        vis = bgr.copy()
        for s in squares:
            cv2.polylines(vis, [s["corners"].astype(np.int32)], True, (0, 0, 255), 3)
        for i, j, _, _ in BONES:
            cv2.line(vis, tuple(lm[i].astype(int)), tuple(lm[j].astype(int)), (0, 255, 0), 2)
        for p in lm:
            cv2.circle(vis, tuple(p.astype(int)), 4, (255, 0, 0), -1)
        cv2.putText(vis, f"{len(squares)} markers  {rec['map_method']}  "
                         f"{rec['scale_px_per_mm']:.2f} px/mm", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        Path(dbg_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(dbg_dir) / f"{Path(source).stem}_{frame_idx:05d}.jpg"),
                    cv2.resize(vis, (w // 3, h // 3)))

    return rec, {"lengths": np.array(lengths), "tilts": np.array(tilts),
                 "landmarks_px": lm}


# --------------------------------------------------------------------------------------
# 7 · Statistics
# --------------------------------------------------------------------------------------

def robust_stats(v, mad_z=3.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n0 = len(v)
    if n0 == 0:
        return {"n_raw": 0, "n_used": 0}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    keep = (np.abs(0.6745 * (v - med) / mad) <= mad_z) if mad > 1e-9 else np.ones(n0, bool)
    k = v[keep]
    if len(k) < 3:
        k = v
        keep = np.ones(n0, bool)
    mean = float(k.mean())
    std = float(k.std(ddof=1)) if len(k) > 1 else 0.0
    return {
        "n_raw": n0, "n_used": int(len(k)), "n_rejected": int(n0 - keep.sum()),
        "mean_mm": mean, "median_mm": float(np.median(k)),
        "std_mm": std, "var_mm2": float(std ** 2),
        "cv_pct": float(100 * std / mean) if mean > 0 else np.nan,
        "mad_mm": float(np.median(np.abs(k - np.median(k)))),
        "min_mm": float(k.min()), "max_mm": float(k.max()),
        "p5_mm": float(np.percentile(k, 5)), "p95_mm": float(np.percentile(k, 95)),
        "iqr_mm": float(np.percentile(k, 75) - np.percentile(k, 25)),
        "sem_mm": float(std / math.sqrt(len(k))) if len(k) > 1 else 0.0,
        "ci95_lo_mm": float(mean - 1.96 * std / math.sqrt(len(k))) if len(k) > 1 else mean,
        "ci95_hi_mm": float(mean + 1.96 * std / math.sqrt(len(k))) if len(k) > 1 else mean,
    }


def fit_parallax(L, D, h_max=45.0):
    """
    L: (F, B) measured lengths (mm, NaN allowed).  D: (F,) camera->paper distance (mm).
    Model  L_meas = L_true * D / (D - h).  Grid-search the single global h.
    Returns (h_mm, L_true (B,), sse_curve).
    """
    D = np.asarray(D, float)
    ok = np.isfinite(D) & (D > 0)
    if ok.sum() < 4:
        return None
    L, D = L[ok], D[ok]
    hs = np.linspace(0.0, min(h_max, 0.5 * float(D.min())), 181)
    sse, Lts = [], []
    for h in hs:
        k = D / (D - h)
        num = np.nansum(L * k[:, None], axis=0)
        den = np.nansum(np.where(np.isfinite(L), (k ** 2)[:, None], np.nan), axis=0)
        Lt = num / np.where(den <= 0, np.nan, den)
        r = L - Lt[None, :] * k[:, None]
        sse.append(float(np.nansum(r ** 2)))
        Lts.append(Lt)
    i = int(np.argmin(sse))
    return {"h_mm": float(hs[i]), "L_true_mm": Lts[i], "sse": np.array(sse),
            "h_grid": hs, "D_range_pct": float(100 * (D.max() - D.min()) / D.mean())}


def focal_px_from_exif(path, width, height):
    try:
        from PIL import Image, ExifTags
        img = Image.open(path)
        ex = img.getexif()
        if not ex:
            return None
        tag = {v: k for k, v in ExifTags.TAGS.items()}.get("FocalLengthIn35mmFilm")
        f35 = ex.get(tag)
        if not f35:
            return None
        return float(f35) / 36.0 * float(max(width, height))
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# 8 · Input enumeration + main
# --------------------------------------------------------------------------------------

def gather_inputs(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted([q for q in p.rglob("*")
                             if q.suffix.lower() in IMAGE_EXT | VIDEO_EXT])
        elif p.exists():
            files.append(p)
    return files


def iter_frames(path, frame_step, max_frames):
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        im = cv2.imread(str(path))
        if im is not None:
            yield 0, im
        return
    cap = cv2.VideoCapture(str(path))
    idx = kept = 0
    while cap.isOpened() and kept < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            kept += 1
            yield idx, fr
        idx += 1
    cap.release()


def make_plots(df, stats, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fmap = {b[2]: b[3] for b in BONES}
    order = [n for n in BONE_NAMES if n in set(df["bone"])]
    data = [df.loc[df.bone == n, "length_mm"].values for n in order]
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=True)
    for patch, n in zip(bp["boxes"], order):
        patch.set_facecolor(FINGER_COLORS[fmap[n]])
    ax.set_ylabel("bone length (mm)")
    ax.set_title(f"Per-bone length distribution — {MARKER_KIND}")
    plt.xticks(rotation=70, ha="right"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(Path(outdir) / "bone_length_distributions.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(stats["bone"], stats["cv_pct"],
           color=[FINGER_COLORS[fmap[n]] for n in stats["bone"]])
    ax.set_ylabel("CV (%)"); ax.set_title("Measurement variability per bone")
    plt.xticks(rotation=70, ha="right"); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(Path(outdir) / "bone_length_cv.png", dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="directory and/or files")
    ap.add_argument("--out", default="calib_filled")
    ap.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM)
    ap.add_argument("--edge-bias-mm", type=float, default=0.0,
                    help="subtract this from --square-mm (e.g. ink bleed beyond the drawn box)")
    ap.add_argument("--model", default=None, help="path to hand_landmarker.task")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--hand", default=None, choices=[None, "Left", "Right"],
                    help="keep only this handedness")
    ap.add_argument("--method", default="auto", choices=["auto", "homography", "scale_field"])
    ap.add_argument("--no-refine", action="store_true", help="skip SciPy homography refinement")
    ap.add_argument("--min-markers", type=int, default=2)
    ap.add_argument("--max-reproj-px", type=float, default=6.0)
    ap.add_argument("--size-tol", type=float, default=0.18)
    ap.add_argument("--max-tilt-deg", type=float, default=22.0,
                    help="reject bones tilted more than this out of the paper plane")
    ap.add_argument("--frame-step", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=60, help="per video")
    ap.add_argument("--parallax", action="store_true",
                    help="solve hand-plane height from multi-height shots")
    ap.add_argument("--focal-px", type=float, default=None, help="override EXIF focal length")
    ap.add_argument("--hand-thickness-mm", type=float, default=None,
                    help="apply this fixed height correction instead of solving for it")
    ap.add_argument("--grid-field", action="store_true",
                    help="EXPERIMENTAL: printed-grid pitch for perspective shape when <5 squares. Unreliable on JPEGs (8x8 block artifacts can dominate the spectrum) - verify grid_shape_resid in frame_report.csv before trusting it.")
    ap.add_argument("--grid-check", action="store_true",
                    help="EXPERIMENTAL diagnostic only, never changes the measured lengths: reports printed-grid pitch vs marker scale.")
    ap.add_argument("--grid-pitch-mm", type=float, default=1.0)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--debug", action="store_true", help="write annotated overlays")
    add_marker_args(ap)
    cfg = ap.parse_args()
    cfg.square_mm = cfg.square_mm - cfg.edge_bias_mm

    outdir = Path(cfg.out); outdir.mkdir(parents=True, exist_ok=True)
    dbg = outdir / "debug" if cfg.debug else None

    files = gather_inputs(cfg.input)
    if not files:
        print("No input images/videos found.", file=sys.stderr); sys.exit(1)
    print(f"[{MARKER_KIND}] {len(files)} input file(s); square = {cfg.square_mm:.2f} mm")

    det = HandDetector(cfg.model, cfg.min_conf)
    print(f"MediaPipe backend: {det.backend}")

    rows, frecs, Lrows, Drows, grid_rows = [], [], [], [], []
    for f in files:
        for idx, frame in iter_frames(f, cfg.frame_step, cfg.max_frames):
            rec, res = process_frame(frame, det, cfg, str(f), idx, dbg)
            frecs.append(rec)
            if res is None:
                continue
            for bi, (i, j, name, finger) in enumerate(BONES):
                tilt = res["tilts"][bi]
                rows.append({
                    "source": f.name, "frame": idx, "bone": name, "finger": finger,
                    "length_mm": res["lengths"][bi], "tilt_deg": tilt,
                    "valid": bool(tilt <= cfg.max_tilt_deg),
                    "scale_px_per_mm": rec["scale_px_per_mm"],
                    "map_method": rec["map_method"], "n_markers": rec["n_markers"],
                })
            Lv = res["lengths"].copy()
            Lv[res["tilts"] > cfg.max_tilt_deg] = np.nan
            Lrows.append(Lv)
            fpx = cfg.focal_px or focal_px_from_exif(f, rec["width"], rec["height"])
            Drows.append(fpx / rec["scale_px_per_mm"] if fpx else np.nan)
            if cfg.grid_check and f.suffix.lower() in IMAGE_EXT:
                P, V = build_grid_field(frame)
                if P is not None:
                    grid_rows.append({"source": f.name,
                                      "grid_px_per_mm": float(np.median(V)) / cfg.grid_pitch_mm,
                                      "square_px_per_mm": rec["scale_px_per_mm"],
                                      "n_patches": int(len(V))})

    if not rows:
        print("No usable frames. Check marker colour thresholds (--hue-lo/--hue-hi) "
              "and run with --debug.", file=sys.stderr)
        pd.DataFrame(frecs).to_csv(outdir / "frame_report.csv", index=False)
        sys.exit(2)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_frame_bone_lengths_mm.csv", index=False)
    pd.DataFrame(frecs).to_csv(outdir / "frame_report.csv", index=False)

    use = df[df.valid]
    stats = []
    for i, j, name, finger in BONES:
        s = robust_stats(use.loc[use.bone == name, "length_mm"].values)
        s.update({"bone": name, "finger": finger, "j_from": i, "j_to": j})
        stats.append(s)
    sdf = pd.DataFrame(stats)
    cols = ["bone", "finger", "j_from", "j_to", "n_raw", "n_used", "n_rejected",
            "mean_mm", "median_mm", "std_mm", "var_mm2", "cv_pct", "mad_mm",
            "sem_mm", "ci95_lo_mm", "ci95_hi_mm", "min_mm", "max_mm",
            "p5_mm", "p95_mm", "iqr_mm"]
    sdf = sdf[[c for c in cols if c in sdf.columns]]
    ref = float(sdf.loc[sdf.bone == BONE_NAMES[MIDDLE_MCP_PIP_BONE_IDX], "median_mm"].iloc[0])
    sdf["normalized"] = sdf["median_mm"] / ref
    sdf.to_csv(outdir / "bone_length_stats.csv", index=False)

    summary = {
        "marker_kind": MARKER_KIND, "square_mm": cfg.square_mm,
        "n_files": len(files), "n_frames_seen": len(frecs),
        "n_frames_used": int(df.groupby(["source", "frame"]).ngroups),
        "mediapipe_backend": det.backend,
        "map_methods": df.map_method.value_counts().to_dict(),
        "mean_cv_pct": float(sdf.cv_pct.mean()),
        "reference_bone": BONE_NAMES[MIDDLE_MCP_PIP_BONE_IDX], "reference_mm": ref,
    }

    if grid_rows:
        gdf = pd.DataFrame(grid_rows)
        gdf["ratio"] = gdf.grid_px_per_mm / gdf.square_px_per_mm
        gdf.to_csv(outdir / "grid_cross_check.csv", index=False)
        summary["grid_vs_square_scale_ratio"] = float(gdf.ratio.median())

    lengths_mm = dict(zip(sdf.bone, sdf.median_mm))
    if cfg.parallax or cfg.hand_thickness_mm is not None:
        L = np.array(Lrows, float)
        D = np.array(Drows, float)
        if cfg.hand_thickness_mm is not None and np.isfinite(D).sum() >= 1:
            h = cfg.hand_thickness_mm
            k = np.nanmedian(D / (D - h))
            lengths_mm = {b: v / k for b, v in lengths_mm.items()}
            summary["parallax"] = {"mode": "fixed", "h_mm": h, "magnification": float(k)}
        else:
            fp = fit_parallax(L, D)
            if fp is None:
                summary["parallax"] = {"mode": "failed",
                                       "reason": "no EXIF focal length / too few frames"}
            else:
                lengths_mm = {n: float(v) for n, v in zip(BONE_NAMES, fp["L_true_mm"])}
                summary["parallax"] = {"mode": "solved", "h_mm": fp["h_mm"],
                                       "D_range_pct": fp["D_range_pct"],
                                       "warning": ("camera-distance range < 15%; h is weakly "
                                                   "identified" if fp["D_range_pct"] < 15 else "")}
                sdf["parallax_corrected_mm"] = [lengths_mm[b] for b in sdf.bone]
                sdf.to_csv(outdir / "bone_length_stats.csv", index=False)

    json.dump({"unit": "mm", "bone_lengths_mm": lengths_mm,
               "bone_lengths_m": {k: v / 1000.0 for k, v in lengths_mm.items()},
               "bone_order": BONE_NAMES,
               "normalized_to_middle_mcp_pip": dict(zip(sdf.bone, sdf.normalized))},
              open(outdir / "bone_lengths.json", "w"), indent=2)
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)

    if cfg.plots:
        make_plots(use, sdf, outdir)

    print("\n=== bone lengths (mm) ===")
    print(sdf[["bone", "n_used", "median_mm", "mean_mm", "std_mm",
               "var_mm2", "cv_pct"]].to_string(index=False,
                                               float_format=lambda x: f"{x:8.3f}"))
    print(f"\nframes used: {summary['n_frames_used']}/{summary['n_frames_seen']}   "
          f"mean CV: {summary['mean_cv_pct']:.2f}%")
    print(f"written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
