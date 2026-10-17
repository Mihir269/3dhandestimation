"""
handedness.py — run the right-handed EKF pipeline on either hand

Why any of this is needed
-------------------------
Three separate things in the pipeline are chirality-dependent:

1. The v12 corrector was trained on ONE handedness (`ckpt['config']['hand']`).
   Feed it the other hand raw and the residual it predicts is meaningless.
2. The v12 palm normal is sign-disambiguated by chirality
   (`cross(wrist->index_MCP, wrist->pinky_MCP)` is palmar for a RIGHT hand),
   so every abduction angle flips sign on a left hand.
3. `palm_dirs` in the calibration are direction vectors, so they mirror.

Bone LENGTHS are scalars and don't care.

The fix is to pick one chirality -- right, matching the checkpoint -- and
normalize everything into it at a single point: negate x on the observed world
landmarks. Everything downstream (angle extraction, corrector, EKF, FK) then
runs unchanged. Only the rendered skeleton is mirrored back, for display.

⚠️ Do not confuse this with `run_hand_ekf.py --mirror`. That flag flips the
IMAGE for selfie webcams, which makes MediaPipe report the OPPOSITE handedness
for the same physical hand. The two interact: with --mirror on, your right hand
is reported as "Left". Trust `result.handedness`, not your own assumption about
which hand you raised, and verify once with the on-screen readout.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

MIRROR_X = np.array([-1.0, 1.0, 1.0])


def mirror_landmarks(kps: np.ndarray) -> np.ndarray:
    """(21,3) -> mirrored across the x axis. Its own inverse."""
    return np.asarray(kps, float).reshape(21, 3) * MIRROR_X


def normalize_to(kps: np.ndarray, observed_hand: str,
                 target_hand: str = "Right") -> np.ndarray:
    """Put landmarks into `target_hand` chirality. No-op when they match."""
    if kps is None:
        return None
    same = str(observed_hand).lower().startswith(str(target_hand).lower()[:1])
    return np.asarray(kps, float).reshape(21, 3) if same else mirror_landmarks(kps)


class HandednessRouter:
    """Holds one PersonalizedHandModel + one HandEKF per hand.

    Separate filters matter: switching hands mid-clip is a discontinuity, and a
    single filter would spend ~10 frames dragging its state across. Separate
    ones each keep their own history, so alternating hands costs nothing.

    Usage per frame:
        out = router.step(kps_world, hand_label, dt)
        if out is not None:
            angles, skeleton_for_display = out["angles"], out["skeleton_display"]
    """

    def __init__(self, models: Dict[str, "object"], make_ekf,
                 target_hand: str = "Right"):
        """
        models     : {"Right": PersonalizedHandModel, "Left": ...}. A single-entry
                     dict is fine -- unseen hands fall back to the one present,
                     with a one-time warning, since the two hands of one person
                     differ by only a couple of percent.
        make_ekf   : callable(model) -> HandEKF
        target_hand: chirality the corrector was trained on. Read it from
                     ckpt['config']['hand'] rather than assuming.
        """
        if not models:
            raise ValueError("need at least one PersonalizedHandModel")
        self.models = dict(models)
        self.target = target_hand
        self.ekfs = {h: make_ekf(m) for h, m in self.models.items()}
        self._warned: set = set()
        self.last_hand: Optional[str] = None
        self.last_model: Optional[str] = None

    def _pick(self, hand: str) -> str:
        h = "Left" if str(hand).lower().startswith("l") else "Right"
        if h in self.models:
            return h
        other = next(iter(self.models))
        if h not in self._warned:
            self._warned.add(h)
            print(f"[handedness] no calibration for {h}; using the {other} model. "
                  f"Bone lengths will be off by the L/R asymmetry (~1-3%).")
        return other

    def step(self, kps_world: Optional[np.ndarray], hand: Optional[str],
             corrected_angles: Optional[np.ndarray], dt: float) -> Optional[dict]:
        if hand is None:
            hand = self.last_hand or self.target
        key = self._pick(hand)
        ekf = self.ekfs[key]
        self.last_hand = hand
        self.last_model = key

        # single normalization point: everything past here is `target` chirality
        kps_n = normalize_to(kps_world, hand, self.target) if kps_world is not None else None
        kps_mm = kps_n * 1000.0 if kps_n is not None else None

        angles = ekf.step(corrected_angles, kps_mm, dt)
        skel = ekf.skeleton()

        # mirror back for display so the drawn hand matches the real one
        display = skel if str(hand).lower().startswith(self.target.lower()[:1]) \
            else mirror_landmarks(skel)

        return {
            "hand": hand, "model_used": key,
            "angles": angles, "angle_std": ekf.angle_std,
            "skeleton": skel, "skeleton_display": display,
            "resid_mm": ekf.last_resid_mm, "nis": ekf.last_nis,
            "align": ekf.last_align, "ekf": ekf,
        }


def landmarks_and_hand(result, frame_shape) -> Tuple[Optional[np.ndarray],
                                                     Optional[np.ndarray],
                                                     Optional[str], float]:
    """Pull (world_landmarks, pixel_landmarks, hand_label, score) from a
    MediaPipe HandLandmarkerResult. Returns (None, None, None, 0.0) on no detect."""
    if not getattr(result, "hand_world_landmarks", None):
        return None, None, None, 0.0
    w3 = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], float)
    h, w = frame_shape[:2]
    px = np.array([[p.x * w, p.y * h] for p in result.hand_landmarks[0]], float)
    label, score = None, 0.0
    hs = getattr(result, "handedness", None)
    if hs:
        cat = hs[0][0]
        label = cat.category_name
        score = float(getattr(cat, "score", 0.0))
    return w3, px, label, score
