"""
anatomical_angles_v12.py
========================
Corrected joint-angle system for the FreiHAND x MediaPipe hand-angle pipeline.

WHY THIS REPLACES THE 19-ANGLE SET
----------------------------------
The old `*_Abd` angles measured the angle between adjacent fingers' WRIST->MCP
vectors -- the *metacarpal fan*. Metacarpals 2-5 are bound near-rigidly by the
deep transverse metacarpal ligament, so that quantity barely varies with pose.
It carries almost no information about finger spread, which is why FK could not
reproduce abduction and fell back on `gt_mcp_dir` anchor vectors.

True MCP abduction is per-joint: the deviation of the proximal phalanx from its
OWN metacarpal's long axis, in the palm plane, about an anterior-posterior axis
through the metacarpal head.

All flexion angles here are SIGNED (arccos -> atan2), so hyperextension no
longer folds onto the flexion branch. Signed flexion + signed abduction makes
the angle -> direction map bijective, which the EKF measurement function needs.

ANGLE SET
---------
ARTICULATION_ANGLES (21) -- the model targets / EKF state:
    Index/Middle/Ring/Pinky : MCP_Abd, MCP_Flex, PIP_Flex, DIP_Flex   (4x4 = 16)
    Thumb                   : CMC_Abd, CMC_Flex, MCP_Abd, MCP_Flex, IP_Flex  (5)

Thumb_MCP_Abd was briefly excluded as a hinge; measurement showed that forcing
it to zero biases the downstream Thumb_IP_Flex by up to 9 deg at 45 deg of true
thumb MCP abduction, so it is RESTORED. Set INCLUDE_THUMB_MCP_ABD = False to
reproduce the 20-angle variant.

FAN_ANGLE_NAMES (4) -- palm shape, NOT model targets:
    Thumb_Index_Fan, Index_Middle_Fan, Middle_Ring_Fan, Ring_Pinky_Fan
These are the OLD abduction angles, renamed to what they actually measure. They
are near-constant within a subject, so they belong in per-subject CALIBRATION
(alongside bone lengths), not in the filter state. They are still computed and
stored for FK and for analysis.

CONVENTIONS
-----------
Local frame at every joint, columns [x_hat, y_hat, z_hat]:
    y_hat = parent bone direction (the neutral / zero-pose direction)
    z_hat = palmar normal         (flexion carries the bone toward +z)
    x_hat = y_hat x z_hat         (radial, i.e. thumb-ward, for a right hand)
Signs:
    flex > 0 -> flexion toward palm;  flex < 0 -> hyperextension
    abd  > 0 -> radial deviation (toward the thumb side)
Clinical "abduction" means away from the middle finger, so clinical abduction
= +abd for index but -abd for ring/pinky. Store the RAW signed radial value and
apply that flip only at report time -- baking per-finger sign flips into the
stored representation is a reliable source of FK bugs.

Joint rotation: R_local = M_x(flex) @ M_z(abd), applied to the parent frame to
produce the child frame, so frames chain exactly and FK round-trips to machine
precision (validated at ~5e-18 m).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Skeleton (MediaPipe / FreiHAND share this ordering -- no remap needed)
# ---------------------------------------------------------------------------
WRIST = 0
FINGER_CHAINS = {
    "Thumb":  (1, 2, 3, 4),      # CMC, MCP, IP, Tip
    "Index":  (5, 6, 7, 8),      # MCP, PIP, DIP, Tip
    "Middle": (9, 10, 11, 12),
    "Ring":   (13, 14, 15, 16),
    "Pinky":  (17, 18, 19, 20),
}
LONG_FINGERS = ("Index", "Middle", "Ring", "Pinky")
MCP_IDS = {"Thumb": 1, "Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}

BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
REF_BONE = (9, 10)   # Middle MCP->PIP, the scale-normalisation reference

# --- the 20 model-target angles, in canonical order ------------------------
ARTICULATION_ANGLES = (
    [f"{f}_MCP_Abd" for f in LONG_FINGERS]
    + [f"{f}_MCP_Flex" for f in LONG_FINGERS]
    + [f"{f}_PIP_Flex" for f in LONG_FINGERS]
    + [f"{f}_DIP_Flex" for f in LONG_FINGERS]
    + ["Thumb_CMC_Abd", "Thumb_CMC_Flex", "Thumb_MCP_Abd",
       "Thumb_MCP_Flex", "Thumb_IP_Flex"]
)
INCLUDE_THUMB_MCP_ABD = True
if not INCLUDE_THUMB_MCP_ABD:
    ARTICULATION_ANGLES = [a for a in ARTICULATION_ANGLES if a != "Thumb_MCP_Abd"]
FAN_ANGLE_NAMES = ["Thumb_Index_Fan", "Index_Middle_Fan",
                   "Middle_Ring_Fan", "Ring_Pinky_Fan"]
ALL_V12_NAMES = ARTICULATION_ANGLES + FAN_ANGLE_NAMES

assert len(ARTICULATION_ANGLES) == 20 + int(INCLUDE_THUMB_MCP_ABD)

# Per-finger joint layout: (label, is_hinge). Hinge -> abduction forced to 0.
JOINT_LAYOUT = {
    "Thumb": [("Thumb_CMC", False),
              ("Thumb_MCP", not INCLUDE_THUMB_MCP_ABD),
              ("Thumb_IP", True)],
    **{f: [(f"{f}_MCP", False), (f"{f}_PIP", True), (f"{f}_DIP", True)]
       for f in LONG_FINGERS},
}

_MIN_BONE = 1e-9   # unit-agnostic guard against duplicated/degenerate landmarks


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.nan_to_num(np.asarray(v, dtype=np.float64))
    n = np.linalg.norm(v)
    if n < _MIN_BONE:
        return np.array([0.0, 1.0, 0.0])
    return v / n


def _orthonormalize(y_hat, z_raw):
    """Gram-Schmidt z against y, then x = y cross z. Columns [x, y, z]."""
    y_hat = _unit(y_hat)
    z_raw = np.nan_to_num(np.asarray(z_raw, dtype=np.float64))
    z_hat = _unit(z_raw - np.dot(z_raw, y_hat) * y_hat)
    x_hat = _unit(np.cross(y_hat, z_hat))
    return np.column_stack([x_hat, y_hat, z_hat])


def _M_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _M_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def palm_frame(kps, hand="right"):
    """
    Palm reference frame, columns [x, y, z].
      y = wrist -> middle MCP
      z = palmar normal (SVD plane fit over wrist + 5 MCPs)
      x = y cross z  (radial for a right hand)

    SVD returns the normal up to an arbitrary sign, so it is disambiguated
    geometrically: cross(wrist->index_MCP, wrist->pinky_MCP) points palmar for
    a right hand by chirality, dorsal for a left hand.
    """
    kps = np.nan_to_num(np.asarray(kps, dtype=np.float64))
    w = kps[WRIST]
    y_hat = _unit(kps[MCP_IDS["Middle"]] - w)

    pts = np.stack([w, kps[5], kps[9], kps[13], kps[17]])
    centred = pts - pts.mean(axis=0)
    n_svd = np.linalg.svd(centred, full_matrices=False)[2][-1]

    ref = np.cross(kps[5] - w, kps[17] - w)
    if str(hand).lower().startswith("l"):
        ref = -ref
    if np.dot(n_svd, ref) < 0:
        n_svd = -n_svd
    return _orthonormalize(y_hat, n_svd)


# ---------------------------------------------------------------------------
# Core decomposition / composition
# ---------------------------------------------------------------------------
def decompose(frame, child_dir, hinge=False):
    """
    Express `child_dir` in local `frame` (columns [x,y,z]) as (abd, flex).

        abd  = arcsin(p . x)         signed, radial-positive.
               Safe because |abd| never approaches 90 deg anatomically.
        flex = atan2(p . z, p . y)   signed, full range.
               atan2 (not arccos/arcsin) because flexion DOES cross 90 deg --
               arcsin would fold 120 deg back onto 60 deg, i.e. every fist.

    Returns (abd, flex, child_frame, off_axis), all radians. `off_axis` is the
    |p . x| a hinge had to discard -- a free per-frame quality signal for
    MediaPipe noise and hinge-model violation.
    """
    p = _unit(child_dir)
    lx, ly, lz = frame.T @ p

    if hinge:
        abd, off_axis = 0.0, float(abs(lx))
    else:
        abd, off_axis = float(np.arcsin(np.clip(lx, -1.0, 1.0))), 0.0
    flex = float(np.arctan2(lz, ly))

    return abd, flex, frame @ _M_x(flex) @ _M_z(abd), off_axis


def compose(frame, abd, flex):
    """Inverse of `decompose`: (abd, flex) -> unit child direction + frame."""
    child_frame = frame @ _M_x(flex) @ _M_z(abd)
    return child_frame[:, 1].copy(), child_frame


# ---------------------------------------------------------------------------
# Extraction: 21 keypoints -> 24 angles (20 articulation + 4 fan)
# ---------------------------------------------------------------------------
def compute_angles_v12(kps, hand="right", degrees=True, return_diagnostics=False):
    """
    kps  : (21, 3) landmarks, any consistent unit (result is scale-invariant).
    hand : 'right' or 'left'.

    Returns dict name -> angle, ordered as ALL_V12_NAMES.
    The result is invariant to global rotation and translation of `kps`, so GT
    (camera frame) and MediaPipe (wrist-relative) need no alignment.
    """
    kps = np.nan_to_num(np.asarray(kps, dtype=np.float64))
    P = palm_frame(kps, hand=hand)
    w = kps[WRIST]
    out, diag = {}, {}

    # --- 20 articulation angles ---------------------------------------------
    for finger, chain in FINGER_CHAINS.items():
        # Neutral for the first joint = this finger's OWN metacarpal.
        F = _orthonormalize(kps[chain[0]] - w, P[:, 2])
        for k, (label, hinge) in enumerate(JOINT_LAYOUT[finger]):
            child_dir = kps[chain[k + 1]] - kps[chain[k]]
            abd, flex, F, off = decompose(F, child_dir, hinge=hinge)
            out[f"{label}_Flex"] = flex
            if hinge:
                diag[f"{label}_off_axis"] = off
            else:
                out[f"{label}_Abd"] = abd

    # --- 4 fan (palm-shape) angles ------------------------------------------
    def _azimuth(idx):
        v = P.T @ _unit(kps[idx] - w)
        return float(np.arctan2(v[0], v[1]))      # +x = radial

    az = {f: _azimuth(i) for f, i in MCP_IDS.items()}
    out["Thumb_Index_Fan"] = az["Thumb"] - az["Index"]
    out["Index_Middle_Fan"] = az["Index"] - az["Middle"]
    out["Middle_Ring_Fan"] = az["Middle"] - az["Ring"]
    out["Ring_Pinky_Fan"] = az["Ring"] - az["Pinky"]

    out = {k: out[k] for k in ALL_V12_NAMES}
    if degrees:
        out = {k: float(np.degrees(v)) for k, v in out.items()}
        diag = {k: float(np.degrees(np.arcsin(min(v, 1.0)))) for k, v in diag.items()}
    return (out, diag) if return_diagnostics else out


def compute_angles_batch(kps_batch, hand="right", degrees=True, progress=None):
    """(N,21,3) -> (N,24) float32 array ordered as ALL_V12_NAMES."""
    kps_batch = np.asarray(kps_batch)
    n = len(kps_batch)
    out = np.zeros((n, len(ALL_V12_NAMES)), dtype=np.float32)
    it = range(n) if progress is None else progress(range(n))
    for i in it:
        d = compute_angles_v12(kps_batch[i], hand=hand, degrees=degrees)
        out[i] = [d[k] for k in ALL_V12_NAMES]
    return out


# ---------------------------------------------------------------------------
# Forward kinematics: angles + bone lengths + palm template -> 21 keypoints
# ---------------------------------------------------------------------------
def forward_kinematics_v12(angles, bone_lengths, palm=None, degrees=True,
                           wrist_origin=None, fan_angles=None):
    """
    angles       : dict with the 20 articulation angles (fan angles may also be
                   present; `fan_angles` overrides them if given).
    bone_lengths : dict (parent_idx, child_idx) -> length, all 20 bones.
    palm         : 3x3 palm frame, columns [x,y,z]. Default = canonical.
    fan_angles   : optional dict of the 4 fan angles -- pass the per-subject
                   CALIBRATED values here rather than per-frame estimates.

    NOTE: metacarpals are placed IN the palm plane. Palm arching / thumb
    opposition out of that plane is not represented by any angle in this set;
    supply it via a calibrated palm template if you need it.
    """
    A = {k: (np.radians(v) if degrees else v) for k, v in angles.items()}
    if fan_angles is not None:
        for k, v in fan_angles.items():
            A[k] = np.radians(v) if degrees else v
    if palm is None:
        palm = _orthonormalize(np.array([0.0, 1.0, 0.0]),
                               np.array([0.0, 0.0, 1.0]))
    w = np.zeros(3) if wrist_origin is None else np.asarray(wrist_origin, float)
    kps = np.zeros((21, 3))
    kps[WRIST] = w

    az = {"Middle": 0.0}
    az["Index"] = az["Middle"] + A["Index_Middle_Fan"]
    az["Ring"] = az["Middle"] - A["Middle_Ring_Fan"]
    az["Pinky"] = az["Ring"] - A["Ring_Pinky_Fan"]
    az["Thumb"] = az["Index"] + A["Thumb_Index_Fan"]

    for finger, chain in FINGER_CHAINS.items():
        a = az[finger]
        ray = palm @ np.array([np.sin(a), np.cos(a), 0.0])
        kps[chain[0]] = w + bone_lengths[(WRIST, MCP_IDS[finger])] * ray

        F = _orthonormalize(ray, palm[:, 2])
        for k, (label, hinge) in enumerate(JOINT_LAYOUT[finger]):
            abd = 0.0 if hinge else A[f"{label}_Abd"]
            direction, F = compose(F, abd, A[f"{label}_Flex"])
            kps[chain[k + 1]] = kps[chain[k]] + \
                bone_lengths[(chain[k], chain[k + 1])] * direction
    return kps


# ---------------------------------------------------------------------------
# Utilities / validation
# ---------------------------------------------------------------------------
def wrap_deg(d):
    """Map an angle DIFFERENCE into (-180, 180].

    atan2-based flexion lives on a circle, so a GT of 179 deg and a prediction
    of -179 deg are 2 deg apart, not 358. Use this for every loss and metric on
    signed angles -- otherwise a handful of wrapped samples dominate the
    gradient. Observed on real data: Thumb_IP_Flex spans -177.9 to 160.7 deg.
    """
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def unwrap_to_reference(angles_deg, reference_deg):
    """Shift each angle by +-360 to sit closest to a reference (e.g. the GT or
    the per-angle median), removing branch-cut discontinuities from a column."""
    a = np.asarray(angles_deg, dtype=np.float64)
    return a - 360.0 * np.round((a - np.asarray(reference_deg)) / 360.0)


def bone_lengths_from_kps(kps):
    kps = np.asarray(kps, float)
    return {b: float(np.linalg.norm(kps[b[1]] - kps[b[0]])) for b in BONES}


def round_trip_error(kps, hand="right"):
    """kps -> angles -> kps. Residual = hinge-model violation + palm arch only;
    the frame algebra itself is exact."""
    kps = np.asarray(kps, float)
    ang = compute_angles_v12(kps, hand=hand)
    rec = forward_kinematics_v12(ang, bone_lengths_from_kps(kps),
                                 palm=palm_frame(kps, hand=hand),
                                 wrist_origin=kps[WRIST])
    return float(np.mean(np.linalg.norm(rec - kps, axis=1))), rec


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bl = {b: 0.04 for b in BONES}
    ang = {n: float(rng.uniform(-20, 20)) for n in ALL_V12_NAMES}
    for n in ALL_V12_NAMES:
        if n.endswith("_Flex"):
            ang[n] = float(rng.uniform(-10, 80))
    kps = forward_kinematics_v12(ang, bl)
    err, _ = round_trip_error(kps)
    rec = compute_angles_v12(kps)
    worst = max(abs(rec[n] - ang[n]) for n in ALL_V12_NAMES)
    print(f"n articulation angles          : {len(ARTICULATION_ANGLES)}")
    print(f"wrap_deg(179 - -179)           : {wrap_deg(179 - -179):.1f} deg")
    print(f"synthetic FK round-trip error  : {err:.3e} m")
    print(f"worst angle recovery error     : {worst:.3e} deg")
