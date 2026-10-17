#!/usr/bin/env python3
"""
calib_adapter.py — bonelengthcalculation/ output  ->  hand_ekf.PersonalizedHandModel

    python calib_adapter.py \
        --primary  bonelengthcalculation/calib_filled_tilt35 \
        --fallback bonelengthcalculation/calib_filled \
        --out calib/hand_calibration_ekf.json --report

Then:
    model = PersonalizedHandModel.from_calibration(
        "calib/hand_calibration_ekf.json", angle_names)

What this handles
-----------------
1. NAME MAPPING. The calibration writes `Thumb_Wrist_CMC`; hand_ekf expects
   `Thumb_Wrist-CMC`. The two `bone_order` lists are positionally identical
   (verified all 20), so the mapping is by INDEX, not by string munging.

2. NaN MERGE. `calib_filled` runs a 25 deg tilt gate, which nulls the four
   wrist->MCP metacarpals (their median out-of-plane tilt is 28-30 deg, from the
   palm arch). `calib_filled_tilt35` admits them at a 35 deg gate. Use tilt35 as
   primary and calib_filled as fallback: identical phalanx values, plus the
   metacarpals.

3. TILT FORESHORTENING — read the note below before enabling.

4. PALM DIRECTIONS. Not produced by the calibration script. See `--palm-dirs`
   and the patch in `PALM_DIRS_PATCH` at the bottom of this file.

THE TILT QUESTION
-----------------
The calibration measures each bone's PROJECTED length in the paper plane
(`norm(paper[j] - paper[i])`) and computes an out-of-plane tilt from MediaPipe's
world landmarks, but uses that tilt ONLY as a rejection gate. It never divides by
cos(tilt). So every bone under-reads by cos(tilt): ~0.2% on the flat mid-
phalanges, ~7% on the distal segments, ~14% on the metacarpals.

Correcting looks obviously right and measurably ISN'T. Cross-checked against
van der Hulst Table IV (phalange length as % of hand length):

    Middle finger      proximal  medial  distal
    van der Hulst        26.0     16.8    10.4
    projected (raw)      25.8     16.0    12.4   <- closer
    cos(tilt) corrected  24.0     14.9    12.4

The tilt estimate comes from MediaPipe's z, which is the least reliable axis —
the same premise that motivates the anisotropic R in the EKF. Correcting with a
biased tilt makes the RATIOS worse, and ratios are what FK uses (Umeyama absorbs
a uniform scale error, so only ratios matter). So `--tilt-correct` defaults OFF.

Enable it only if you re-measure tilt from something other than MediaPipe z.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# hand_ekf ordering. Index-aligned with the calibration's `bone_order`.
EKF_BONE_NAMES: Tuple[str, ...] = (
    "Thumb_Wrist-CMC", "Thumb_CMC-MCP", "Thumb_MCP-IP", "Thumb_IP-Tip",
    "Index_Wrist-MCP", "Index_MCP-PIP", "Index_PIP-DIP", "Index_DIP-Tip",
    "Middle_Wrist-MCP", "Middle_MCP-PIP", "Middle_PIP-DIP", "Middle_DIP-Tip",
    "Ring_Wrist-MCP", "Ring_MCP-PIP", "Ring_PIP-DIP", "Ring_DIP-Tip",
    "Pinky_Wrist-MCP", "Pinky_MCP-PIP", "Pinky_PIP-DIP", "Pinky_DIP-Tip",
)
METACARPAL_IDX = (0, 4, 8, 12, 16)          # wrist -> first child, per finger
REF_BONE_IDX = 9                            # Middle MCP->PIP. NOT 11.

# Fallback metacarpal fan (radians from the palm's +Y), used only when no
# measured palm directions are supplied. Nominal adult right hand.
NOMINAL_FAN_DEG = (-38.0, -13.0, 0.0, 13.0, 25.0)


def load_run(run_dir: str) -> Tuple[np.ndarray, List[str], dict]:
    """Read one calib_* directory. Returns (lengths_mm(20,), bone_order, summary)."""
    with open(os.path.join(run_dir, "bone_lengths.json")) as fh:
        bl = json.load(fh)
    order = bl["bone_order"]
    if len(order) != 20:
        raise ValueError(f"{run_dir}: expected 20 bones, got {len(order)}")
    L = np.array([bl["bone_lengths_mm"][b] for b in order], float)
    summary = {}
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            summary = json.load(fh)
    return L, order, summary


def median_tilts(run_dir: str, order: List[str],
                 max_tilt: float = 90.0) -> np.ndarray:
    """Median out-of-plane tilt per bone, degrees, from per_frame_bone_lengths_mm.csv."""
    path = os.path.join(run_dir, "per_frame_bone_lengths_mm.csv")
    if not os.path.exists(path):
        return np.full(20, np.nan)
    acc: Dict[str, List[float]] = {b: [] for b in order}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = row.get("tilt_deg", "")
            if t in ("", "nan") or row["bone"] not in acc:
                continue
            tv = float(t)
            if tv <= max_tilt:
                acc[row["bone"]].append(tv)
    return np.array([np.median(acc[b]) if acc[b] else np.nan for b in order])


def load_palm_dirs(path: Optional[str], lengths: np.ndarray) -> Tuple[np.ndarray, bool]:
    """(5,3) unit wrist->knuckle directions. Returns (dirs, measured?)."""
    if path and os.path.exists(path):
        with open(path) as fh:
            blob = json.load(fh)
        if "palm_dirs" in blob:
            P = np.asarray(blob["palm_dirs"], float).reshape(5, 3)
        elif "mcp_paper_mm" in blob:
            # paper-plane MCP positions relative to the wrist, z = 0
            M = np.asarray(blob["mcp_paper_mm"], float).reshape(5, 2)
            P = np.column_stack([M, np.zeros(5)])
        else:
            raise KeyError(f"{path}: need 'palm_dirs' or 'mcp_paper_mm'")
        return P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12), True

    fan = np.deg2rad(NOMINAL_FAN_DEG)
    P = np.stack([np.sin(fan), np.cos(fan), np.zeros(5)], axis=1)
    return P, False


def build(primary: str, fallback: Optional[str] = None,
          tilt_correct: bool = False, palm_dirs_path: Optional[str] = None,
          max_tilt: float = 35.0, mirror_to_right: bool = False,
          hand: str = "Right") -> dict:
    L, order, summary = load_run(primary)
    provenance = [os.path.basename(primary)] * 20

    if fallback:
        Lf, order_f, _ = load_run(fallback)
        if order_f != order:
            raise ValueError("bone_order differs between runs; refusing to merge")
        fill = ~np.isfinite(L) & np.isfinite(Lf)
        L = np.where(fill, Lf, L)
        for i in np.flatnonzero(fill):
            provenance[i] = os.path.basename(fallback)

    tilts = median_tilts(primary, order, max_tilt)
    if tilt_correct:
        c = np.cos(np.deg2rad(np.nan_to_num(tilts, nan=0.0)))
        L = L / np.where(c > 0.2, c, 1.0)

    missing = [order[i] for i in np.flatnonzero(~np.isfinite(L))]
    if missing:
        raise ValueError(
            f"still NaN after merge: {missing}\n"
            f"Re-run the calibration with a higher --max-tilt-deg, or pass "
            f"--fallback pointing at a run that has them.")

    P, measured = load_palm_dirs(palm_dirs_path, L)

    # Chirality. Bone LENGTHS are scalars and mirror trivially; palm DIRECTIONS
    # do not -- mirroring a hand negates x. The corrector was trained on one
    # handedness (see the checkpoint's config['hand']), and the v12 palm normal
    # is sign-disambiguated by chirality, so a left hand fed in raw produces
    # sign-flipped abduction throughout. Emitting the left calibration already
    # mirrored into right chirality means the whole downstream pipeline -- angle
    # extraction, corrector, FK -- stays right-handed, and the only runtime step
    # is negating x on the observed landmarks.
    if mirror_to_right:
        P = P * np.array([-1.0, 1.0, 1.0])

    return {
        "schema": "hand_calibration_ekf/v1",
        "unit": "mm",
        "hand": hand,
        "chirality": "Right" if (hand == "Right" or mirror_to_right) else hand,
        "mirrored_to_right": bool(mirror_to_right),
        "lengths_mm": {n: float(v) for n, v in zip(EKF_BONE_NAMES, L)},
        "palm_dirs": P.tolist(),
        "palm_dirs_measured": measured,
        "median_tilt_deg": {n: (None if not np.isfinite(t) else float(t))
                            for n, t in zip(EKF_BONE_NAMES, tilts)},
        "provenance": {n: p for n, p in zip(EKF_BONE_NAMES, provenance)},
        "tilt_corrected": bool(tilt_correct),
        "source_summary": summary,
        "hand_length_mm": float(L[8:12].sum()),
        "reference_bone": EKF_BONE_NAMES[REF_BONE_IDX],
        "reference_mm": float(L[REF_BONE_IDX]),
    }


def report(cal: dict) -> str:
    L = np.array(list(cal["lengths_mm"].values()))
    T = [cal["median_tilt_deg"][n] for n in cal["lengths_mm"]]
    H = cal["hand_length_mm"]
    out = [f"hand length (wrist->middle tip): {H:.1f} mm"
           f"   reference {cal['reference_bone']} = {cal['reference_mm']:.2f} mm",
           f"tilt-corrected: {cal['tilt_corrected']}   "
           f"palm dirs measured: {cal['palm_dirs_measured']}",
           "",
           f"{'bone':<20}{'mm':>8}{'/ref':>8}{'tilt':>7}  source"]
    for i, (n, v) in enumerate(cal["lengths_mm"].items()):
        t = T[i]
        out.append(f"{n:<20}{v:8.2f}{v / L[REF_BONE_IDX]:8.3f}"
                   f"{('  --' if t is None else f'{t:7.1f}')}  {cal['provenance'][n]}")

    # van der Hulst Table IV cross-check (% of hand length)
    vdh = {"Index": (23.5, 15.2, 9.3), "Middle": (26.0, 16.8, 10.4),
           "Ring": (22.2, 15.3, 9.7), "Pinky": (17.7, 10.8, 8.6)}
    out += ["", "phalange % of hand length vs van der Hulst Table IV:",
            f"{'finger':<8}{'prox':>16}{'med':>16}{'dist':>16}"]
    for fi, f in enumerate(("Index", "Middle", "Ring", "Pinky")):
        b = (fi + 1) * 4
        mine = [100 * L[b + k] / H for k in (1, 2, 3)]
        ref = vdh[f]
        cells = "".join(f"{m:9.1f} ({r:4.1f})" for m, r in zip(mine, ref))
        out.append(f"{f:<8}{cells}")
    out.append("  (yours, with van der Hulst in brackets)")
    return "\n".join(out)


def compare(a: dict, b: dict) -> str:
    """Diff two calibrations. Your own left and right hands should agree to a
    couple of percent -- real anatomical asymmetry is small, so a large gap
    means a measurement problem, not an unusual hand. This is the strongest
    independent check available on the whole calibration pipeline, because the
    two sets of photos share no frames."""
    La = np.array([a["lengths_mm"][n] for n in EKF_BONE_NAMES])
    Lb = np.array([b["lengths_mm"][n] for n in EKF_BONE_NAMES])
    d = 100.0 * (La - Lb) / Lb
    # scale-free comparison: ratios to the reference bone
    Ra, Rb = La / La[REF_BONE_IDX], Lb / Lb[REF_BONE_IDX]
    dr = 100.0 * (Ra - Rb) / Rb
    out = [f"{a.get('hand','A')} vs {b.get('hand','B')}",
           f"  hand length   {a['hand_length_mm']:.1f} vs {b['hand_length_mm']:.1f} mm"
           f"   ({100 * (a['hand_length_mm'] / b['hand_length_mm'] - 1):+.1f}%)",
           f"  mean |diff|   {np.abs(d).mean():.1f}%   "
           f"scale-free {np.abs(dr).mean():.1f}%",
           "",
           f"{'bone':<20}{'A':>9}{'B':>9}{'diff%':>8}{'ratio diff%':>13}"]
    for i, n in enumerate(EKF_BONE_NAMES):
        out.append(f"{n:<20}{La[i]:9.2f}{Lb[i]:9.2f}{d[i]:+8.1f}{dr[i]:+13.1f}")
    worst = int(np.argmax(np.abs(dr)))
    out += ["", f"  largest scale-free disagreement: {EKF_BONE_NAMES[worst]} "
                f"({dr[worst]:+.1f}%)",
            "  under ~3% scale-free is normal anatomical asymmetry;",
            "  a single bone far outside that is a landmark or tilt problem."]
    return "\n".join(out)


PALM_DIRS_PATCH = r'''
# ---------------------------------------------------------------------------
# Patch for calibrate_bones_filled_squares.py — dump paper-plane MCP directions
#
# The calibration already computes `paper` (landmarks warped into millimetre
# paper coordinates). It just never saves them. Right after
#
#     paper = cv2.perspectiveTransform(...).reshape(-1, 2)
#
# add:
#
#     rec["mcp_paper_mm"] = [(float(paper[j][0] - paper[0][0]),
#                             float(paper[j][1] - paper[0][1]))
#                            for j in (1, 5, 9, 13, 17)]
#
# then after the per-frame loop, write the median across frames:
#
#     import numpy as _np, json as _json
#     M = _np.array([r["mcp_paper_mm"] for r in records
#                    if r.get("status") == "ok" and "mcp_paper_mm" in r], float)
#     if len(M):
#         _json.dump({"mcp_paper_mm": _np.median(M, axis=0).tolist()},
#                    open(Path(cfg.out) / "palm_dirs.json", "w"), indent=2)
#
# Then:  python calib_adapter.py ... --palm-dirs calib_filled_tilt35/palm_dirs.json
#
# Without this the adapter falls back to a nominal anatomical fan, which biases
# every keypoint residual in the EKF's measurement block.
# ---------------------------------------------------------------------------
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--primary", required=True, help="calib_* dir (use the tilt35 run)")
    ap.add_argument("--fallback", default=None, help="second calib_* dir to fill NaNs")
    ap.add_argument("--palm-dirs", default=None, help="palm_dirs.json (see PALM_DIRS_PATCH)")
    ap.add_argument("--tilt-correct", action="store_true",
                    help="divide by cos(tilt). OFF by default — read the module docstring")
    ap.add_argument("--max-tilt", type=float, default=35.0)
    ap.add_argument("--hand", default="Right", choices=["Right", "Left"],
                    help="which hand this calibration is of")
    ap.add_argument("--mirror-to-right", action="store_true",
                    help="negate palm_dirs x so a Left calibration is usable in "
                         "the right-handed pipeline. Use for --hand Left.")
    ap.add_argument("--compare", default=None,
                    help="another hand_calibration_ekf.json to diff against "
                         "(e.g. compare your left and right hands)")
    ap.add_argument("--out", default="hand_calibration_ekf.json")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cal = build(args.primary, args.fallback, args.tilt_correct,
                args.palm_dirs, args.max_tilt, args.mirror_to_right, args.hand)
    with open(args.out, "w") as fh:
        json.dump(cal, fh, indent=2)
    print(f"wrote {args.out}")
    if args.report:
        print()
        print(report(cal))
    if args.compare:
        with open(args.compare) as fh:
            other = json.load(fh)
        print()
        print(compare(cal, other))
    if not cal["palm_dirs_measured"]:
        print("\n[warn] palm directions are a NOMINAL anatomical fan, not measured.")
        print("       See PALM_DIRS_PATCH in this file — it is ~6 lines.")


if __name__ == "__main__":
    main()
