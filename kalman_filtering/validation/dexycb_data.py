#!/usr/bin/env python3
"""
dexycb_data.py -- a compact DexYCB cache, so tuning and analysis need no images.

Running MediaPipe over the images is the only slow step, and the 12 GB subject
tars only fit in Colab. This module stores what the filters actually use, one
small npz per (sequence, camera), about 20-70 KB each:

    gt       (T,21,3) float32  GT joints, metres, camera frame, NaN = no hand
    gt_ang   (T,21)   float32  GT v12 angles (right chirality), NaN = no hand
    world    (T,21,3) float32  MediaPipe world landmarks, metres, NaN = no detection
    label    (T,)     int8     MediaPipe handedness 1=right 0=left -1=none
    side     str               'right' | 'left' (from meta.yml, ground truth)
    subject, name, serial      str

Everything downstream (validate_dexycb.py --compact, tune_dexycb.py --compact,
analyse_dexycb_frames.py) reads this folder only, so it runs on a laptop too.

Build (Colab, from extracted DexYCB):
    python validation/dexycb_data.py build --root /content/dexycb --out <compact dir> \
        --serials all --mp-cache <old mp_cache dir, reused if present>

Convert the subject-01 results zip you already have (no GT keypoints in it,
only GT angles, so PA-MPJPE and the oracle calibration are unavailable):
    python validation/dexycb_data.py from-results --results <dexycb results dir> --out <compact dir>

Inspect:
    python validation/dexycb_data.py info --compact <compact dir>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, HERE)

import hand_angles_v2 as HA  # noqa: E402
import validate_dexycb as V  # noqa: E402

ANGLES = HA.ARTICULATION_ANGLES


def key_of(subject, name, serial):
    return f"{subject}__{name}__{serial}"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def save_record(out_dir, subject, name, serial, side, world, label, gt=None, gt_ang=None):
    os.makedirs(out_dir, exist_ok=True)
    T = len(world)
    if gt_ang is None and gt is not None:
        flip = V.MIRROR_X if side == "left" else 1.0
        gt_ang = V.angles_of(gt * 1000.0 * flip)
    np.savez_compressed(
        os.path.join(out_dir, key_of(subject, name, serial) + ".npz"),
        gt=(np.full((T, 21, 3), np.nan) if gt is None else gt).astype(np.float32),
        has_gt_kps=np.array(gt is not None),
        gt_ang=np.asarray(gt_ang, np.float32),
        world=np.asarray(world, np.float32), label=np.asarray(label, np.int8),
        side=np.array(side), subject=np.array(subject), name=np.array(name),
        serial=np.array(serial))


def all_camera_sequences(root, serials):
    """Every (sequence, camera) pair with images, unlike V.find_sequences,
    which keeps only the first camera it finds per sequence."""
    out = []
    for m in sorted(glob.glob(os.path.join(root, "**", "meta.yml"), recursive=True)):
        sd = os.path.dirname(m)
        side, nf = V.read_meta(m)
        for s in serials:
            cam = os.path.join(sd, s)
            if os.path.isdir(cam) and glob.glob(os.path.join(cam, "color_*.jpg")):
                out.append(V.Sequence(subject=os.path.basename(os.path.dirname(sd)),
                                      name=os.path.basename(sd), cam_dir=cam, serial=s,
                                      side=side, num_frames=nf))
    return out


def build(root, out_dir, serials, mp_model, mp_cache, recompute=False, min_valid=10):
    seqs = all_camera_sequences(root, serials)
    print(f"[compact] {len(seqs)} (sequence, camera) pairs under {root}")
    done = skipped = 0
    for seq in V._progress(seqs, "compact cache"):
        target = os.path.join(out_dir, seq.key + ".npz")
        if os.path.exists(target) and not recompute:
            done += 1
            continue
        gt = V.load_gt(seq)
        if np.isfinite(gt).all((1, 2)).sum() < min_valid:
            skipped += 1
            continue
        world, label = V.detect_sequence(seq, mp_model, mp_cache, recompute)
        save_record(out_dir, seq.subject, seq.name, seq.serial, seq.side, world, label, gt=gt)
        done += 1
    print(f"[compact] {done} records in {out_dir}, {skipped} skipped (too few GT frames)")


def from_results(results_dir, out_dir, run="full"):
    """Rebuild records from a validate_dexycb.py results folder: GT angles from
    per_frame.csv.gz, hand side from per_sequence.csv, MediaPipe from mp_cache/."""
    import pandas as pd
    frames = pd.read_csv(os.path.join(results_dir, run, "per_frame.csv.gz"))
    sides = pd.read_csv(os.path.join(results_dir, run, "per_sequence.csv")) \
        .set_index("sequence")["side"].to_dict()
    n = 0
    for key, df in frames.groupby("sequence", sort=False):
        cache = os.path.join(results_dir, "mp_cache", key + ".npz")
        if not os.path.exists(cache):
            print(f"  no MediaPipe cache for {key}, skipped")
            continue
        subject, name, serial = key.split("__")
        d = np.load(cache)
        df = df.sort_values("frame")
        T = len(d["world"])
        gt_ang = np.full((T, len(ANGLES)), np.nan)
        rows = df["frame"].to_numpy()
        gt_ang[rows] = df[[f"gt_{a}" for a in ANGLES]].to_numpy()
        gt_ang[rows[df["gt_valid"].to_numpy() == 0]] = np.nan
        save_record(out_dir, subject, name, serial, sides[key], d["world"], d["label"],
                    gt=None, gt_ang=gt_ang)
        n += 1
    print(f"[compact] wrote {n} records (GT angles only) -> {out_dir}")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def load(compact_dir, subjects=None, serials=None, max_per_group=0, seed=0):
    """-> list of record dicts from one cache dir or several ("dirA,dirB").
    subjects/serials filter by substring/exact match;
    max_per_group keeps at most that many sequences per (subject, camera)."""
    dirs = [d for d in str(compact_dir).split(",") if d]       # "a,b" = several caches
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.npz")))
    if not files:
        raise SystemExit(f"no compact records in {compact_dir}")
    recs = []
    for f in files:
        with np.load(f) as d:
            r = {k: d[k] for k in d.files}
        for k in ("side", "subject", "name", "serial"):
            r[k] = str(r[k])
        r["has_gt_kps"] = bool(r.get("has_gt_kps", False))
        if subjects and not any(s in r["subject"] for s in subjects):
            continue
        if serials and r["serial"] not in serials:
            continue
        recs.append(r)
    if max_per_group:
        rng = np.random.default_rng(seed)
        groups = {}
        for r in recs:
            groups.setdefault((r["subject"], r["serial"]), []).append(r)
        recs = []
        for g in sorted(groups):
            items = groups[g]
            idx = sorted(rng.choice(len(items), min(max_per_group, len(items)), replace=False))
            recs += [items[i] for i in idx]
    return recs


def prepare(records, corrector=None, calib="mediapipe", min_valid=10):
    """Same dict schema as validate_dexycb.prepare, plus 'subject', 'serial'.
    gt_mm is None when the record has GT angles only."""
    prepared = []
    if calib == "gt" and records and not all(r["has_gt_kps"] for r in records):
        print("[compact] some records have GT angles only: those use the MediaPipe-derived "
              "hand model (calib 'mediapipe') and have no PA-MPJPE")
    for r in V._progress(records, "preparing"):
        side = r["side"]
        flip = V.MIRROR_X if side == "left" else 1.0
        gt_ang = r["gt_ang"].astype(float)
        valid = np.isfinite(gt_ang).all(1)
        world = r["world"].astype(float)
        det = np.isfinite(world).all((1, 2))
        if valid.sum() < min_valid or det.sum() < min_valid:
            continue
        mp_m = world * flip
        mp_ang = V.angles_of(mp_m)
        corr_ang = np.full_like(mp_ang, np.nan)
        if corrector is not None:
            for f in np.flatnonzero(det):
                corr_ang[f] = corrector(mp_m[f], mp_ang[f])
        gt_mm = r["gt"].astype(float) * 1000.0 * flip if r["has_gt_kps"] else None
        if calib == "gt" and gt_mm is not None:
            model = V.model_from_keypoints(gt_mm[valid])
        else:
            model = V.model_from_keypoints(mp_m[det] * 1000.0)
        seq = V.Sequence(subject=r["subject"], name=r["name"], cam_dir="", serial=r["serial"],
                         side=side, num_frames=len(world))
        want = 1 if side == "right" else 0
        prepared.append(dict(
            seq=seq, subject=r["subject"], serial=r["serial"], gt_mm=gt_mm, gt_ang=gt_ang,
            valid=valid, det=det, mp_mm=mp_m * 1000.0, mp_ang=mp_ang, corr_ang=corr_ang,
            model=model, label=r["label"],
            label_agree=float(np.mean(r["label"][det] == want)) if det.any() else np.nan))
    return prepared


def info(compact_dir):
    import pandas as pd
    recs = load(compact_dir)
    rows = [dict(subject=r["subject"], serial=r["serial"], side=r["side"],
                 frames=len(r["world"]), gt=int(np.isfinite(r["gt_ang"]).all(1).sum()),
                 det=int(np.isfinite(r["world"]).all((1, 2)).sum()),
                 gt_kps=r["has_gt_kps"]) for r in recs]
    df = pd.DataFrame(rows)
    g = df.groupby(["subject", "serial"]).agg(sequences=("frames", "size"),
                                              left=("side", lambda s: int((s == "left").sum())),
                                              gt_frames=("gt", "sum"), detected=("det", "sum"),
                                              gt_kps=("gt_kps", "all"))
    g["detect_%"] = (100 * g["detected"] / g["gt_frames"].clip(lower=1)).round(1)
    print(g.to_string())
    print(f"\n{len(df)} records, {df['subject'].nunique()} subjects, "
          f"{df['serial'].nunique()} cameras")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--serials", nargs="+", default=["all"],
                   help="camera serials, or 'all' for the 8 DexYCB cameras")
    b.add_argument("--mp-cache", default=None, help="MediaPipe cache dir (reused)")
    b.add_argument("--mp-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    b.add_argument("--recompute", action="store_true")
    c = sub.add_parser("from-results")
    c.add_argument("--results", required=True, help="folder with full/ and mp_cache/")
    c.add_argument("--out", required=True)
    c.add_argument("--run", default="full")
    i = sub.add_parser("info")
    i.add_argument("--compact", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        serials = V.SERIALS if args.serials == ["all"] else args.serials
        build(args.root, args.out, serials, args.mp_model,
              args.mp_cache or os.path.join(args.out, "_mp_cache"), args.recompute)
    elif args.cmd == "from-results":
        from_results(args.results, args.out, args.run)
    else:
        info(args.compact)


if __name__ == "__main__":
    main()
