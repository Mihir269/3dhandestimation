#!/usr/bin/env python3
"""
validate_dexycb.py -- accuracy of the filters against DexYCB ground truth.

DexYCB (https://dex-ycb.github.io/) is real 30 fps RGB video with per-frame 3D
hand joints in MediaPipe's 21-joint order, so it can measure what the live
benchmarks cannot: ERROR against truth, not just smoothness.

Six methods, all on the same frames:

    method          angles fed in          filter
    --------------  ---------------------  ------------------------------
    mp              MediaPipe              none
    corr            v12 corrector          none
    kf_angle_mp     MediaPipe              angle-only KF      (angle_kf.py)
    kf_angle_corr   v12 corrector          angle-only KF
    ekf_mp          MediaPipe              two-block EKF + MediaPipe 3D
    ekf_corr        v12 corrector          two-block EKF + MediaPipe 3D

Expected layout (what the Colab notebook extracts; any nesting depth works):

    <root>/<subject>/<sequence>/meta.yml
    <root>/<subject>/<sequence>/<camera serial>/color_000000.jpg
    <root>/<subject>/<sequence>/<camera serial>/labels_000000.npz   (joint_3d, metres, -1 = no hand)

Usage:
    python validation/validate_dexycb.py --root /content/dexycb
    python validation/validate_dexycb.py --root D:/dexycb --max-seqs 5          # quick look
    python validation/validate_dexycb.py --root ... --sigma-angle 4 --tag s4    # re-tune: MediaPipe is cached
    python validation/validate_dexycb.py --compact output/dexycb_compact          # no images needed
    python validation/validate_dexycb.py --compact ... --robust --tag robust     # innovation-weighted R

Assumptions (see Flow.md for the full log):
  * Handedness comes from meta.yml (ground truth), not MediaPipe's label, so
    this tests the filters, not handedness detection. Label agreement is
    reported separately.
  * The hand model for FK / block 2 is built from each sequence's own GT
    (median bone lengths + fan) -- an ORACLE calibration, the best case for
    block 2. `--calib mediapipe` builds it from MediaPipe's detections instead.
  * Metrics use frames where GT exists AND MediaPipe detected the hand, the
    same frames for every method. Frames the filters bridge without a
    detection are scored separately (coast_mae).
  * 3D error is PA-MPJPE (similarity-aligned per frame), because MediaPipe's
    world frame and DexYCB's camera frame differ by an unknown rotation.
    'mp' is scored on MediaPipe's own landmarks; every other method on FK of
    its angles, which carries the FK model floor (reported as gt_fk_floor).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import hand_angles_v2 as HA  # noqa: E402
from angle_kf import AngleKF  # noqa: E402
from hand_ekf import EKFConfig, HandEKF, PersonalizedHandModel, WRAP, umeyama  # noqa: E402

ANGLES = HA.ARTICULATION_ANGLES
FPS = 30.0
SERIALS = ["836212060125", "839512060362", "840412060917", "841412060263",
           "932122060857", "932122060861", "932122061900", "932122062010"]
MIRROR_X = np.array([-1.0, 1.0, 1.0])

# name -> (angle source, filter)
METHODS = {
    "mp": ("mp", None),
    "corr": ("corr", None),
    "kf_angle_mp": ("mp", "angle"),
    "kf_angle_corr": ("corr", "angle"),
    "ekf_mp": ("mp", "ekf"),
    "ekf_corr": ("corr", "ekf"),
}
LABELS = {
    "mp": "MediaPipe (raw)",
    "corr": "v12 corrector (unfiltered)",
    "kf_angle_mp": "angle-only KF on MediaPipe",
    "kf_angle_corr": "angle-only KF on corrector",
    "ekf_mp": "EKF (angles + 3D) on MediaPipe",
    "ekf_corr": "EKF (angles + 3D) on corrector",
}


def _progress(it, desc, total=None):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc, total=total)
    except ImportError:
        return it


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------
@dataclass
class Sequence:
    subject: str
    name: str
    cam_dir: str
    serial: str
    side: str            # 'right' | 'left'
    num_frames: int

    @property
    def key(self):
        return f"{self.subject}__{self.name}__{self.serial}"


def read_meta(path):
    text = open(path).read()
    try:
        import yaml
        meta = yaml.safe_load(text)
        return str(meta["mano_sides"][0]).lower(), int(meta["num_frames"])
    except Exception:
        side = re.search(r"mano_sides:\s*\n?\s*-?\s*\[?\s*'?\"?(\w+)", text)
        nf = re.search(r"num_frames:\s*(\d+)", text)
        if not (side and nf):
            raise ValueError(f"cannot parse {path}")
        return side.group(1).lower(), int(nf.group(1))


def find_sequences(root, serial=None):
    metas = sorted(glob.glob(os.path.join(root, "**", "meta.yml"), recursive=True))
    if not metas:
        raise SystemExit(f"no meta.yml found under {root} -- is this the extracted DexYCB folder?")
    seqs, skipped = [], 0
    for m in metas:
        sd = os.path.dirname(m)
        serials = [serial] if serial else SERIALS
        cam = next((os.path.join(sd, s) for s in serials
                    if os.path.isdir(os.path.join(sd, s))
                    and glob.glob(os.path.join(sd, s, "color_*.jpg"))), None)
        if cam is None:
            skipped += 1
            continue
        side, nf = read_meta(m)
        seqs.append(Sequence(subject=os.path.basename(os.path.dirname(sd)),
                             name=os.path.basename(sd), cam_dir=cam,
                             serial=os.path.basename(cam), side=side, num_frames=nf))
    if skipped:
        print(f"[dexycb] skipped {skipped} sequences with no images for the requested camera")
    return seqs


def load_gt(seq):
    """(T,21,3) metres in camera coordinates, NaN where there is no hand."""
    gt = np.full((seq.num_frames, 21, 3), np.nan)
    for f in range(seq.num_frames):
        p = os.path.join(seq.cam_dir, f"labels_{f:06d}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as lab:
            j = np.asarray(lab["joint_3d"], float).reshape(21, 3)
        if not np.all(j == -1):
            gt[f] = j
    return gt


# --------------------------------------------------------------------------
# MediaPipe (cached)
# --------------------------------------------------------------------------
def detect_sequence(seq, mp_model, cache_dir, recompute=False):
    """-> world (T,21,3) metres, NaN when not detected; label (T,) 1=right 0=left -1=none."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, seq.key + ".npz")
    if os.path.exists(cache) and not recompute:
        d = np.load(cache)
        return d["world"], d["label"]

    import cv2
    from pipeline import HandTracker
    tracker = HandTracker(mp_model)          # fresh per sequence: VIDEO mode keeps state
    world = np.full((seq.num_frames, 21, 3), np.nan)
    label = np.full(seq.num_frames, -1, np.int8)
    try:
        for f in range(seq.num_frames):
            img = cv2.imread(os.path.join(seq.cam_dir, f"color_{f:06d}.jpg"))
            if img is None:
                continue
            det = tracker.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f * 1000.0 / FPS)
            if det is not None:
                world[f] = det["world"]
                label[f] = 1 if det["hand"].startswith("r") else 0
    finally:
        try:
            tracker.landmarker.close()
        except Exception:
            pass
    np.savez_compressed(cache, world=world, label=label)
    return world, label


# --------------------------------------------------------------------------
# Preparation: GT angles, MediaPipe angles, corrected angles, hand model
# --------------------------------------------------------------------------
def model_from_keypoints(kps_mm):
    """Median bones + fan over valid frames -> PersonalizedHandModel."""
    valid = [k for k in kps_mm if np.isfinite(k).all()]
    bones = np.median([[HA.bone_lengths_from_kps(k)[b] for b in HA.BONES] for k in valid], 0)
    fans = np.median([WRAP(HA.compute_fan(k)) for k in valid], 0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return PersonalizedHandModel(bones, fans, name="sequence")


def angles_of(kps_seq):
    out = np.full((len(kps_seq), len(ANGLES)), np.nan)
    for f, k in enumerate(kps_seq):
        if np.isfinite(k).all():
            out[f] = HA.compute_angles(k, hand="right")
    return out


def prepare(seqs, mp_model, cache_dir, corrector=None, calib="gt", recompute=False,
            min_valid=10):
    """Everything that does not depend on filter settings. Slow part (MediaPipe)
    is cached on disk; the rest takes seconds per sequence."""
    prepared = []
    for seq in _progress(seqs, "preparing sequences"):
        gt = load_gt(seq)
        valid = np.isfinite(gt).all((1, 2))
        if valid.sum() < min_valid:
            continue
        world, label = detect_sequence(seq, mp_model, cache_dir, recompute)
        det = np.isfinite(world).all((1, 2))
        flip = MIRROR_X if seq.side == "left" else 1.0      # everything right-handed
        gt_mm = gt * 1000.0 * flip
        mp_m = world * flip
        mp_ang = angles_of(mp_m)
        corr_ang = np.full_like(mp_ang, np.nan)
        if corrector is not None:
            for f in np.flatnonzero(det):
                corr_ang[f] = corrector(mp_m[f], mp_ang[f])
        if calib == "gt":
            model = model_from_keypoints(gt_mm[valid])
        else:
            if det.sum() < min_valid:
                continue
            model = model_from_keypoints(mp_m[det] * 1000.0)
        want = 1 if seq.side == "right" else 0
        prepared.append(dict(
            seq=seq, gt_mm=gt_mm, gt_ang=angles_of(gt_mm), valid=valid, det=det,
            mp_mm=mp_m * 1000.0, mp_ang=mp_ang, corr_ang=corr_ang, model=model,
            label_agree=float(np.mean(label[det] == want)) if det.any() else np.nan))
    return prepared


# --------------------------------------------------------------------------
# Running the methods
# --------------------------------------------------------------------------
def run_methods(prepared, cfg_kwargs=None, methods=None):
    """-> {seq key: {method: dict(ang (T,21), kps (T,21,3), nis1, nis2)}}"""
    cfg_kwargs = cfg_kwargs or {}
    methods = methods or list(METHODS)
    results = {}
    for p in prepared:
        T = len(p["valid"])
        res = {}
        for m in methods:
            src, filt = METHODS[m]
            z = p["mp_ang"] if src == "mp" else p["corr_ang"]
            if src == "corr" and not np.isfinite(z[p["det"]]).all():
                continue                                   # no corrector loaded
            ang = np.full((T, len(ANGLES)), np.nan)
            kps = np.full((T, 21, 3), np.nan)
            nis1, nis2 = [], []
            if filt is None:
                for f in np.flatnonzero(p["det"]):
                    ang[f] = z[f]
                    kps[f] = p["mp_mm"][f] if m == "mp" else p["model"].fk(z[f])
            else:
                cfg = EKFConfig(**cfg_kwargs)
                kf = (AngleKF if filt == "angle" else HandEKF)(p["model"], cfg)
                for f in range(T):
                    d = p["det"][f]
                    est = kf.step(z[f] if d else None, p["mp_mm"][f] if d else None, 1.0 / FPS)
                    if est is not None:
                        ang[f] = est
                        kps[f] = kf.skeleton(oriented=False)
                    if np.isfinite(kf.last_nis1):
                        nis1.append(kf.last_nis1)
                    if np.isfinite(kf.last_nis2):
                        nis2.append(kf.last_nis2)
            res[m] = dict(ang=ang, kps=kps,
                          nis1=float(np.mean(nis1)) if nis1 else np.nan,
                          nis2=float(np.mean(nis2)) if nis2 else np.nan)
        results[p["seq"].key] = res
    return results


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def pa_mpjpe(pred, gt):
    s, R, t = umeyama(pred, gt)
    aligned = s * pred @ R.T + t
    return float(np.linalg.norm(aligned - gt, axis=1).mean()), aligned


def runs(mask, min_len=4):
    out, cur = [], []
    for i, v in enumerate(mask):
        if v:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                out.append(np.array(cur))
            cur = []
    if len(cur) >= min_len:
        out.append(np.array(cur))
    return out


def unwrap(a):
    return np.degrees(np.unwrap(np.radians(a), axis=0))


def lag_frames(pred, gt, max_lag=5):
    lags = []
    for j in range(pred.shape[1]):
        a, b = gt[:, j] - gt[:, j].mean(), pred[:, j] - pred[:, j].mean()
        if a.std() < 1.0:                     # angle barely moves: lag is undefined
            continue
        cc = [np.dot(a[max(0, -k):len(a) - max(0, k)], b[max(0, k):len(b) - max(0, -k)])
              for k in range(-max_lag, max_lag + 1)]
        lags.append(int(np.argmax(cc)) - max_lag)
    return lags


def evaluate(prepared, results):
    import pandas as pd
    methods = [m for m in METHODS if any(m in r for r in results.values())]
    acc = {m: dict(err=[], pa=[], accel=[], accel_kp=[], jit=[], jit_gt=[], lag=[],
                   coast=[], nis1=[], nis2=[]) for m in methods}
    floor, per_seq, frames = [], [], []

    for p in prepared:
        key = p["seq"].key
        mask = p["valid"] & p["det"]
        gt_ang, gt_mm = p["gt_ang"], p["gt_mm"]
        has_kps = gt_mm is not None                      # compact records may hold angles only
        for f in (np.flatnonzero(mask)[::5] if has_kps else []):   # model floor, subsampled
            floor.append(pa_mpjpe(p["model"].fk(gt_ang[f]), gt_mm[f])[0])
        row = dict(sequence=key, subject=p["seq"].subject, serial=p["seq"].serial,
                   side=p["seq"].side, frames=len(mask),
                   gt_frames=int(p["valid"].sum()), evaluated=int(mask.sum()),
                   mp_label_agreement=p["label_agree"])
        segs = runs(mask)
        for m in methods:
            r = results[key].get(m)
            if r is None:
                continue
            a = acc[m]
            e = WRAP(r["ang"][mask] - gt_ang[mask])
            a["err"].append(e)
            row[m] = float(np.nanmean(np.abs(e))) if len(e) else np.nan
            coast = p["valid"] & ~p["det"] & np.isfinite(r["ang"]).all(1)
            if coast.any():
                a["coast"].append(np.abs(WRAP(r["ang"][coast] - gt_ang[coast])))
            aligned = np.full_like(r["kps"], np.nan)
            for f in (np.flatnonzero(mask) if has_kps else []):
                err3d, aligned[f] = pa_mpjpe(r["kps"][f], gt_mm[f])
                a["pa"].append(err3d)
            for s in segs:
                up, ug = unwrap(r["ang"][s]), unwrap(gt_ang[s])
                d2p, d2g = np.diff(up, 2, axis=0), np.diff(ug, 2, axis=0)
                a["accel"].append(np.abs(d2p - d2g).ravel())
                a["jit"].append(d2p.ravel())
                a["jit_gt"].append(d2g.ravel())
                if has_kps:
                    k2p = np.diff(aligned[s], 2, axis=0)
                    k2g = np.diff(gt_mm[s], 2, axis=0)
                    a["accel_kp"].append(np.linalg.norm(k2p - k2g, axis=2).ravel())
            if segs:
                longest = max(segs, key=len)
                if len(longest) >= 20:
                    a["lag"] += lag_frames(unwrap(r["ang"][longest]), unwrap(gt_ang[longest]))
            for n in ("nis1", "nis2"):
                if np.isfinite(r[n]):
                    a[n].append(r[n])
        per_seq.append(row)

        fr = dict(sequence=np.repeat(key, len(mask)), frame=np.arange(len(mask)),
                  gt_valid=p["valid"].astype(int), detected=p["det"].astype(int))
        for i, n in enumerate(ANGLES):
            fr[f"gt_{n}"] = gt_ang[:, i]
            for m in methods:
                if m in results[key]:
                    fr[f"{m}_{n}"] = results[key][m]["ang"][:, i]
        frames.append(pd.DataFrame(fr))

    summary, per_angle = [], []
    for m in methods:
        a = acc[m]
        if not a["err"]:
            continue
        E = np.concatenate(a["err"])
        absE = np.abs(E)
        cat = lambda k: np.concatenate(a[k]) if a[k] else np.array([np.nan])  # noqa: E731
        jit, jit_gt = cat("jit"), cat("jit_gt")
        summary.append(dict(
            method=m, description=LABELS[m], frames=len(E),
            angle_mae=float(absE.mean()), angle_rmse=float(np.sqrt((E ** 2).mean())),
            angle_median=float(np.median(absE)),
            pck_10deg=float((absE < 10).mean() * 100),
            pa_mpjpe_mm=float(np.mean(a["pa"])) if a["pa"] else np.nan,
            accel_err_deg=float(np.nanmean(cat("accel"))),
            accel_err_mm=float(np.nanmean(cat("accel_kp"))),
            jitter=float(np.sqrt(np.nanmean(jit ** 2))),
            jitter_vs_gt=float(np.sqrt(np.nanmean(jit ** 2)) / np.sqrt(np.nanmean(jit_gt ** 2))),
            lag_frames=float(np.median(a["lag"])) if a["lag"] else np.nan,
            coast_mae=float(np.mean(np.concatenate(a["coast"]))) if a["coast"] else np.nan,
            nis1=float(np.mean(a["nis1"])) if a["nis1"] else np.nan,
            nis2=float(np.mean(a["nis2"])) if a["nis2"] else np.nan))
        for i, n in enumerate(ANGLES):
            per_angle.append(dict(angle=n, method=m, mae=float(absE[:, i].mean()),
                                  bias=float(E[:, i].mean()),
                                  rmse=float(np.sqrt((E[:, i] ** 2).mean()))))
    summary = pd.DataFrame(summary)
    extra = dict(gt_fk_floor_mm=float(np.mean(floor)) if floor else np.nan,
                 sequences=len(prepared),
                 frames_with_gt=int(sum(p["valid"].sum() for p in prepared)),
                 frames_evaluated=int(sum((p["valid"] & p["det"]).sum() for p in prepared)))
    return (summary, pd.DataFrame(per_angle), pd.DataFrame(per_seq),
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), extra)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    from pandas.api.types import is_integer_dtype
    is_int = [is_integer_dtype(df[c].dtype) for c in cols]
    for row in df.itertuples(index=False):     # itertuples keeps each column's dtype
        cells = [str(int(v)) if ii else
                 floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v, ii in zip(row, is_int)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out, summary, per_angle, per_seq, extra, args_dict):
    cols = ["method", "angle_mae", "angle_rmse", "pck_10deg", "pa_mpjpe_mm",
            "accel_err_deg", "jitter_vs_gt", "lag_frames", "coast_mae", "nis1", "nis2"]
    base = summary.set_index("method")["angle_mae"]
    lines = [
        "# DexYCB validation report", "",
        f"- sequences: {extra['sequences']}, frames with GT: {extra['frames_with_gt']}, "
        f"evaluated (GT and MediaPipe detection): {extra['frames_evaluated']}",
        f"- FK model floor (FK of GT angles vs GT, PA-MPJPE): {extra['gt_fk_floor_mm']:.1f} mm",
        f"- MediaPipe handedness label agreement: {np.nanmean(per_seq['mp_label_agreement']) * 100:.1f} %",
        f"- settings: {args_dict}", "",
        "## Summary", "", md_table(summary[[c for c in cols if c in summary]]), "",
        "Columns: angle MAE/RMSE in degrees (wrap-aware), PCK = % of angles within 10 deg, "
        "PA-MPJPE = 3D joint error after similarity alignment (mm), accel_err = mean "
        "|2nd difference of prediction - of GT| (deg/frame^2), jitter_vs_gt = RMS 2nd "
        "difference relative to the GT's own, lag in frames (positive = behind GT), "
        "coast_mae = error on frames the filter bridged without a detection.", "",
    ]
    if "mp" in base:
        lines += ["## Change in angle MAE vs raw MediaPipe", ""]
        for m, v in base.items():
            lines.append(f"- {m}: {v:.2f} deg ({100 * (v / base['mp'] - 1):+.1f} %)")
        lines.append("")
    pa = per_angle.pivot(index="angle", columns="method", values="mae").reindex(ANGLES)
    bias = per_angle.pivot(index="angle", columns="method", values="bias").reindex(ANGLES)
    lines += ["## Per-angle MAE (deg)", "", md_table(pa.reset_index()), "",
              "## Per-angle signed bias (deg) -- what no Kalman filter can remove", "",
              md_table(bias.reset_index()), ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def make_plots(out, prepared, results, summary, per_angle, plot_angle, plot_seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mp": "#9e9e9e", "corr": "#8d6e63", "kf_angle_mp": "#ffb74d",
              "kf_angle_corr": "#e65100", "ekf_mp": "#64b5f6", "ekf_corr": "#0d47a1"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(axes, ("angle_mae", "pa_mpjpe_mm", "jitter_vs_gt"),
                              ("angle MAE (deg)", "PA-MPJPE (mm)", "jitter / GT jitter")):
        ax.bar(summary["method"], summary[col], color=[colors[m] for m in summary["method"]])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for i, v in enumerate(summary[col]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].axhline(1.0, color="k", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=130)
    plt.close(fig)

    pv = per_angle.pivot(index="angle", columns="method", values="mae").reindex(ANGLES)
    fig, ax = plt.subplots(figsize=(16, 5))
    w = 0.8 / len(pv.columns)
    x = np.arange(len(pv))
    for i, m in enumerate(pv.columns):
        ax.bar(x + i * w, pv[m], w, label=m, color=colors[m])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(pv.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("MAE (deg)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_angle_mae.png"), dpi=130)
    plt.close(fig)

    if not prepared:
        return
    p = prepared[min(plot_seq, len(prepared) - 1)]
    j = ANGLES.index(plot_angle)
    t = np.arange(len(p["valid"])) / FPS
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(t, p["gt_ang"][:, j], color="k", lw=2.2, label="ground truth")
    for m, r in results[p["seq"].key].items():
        ax.plot(t, r["ang"][:, j], color=colors[m], lw=1.2 if m in ("mp", "corr") else 1.8,
                ls=":" if m in ("mp", "corr") else "-", label=m)
    ax.set_title(f"{plot_angle} - {p['seq'].key} ({p['seq'].side} hand)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"timeseries_{plot_angle}.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    d = EKFConfig()
    ap = argparse.ArgumentParser(description="Validate the Kalman filters on DexYCB",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="extracted DexYCB folder (images + labels)")
    src.add_argument("--compact", help="compact cache from dexycb_data.py (no images needed)")
    ap.add_argument("--serial", default=None, help="camera serial (default: first found)")
    ap.add_argument("--subjects", nargs="+", default=None, help="--compact: subject filter")
    ap.add_argument("--serials", nargs="+", default=None, help="--compact: camera filter")
    ap.add_argument("--max-per-group", type=int, default=0,
                    help="--compact: at most N sequences per (subject, camera)")
    ap.add_argument("--clip-len", type=int, default=0,
                    help="cut sequences into windows of N frames (0 = whole sequences)")
    ap.add_argument("--clips-per-seq", type=int, default=0,
                    help="with --clip-len: keep N evenly spaced windows per sequence (0 = all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "validation", "dexycb"))
    ap.add_argument("--tag", default="", help="subfolder for this run's results")
    ap.add_argument("--cache-dir", default=None, help="MediaPipe cache (default <out>/mp_cache)")
    ap.add_argument("--recompute", action="store_true", help="ignore the MediaPipe cache")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--calib", choices=["gt", "mediapipe"], default="gt",
                    help="hand model for FK/block 2: oracle from GT, or from MediaPipe")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--no-corrector", action="store_true")
    ap.add_argument("--mp-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "angle_corrector_v12.pth"))
    ap.add_argument("--plot-angle", default="Index_PIP_Flex", choices=ANGLES)
    ap.add_argument("--plot-seq", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    g = ap.add_argument_group("filter settings")
    g.add_argument("--jerk-psd", type=float, default=d.jerk_psd)
    g.add_argument("--sigma-angle", type=float, default=d.sigma_angle)
    g.add_argument("--r-scale-angles", type=float, default=d.r_scale_angles)
    g.add_argument("--sigma-xy", type=float, default=d.sigma_xy)
    g.add_argument("--sigma-z", type=float, default=d.sigma_z)
    g.add_argument("--r-scale-kps", type=float, default=d.r_scale_kps)
    g.add_argument("--angle-noise-csv", default=None, help="per-angle sigma (overrides --sigma-angle)")
    g.add_argument("--coast-decay", type=float, default=d.coast_decay,
                   help="velocity/acceleration multiplier per coasting frame (1 = none)")
    g.add_argument("--robust", action="store_true", help="innovation-weighted R (EKFConfig.robust)")
    g.add_argument("--robust-k", type=float, default=d.robust_k)
    g.add_argument("--robust-k-kp", type=float, default=d.robust_k_kp)
    g.add_argument("--robust-power", type=float, default=d.robust_power)
    return ap


def cfg_from_args(args):
    sigma = args.sigma_angle
    if getattr(args, "angle_noise_csv", None):
        from hand_ekf import load_angle_sigma_csv
        sigma = load_angle_sigma_csv(args.angle_noise_csv, "Corr")
    return dict(jerk_psd=args.jerk_psd, sigma_angle=sigma,
                r_scale_angles=args.r_scale_angles, sigma_xy=args.sigma_xy,
                sigma_z=args.sigma_z, r_scale_kps=args.r_scale_kps,
                robust=args.robust, robust_k=args.robust_k,
                robust_k_kp=args.robust_k_kp, robust_power=args.robust_power,
                coast_decay=args.coast_decay)


_CLIP_KEYS = ("gt_ang", "valid", "det", "mp_mm", "mp_ang", "corr_ang", "label", "gt_mm")


def clip_prepared(prepared, clip_len=0, clips_per_seq=0):
    """Cut long sequences into windows of clip_len frames (HO-3D clips are
    1000-2800 frames, DexYCB ~70). clips_per_seq > 0 keeps that many evenly
    spaced windows per sequence, so tuning costs about the same per sequence on
    both datasets. Windows keep the sequence's subject/camera, so subject- and
    camera-level splits never separate them. The filter restarts per window."""
    if not clip_len:
        return prepared
    out = []
    for p in prepared:
        T = len(p["valid"])
        if T <= clip_len:
            out.append(p)
            continue
        starts = list(range(0, T - clip_len + 1, clip_len))
        if clips_per_seq and len(starts) > clips_per_seq:
            starts = [starts[int(round(i))] for i in np.linspace(0, len(starts) - 1, clips_per_seq)]
        for s0 in starts:
            q = dict(p)
            for k in _CLIP_KEYS:
                if q.get(k) is not None:
                    q[k] = p[k][s0:s0 + clip_len]
            if (q["valid"] & q["det"]).sum() < 10:
                continue
            sq = p["seq"]
            q["seq"] = Sequence(subject=sq.subject, name=f"{sq.name}@{s0}", cam_dir=sq.cam_dir,
                                serial=sq.serial, side=sq.side, num_frames=clip_len)
            out.append(q)
    print(f"[dexycb] clipped {len(prepared)} sequences into {len(out)} windows of {clip_len} frames")
    return out


def load_prepared(args, corrector, calib):
    """Prepared sequences from --root (images) or --compact (cache)."""
    if getattr(args, "compact", None):
        import dexycb_data as D
        recs = D.load(args.compact, args.subjects, args.serials, args.max_per_group)
        if args.max_seqs:
            recs = recs[:args.max_seqs]
        print(f"[dexycb] {len(recs)} compact records from {args.compact}")
        return clip_prepared(D.prepare(recs, corrector, calib),
                             getattr(args, "clip_len", 0), getattr(args, "clips_per_seq", 0))
    seqs = find_sequences(args.root, args.serial)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    print(f"[dexycb] {len(seqs)} sequences, camera {seqs[0].serial if seqs else '-'}, "
          f"{sum(s.side == 'left' for s in seqs)} left-handed")
    cache_dir = args.cache_dir or os.path.join(args.out, "mp_cache")
    return prepare(seqs, args.mp_model, cache_dir, corrector, calib, args.recompute)


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()
    out = os.path.join(args.out, args.tag) if args.tag else args.out
    os.makedirs(out, exist_ok=True)
    corrector = None
    if not args.no_corrector:
        from pipeline import AngleCorrector
        corrector = AngleCorrector(args.ckpt)
    prepared = load_prepared(args, corrector, args.calib)
    if not prepared:
        raise SystemExit("no usable sequences (too few GT frames)")

    cfg = cfg_from_args(args)
    results = run_methods(prepared, cfg, args.methods)
    summary, per_angle, per_seq, frames, extra = evaluate(prepared, results)

    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    per_angle.to_csv(os.path.join(out, "per_angle.csv"), index=False)
    per_seq.to_csv(os.path.join(out, "per_sequence.csv"), index=False)
    frames.to_csv(os.path.join(out, "per_frame.csv.gz"), index=False)
    cams = sorted({p["seq"].serial for p in prepared})
    write_report(out, summary, per_angle, per_seq, extra,
                 dict(calib=args.calib, cameras=cams,
                      **{k: ("per-angle" if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}))
    if not args.no_plots:
        make_plots(out, prepared, results, summary, per_angle, args.plot_angle, args.plot_seq)

    import pandas as pd
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    show = ["method", "angle_mae", "angle_rmse", "pck_10deg", "pa_mpjpe_mm",
            "accel_err_deg", "jitter_vs_gt", "lag_frames", "coast_mae", "nis1", "nis2"]
    print()
    print(summary[show].round(2).to_string(index=False))
    print(f"\nFK model floor {extra['gt_fk_floor_mm']:.1f} mm | evaluated "
          f"{extra['frames_evaluated']}/{extra['frames_with_gt']} GT frames | "
          f"{time.time() - t0:.0f} s")
    print(f"results -> {out}")
    return summary


if __name__ == "__main__":
    main()
