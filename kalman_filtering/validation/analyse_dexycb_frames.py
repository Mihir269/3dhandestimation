#!/usr/bin/env python3
"""
analyse_dexycb_frames.py -- WHERE and WHY the pipeline is wrong, frame by frame.

The summary tables say "13.9 deg MAE". This script asks which frames carry that
error and what those frames have in common, so the next fix can be aimed.

For every evaluated frame (GT present and MediaPipe detected) it records the
error of raw MediaPipe, the corrector and the filter, plus candidate causes:

    gt_speed        how fast the true hand moves (deg/s, mean over angles)
    gt_flex         how closed the hand is (mean GT flexion, deg)
    meas_jump       how much the corrected measurement jumps frame to frame
                    beyond what GT does (|2nd diff corr| - |2nd diff GT|)
    corr_shift      how far the corrector moves MediaPipe (deg)
    filt_dev        how far the filter sits from its own measurement (deg)
    since_gap       frames since MediaPipe last lost the hand
    label_wrong     MediaPipe's handedness label disagrees with the truth
    mp_scale_dev    MediaPipe hand size vs that sequence's median (%), a sign of
                    a bad detection
    nis1, nis2      filter innovation statistics; infl1 = angles down-weighted
                    by the robust R (if enabled)
    t_norm          position in the sequence (0 = start, 1 = end); DexYCB
                    clips go open hand -> reach -> grasp
    palm_facing, depth_m   (only with GT keypoints) palm orientation to the
                    camera, hand distance

It then splits each sequence's error into BIAS (the per-sequence, per-angle
mean signed error, which no filter can remove) and TRANSIENT (the rest), and
reports error against each cause, the worst frames and sequences, what happens
after a detection gap, and whether the filter follows measurement glitches.

    python validation/analyse_dexycb_frames.py --compact output/dexycb_compact
    python validation/analyse_dexycb_frames.py --compact ... --robust --tag robust
    python validation/analyse_dexycb_frames.py --root /content/dexycb --serial 836212060125 \
        --cache-dir <mp_cache> --images 24          # adds a contact sheet of the worst frames

Outputs (--out/<tag>): frames.csv.gz, worst_frames.csv, sequences.csv,
report.md, error_vs_cause.png, worst_sequences.png, after_gap.png,
[worst_frames_sheet.jpg].
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN rows on missed frames

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, HERE)

import hand_angles_v2 as HA  # noqa: E402
import validate_dexycb as V  # noqa: E402
from angle_kf import AngleKF  # noqa: E402
from hand_ekf import EKFConfig, HandEKF, WRAP  # noqa: E402

ANGLES = HA.ARTICULATION_ANGLES
FPS = V.FPS
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_COLS = {f: [i for i, n in enumerate(ANGLES) if n.startswith(f)] for f in FINGERS}
FLEX = [i for i, n in enumerate(ANGLES) if n.endswith("_Flex") and not n.startswith("Thumb")]


# --------------------------------------------------------------------------
# Running the filter and collecting per-frame diagnostics
# --------------------------------------------------------------------------
def run_filter(p, cfg_kwargs, filt="ekf", source="corr"):
    z = p["corr_ang"] if source == "corr" else p["mp_ang"]
    T = len(z)
    out = dict(ang=np.full((T, len(ANGLES)), np.nan), nis1=np.full(T, np.nan),
               nis2=np.full(T, np.nan), infl1=np.zeros(T), infl2=np.zeros(T),
               std=np.full(T, np.nan))
    kf = (AngleKF if filt == "angle" else HandEKF)(p["model"], EKFConfig(**cfg_kwargs))
    for f in range(T):
        d = p["det"][f]
        est = kf.step(z[f] if d else None, p["mp_mm"][f] if d else None, 1.0 / FPS)
        if est is None:
            continue
        out["ang"][f] = est
        out["nis1"][f] = kf.last_nis1
        out["nis2"][f] = kf.last_nis2
        out["infl1"][f] = int(np.sum(getattr(kf, "last_infl1", np.ones(1)) > 1.0))
        out["infl2"][f] = int(np.sum(getattr(kf, "last_infl2", np.ones(1)) > 1.0))
        out["std"][f] = float(kf.angle_std.mean())
    return out


def frames_since(mask_missing):
    """For each frame, frames since the last True in mask_missing (start counts as a gap)."""
    out = np.zeros(len(mask_missing), int)
    c = 0
    for i, m in enumerate(mask_missing):
        c = 0 if m else c + 1
        out[i] = c
    return out


def palm_geometry(gt_mm):
    """Palm normal's angle to the camera axis (deg; 0 = palm facing the camera
    or its back, 90 = edge-on) and wrist depth (m). GT is in camera frame."""
    T = len(gt_mm)
    facing, depth = np.full(T, np.nan), np.full(T, np.nan)
    for f in range(T):
        k = gt_mm[f]
        if not np.isfinite(k).all():
            continue
        n = np.cross(k[5] - k[0], k[17] - k[0])
        nn = np.linalg.norm(n)
        if nn > 0:
            facing[f] = np.degrees(np.arccos(min(1.0, abs(n[2]) / nn)))
        depth[f] = abs(k[0, 2]) / 1000.0
    return facing, depth


def per_frame_table(prepared, cfg_kwargs, filt):
    import pandas as pd
    tables = []
    for p in V._progress(prepared, "analysing"):
        T = len(p["valid"])
        mask = p["valid"] & p["det"]
        if mask.sum() < 3:
            continue
        gt = p["gt_ang"]
        fr = run_filter(p, cfg_kwargs, filt)
        e_mp = WRAP(p["mp_ang"] - gt)
        e_corr = WRAP(p["corr_ang"] - gt)
        e_f = WRAP(fr["ang"] - gt)

        ugt = V.unwrap(np.where(np.isfinite(gt), gt, 0.0))
        speed = np.full(T, np.nan)
        speed[1:-1] = np.mean(np.abs(ugt[2:] - ugt[:-2]), 1) * FPS / 2
        uc = V.unwrap(np.where(np.isfinite(p["corr_ang"]), p["corr_ang"], 0.0))
        meas_jump = np.full(T, np.nan)
        d2c = np.abs(uc[2:] - 2 * uc[1:-1] + uc[:-2])
        d2g = np.abs(ugt[2:] - 2 * ugt[1:-1] + ugt[:-2])
        ok3 = mask[2:] & mask[1:-1] & mask[:-2]
        meas_jump[1:-1] = np.where(ok3, np.mean(d2c - d2g, 1), np.nan)

        hand_len = np.full(T, np.nan)
        det_idx = np.flatnonzero(p["det"])
        hand_len[det_idx] = np.linalg.norm(p["mp_mm"][det_idx, 9] - p["mp_mm"][det_idx, 0], axis=1)
        scale_dev = 100 * (hand_len / np.nanmedian(hand_len) - 1)

        want = 1 if p["seq"].side == "right" else 0
        label = p.get("label")
        label_wrong = (np.asarray(label) != want).astype(float) if label is not None \
            else np.full(T, np.nan)

        facing, depth = (palm_geometry(p["gt_mm"]) if p.get("gt_mm") is not None
                         else (np.full(T, np.nan), np.full(T, np.nan)))

        # bias = per-(sequence, angle) mean signed error of the filter output
        bias = np.nanmean(np.where(mask[:, None], e_f, np.nan), 0)
        trans = e_f - bias

        row = dict(sequence=p["seq"].key, subject=p["seq"].subject, serial=p["seq"].serial,
                   side=p["seq"].side, frame=np.arange(T), t_norm=np.arange(T) / max(T - 1, 1),
                   evaluated=mask.astype(int), detected=p["det"].astype(int),
                   err_mp=np.nanmean(np.abs(e_mp), 1), err_corr=np.nanmean(np.abs(e_corr), 1),
                   err_filt=np.nanmean(np.abs(e_f), 1),
                   err_filt_transient=np.nanmean(np.abs(trans), 1),
                   worst_angle=[ANGLES[int(np.nanargmax(np.abs(r)))] if np.isfinite(r).any()
                                else "" for r in e_f],
                   worst_angle_err=np.nanmax(np.abs(e_f), 1, initial=0,
                                             where=np.isfinite(e_f)),
                   gt_speed=speed, gt_flex=np.nanmean(gt[:, FLEX], 1), meas_jump=meas_jump,
                   corr_shift=np.nanmean(np.abs(WRAP(p["corr_ang"] - p["mp_ang"])), 1),
                   filt_dev=np.nanmean(np.abs(WRAP(fr["ang"] - p["corr_ang"])), 1),
                   since_gap=frames_since(~p["det"]), label_wrong=label_wrong,
                   mp_scale_dev=scale_dev, nis1=fr["nis1"], nis2=fr["nis2"],
                   infl1=fr["infl1"], infl2=fr["infl2"], filt_std=fr["std"],
                   palm_facing=facing, depth_m=depth)
        for f_name, cols in FINGER_COLS.items():
            row[f"err_{f_name.lower()}"] = np.nanmean(np.abs(e_f[:, cols]), 1)
        for i, n in enumerate(ANGLES):
            row[f"e_{n}"] = e_f[:, i]
        tables.append(pd.DataFrame(row))
    df = pd.concat(tables, ignore_index=True)
    return df


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
CAUSES = [("gt_speed", "true hand speed (deg/s)"),
          ("gt_flex", "hand closure: mean GT flexion (deg)"),
          ("meas_jump", "measurement jump beyond GT (deg/f^2)"),
          ("corr_shift", "|corrector - MediaPipe| (deg)"),
          ("filt_dev", "|filter - measurement| (deg)"),
          ("mp_scale_dev", "MediaPipe hand size vs median (%)"),
          ("nis1", "NIS block 1"),
          ("t_norm", "position in sequence (0-1)"),
          ("palm_facing", "palm normal vs camera axis (deg)"),
          ("depth_m", "hand distance (m)")]


def binned(ev, col, q=5):
    import pandas as pd
    ev = ev[ev[col].notna()]
    x = ev[col]
    if len(x) < 50 or x.nunique() < 3:
        return None
    bins = pd.qcut(x.rank(method="first"), q, labels=False).astype(int)
    g = ev.groupby(bins)
    out = pd.DataFrame({"range": g[col].agg(lambda s: f"{s.min():.3g} - {s.max():.3g}"),
                        "frames": g.size(),
                        "mp": g["err_mp"].mean(), "corr": g["err_corr"].mean(),
                        "filter": g["err_filt"].mean(),
                        "filter_transient": g["err_filt_transient"].mean()})
    out.index = [f"Q{i + 1}" for i in out.index]
    return out


def spearman(a, b):
    import pandas as pd
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return np.nan
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))


def analyse(df, out, top, cfg_desc):
    import pandas as pd
    ev = df[df.evaluated == 1].copy()
    E = ev[[f"e_{n}" for n in ANGLES]].to_numpy()
    lines = ["# Frame-level error analysis (DexYCB)", "",
             f"- {ev.sequence.nunique()} sequences, {len(ev)} evaluated frames, "
             f"{ev.subject.nunique()} subject(s), {ev.serial.nunique()} camera(s)",
             f"- filter: {cfg_desc}", ""]

    # ---- overall + bias / transient split
    mse = np.nanmean(E ** 2)
    trans = ev[[f"e_{n}" for n in ANGLES]] - ev.groupby("sequence")[[f"e_{n}" for n in ANGLES]].transform("mean")
    mse_t = np.nanmean(trans.to_numpy() ** 2)
    lines += ["## 1 · How much of the error is bias?", "",
              f"| | MAE (deg) |", "|---|---|",
              f"| MediaPipe | {ev.err_mp.mean():.2f} |",
              f"| corrector | {ev.err_corr.mean():.2f} |",
              f"| filter | {ev.err_filt.mean():.2f} |", "",
              f"Of the filter's mean squared error ({mse:.0f} deg^2), **{100 * (1 - mse_t / mse):.0f} %** "
              f"is per-sequence bias (a constant offset per angle for the whole clip) and "
              f"{100 * mse_t / mse:.0f} % is frame-to-frame (transient) error. A Kalman filter can "
              "only act on the transient part.", ""]

    # ---- per finger / per angle
    fing = {f: ev[f"err_{f.lower()}"].mean() for f in FINGERS}
    pa = pd.DataFrame({"angle": ANGLES, "mae": np.nanmean(np.abs(E), 0),
                       "bias": np.nanmean(E, 0),
                       "within_seq_bias_rms": np.sqrt(np.nanmean(
                           ev.groupby("sequence")[[f"e_{n}" for n in ANGLES]].mean().to_numpy() ** 2, 0)),
                       "transient_rms": np.sqrt(np.nanmean(trans.to_numpy() ** 2, 0))})
    pa = pa.sort_values("mae", ascending=False)
    pa.to_csv(os.path.join(out, "per_angle.csv"), index=False)
    lines += ["## 2 · Which fingers and angles", "",
              "| finger | MAE |", "|---|---|"] + [f"| {f} | {v:.2f} |" for f, v in fing.items()]
    lines += ["", "Worst 8 angles (bias = mean signed error over all frames; "
              "within_seq_bias_rms = typical per-clip offset):", "",
              V.md_table(pa.head(8).reset_index(drop=True)), ""]

    # ---- causes
    corr_rows, tables = [], {}
    for col, label in CAUSES:
        if col not in ev or ev[col].notna().sum() < 50:
            continue
        corr_rows.append(dict(cause=label, column=col,
                              rho_filter=spearman(ev[col], ev.err_filt),
                              rho_transient=spearman(ev[col], ev.err_filt_transient),
                              rho_corrector=spearman(ev[col], ev.err_corr)))
        tables[col] = binned(ev, col)
    cr = pd.DataFrame(corr_rows).sort_values("rho_filter", key=np.abs, ascending=False)
    cr.to_csv(os.path.join(out, "cause_correlations.csv"), index=False)
    lines += ["## 3 · What high-error frames have in common", "",
              "Spearman rank correlation of each candidate cause with the per-frame error "
              "(|rho| > 0.2 is worth a look, > 0.4 is strong):", "",
              V.md_table(cr.drop(columns="column").reset_index(drop=True)), ""]
    for col, label in CAUSES:
        t = tables.get(col)
        if t is None:
            continue
        lines += [f"**{label}** (quintiles, MAE in deg)", "", V.md_table(t.reset_index().rename(
            columns={"index": "bin"})), ""]

    # ---- events: after a gap, wrong label
    ev["gap_bin"] = pd.cut(ev.since_gap, [0, 1, 2, 5, 10, 1e9],
                           labels=["1 (first frame back)", "2", "3-5", "6-10", ">10"])
    g = ev.groupby("gap_bin", observed=True).agg(frames=("err_filt", "size"),
                                                  mp=("err_mp", "mean"), corr=("err_corr", "mean"),
                                                  filter=("err_filt", "mean"))
    lines += ["## 4 · After MediaPipe loses and re-finds the hand", "",
              "Frames since the last missed detection (the first frame of each clip counts as "
              "a gap, so this also shows the filter's start-up):", "",
              V.md_table(g.reset_index()), ""]
    if ev.label_wrong.notna().any():
        g2 = ev.assign(label_wrong=ev.label_wrong.astype(int)).groupby("label_wrong").agg(frames=("err_filt", "size"), mp=("err_mp", "mean"),
                                           corr=("err_corr", "mean"), filter=("err_filt", "mean"))
        lines += ["MediaPipe handedness label wrong (1) vs right (0). The evaluation uses the "
                  "true side, so this shows how bad MediaPipe's landmarks are on frames where "
                  "it is confused about the hand:", "", V.md_table(g2.reset_index()), ""]

    # ---- glitches: big measurement jumps
    thr = float(np.nanpercentile(ev.meas_jump, 97))
    gl = ev.meas_jump > thr
    lines += ["## 5 · Measurement glitches", "",
              f"Frames whose corrected measurement jumps more than the 97th percentile "
              f"({thr:.1f} deg/f^2 beyond GT): {int(gl.sum())} frames.", "",
              "| | glitch frames | other frames |", "|---|---|---|",
              f"| corrector MAE | {ev.err_corr[gl].mean():.2f} | {ev.err_corr[~gl].mean():.2f} |",
              f"| filter MAE | {ev.err_filt[gl].mean():.2f} | {ev.err_filt[~gl].mean():.2f} |",
              f"| filter - corrector | {(ev.err_filt - ev.err_corr)[gl].mean():+.2f} | "
              f"{(ev.err_filt - ev.err_corr)[~gl].mean():+.2f} |",
              f"| angles down-weighted by robust R | {ev.infl1[gl].mean():.2f} | "
              f"{ev.infl1[~gl].mean():.2f} |", "",
              "A negative 'filter - corrector' on glitch frames means the filter is "
              "already refusing to follow them.", ""]

    # ---- worst frames and sequences
    worst = ev.sort_values("err_filt", ascending=False).head(top)
    keep = ["sequence", "frame", "side", "err_mp", "err_corr", "err_filt", "worst_angle",
            "worst_angle_err", "gt_speed", "gt_flex", "meas_jump", "since_gap", "label_wrong",
            "mp_scale_dev", "nis1"]
    worst[[c for c in keep if c in worst]].to_csv(os.path.join(out, "worst_frames.csv"), index=False)
    seq = ev.groupby("sequence").agg(subject=("subject", "first"), side=("side", "first"),
                                     frames=("err_filt", "size"), mp=("err_mp", "mean"),
                                     corr=("err_corr", "mean"), filter=("err_filt", "mean"),
                                     transient=("err_filt_transient", "mean"),
                                     label_wrong=("label_wrong", "mean"),
                                     speed=("gt_speed", "mean"), flex=("gt_flex", "mean"))
    det = df.groupby("sequence").detected.mean()
    seq["detect_rate"] = det.reindex(seq.index)
    seq["bias_share"] = 1 - seq["transient"] / seq["filter"]
    seq = seq.sort_values("filter", ascending=False)
    seq.to_csv(os.path.join(out, "sequences.csv"))
    rho_side = ev.groupby("side").err_filt.mean()
    lines += ["## 6 · Worst sequences", "",
              f"Sequence MAE ranges {seq['filter'].min():.1f}-{seq['filter'].max():.1f} deg "
              f"(median {seq['filter'].median():.1f}). By hand: " + ", ".join(
                  f"{k} {v:.2f}" for k, v in rho_side.items()) + ".", "",
              V.md_table(seq.head(10).reset_index().round(2)), "",
              f"## 7 · Worst {min(top, 15)} frames", "",
              V.md_table(worst[[c for c in keep if c in worst]].head(15).round(2)), "",
              "Full lists: `worst_frames.csv`, `sequences.csv`, `per_angle.csv`, "
              "`cause_correlations.csv`; every frame with every feature: `frames.csv.gz`."]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    return ev, seq, cr, tables


def plots(df, ev, seq, tables, out, n_seq=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = [(c, l) for c, l in CAUSES if tables.get(c) is not None]
    nc = 4
    nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.3 * nr), squeeze=False)
    for ax, (c, label) in zip(axes.ravel(), cols):
        t = tables[c]
        x = np.arange(len(t))
        for k, col, st in (("mp", "#9e9e9e", ":"), ("corr", "#8d6e63", "--"),
                           ("filter", "#0d47a1", "-"), ("filter_transient", "#64b5f6", "-")):
            ax.plot(x, t[k], st, color=col, marker="o", ms=3, label=k)
        ax.set_xticks(x)
        ax.set_xticklabels(t.index)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("MAE (deg)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("error by quintile of each candidate cause (Q1 = lowest)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_vs_cause.png"), dpi=120)
    plt.close(fig)

    worst = list(seq.index[:n_seq])
    fig, axes = plt.subplots(len(worst), 1, figsize=(13, 2.3 * len(worst)), squeeze=False)
    for ax, key in zip(axes[:, 0], worst):
        d = df[df.sequence == key]
        ax.plot(d.frame, d.err_mp, ":", color="#9e9e9e", label="MediaPipe")
        ax.plot(d.frame, d.err_corr, "--", color="#8d6e63", label="corrector")
        ax.plot(d.frame, d.err_filt, color="#0d47a1", lw=1.8, label="filter")
        for f in d.frame[d.detected == 0]:
            ax.axvspan(f - 0.5, f + 0.5, color="red", alpha=0.15, lw=0)
        if d.label_wrong.notna().any():
            for f in d.frame[(d.label_wrong == 1) & (d.detected == 1)]:
                ax.axvspan(f - 0.5, f + 0.5, color="orange", alpha=0.15, lw=0)
        ax2 = ax.twinx()
        ax2.plot(d.frame, d.gt_speed, color="green", lw=0.8, alpha=0.6)
        ax2.set_ylabel("GT speed", color="green", fontsize=7)
        ax.set_title(f"{key}  ({d.side.iloc[0]} hand)   red = no detection, "
                     "orange = wrong handedness label", fontsize=8)
        ax.set_ylabel("MAE (deg)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "worst_sequences.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    g = ev.assign(sg=ev.since_gap.clip(upper=20)).groupby("sg")[["err_mp", "err_corr", "err_filt"]].mean()
    for k, col in (("err_mp", "#9e9e9e"), ("err_corr", "#8d6e63"), ("err_filt", "#0d47a1")):
        ax.plot(g.index, g[k], color=col, marker="o", ms=3, label=k.replace("err_", ""))
    ax.set_xlabel("frames since the last missed detection (20 = 20+)")
    ax.set_ylabel("MAE (deg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "after_gap.png"), dpi=120)
    plt.close(fig)


def contact_sheet(df, prepared, out, n):
    """Worst frames as images with MediaPipe's 2D-projected landmarks (needs --root)."""
    import cv2
    by_key = {p["seq"].key: p for p in prepared}
    worst = df[df.evaluated == 1].sort_values("err_filt", ascending=False).head(n)
    tiles = []
    for _, r in worst.iterrows():
        p = by_key.get(r.sequence)
        if p is None or not p["seq"].cam_dir:
            continue
        img = cv2.imread(os.path.join(p["seq"].cam_dir, f"color_{int(r.frame):06d}.jpg"))
        if img is None:
            continue
        img = cv2.resize(img, (320, 180))
        for i, txt in enumerate((f"{r.sequence.split('__')[1]} f{int(r.frame)} {r.side}",
                                 f"filter {r.err_filt:.0f}  corr {r.err_corr:.0f}  mp {r.err_mp:.0f}",
                                 f"worst {r.worst_angle} {r.worst_angle_err:.0f}")):
            cv2.putText(img, txt, (4, 14 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(out, "worst_frames_sheet.jpg"), sheet)


def main(argv=None):
    ap = V.build_parser()
    ap.description = __doc__
    ap.set_defaults(out=os.path.join(ROOT, "output", "validation", "frame_analysis"),
                    calib="mediapipe")
    ap.add_argument("--filter", choices=["ekf", "angle"], default="ekf")
    ap.add_argument("--top", type=int, default=50, help="worst frames to list")
    ap.add_argument("--images", type=int, default=0, help="--root only: contact sheet of N worst frames")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = os.path.join(args.out, args.tag) if args.tag else args.out
    os.makedirs(out, exist_ok=True)
    from pipeline import AngleCorrector
    prepared = V.load_prepared(args, AngleCorrector(args.ckpt), args.calib)
    cfg = V.cfg_from_args(args)
    desc = (f"{args.filter}, calib {args.calib}, " + ", ".join(
        f"{k}={'per-angle' if isinstance(v, np.ndarray) else v}" for k, v in cfg.items()))
    df = per_frame_table(prepared, cfg, args.filter)
    df.to_csv(os.path.join(out, "frames.csv.gz"), index=False)
    ev, seq, cr, tables = analyse(df, out, args.top, desc)
    plots(df, ev, seq, tables, out)
    if args.images and args.root:
        contact_sheet(df, prepared, out, args.images)
    print(open(os.path.join(out, "report.md"), encoding="utf-8").read()[:4000])
    print(f"\n[analyse] {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
