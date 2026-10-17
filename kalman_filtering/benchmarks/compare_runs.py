#!/usr/bin/env python3
"""
compare_runs.py -- benchmark the CSVs written by the three runners.

    python benchmarks\\compare_runs.py raw=output\\raw\\clip.csv kf_mp=output\\kf_mp\\clip.csv ekf=output\\ekf\\clip.csv
    python benchmarks\\compare_runs.py ... --plot-angle Thumb_IP_Flex --out-dir output\\benchmarks\\clip

Run all three on the SAME video file for a fair comparison (a webcam session
can't be replayed). There is no ground truth on your own footage, so these are
smoothness / consistency / lag metrics, not accuracy:

  ang_jitter   RMS 2nd difference of the angles, deg/frame^2 (wrap-aware)
  ang_jerk     RMS 3rd difference, deg/frame^3
  kp_jitter_local  RMS 2nd difference of the 3D output keypoints in each
               frame's own palm frame, mm/frame^2 -- articulation jitter only
  kp_jitter    same in the wrist-centred camera-oriented frame; includes the
               (unfiltered) per-frame root rotation
  bone_cv_%    per-bone length coefficient of variation over time, mean of 20.
               FK output is constant by construction; MediaPipe's is not.
  dev_mp       mean |output - raw MediaPipe| angle, deg. How far the method moves
               away from the measurement (filter lag + corrector offset).
  dev_corr     mean |output - corrected| angle, deg (ekf runs only)
  lag_frames   median over angles of the cross-correlation peak between output
               and raw MediaPipe (positive = output lags)
  nis1/nis2    mean NIS (targets 21 / 45)

Accuracy against GT needs FreiHAND-style labels; see the README.
Writes <out-dir>/benchmark_summary.csv, benchmark_per_angle.csv and a plot.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
import hand_angles_v2 as HA  # noqa: E402

ANGLES = HA.ARTICULATION_ANGLES
wrap = HA.wrap_deg


def segments(valid, frames, min_len=5):
    """Contiguous runs of valid, consecutive frames -> list of index arrays."""
    out, cur = [], []
    for i, (v, f) in enumerate(zip(valid, frames)):
        if v and cur and f == frames[cur[-1]] + 1:
            cur.append(i)
        else:
            if len(cur) >= min_len:
                out.append(np.array(cur))
            cur = [i] if v else []
    if len(cur) >= min_len:
        out.append(np.array(cur))
    return out


def unwrap_deg(a):
    """Per-column unwrap so differences across +-180 are continuous."""
    return np.degrees(np.unwrap(np.radians(a), axis=0))


def angle_metrics(A, segs):
    d2, d3 = [], []
    for s in segs:
        u = unwrap_deg(A[s])
        if len(s) > 3:
            d2.append(np.diff(u, 2, axis=0))
            d3.append(np.diff(u, 3, axis=0))
    d2 = np.concatenate(d2) if d2 else np.full((1, A.shape[1]), np.nan)
    d3 = np.concatenate(d3) if d3 else np.full((1, A.shape[1]), np.nan)
    return np.sqrt(np.nanmean(d2 ** 2, 0)), np.sqrt(np.nanmean(d3 ** 2, 0))


def lag_frames(out, ref, segs, max_lag=10):
    s = max(segs, key=len) if segs else None
    if s is None or len(s) < 4 * max_lag:
        return np.full(out.shape[1], np.nan)
    lags = []
    for j in range(out.shape[1]):
        a = unwrap_deg(ref[s, j]); b = unwrap_deg(out[s, j])
        a, b = a - a.mean(), b - b.mean()
        if a.std() < 1e-6 or b.std() < 1e-6:
            lags.append(np.nan)
            continue
        cc = [np.dot(a[max(0, -k):len(a) - max(0, k)], b[max(0, k):len(b) - max(0, -k)])
              for k in range(-max_lag, max_lag + 1)]
        lags.append(np.argmax(cc) - max_lag)
    return np.array(lags, float)


def analyse(label, path):
    df = pd.read_csv(path)
    frames = df["frame"].to_numpy()
    out = df[[f"out_{n}" for n in ANGLES]].to_numpy(float)
    mp = df[[f"mp_{n}" for n in ANGLES]].to_numpy(float)
    kp = df[[f"kpout_{j}_{a}" for j in range(21) for a in "xyz"]].to_numpy(float).reshape(-1, 21, 3)
    valid = np.isfinite(out).all(1)
    segs = segments(valid, frames)

    jit, jerk = angle_metrics(out, segs)
    # local = expressed in each frame's own palm frame, which removes the global
    # hand rotation. The filters' root rotation is a per-frame Umeyama fit that
    # is NOT filtered, so the global number mixes root-pose noise into it.
    kp_local = np.full_like(kp, np.nan)
    for i in np.flatnonzero(np.isfinite(kp).all((1, 2))):
        kp_local[i] = kp[i] @ HA.palm_frame(kp[i], hand="right")

    def rms_d2(K):
        d = [np.linalg.norm(np.diff(K[s], 2, axis=0), axis=2) for s in segs if len(s) > 3]
        return float(np.sqrt(np.mean(np.concatenate(d) ** 2))) if d else np.nan

    kp_jit, kp_jit_local = rms_d2(kp), rms_d2(kp_local)

    kv = kp[valid]
    bl = np.stack([np.linalg.norm(kv[:, b] - kv[:, a], axis=1) for a, b in HA.BONES], 1)
    bone_cv = float(np.mean(bl.std(0) / np.maximum(bl.mean(0), 1e-9)) * 100) if len(kv) else np.nan

    both = valid & np.isfinite(mp).all(1)
    dev_mp = np.nanmean(np.abs(wrap(out[both] - mp[both])), 0)
    row = dict(run=label, frames=len(df), valid=int(valid.sum()),
               ang_jitter=float(np.nanmean(jit)), ang_jerk=float(np.nanmean(jerk)),
               kp_jitter_local_mm=kp_jit_local, kp_jitter_mm=kp_jit, bone_cv_pct=bone_cv,
               dev_mp_deg=float(np.nanmean(dev_mp)))
    per = pd.DataFrame({"angle": ANGLES, "run": label, "jitter": jit, "jerk": jerk,
                        "dev_mp": dev_mp})
    if f"corr_{ANGLES[0]}" in df:
        corr = df[[f"corr_{n}" for n in ANGLES]].to_numpy(float)
        ok = valid & np.isfinite(corr).all(1)
        row["dev_corr_deg"] = float(np.nanmean(np.abs(wrap(out[ok] - corr[ok]))))
    lag = lag_frames(out, mp, segments(both, frames))
    row["lag_frames"] = float(np.nanmedian(lag)) if np.isfinite(lag).any() else np.nan
    per["lag_frames"] = lag
    for c in ("nis1", "nis2", "resid_mm"):
        if c in df:
            row[c] = float(df[c].mean())
    return row, per, df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="csv paths, optionally label=path")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output", "benchmarks"))
    ap.add_argument("--plot-angle", default="Index_PIP_Flex")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows, pers, dfs = [], [], {}
    for r in args.runs:
        label, path = r.split("=", 1) if "=" in r and not os.path.exists(r) else \
            (os.path.splitext(os.path.basename(r))[0], r)
        row, per, df = analyse(label, path)
        rows.append(row); pers.append(per); dfs[label] = df

    summary = pd.DataFrame(rows)
    base = summary.iloc[0]
    summary["jitter_vs_first"] = summary["ang_jitter"] / base["ang_jitter"]
    os.makedirs(args.out_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.out_dir, "benchmark_summary.csv"), index=False)
    pd.concat(pers).to_csv(os.path.join(args.out_dir, "benchmark_per_angle.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.round(3).to_string(index=False))
    print(f"\n(jitter_vs_first is relative to '{base['run']}')")
    print(f"wrote {os.path.join(args.out_dir, 'benchmark_summary.csv')}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = args.plot_angle
    fig, ax = plt.subplots(figsize=(12, 4.5))
    first = next(iter(dfs.values()))
    ax.plot(first["time_s"], first[f"mp_{n}"], color="#bbbbbb", lw=1, label="raw MediaPipe")
    for label, df in dfs.items():
        if f"corr_{n}" in df:
            ax.plot(df["time_s"], df[f"corr_{n}"], lw=0.8, ls=":", label=f"{label}: corrected (unfiltered)")
        ax.plot(df["time_s"], df[f"out_{n}"], lw=1.6, label=f"{label}: output")
    ax.set_xlabel("time (s)"); ax.set_ylabel(f"{n} (deg)")
    ax.set_title(f"{n} over time"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.out_dir, f"benchmark_{n}.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
