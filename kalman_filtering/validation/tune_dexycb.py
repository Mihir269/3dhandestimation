#!/usr/bin/env python3
"""
tune_dexycb.py -- optimise the filter's noise parameters against DexYCB truth.

WHAT IS BEING TUNED
-------------------
  Q(dt) = jerk_psd * [CWNJ 3x3 per angle]      how much the hand is allowed to
                                               change its acceleration
  R1    = diag(sigma_angle^2) * r_scale_angles measurement noise on the angles
  R2    = diag(sigma_xy^2, sigma_xy^2, sigma_z^2) * r_scale_kps + C_align
                                               noise on MediaPipe's 3D keypoints
  robust_k, robust_k_kp  (--tune-robust)       how many sigmas a measurement may
                                               jump from the filter's prediction
                                               before its R is inflated

`r_scale_*` only multiplies the sigmas, so it is DEGENERATE with them: the
tuner fixes r_scale = 1 and tunes the sigmas.

sigma_angle may be one number (`--sigma-mode uniform`) or a 21-vector
(`--sigma-mode perangle`). The per-angle shape is NOT taken from the ground
truth; it is estimated from the measured angle stream alone (white noise:
Var(2nd difference) = 6 sigma^2), and only one global multiplier is optimised.

HONEST SPLITS
-------------
  --split parity    alternate sequences train / held out (default)
  --split subject   whole subjects held out (needs several subjects)
  --split camera    whole cameras held out (needs several cameras)

GROUP MODE: is the tuning the same on different data?  (--group-by)
--------------------------------------------------------------------
`--group-by subject` (or `camera`) tunes the filter separately on every group
(each on its own train half), then scores every group's best settings on every
group's held-out half. If the parameters differ but the cross-scores barely
move, the objective is flat and one setting serves everyone; if a group's own
settings clearly beat the others on its data, the tuning does not transfer.

    python validation/tune_dexycb.py --compact output/dexycb_compact --method ekf_corr --workers 8
    python validation/tune_dexycb.py --compact ... --tune-robust --accel-weight 0.5
    python validation/tune_dexycb.py --compact ... --group-by subject --max-per-group 30
    python validation/tune_dexycb.py --root /content/dexycb --method kf_angle_corr   # from images

Outputs (in --out): best_config.json, tuned_flags.txt, tuned_sigma_per_angle.csv
(feeds `--angle-noise-csv`), trials.csv, report.md, plots; group mode adds
group_params.csv, group_cross_mae.csv, group_report.md.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mproc
import os
import sys
import time

import numpy as np

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


# --------------------------------------------------------------------------
# Noise-floor estimate from the measurement stream (no ground truth)
# --------------------------------------------------------------------------
def estimate_sigma(prepared, source="corr"):
    """Robust per-angle white-noise sigma from second differences."""
    per = [[] for _ in ANGLES]
    for p in prepared:
        z = p["mp_ang"] if source == "mp" else p["corr_ang"]
        for seg in V.runs(p["det"], min_len=8):
            d2 = np.diff(V.unwrap(z[seg]), 2, axis=0)
            for i in range(len(ANGLES)):
                per[i].append(d2[:, i])
    out = np.zeros(len(ANGLES))
    for i, chunks in enumerate(per):
        d = np.concatenate(chunks) if chunks else np.array([0.0])
        mad = np.median(np.abs(d - np.median(d)))
        out[i] = 1.4826 * mad / np.sqrt(6.0)
    return np.maximum(out, 0.2)


# --------------------------------------------------------------------------
# Objective: per-sequence parts, then one aggregate
# --------------------------------------------------------------------------
def score_one(p, cfg_kwargs, source="corr", filt="ekf"):
    """Run one sequence; return the raw pieces the aggregate needs."""
    z = p["mp_ang"] if source == "mp" else p["corr_ang"]
    T = len(z)
    ang = np.full((T, len(ANGLES)), np.nan)
    nis1, nis2 = [], []
    if filt is None:
        ang[p["det"]] = z[p["det"]]
    else:
        cfg = EKFConfig(**cfg_kwargs)
        kf = (AngleKF if filt == "angle" else HandEKF)(p["model"], cfg)
        for f in range(T):
            d = p["det"][f]
            est = kf.step(z[f] if d else None, p["mp_mm"][f] if d else None, 1.0 / FPS)
            if est is not None:
                ang[f] = est
            if np.isfinite(kf.last_nis1):
                nis1.append(kf.last_nis1)
            if np.isfinite(kf.last_nis2):
                nis2.append(kf.last_nis2)
    m = p["valid"] & p["det"]
    if not m.any():
        return None
    acc, jit, jit_gt = [], [], []
    for seg in V.runs(m):
        d2p = np.diff(V.unwrap(ang[seg]), 2, axis=0)
        d2g = np.diff(V.unwrap(p["gt_ang"][seg]), 2, axis=0)
        acc.append(np.abs(d2p - d2g).ravel())
        jit.append(d2p.ravel())
        jit_gt.append(d2g.ravel())
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    return dict(err=WRAP(ang[m] - p["gt_ang"][m]), acc=cat(acc), jit=cat(jit),
                jit_gt=cat(jit_gt), nis1=np.array(nis1), nis2=np.array(nis2))


def aggregate(parts):
    parts = [q for q in parts if q is not None]
    E = np.concatenate([q["err"] for q in parts])
    J = np.concatenate([q["jit"] for q in parts])
    JG = np.concatenate([q["jit_gt"] for q in parts])
    A = np.concatenate([q["acc"] for q in parts])
    n1 = np.concatenate([q["nis1"] for q in parts])
    n2 = np.concatenate([q["nis2"] for q in parts])
    return dict(mae=float(np.nanmean(np.abs(E))),
                rmse=float(np.sqrt(np.nanmean(E ** 2))),
                pck10=float(np.nanmean(np.abs(E) < 10) * 100),
                accel=float(np.nanmean(A)) if A.size else np.nan,
                jitter_vs_gt=float(np.sqrt(np.nanmean(J ** 2)) / np.sqrt(np.nanmean(JG ** 2)))
                if J.size else np.nan,
                nis1=float(n1.mean()) if n1.size else np.nan,
                nis2=float(n2.mean()) if n2.size else np.nan)


def score(prepared, cfg_kwargs, source="corr", filt="ekf"):
    """Serial scorer over a list of prepared sequences (kept for notebooks)."""
    return aggregate([score_one(p, cfg_kwargs, source, filt) for p in prepared])


_PREP = None


def _init_worker(prepared):
    global _PREP
    _PREP = prepared


def _score_idx(job):
    i, cfg, source, filt = job
    return score_one(_PREP[i], cfg, source, filt)


class Scorer:
    """Scores a subset (by index) of one prepared list, optionally in parallel.
    The prepared data is sent to each worker once, not per call."""

    def __init__(self, prepared, workers=1):
        self.prepared = prepared
        self.pool = None
        if workers > 1 and len(prepared) > 1:
            self.pool = mproc.get_context("spawn").Pool(workers, _init_worker, (prepared,))

    def __call__(self, idx, cfg, source, filt):
        idx = list(idx)
        if self.pool is None:
            return aggregate([score_one(self.prepared[i], cfg, source, filt) for i in idx])
        jobs = [(i, cfg, source, filt) for i in idx]
        return aggregate(self.pool.map(_score_idx, jobs, chunksize=max(1, len(jobs) // 64)))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# --------------------------------------------------------------------------
# Search space: everything in log10
# --------------------------------------------------------------------------
class Space:
    def __init__(self, use_ekf, sigma_mode, base_sigma, tune_robust=False, robust_power=1.0):
        self.use_ekf = use_ekf
        self.sigma_mode = sigma_mode
        self.base_sigma = base_sigma
        self.tune_robust = tune_robust
        self.robust_power = robust_power
        self.names = ["log10_jerk_psd",
                      "log10_sigma_scale" if sigma_mode == "perangle" else "log10_sigma_angle"]
        self.bounds = [(3.0, 7.0), (-0.7, 0.7) if sigma_mode == "perangle" else (0.0, 1.5)]
        if use_ekf:
            self.names += ["log10_sigma_xy", "log10_sigma_z"]
            self.bounds += [(0.3, 1.9), (0.7, 2.3)]
        if tune_robust:
            k_bounds = (np.log10(0.8), np.log10(10.0))
            self.names.append("log10_robust_k")
            self.bounds.append(k_bounds)
            if use_ekf:
                self.names.append("log10_robust_k_kp")
                self.bounds.append(k_bounds)

    def clip(self, x):
        """The point actually used: the search may step outside the bounds."""
        return np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

    def cfg(self, x):
        x = self.clip(x)
        v = dict(zip(self.names, x))
        s = 10 ** x[1]
        cfg = dict(jerk_psd=float(10 ** v["log10_jerk_psd"]),
                   sigma_angle=(s * self.base_sigma if self.sigma_mode == "perangle" else float(s)),
                   r_scale_angles=1.0, r_scale_kps=1.0)
        if self.use_ekf:
            cfg.update(sigma_xy=float(10 ** v["log10_sigma_xy"]),
                       sigma_z=float(10 ** v["log10_sigma_z"]))
        if self.tune_robust:
            cfg.update(robust=True, robust_power=self.robust_power,
                       robust_k=float(10 ** v["log10_robust_k"]))
            if self.use_ekf:
                cfg["robust_k_kp"] = float(10 ** v["log10_robust_k_kp"])
        return cfg

    def x_from_cfg(self, cfg):
        """Point in the space for a config (used to seed the search)."""
        sig = np.asarray(cfg.get("sigma_angle", 6.0), float)
        if self.sigma_mode == "perangle":
            s = float(np.median((sig if sig.ndim else np.full(len(ANGLES), sig)) / self.base_sigma))
        else:
            s = float(sig if sig.ndim == 0 else np.median(sig))
        vals = {"log10_jerk_psd": np.log10(cfg.get("jerk_psd", 3e4)),
                "log10_sigma_scale": np.log10(s), "log10_sigma_angle": np.log10(s),
                "log10_sigma_xy": np.log10(cfg.get("sigma_xy", 15.0)),
                "log10_sigma_z": np.log10(cfg.get("sigma_z", 25.0)),
                "log10_robust_k": np.log10(cfg.get("robust_k", 3.0)),
                "log10_robust_k_kp": np.log10(cfg.get("robust_k_kp", 3.0))}
        x = np.array([vals[n] for n in self.names])
        return np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

    def sample(self, rng, n):
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi, size=(n, len(lo)))

    def pretty(self, cfg):
        out = {"jerk_psd": f"{cfg['jerk_psd']:.3g}"}
        sig = np.asarray(cfg["sigma_angle"], float)
        out["sigma_angle"] = f"{float(np.median(sig)):.2f}" + (" (median)" if sig.ndim else "")
        for k in ("sigma_xy", "sigma_z", "robust_k", "robust_k_kp"):
            if k in cfg:
                out[k] = f"{cfg[k]:.3g}"
        return out


def optimise(scorer, idx, space, source, filt, n_random=40, n_refine=60, seed=0,
             accel_weight=0.0, seed_cfgs=(), verbose=True, tag=""):
    rng = np.random.default_rng(seed)
    trials = []

    def f(x):
        cfg = space.cfg(x)
        s = scorer(idx, cfg, source, filt)
        val = s["mae"] + accel_weight * s["accel"]
        x = space.clip(x)                      # record and print what was evaluated
        trials.append(dict(**{n: float(v) for n, v in zip(space.names, x)}, **s, objective=val))
        if verbose:
            print(f"  {tag}[{len(trials):3d}] " + "  ".join(
                f"{n.replace('log10_', '')} {10 ** v:8.3g}" for n, v in zip(space.names, x))
                + f" | MAE {s['mae']:.3f}  accel {s['accel']:.2f}  obj {val:.3f}", flush=True)
        return val

    X = [space.x_from_cfg(c) for c in seed_cfgs] + list(space.sample(rng, n_random))
    vals = [f(x) for x in X]
    best = X[int(np.argmin(vals))]
    if n_refine:
        try:
            from scipy.optimize import minimize
            minimize(f, best, method="Nelder-Mead",
                     options=dict(maxfev=n_refine, xatol=0.02, fatol=1e-3))
        except ImportError:
            print("  scipy missing: refining by local random search instead")
            step = 0.15
            for _ in range(n_refine):
                cand = best + rng.normal(0, step, len(best))
                if f(cand) <= min(t["objective"] for t in trials[:-1]):
                    best = cand
                step *= 0.97
    import pandas as pd
    trials = pd.DataFrame(trials)
    best = trials.loc[trials["objective"].idxmin(), space.names].to_numpy(float)
    return best, trials


# --------------------------------------------------------------------------
# Splits and groups
# --------------------------------------------------------------------------
def group_key(p, by):
    if by == "dataset":
        return "HO3D" if p["seq"].subject.startswith("HO3D") else "DexYCB"
    return p["seq"].subject if by == "subject" else p["seq"].serial


def split(prepared, how):
    idx = np.arange(len(prepared))
    if how == "parity":
        return idx[0::2], idx[1::2]
    keys = sorted({group_key(p, how) for p in prepared})
    if len(keys) < 2:
        print(f"[tune] only one {how} present: falling back to the parity split")
        return idx[0::2], idx[1::2]
    held = set(keys[1::2])
    test = [i for i, p in enumerate(prepared) if group_key(p, how) in held]
    train = [i for i in idx if i not in set(test)]
    print(f"[tune] {how} split: train {keys[0::2]} | held out {keys[1::2]}")
    return np.array(train), np.array(test)


def defaults_cfg(filt):
    d = EKFConfig()
    cfg = dict(jerk_psd=d.jerk_psd, sigma_angle=d.sigma_angle,
               r_scale_angles=d.r_scale_angles, r_scale_kps=d.r_scale_kps)
    if filt == "ekf":
        cfg.update(sigma_xy=d.sigma_xy, sigma_z=d.sigma_z)
    return cfg


def flags_for(cfg, filt, sigma_mode, csv_path):
    flags = [f"--jerk-psd {cfg['jerk_psd']:.4g}", "--r-scale-angles 1.0"]
    sig = np.asarray(cfg["sigma_angle"], float)
    if sigma_mode == "perangle" and sig.ndim:
        flags.append(f"--angle-noise-csv {csv_path}")
    else:
        flags.append(f"--sigma-angle {float(np.median(sig)):.4g}")
    if filt == "ekf":
        flags += [f"--sigma-xy {cfg['sigma_xy']:.4g}", f"--sigma-z {cfg['sigma_z']:.4g}",
                  "--r-scale-kps 1.0"]
    if cfg.get("robust"):
        flags += ["--robust", f"--robust-k {cfg['robust_k']:.3g}",
                  f"--robust-power {cfg.get('robust_power', 1.0):g}"]
        if "robust_k_kp" in cfg:
            flags.append(f"--robust-k-kp {cfg['robust_k_kp']:.3g}")
    return " ".join(flags)


def jsonable(cfg):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}


# --------------------------------------------------------------------------
# Single tuning run
# --------------------------------------------------------------------------
def tune_once(args, prepared, scorer, source, filt, out):
    train, test = split(prepared, args.split)
    print(f"[tune] {len(prepared)} sequences -> {len(train)} train / {len(test)} held out, "
          f"method {args.method}, calib {args.calib}")
    base_sigma = estimate_sigma([prepared[i] for i in train], source)
    print(f"[tune] estimated angle noise: median {np.median(base_sigma):.2f} deg, "
          f"range {base_sigma.min():.2f}-{base_sigma.max():.2f} "
          f"(worst: {ANGLES[int(np.argmax(base_sigma))]})")
    space = Space(filt == "ekf", args.sigma_mode, base_sigma, args.tune_robust, args.robust_power)
    base = defaults_cfg(filt)
    ref = {"unfiltered (held out)": scorer(test, {}, source, None),
           "current defaults (train)": scorer(train, base, source, filt),
           "current defaults (held out)": scorer(test, base, source, filt)}
    if args.tune_robust:
        rb = dict(base, robust=True, robust_power=args.robust_power)
        ref["defaults + robust k=3 (held out)"] = scorer(test, rb, source, filt)
    for k, v in ref.items():
        print(f"[tune] {k:34s} MAE {v['mae']:.3f}  accel {v['accel']:.2f}")

    print("[tune] searching ...")
    best, trials = optimise(scorer, train, space, source, filt, args.n_random, args.n_refine,
                            args.seed, args.accel_weight, seed_cfgs=[base],
                            verbose=not args.quiet)
    best_cfg = space.cfg(best)
    res = {"train (tuned)": scorer(train, best_cfg, source, filt),
           "held out (tuned)": scorer(test, best_cfg, source, filt)}
    for k, v in res.items():
        print(f"[tune] {k:34s} MAE {v['mae']:.3f}  accel {v['accel']:.2f}  "
              f"nis1 {v['nis1']:.1f}  nis2 {v['nis2']:.1f}")

    sigma = np.asarray(best_cfg["sigma_angle"], float)
    sigma_vec = np.full(len(ANGLES), float(sigma)) if sigma.ndim == 0 else sigma
    import pandas as pd
    csv_path = os.path.join(out, "tuned_sigma_per_angle.csv")
    pd.DataFrame({"Angle": ANGLES, "RMSE_Corr": sigma_vec, "RMSE_MP": sigma_vec}).to_csv(
        csv_path, index=False)
    trials.to_csv(os.path.join(out, "trials.csv"), index=False)
    flag_str = flags_for(best_cfg, filt, args.sigma_mode, csv_path)
    open(os.path.join(out, "tuned_flags.txt"), "w").write(flag_str + "\n")
    json.dump({"method": args.method, "calib": args.calib, "sigma_mode": args.sigma_mode,
               "split": args.split, "accel_weight": args.accel_weight,
               "config": jsonable(best_cfg), "readable": space.pretty(best_cfg),
               "flags": flag_str, "reference": ref, "tuned": res,
               "sigma_estimate_deg": base_sigma.tolist()},
              open(os.path.join(out, "best_config.json"), "w"), indent=2)

    def_key = "current defaults (held out)"
    gain = 100 * (1 - res["held out (tuned)"]["mae"] / ref[def_key]["mae"])
    lines = ["# Filter tuning on DexYCB", "",
             f"- method `{args.method}`, calibration `{args.calib}`, sigma mode "
             f"`{args.sigma_mode}`, split `{args.split}`, objective MAE + "
             f"{args.accel_weight} x accel, robust tuned: {args.tune_robust}",
             f"- {len(train)} training / {len(test)} held-out sequences, "
             f"{len(trials)} evaluations, {time.time() - args._t0:.0f} s",
             f"- tuned: {space.pretty(best_cfg)}", "",
             "| configuration | MAE | accel err | jitter/GT | NIS1 | NIS2 |",
             "|---|---|---|---|---|---|"]
    for k, v in list(ref.items()) + list(res.items()):
        lines.append(f"| {k} | {v['mae']:.3f} | {v['accel']:.2f} | {v['jitter_vs_gt']:.2f} | "
                     f"{v['nis1']:.1f} | {v['nis2']:.1f} |")
    lines += ["", f"Held-out MAE change vs current defaults: **{gain:+.2f} %**", "",
              "Run the live scripts with:", "", "```", flag_str, "```", "",
              "A gain on `train` that does not appear on `held out` is overfitting."]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    _scatter(trials, space, args.method, out)
    print(f"\n[tune] best flags: {flag_str}")
    print(f"[tune] held-out MAE {res['held out (tuned)']['mae']:.3f} vs default "
          f"{ref[def_key]['mae']:.3f} ({gain:+.2f} %)")
    return best_cfg


def _scatter(trials, space, method, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(space.names), figsize=(3.8 * len(space.names), 3.4))
    for ax, n in zip(np.atleast_1d(axes), space.names):
        ax.scatter(10 ** trials[n], trials["mae"], s=12, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(n.replace("log10_", ""))
        ax.set_ylabel("MAE (deg)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"tuning {method}: MAE vs parameter")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tuning_scatter.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Group mode: tune per subject / camera, then cross-evaluate
# --------------------------------------------------------------------------
def tune_groups(args, prepared, scorer, source, filt, out):
    import pandas as pd
    by = args.group_by
    keys = sorted({group_key(p, by) for p in prepared})
    if len(keys) < 2:
        raise SystemExit(f"--group-by {by} needs at least two {by}s in the data, found {keys}")
    halves = {}
    for g in keys:
        idx = [i for i, p in enumerate(prepared) if group_key(p, by) == g]
        if by == "dataset" and len({prepared[i]["seq"].subject for i in idx}) >= 2:
            # alternate whole subjects: HO-3D's 5 cameras (and the windows of one
            # clip) show the same motion and must not straddle train/held-out
            subs = sorted({prepared[i]["seq"].subject for i in idx})
            held = set(subs[1::2])
            halves[g] = ([i for i in idx if prepared[i]["seq"].subject not in held],
                         [i for i in idx if prepared[i]["seq"].subject in held])
        else:
            halves[g] = (idx[0::2], idx[1::2])
    all_train = [i for g in keys for i in halves[g][0]]
    base_sigma = estimate_sigma([prepared[i] for i in all_train], source)
    space = Space(filt == "ekf", args.sigma_mode, base_sigma, args.tune_robust, args.robust_power)
    base = defaults_cfg(filt)
    print(f"[group] {len(keys)} {by}s: " + ", ".join(
        f"{g} ({len(halves[g][0])}+{len(halves[g][1])})" for g in keys))

    tuned, rows = {}, []
    for gi, g in enumerate(keys + ["ALL"]):
        tr = all_train if g == "ALL" else halves[g][0]
        print(f"\n[group] tuning on {g} ({len(tr)} sequences)")
        done = os.path.join(out, f"group_best_{g}.json")
        if os.path.exists(done):                       # resume after a Colab pause
            best = np.array(json.load(open(done))["x"])
            print(f"[group] {g}: already tuned, resumed from {os.path.basename(done)}")
        else:
            best, trials = optimise(scorer, tr, space, source, filt, args.n_random,
                                    args.n_refine, args.seed + gi, args.accel_weight,
                                    seed_cfgs=[base], verbose=not args.quiet, tag=f"{g[-12:]} ")
            trials.to_csv(os.path.join(out, f"trials_{g}.csv"), index=False)
            json.dump({"x": [float(v) for v in best], "names": space.names}, open(done, "w"))
        tuned[g] = space.cfg(best)
        rows.append(dict(group=g, n_train=len(tr), **{
            n.replace("log10_", ""): float(10 ** v) for n, v in zip(space.names, space.clip(best))}))
    params = pd.DataFrame(rows)
    params.to_csv(os.path.join(out, "group_params.csv"), index=False)

    # cross evaluation: settings tuned on row-group, scored on column-group's held-out half
    cross, cross_acc = {}, {}
    sources = keys + ["ALL"]
    for g_cfg in sources + ["defaults"]:
        cfg = base if g_cfg == "defaults" else tuned[g_cfg]
        cross[g_cfg], cross_acc[g_cfg] = {}, {}
        for g_eval in keys:
            s = scorer(halves[g_eval][1], cfg, source, filt)
            cross[g_cfg][g_eval] = s["mae"]
            cross_acc[g_cfg][g_eval] = s["accel"]
        print(f"[group] settings from {g_cfg:>24}: held-out MAE " + "  ".join(
            f"{cross[g_cfg][g]:.3f}" for g in keys))
    C = pd.DataFrame(cross).T[keys]
    A = pd.DataFrame(cross_acc).T[keys]
    C.to_csv(os.path.join(out, "group_cross_mae.csv"))
    A.to_csv(os.path.join(out, "group_cross_accel.csv"))

    own = np.array([C.loc[g, g] for g in keys])
    other = np.array([C.loc[[h for h in keys if h != g], g].mean() for g in keys])
    pooled = C.loc["ALL", keys].to_numpy()
    spread = {c: float(np.std(np.log10(params.loc[params.group != "ALL", c])))
              for c in params.columns if c not in ("group", "n_train")}
    lines = [f"# Is the tuning the same across {by}s?", "",
             f"- method `{args.method}`, objective MAE + {args.accel_weight} x accel, "
             f"robust tuned: {args.tune_robust}, sequences per group capped at "
             f"{args.max_per_group or 'no cap'}", "",
             "## Tuned parameters per group", "", V.md_table(params, "{:.4g}"), "",
             "Spread across groups (std of log10; 0.1 = about +-26 %, 0.3 = about 2x):", ""]
    lines += [f"- {k}: {v:.2f}" for k, v in spread.items()]
    lines += ["", "## Held-out MAE: settings tuned on the ROW group, scored on the COLUMN "
              "group's held-out half", "", V.md_table(C.reset_index().rename(
                  columns={"index": "tuned on"}), "{:.3f}"), "",
              "## What it means", "",
              f"- own settings on own data: mean {own.mean():.3f} deg",
              f"- another group's settings: mean {other.mean():.3f} deg "
              f"(transfer cost {other.mean() - own.mean():+.3f} deg)",
              f"- one pooled setting (ALL): mean {pooled.mean():.3f} deg "
              f"({pooled.mean() - own.mean():+.3f} vs own)",
              f"- shipped defaults: mean {C.loc['defaults', keys].mean():.3f} deg", "",
              "If the transfer cost is a few hundredths of a degree, the tuning is effectively "
              "the same everywhere even where the parameter values differ (the objective is "
              "flat), and the pooled setting is the one to ship."]
    open(os.path.join(out, "group_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump({g: jsonable(c) for g, c in tuned.items()},
              open(os.path.join(out, "group_configs.json"), "w"), indent=2)
    print("\n" + "\n".join(lines[-8:]))
    return tuned


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="extracted DexYCB folder (images + labels)")
    src.add_argument("--compact", help="compact cache from dexycb_data.py (no images needed)")
    ap.add_argument("--serial", default=None, help="--root: camera serial")
    ap.add_argument("--subjects", nargs="+", default=None, help="--compact: subject filter")
    ap.add_argument("--serials", nargs="+", default=None, help="--compact: camera filter")
    ap.add_argument("--max-per-group", type=int, default=0,
                    help="--compact: at most N sequences per (subject, camera)")
    ap.add_argument("--clip-len", type=int, default=0,
                    help="cut sequences into windows of N frames (for long HO-3D clips)")
    ap.add_argument("--clips-per-seq", type=int, default=0,
                    help="with --clip-len: keep N evenly spaced windows per sequence")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "validation", "tuning"))
    ap.add_argument("--cache-dir", default=None, help="--root: MediaPipe cache (reused)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--method", default="ekf_corr", choices=list(V.METHODS))
    ap.add_argument("--calib", choices=["gt", "mediapipe"], default="mediapipe",
                    help="hand model for FK/block 2; 'mediapipe' is the realistic live case")
    ap.add_argument("--sigma-mode", choices=["uniform", "perangle"], default="perangle")
    ap.add_argument("--split", choices=["parity", "subject", "camera", "dataset"],
                    default="parity")
    ap.add_argument("--group-by", choices=["subject", "camera", "dataset"], default=None,
                    help="tune per group and cross-evaluate (see the docstring)")
    ap.add_argument("--tune-robust", action="store_true",
                    help="also tune robust_k (and robust_k_kp for the EKF)")
    ap.add_argument("--robust-power", type=float, default=1.0)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--n-refine", type=int, default=60)
    ap.add_argument("--accel-weight", type=float, default=0.0,
                    help="objective = MAE + w * acceleration error (0 = pure MAE; "
                         "0.5 keeps the output smooth, see HANDOFF 3.3b)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mp-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "angle_corrector_v12.pth"))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args._t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    source, filt = V.METHODS[args.method]
    if filt is None:
        raise SystemExit(f"{args.method} has no filter to tune")
    corrector = None
    if source == "corr":
        from pipeline import AngleCorrector
        corrector = AngleCorrector(args.ckpt)
    if args.root and not args.cache_dir:
        args.cache_dir = os.path.join(ROOT, "output", "validation", "dexycb", "mp_cache")
    args.no_corrector = corrector is None
    prepared = V.load_prepared(args, corrector, args.calib)
    if not prepared:
        raise SystemExit("no usable sequences")
    scorer = Scorer(prepared, args.workers)
    print(f"[tune] scoring with {args.workers} worker(s)")
    try:
        if args.group_by:
            tune_groups(args, prepared, scorer, source, filt, args.out)
        else:
            tune_once(args, prepared, scorer, source, filt, args.out)
    finally:
        scorer.close()
    print(f"[tune] results -> {args.out}  ({time.time() - args._t0:.0f} s)")


if __name__ == "__main__":
    main()
