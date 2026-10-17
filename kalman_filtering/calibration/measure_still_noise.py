#!/usr/bin/env python3
"""
measure_still_noise.py -- measure the filter's measurement noise (R) from a
still hand, and write it in the form the runners read.

Hold your hand still in a few poses. Because the true pose barely changes,
almost all frame-to-frame movement in MediaPipe's output is measurement
noise -- exactly what R1 (angles) and R2 (3D keypoints) are meant to be.

    python calibration\\measure_still_noise.py                    # webcam, 5 guided poses
    python calibration\\measure_still_noise.py --seconds 10 --poses 3
    python calibration\\measure_still_noise.py --source still.mp4 --no-display
    python calibration\\measure_still_noise.py --analyse output\\still_noise\\<run>\\recording.npz

ESTIMATOR
---------
A hand is never perfectly still: it drifts slowly. The noise is estimated
from SECOND DIFFERENCES, which cancel slow drift. For white noise of std s,
Var(2nd difference) = 6 s^2, so the robust estimate is

    s = 1.4826 * MAD(2nd difference) / sqrt(6)

-- the same estimator tune_dexycb.py uses, so the two are directly comparable.
Keypoints are first rigidly aligned (rotation + translation) to each pose's
median shape, so whole-hand wobble is not counted as keypoint noise, then
measured per axis (x, y in the image plane; z = depth).

WHAT IT CAN AND CANNOT TELL YOU
-------------------------------
* It measures NOISE, not BIAS. The corrector's consistent errors do not show
  up on a still hand; DexYCB tuning preferred larger R partly because of that.
* Still-hand noise is a LOWER BOUND: motion blur and fingers hiding each other
  add noise when the hand moves.
* Measured on clip.mp4 (a held fist): 0.58 deg per angle and 0.21 mm per
  keypoint -- 15-70x below the DexYCB-tuned 8.5 deg / 15 mm. MediaPipe's VIDEO
  mode smooths landmarks between frames, so still-hand jitter is tiny.
* So the RECOMMENDED output keeps the per-angle SHAPE (which angles are
  noisier) and scales it to the DexYCB-validated level; the raw values are
  written separately for experiments.

OUTPUTS (output/still_noise/<timestamp>/)
    angle_noise.csv      RECOMMENDED per-angle sigma (still-hand shape, DexYCB
                         level); loads with --angle-noise-csv. Columns are named
                         RMSE_Corr / RMSE_MP because that is what the loader
                         looks for; the values are noise sigmas
    angle_noise_raw.csv  the measured values as-is, plus per-pose columns
    keypoint_noise.csv   per-joint, per-axis sigma in mm
    flags.txt            recommended flags;  flags_raw.txt  raw flags
    report.md, summary.json, recording.npz (raw data, for --analyse)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

import hand_angles_v2 as HA  # noqa: E402
from hand_ekf import EKFConfig, FINGER_IDS, umeyama  # noqa: E402

ANGLES = HA.ARTICULATION_ANGLES
POSES = [
    "Open hand, fingers spread, palm facing the camera",
    "Relaxed hand, fingers half curled",
    "Loose fist",
    "Pinch: thumb tip touching index tip",
    "Hand turned sideways (little finger towards the camera)",
]


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def runs(mask, min_len=8):
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


def sigma_d2(x, segs):
    """Robust white-noise sigma per column from second differences."""
    chunks = [np.diff(x[s], 2, axis=0) for s in segs if len(s) >= 8]
    if not chunks:
        return np.full(x.shape[1:], np.nan)
    d = np.concatenate(chunks)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    return 1.4826 * mad / np.sqrt(6.0)


def unwrap(a):
    return np.degrees(np.unwrap(np.radians(a), axis=0))


def analyse(rec):
    """rec: dict of arrays from a recording -> per-pose and pooled noise."""
    pose_ids = rec["pose"]
    det = rec["detected"].astype(bool)
    out = {"poses": []}
    ang_mp_p, ang_corr_p, kp_p, weights = [], [], [], []
    for pid in np.unique(pose_ids[pose_ids >= 0]):
        sel = np.flatnonzero(pose_ids == pid)
        mask = det[sel]
        segs = runs(mask)
        n = int(sum(len(s) for s in segs))
        if n < 20:
            print(f"[noise] pose {pid}: only {n} usable frames, skipped")
            continue
        mp = unwrap(np.nan_to_num(rec["mp_ang"][sel]))
        s_mp = sigma_d2(mp, segs)
        s_corr = None
        if "corr_ang" in rec and np.isfinite(rec["corr_ang"][sel][mask]).all():
            s_corr = sigma_d2(unwrap(np.nan_to_num(rec["corr_ang"][sel])), segs)

        kps = rec["kps_mm"][sel]
        ref = np.median(kps[mask], axis=0)
        aligned = np.zeros_like(kps)
        for i in np.flatnonzero(mask):
            _, R, t = umeyama(kps[i], ref, with_scale=False)
            aligned[i] = kps[i] @ R.T + t
        s_kp = sigma_d2(aligned.reshape(len(sel), -1), segs).reshape(21, 3)

        drift = float(np.nanmean(np.nanstd(mp[mask], axis=0)))
        out["poses"].append(dict(pose=int(pid), name=str(rec["pose_names"][pid]),
                                 frames=n, sigma_mp=s_mp, sigma_corr=s_corr,
                                 sigma_kp=s_kp, total_std_mp=drift))
        ang_mp_p.append(s_mp)
        if s_corr is not None:
            ang_corr_p.append(s_corr)
        kp_p.append(s_kp)
        weights.append(n)

    if not out["poses"]:
        raise SystemExit("no pose had enough frames with a detected hand")
    # RMS across poses: noise is pose-dependent, and the filter must cope with all of them
    rms = lambda arrs: np.sqrt(np.mean(np.square(arrs), axis=0))  # noqa: E731
    out["sigma_mp"] = rms(ang_mp_p)
    out["sigma_corr"] = rms(ang_corr_p) if ang_corr_p else None
    out["sigma_kp"] = rms(kp_p)
    fk = out["sigma_kp"][list(FINGER_IDS)]
    out["sigma_xy"] = float(np.sqrt(np.mean(fk[:, :2] ** 2)))
    out["sigma_z"] = float(np.sqrt(np.mean(fk[:, 2] ** 2)))
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def write_outputs(out_dir, res):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    s_mp = res["sigma_mp"]
    s_corr = res["sigma_corr"] if res["sigma_corr"] is not None else s_mp
    d = EKFConfig()
    eff_default = d.sigma_angle * np.sqrt(d.r_scale_angles)

    # RECOMMENDED: still-hand SHAPE at the DexYCB-validated LEVEL.
    # MediaPipe's VIDEO mode smooths landmarks between frames, so a still hand
    # shows far less jitter than the error a moving hand actually has (bias +
    # motion noise). Using the raw values as R makes the filter trust every
    # measurement almost completely, i.e. barely filter. So keep which angles
    # are noisier than others, and scale so the median matches the default
    # effective sigma that DexYCB showed works. One factor for both columns
    # keeps the measured MediaPipe / corrected ratio.
    k = eff_default / max(float(np.median(s_corr)), 1e-6)
    scaled = pd.DataFrame({"Angle": ANGLES, "RMSE_Corr": s_corr * k, "RMSE_MP": s_mp * k})
    ang_path = os.path.join(out_dir, "angle_noise.csv")
    scaled.round(4).to_csv(ang_path, index=False)

    floor = 0.3                                   # never hand the filter a ~0 variance
    raw = pd.DataFrame({"Angle": ANGLES, "RMSE_Corr": np.maximum(s_corr, floor),
                        "RMSE_MP": np.maximum(s_mp, floor)})
    for p in res["poses"]:
        raw[f"mp_pose{p['pose']}"] = p["sigma_mp"]
        if p["sigma_corr"] is not None:
            raw[f"corr_pose{p['pose']}"] = p["sigma_corr"]
    raw_path = os.path.join(out_dir, "angle_noise_raw.csv")
    raw.round(4).to_csv(raw_path, index=False)

    kp = pd.DataFrame([dict(joint=j, sigma_x=res["sigma_kp"][j, 0],
                            sigma_y=res["sigma_kp"][j, 1], sigma_z=res["sigma_kp"][j, 2])
                       for j in range(21)])
    kp.round(4).to_csv(os.path.join(out_dir, "keypoint_noise.csv"), index=False)

    # Keypoints: keep the DexYCB-tuned sigma_xy (it also absorbs the FK model
    # mismatch, which a still hand cannot measure); take only the measured
    # depth/image-plane RATIO for sigma_z.
    ratio = res["sigma_z"] / max(res["sigma_xy"], 1e-9)
    sz = max(d.sigma_xy * ratio, d.sigma_xy)
    flags = (f"--angle-noise-csv {ang_path} --r-scale-angles 1.0 "
             f"--sigma-xy {d.sigma_xy:.2f} --sigma-z {sz:.2f} --r-scale-kps {d.r_scale_kps}")
    flags_raw = (f"--angle-noise-csv {raw_path} --r-scale-angles 1.0 "
                 f"--sigma-xy {max(res['sigma_xy'], 0.5):.2f} "
                 f"--sigma-z {max(res['sigma_z'], 0.5):.2f} --r-scale-kps 1.0")
    open(os.path.join(out_dir, "flags.txt"), "w").write(flags + "\n")
    open(os.path.join(out_dir, "flags_raw.txt"), "w").write(flags_raw + "\n")
    lines = [
        "# Still-hand measurement noise", "",
        f"- poses measured: {len(res['poses'])}",
        f"- angle noise, median over angles: MediaPipe **{np.median(s_mp):.2f}°**"
        + (f", corrected **{np.median(res['sigma_corr']):.2f}°**"
           if res["sigma_corr"] is not None else ""),
        f"- keypoint noise (finger joints): sigma_xy **{res['sigma_xy']:.2f} mm**, "
        f"sigma_z **{res['sigma_z']:.2f} mm** (depth / image-plane ratio "
        f"{res['sigma_z'] / max(res['sigma_xy'], 1e-9):.1f}x)",
        f"- current defaults for comparison: effective angle sigma {eff_default:.2f}°, "
        f"sigma_xy {d.sigma_xy} mm, sigma_z {d.sigma_z} mm (DexYCB-tuned)", "",
        "## Per pose", "",
        "| pose | frames | MP angle σ (median) | corrected σ (median) | total spread (drift + noise) |",
        "|---|---|---|---|---|",
    ]
    for p in res["poses"]:
        c = f"{np.median(p['sigma_corr']):.2f}°" if p["sigma_corr"] is not None else "–"
        lines.append(f"| {p['name']} | {p['frames']} | {np.median(p['sigma_mp']):.2f}° | "
                     f"{c} | {p['total_std_mp']:.2f}° |")
    lines += ["", "## Per angle (RMS over poses)", "",
              "| angle | MediaPipe σ | corrected σ |", "|---|---|---|"]
    for i, n in enumerate(ANGLES):
        c = f"{res['sigma_corr'][i]:.2f}" if res["sigma_corr"] is not None else "–"
        lines.append(f"| {n} | {s_mp[i]:.2f} | {c} |")
    lines += ["", "## Use it", "",
              f"**Recommended** (`flags.txt`): the still-hand per-angle shape, scaled so "
              f"its median is the DexYCB-validated {eff_default:.2f}° (factor {k:.1f}x), "
              f"DexYCB sigma_xy, and sigma_z from the measured depth ratio "
              f"({ratio:.1f}x):", "", "```", flags, "```", "",
              "**Raw** (`flags_raw.txt`, for experiments only): the measured values as-is. "
              "The filter will then trust MediaPipe almost completely and barely smooth.",
              "", "```", flags_raw, "```", "",
              "Why not use the raw values: this is noise on a STILL hand. MediaPipe's",
              "video mode smooths landmarks between frames, so a still hand shows much",
              "less jitter than the error a moving hand really has; and a still hand",
              "shows no bias at all. On DexYCB the best settings were an order of",
              "magnitude above still-hand noise for exactly those reasons."]
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(dict(scale_factor=k, flags_raw=flags_raw, sigma_mp=s_mp.tolist(),
                   sigma_corr=None if res["sigma_corr"] is None else res["sigma_corr"].tolist(),
                   sigma_xy=res["sigma_xy"], sigma_z=res["sigma_z"], flags=flags,
                   poses=[dict(name=p["name"], frames=p["frames"],
                               median_sigma_mp=float(np.median(p["sigma_mp"])))
                          for p in res["poses"]]),
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return flags


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
def capture(args):
    import cv2
    from pipeline import (AngleCorrector, HandTracker, MIRROR_X, draw_landmarks_2d,
                          resolve_hand)

    is_file = args.source is not None
    mirror = not is_file and not args.no_mirror
    tracker = HandTracker(args.mp_model)
    corrector = None if args.no_corrector else AngleCorrector(args.ckpt)
    cap = cv2.VideoCapture(args.source if is_file else args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source or 'camera ' + str(args.camera)}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
    if is_file and args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    poses = POSES[:args.poses] if not is_file else ["recorded clip"]
    rec = {k: [] for k in ("pose", "detected", "mp_ang", "corr_ang", "kps_mm", "t")}
    t0 = time.perf_counter()
    frame_i = 0

    def grab():
        nonlocal frame_i
        ok, frame = cap.read()
        if not ok:
            return None, None
        if mirror:
            frame = cv2.flip(frame, 1)
        t_ms = frame_i * 1000.0 / fps if is_file else (time.perf_counter() - t0) * 1000.0
        frame_i += 1
        return frame, tracker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t_ms)

    def put(frame, lines, color=(255, 255, 255)):
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (16, 38 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, txt, (16, 38 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2, cv2.LINE_AA)

    for pid, name in enumerate(poses):
        # get-ready phase (webcam only): not recorded
        if not is_file:
            t_ready = time.perf_counter()
            while time.perf_counter() - t_ready < args.ready:
                frame, det = grab()
                if frame is None:
                    break
                if det is not None:
                    draw_landmarks_2d(frame, det["px"])
                left = args.ready - (time.perf_counter() - t_ready)
                put(frame, [f"Pose {pid + 1}/{len(poses)}: {name}",
                            f"Get into position... recording in {left:3.1f} s",
                            "Then hold completely still"], (80, 220, 255))
                if not args.no_display:
                    cv2.imshow("still-hand noise", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        raise KeyboardInterrupt
        # recording phase
        n_target = int(args.seconds * (fps if is_file else 30.0)) if not is_file or args.frames == 0 \
            else args.frames
        t_rec = time.perf_counter()
        n = 0
        while True:
            if is_file and args.frames and n >= args.frames:
                break
            if not is_file and time.perf_counter() - t_rec >= args.seconds:
                break
            frame, det = grab()
            if frame is None:
                break
            n += 1
            rec["pose"].append(pid)
            rec["t"].append(time.perf_counter() - t0)
            if det is None:
                rec["detected"].append(0)
                rec["mp_ang"].append(np.full(len(ANGLES), np.nan))
                rec["corr_ang"].append(np.full(len(ANGLES), np.nan))
                rec["kps_mm"].append(np.full((21, 3), np.nan))
            else:
                geometric, _ = resolve_hand(det["hand"], mirror, args.hand)
                kps_m = det["world"] * (MIRROR_X if geometric == "left" else 1.0)
                mp_ang = HA.compute_angles(kps_m, hand="right")
                rec["detected"].append(1)
                rec["mp_ang"].append(mp_ang)
                rec["corr_ang"].append(corrector(kps_m, mp_ang) if corrector
                                       else np.full(len(ANGLES), np.nan))
                rec["kps_mm"].append(kps_m * 1000.0)
                draw_landmarks_2d(frame, det["px"])
            if not args.no_display:
                done = (time.perf_counter() - t_rec) / args.seconds if not is_file \
                    else n / max(n_target, 1)
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (16, h - 40), (16 + int((w - 32) * min(done, 1)), h - 20),
                              (60, 200, 60), -1)
                put(frame, [f"Pose {pid + 1}/{len(poses)}: {name}",
                            "RECORDING - hold still" if det is not None else "no hand detected!"],
                    (60, 230, 60) if det is not None else (60, 60, 255))
                cv2.imshow("still-hand noise", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    raise KeyboardInterrupt
        print(f"[noise] pose {pid + 1} '{name}': {n} frames, "
              f"{int(np.sum(np.array(rec['detected'])[-n:])) if n else 0} with a hand")

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["pose_names"] = np.array(poses)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="video of a still hand instead of the webcam")
    ap.add_argument("--start-frame", type=int, default=0, help="video only")
    ap.add_argument("--frames", type=int, default=0, help="video only: frames to use (0 = all)")
    ap.add_argument("--analyse", default=None, help="re-analyse a saved recording.npz")
    ap.add_argument("--poses", type=int, default=5, choices=range(1, len(POSES) + 1))
    ap.add_argument("--seconds", type=float, default=8.0, help="recording time per pose")
    ap.add_argument("--ready", type=float, default=4.0, help="get-ready time per pose")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto",
                    help="the PHYSICAL hand you hold up")
    ap.add_argument("--no-corrector", action="store_true")
    ap.add_argument("--no-display", action="store_true", help="video files only")
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "still_noise"))
    ap.add_argument("--mp-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "angle_corrector_v12.pth"))
    args = ap.parse_args(argv)

    if args.analyse:
        d = np.load(args.analyse, allow_pickle=False)
        rec = {k: d[k] for k in d.files}
        out_dir = os.path.dirname(os.path.abspath(args.analyse))
    else:
        if args.no_display and args.source is None:
            ap.error("--no-display needs --source")
        try:
            rec = capture(args)
        except KeyboardInterrupt:
            raise SystemExit("stopped before finishing; nothing saved")
        out_dir = os.path.join(args.out, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "recording.npz"), **rec)

    res = analyse(rec)
    flags = write_outputs(out_dir, res)
    med_mp = np.median(res["sigma_mp"])
    print(f"\n[noise] angle sigma (median): MediaPipe {med_mp:.2f} deg"
          + (f", corrected {np.median(res['sigma_corr']):.2f} deg"
             if res["sigma_corr"] is not None else ""))
    print(f"[noise] keypoint sigma: xy {res['sigma_xy']:.2f} mm, z {res['sigma_z']:.2f} mm")
    print(f"[noise] results -> {out_dir}")
    print("\nrecommended (still-hand shape at the DexYCB-validated level):"
          f"\n  python run_ekf_corrected.py {flags}")
    print(f"raw measured values (experiments only): see {os.path.join(out_dir, 'flags_raw.txt')}")


if __name__ == "__main__":
    main()
