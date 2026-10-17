"""
live_compare.py
===============
Live side-by-side comparison of two methods on ONE webcam feed.

Per frame, MediaPipe runs once and the v12 corrector runs once (only if a side
needs it). The same measurement then goes into two independent "sides", each
with its own angle source and filter. The feeds differ only by method.

    screen layout
    +----------------+----------------+----------------+
    |  camera        |  LEFT method   |  RIGHT method  |
    |  + landmarks   |  3D hand + HUD |  3D hand + HUD |
    +----------------+----------------+----------------+
    |  scrolling plot of one angle: raw MediaPipe (grey), left, right   |
    +------------------------------------------------------------------+

Both 3D hands are drawn the same way, whatever the method: FK of that side's
angles with your calibrated bones, turned to match the palm MediaPipe sees.
Differences on screen are the angles, not the drawing.

HUD per side: rolling jitter (RMS 2nd difference of its angles over the last
60 frames, deg/frame^2), jitter relative to raw MediaPipe, filter cost (ms),
NIS where the filter has one.

Keys: q quit | space pause | n / p next / previous plotted angle |
      a / d rotate | m raw MediaPipe ghost | r reset filters | s screenshot
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

CORE = os.path.dirname(os.path.abspath(__file__))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import hand_angles_v2 as HA  # noqa: E402
from angle_kf import AngleKF  # noqa: E402
from hand_ekf import (PALM_IDS, EKFConfig, HandEKF, PersonalizedHandModel,  # noqa: E402
                      RootSmoother, umeyama)
from pipeline import (DEFAULT_CALIB, DEFAULT_CKPT, DEFAULT_MP_MODEL, MIRROR_X,  # noqa: E402
                      OUTPUT_DIR, AngleCorrector, DetectionStream, HandTracker, Renderer3D,
                      draw_landmarks_2d, resolve_hand, to_view)

ANGLES = HA.ARTICULATION_ANGLES
HISTORY = 180                       # frames kept for the plot
JITTER_WINDOW = 60                  # frames for the rolling jitter

FILTER_NAMES = {"none": "no filter", "angle": "angle-only KF",
                "ekf": "EKF: angles + MediaPipe 3D"}
SOURCE_NAMES = {"mediapipe": "MediaPipe angles", "corrected": "v12-corrected angles"}
LEFT_BGR, RIGHT_BGR, RAW_BGR = (60, 160, 255), (230, 200, 70), (140, 140, 140)


@dataclass
class Spec:
    source: str      # 'mediapipe' | 'corrected'
    filter: str      # 'none' | 'angle' | 'ekf'
    label: str = ""

    def title(self):
        return self.label or f"{SOURCE_NAMES[self.source]} + {FILTER_NAMES[self.filter]}"


class Side:
    """One method: an angle source, a per-hand filter (or none), a history."""

    def __init__(self, spec: Spec, models, make_cfg):
        self.spec = spec
        self.models = models
        self.cfg_root_tau = make_cfg().root_tau_s
        self.filters = {}
        if spec.filter != "none":
            cls = AngleKF if spec.filter == "angle" else HandEKF
            self.filters = {h: cls(m, make_cfg()) for h, m in models.items()}
        self.hist = deque(maxlen=HISTORY)
        self.last_seen = {}
        self.R = np.eye(3)
        self._root = RootSmoother()      # unfiltered side: same display smoothing
        self.out = None
        self.skel = None
        self.cost_ms = 0.0
        self.active = None

    def reset(self):
        for f in self.filters.values():
            f.reset()

    def step(self, z, kps_mm, dt, hand, frame_i):
        t0 = time.perf_counter()
        model = self.models[hand]
        if self.spec.filter == "none":
            self.out = None if z is None else np.asarray(z, float)
            if self.out is not None:
                if kps_mm is not None:
                    ids = list(PALM_IDS)
                    _, R, _ = umeyama(model.palm_points[ids], kps_mm[ids])
                    self.R = self._root.update(R, self.cfg_root_tau, dt)
                self.skel = model.fk(self.out) @ self.R.T
            else:
                self.skel = None
            self.active = None
        else:
            f = self.filters[hand]
            if f.initialised and frame_i - self.last_seen.get(hand, frame_i) > f.cfg.max_coast_frames:
                f.reset()                          # idle while the other hand was tracked
            self.last_seen[hand] = frame_i
            est = f.step(z, kps_mm, dt)
            self.out = est
            self.skel = None if est is None else f.skeleton(oriented=True, model=model)
            self.active = f
        self.cost_ms = 0.9 * self.cost_ms + 0.1 * 1e3 * (time.perf_counter() - t0)
        self.hist.append(None if self.out is None else self.out.copy())


def rolling_jitter(hist, n=JITTER_WINDOW):
    """Mean over angles of RMS 2nd difference in the latest unbroken run."""
    run = []
    for a in reversed(hist):
        if a is None:
            break
        run.append(a)
        if len(run) >= n:
            break
    if len(run) < 5:
        return np.nan
    A = np.array(run[::-1])
    d2 = np.diff(HA.wrap_deg(np.diff(A, axis=0)), axis=0)
    return float(np.sqrt(np.mean(d2 ** 2)))


def draw_plot(width, height, series, name):
    """Scrolling time-series of one angle. series: [(deque of arrays|None, idx, bgr, label)]."""
    import cv2
    img = np.full((height, width, 3), 18, np.uint8)
    top, bot, left = 26, height - 12, 50
    vals = [a[idx] for h, idx, _, _ in series for a in h if a is not None]
    if vals:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        pad = max(5.0, 0.1 * (hi - lo))
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = -10.0, 100.0
    for frac in (0.0, 0.5, 1.0):
        y = int(bot - frac * (bot - top))
        cv2.line(img, (left, y), (width - 8, y), (45, 45, 45), 1)
        cv2.putText(img, f"{lo + frac * (hi - lo):5.0f}", (4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    xs = np.linspace(left, width - 8, HISTORY)
    for h, idx, col, _ in series:
        pts = []
        off = HISTORY - len(h)
        for i, a in enumerate(h):
            if a is None:
                if len(pts) > 1:
                    cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
                pts = []
                continue
            y = bot - (a[idx] - lo) / (hi - lo) * (bot - top)
            pts.append((int(xs[off + i]), int(y)))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    cv2.putText(img, f"{name} (deg)   last {HISTORY} frames   n/p: change angle",
                (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    x = width - 8
    for _, _, col, label in reversed(series):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x -= tw + 34
        cv2.line(img, (x, 14), (x + 22, 14), col, 3)
        cv2.putText(img, label, (x + 26, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def build_parser(description):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--source", default=None,
                    help="video file instead of the webcam (for testing)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--panel", type=int, default=420, help="3D panel size, px")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the webcam image")
    ap.add_argument("--hand", choices=["auto", "right", "left"], default="auto",
                    help="the PHYSICAL hand you hold up; auto = trust MediaPipe")
    ap.add_argument("--plot-angle", default="Index_PIP_Flex", choices=ANGLES)
    ap.add_argument("--no-record", action="store_true", help="do not save an mp4")
    ap.add_argument("--no-display", action="store_true", help="headless (video files only)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-thread", action="store_true",
                    help="run MediaPipe in the main loop (the old, slower behaviour)")
    ap.add_argument("--mp-model", default=DEFAULT_MP_MODEL)
    ap.add_argument("--calib-right", default=DEFAULT_CALIB["right"])
    ap.add_argument("--calib-left", default=DEFAULT_CALIB["left"])
    d = EKFConfig()
    g = ap.add_argument_group("filter settings (applied to every filtered side)")
    g.add_argument("--jerk-psd", type=float, default=d.jerk_psd)
    g.add_argument("--sigma-angle", type=float, default=d.sigma_angle,
                   help="4 was NIS-consistent on clip.mp4 (Flow.md 7.3)")
    g.add_argument("--r-scale-angles", type=float, default=d.r_scale_angles)
    g.add_argument("--sigma-xy", type=float, default=d.sigma_xy)
    g.add_argument("--sigma-z", type=float, default=d.sigma_z)
    g.add_argument("--r-scale-kps", type=float, default=d.r_scale_kps)
    g.add_argument("--root-tau", type=float, default=d.root_tau_s,
                   help="seconds of smoothing on the drawn hand's orientation "
                        "(display only, both sides; 0 = raw per-frame fit)")
    adapt = g.add_mutually_exclusive_group()
    adapt.add_argument("--adaptive", dest="adaptive", action="store_true",
                       help="smooth harder when the fingers are slow, less when fast")
    adapt.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    g.set_defaults(adaptive=d.adaptive)
    g.add_argument("--adapt-ref", type=float, default=d.adapt_ref_speed)
    g.add_argument("--coast-decay", type=float, default=d.coast_decay)
    rob = g.add_mutually_exclusive_group()
    rob.add_argument("--robust", dest="robust", action="store_true",
                     help="down-weight measurements that jump further from the filter's "
                          "prediction than --robust-k sigmas")
    rob.add_argument("--no-robust", dest="robust", action="store_false")
    g.set_defaults(robust=d.robust)
    g.add_argument("--robust-k", type=float, default=d.robust_k,
                   help="sigmas of innovation before an angle is down-weighted")
    g.add_argument("--robust-k-kp", type=float, default=d.robust_k_kp,
                   help="same for a block-2 keypoint")
    g.add_argument("--robust-power", type=float, default=d.robust_power,
                   help="R is inflated by (d/k)^power; 1 = Huber")
    return ap


def run_live(left: Spec, right: Spec, name: str, args):
    import cv2

    is_file = args.source is not None
    mirror = (not args.no_mirror) and not is_file
    if args.no_display and not is_file:
        raise SystemExit("--no-display needs --source <video>; a webcam run has no end")

    print(f"=== live comparison: {name} ===")
    print(f"  LEFT : {left.title()}")
    print(f"  RIGHT: {right.title()}")
    print("[hand_ekf] using hand_angles_v2.compute_angles "
          f"({HA.N_ANGLES} articulation angles, v12)")

    models = {}
    for h, path in (("right", args.calib_right), ("left", args.calib_left)):
        if path and os.path.exists(path):
            models[h] = PersonalizedHandModel.from_calibration(path)
    if not models:
        raise SystemExit("no calibration file found; pass --calib-right")
    for h in ("right", "left"):
        models.setdefault(h, next(iter(models.values())))

    def make_cfg():
        return EKFConfig(jerk_psd=args.jerk_psd, sigma_angle=args.sigma_angle,
                         r_scale_angles=args.r_scale_angles, sigma_xy=args.sigma_xy,
                         sigma_z=args.sigma_z, r_scale_kps=args.r_scale_kps,
                         root_tau_s=args.root_tau, adaptive=args.adaptive,
                         adapt_ref_speed=args.adapt_ref, robust=args.robust,
                         robust_k=args.robust_k, robust_k_kp=args.robust_k_kp,
                         robust_power=args.robust_power, coast_decay=args.coast_decay)

    sides = [Side(left, models, make_cfg), Side(right, models, make_cfg)]
    corrector = None
    if "corrected" in (left.source, right.source):
        corrector = AngleCorrector(DEFAULT_CKPT)
        print(f"corrector on {corrector.device}")
    tracker = HandTracker(args.mp_model)

    cap = cv2.VideoCapture(args.source if is_file else args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {'video ' + args.source if is_file else 'camera ' + str(args.camera)}")
    if not is_file:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
    if not (1.0 < src_fps < 1000.0):
        src_fps = 30.0
    stream = DetectionStream(cap, tracker, is_cam=not is_file, mirror=mirror, src_fps=src_fps,
                             max_frames=args.max_frames, threaded=not args.no_thread)
    print(f"MediaPipe: {'main loop' if args.no_thread else 'background thread'}")

    P = args.panel
    rend = Renderer3D(size=P, scale=1.45)
    raw_hist = deque(maxlen=HISTORY)
    plot_i = ANGLES.index(args.plot_angle)
    show_ghost, paused = False, False
    t_prev = time.perf_counter()
    last_t_s = None
    fps = 0.0
    frame_i = 0
    last_hand = "right"
    timing = {}
    t_start = time.perf_counter()
    writer, rec_buf, rec_path = None, [], None
    out_dir = os.path.join(OUTPUT_DIR, "live")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    combo = None

    def open_writer(frame_shape, rate):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}_{stamp}.mp4")
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), rate,
                               (frame_shape[1], frame_shape[0])), path

    try:
        while True:
            if paused:
                k = cv2.waitKey(30) & 0xFF
                if k == ord(" "):
                    paused = False
                    t_prev = time.perf_counter()
                    last_t_s = None
                elif k == ord("q"):
                    break
                continue

            pkt = stream.get()
            if pkt is None:
                break
            frame, det = pkt.frame, pkt.det
            if is_file:
                dt = 1.0 / src_fps
            else:
                dt = 1.0 / 30.0 if last_t_s is None else max(pkt.t_s - last_t_s, 1e-3)
            last_t_s = pkt.t_s
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            tm1 = time.perf_counter()
            kps_mm = mp_ang = corr_ang = None
            geometric = physical = None
            if det is not None:
                geometric, physical = resolve_hand(det["hand"], mirror, args.hand)
                kps_m = det["world"] * (MIRROR_X if geometric == "left" else 1.0)
                mp_ang = HA.compute_angles(kps_m, hand="right")
                kps_mm = kps_m * 1000.0
                if corrector is not None:
                    corr_ang = corrector(kps_m, mp_ang)
                draw_landmarks_2d(frame, det["px"])
            raw_hist.append(mp_ang)
            tm2 = time.perf_counter()

            if physical is not None:
                last_hand = physical           # keep the hand through dropouts
            for s in sides:
                z = corr_ang if s.spec.source == "corrected" else mp_ang
                s.step(z, kps_mm, dt, last_hand, frame_i)

            # ---------------- panels ----------------
            raw_jit = rolling_jitter(raw_hist)
            ghost = None
            if show_ghost and kps_mm is not None:
                g = kps_mm - kps_mm[0]
                ghost = to_view(g * MIRROR_X if geometric == "left" else g)
            panels = []
            for s, col in zip(sides, (LEFT_BGR, RIGHT_BGR)):
                skel = None
                if s.skel is not None:
                    skel = to_view(s.skel * MIRROR_X if geometric == "left" else s.skel)
                jit = rolling_jitter(s.hist)
                lines = []
                if s.active is not None and s.active.initialised:
                    nis = f"nis1 {s.active.last_nis1:5.1f}"
                    if s.spec.filter == "ekf":
                        nis += f"  nis2 {s.active.last_nis2:5.1f}"
                    lines.append((nis + ("  COASTING" if det is None else ""), (170, 220, 170)))
                lines.append((f"filter cost {s.cost_ms:5.2f} ms", (190, 190, 190)))
                ratio = jit / raw_jit if np.isfinite(jit) and raw_jit and np.isfinite(raw_jit) else np.nan
                lines.append((f"jitter {jit:5.2f} deg/f^2   x{ratio:4.2f} of raw MP"
                              if np.isfinite(jit) else "jitter  --", col))
                img = rend.draw([(ghost, "grey"), (skel, "color")],
                                ("LEFT: " if col == LEFT_BGR else "RIGHT: ") + s.spec.title(),
                                lines)
                cv2.rectangle(img, (0, 0), (P - 1, P - 1), col, 2)
                panels.append(img)

            cam = cv2.resize(frame, (int(frame.shape[1] * P / frame.shape[0]), P))
            tm3 = time.perf_counter()
            for key, v in (("mediapipe", pkt.det_ms / 1e3), ("angles+corrector", tm2 - tm1),
                           ("filters+draw", tm3 - tm2)):
                timing[key] = 0.9 * timing.get(key, v) + 0.1 * v
            cv2.putText(cam, "  ".join(f"{k} {1e3 * v:4.1f}ms" for k, v in timing.items()),
                        (10, P - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            hud = f"{fps:4.1f} fps"
            if det is not None:
                hud += f"   {physical} hand" + ("  (image mirrored)" if mirror else "")
            else:
                hud += "   no hand"
            cv2.putText(cam, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            top = np.hstack([cam] + panels)
            plot = draw_plot(top.shape[1], 170,
                             [(raw_hist, plot_i, RAW_BGR, "raw MediaPipe"),
                              (sides[0].hist, plot_i, LEFT_BGR, "LEFT"),
                              (sides[1].hist, plot_i, RIGHT_BGR, "RIGHT")],
                             ANGLES[plot_i])
            combo = np.vstack([top, plot])

            # ---------------- record ----------------
            if not args.no_record:
                if writer is None:
                    if is_file:
                        writer, rec_path = open_writer(combo.shape, src_fps)
                    else:
                        rec_buf.append(combo)
                        if len(rec_buf) >= 40:
                            writer, rec_path = open_writer(combo.shape, float(np.clip(fps, 5, 60)))
                    if writer is not None:
                        for b in rec_buf:
                            writer.write(b)
                        rec_buf.clear()
                        if is_file:
                            writer.write(combo)
                else:
                    writer.write(combo)

            frame_i += 1
            if args.no_display:
                continue
            cv2.imshow(f"live comparison - {name}", combo)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                paused = True
            elif k == ord("n"):
                plot_i = (plot_i + 1) % len(ANGLES)
            elif k == ord("p"):
                plot_i = (plot_i - 1) % len(ANGLES)
            elif k == ord("a"):
                rend.yaw -= 8
            elif k == ord("d"):
                rend.yaw += 8
            elif k == ord("m"):
                show_ghost = not show_ghost
            elif k == ord("r"):
                for s in sides:
                    s.reset()
                print("filters reset")
            elif k == ord("s"):
                os.makedirs(out_dir, exist_ok=True)
                shot = os.path.join(out_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(shot, combo)
                print(f"screenshot -> {shot}")
    finally:
        if writer is None and rec_buf and not args.no_record:
            writer, rec_path = open_writer(rec_buf[0].shape, float(np.clip(fps, 5, 60)))
            for b in rec_buf:
                writer.write(b)
        if writer is not None:
            writer.release()
            print(f"saved recording -> {rec_path}")
        stream.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - t_start
        print(f"processed {frame_i} frames in {elapsed:.1f} s ({frame_i / max(elapsed, 1e-9):.1f} fps)  "
              + "  ".join(f"{k} {1e3 * v:.1f} ms" for k, v in timing.items()))
    return combo
