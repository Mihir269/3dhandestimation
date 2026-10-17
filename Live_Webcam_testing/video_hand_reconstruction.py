#!/usr/bin/env python3
"""
video_hand_reconstruction.py
============================
Offline: read a video, run MediaPipe + the v12 corrector on every frame, and
write an output video showing the source alongside a 3D matplotlib plot of the
hand rebuilt by forward kinematics from the corrected joint angles.

    python video_hand_reconstruction.py --input clip.mp4 --ckpt angle_corrector_v12.pth

Average bone lengths and an average palm are used, wrist pinned at the origin,
so only the joint angles change between frames. Pass --calib to substitute a
real per-subject hand_calibration.json.

Also writes a per-frame CSV of MediaPipe and corrected angles (--csv).

This one runs fine in Colab -- it needs no camera. In Colab, prefix the shell
call with ! and pass --no-progress if tqdm renders badly.

No temporal filtering: the jitter is raw measurement noise, for the Kalman
filter to absorb later.
"""

import argparse
import os
import urllib.request

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from hand_recon_common import (
    AngleCorrector, HandTracker, DEFAULT_BONE_MM, DEFAULT_FAN_DEG, FINGER_COLOR,
    FINGER_OF, load_calibration, reconstruct, to_right_hand, to_view_frame,
    draw_landmarks_2d,
)
from anatomical_angles_v12 import BONES, palm_frame

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')


def ensure_model(path):
    if not os.path.exists(path):
        print(f'downloading HandLandmarker -> {path}')
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class Plot3D:
    """Persistent matplotlib 3D axes reused across frames (creating a new figure
    per frame is what makes naive versions of this unbearably slow)."""

    def __init__(self, size_px, lim=105.0, cy=78.0, elev=18.0, azim=-70.0, dpi=100):
        self.fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.size_px, self.lim, self.cy = size_px, lim, cy
        self.elev, self.azim = elev, azim
        self.fig.patch.set_facecolor('#111111')

    def render(self, pts, ghost=None, title='', subtitle=''):
        ax = self.ax
        ax.clear()
        ax.set_facecolor('#111111')

        if ghost is not None:
            for a, b in BONES:
                ax.plot(*[[ghost[a][i], ghost[b][i]] for i in range(3)],
                        c='#666666', lw=1.4, alpha=0.65, zorder=1)
        if pts is not None:
            for a, b in BONES:
                ax.plot(*[[pts[a][i], pts[b][i]] for i in range(3)],
                        c=FINGER_COLOR[FINGER_OF[b]], lw=3.0, zorder=3)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18,
                       c=[FINGER_COLOR[FINGER_OF[j]] for j in range(21)], zorder=4)
            ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], s=70,
                       facecolors='none', edgecolors='w', linewidths=1.4, zorder=5)

        # equal spans on all three axes, centred on the hand, so the aspect is
        # honest and the skeleton does not drift around the box between frames
        L = self.lim
        ax.set_xlim(-L, L); ax.set_ylim(self.cy - L, self.cy + L); ax.set_zlim(-L, L)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=self.elev, azim=self.azim)
        for axis, lbl in ((ax.xaxis, 'x (mm)'), (ax.yaxis, 'y (mm)'), (ax.zaxis, 'z (mm)')):
            axis.pane.set_facecolor('#181818')
            axis.pane.set_edgecolor('#333333')
            axis.label.set_color('#999999')
        ax.tick_params(colors='#777777', labelsize=7)
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.set_zlabel('z (mm)', fontsize=8)
        ax.set_title(f'{title}\nwrist at origin | average bone lengths | no filtering',
                     color='#dddddd', fontsize=10, pad=2, linespacing=1.6)
        if subtitle:
            ax.text2D(0.5, 0.895, subtitle, transform=ax.transAxes, ha='center',
                      color='#9adcd0', fontsize=8)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    def close(self):
        plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default=None)
    ap.add_argument('--ckpt', default='angle_corrector_v12.pth')
    ap.add_argument('--mp-model', default='hand_landmarker.task')
    ap.add_argument('--calib', default=None)
    ap.add_argument('--csv', default=None, help='per-frame angle dump')
    ap.add_argument('--panel', type=int, default=640)
    ap.add_argument('--palm', choices=['observed', 'canonical'], default='observed')
    ap.add_argument('--ghost', action='store_true',
                    help='also draw the raw MediaPipe skeleton in grey')
    ap.add_argument('--spin', type=float, default=0.0,
                    help='degrees of azimuth rotation per frame')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--no-progress', action='store_true')
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + '_recon.mp4'
    ensure_model(args.mp_model)
    corrector = AngleCorrector(args.ckpt)
    tracker = HandTracker(args.mp_model, running_mode='VIDEO', num_hands=1)

    if args.calib:
        bone_mm, fan_deg = load_calibration(args.calib)
        src = os.path.basename(args.calib)
    else:
        bone_mm, fan_deg = DEFAULT_BONE_MM, DEFAULT_FAN_DEG
        src = 'average hand'

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.input}')
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if args.max_frames:
        n_tot = min(n_tot, args.max_frames)
    print(f'{args.input}: {n_tot} frames @ {fps:.1f} fps | geometry: {src}')

    P = args.panel
    plot = Plot3D(P)
    writer, rows, n_det = None, [], 0

    rng = range(n_tot if n_tot > 0 else 10 ** 9)
    if not args.no_progress:
        try:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc='rendering')
        except ImportError:
            pass

    for i in rng:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = tracker.detect(rgb, timestamp_ms=i * 1000.0 / fps)

        rec_c = rec_m = None
        sub = 'no hand detected'
        if det is not None:
            n_det += 1
            kps, mirrored = to_right_hand(det['world'], det['hand'])
            kps = to_view_frame(kps)          # MediaPipe y-down -> y-up
            mp_ang, corr_ang, _ = corrector(kps)
            palm = palm_frame(kps) if args.palm == 'observed' else None
            rec_c = reconstruct(corr_ang, corrector.angle_names, fan_deg, bone_mm, palm=palm)
            rec_m = reconstruct(mp_ang, corrector.angle_names, fan_deg, bone_mm, palm=palm)
            d = float(np.mean(np.abs(corr_ang - mp_ang)))
            sub = (f'mean |correction| {d:.1f}deg'
                   + ('   [left hand mirrored]' if mirrored else ''))
            draw_landmarks_2d(frame, det['px'])
            if args.csv:
                r = {'frame': i, 'time_s': i / fps, 'hand': det['hand']}
                r.update({f'mp_{n}': float(v)
                          for n, v in zip(corrector.angle_names, mp_ang)})
                r.update({f'corr_{n}': float(v)
                          for n, v in zip(corrector.angle_names, corr_ang)})
                rows.append(r)

        if args.spin:
            plot.azim = (plot.azim + args.spin) % 360
        panel = plot.render(rec_c, rec_m if args.ghost else None,
                            title='FK from corrected angles', subtitle=sub)

        fh = cv2.resize(frame, (int(frame.shape[1] * P / frame.shape[0]), P))
        combo = np.hstack([fh, panel])
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)

    cap.release()
    if writer:
        writer.release()
    plot.close()

    print(f'\nwrote {out_path}')
    print(f'hand detected in {n_det}/{i + 1} frames ({100 * n_det / max(i + 1, 1):.1f}%)')
    if args.csv and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f'wrote {args.csv}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()