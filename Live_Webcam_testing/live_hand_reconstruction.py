#!/usr/bin/env python3
"""
live_hand_reconstruction.py
===========================
Live webcam hand reconstruction with the v12 angle corrector.

    python live_hand_reconstruction.py --ckpt angle_corrector_v12.pth

Left panel  : webcam with MediaPipe's 2D landmarks
Right panel : hand rebuilt by forward kinematics from the CORRECTED joint angles,
              using average bone lengths and an average palm, wrist at the origin

The session is recorded to ./output/live_<timestamp>.mp4 (camera + 3D panel
side by side). Pass --no-record to disable.

Keys:  q quit   m show/hide the raw MediaPipe skeleton (off by default)
       r auto-rotate   a/d yaw   w/s pitch   c toggle camera panel   SPACE pause

Run this LOCALLY -- Colab cannot reach your webcam. Use
video_hand_reconstruction.py for a recorded clip instead.

No temporal filtering is applied. The jitter you see is the raw per-frame
measurement noise; smoothing it is the Kalman filter's job later.
"""

import argparse
import os
import time
import urllib.request

import cv2
import numpy as np

from hand_recon_common import (
    AngleCorrector, HandTracker, Renderer3D, DEFAULT_BONE_MM, DEFAULT_FAN_DEG,
    load_calibration, reconstruct, to_right_hand, to_view_frame, draw_landmarks_2d,
)
from anatomical_angles_v12 import palm_frame

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
             'hand_landmarker/float16/1/hand_landmarker.task')


def ensure_model(path):
    if not os.path.exists(path):
        print(f'downloading HandLandmarker -> {path}')
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='angle_corrector_v12.pth')
    ap.add_argument('--mp-model', default='hand_landmarker.task')
    ap.add_argument('--calib', default=None,
                    help='hand_calibration.json; overrides the average hand')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--panel', type=int, default=540)
    ap.add_argument('--palm', choices=['observed', 'canonical'], default='observed',
                    help="'observed' lets the rebuilt hand rotate with yours; "
                         "'canonical' pins it to a fixed frame so only the pose changes")
    ap.add_argument('--no-mirror', action='store_true',
                    help='do not mirror the webcam image')
    ap.add_argument('--outdir', default='output', help='where recordings are written')
    ap.add_argument('--no-record', dest='record', action='store_false',
                    help='do not save the session to a video file')
    ap.add_argument('--rec-fps', type=float, default=0.0,
                    help='0 = measure the real capture rate over the first '
                         'frames so playback runs at true speed')
    ap.set_defaults(record=True)
    args = ap.parse_args()

    ensure_model(args.mp_model)
    corrector = AngleCorrector(args.ckpt)
    tracker = HandTracker(args.mp_model, running_mode='VIDEO', num_hands=1)

    if args.calib:
        bone_mm, fan_deg = load_calibration(args.calib)
        src = os.path.basename(args.calib)
    else:
        bone_mm, fan_deg = DEFAULT_BONE_MM, DEFAULT_FAN_DEG
        src = 'average hand'
    print(f'model: {args.ckpt}  ({corrector.n_ang} angles, {corrector.device})')
    print(f'hand geometry: {src}')

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {args.camera}')

    rend = Renderer3D(size=args.panel)
    show_mp, spin, show_cam, paused = False, False, True, False
    fps, t_prev, t0 = 0.0, time.time(), time.time()
    last = None

    # Recording. The writer's frame rate is fixed at creation, so the first
    # WARMUP frames are buffered to measure the real capture rate -- otherwise a
    # guessed value makes playback run fast or slow.
    WARMUP = 40
    rec_buf, writer, rec_path = [], None, None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = tracker.detect(rgb, timestamp_ms=(time.time() - t0) * 1000)

            if det is not None:
                kps, mirrored = to_right_hand(det['world'], det['hand'])
                kps = to_view_frame(kps)      # MediaPipe y-down -> y-up
                mp_ang, corr_ang, fan_obs = corrector(kps)
                palm = palm_frame(kps) if args.palm == 'observed' else None
                rec_c = reconstruct(corr_ang, corrector.angle_names, fan_deg,
                                    bone_mm, palm=palm)
                rec_m = reconstruct(mp_ang, corrector.angle_names, fan_deg,
                                    bone_mm, palm=palm)
                last = dict(frame=frame, px=det['px'], rec_c=rec_c, rec_m=rec_m,
                            delta=float(np.mean(np.abs(corr_ang - mp_ang))),
                            hand=det['hand'], mirrored=mirrored)
            else:
                last = dict(frame=frame, px=None)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

        if last is None:
            continue
        if spin:
            rend.yaw = (rend.yaw + 1.2) % 360

        frame = last['frame'].copy()
        if last.get('px') is not None:
            draw_landmarks_2d(frame, last['px'])
            skels = [(last['rec_c'], 'color')]
            if show_mp:
                skels.insert(0, (last['rec_m'], 'grey'))
            labels = [(f"mean |correction| {last['delta']:.1f} deg", (200, 200, 200))]
            if show_mp:
                labels.append(('grey = MediaPipe   colour = corrected', (150, 150, 150)))
            if last.get('mirrored'):
                labels.append(('left hand mirrored to right', (120, 200, 255)))
            panel = rend.draw(skels, labels, title='FK from corrected angles (mm)')
        else:
            panel = rend.draw([], [('no hand detected', (90, 90, 220))],
                              title='FK from corrected angles (mm)')

        cv2.putText(frame, f'{fps:4.1f} fps', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if paused:
            cv2.putText(frame, 'PAUSED', (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2, cv2.LINE_AA)

        # Always build the combined view so the recording keeps a constant
        # frame size even when the camera panel is toggled off on screen.
        h = args.panel
        f2 = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
        combo = np.hstack([f2, panel])

        if args.record and not paused:
            if writer is None:
                rec_buf.append(combo)
                if len(rec_buf) >= WARMUP:
                    rate = args.rec_fps or float(np.clip(fps, 5.0, 60.0))
                    os.makedirs(args.outdir, exist_ok=True)
                    rec_path = os.path.join(
                        args.outdir, f'live_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
                    writer = cv2.VideoWriter(
                        rec_path, cv2.VideoWriter_fourcc(*'mp4v'), rate,
                        (combo.shape[1], combo.shape[0]))
                    for b in rec_buf:
                        writer.write(b)
                    rec_buf.clear()
                    print(f'recording at {rate:.1f} fps -> {rec_path}')
            else:
                writer.write(combo)

        disp = combo if show_cam else panel
        if args.record:
            cv2.circle(disp, (disp.shape[1] - 26, 26), 8, (0, 0, 235), -1)
            cv2.putText(disp, 'REC', (disp.shape[1] - 78, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 235), 2, cv2.LINE_AA)
        cv2.imshow('v12 hand reconstruction', disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('m'):
            show_mp = not show_mp
        elif k == ord('r'):
            spin = not spin
        elif k == ord('c'):
            show_cam = not show_cam
        elif k == ord(' '):
            paused = not paused
        elif k == ord('a'):
            rend.yaw -= 6
        elif k == ord('d'):
            rend.yaw += 6
        elif k == ord('w'):
            rend.pitch -= 6
        elif k == ord('s'):
            rend.pitch += 6

    # flush a session shorter than the warm-up window
    if args.record and writer is None and rec_buf:
        rate = args.rec_fps or float(np.clip(fps, 5.0, 60.0))
        os.makedirs(args.outdir, exist_ok=True)
        rec_path = os.path.join(args.outdir,
                                f'live_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        writer = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 rate, (rec_buf[0].shape[1], rec_buf[0].shape[0]))
        for b in rec_buf:
            writer.write(b)
    if writer is not None:
        writer.release()
        print(f'saved recording -> {os.path.abspath(rec_path)}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()