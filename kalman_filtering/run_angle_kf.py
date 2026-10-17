#!/usr/bin/env python3
"""
run_angle_kf.py -- joint-angle-only, jerk-minimising Kalman filter.

MediaPipe -> v12 angles -> v12 corrector -> 21 independent per-angle KFs
(state: angle, velocity, acceleration; process: white-noise jerk) -> filtered
angles. The 3D keypoints never enter the estimate; the calibrated FK model is
used only to draw the filtered angles as a hand.

    python run_angle_kf.py                                   # webcam 0
    python run_angle_kf.py --source ..\\Live_Webcam_testing\\clip.mp4 --no-display --tag clip
    python run_angle_kf.py --angle-source mediapipe          # smooth raw MediaPipe instead
    python run_angle_kf.py --jerk-psd 3e4 --sigma-angle 8    # smoother, laggier

Read the NIS readout: nis1 ~ 21 when --sigma-angle / --jerk-psd match the data.
Keys: q quit, space pause, a/d rotate, m grey raw MediaPipe ghost, r reset.
Writes output/angles/<tag>.mp4 and <tag>.csv. Flow.md explains the maths.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from pipeline import run  # noqa: E402

if __name__ == "__main__":
    run("angles")
