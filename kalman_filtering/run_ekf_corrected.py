#!/usr/bin/env python3
"""
run_ekf_corrected.py -- the full pipeline.

MediaPipe -> v12 angles -> v12 corrector (models/angle_corrector_v12.pth)
          -> two-block EKF (block 1: corrected angles, block 2: 3D keypoints
             through the calibrated FK model) -> smooth, bone-length-consistent
             3D hand.

    python run_ekf_corrected.py                          # webcam 0
    python run_ekf_corrected.py --source ..\\Live_Webcam_testing\\clip.mp4 --no-display --tag clip
    python run_ekf_corrected.py --angle-noise-csv test_per_angle_improvement.csv
    python run_ekf_corrected.py --kp-model mediapipe     # see README "Known issue"

On the first run read two things only: the startup lines (the
"[hand_ekf] using hand_angles_v2.compute_angles" line must be there) and the
NIS readout (nis1 ~ 21, nis2 ~ 45 when R is right).

Keys: q quit, space pause, a/d rotate, m grey raw MediaPipe ghost,
      k toggle block 2 (live ablation), r reset filter.
Writes output/ekf/<tag>.mp4 and <tag>.csv.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
from pipeline import run  # noqa: E402

if __name__ == "__main__":
    run("ekf")
