#!/usr/bin/env python3
"""
run_kf_mediapipe.py -- BENCHMARK B: Kalman filter on MediaPipe alone.

The SAME two-block EKF, tuning and calibration as run_ekf_corrected.py, but
block 1 is fed the RAW MediaPipe angles -- no v12 corrector. Block 2 fuses
MediaPipe's 3D keypoints through the calibrated FK model, as in the main
script. Comparing this against run_ekf_corrected.py isolates what the
corrector adds on top of filtering.

    python benchmarks\\run_kf_mediapipe.py                      # webcam 0
    python benchmarks\\run_kf_mediapipe.py --source ..\\Live_Webcam_testing\\clip.mp4 --no-display --tag clip
    python benchmarks\\run_kf_mediapipe.py --no-block2          # angles-only KF

Keys: q quit, space pause, a/d rotate, m grey raw MediaPipe ghost,
      k toggle block 2 (live ablation), r reset filter.
Writes output/kf_mp/<tag>.mp4 and <tag>.csv.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from pipeline import run  # noqa: E402

if __name__ == "__main__":
    run("kf_mp")
