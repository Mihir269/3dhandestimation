#!/usr/bin/env python3
"""
compare_3d_vs_angles.py -- LIVE: Kalman filter on angles only vs angles + MediaPipe 3D.

    LEFT : corrected angles -> angle-only KF
    RIGHT: corrected angles + MediaPipe 3D keypoints -> two-block EKF
           (block 2 fits your calibrated FK skeleton to MediaPipe's 3D points)

Same corrected angles and same process model on both sides. The only
difference is whether MediaPipe's 3D keypoints also go into the filter. On
clip.mp4 the angle-only side was smoother and closer to its input, because
MediaPipe's skeleton proportions do not match your calibration (README,
"Known issue"). Watch nis2 on the right: target ~45.

    python live\\compare_3d_vs_angles.py
    python live\\compare_3d_vs_angles.py --angle-source mediapipe
    python live\\compare_3d_vs_angles.py --sigma-xy 10 --sigma-z 30   # trust the 3D less

Keys: q quit | space pause | n/p plotted angle | a/d rotate | m MediaPipe ghost
      r reset filters | s screenshot.   Recording -> output/live/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from live_compare import Spec, build_parser, run_live  # noqa: E402

if __name__ == "__main__":
    ap = build_parser(__doc__.strip().splitlines()[0])
    ap.add_argument("--angle-source", choices=["corrected", "mediapipe"], default="corrected")
    args = ap.parse_args()
    run_live(Spec(args.angle_source, "angle", "KF: joint angles only"),
             Spec(args.angle_source, "ekf", "EKF: angles + MediaPipe 3D"),
             "3d_vs_angles", args)
