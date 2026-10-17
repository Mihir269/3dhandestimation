#!/usr/bin/env python3
"""
compare_mediapipe_vs_corrector.py -- LIVE: Kalman filter on MediaPipe vs on the residual model.

    LEFT : raw MediaPipe angles         -> Kalman filter
    RIGHT: v12 residual corrector angles -> the same Kalman filter

Identical filter and settings on both sides. The only difference is whether
the v12 corrector sits in front of the filter, so this isolates what your
model adds on top of filtering.

    python live\\compare_mediapipe_vs_corrector.py                  # angle-only KF (default)
    python live\\compare_mediapipe_vs_corrector.py --filter ekf     # two-block EKF on both sides

Keys: q quit | space pause | n/p plotted angle | a/d rotate | m MediaPipe ghost
      r reset filters | s screenshot.   Recording -> output/live/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from live_compare import Spec, build_parser, run_live  # noqa: E402

if __name__ == "__main__":
    ap = build_parser(__doc__.strip().splitlines()[0])
    ap.add_argument("--filter", choices=["angle", "ekf"], default="angle",
                    help="angle = angle-only KF, ekf = angles + MediaPipe 3D")
    args = ap.parse_args()
    kind = "angle-only KF" if args.filter == "angle" else "EKF"
    run_live(Spec("mediapipe", args.filter, f"MediaPipe + {kind}"),
             Spec("corrected", args.filter, f"v12 corrector + {kind}"),
             "mediapipe_vs_corrector", args)
