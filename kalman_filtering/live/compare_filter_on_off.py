#!/usr/bin/env python3
"""
compare_filter_on_off.py -- LIVE: the same angles, without vs with Kalman filtering.

    LEFT : v12-corrected angles, no filter           (the jittery input)
    RIGHT: v12-corrected angles + angle-only KF      (the same input, filtered)

Both sides get the identical corrected angles every frame, so everything you
see differ is the filter. Use --angle-source mediapipe to compare raw MediaPipe
with and without filtering instead.

    python live\\compare_filter_on_off.py
    python live\\compare_filter_on_off.py --angle-source mediapipe
    python live\\compare_filter_on_off.py --sigma-angle 4           # NIS-tuned value

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
    run_live(Spec(args.angle_source, "none", "WITHOUT Kalman filter"),
             Spec(args.angle_source, "angle", "WITH Kalman filter (angle-only)"),
             "filter_on_off", args)
