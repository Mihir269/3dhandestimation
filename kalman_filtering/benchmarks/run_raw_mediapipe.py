#!/usr/bin/env python3
"""
run_raw_mediapipe.py -- BENCHMARK A: raw, unfiltered MediaPipe.

No corrector, no filter. Angles are v12 angles computed straight from
MediaPipe's world landmarks; the 3D panel draws MediaPipe's own landmarks.
The jitter you see is the baseline the other two scripts are measured against.

    python benchmarks\\run_raw_mediapipe.py                     # webcam 0
    python benchmarks\\run_raw_mediapipe.py --source ..\\Live_Webcam_testing\\clip.mp4 --no-display --tag clip

Keys: q quit, space pause, a/d rotate view.
Writes output/raw/<tag>.mp4 and <tag>.csv (columns: see core/pipeline.py).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from pipeline import run  # noqa: E402

if __name__ == "__main__":
    run("raw")
