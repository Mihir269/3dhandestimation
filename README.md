# Studying 2D → 3D Hand Pose

Experiments in recovering 3D hand pose from a single RGB camera, building on
Zimmermann & Brox, *Learning to Estimate 3D Hand Pose from Single RGB Images* (ICCV 2017),
and MediaPipe Hands. The repo covers the full pipeline: hand segmentation, 2D keypoint
detection, 2D→3D lifting, bone-length calibration, joint-angle reconstruction, and
temporal smoothing with Kalman filters.

## Repository layout

| Folder | What it does |
|---|---|
| [mediapipe_baseline/](mediapipe_baseline) | MediaPipe hand detection on images, videos, webcam and InterHand-style test sets |
| [handsegnet/](handsegnet) | HandSegNet (hand segmentation) — six training attempts using RHD, CPM, Roboflow and FreiHAND data |
| [posenet/](posenet) | PoseNet 2D keypoint network trained on FreiHAND |
| [poseprior/](poseprior) | PosePrior network that lifts 2D keypoints to canonical 3D pose (`train.py`, `evaluate.py`) |
| [models/](models) | Shared model definitions |
| [bonelengthcalculation/](bonelengthcalculation) | Per-user bone-length calibration from images of squares of known size |
| [Live_Webcam_testing/](Live_Webcam_testing) | Live and offline hand reconstruction and anatomical joint angles |
| [kalman_filtering/](kalman_filtering) | Angle corrector + EKF for stable joint angles, validated on DexYCB. See its [README](kalman_filtering/README.md) and [Flow.md](kalman_filtering/Flow.md) |
| [hand_calibration_images/](hand_calibration_images), [final_output/](final_output) | Sample calibration images and result figures |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Requires Python 3.9–3.11 (MediaPipe) and, for training, a CUDA-capable PyTorch install.

## Data

Datasets are **not** included (they are several GB). Download them and place them under
`alldata/`:

- [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- [RHD (Rendered Hand Pose)](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.html)
- Roboflow hand-detection data and your own test images/videos (`alldata/test_images`, `alldata/test_video`, …)
- DexYCB for the Kalman-filter validation (see `kalman_filtering/validation/`)

## Usage

Run scripts from the repository root, since default paths such as `alldata/...` are relative to it.

```bash
# MediaPipe baselines
python mediapipe_baseline/test_images_detection.py
python mediapipe_baseline/video_hand_detection.py
python mediapipe_baseline/webcam_hand_detection.py

# Train the 2D→3D lifting network
cd poseprior && python train.py --dataset freihand --epochs 100

# Kalman-filtered joint angles on a webcam
python kalman_filtering/run_ekf_corrected.py
```

Generated results (`output_*/`, `webcam_screenshots/`, videos, training checkpoints) are git-ignored.

## References

- C. Zimmermann and T. Brox, "Learning to Estimate 3D Hand Pose from Single RGB Images", ICCV 2017.
- Zimmermann et al., FreiHAND, ICCV 2019.
- Chao et al., DexYCB, CVPR 2021.
- MediaPipe Hands, Google.
