# kalman_filtering — v12 corrector + EKF, with benchmarks

Self-contained. Everything needed to run is inside this folder.

**New here? Read [Flow.md](Flow.md)** — the per-frame flow, the maths, and
every assumption the code makes.

```
kalman_filtering/
├── Flow.md                     how it works + assumptions log
├── run_ekf_corrected.py        MAIN   MediaPipe -> v12 corrector -> two-block EKF
├── run_angle_kf.py             ANGLE-ONLY jerk-minimising KF (no keypoint block)
├── live/                       LIVE side-by-side comparisons on one webcam
│   ├── compare_filter_on_off.py          same angles: no filter | Kalman filter
│   ├── compare_3d_vs_angles.py           angle-only KF | EKF with MediaPipe 3D
│   └── compare_mediapipe_vs_corrector.py KF on MediaPipe | KF on v12 corrector
├── benchmarks/
│   ├── run_raw_mediapipe.py    BENCH A  raw MediaPipe, no filter (the jitter baseline)
│   ├── run_kf_mediapipe.py     BENCH B  same EKF on raw MediaPipe angles + 3D, no corrector
│   └── compare_runs.py         jitter / jerk / bone-CV / lag / NIS table + plot from the CSVs
├── core/
│   ├── anatomical_angles_v12.py  THE canonical angle module (restored copy). Never copy it.
│   ├── hand_angles_v2.py         verified shim: arrays, bone names, units, fan guards
│   ├── hand_ekf.py               PersonalizedHandModel (FK + analytic Jacobian), HandEKF
│   ├── angle_kf.py               AngleKF: 21 independent 3-state angle filters
│   ├── live_compare.py           engine for live/: one detection, two methods, one screen
│   └── pipeline.py               MediaPipe, corrector, per-hand filters, rendering, CSV
├── calibration/
│   └── measure_still_noise.py  still-hand recording → per-angle / per-axis noise → runner flags
├── tests/
│   └── selftest.py             run first: FK/Jacobian/wrap/NIS/checkpoint checks
├── validation/                 ACCURACY against ground truth (DexYCB, 30 fps video)
│   ├── validate_dexycb.py        6 methods × MAE / PA-MPJPE / accel error / jitter / lag
│   ├── validate_dexycb_colab.ipynb  download → extract one camera → run → save to Drive
│   ├── tune_dexycb.py            optimise Q / R₁ / R₂ to minimise MAE, held-out checked
│   └── tune_dexycb_colab.ipynb   the same, as a Colab notebook
├── models/                     angle_corrector_v12.pth, hand_landmarker.task
├── calib/                      hand_calibration_ekf_{right,left}.json
└── output/                     raw/  kf_mp/  ekf/  angles/  live/  benchmarks/
```

## Measured accuracy — DexYCB, 100 sequences (2026-09-17)

Real 30 fps video, per-frame GT 3D joints, one unseen subject, hands holding
objects. 6,928 GT frames, 83 % with a MediaPipe detection. Full detail and
caveats in [Flow.md](Flow.md) §7.5.

| method | angle MAE | PCK@10° | PA-MPJPE | accel err | jitter/GT | lag |
|---|---|---|---|---|---|---|
| MediaPipe raw | 20.67° | 31.6 % | 16.3 mm | 9.90 | 6.89 | 0 |
| corrector, unfiltered | 14.01° | 44.5 % | 16.3 mm | 5.36 | 3.38 | 0 |
| angle KF on MediaPipe | 21.10° | 31.1 % | 17.2 mm | 3.84 | 3.08 | 0 |
| angle KF on corrector | 14.08° | 44.2 % | 16.3 mm | 2.40 | 1.06 | 0 |
| **EKF on corrector** | **13.76°** | **46.4 %** | **13.2 mm** | 2.46 | 1.10 | 0 |

* **The corrector gives the accuracy: −32 % MAE** on an unseen subject.
* **The filter gives the smoothness:** acceleration error halves, jitter goes
  from 3.4× the real motion to ~1.0×, at zero lag. MAE barely moves --
  a Kalman filter removes noise, not bias.
* **Do not filter raw MediaPipe for accuracy** (20.67° → 21.10°): its error is
  persistent, not white.
* `run_ekf_corrected.py` is the best method on every accuracy measure.
* Defaults updated from this run: `jerk_psd` 3e4, `sigma_xy` 15 mm.

## Validation against ground truth (DexYCB)

Open `validation/validate_dexycb_colab.ipynb` in Colab after uploading this
folder to `MyDrive/kalman_filtering`, then run the cells top to bottom. Locally,
with an extracted DexYCB folder:

```powershell
python validation\validate_dexycb.py --root D:\dexycb --max-seqs 5
```

It scores `mp`, `corr`, `kf_angle_mp`, `kf_angle_corr`, `ekf_mp` and `ekf_corr`
on the same frames. Results go to `output/validation/dexycb/`: `report.md`,
`summary.csv`, `per_angle.csv` (MAE and signed bias), `per_sequence.csv`,
`per_frame.csv.gz` and plots. MediaPipe detections are cached, so re-tuning
with `--sigma-angle`, `--jerk-psd` and so on takes seconds.

## Tuning the covariances against ground truth

`validation/tune_dexycb_colab.ipynb` (or the script) searches `jerk_psd` (Q),
`sigma_angle` (R₁) and `sigma_xy` / `sigma_z` (R₂) to minimise MAE:

```powershell
python validation\tune_dexycb.py --root D:\dexycb --method ekf_corr --sigma-mode perangle
```

* `r_scale_*` is pinned to 1 while tuning: it only multiplies the sigmas, so
  tuning both searches a ridge instead of a point.
* The per-angle noise *shape* is estimated from the measured angles alone
  (robust spread of their second differences), never from the ground truth.
  Only one global multiplier is fitted.
* Sequences are split by parity: fitted on one half, **reported on the other**.
* Outputs `tuned_flags.txt` (paste straight onto any runner),
  `tuned_sigma_per_angle.csv` (feeds `--angle-noise-csv`), `trials.csv`,
  `report.md` and a scatter of MAE against each parameter.
* `--accel-weight 0.5` optimises smoothness alongside accuracy; pure MAE
  tends to prefer a filter that barely smooths.

## Measuring noise from a still hand

```powershell
python calibration\measure_still_noise.py
```

Guides you through 5 poses (4 s to get ready, 8 s held still each) and writes
`output/still_noise/<time>/`: `report.md`, `angle_noise.csv`,
`keypoint_noise.csv` and `flags.txt` to paste onto any runner.

⚠️ Still-hand noise is **far smaller** than the error the filter must handle.
On a held fist in `clip.mp4`: 0.58° per angle and 0.21 mm per keypoint,
against the DexYCB-tuned 8.5° and 15 mm. MediaPipe's video mode smooths
landmarks between frames, and a still hand shows no bias or motion noise. So
the recommended `flags.txt` uses the still hand only for the **per-angle
shape** (which angles are noisier) and the depth/image-plane ratio, scaled to
the DexYCB-validated level. The raw values are in `flags_raw.txt`; using them
makes the filter trust MediaPipe almost completely and barely smooth.

## Live comparisons (webcam, two feeds side by side)

```powershell
..\Live_Webcam_testing\myenv\Scripts\Activate.ps1
python live\compare_filter_on_off.py            # LEFT no filter        | RIGHT Kalman filter
python live\compare_3d_vs_angles.py             # LEFT angle-only KF    | RIGHT EKF + MediaPipe 3D
python live\compare_mediapipe_vs_corrector.py   # LEFT KF on MediaPipe  | RIGHT KF on v12 corrector
```

Screen: camera | LEFT 3D hand | RIGHT 3D hand, with a scrolling plot of one
angle underneath (grey raw MediaPipe, orange LEFT, blue RIGHT). Each panel
shows its **rolling jitter** and its ratio to raw MediaPipe, the filter cost,
and NIS. The camera panel shows fps and a timing breakdown. MediaPipe runs
once per frame and both sides get the identical measurement, so any
difference is the method.

Keys: `q` quit · `space` pause · `n`/`p` change the plotted angle · `a`/`d`
rotate · `m` raw MediaPipe ghost · `r` reset filters · `s` screenshot.
Every session is recorded to `output/live/` (`--no-record` to skip).

Useful flags: `--sigma-angle 4` (NIS-tuned on clip.mp4), `--camera 1`,
`--hand right` (the physical hand, if MediaPipe's label is wrong),
`--no-mirror`, `--angle-source mediapipe` (first two scripts),
`--filter ekf` (third script), `--source clip.mp4` to test on a video.

Speed on this machine (CPU, 1280×720 clip): ~14 fps end to end. MediaPipe
takes ~35–40 ms of each frame; the corrector ~1 ms; the angle-only KF
0.2 ms; the two-block EKF 4.5 ms.

All three runners share `core/pipeline.py` and differ only in `mode`, so a
difference in the benchmark can only come from the method:

| script | block-1 input | filter | 3D output |
|---|---|---|---|
| `benchmarks/run_raw_mediapipe.py` | MediaPipe angles | none | MediaPipe world landmarks |
| `benchmarks/run_kf_mediapipe.py` | MediaPipe angles | EKF | FK of filtered angles |
| `run_ekf_corrected.py` | **v12-corrected** angles | EKF | FK of filtered angles |
| `run_angle_kf.py` | corrected (or `--angle-source mediapipe`) | angle-only KF | FK of filtered angles (drawing only) |

## Setup

Use the environment that already has the packages:

```powershell
..\Live_Webcam_testing\myenv\Scripts\Activate.ps1
python tests\selftest.py          # must end with ALL CHECKS PASSED
```

(or `pip install -r requirements.txt` into a fresh venv)

## Run

Webcam (default source 0; image is mirrored by default for webcams):

```powershell
python run_ekf_corrected.py
python benchmarks\run_kf_mediapipe.py
python benchmarks\run_raw_mediapipe.py
```

Fair benchmark — the same video through all three, headless:

```powershell
$clip = "..\Live_Webcam_testing\clip.mp4"
python benchmarks\run_raw_mediapipe.py --source $clip --no-display --tag clip
python benchmarks\run_kf_mediapipe.py  --source $clip --no-display --tag clip
python run_ekf_corrected.py            --source $clip --no-display --tag clip
python benchmarks\compare_runs.py raw=output\raw\clip.csv kf_mp=output\kf_mp\clip.csv ekf=output\ekf\clip.csv
```

Tip: record a webcam session once with the raw script, then feed the saved
camera footage to all three — a live session cannot be replayed.

Keys: `q` quit · `space` pause · `a`/`d` rotate · `m` grey raw-MediaPipe ghost ·
`k` toggle block 2 (live ablation) · `r` reset filter.

Each run writes `output/<mode>/<tag>.mp4` (camera | front | side) and
`<tag>.csv`: `mp_*`, `out_*` angles, `kpmp_*`/`kpout_*` 3D keypoints (mm,
wrist-centred), plus `corr_*`, `std_*`, `nis1`, `nis2`, `resid_mm` for the
filter modes.

## First results — `Live_Webcam_testing/clip.mp4` (991 frames, default knobs)

From `output/benchmarks/clip/benchmark_summary.csv`. No ground truth, so this
measures smoothness and consistency, **not accuracy**.

| run | angle jitter (×raw) | angle jerk | 3D jitter, palm-local mm | dev from corrected | nis1 | nis2 |
|---|---|---|---|---|---|---|
| raw MediaPipe | 8.70 (1.00) | 14.99 | 11.04 | – | – | – |
| KF on MediaPipe | 4.54 (0.52) | 7.68 | 7.72 | – | 27.5 | 25.4 |
| corrector + EKF | 3.12 (0.36) | 5.25 | 6.91 | 5.8° | 24.1 | 30.9 |
| … `--kp-model mediapipe` | 3.07 (0.35) | 5.16 | 6.80 | 4.7° | 16.4 | 18.5 |
| … `--no-block2` | 2.42 (0.28) | 4.00 | 5.46 | 2.8° | 10.1 | – |
| `run_angle_kf.py` (angle-only) | 2.42 (0.28) | 4.00 | 5.46 | 2.8° | 10.1 | – |
| `run_angle_kf.py --angle-source mediapipe` | 4.30 (0.49) | 7.27 | 7.32 | – | 16.9 | – |

The angle-only KF is exactly the two-block EKF with block 2 off, just ~50×
cheaper per frame. NIS₁ = 10 says σ = 6° is too large; `--sigma-angle 4` is
NIS-consistent. See Flow.md §7.3 for the sweep.

Read honestly:
* **On this clip block 2 looks worse** -- more jitter, and 3° further from the
  corrected angles than block 1 alone. ⚠️ **That conclusion did not survive
  ground truth.** On DexYCB (below) block 2 gives the *best* accuracy. A clip
  with no labels can only show deviation and smoothness, never error.
* The constant-jerk model **overshoots on fast flex/extend steps** (visible in
  the plot around t = 6 s and 9 s). Lower `--jerk-psd` trades overshoot and
  jitter for lag.
* `kp_jitter_mm` (camera-oriented) is *higher* for the filters than for raw.
  The root rotation is a per-frame palm fit that is not in the filter state,
  so it jitters. Use the palm-local column for articulation. Putting the
  3-DOF root rotation into the state is the upgrade path.
* **Handedness:** MediaPipe labelled 414 of 946 detected frames *left*,
  from frame 548 on. Checked against the video: that is correct, the clip
  switches to the left hand (the one with the watch). The switch moves to the
  left hand's filter and calibration, which is why `bone_cv_pct` is 2.1 %
  rather than 0.

## Filter knobs (both filter scripts)

| flag | default | meaning |
|---|---|---|
| `--jerk-psd` | 3e4 | deg²/s⁵. Smaller = smoother but laggier (set from DexYCB) |
| `--sigma-angle` | 6 | block-1 angle noise, deg |
| `--angle-noise-csv` | – | per-angle σ (e.g. a tuner's `tuned_sigma_per_angle.csv`) |
| `--sigma-xy` / `--sigma-z` | 15 / 25 mm | MediaPipe keypoint noise (set from DexYCB) |
| `--robust` / `--robust-k` / `--robust-k-kp` / `--robust-power` | off / 3 / 3 / 1 | innovation-weighted R: a measurement more than k sigmas from the filter's prediction gets its R inflated by (d/k)^power, so a one-frame glitch pulls less. Small gain on DexYCB (below) |
| `--coast-decay` | 0.3 | while MediaPipe has lost the hand, velocity and acceleration are multiplied by this each frame instead of extrapolated (coast error 21.9° → 18.2° on DexYCB) |
| `--no-thread` | – | run MediaPipe in the main loop (old behaviour, ~18 instead of ~26 fps) |
| `--unity [HOST:PORT]` | – | stream the filtered hand to Unity over UDP, see `unity/README.md` |
| `--r-scale-angles` / `--r-scale-kps` | 2 / 2 | correlated-blocks hedge (variance multiplier) |
| `--kp-model` | calib | skeleton block 2 fits against: `calib` or `mediapipe` (see below) |
| `--adaptive` / `--no-adaptive` | on | smooth harder when a finger is slow, less when it moves fast (per angle). `--adapt-ref` (120 °/s) sets what counts as fast |
| `--root-tau` | 0.08 s | display-only smoothing of the hand's overall orientation. The root pose is not in the filter state, so without it the drawn hand shakes even when the angles are smooth. Raise to 0.15 for a calmer hand, 0 for the raw per-frame fit |
| `--no-block1` / `--no-block2` | – | ablations |
| `--hand` | auto | override MediaPipe's handedness label |

**Tune by NIS, not by eye.** Targets: `nis1 ≈ 21`, `nis2 ≈ 45`. Far below →
R too large (that block barely contributes); far above → R too small.

## What changed from the 2026-09-05 EKF design, and why

1. **Block 2 measures the 15 finger keypoints, not all 21**, and adds the
   palm alignment's own error covariance to R₂. Palm points carry no angle
   information (in v12 FK they depend only on bones + fan). On a synthetic
   clip with matched noise, the plain-diagonal version made block 2 *worse
   than raw* (12–19° vs 6°, NIS ≈ 210); with the alignment covariance,
   block-2-only is 5.0°, both blocks 2.6°, NIS₂ 47 against a target of 45.
   Aligning on all 21 points was also tried: angle combinations that mimic a
   palm rotation become unobservable and drift (RMSE > 90°).
2. **The corrector is fed MediaPipe's own world frame**, not the y-up view
   frame. `live_hand_reconstruction.py` rotated the landmarks before the
   corrector on the assumption that the model is rotation-invariant. The
   angles are, but the GCN branch reads raw wrist-centred coordinates, and
   the training NPZ was cached in MediaPipe's frame. Confirm against the
   training notebook.
3. **Wrap-aware block-1 innovation** is implemented (`wrap_deg`), tested
   across ±180°.

## Known issue: MediaPipe's skeleton ≠ your calibrated skeleton

Measured on `Live_Webcam_testing/clip.mp4` (ratios to Middle MCP→PIP):
MediaPipe's wrist→MCP bones read **+38–48 %** longer than the calibration,
distal phalanges **+34–60 %**. Consequences:

* FK with calibrated bones cannot fit MediaPipe's points better than
  **~26 mm** mean per finger keypoint; with MediaPipe's own skeleton the floor
  is **~13 mm**.
* Block 2 pulls the angles toward whatever makes the wrong skeleton fit — a
  **bias**, which no Kalman filter removes (visible as a few-degree offset
  from the corrected angles in `benchmark_Index_PIP_Flex.png`).

`--kp-model mediapipe` estimates the block-2 skeleton from the first 15
detections (the displayed hand still uses your calibration). Benchmark both;
whether the calibrated skeleton should drive block 2 at all is an open design
question for the thesis. This is the §16.4 metacarpal-ratio question showing
up on real footage.

## Added 2026-10-03

* **Threaded MediaPipe** (`core/pipeline.py::DetectionStream`, all runners and
  live comparisons): detection of frame k+1 overlaps the rest of frame k.
  clip.mp4 headless: 17.8 → 25.9 fps, output byte-identical. Webcam: the
  queue keeps only the newest frame and dt is the real capture interval.
* **Robust (innovation-weighted) R** and **coast decay**: see the knobs table.
* **Compact DexYCB cache** (`validation/dexycb_data.py`): ~25 KB per
  (sequence, camera) with GT, MediaPipe and hand side, so validation, tuning
  and analysis run without the images, also on this laptop:
  `python validation/validate_dexycb.py --compact output/dexycb_compact`.
  The subject-01 results zip is already converted (`output/dexycb_compact`).
* **Many subjects / cameras**: `validation/dexycb_multi_colab.ipynb` builds the
  cache subject by subject (resumable), then tunes with whole subjects held out
  and per group. `tune_dexycb.py --group-by subject|camera` tunes each group
  separately and cross-evaluates (is the tuning the same everywhere?);
  `--split subject|camera`, `--tune-robust`, `--workers N` (parallel).
* **Frame-level error analysis**: `validation/analyse_dexycb_frames.py`
  (bias vs transient error, error against candidate causes, detection gaps,
  glitches, worst frames and sequences, plots).
* **Unity viewer**: `unity/` (UDP receiver, procedural hand, rigged-hand driver).

## Still open

* Measure σ_xy / σ_z from `mp_keypoints_3d.npz` vs FreiHAND GT (Procrustes).
* Full 21×21 angle covariance for block 1.
* Accuracy (not just smoothness) needs GT sequences; `compare_runs.py`
  reports smoothness, consistency and lag only.
* Kalman filters remove noise, not bias.

## Second dataset: HO-3D v3 (added 2026-10-04)

`validation/ho3d_colab.ipynb` downloads HO-3D v3 from its official OneDrive
share straight into Colab. `validation/ho3d_data.py` reads the 31.9 GB zip's
index over HTTP and fetches only each sequence's RGB frames and annotations
(one range request each, CRC-checked, resumable), and keeps them in
`MyDrive/ho3d_v3/`. It then converts them to the same compact cache, so every
tool above runs on HO-3D unchanged. `--compact a,b` reads several caches
together, and `--split dataset` / `--group-by dataset` tune on DexYCB and score
on HO-3D.

* Joint order: MANO, mapped to MediaPipe. Axes: OpenGL, converted to OpenCV.
  Checked by bone lengths and depth (`ho3d_data.py check`).
* **The single-camera sequences (MC, ND, SM, SMu, SS, SiS) are annotated with
  one fixed grasp**: the finger angles never change. `scan` finds them from the
  annotations alone, and only sequences with moving fingers are used (about 49,
  about 15 GB of frames).
* Frame rate assumed 30 fps (not stated by the authors).
* Local end-to-end test (GPMF10 + MC1): MediaPipe 19.7°, corrector 13.9°,
  EKF 13.8°, close to DexYCB.
* Academic, non-commercial use only; do not redistribute the data.
