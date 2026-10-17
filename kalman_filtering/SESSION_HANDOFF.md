# Session handoff — 2026-10-03 → 2026-10-06

Read this first for what happened in this session. `HANDOFF.md` has the full
history and every number (this session = §9–§13), `Flow.md` the maths and
assumptions, `README.md` how to run things, `unity/README.md` the Unity viewer.

---

## 0 · Status in one paragraph

The hand-tracking pipeline (MediaPipe → v12 corrector → two-block EKF) is
**finished on the filter side and validated on two datasets**: DexYCB
(5 subjects × 2 cameras) and HO-3D v3 (10 subjects × 5 cameras), 95K evaluated
frames. Final EKF error is **14.0° on DexYCB and 15.6° on HO-3D**, against
21.2° and 22.3° for raw MediaPipe. The corrector provides the accuracy
(about −30 %); the filter provides smoothness (jitter cut 3–6×), robustness
to glitches and gap bridging. Tuning studies showed **one fixed filter
setting serves all people, cameras and both datasets; the shipped defaults
were kept**. MediaPipe now runs in its own thread (about 18 → 26 fps). A Unity
viewer exists (Python side tested; C# never compiled). The next real accuracy
lever is the **corrector's constant bias** (thumb CMC flexion about −20° on
both datasets), not the filter.

---

## 1 · What was done this session

| # | work | result |
|---|---|---|
| 1 | Verified the corrector checkpoint | `models/angle_corrector_v12.pth` = the full-dataset model (80,670 train, FreiHAND test 8.69°), byte-identical to the user's notebook output. Old "trained on 1,489 samples" note was wrong |
| 2 | Explained scale handling | MediaPipe world landmarks are metric; a per-frame palm Umeyama fit (scale EMA) matches them to the calibrated hand; measured scale ≈ 0.995 |
| 3 | User ran `measure_still_noise.py` | median 0.48° (MediaPipe) / 0.25° (corrected), keypoints about 0.2 mm: still-hand noise is tiny, error is motion noise and bias |
| 4 | **Threaded MediaPipe** (`pipeline.DetectionStream`) | 17.8 → 25.9 fps on clip.mp4, byte-identical output on video; webcam keeps only the newest frame. `--no-thread` = old behaviour |
| 5 | **Robust (innovation-weighted) R** (`--robust`, `robust_k`, `robust_k_kp`, `robust_power`) | synthetic spikes 7.49 → 4.92°; on real data no MAE gain, a little smoother at k=3. **Off by default** |
| 6 | **Coast decay** (`coast_decay = 0.3`, **new default**) | velocity/acceleration damped while MediaPipe has lost the hand: coasting error 21.9 → 18.2°, normal frames unchanged |
| 7 | Fixed: coasting left hand drawn un-mirrored | it now keeps the last chirality during dropouts |
| 8 | Fixed: Flow.md axis note | MediaPipe world z points **away** from the camera (angles unaffected; matters for Unity) |
| 9 | **Compact dataset cache** (`validation/dexycb_data.py`) | GT + MediaPipe per sequence, a few KB each; validation, tuning and analysis run without images, on the laptop too |
| 10 | **Tuner upgrades** (`validation/tune_dexycb.py`) | parallel workers, `--split subject|camera|dataset`, `--group-by subject|camera|dataset` (cross-evaluation), `--tune-robust`, `--clip-len/--clips-per-seq`, resume per group, reports the clamped values actually evaluated |
| 11 | **Frame-level error analysis** (`validation/analyse_dexycb_frames.py`) | bias vs frame-to-frame error, error against causes, gaps, wrong labels, glitches, worst frames/sequences, plots |
| 12 | **DexYCB multi-subject** (`validation/dexycb_multi_colab.ipynb`, user ran it) | 5 subjects × 2 cameras, 1000 records cached (`compact.zip`, 34 MB) |
| 13 | **HO-3D v3** (`validation/ho3d_data.py`, `ho3d_colab.ipynb`, `ho3d_colab_part2.ipynb`, user ran both) | auto-download from OneDrive by range requests (no 32 GB zip), joint/axis conversion, static-grasp sequences detected and excluded; 49 moving sequences |
| 14 | **Unity viewer** (`core/unity_stream.py`, `--unity`, `unity/`) | UDP stream verified with a Python listener; `HandUdpReceiver.cs`, `ProceduralHand.cs`, `RiggedHandDriver.cs` written but **not compiled** |

---

## 2 · Final results

### Accuracy (GT calibration; finger-angle MAE)

| | DexYCB s01 | DexYCB 5 subj × 2 cam | HO-3D 10 subj × 5 cam |
|---|---|---|---|
| frames evaluated | 5,765 | 23,564 | 71,986 |
| MediaPipe | 20.67° | 21.24° | 22.28° |
| + corrector | 14.01° | 14.22° | 16.03° |
| **+ EKF (final)** | **13.87°** | **13.97°** | **15.61°** |
| EKF PA-MPJPE | 13.2 mm | 15.8 mm | 16.9 mm |

- Bones from MediaPipe instead of GT ("no calibration"): 14.07° (DexYCB),
  15.58° (HO-3D). Personal calibration is worth about 0.1°.
- On frames where the measurement jumps, the filter is 3.7° (DexYCB) /
  4.4° (HO-3D) better than the corrector.
- Live speed about 26 fps (CPU).

### Tuning conclusions

| question | answer |
|---|---|
| Does tuning carry over to held-out people (DexYCB)? | slightly: 13.81 → 13.61° |
| Per-person settings needed? | no: transfer cost +0.08° |
| Per-camera settings needed? | no: −0.03° |
| Do DexYCB-tuned settings work on HO-3D? | **no**: −0.05° MAE for +50 % jitter |
| Same tuning on both datasets? | opposite preferences (DexYCB jerk about 4e5, HO-3D at the 1e3 bound), but the objective is flat (transfer cost 0.10°) and **current defaults are within 0.03° of the best on average and best on HO-3D** |
| Robust R? | no accuracy gain; a little smoother at k=3 |

**Decision: keep the current defaults.**
`jerk_psd 3e4, sigma_angle 6°, r_scale_angles 2, sigma_xy 15 mm, sigma_z 25 mm,
r_scale_kps 2, adaptive Q on, root_tau 0.08 s, coast_decay 0.3, robust off,
threaded MediaPipe on.`

### Where the remaining error is

- Constant per-clip offsets: **62 %** of the error on DexYCB, 34 % on HO-3D.
  No filter can remove them.
- **Thumb CMC flexion bias about −20° on both datasets**, every subject and
  camera. It is systematic in the corrector and learnable.
- Wrong MediaPipe handedness label on about 6 % of frames: MediaPipe's error
  there is 34° vs 20–21°. Live, this also means the wrong mirror and the wrong
  calibration.
- Left hands +1.1° (DexYCB); first frame after a lost detection +1.5–2.3°;
  closed grasps and hands closer than 0.4 m are worse on HO-3D.
- Some "worst frames" are GT glitches (DexYCB s04 112037 f11–12, GT moving at
  770°/s).

---

## 3 · Where things are

### Code (all in `kalman_filtering/`)

| file | role |
|---|---|
| `core/pipeline.py` | runners; `DetectionStream` (threading), `--unity`, `--robust*`, `--coast-decay`, `--no-thread` |
| `core/hand_ekf.py`, `core/angle_kf.py` | filters; `robust_inflation`, `coast_decay` |
| `core/live_compare.py` | side-by-side live comparisons (threaded too) |
| `core/unity_stream.py` | UDP sender; `--listen` test receiver |
| `validation/dexycb_data.py` | compact cache: `build`, `from-results`, `info`; `load("dirA,dirB")` |
| `validation/validate_dexycb.py` | accuracy; `--compact`, `--clip-len`, `--angle-noise-csv`, robust and coast flags |
| `validation/tune_dexycb.py` | tuning, splits, group cross-evaluation, resume |
| `validation/analyse_dexycb_frames.py` | frame-level error analysis |
| `validation/ho3d_data.py` | HO-3D `index`, `download`, `gt-eval`, `check`, `scan`, `build` |
| `validation/*_colab.ipynb` | `dexycb_multi_colab`, `ho3d_colab`, `ho3d_colab_part2` (generated by `make_*_notebook.py`) |
| `unity/` | Unity scripts + README |
| `tests/selftest.py` | must end "ALL CHECKS PASSED" (includes robust-R checks) |

### Data

| where | what |
|---|---|
| `output/dexycb_compact/` (local) | DexYCB s01, cam 836212060125, GT angles only (from the old results zip) |
| `MyDrive/kalman_filtering_results/dexycb/compact` + `compact.zip` | DexYCB 5 subjects × 2 cameras, 1000 records, with GT joints |
| `MyDrive/kalman_filtering_results/dexycb/multi/` | multi-subject validation, tuning, group and frame reports |
| `MyDrive/kalman_filtering_results/ho3d/compact` | HO-3D, 49 moving sequences |
| `MyDrive/kalman_filtering_results/ho3d/part2/` | transfer test, dataset group tuning, frame analysis |
| `MyDrive/ho3d_v3/` | HO-3D zip index, GT, scan, **raw frames about 15 GB (deletable)** |
| `kalman_filtering/calib/hand_calibration_ekf_{right,left}.json` | the user's own bone lengths (from `bonelengthcalculation/`) |

---

## 4 · Quick commands

```powershell
cd C:\Users\gunem\mihir_projects\studying-2dto3d\kalman_filtering
..\Live_Webcam_testing\myenv\Scripts\Activate.ps1
python tests\selftest.py                                    # ALL CHECKS PASSED
python run_ekf_corrected.py                                 # webcam, final defaults
python run_ekf_corrected.py --unity                         # + stream to Unity
python run_ekf_corrected.py --robust                        # optional glitch down-weighting
python core\unity_stream.py --listen                        # check the Unity stream
python validation\validate_dexycb.py --compact output\dexycb_compact --calib mediapipe
python validation\analyse_dexycb_frames.py --compact output\dexycb_compact
python validation\tune_dexycb.py --compact <dirA>,<dirB> --group-by dataset --clip-len 150 --clips-per-seq 2
```

---

## 5 · Open items, in priority order

1. **Corrector bias correction**, the largest remaining lever (likely 1–3°).
   Learn per-angle offsets (then optionally fine-tune v12) on DexYCB subjects
   01/03/05; test on 02/04 **and** HO-3D. Needs the Drive caches downloaded to
   `kalman_filtering/output/`.
2. **Handedness-label stability**: hysteresis or majority vote within a track,
   so MediaPipe's label (wrong on about 6 % of frames) cannot flip the mirror
   and calibration mid-track.
3. **Compile and test the Unity scripts** in a real editor (Unity 2021.3+).
4. **First git commit**: the repository still has **no commits**.
5. Optional: delete `MyDrive/ho3d_v3/raw` (about 15 GB); not needed for
   anything planned.
6. Not built, low priority: root rotation in the filter state, offline RTS
   smoother, two-hand tracking.

---

## 6 · Traps (each cost time)

- **Smoothness ≠ accuracy; filter tuning is worth ≤ 0.2° and does not travel
  between datasets.** Don't propose more filter tuning.
- **HO-3D single-camera sequences (MC, ND, SM, SMu, SS, SiS) have a frozen
  grasp annotation**: finger angles never change. They are excluded by
  `scan`/`build`.
- **HO-3D's 5 cameras of one recording share the same motion.** Always hold
  out whole subjects or recordings, never single cameras or windows.
- **HO-3D frame rate is assumed to be 30 fps** (not stated by the authors).
  HO-3D's smooth GT inflates jitter/GT ratios.
- **Long HO-3D clips make Colab tuning time out.** Use
  `--clip-len 150 --clips-per-seq 2`; group mode resumes per group.
- **Re-upload `kalman_filtering/` to Drive before any Colab run.** The
  notebooks assert on new options and stop if the copy is stale.
- **OneDrive downloads** use the anonymous Badger token
  (`ho3d_data.share_children`); URLs expire and are refreshed automatically.
- **The tuner's search can step outside the bounds.** The printed values are
  now the clamped ones that were actually evaluated.
- **PowerShell / Git-Bash:** inline here-strings and `\v` in sed broke edits;
  write patch scripts to files.
- **Resume line (verified numbers):** "Validated and tuned the Kalman filter
  on 95K frames across DexYCB and HO-3D (15 subjects, 7 cameras),
  generalizing to unseen users".

---

## 7 · For the next session

Open with: **"Shall I start the corrector bias correction (§5.1), or the
handedness-label fix (§5.2)? Have you tried the Unity viewer?"** Ask the user
to download `MyDrive/kalman_filtering_results/dexycb/compact.zip` and the
`ho3d/compact` folder into `kalman_filtering/output/` before the corrector
work.
