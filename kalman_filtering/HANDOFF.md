# Session handoff — Kalman filtering (2026-09-16 → 2026-10-03)

**Read this first, then `Flow.md` (how the code works + assumptions log) and
`README.md` (how to run).** `../PROJECT_CONTEXT.md` / the Drive copy covers the
project history before this work (v12 angles, corrector training, calibration).

---

## 0 · One-paragraph status

The Kalman filtering pipeline is **built, running end to end on real video and
live webcam, validated against ground truth** (DexYCB, 100 sequences),
**tuned** (the user ran the tuning notebook), and **tuned again for live
smoothness** after the user reported visible webcam jitter. Headline: the
**v12 corrector cuts MediaPipe's angle error by 32 %** (20.67° → 14.01°) on an
unseen subject; the **filter buys smoothness, not accuracy** (jitter ≈ real
hand motion, MAE unchanged); the **EKF with MediaPipe 3D keypoints is the best
method** (13.76°, 13.2 mm PA-MPJPE). The user's last report on the webcam was
"far better than earlier, less jittery" after the root-orientation fix. Still
not run by the user: `calibration/measure_still_noise.py` (5-pose webcam
session). Next agreed step, **not started**: threaded MediaPipe to lift the
live frame rate from ~14 fps.

**Update 2026-10-03:** the corrector in `models/angle_corrector_v12.pth` is
**already the full-dataset model** (80,670 FreiHAND training samples, FreiHAND
test MAE 8.69° vs MediaPipe 19.86°) — verified byte-identical (MD5
`e9f0a207…`) to the checkpoint produced by the user's
`freihand_angle_correction_v12 (2).ipynb`. Every DexYCB number in this file
was therefore measured with the full-data model. The old "retrain on the full
extraction" item is **done**, not open (see §3.6).

**Session 2026-10-03 (see §9 for details).** Built and tested locally:
- threaded MediaPipe: 17.8 → 25.9 fps on clip.mp4, output byte-identical
- robust (innovation-weighted) R
- coast decay: now the default, cuts coasting error 21.9 → 18.2°
- a compact DexYCB cache, so validation and tuning run on the laptop
- a parallel tuner with subject/camera splits and per-group cross-evaluation
- a frame-level error analysis script
- a Unity UDP stream with Python sender plus C# viewer (**C# not compiled**:
  there is no Unity on this machine)

The user ran `measure_still_noise.py`: median 0.48° (MediaPipe) / 0.25°
(corrected), keypoints 0.18 / 0.22 mm. Same message as before: noise is tiny,
bias dominates. **Not yet run:** `validation/dexycb_multi_colab.ipynb`. It
builds the cache for subjects 02–10 and answers "is the tuning the same on
other subjects and cameras?".

---

## 1 · Where everything is

All work lives in `mihir_projects/studying-2dto3d/kalman_filtering/`
(self-contained; nothing imports from other folders).

```
kalman_filtering/
├── HANDOFF.md  Flow.md  README.md  requirements.txt
├── run_ekf_corrected.py     MAIN: MediaPipe → corrector → two-block EKF (webcam/video)
├── run_angle_kf.py          angle-only KF (no 3D block)
├── benchmarks/  run_raw_mediapipe.py  run_kf_mediapipe.py  compare_runs.py
├── live/        compare_filter_on_off.py  compare_3d_vs_angles.py
│                compare_mediapipe_vs_corrector.py      (two feeds, one webcam)
├── calibration/ measure_still_noise.py                 (still hand → noise → flags)
├── validation/  validate_dexycb.py  validate_dexycb_colab.ipynb
│                tune_dexycb.py      tune_dexycb_colab.ipynb
├── core/        anatomical_angles_v12.py (CANONICAL, restored)  hand_angles_v2.py
│                hand_ekf.py  angle_kf.py  pipeline.py  live_compare.py
├── tests/       selftest.py         (must end "ALL CHECKS PASSED")
├── models/      angle_corrector_v12.pth  hand_landmarker.task
├── calib/       hand_calibration_ekf_{right,left}.json
└── output/      raw/ kf_mp/ ekf/ angles/ live/ benchmarks/   (+ validation/, still_noise/)
```

**Python environment:** `..\Live_Webcam_testing\myenv` (torch 2.13 CPU,
mediapipe 1.0.1, OpenCV 5, pandas; **no scipy**). The repo-root `.venv` has
scipy 1.17 + mediapipe 0.10.33 and was used to test the tuner's Nelder-Mead
path. Colab has everything.

**Google Drive (user's):**
- `MyDrive/kalman_filtering/` — code upload for Colab (re-upload after changes)
- `MyDrive/dexycb_s01_cam836212060125.tar` — DexYCB subject-01, camera 836212060125 subset
- `MyDrive/kalman_filtering_results/dexycb/` — `full/`, `quick/`, `mp_cache/` (100 npz), sweeps
- DexYCB source IDs: subject-01 `1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk` (12.4 GB), calibration `1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ`

**Local test assets:** `Live_Webcam_testing/clip.mp4` (991 frames, 1280×720,
30 fps; **right hand until frame ~547, then the LEFT hand** — verified by eye).

---

## 2 · How it works (short — full version in Flow.md §3–4)

1. MediaPipe VIDEO mode → 21 world landmarks (m) + handedness label.
2. `resolve_hand()`: **geometric** hand (mirror x if left → right chirality)
   vs **physical** hand (picks calibration + filter). MediaPipe's label
   follows the image as given, so a selfie-flipped webcam swaps them.
3. v12 angles (21 articulation; 4 fan angles come from calibration).
4. v12 corrector (GCN + angle MLP, residual) — fed MediaPipe's **native** frame.
5. Filter, state `[θ, θ̇, θ̈]` per angle, constant-jerk process model:
   - **Block 1** angles: `H=[I 0 0]`, wrapped innovation.
   - **Block 2** (EKF only) 15 finger keypoints: FK with calibrated bones,
     palm-only Umeyama alignment, analytic Jacobian (×π/180), R₂ =
     per-axis noise **+ alignment error covariance** (full 45×45).
   - Angle-only KF = exactly 21 independent 3-state filters (verified 1e-13°).
6. Output: filtered angles, σ, NIS; 3D hand = FK with calibrated bones.
7. Dropout: predict ≤ 8 frames, then reset. One filter per physical hand.

**Current defaults** (`core/hand_ekf.py::EKFConfig`, set from DexYCB):
`jerk_psd 3e4`, `sigma_angle 6°`, `r_scale_angles 2`, `sigma_xy 15 mm`,
`sigma_z 25 mm`, `r_scale_kps 2`, `max_coast_frames 8`,
`init_std (10°, 200°/s, 2000°/s²)`, plus two live-smoothness additions
(2026-09-28): `root_tau_s 0.08` (display-only smoothing of the hand's overall
orientation — the root pose is not in the state, and this was the main visible
webcam jitter: drawn-hand jitter 20.5 → 13.8) and `adaptive` Q with
`adapt_ref_speed 120`, `adapt_lo 0.02` (angle jitter 2.46 → 2.30, resting
jitter 0.25 → 0.21, +1 frame lag, DexYCB-neutral).

**Tuning outcome (2026-09-28, user ran the notebook):** held-out MAE gains are
~0.1° (0.5–0.9 %). Pure-MAE tuning pushes `jerk_psd` to its upper bound (1e7)
and **wrecks smoothness** (jitter 2.1× GT, NIS 126/310) — do not ship it. The
`--accel-weight 0.5` run is the sensible one: 13.665° held out (vs 13.752°
default) at accel 2.47, settings close to the current defaults. The one thing
worth adopting is its **per-angle sigma CSV**
(`kalman_filtering_results/dexycb/tuning/ekf_corr_smooth/tuned_sigma_per_angle.csv`,
use with `--angle-noise-csv … --r-scale-angles 1.0`).

---

## 3 · Results so far

### 3.1 DexYCB ground truth (subject 01, camera 836212060125, oracle calibration)
6,928 GT frames, 5,765 evaluated (83 % MediaPipe detection), 91.1 %
handedness-label agreement, FK model floor 8.8 mm.

| method | MAE | PCK@10° | PA-MPJPE | accel err | jitter/GT | lag | coast MAE |
|---|---|---|---|---|---|---|---|
| MediaPipe raw | 20.67° | 31.6 % | 16.3 mm | 9.90 | 6.89 | 0 | – |
| corrector, unfiltered | 14.01° | 44.5 % | 16.3 mm | 5.36 | 3.38 | 0 | – |
| angle KF on MediaPipe | 21.10° | 31.1 % | 17.2 mm | 3.84 | 3.08 | 0 | 39.7° |
| angle KF on corrector | 14.08° | 44.2 % | 16.3 mm | 2.40 | 1.06 | 0 | 22.6° |
| EKF on MediaPipe | 20.99° | 31.3 % | 14.9 mm | 3.87 | 3.33 | 0 | 39.8° |
| **EKF on corrector** | **13.76°** | **46.4 %** | **13.2 mm** | 2.46 | 1.10 | 0 | 23.8° |

**Realistic calibration** (block-2 skeleton from MediaPipe, not GT; re-run
locally from the results zip): EKF on corrector 13.91° at the old defaults,
**13.83–13.86°** after the grid that set the current defaults; angle-only 14.08°.

Conclusions:
- Corrector = accuracy. Filter = smoothness. MAE moves ±0.3° across every
  filter setting tried: the remaining error is **bias**, which no KF removes.
- Filtering raw MediaPipe does **not** improve MAE (its error is persistent).
- MediaPipe angle noise ≈ 2–3× the corrector's (NIS₁ 64 vs 11.5 at equal R).
- **NIS and MAE disagree** once the FK model is imperfect: best MAE had
  NIS₂ ≈ 9.5 (target 45). With ground truth, tune by MAE, not NIS.

### 3.2 Live smoothness work (2026-09-28) — the user's jitter complaint

The user ran the webcam and reported it "still feels a little jittery". Two
causes were found and fixed; the user then reported it "far better".

**(a) The root orientation was never filtered.** The drawn hand's rotation is
re-fitted per frame by Umeyama on 6 palm points, so it carried raw MediaPipe
noise even when the angles were smooth — and it is invisible to every metric
reported before, because DexYCB scoring removes orientation. Fixed with
`RootSmoother` (quaternion EMA, time-constant based so it is frame-rate
independent), display-only: block 2 still uses the raw fit, so the estimate and
NIS are untouched.

| drawn-hand jitter, camera frame (clip.mp4) | mm/frame² |
|---|---|
| raw MediaPipe | 17.38 |
| filtered, before the fix | 20.46 ← **worse than raw** |
| filtered, `root_tau_s 0.08` (default) | 13.77 |
| filtered, `root_tau_s 0.15` | 12.02 |

**(b) Adaptive process noise** (`adaptive`, on by default): per-angle `q` is
scaled by `clip((|θ̇| / 120)², 0.02, 8)` from that angle's own smoothed speed —
the 1-euro idea applied to Q. Measured **at matched fidelity to the input**,
~6 % less jitter than the equivalent fixed `q`; against the previous default,
angle jitter 2.46 → 2.30, resting jitter 0.254 → 0.212, cost +0.8° deviation
and one frame of lag. DexYCB-neutral (13.87° vs 13.86°), because those
sequences are almost always moving.

**Current with/without comparison** (clip.mp4, current defaults):

| | no filter | with filter |
|---|---|---|
| angle jitter | 8.70 | 2.30 (0.26×) |
| angle jerk | 14.99 | 3.89 |
| hand jitter, fingers only | 11.04 | 5.21 |
| hand jitter, as drawn | 17.38 | 13.63 |
| bone-length consistency | 10.85 % | 2.13 % |
| lag | – | 2 frames (66 ms) |

Honest note: much of (b) is equivalent to simply lowering `jerk_psd`
(`--jerk-psd 1e4` gets most of it); adaptive's advantage is calm at rest
without being sluggish when moving.

### 3.3 Earlier smoothness-only runs (clip.mp4, no GT) — superseded
Angle jitter vs raw: KF on MediaPipe 0.52×, EKF on corrector 0.36×,
angle-only on corrector 0.28×. These suggested block 2 hurts; **DexYCB
overturned that.** Never conclude accuracy from smoothness.

### 3.3b Tuning results (2026-09-28, user ran `tune_dexycb_colab.ipynb`)

98 sequences, 49 train / 49 held out, MediaPipe-derived hand model.

| run | held-out MAE | accel err | NIS |
|---|---|---|---|
| corrector, unfiltered | 14.014° | 5.36 | – |
| angle-only, current defaults | 14.030° | 2.36 | 11 |
| angle-only, MAE-tuned | 13.960° | 3.08 | 12 |
| EKF, current defaults | 13.752° | 2.34 | sane |
| EKF, MAE-tuned | 13.631° | 3.65 | 126 / 310 |
| EKF, `--accel-weight 0.5` | 13.665° | 2.47 | 17 / 21 |

* MAE gains are ~0.1° (0.5–0.9 %) — tuning does not fix accuracy.
* **Pure-MAE tuning pushed `jerk_psd` to its upper bound (1e7)** and wrecked
  smoothness (jitter 2.1× GT). Do not ship those flags.
* The `--accel-weight 0.5` result is the sensible one and lands close to the
  current defaults.
* Worth adopting: its **per-angle sigma CSV**, in Drive at
  `kalman_filtering_results/dexycb/tuning/ekf_corr_smooth/tuned_sigma_per_angle.csv`
  → `--angle-noise-csv <file> --r-scale-angles 1.0`. **Not yet wired in as a
  default.** A copy is now in the repo at `kalman_filtering/tuned_sigma_per_angle.csv`.
* Signal-estimated angle noise: median 1.66° vs the ~8.5° that works against
  GT — same message as the still-hand result: noise is small, bias dominates.

### 3.4 Still-hand noise (clip.mp4 held fist, 40 frames — test only)
0.58° (MediaPipe) / 0.36° (corrected) per angle; 0.21 mm (xy) / 0.30 mm (z)
per keypoint — **15–70× below** the DexYCB-tuned values. MediaPipe's VIDEO
mode smooths landmarks; still hands show no bias or motion noise. Hence
`measure_still_noise.py` recommends the still-hand **shape**, scaled to the
DexYCB level; raw values only in `flags_raw.txt`.

### 3.5 Speed (CPU)
MediaPipe 35–40 ms/frame (the bottleneck; downscaling did not help) · v12
angles 3.6 ms · corrector 0.7 ms · angle-only KF 0.03–0.2 ms · two-block EKF
0.85–4.5 ms · FK 0.36 ms. Live scripts ≈ 14 fps, and the stages run
sequentially — MediaPipe on its own thread should give 20–25 fps, which also
lets the filter smooth harder at the same lag.

### 3.6 Corrector model check (2026-10-03)

The user supplied `Downloads/freihand_angle_correction_v12_(2).ipynb` and
`Downloads/angle_corrector_v12 (2).pth`.

* **Same model.** MD5 `e9f0a207af2c9c610fa793663a6debee` for the supplied
  checkpoint, `kalman_filtering/models/angle_corrector_v12.pth` and
  `Live_Webcam_testing/angle_corrector_v12.pth`: all three identical.
* **Trained on the full extraction.** The notebook prints "USING FULL DATASET:
  freihand_master_data". It uses 130,240 MediaPipe runs (4 backgrounds × 32,560
  scenes). 100,798 are valid (22.6 % no detection; the failure rate rises with
  the background block, 5 % → 46.5 %), minus 32 rejected samples. The split is
  by scene: **train 80,670 · val 20,096 · test 3,696**. The checkpoint holds
  `best_val_mae 8.950`, 138k parameters, and the `config` values (scene split,
  seed 42, right hand) match the notebook.
* **FreiHAND results:** train 20.59 → 8.62°, val 20.47 → 8.95°, **test
  19.86 → 8.69° (−56 %)**. The worst angles after correction are Pinky
  MCP/PIP flexion (12.6–13.0°); the best are the abduction angles (5.5–7.4°).
* **The input encoding matches training.** `AngleCorrector._encode` /
  `_nodes` in `core/pipeline.py` reproduce the notebook's `encode_angles` /
  `norm_landmarks`: clip(−180, 360) → [z-score | sin | cos], wrist-centred
  landmarks divided by the Middle MCP→PIP length (kp 9→10), no rotation. That
  **confirms Flow.md A2**: the corrector sees MediaPipe's native frame.
* **What this means.** The 1,489-sample figure quoted earlier was the old
  partial cache and is wrong for the shipped model. The gap between FreiHAND
  test (8.7°) and DexYCB (14.0°) is a **domain gap** (other subject, camera,
  viewpoints, motion, object grasps), not too little training data.

### 3.7 Scale of MediaPipe's 3D keypoints in block 2 (2026-10-03)

* Block 2 uses MediaPipe's **world landmarks**, which are metric (metres,
  origin near the hand centre). It does **not** use pixels: pixels only draw
  the overlay. The pipeline converts to mm (`kps_m * 1000`,
  `pipeline.py`).
* MediaPipe's world-landmark size comes from its own learned hand model, not
  from true depth, so it does not have to match the user's real hand. Block 2
  handles this with the **Umeyama similarity fit on the 6 palm points**. That
  fit estimates a scale `s` (EMA 0.05), and the prediction is
  `s · R · FK(θ) + t`, so the calibrated FK hand is resized to MediaPipe's
  hand before the comparison.
* Measured on clip.mp4 (`output/ekf/clip*.csv`, column `scale`): median
  **s ≈ 0.995**, IQR 0.93–1.01, minimum 0.64 around the right→left switch. So
  MediaPipe's hand size matches the user's calibrated hand closely.
* What scale cannot fix is **proportion** (A18: MediaPipe's wrist→MCP is
  38–48 % longer relative to the fingers). Mean block-2 residual is 27.9 mm
  with the calibrated skeleton and 18.8 mm with `--kp-model mediapipe`. A
  single uniform scale cannot remove a per-bone mismatch.

---

## 4 · Decisions taken (and why) — full table in Flow.md §6

1. Block 2 on **15 finger keypoints**, palm points used only for alignment;
   **alignment covariance added to R₂** (without it block 2 was worse than raw
   on synthetic data, NIS ≈ 210; with it NIS 47 vs target 45).
2. Umeyama on all 21 points rejected (unobservable directions drift, >90°).
3. Corrector fed MediaPipe's native frame (GCN branch not rotation-invariant).
4. Angle-only filter implemented as 21 × 3-state filters (identical, cheaper).
5. `r_scale_*` pinned to 1 when tuning (degenerate with the sigmas).
6. Tuner's per-angle noise shape comes from the signal (MAD of 2nd
   differences), never from GT; parity split of sequences, held-out reporting.
7. Defaults changed on DexYCB evidence: jerk 1e5→3e4, σ_xy 6→15 mm.
8. `--hand` means the **physical** hand.
9. Root orientation smoothed for **display only**, not inside the filter
   (§3.2a): cheap, cannot affect the validated estimate. Putting rotation in
   the state is the principled version and is still open.
10. Adaptive Q shipped **on** because it is DexYCB-neutral and helps at rest;
    `--no-adaptive` restores the exact validated behaviour.
11. Tuned MAE-optimal flags **rejected** (they destroy smoothness, §3.3b).

---

## 5 · Open items, in priority order

0a. **(2026-10-04) Run `validation/ho3d_colab.ipynb`** (HO-3D v3, second dataset,
   see §10). Can run before or after the DexYCB multi notebook; its last step
   compares the two only if the DexYCB cache exists in Drive.
0. **(2026-10-03) Run `validation/dexycb_multi_colab.ipynb`.** Re-upload
   `kalman_filtering/` first. It produces the multi-subject numbers, pooled
   tuning with subjects held out, and the group cross-evaluation (is the
   tuning the same everywhere?). Then decide whether `--robust` with the tuned
   k becomes a default. Also: compile the Unity scripts in a real editor;
   investigate left-hand / wrong-label error (§9).
1. ~~Threaded MediaPipe~~ **done 2026-10-03** (§9).
2. **Adopt the tuned per-angle sigma CSV** (§3.3b): copy it out of Drive into
   the repo, re-check it on the clip and on DexYCB, then decide whether it
   becomes a default.
3. **Run `calibration/measure_still_noise.py` on the webcam** (5 poses), then
   compare its per-angle shape with the tuner's signal-based estimate.
4. **Second DexYCB subject** to confirm the numbers generalise
   (one subject, one camera so far).
5. **Per-angle error is dominated by MCP/PIP flexion (15–19°)** vs abduction
   (5–8°) — a corrector issue, not a filter one. ~~Retrain v12 on the full
   extraction~~ **done**: the shipped model was already trained on 80,670
   samples (§3.6). The remaining lever is the **domain gap** (FreiHAND test
   8.7° vs DexYCB 14.0°). Options: fine-tune or evaluate on DexYCB/HO-3D
   subjects other than s01, augment with rotations and viewpoints, or add
   temporal input.
6. Validate the **user's own calibration**: needs GT of the user's hand
   (DexYCB cannot test it).
7. Not built: offline RTS smoother; **root rotation inside the filter state**
   (only display-smoothed today); innovation gating for MediaPipe spikes;
   full 21×21 R₁; two-hand tracking (`num_hands=1`).
8. Housekeeping: `Live_Webcam_testing/anatomical_angles_v12.py` is **still the
   overwritten shim** (broken) and that folder's `hand_angles_v2.py` is the old
   guessing version — untouched on purpose; the good copies are in `core/`.
   **The repo has no commits** — make a first commit before anything else.
   The Drive copy of `kalman_filtering/` also lags the local one (this bit the
   user once: `ModuleNotFoundError: tune_dexycb` in Colab) — **re-upload before
   any Colab work.**

---

## 6 · Traps (each has cost time)

- **Smoothness ≠ accuracy.** Block 2 looked harmful on the clip and was best on GT.
- **Perceived jitter ≠ measured jitter.** Every metric reported for two weeks
  ignored the hand's overall orientation, which was the jitter the user
  actually saw. When someone says "it looks jittery", ask whether the
  *fingers* or the *whole hand* shakes.
- **Do not chase MAE with filter parameters.** Two independent tuning runs put
  the gain at ~0.1°, and the MAE-optimal settings are visibly worse to use.
- **Kalman removes noise, not bias.** Don't expect MAE gains from tuning.
- **Still-hand noise is not R.** 15–70× too small; use its shape only.
- **Wrap every angle difference** (`wrap_deg`); Thumb_IP_Flex spans ±180°.
- **Jacobian is mm per radian, state is degrees** → ×π/180 (already done).
- **Mirroring:** selfie-flipped webcam ⇒ MediaPipe's label is the opposite of
  the physical hand. Only the unmirrored direction was verified on video.
- **Self-test pins its own `jerk_psd=1e5`** because its synthetic motion is
  faster than real hands; the shipped default would lag it.
- **MediaPipe VIDEO mode needs strictly increasing timestamps** and a fresh
  tracker per independent sequence.
- **`load_angle_sigma_csv`** needs an `Angle` column and one containing
  `RMSE` + `Corr`/`MP`; the noise CSVs use those names even though the values
  are sigmas.
- **PowerShell:** inline here-strings containing `strip("\n")` were blocked by
  a safety check — write generator scripts to files instead.
- **Notebook convention:** every code cell is `ast.parse`-validated before
  handover (shell calls go through a `sh()` helper, not `!`).

---

## 7 · Quick commands

```powershell
cd C:\Users\gunem\mihir_projects\studying-2dto3d\kalman_filtering
..\Live_Webcam_testing\myenv\Scripts\Activate.ps1
python tests\selftest.py                              # ALL CHECKS PASSED
python run_ekf_corrected.py                           # main, webcam
python run_ekf_corrected.py --root-tau 0.15 --jerk-psd 1e4   # calmer, laggier
python run_ekf_corrected.py --no-adaptive             # exact validated behaviour
python live\compare_filter_on_off.py                  # no filter | filter
python live\compare_3d_vs_angles.py                   # angle-only | EKF + 3D
python live\compare_mediapipe_vs_corrector.py         # KF on MP | KF on corrector
python calibration\measure_still_noise.py             # still-hand noise → flags
python validation\validate_dexycb.py --root <dexycb> --max-seqs 5
python validation\tune_dexycb.py --root <dexycb> --method ekf_corr --sigma-mode perangle
```

Keys in live windows: `q` quit · `space` pause · `n`/`p` plotted angle ·
`a`/`d` rotate · `m` MediaPipe ghost · `k` toggle block 2 · `r` reset · `s` screenshot.

---

## 9 · Session 2026-10-03: what was added, with numbers

All numbers on DexYCB s01 / cam 836212060125, MediaPipe-derived calibration,
reproduced locally from the compact cache (mp 20.67, corr 14.01, EKF 13.87:
identical to the Colab run).

| item | where | result |
|---|---|---|
| 1 threaded MediaPipe | `pipeline.DetectionStream`, every runner + `live/*`, `--no-thread` | 17.8 → 25.9 fps (runner), 17.3 → 25.7 (live compare), byte-identical CSV on video |
| 3 robust R | `EKFConfig.robust*`, `--robust --robust-k --robust-k-kp --robust-power` | synthetic spikes 7.49 → 4.92°; DexYCB k=3: 13.87 → 13.85°, jitter/GT 0.98 → 0.90; tuned (k 1.5, k_kp 10 = keypoints ungated, jerk 3.5e5): held out 13.741 → **13.593°** at equal accel (plain re-tune: 13.605°, accel 2.36). Off by default |
| 6 → fix | `EKFConfig.coast_decay = 0.3` (**new default**), `--coast-decay` | coasting MAE 21.9 → 18.2°, detected MAE 13.87 → 13.88° |
| 2/4 multi-subject | `validation/dexycb_data.py` (compact cache), `dexycb_multi_colab.ipynb` (generated by `make_multi_notebook.py`), `tune_dexycb.py --compact --split subject\|camera --group-by subject\|camera --tune-robust --workers N` | group mode tested on fake 2-camera data only; **real run pending** |
| 6 analysis | `validation/analyse_dexycb_frames.py` → `output/validation/frame_analysis/{default,robust_tuned}/` | see below |
| 5 Unity | `core/unity_stream.py`, `--unity`, `unity/` (README, HandUdpReceiver, ProceduralHand, RiggedHandDriver) | Python stream verified with a listener; C# untested |

**What the frame analysis says (s01):**
- 64 % of the filter's squared error is per-sequence **bias**, 36 % transient.
  No Kalman setting reaches the bias.
- Worst angles: Thumb CMC/IP flexion, about −21° bias, and Thumb MCP
  flexion/abduction, about ±18°. These are consistent offsets: the corrector
  under-corrects the thumb on DexYCB.
- **Left hands are worse**: 15.05° vs 12.99°. All 10 worst sequences are left.
- On the 7 % of frames where MediaPipe's handedness label is wrong, its error
  is 37° vs 19°. Live, a wrong label also means a wrong mirror, so this is a
  real webcam risk.
- Error is highest right after re-detection: 15.7° on the first frame back vs
  13.6° after 10 frames.
- Error correlates with measurement jumps and NIS (rho ≈ 0.2–0.37), not with
  hand speed or closure.
- The filter already refuses to follow glitches: on the 3 % jump frames it is
  3° better than the corrector.

**Two corrections made along the way:**
- MediaPipe world z points **away** from the camera (Flow A1 had it the other
  way; angles are unaffected).
- In `pipeline.run` a coasting left hand was drawn un-mirrored, a flip on every
  dropout. It now keeps the last chirality.

## 10 · HO-3D v3 as a second dataset (2026-10-04)

- **Code.** `validation/ho3d_data.py` has commands `index`, `download`,
  `gt-eval`, `check`, `scan` and `build`. The notebook is
  `validation/ho3d_colab.ipynb`, generated by `make_ho3d_notebook.py`.
- **Download.** It comes from the official OneDrive share. The share is
  resolved anonymously with OneDrive's web-viewer token, and the URLs refresh
  themselves. Each (split, sequence, rgb|meta|depth) block in the 31.9 GB zip
  is contiguous, so each one is a single range request. Depth (14 GB) is never
  downloaded. Every file is CRC-checked. The downloads are kept in Drive at
  `MyDrive/ho3d_v3/raw/<split>_<seq>.tar`, `gt/` and `zip_index.json.gz`.
- **Conversion.**
  - Joints: MANO order → `MANO_TO_MP`, verified with tips next to their chains
    and plausible bone lengths.
  - Axes: OpenGL → OpenCV via (x, −y, −z). This is a rotation, so chirality is
    unchanged.
  - The released evaluation GT (`evaluation_xyz.json`) is also in MANO order,
    which is detected automatically.
- **Finding: static grasps.** The single-camera sequences (MC, ND, SM, SMu1,
  SMu40–42, SS, SiS; SM1 in evaluation) have one fixed grasp for the whole
  clip, with 1–2 distinct finger poses. Only the multi-camera recordings have
  moving fingers (0.26–1.36°/frame): ABF, BB, GPMF, GSF, MDF, SB, ShSu, SiBF,
  plus AP and MPM in evaluation. That is about 49 sequences. `scan`/`build`
  exclude static clips by default. The 5 cameras of one recording share the
  same motion, so always hold out whole recordings (`--split subject`).
- **Local test** (GPMF10 + MC1; the downloads were deleted afterwards):
  - MediaPipe 19.7°, corrector 13.9°, EKF 13.8°, lag 2 frames
  - DexYCB-like numbers, on a different lab and people
- **Tooling additions.**
  - `dexycb_data.load` accepts `"dirA,dirB"`.
  - The tuner has `--split dataset` and `--group-by dataset` (DexYCB vs HO-3D).
- Frame rate is assumed to be 30 fps; the authors don't state it.

## 11 · Multi-subject DexYCB results (user's Colab run, 2026-10-04)

- **Data and setup.** Cache built for subjects 01–05 × cameras 836212060125 /
  932122062010, 1000 records (`MyDrive/kalman_filtering_results/dexycb/compact.zip`,
  34 MB). Evaluation used 40 sequences per (subject, camera): 400 sequences,
  23,564 frames.
- **Accuracy generalises.** With the oracle (GT) calibration:
  - MediaPipe 21.24°, corrector 14.22°, EKF **13.97°**, PA-MPJPE 15.8 mm
  - with the MediaPipe calibration: EKF 14.07°
  - jitter/GT 1.44, coasting MAE 16.7°
  - EKF per subject and camera: 13.4–15.0°; subject 03 is the hardest
  - Camera 932… is worse for raw MediaPipe (22.3 vs 20.4°), but the corrector
    removes most of that gap.
- **Pooled tuning, whole subjects held out** (train 01/03/05, test 02/04,
  objective MAE + 0.5·accel):

  | | held-out MAE | accel error |
  |---|---|---|
  | defaults | 13.810° | 2.36 |
  | tuned | **13.613° (−1.4 %)** | 2.44 |

  Tuned settings: jerk 3.6e5, σ_angle median 9.7° (per-angle CSV),
  σ_xy 12.2, σ_z 14.3, r_scale 1.
- **Robust R.** Tuned: 13.685°, worse than plain. Defaults + robust k=3: MAE
  unchanged (13.810°), accel 2.33, jitter/GT 0.67 → 0.60. It buys a little
  smoothness, no accuracy.
- **Is the tuning the same per subject?**
  - Parameters vary a lot (jerk_psd spread about 3×), but the cost of using
    another subject's settings is only **+0.08°**. The objective is flat.
  - One pooled setting beats each subject's own setting by 0.17°, because 30
    sequences per subject overfit.
  - The shipped defaults are 0.43° worse than the pooled setting, and 0.9°
    worse on subject 05.
- **Is the tuning the same per camera?** Transfer cost −0.03°. There is no
  camera effect.
- **The same direction every time.** Every tuning run (s01 local, pooled,
  per-group) wants jerk_psd about 10× the default, with larger σ_angle.
  Candidate new defaults are the `pooled_plain` values. Check them on the
  webcam first.
- **Frame analysis (400 sequences):**
  - 62 % of the error is per-clip bias.
  - Thumb bias is the same sign on all 5 people: CMC −21.9°, IP −20.7°,
    MCP +18.3°. It is systematic and learnable.
  - Left hands are worse: 14.67° vs 13.57°.
  - Wrong-label frames: 6 %, where MediaPipe errs by 34° vs 20°.
  - Error is higher late in the clip, during the grasp: 13.2 → 14.5°.
  - Error grows with distance (1.0 m+); palm orientation has no effect.
  - The filter is 3.7° better than the corrector on glitch frames.
  - The worst frames (s04 112037 f11–12, GT speed 770°/s) look like **GT
    annotation glitches**, not pipeline errors.
- **Implication.** The filter is done; tuning is worth about 0.2°. The lever
  is the corrector's systematic bias: fine-tune or bias-correct v12 on DexYCB
  subjects (train 01/03/05, test 02/04; the compact cache has GT joints +
  MediaPipe landmarks), then check on HO-3D whether the thumb offset is
  DexYCB-specific.

## 12 · HO-3D results, part 1 (user's Colab run, 2026-10-05)

- **Data and setup.** 49 moving sequences (10 subject groups × 5 cameras),
  71,986 evaluated frames. GT calibration, current defaults.
- **Accuracy:**

  | | HO-3D | DexYCB |
  |---|---|---|
  | MediaPipe | 22.28° | 21.24° |
  | corrector | 16.03° | 14.22° |
  | angle KF | 15.87° | |
  | **EKF** | **15.61°** | 13.97° |

  EKF PA-MPJPE 16.9 mm; jitter/GT 1.88 (angle KF 1.23). Same ranking as on
  DexYCB, about 1.6° harder because of object occlusion.
- **Bias share is only 34 %** (DexYCB 62 %). HO-3D's error is mostly
  frame-to-frame, so there is more room for filtering here.
- **The thumb CMC flexion bias appears again:** −18.7° (DexYCB −21.9°), and
  Thumb_CMC_Abd is +16.5°. So it is not a DexYCB annotation quirk; it is a
  systematic corrector bias.
- **What makes frames worse:**
  - hand closure correlates with error here (rho 0.24; 13.5 → 16.7° from
    open to closed); DexYCB showed no such effect
  - very close hands (< 0.4 m)
  - the first frame after a gap: 17.8°
  - a wrong handedness label (6 % of frames): MediaPipe 34.5° vs 21.5°
- **Glitches:** the filter is 4.4° better than the corrector on glitch frames.
- **Worst cases:** the BB recording on all 5 cameras (17.5–20.1°); GSF14 cam4
  frames 335–342, where MediaPipe errs by 87°.
- **Detection by camera:** AP cameras 1–3 detect only 52–64 % of frames.
- **Why it stopped.** Part 1 ended in step 9 (tuning): about 72k frames per
  trial was too slow for a Colab session.
- **Part 2** (`validation/ho3d_colab_part2.ipynb`, generated by
  `make_ho3d_part2_notebook.py`) needs no images and writes straight to Drive.
  Its steps:
  1. DexYCB-tuned settings vs defaults on all of HO-3D.
  2. `--group-by dataset` with `--clip-len 150 --clips-per-seq 2`. The halves
     alternate whole subjects, and the run resumes per group
     (`group_best_<g>.json`).
  3. Optional: the frame analysis, saved to Drive.
- **New options.**
  - `--clip-len` / `--clips-per-seq` in `validate_dexycb.py` and
    `tune_dexycb.py` (via `V.clip_prepared`).
  - Group-mode resume.

## 13 · HO-3D part 2 and the FINAL filter verdict (2026-10-05)

**Transfer test on all 71,986 HO-3D frames** (the DexYCB `pooled_plain`
settings were never tuned on HO-3D):

| EKF on HO-3D | MAE | PA-MPJPE | accel err | jitter/GT | coast MAE |
|---|---|---|---|---|---|
| current defaults | 15.58° | 18.6 mm | 1.34 | 1.88 | 18.8° |
| DexYCB-tuned | 15.53° | 17.4 mm | 1.63 (+22 %) | 2.83 (+50 %) | 19.3° |

The angle-only KF with the DexYCB-tuned settings is 15.89° vs 15.87°, with
jitter 2.61 vs 1.23. **The DexYCB tuning does not carry over:** −0.05° MAE for
50 % more jitter.

**Group-by-dataset tuning** (150-frame windows; the halves alternate whole
subjects):

| tuned on | DexYCB held out | HO-3D held out |
|---|---|---|
| DexYCB (jerk 4.3e5) | **13.73** | 14.44 |
| HO-3D (jerk at the 1e3 lower bound) | 14.04 | 14.54 |
| both (jerk 2.4e4) | 13.87 | 14.45 |
| **current defaults (jerk 3e4)** | 13.93 | **14.40** |

- The two datasets want opposite smoothing. DexYCB has short, fast grasps
  (jerk about 4e5). HO-3D has slow manipulation, smooth GT and occlusion noise
  (jerk at the 1e3 lower bound).
- The objective is flat, though. The transfer cost is 0.10°, and **the shipped
  defaults sit in between and are within 0.03° of the best on average; on
  HO-3D they are the best.** HO-3D's own tuning loses to the defaults on
  held-out HO-3D subjects, i.e. it overfits.
- (Reporting fix: the tuner printed unclamped values below the jerk bound,
  e.g. "10.76". It now reports the clamped values actually evaluated.)

**VERDICT: keep the current defaults.** jerk 3e4, σ_angle 6, σ_xy 15,
σ_z 25, r_scale 2, adaptive Q, coast_decay 0.3, robust off, threading on.
They are validated on 3 evaluations (s01, 5 subjects × 2 cameras, HO-3D
10 × 5 cameras). Retuning is worth ≤ 0.2° and does not travel. The filter
work is finished. The remaining lever is the corrector's bias: thumb CMC
flexion is about −20° on both datasets, and 62 % (DexYCB) / 34 % (HO-3D) of
the error is per-clip offset.

## 8 · For the next session

The pipeline is in a good state and the user is happy with the live output.
Open with: **"Have you run dexycb_multi_colab.ipynb, and did the Unity scripts compile?"** (§5.0, §9). Threading is done;
the multi-subject results decide whether robust R becomes a default.

Before any Colab work, re-upload `kalman_filtering/` to Drive (§5.8). Ask for
`measure_still_noise.py` results if the user has run it, and for the tuned
per-angle sigma CSV if they want it adopted.

Everything beyond that is accuracy, and accuracy now lives in the corrector,
not the filter. The corrector is already trained on the full FreiHAND
extraction (§3.6), so the largest remaining lever is the FreiHAND→real-video
domain gap (§5.5).

**Résumé line** (validated numbers, if it is needed again):
`Built a two-block EKF fusing MediaPipe 3D keypoints, cutting jitter 6x at
zero lag with 13.8° MAE on DexYCB`
