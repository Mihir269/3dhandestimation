"""Generates validation/ho3d_colab_part2.ipynb (every code cell is ast-checked)."""

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    src = text.strip("\n")
    ast.parse(src)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})


md(r"""
# HO-3D v3, part 2: does the DexYCB tuning carry over?

Part 1 (`ho3d_colab.ipynb`) finished steps 1–8:
- download
- conversion
- compact cache: 49 sequences in `MyDrive/kalman_filtering_results/ho3d/compact`
- validation
- frame analysis

It stopped in the tuning step because HO-3D clips are 1000–2800 frames long, so every tuning trial ran the filter over about 72k frames. This notebook finishes the job **without runtime trouble**:

- It needs **no images**, only the two small caches in Drive, so start-up takes about 2 minutes instead of restoring 16 GB.
- Long HO-3D clips are cut into **two 150-frame windows** per sequence for tuning, about the same cost per sequence as DexYCB.
- **Every step writes its results straight to Drive and is skipped next time** once done.
- The long tuning step **resumes per group** after a pause or disconnect: just re-run the cell.
- Each step shows progress, and no single step should run much longer than about 30 minutes.

Tip: keep the Colab tab in the foreground. Free Colab pauses idle sessions.

### Part 1 results, recorded here because those files were only in `/content`

| method | MAE | PA-MPJPE | jitter vs GT | DexYCB (5 subjects) |
|---|---|---|---|---|
| MediaPipe | 22.28° | 17.6 mm | 11.6 | 21.24° |
| corrector | 16.03° | 18.8 mm | 6.39 | 14.22° |
| angle-only KF | 15.87° | 18.7 mm | 1.23 | 14.35° |
| **EKF (angles + 3D)** | **15.61°** | **16.9 mm** | 1.88 | 13.97° |

(GT calibration, 71,986 frames, 10 subject groups × 5 cameras. With the MediaPipe calibration the EKF scores 15.58°.)
""")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
""")
md("## 1 · Settings")
code(r"""
KF_DRIVE = '/content/drive/MyDrive/kalman_filtering'
HO3D_COMPACT = '/content/drive/MyDrive/kalman_filtering_results/ho3d/compact'
DEXYCB_COMPACT = '/content/drive/MyDrive/kalman_filtering_results/dexycb/compact'
DEXYCB_TUNED = '/content/drive/MyDrive/kalman_filtering_results/dexycb/multi/tune/pooled_plain'
OUT = '/content/drive/MyDrive/kalman_filtering_results/ho3d/part2'   # straight to Drive

CLIP_LEN, CLIPS_PER_SEQ = 150, 2   # HO-3D windows used for tuning
DEX_PER_GROUP = 20                 # DexYCB sequences per (subject, camera) for tuning
N_RANDOM, N_REFINE = 20, 25        # search effort per tuning run
WORKERS = 2                        # Colab has 2 CPUs
KF = '/content/kalman_filtering'
""")
md("## 2 · Code (about 2 min, no images needed)")
code(r"""
import os, shutil, subprocess, sys, glob, json, time
from IPython.display import Markdown, display
import pandas as pd
pd.set_option('display.width', 250)

def sh(cmd, check=True):
    '''Run a shell command and stream its output into the cell.'''
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if 'inference_feedback' in line or line.startswith(('W0000', 'I0000')) or 'it/s]' in line:
            continue
        print(line, end='', flush=True)
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f'command failed ({rc}): {cmd}')
    return rc

assert os.path.isdir(KF_DRIVE), f'upload the kalman_filtering folder to {KF_DRIVE} first'
if os.path.exists(KF):
    shutil.rmtree(KF)
shutil.copytree(KF_DRIVE, KF, ignore=shutil.ignore_patterns('output', '__pycache__', '*.mp4', 'unity'))
assert '--clip-len' in open(f'{KF}/validation/tune_dexycb.py').read(), \
    'Drive copy is stale: re-upload kalman_filtering/ (needs --clip-len and group resume)'
sh('pip -q install mediapipe')
os.makedirs(OUT, exist_ok=True)
n_ho3d = len(glob.glob(f'{HO3D_COMPACT}/*.npz'))
n_dex = len(glob.glob(f'{DEXYCB_COMPACT}/*.npz'))
print(f'HO-3D cache: {n_ho3d} sequences | DexYCB cache: {n_dex} records')
assert n_ho3d, 'HO-3D cache missing: finish part 1 (steps 1-6) first'
""")
md(r"""
## 3 · The quickest answer: DexYCB's tuned settings, scored on HO-3D (about 10–15 min)

The settings tuned on DexYCB (`pooled_plain`: jerk 3.6e5, larger angle noise, σ_z 14 mm) are compared with the current defaults on **all** HO-3D frames. HO-3D played no part in that tuning, so this is a clean transfer test. If the DexYCB-tuned settings also beat the defaults here, they are safe to adopt.
""")
code(r"""
def dex_tuned_flags():
    cfg = json.load(open(f'{DEXYCB_TUNED}/best_config.json'))['config']
    return (f"--jerk-psd {cfg['jerk_psd']} --angle-noise-csv {DEXYCB_TUNED}/tuned_sigma_per_angle.csv "
            f"--r-scale-angles 1.0 --sigma-xy {cfg['sigma_xy']} --sigma-z {cfg['sigma_z']} --r-scale-kps 1.0")

runs = {'defaults': '', 'dexycb_tuned': dex_tuned_flags() if os.path.exists(f'{DEXYCB_TUNED}/best_config.json') else None}
for tag, flags in runs.items():
    if flags is None:
        print('no DexYCB tuning in Drive (dexycb_multi_colab stage 2b): skipping', tag)
        continue
    if os.path.exists(f'{OUT}/transfer/{tag}/summary.csv'):
        print(f'{tag}: done earlier, skipped')
        continue
    sh(f'cd {KF} && python validation/validate_dexycb.py --compact {HO3D_COMPACT} --calib mediapipe '
       f'--methods corr kf_angle_corr ekf_corr --no-plots --out {OUT}/transfer --tag {tag} {flags}')
rows = []
for tag in runs:
    p = f'{OUT}/transfer/{tag}/summary.csv'
    if os.path.exists(p):
        s = pd.read_csv(p)
        s.insert(0, 'settings', tag)
        rows.append(s)
if rows:
    display(pd.concat(rows)[['settings', 'method', 'angle_mae', 'pa_mpjpe_mm', 'accel_err_deg',
                             'jitter_vs_gt', 'lag_frames', 'coast_mae']].round(3))
""")
md(r"""
## 4 · Is the tuning the same on both datasets? (resumable, about 3 × 15–25 min)

The filter is tuned three times:
- on DexYCB alone
- on HO-3D alone
- on both together ("ALL")

Every result is then scored on both datasets' held-out subjects. Each tuning run is saved to Drive the moment it finishes. **If Colab pauses, re-run this cell: finished runs are skipped and the search continues with the next one.**

Robust R is not tuned here: on DexYCB it added nothing (stage 2b of the multi notebook).
""")
code(r"""
both = f'{DEXYCB_COMPACT},{HO3D_COMPACT}'
if not glob.glob(f'{DEXYCB_COMPACT}/*.npz'):
    print('No DexYCB cache in Drive: run dexycb_multi_colab.ipynb stage 1 first.')
else:
    sh(f'cd {KF} && python validation/tune_dexycb.py --compact {both} --method ekf_corr '
       f'--group-by dataset --max-per-group {DEX_PER_GROUP} '
       f'--clip-len {CLIP_LEN} --clips-per-seq {CLIPS_PER_SEQ} --accel-weight 0.5 '
       f'--n-random {N_RANDOM} --n-refine {N_REFINE} --workers {WORKERS} --out {OUT}/groups_dataset')
    rep = f'{OUT}/groups_dataset/group_report.md'
    if os.path.exists(rep):
        display(Markdown(open(rep, encoding='utf-8').read()))
""")
md(r"""
How to read it:
- **Transfer cost** is the price of using the other dataset's settings. A few hundredths of a degree means one setting works on both.
- **ALL vs defaults** is how much a jointly tuned setting would gain everywhere.
- If HO-3D strongly prefers different settings, look at the parameter table to see which knob differs. A likely candidate is jerk_psd, since HO-3D's hands move more slowly.

## 5 · Optional: save the HO-3D frame analysis files to Drive (about 20 min)

Part 1 showed this report, but its files (`frames.csv.gz`, worst frames and so on) were only in `/content`. Run this cell if you want them in Drive for later digging. It is skipped once done.
""")
code(r"""
if os.path.exists(f'{OUT}/frames/defaults/report.md'):
    print('frame analysis already in Drive, skipped')
else:
    sh(f'cd {KF} && python validation/analyse_dexycb_frames.py --compact {HO3D_COMPACT} '
       f'--out {OUT}/frames --tag defaults > /content/frames.log 2>&1')
print('saved to', f'{OUT}/frames/defaults')
""")
md(r"""
## Done

Everything is in `MyDrive/kalman_filtering_results/ho3d/part2/`. Send back these two reports (or the folder):
- `transfer/*/summary.csv`
- `groups_dataset/group_report.md`

For the laptop, the HO-3D cache itself is `MyDrive/kalman_filtering_results/ho3d/compact` (49 small files). Download that folder into `kalman_filtering/output/compact_ho3d`.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(HERE, "ho3d_colab_part2.ipynb")
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"wrote {path} ({sum(c['cell_type'] == 'code' for c in cells)} code cells, all parse)")
