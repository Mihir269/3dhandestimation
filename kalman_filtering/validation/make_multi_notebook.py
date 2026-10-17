"""Generates validation/dexycb_multi_colab.ipynb (every code cell is ast-checked)."""

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
# DexYCB on many subjects and cameras: cache, tune, compare, analyse

Every result so far comes from **one subject and one camera** (subject 01, camera 836212060125). This notebook extends that to the other subjects and cameras, in three stages.

1. **Build a compact cache.** Each subject is a 12 GB download. For every subject the notebook downloads the tar, extracts the chosen cameras, runs MediaPipe once, and keeps only what the filters need: about 30 KB per sequence and camera (GT joints, MediaPipe landmarks, hand side). The tar is then deleted. Progress is saved to Drive after each subject, so you can stop at any time and continue in a new session.
2. **Tune and compare:**
   - pooled tuning with whole subjects held out
   - per-subject and per-camera tuning, cross-evaluated, to see whether the tuning is the same everywhere
3. **Frame-level analysis**: which frames are wrong, and why.

**Tip:** after stage 1, download `compact.zip` (a few tens of MB). Stages 2 and 3 then run on your laptop with all its cores, usually faster than on Colab's 2 CPUs. The commands are in the last cell.

**Before you start:** re-upload the current `kalman_filtering/` folder to `MyDrive/kalman_filtering`. Stale code in Drive caused a `ModuleNotFoundError` once before.
""")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
""")
md("## 1 · Settings")
code(r"""
KF_DRIVE = '/content/drive/MyDrive/kalman_filtering'
RESULTS_DRIVE = '/content/drive/MyDrive/kalman_filtering_results/dexycb'
COMPACT_DRIVE = RESULTS_DRIVE + '/compact'     # the small cache, kept across sessions
MP_CACHE = RESULTS_DRIVE + '/mp_cache'         # MediaPipe per (sequence, camera), reused

# DexYCB download IDs (https://dex-ycb.github.io/)
SUBJECTS = {
    '01': ('20200709-subject-01', '1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk'),
    '02': ('20200813-subject-02', '1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62'),
    '03': ('20200820-subject-03', '1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq'),
    '04': ('20200903-subject-04', '14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA'),
    '05': ('20200908-subject-05', '1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS'),
    '06': ('20200918-subject-06', '1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ'),
    '07': ('20200928-subject-07', '1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0'),
    '08': ('20201002-subject-08', '1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv'),
    '09': ('20201015-subject-09', '1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH'),
    '10': ('20201022-subject-10', '1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-'),
}

# Which subjects to process this session. Each takes roughly 15-30 min
# (download + extract + MediaPipe on the cameras below). Done subjects are skipped.
RUN_SUBJECTS = ['01', '02', '03', '04', '05']

# Cameras. 836212060125 is the one used so far; adding a second, different
# viewpoint gives the camera comparison. All 8 = 4x the MediaPipe time.
SERIALS = ['836212060125', '932122062010']
# ALL 8: ['836212060125', '839512060362', '840412060917', '841412060263',
#         '932122060857', '932122060861', '932122061900', '932122062010']

KF = '/content/kalman_filtering'
WORK = '/content/dexycb_work'
LOCAL_COMPACT = '/content/compact'
OUT = '/content/results/multi'

# stage 2/3 effort
MAX_PER_GROUP = 40        # sequences per (subject, camera) used for tuning/analysis
N_RANDOM, N_REFINE = 30, 50
ACCEL_WEIGHT = 0.5        # MAE + 0.5 x acceleration error (pure MAE wrecks smoothness)
""")
md("## 2 · Code and packages")
code(r"""
import os, shutil, subprocess, sys, glob, time

def sh(cmd, check=True):
    '''Run a shell command and stream its output into the cell.'''
    print('$', cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if 'inference_feedback' in line or line.startswith(('W0000', 'I0000')):
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
assert os.path.exists(f'{KF}/validation/dexycb_data.py'), 'Drive copy is stale: re-upload kalman_filtering/'
for d in (COMPACT_DRIVE, MP_CACHE, LOCAL_COMPACT, OUT):
    os.makedirs(d, exist_ok=True)
sh('pip -q install mediapipe gdown pyyaml')
sh(f'cd {KF} && python tests/selftest.py 2>&1 | tail -n 3')
""")
md(r"""
## 3 · Stage 1: build the compact cache, one subject at a time

For each subject: download (or copy from a Drive shortcut), extract only the chosen cameras' colour images and labels, run MediaPipe, write the records, copy them to Drive, then delete the tar and the images.

**If `gdown` fails** with "Too many users have viewed or downloaded this file", open the file's Drive link from the table above in a browser, choose *Add shortcut to Drive* → *My Drive*, and re-run. The cell copies from the shortcut instead.
""")
code(r"""
def subject_done(sid):
    return os.path.exists(f'{COMPACT_DRIVE}/_done_{sid}_' + '-'.join(SERIALS))

for sid in RUN_SUBJECTS:
    folder, file_id = SUBJECTS[sid]
    if subject_done(sid):
        print(f'subject {sid}: already in the cache, skipped')
        continue
    t0 = time.time()
    tar = f'/content/{folder}.tar.gz'
    shortcut = f'/content/drive/MyDrive/{folder}.tar.gz'
    if not os.path.exists(tar):
        if os.path.exists(shortcut):
            sh(f'cp "{shortcut}" {tar}')
        elif sh(f'gdown {file_id} -O {tar}', check=False):
            print(f'!! download of subject {sid} failed: add a Drive shortcut (see above) and re-run')
            continue
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    pats = ' '.join(f"'*/{s}/color_*' '*/{s}/labels_*'" for s in SERIALS)
    sh(f"tar -xzf {tar} -C {WORK} --wildcards {pats} '*/meta.yml'")
    # one MediaPipe process per camera: Colab has 2 CPUs
    procs = [subprocess.Popen(
        f'cd {KF} && python validation/dexycb_data.py build --root {WORK} --out {LOCAL_COMPACT} '
        f'--serials {s} --mp-cache {MP_CACHE} > /content/build_{sid}_{s}.log 2>&1', shell=True)
        for s in SERIALS]
    codes = [p.wait() for p in procs]
    for s in SERIALS:
        print(open(f'/content/build_{sid}_{s}.log').read().splitlines()[-1])
    if any(codes):
        print(f'!! build failed for subject {sid}; see /content/build_{sid}_*.log')
        continue
    for f in glob.glob(f'{LOCAL_COMPACT}/{folder}__*.npz'):
        shutil.copy(f, COMPACT_DRIVE)
    open(f'{COMPACT_DRIVE}/_done_{sid}_' + '-'.join(SERIALS), 'w').write('ok')
    shutil.rmtree(WORK, ignore_errors=True)
    os.remove(tar)
    print(f'subject {sid}: done in {(time.time() - t0) / 60:.1f} min')
""")
code(r"""
sh(f'cd {KF} && python validation/dexycb_data.py info --compact {COMPACT_DRIVE}')
""")
md(r"""
## 4 · Stage 2a: accuracy on everything, current defaults

All six methods on every cached subject and camera, using the oracle calibration (`--calib gt`, available now because the cache holds the GT joints), and the realistic one (`--calib mediapipe`).
""")
code(r"""
for calib in ('gt', 'mediapipe'):
    sh(f'cd {KF} && python validation/validate_dexycb.py --compact {COMPACT_DRIVE} --calib {calib} '
       f'--max-per-group {MAX_PER_GROUP} --out {OUT}/validate --tag {calib} --no-plots')
""")
code(r"""
import pandas as pd
from IPython.display import Markdown, display
pd.set_option('display.width', 250)
for calib in ('gt', 'mediapipe'):
    display(Markdown(f'### calib = {calib}'))
    display(pd.read_csv(f'{OUT}/validate/{calib}/summary.csv').drop(columns=['description']).round(2))
    ps = pd.read_csv(f'{OUT}/validate/{calib}/per_sequence.csv')
    display(ps.groupby(['subject', 'serial'])[['mp', 'corr', 'ekf_corr']].mean().round(2))
""")
md(r"""
## 5 · Stage 2b: pooled tuning, whole subjects held out

`--split subject` fits on half the subjects and reports on the other half: the honest test of whether a tuned setting generalises to people it has never seen. `--tune-robust` also tunes the innovation-weighting thresholds (`robust_k`, `robust_k_kp`).
""")
code(r"""
for name, extra in (('pooled_plain', ''), ('pooled_robust', '--tune-robust')):
    sh(f'cd {KF} && python validation/tune_dexycb.py --compact {COMPACT_DRIVE} --method ekf_corr '
       f'--split subject --max-per-group {MAX_PER_GROUP} --accel-weight {ACCEL_WEIGHT} '
       f'--n-random {N_RANDOM} --n-refine {N_REFINE} --workers 2 --quiet {extra} '
       f'--out {OUT}/tune/{name}')
    display(Markdown(open(f'{OUT}/tune/{name}/report.md', encoding='utf-8').read()))
""")
md(r"""
## 6 · Stage 2c: is the tuning the same on other data?

The filter is tuned separately on each subject (and then each camera), and each group's best settings are scored on every group's held-out half. Read `group_report.md`:

- **Small transfer cost** (a few hundredths of a degree): one setting serves everyone, even if the numbers differ. The objective is flat.
- **Large transfer cost**: the best settings depend on the person or the viewpoint.
""")
code(r"""
for by in ('subject', 'camera'):
    sh(f'cd {KF} && python validation/tune_dexycb.py --compact {COMPACT_DRIVE} --method ekf_corr '
       f'--group-by {by} --tune-robust --max-per-group 30 --accel-weight {ACCEL_WEIGHT} '
       f'--n-random 20 --n-refine 30 --workers 2 --quiet --out {OUT}/groups_{by}', check=False)
    p = f'{OUT}/groups_{by}/group_report.md'
    if os.path.exists(p):
        display(Markdown(open(p, encoding='utf-8').read()))
""")
md(r"""
## 7 · Stage 3: where and why it fails, frame by frame

Runs `analyse_dexycb_frames.py` with the current defaults (add `--robust` to analyse the robust filter). The report shows:
- how much of the error is per-sequence **bias**, which no filter can remove
- which causes go with high error: speed, hand closure, measurement jumps, palm orientation, distance
- what happens after a detection gap
- the worst sequences and frames
""")
code(r"""
sh(f'cd {KF} && python validation/analyse_dexycb_frames.py --compact {COMPACT_DRIVE} '
   f'--max-per-group {MAX_PER_GROUP} --out {OUT}/frames --tag defaults > /content/frames.log 2>&1')
from IPython.display import Image
display(Markdown(open(f'{OUT}/frames/defaults/report.md', encoding='utf-8').read()))
for png in ('error_vs_cause.png', 'worst_sequences.png', 'after_gap.png'):
    display(Image(filename=f'{OUT}/frames/defaults/{png}'))
""")
md("## 8 · Save, and take the cache to your laptop")
code(r"""
shutil.copytree(OUT, f'{RESULTS_DRIVE}/multi', dirs_exist_ok=True)
sh(f'cd {os.path.dirname(COMPACT_DRIVE)} && zip -qr /content/compact.zip compact -x "*/_done_*"')
shutil.copy('/content/compact.zip', f'{RESULTS_DRIVE}/compact.zip')
sh(f'ls -lh {RESULTS_DRIVE}/compact.zip')
print('saved to', RESULTS_DRIVE)
""")
md(r"""
On the laptop, unzip `compact.zip` into `kalman_filtering/output/` and run everything locally (12 cores):

```
python validation\dexycb_data.py info --compact output\compact
python validation\tune_dexycb.py --compact output\compact --method ekf_corr --split subject --tune-robust --accel-weight 0.5
python validation\tune_dexycb.py --compact output\compact --method ekf_corr --group-by subject --tune-robust --accel-weight 0.5 --max-per-group 30
python validation\analyse_dexycb_frames.py --compact output\compact --max-per-group 40
```
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(HERE, "dexycb_multi_colab.ipynb")
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"wrote {path} ({sum(c['cell_type'] == 'code' for c in cells)} code cells, all parse)")
