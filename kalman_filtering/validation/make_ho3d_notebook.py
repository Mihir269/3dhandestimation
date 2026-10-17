"""Generates validation/ho3d_colab.ipynb (every code cell is ast-checked)."""

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
# HO-3D v3: a second dataset for verifying the filter

Everything so far was measured on DexYCB. HO-3D v3 is a different lab, different cameras and different people (10 people, 68 videos of a right hand handling objects, MANO-fitted 3D joints). Running the same tools on it checks whether the DexYCB conclusions and tuned settings carry over.

**What this notebook does**
1. **Downloads HO-3D v3 from its official OneDrive share** into Colab, sequence by sequence. The dataset is one 31.9 GB zip, half of it depth maps. `validation/ho3d_data.py` reads the zip's file index over HTTP and fetches only each sequence's RGB frames and annotations (about 21 GB), checking every file's checksum. No 32 GB file is ever stored, and an interrupted download resumes.
2. **Stores the downloads in Google Drive.** Each sequence is saved as a tar in `MyDrive/ho3d_v3/raw/`, so later sessions restore from Drive instead of OneDrive. The released evaluation-split GT and the zip index go there too.
3. **Builds the compact cache.** MediaPipe runs once per frame; GT joints are converted to MediaPipe's order and the DexYCB camera axes.
4. **Validates, analyses frame by frame, and compares tuning with DexYCB.**

**Terms:** HO-3D is for academic, non-commercial research only and may not be redistributed. The index cell prints the dataset's terms. Running the download means you accept them. Cite *HOnnotate, Hampali et al., CVPR 2020* if you use it.

**Time and space.**
- Download: roughly 10–30 min, depending on OneDrive's speed.
- MediaPipe over about 75k frames: about 25–35 min on Colab's 2 CPUs.
- Colab disk: about 20 GB.
- Drive: about 15 GB if `KEEP_IMAGES_IN_DRIVE = True`, about 100 MB otherwise.

Every stage is resumable. If the session dies, re-run from the top: finished work is skipped.

**Note on HO-3D's sequences.** The multi-camera recordings (5 cameras each) have per-frame finger annotations. The single-camera recordings are annotated with one fixed grasp, so only those with moving fingers are used (step 4). The cameras of one recording share the same motion, so tuning always holds out whole recordings (`--split subject`), never single cameras.

**Before you start:** re-upload the current `kalman_filtering/` folder to `MyDrive/kalman_filtering`. It must contain `validation/ho3d_data.py`.
""")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
""")
md("## 1 · Settings")
code(r"""
KF_DRIVE = '/content/drive/MyDrive/kalman_filtering'
HO3D_DRIVE = '/content/drive/MyDrive/ho3d_v3'            # downloads are kept here
RESULTS_DRIVE = '/content/drive/MyDrive/kalman_filtering_results/ho3d'
COMPACT_DRIVE = RESULTS_DRIVE + '/compact'                # the small per-sequence cache
DEXYCB_COMPACT = '/content/drive/MyDrive/kalman_filtering_results/dexycb/compact'  # from dexycb_multi_colab (optional)

KEEP_IMAGES_IN_DRIVE = True   # True: RGB frames saved to Drive as one tar per sequence (~15 GB)
                              # False: only GT + compact cache in Drive (~100 MB); re-runs download again
SPLITS = ['train', 'evaluation']   # 55 + 13 sequences; evaluation GT was released in Nov 2024
FPS = 30.0                         # not stated by the authors; RealSense default (sets the filter's dt)
DOWNLOAD_WORKERS = 6               # parallel range downloads from OneDrive

KF = '/content/kalman_filtering'
LOCAL = '/content/ho3d'            # extracted frames (Colab disk)
LOCAL_COMPACT = '/content/compact_ho3d'
OUT = '/content/results/ho3d'
MAX_PER_GROUP = 3                  # tuning: sequences per (subject, camera); HO-3D clips are long
""")
md("## 2 · Code and packages")
code(r"""
import os, shutil, subprocess, sys, glob, time, json, tarfile

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
assert os.path.exists(f'{KF}/validation/ho3d_data.py'), 'Drive copy is stale: re-upload kalman_filtering/'
for d in (HO3D_DRIVE + '/raw', HO3D_DRIVE + '/gt', COMPACT_DRIVE, LOCAL, LOCAL_COMPACT, OUT):
    os.makedirs(d, exist_ok=True)
sh('pip -q install mediapipe')
sh(f'cd {KF} && python tests/selftest.py 2>&1 | tail -n 2')
sys.path.insert(0, f'{KF}/validation'); sys.path.insert(0, f'{KF}/core')
import ho3d_data as H

total, used, free = shutil.disk_usage('/content/drive/MyDrive')
print(f'Google Drive: {free / 1e9:.0f} GB free of {total / 1e9:.0f} GB')
if KEEP_IMAGES_IN_DRIVE and free < 17e9:
    print('!! less than 17 GB free in Drive: switching KEEP_IMAGES_IN_DRIVE off '
          '(GT and the compact cache are still saved)')
    KEEP_IMAGES_IN_DRIVE = False
""")
md(r"""
## 3 · Index the zip (and read the terms)

Reads the zip's central directory over HTTP (a few MB, under a minute) and saves it to Drive, so later sessions skip this step.
""")
code(r"""
INDEX = f'{HO3D_DRIVE}/zip_index.json.gz'
sh(f'cd {KF} && python validation/ho3d_data.py index --cache {INDEX}')
""")
md(r"""
## 4 · Annotations first, then frames only for sequences whose fingers move

HO-3D annotates its single-camera sequences (MC, ND, SM, SMu, SS, SiS) with **one fixed grasp for the whole clip**: only the wrist moves, the finger angles never change. They cannot test a filter (the truth never moves), so the notebook downloads every sequence's annotations first (about 1.2 GB), measures the finger motion (`scan`), and fetches the RGB frames only for the moving ones (about 49 sequences, about 15 GB). Set `ONLY_MOVING = False` to take everything.

Frames saved in an earlier session are restored from `MyDrive/ho3d_v3/raw/` instead of being downloaded again.
""")
code(r"""
ONLY_MOVING = True
idx = H.build_index(INDEX)
seqs = sorted({(s, q) for (s, q, k) in H.blocks_of(idx) if s in SPLITS})
print(f'{len(seqs)} sequences in {SPLITS}')

# (a) annotations + evaluation GT, then which sequences move
SCAN = f'{HO3D_DRIVE}/gt/sequences.json'
if not os.path.exists(SCAN):
    sh(f'cd {KF} && python validation/ho3d_data.py download --out {LOCAL} --index {INDEX} '
       f'--splits train --kinds meta --workers {DOWNLOAD_WORKERS}')
    if 'evaluation' in SPLITS:
        sh(f'cd {KF} && python validation/ho3d_data.py gt-eval --out {LOCAL}')
        shutil.copy(f'{LOCAL}/evaluation_xyz.json', f'{HO3D_DRIVE}/gt/')
    for name in ('train.txt', 'evaluation.txt'):
        shutil.copy(f'{LOCAL}/{name}', f'{HO3D_DRIVE}/gt/{name}')
    sh(f'cd {KF} && python validation/ho3d_data.py scan --root {LOCAL} --out {SCAN} '
       f'--splits {" ".join(SPLITS)}')
else:
    for name in ('train.txt', 'evaluation.txt', 'evaluation_xyz.json'):
        if os.path.exists(f'{HO3D_DRIVE}/gt/{name}'):
            shutil.copy(f'{HO3D_DRIVE}/gt/{name}', LOCAL)
scan = json.load(open(SCAN))
moving = {(r['split'], r['seq']) for r in scan if r['moving']}
use = [s for s in seqs if (s in moving or not ONLY_MOVING)]
print(f'using {len(use)} of {len(seqs)} sequences '
      f'({len(seqs) - len(use)} static-grasp skipped)' if ONLY_MOVING else f'using all {len(use)}')

# (b) frames: restore from Drive, or download and save to Drive
def local_ok(split, seq):
    return all(os.path.exists(f'{LOCAL}/{split}/{seq}/.{k}_done') for k in ('rgb', 'meta'))

restored, missing = 0, []
for split, seq in use:
    if local_ok(split, seq):
        continue
    tar = f'{HO3D_DRIVE}/raw/{split}_{seq}.tar'
    if os.path.exists(tar):
        with tarfile.open(tar) as t:
            t.extractall(LOCAL)
        restored += 1
    else:
        missing.append((split, seq))
print(f'restored {restored} sequences from Drive, {len(missing)} to download')
for split in SPLITS:
    names = [q for s, q in missing if s == split]
    if names:
        sh(f'cd {KF} && python validation/ho3d_data.py download --out {LOCAL} --index {INDEX} '
           f'--splits {split} --kinds rgb meta --workers {DOWNLOAD_WORKERS} --seqs {" ".join(names)}')
if KEEP_IMAGES_IN_DRIVE:
    for split, seq in missing:
        tar = f'{HO3D_DRIVE}/raw/{split}_{seq}.tar'
        if local_ok(split, seq) and not os.path.exists(tar):
            with tarfile.open(tar + '.part', 'w') as t:      # uncompressed: jpgs do not shrink
                t.add(f'{LOCAL}/{split}/{seq}', arcname=f'{split}/{seq}')
            os.replace(tar + '.part', tar)
    print('frames saved to', HO3D_DRIVE + '/raw')
sh(f'du -sh {LOCAL}')
""")
md(r"""
## 5 · Sanity check: joint order, units, axes

Bone lengths should look like a real hand: wrist→knuckles 7–10 cm, phalanges 2–5 cm. The wrist depth should be positive, 0.4–0.9 m in front of the camera. Anything else means the conversion is wrong. Stop and check before going further.
""")
code(r"""
sh(f'cd {KF} && python validation/ho3d_data.py check --root {LOCAL}')
""")
md(r"""
## 6 · Build the compact cache (MediaPipe, once per frame)

Two processes, one per Colab CPU. Finished sequences are copied to Drive and skipped on re-runs.
""")
code(r"""
for f in glob.glob(f'{COMPACT_DRIVE}/*.npz'):          # resume: what earlier sessions finished
    if not os.path.exists(f'{LOCAL_COMPACT}/{os.path.basename(f)}'):
        shutil.copy(f, LOCAL_COMPACT)
names = [q for _, q in use]
halves = [names[0::2], names[1::2]]
procs = [subprocess.Popen(
    f'cd {KF} && python validation/ho3d_data.py build --root {LOCAL} --out {LOCAL_COMPACT} '
    f'--splits {" ".join(SPLITS)} --fps {FPS} --seqs {" ".join(h)} > /content/ho3d_build_{i}.log 2>&1',
    shell=True) for i, h in enumerate(halves) if h]
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
    done = len(glob.glob(f'{LOCAL_COMPACT}/*.npz'))
    print(f'{(time.time() - t0) / 60:5.1f} min  {done}/{len(use)} sequences', flush=True)
for i in range(len(procs)):
    print(''.join(open(f'/content/ho3d_build_{i}.log').readlines()[-3:]))
for f in glob.glob(f'{LOCAL_COMPACT}/*.npz'):
    shutil.copy(f, COMPACT_DRIVE)
sh(f'cd {KF} && python validation/dexycb_data.py info --compact {LOCAL_COMPACT}')
""")
md(r"""
## 7 · Accuracy on HO-3D, current defaults

The same six methods as on DexYCB (MediaPipe, corrector, angle KF and EKF on each), with the oracle calibration (`gt`) and the realistic one (`mediapipe`). Compare with DexYCB: MediaPipe 20.7°, corrector 14.0°, EKF 13.9°.
""")
code(r"""
import pandas as pd
from IPython.display import Markdown, Image, display
pd.set_option('display.width', 250)
for calib in ('gt', 'mediapipe'):
    sh(f'cd {KF} && python validation/validate_dexycb.py --compact {LOCAL_COMPACT} --calib {calib} '
       f'--out {OUT}/validate --tag {calib} --no-plots')
    display(Markdown(f'### calib = {calib}'))
    display(pd.read_csv(f'{OUT}/validate/{calib}/summary.csv').drop(columns=['description']).round(2))
""")
md(r"""
## 8 · Where and why it fails (frame by frame)

The same analysis as on DexYCB. It shows:
- the bias share of the error
- error against speed, hand closure, measurement jumps, palm orientation and distance
- what happens after detection gaps
- the worst sequences and frames

HO-3D has heavy object occlusion, so expect more detection gaps than on DexYCB.
""")
code(r"""
sh(f'cd {KF} && python validation/analyse_dexycb_frames.py --compact {LOCAL_COMPACT} '
   f'--out {OUT}/frames --tag defaults > /content/ho3d_frames.log 2>&1')
display(Markdown(open(f'{OUT}/frames/defaults/report.md', encoding='utf-8').read()))
for png in ('error_vs_cause.png', 'worst_sequences.png', 'after_gap.png'):
    display(Image(filename=f'{OUT}/frames/defaults/{png}'))
""")
md(r"""
## 9 · Is the tuning the same on HO-3D as on DexYCB?

- **9a** tunes the filter on HO-3D alone: whole subject groups are held out, and robust R is tuned too.
- **9b** needs the DexYCB cache from `dexycb_multi_colab.ipynb`. It tunes on DexYCB and scores on HO-3D (`--split dataset`), then tunes each dataset separately and cross-scores them (`--group-by dataset`).

A small transfer cost means one setting works for both, and the DexYCB-tuned defaults can be trusted on new data.
""")
code(r"""
sh(f'cd {KF} && python validation/tune_dexycb.py --compact {LOCAL_COMPACT} --method ekf_corr '
   f'--split subject --tune-robust --accel-weight 0.5 --max-per-group {MAX_PER_GROUP} '
   f'--n-random 25 --n-refine 35 --workers 2 --quiet --out {OUT}/tune_ho3d')
display(Markdown(open(f'{OUT}/tune_ho3d/report.md', encoding='utf-8').read()))
""")
code(r"""
if glob.glob(f'{DEXYCB_COMPACT}/*.npz'):
    both = f'{DEXYCB_COMPACT},{LOCAL_COMPACT}'
    sh(f'cd {KF} && python validation/tune_dexycb.py --compact {both} --method ekf_corr '
       f'--split dataset --tune-robust --accel-weight 0.5 --max-per-group {MAX_PER_GROUP} '
       f'--n-random 25 --n-refine 35 --workers 2 --quiet --out {OUT}/dexycb_to_ho3d')
    display(Markdown(open(f'{OUT}/dexycb_to_ho3d/report.md', encoding='utf-8').read()))
    sh(f'cd {KF} && python validation/tune_dexycb.py --compact {both} --method ekf_corr '
       f'--group-by dataset --tune-robust --accel-weight 0.5 --max-per-group {MAX_PER_GROUP} '
       f'--n-random 20 --n-refine 30 --workers 2 --quiet --out {OUT}/groups_dataset')
    display(Markdown(open(f'{OUT}/groups_dataset/group_report.md', encoding='utf-8').read()))
else:
    print('No DexYCB cache in Drive yet: run dexycb_multi_colab.ipynb first, or do this step '
          'on the laptop (commands in the last cell).')
""")
md("## 10 · Save results, and the cache for the laptop")
code(r"""
shutil.copytree(OUT, f'{RESULTS_DRIVE}/results', dirs_exist_ok=True)
sh(f'cd {os.path.dirname(LOCAL_COMPACT)} && zip -qr /content/compact_ho3d.zip {os.path.basename(LOCAL_COMPACT)}')
shutil.copy('/content/compact_ho3d.zip', f'{RESULTS_DRIVE}/compact_ho3d.zip')
sh(f'ls -lh {RESULTS_DRIVE}/compact_ho3d.zip')
print('saved to', RESULTS_DRIVE)
""")
md(r"""
On the laptop, unzip `compact_ho3d.zip` into `kalman_filtering/output/`. Then everything runs locally with all 12 cores, including the comparison against the subject-01 DexYCB cache that is already there:

```
python validation\dexycb_data.py info --compact output\compact_ho3d
python validation\validate_dexycb.py --compact output\compact_ho3d --calib gt --tag ho3d
python validation\tune_dexycb.py --compact output\dexycb_compact,output\compact_ho3d --method ekf_corr --group-by dataset --tune-robust --accel-weight 0.5 --max-per-group 3
python validation\analyse_dexycb_frames.py --compact output\compact_ho3d --tag ho3d
```
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(HERE, "ho3d_colab.ipynb")
json.dump(nb, open(path, "w", encoding="utf-8"), indent=1)
print(f"wrote {path} ({sum(c['cell_type'] == 'code' for c in cells)} code cells, all parse)")
