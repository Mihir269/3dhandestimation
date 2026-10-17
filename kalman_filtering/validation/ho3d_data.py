#!/usr/bin/env python3
"""
ho3d_data.py -- download HO-3D v3 and turn it into the compact cache that
validate_dexycb.py, tune_dexycb.py and analyse_dexycb_frames.py read.

HO-3D v3 (https://github.com/shreyashampali/ho3d): RGB video of a right hand
manipulating YCB objects, 10 people, 68 sequences (55 train, 13 evaluation),
MANO-fitted 3D joints. Academic, non-commercial use only; by downloading you
accept the terms in the dataset's README (shown by `index`).

WHY NOT JUST DOWNLOAD THE ZIP
-----------------------------
It is ONE 31.9 GB zip on OneDrive, and half of it is depth maps we do not use.
Inside the zip every (split, sequence, rgb|meta|depth) block is stored
contiguously, so this script reads the zip's central directory over HTTP
(a few requests), then fetches each sequence's rgb and meta block with ONE
range request each (~170-500 MB), CRC-checks and extracts the files. No 32 GB
file ever touches the disk, depth is skipped (~14 GB saved), and the work is
resumable per sequence. OneDrive share links are resolved anonymously through
the same public token OneDrive's own web viewer uses; download URLs expire, so
they are refreshed automatically.

    python validation/ho3d_data.py index --cache <dir>/zip_index.json.gz
    python validation/ho3d_data.py download --out /content/ho3d --index <dir>/zip_index.json.gz \
        --splits train evaluation --kinds rgb meta
    python validation/ho3d_data.py gt-eval --out /content/ho3d          # evaluation_xyz.json (28 MB)
    python validation/ho3d_data.py build --root /content/ho3d --out <compact dir> --fps 30
    python validation/ho3d_data.py check --root /content/ho3d           # joint order / units sanity
    python validation/ho3d_data.py scan --root /content/ho3d --out <dir>/sequences.json   # which move

CONVERSION
----------
* Joint order: HO-3D stores MANO order (wrist, index 1-3, middle 1-3, pinky
  1-3, ring 1-3, thumb 1-3, then tips thumb..pinky). Reordered to MediaPipe's
  with MANO_TO_MP. Verified on train/MC1: each tip sits next to its own chain.
* Axes: OpenGL (y up, hand at -z). Converted to the OpenCV camera frame used
  for DexYCB with (x, -y, -z): a 180 deg rotation, so chirality is unchanged.
* Hand: always the right hand. Frames without annotation (handJoints3D None)
  are NaN, exactly like DexYCB frames without a hand.
* Grouping: `subject` = 'HO3D-' + the sequence's letter prefix (ABF, MC, ...;
  person+object), `serial` = the camera digit of multi-camera sequences
  (ABF10..ABF14 -> cam0..cam4), 'cam0' otherwise.
* Frame rate: not stated in the paper; RealSense D415 default 30 fps is
  assumed (--fps). It only sets the filter's dt.

STATIC-GRASP SEQUENCES (measured 2026-10-04)
-------------------------------------------
The single-camera sequences (MC, ND, SM, SMu, SS, SiS) are annotated with ONE
fixed grasp for the whole clip: only the wrist moves, the finger angles are
constant (1-2 distinct poses in ~900-1800 frames). They cannot test a filter's
smoothness or lag, and their per-frame accuracy is against a fixed pose. The
multi-camera sequences (ABF, BB, GPMF, GSF, MDF, SB, ShSu, SiBF, and AP, MPM in
evaluation) have per-frame finger poses (0.3-1.4 deg/frame). `scan` measures
this from the annotations alone, so only the moving sequences' images need
downloading; `build` skips static ones unless --include-static.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, HERE)

SHARE_DATA = "https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO"
SHARE_GT = ("https://1drv.ms/f/c/11742dd40d1cbdc1/"
            "IgCs3QZu1A1lQpncQsj9-93kAaWblgr_-bTyXC1UVSRf91o?e=268sU9")
ZIP_NAME = "HO3D_v3.zip"
# OneDrive's own web viewer app id: yields an anonymous token for public shares
BADGER_APP_ID = "5cbed6ac-a083-4e14-b191-b4ba07653de2"
UA = {"User-Agent": "Mozilla/5.0 (kalman_filtering ho3d_data.py)"}

MANO_TO_MP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
GL_TO_CV = np.array([1.0, -1.0, -1.0])
BLOCK_RE = re.compile(r"^(train|evaluation)/([^/]+)/(rgb|meta|depth)/")


# --------------------------------------------------------------------------
# OneDrive
# --------------------------------------------------------------------------
def _json(url, data=None, headers=None):
    h = dict(UA, **(headers or {}))
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=h), timeout=60) as r:
        return json.load(r)


def share_children(link):
    """{name: {'size': int, 'url': str}} for a public OneDrive folder link."""
    tok = _json("https://api-badgerp.svc.ms/v1.0/token",
                data=json.dumps({"appId": BADGER_APP_ID}).encode(),
                headers={"Content-Type": "application/json"})["token"]
    enc = "u!" + base64.urlsafe_b64encode(link.encode()).decode().rstrip("=")
    d = _json(f"https://my.microsoftpersonalcontent.com/_api/v2.0/shares/{enc}/driveitem"
              "?$expand=children", headers={"Authorization": "Badger " + tok,
                                            "Prefer": "autoredeem"})
    return {c["name"]: {"size": c.get("size"), "url": c.get("@content.downloadUrl")}
            for c in d.get("children", [])}


class Url:
    """A download URL that refreshes itself when OneDrive expires it."""

    def __init__(self, link, name):
        self.link, self.name = link, name
        self.url, self.size, self.t = None, None, 0.0

    def get(self, force=False):
        if force or self.url is None or time.time() - self.t > 1800:
            c = share_children(self.link)[self.name]
            self.url, self.size, self.t = c["url"], c["size"], time.time()
        return self.url


def http_range(url_obj, start, end, sink, chunk=4 << 20, retries=8):
    """Write bytes [start, end] to `sink` (a file opened 'ab'), resuming and
    refreshing the URL on failure. Returns bytes written."""
    done = sink.tell()
    want = end - start + 1
    attempt = 0
    while done < want:
        try:
            req = urllib.request.Request(url_obj.get(force=attempt > 0), headers=dict(
                UA, Range=f"bytes={start + done}-{end}"))
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"server ignored the range request (HTTP {r.status})")
                while True:
                    b = r.read(chunk)
                    if not b:
                        break
                    sink.write(b)
                    done += len(b)
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(min(30, 2 ** attempt))
            print(f"    retry {attempt} at {done / 1e6:.0f} MB ({e})", flush=True)
    return done


class _HTTPReader(io.RawIOBase):
    """Seekable read-only view of a remote file (for zipfile's directory)."""

    def __init__(self, url_obj, size):
        self.u, self.size, self.pos = url_obj, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        buf = io.BytesIO()
        http_range(self.u, self.pos, self.pos + n - 1, buf)
        data = buf.getvalue()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


# --------------------------------------------------------------------------
# Zip index and block download
# --------------------------------------------------------------------------
def build_index(cache=None):
    if cache and os.path.exists(cache):
        with gzip.open(cache, "rt") as fh:
            return json.load(fh)
    import zipfile
    u = Url(SHARE_DATA, ZIP_NAME)
    u.get()
    print(f"[ho3d] reading the central directory of {ZIP_NAME} ({u.size / 1e9:.1f} GB) over HTTP")
    z = zipfile.ZipFile(io.BufferedReader(_HTTPReader(u, u.size), buffer_size=8 << 20))
    ents = sorted(({"name": i.filename, "off": i.header_offset, "csize": i.compress_size,
                    "size": i.file_size, "ctype": i.compress_type, "crc": i.CRC}
                   for i in z.infolist()), key=lambda e: e["off"])
    idx = {"zip_size": u.size, "dir_start": z.start_dir, "entries": ents}
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with gzip.open(cache, "wt") as fh:
            json.dump(idx, fh)
    print(f"[ho3d] {len(ents)} entries indexed" + (f" -> {cache}" if cache else ""))
    return idx


def blocks_of(idx):
    """{(split, seq, kind): (first position, last position)} in offset order."""
    out = {}
    for k, e in enumerate(idx["entries"]):
        m = BLOCK_RE.match(e["name"])
        if not m or e["name"].endswith("/"):
            continue
        key = m.groups()
        lo, hi = out.get(key, (k, k))
        out[key] = (min(lo, k), max(hi, k))
    return out


def _extract(tmp_path, ents, base, out_root):
    with open(tmp_path, "rb") as f:
        for e in ents:
            if e["name"].endswith("/"):
                continue
            f.seek(e["off"] - base)
            hdr = f.read(30)
            if hdr[:4] != b"PK\x03\x04":
                raise IOError(f"bad local header for {e['name']}")
            n, m = struct.unpack("<HH", hdr[26:30])
            f.seek(n + m, 1)
            data = f.read(e["csize"])
            if e["ctype"] == 8:
                data = zlib.decompress(data, -15)
            elif e["ctype"] != 0:
                raise IOError(f"unsupported compression {e['ctype']} for {e['name']}")
            if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
                raise IOError(f"CRC mismatch for {e['name']}")
            path = os.path.join(out_root, e["name"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as g:
                g.write(data)


def fetch_block(u, idx, key, span, out_root, tmp_dir):
    split, seq, kind = key
    marker = os.path.join(out_root, split, seq, f".{kind}_done")
    if os.path.exists(marker):
        return key, 0, "cached"
    ents = idx["entries"]
    lo, hi = span
    start = ents[lo]["off"]
    end = (ents[hi + 1]["off"] if hi + 1 < len(ents) else idx["dir_start"]) - 1
    tmp = os.path.join(tmp_dir, f"{split}_{seq}_{kind}.part")
    t0 = time.time()
    with open(tmp, "ab") as sink:
        n = http_range(u, start, end, sink)
    _extract(tmp, ents[lo:hi + 1], start, out_root)
    os.remove(tmp)
    open(marker, "w").write(f"{hi - lo + 1} files\n")
    return key, n, f"{n / 1e6:.0f} MB in {time.time() - t0:.0f} s"


def download(out_root, index_cache, splits, kinds, seqs=None, workers=4):
    idx = build_index(index_cache)
    u = Url(SHARE_DATA, ZIP_NAME)
    todo = {k: v for k, v in blocks_of(idx).items()
            if k[0] in splits and k[2] in kinds and (not seqs or k[1] in seqs)}
    tmp_dir = os.path.join(out_root, "_parts")
    os.makedirs(tmp_dir, exist_ok=True)
    total = sum(idx["entries"][hi]["off"] - idx["entries"][lo]["off"] for lo, hi in todo.values())
    print(f"[ho3d] {len(todo)} blocks ({total / 1e9:.1f} GB) for splits {splits}, kinds {kinds}")
    # the two frame lists (needed to read evaluation_xyz.json); calibration/ and
    # manual_annotations/ are not used here
    for k, e in enumerate(idx["entries"]):
        if e["name"] in ("train.txt", "evaluation.txt") and \
                not os.path.exists(os.path.join(out_root, e["name"])):
            fetch_one(u, idx, k, out_root, tmp_dir)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch_block, u, idx, k, v, out_root, tmp_dir) for k, v in sorted(todo.items())]
        for i, f in enumerate(as_completed(futs), 1):
            key, n, msg = f.result()
            print(f"  [{i:3d}/{len(futs)}] {'/'.join(key):28s} {msg}", flush=True)
    print(f"[ho3d] download done in {(time.time() - t0) / 60:.1f} min -> {out_root}")


def fetch_one(u, idx, k, out_root, tmp_dir):
    ents = idx["entries"]
    start = ents[k]["off"]
    end = (ents[k + 1]["off"] if k + 1 < len(ents) else idx["dir_start"]) - 1
    tmp = os.path.join(tmp_dir, f"single_{k}.part")
    with open(tmp, "wb") as sink:
        http_range(u, start, end, sink)
    _extract(tmp, [ents[k]], start, out_root)
    os.remove(tmp)


def gt_eval(out_root):
    """evaluation_xyz.json (joints for the 13 evaluation sequences, released Nov 2024)."""
    path = os.path.join(out_root, "evaluation_xyz.json")
    if os.path.exists(path):
        print(f"[ho3d] {path} already there")
        return path
    u = Url(SHARE_GT, "evaluation_xyz.json")
    u.get()
    with open(path + ".part", "ab") as sink:
        http_range(u, 0, u.size - 1, sink)
    os.replace(path + ".part", path)
    print(f"[ho3d] evaluation GT -> {path} ({u.size / 1e6:.0f} MB)")
    return path


# --------------------------------------------------------------------------
# Ground truth and compact records
# --------------------------------------------------------------------------
def seq_group(seq):
    m = re.match(r"^([A-Za-z]+)(\d+)$", seq)
    letters, digits = (m.group(1), m.group(2)) if m else (seq, "")
    cam = f"cam{digits[-1]}" if len(digits) == 2 and digits[0] in "14" else "cam0"
    return f"HO3D-{letters}", cam


def frame_ids(root, split, seq):
    d = os.path.join(root, split, seq, "rgb")
    return sorted(int(os.path.splitext(f)[0]) for f in os.listdir(d) if f.endswith(".jpg"))


def to_mp(j_mano_gl):
    j = np.asarray(j_mano_gl, float).reshape(21, 3)[MANO_TO_MP]
    return j * GL_TO_CV


def load_gt_train(root, seq, ids):
    import pickle
    gt = np.full((len(ids), 21, 3), np.nan)
    for t, fid in enumerate(ids):
        p = os.path.join(root, "train", seq, "meta", f"{fid:04d}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        j = d.get("handJoints3D")
        if j is not None and np.asarray(j).size == 63:
            gt[t] = to_mp(j)
    return gt


_EVAL_CACHE = {}


def load_gt_eval(root, seq, ids):
    """evaluation_xyz.json is one 21x3 list per line of evaluation.txt."""
    if "map" not in _EVAL_CACHE:
        xyz = json.load(open(os.path.join(root, "evaluation_xyz.json")))
        names = [ln.strip() for ln in open(os.path.join(root, "evaluation.txt")) if ln.strip()]
        if len(names) != len(xyz):
            raise ValueError(f"evaluation.txt has {len(names)} lines, evaluation_xyz.json {len(xyz)}")
        arr = np.asarray(xyz, float)
        order = detect_order(arr)
        print(f"[ho3d] evaluation_xyz.json joint order looks like: {order}")
        _EVAL_CACHE["map"] = {n: a for n, a in zip(names, arr)}
        _EVAL_CACHE["order"] = order
    gt = np.full((len(ids), 21, 3), np.nan)
    for t, fid in enumerate(ids):
        a = _EVAL_CACHE["map"].get(f"{seq}/{fid:04d}")
        if a is None:
            continue
        j = a if _EVAL_CACHE["order"] == "mano" else a[np.argsort(MANO_TO_MP)]
        gt[t] = to_mp(j)
    return gt


def detect_order(joints):
    """'mano' or 'mediapipe' from geometry: in MANO order joints 1 and 4 are
    the index and middle knuckles (~2 cm apart); in MediaPipe order they are
    the thumb CMC and thumb tip (~8 cm)."""
    j = np.asarray(joints, float).reshape(-1, 21, 3)
    d = np.nanmedian(np.linalg.norm(j[:, 1] - j[:, 4], axis=1))
    return "mano" if d < 0.045 else "mediapipe"


def check(root, n=200):
    """Sanity: joint order, units, chain lengths on a few train frames."""
    import pickle
    seqs = sorted(os.listdir(os.path.join(root, "train")))
    J = []
    for seq in seqs:
        mdir = os.path.join(root, "train", seq, "meta")
        if not os.path.isdir(mdir):
            continue
        for f in sorted(os.listdir(mdir))[::50]:
            d = pickle.load(open(os.path.join(mdir, f), "rb"), encoding="latin1")
            if d.get("handJoints3D") is not None:
                J.append(np.asarray(d["handJoints3D"], float))
        if len(J) >= n:
            break
    J = np.array(J)
    print(f"[check] {len(J)} train frames; detected order: {detect_order(J)}")
    mp = np.array([to_mp(j) for j in J])
    chains = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
              "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
    for name, c in chains.items():
        L = [np.nanmedian(np.linalg.norm(mp[:, c[k + 1]] - mp[:, c[k]], axis=1)) * 100
             for k in range(3)]
        base = np.nanmedian(np.linalg.norm(mp[:, c[0]] - mp[:, 0], axis=1)) * 100
        print(f"  {name:6s} wrist->base {base:4.1f} cm, phalanges " + " ".join(f"{x:4.1f}" for x in L))
    print(f"  wrist depth (camera z) median {np.nanmedian(mp[:, 0, 2]):.2f} m (positive = in front)")


def motion(gt_mm_angles):
    """Median per-frame finger-angle change (deg/frame) over consecutive annotated frames."""
    d = np.abs(np.diff(gt_mm_angles, axis=0))
    fin = np.isfinite(d).all(1)
    return float(np.median(d[fin].mean(1))) if fin.any() else 0.0


def eval_ids(root, seq):
    names = [ln.strip() for ln in open(os.path.join(root, "evaluation.txt")) if ln.strip()]
    return sorted(int(n.split("/")[1]) for n in names if n.split("/")[0] == seq)


def scan(root, out_json, splits=("train", "evaluation"), thr=0.05):
    """Classify sequences as moving / static-grasp from the annotations alone
    (train: meta/*.pkl, evaluation: evaluation_xyz.json)."""
    import validate_dexycb as V
    rows = []
    for split in splits:
        if split == "train":
            base = os.path.join(root, "train")
            seqs = sorted(q for q in os.listdir(base) if os.path.isdir(os.path.join(base, q, "meta")))
        else:
            if not os.path.exists(os.path.join(root, "evaluation_xyz.json")):
                print("[scan] evaluation_xyz.json missing (run gt-eval); evaluation skipped")
                continue
            names = [ln.strip() for ln in open(os.path.join(root, "evaluation.txt")) if ln.strip()]
            seqs = sorted({n.split("/")[0] for n in names})
        for seq in seqs:
            if split == "train":
                md = os.path.join(root, "train", seq, "meta")
                ids = sorted(int(f[:-4]) for f in os.listdir(md) if f.endswith(".pkl"))
                gt = load_gt_train(root, seq, ids)
            else:
                ids = eval_ids(root, seq)
                gt = load_gt_eval(root, seq, ids)
            ang = V.angles_of(gt * 1000.0)
            m = motion(ang)
            rows.append(dict(split=split, seq=seq, frames=len(ids),
                             annotated=int(np.isfinite(ang).all(1).sum()),
                             motion_deg_per_frame=round(m, 4), moving=bool(m > thr)))
    json.dump(rows, open(out_json, "w"), indent=1)
    mv = [r for r in rows if r["moving"]]
    print(f"[scan] {len(rows)} sequences: {len(mv)} with moving fingers, "
          f"{len(rows) - len(mv)} static-grasp -> {out_json}")
    for r in rows:
        print(f"  {r['split']:10s} {r['seq']:8s} {r['annotated']:5d} annotated  "
              f"{r['motion_deg_per_frame']:7.3f} deg/frame  {'moving' if r['moving'] else 'STATIC'}")
    return rows


def build(root, out_dir, splits, fps, mp_model, seqs=None, min_valid=10, include_static=False):
    import cv2
    import dexycb_data as D
    from pipeline import HandTracker
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, q) for s in splits if os.path.isdir(os.path.join(root, s))
            for q in sorted(os.listdir(os.path.join(root, s)))
            if os.path.isdir(os.path.join(root, s, q, "rgb")) and (not seqs or q in seqs)]
    print(f"[ho3d] {len(jobs)} sequences to convert at {fps} fps")
    for split, seq in jobs:
        subject, cam = seq_group(seq)
        target = os.path.join(out_dir, D.key_of(subject, f"{split}-{seq}", cam) + ".npz")
        if os.path.exists(target):
            print(f"  {split}/{seq}: cached")
            continue
        ids = frame_ids(root, split, seq)
        try:
            gt = (load_gt_train if split == "train" else load_gt_eval)(root, seq, ids)
        except FileNotFoundError as e:
            print(f"  {split}/{seq}: no GT ({e}); skipped")
            continue
        n_gt = int(np.isfinite(gt).all((1, 2)).sum())
        if n_gt < min_valid:
            print(f"  {split}/{seq}: only {n_gt} annotated frames, skipped")
            continue
        if not include_static:
            import validate_dexycb as V
            m = motion(V.angles_of(gt * 1000.0))
            if m <= 0.05:
                print(f"  {split}/{seq}: static grasp ({m:.3f} deg/frame), skipped "
                      "(--include-static to keep)")
                continue
        t0 = time.time()
        tracker = HandTracker(mp_model)          # fresh per sequence (VIDEO mode keeps state)
        world = np.full((len(ids), 21, 3), np.nan)
        label = np.full(len(ids), -1, np.int8)
        try:
            for t, fid in enumerate(ids):
                img = cv2.imread(os.path.join(root, split, seq, "rgb", f"{fid:04d}.jpg"))
                if img is None:
                    continue
                det = tracker.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), t * 1000.0 / fps)
                if det is not None:
                    world[t] = det["world"]
                    label[t] = 1 if det["hand"].startswith("r") else 0
        finally:
            tracker.landmarker.close()
        D.save_record(out_dir, subject, f"{split}-{seq}", cam, "right", world, label, gt=gt)
        det_n = int(np.isfinite(world).all((1, 2)).sum())
        print(f"  {split}/{seq}: {len(ids)} frames, {n_gt} annotated, {det_n} detected, "
              f"{time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index", help="read the zip directory; print the dataset README")
    i.add_argument("--cache", default=None)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    d.add_argument("--index", default=None, help="zip index cache (made by 'index')")
    d.add_argument("--splits", nargs="+", default=["train", "evaluation"])
    d.add_argument("--kinds", nargs="+", default=["rgb", "meta"], choices=["rgb", "meta", "depth"])
    d.add_argument("--seqs", nargs="+", default=None)
    d.add_argument("--workers", type=int, default=4)
    g = sub.add_parser("gt-eval")
    g.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--root", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--splits", nargs="+", default=["train", "evaluation"])
    b.add_argument("--seqs", nargs="+", default=None)
    b.add_argument("--fps", type=float, default=30.0)
    b.add_argument("--include-static", action="store_true",
                   help="also convert static-grasp sequences (see the docstring)")
    sc = sub.add_parser("scan", help="which sequences have moving fingers (needs meta + eval GT)")
    sc.add_argument("--root", required=True)
    sc.add_argument("--out", required=True, help="json written here")
    sc.add_argument("--splits", nargs="+", default=["train", "evaluation"])
    b.add_argument("--mp-model", default=os.path.join(ROOT, "models", "hand_landmarker.task"))
    a = ap.parse_args(argv)
    if a.cmd == "index":
        idx = build_index(a.cache)
        blocks = blocks_of(idx)
        size = {}
        for (split, _, kind), (lo, hi) in blocks.items():
            e = idx["entries"]
            size[(split, kind)] = size.get((split, kind), 0) + e[hi]["off"] + e[hi]["csize"] - e[lo]["off"]
        for k, v in sorted(size.items()):
            print(f"  {k[0]:10s} {k[1]:5s} {v / 1e9:6.2f} GB")
        readme = share_children(SHARE_DATA).get("README.txt")
        if readme:
            with urllib.request.urlopen(urllib.request.Request(readme["url"], headers=UA)) as r:
                text = r.read().decode("utf-8", "replace")
            print("\n" + text[:text.find("Contents:")].strip())
    elif a.cmd == "download":
        download(a.out, a.index, a.splits, a.kinds, a.seqs, a.workers)
    elif a.cmd == "gt-eval":
        gt_eval(a.out)
    elif a.cmd == "check":
        check(a.root)
    elif a.cmd == "scan":
        scan(a.root, a.out, a.splits)
    else:
        build(a.root, a.out, a.splits, a.fps, a.mp_model, a.seqs, include_static=a.include_static)


if __name__ == "__main__":
    main()
