# Hand bone-length calibration — two marker pipelines

Two standalone scripts. Each takes photos and/or videos of your flat hand on the marked
sheet, measures the 20 bone lengths with MediaPipe 2D, converts pixels to millimetres
using the printed markers, and reports mean / median / **std / variance** / CV / CI.

| Script | Marker | Nominal size |
|---|---|---|
| `calibrate_bones_filled_squares.py` | filled purple squares | 20 mm (`--square-mm 20`) |
| `calibrate_bones_border_squares.py` | hand-drawn outline squares | 30 mm (`--square-mm 30`) |

Both are self-contained — copy either one to Colab on its own.

---

## Quick start

```bash
pip install opencv-python mediapipe numpy pandas scipy matplotlib pillow

python calibrate_bones_filled_squares.py \
    --input ./filled_shots --out ./calib_filled \
    --square-mm 20 --model hand_landmarker.task \
    --hand Right --plots --debug --parallax

python calibrate_bones_border_squares.py \
    --input ./border_shots --out ./calib_border \
    --square-mm 30 --model hand_landmarker.task \
    --hand Right --min-markers 1 --plots --debug --parallax
```

`--input` takes a directory (recursively picks up images **and** videos) or a list of
files. Videos are sampled every `--frame-step` frames, capped at `--max-frames`.
`--model` is optional: with a `hand_landmarker.task` path it uses the Tasks API
(same as the rest of the project); without it, it falls back to `mp.solutions.hands`.

---

## What the scripts actually do

**1. Find the markers.**
*Filled:* HSV threshold → contour → quad → sub-pixel corner refinement.
*Outline:* the pen line is barely darker than the printed major grid, so plain
thresholding fails. The discriminator that works is **stroke thickness** —
illumination-flatten, threshold, then `MORPH_OPEN` with an ellipse sized to the pen
width. The thin printed grid vanishes; the pen strokes survive. Rings are then kept by a
*low* fill ratio (a ring, not a disc), and corners come from fitting lines to the stroke
**mid-line**, so the measured side is the 30 mm centre-line.

**2. Map image → paper plane.**
With ≥3 squares it solves a full **perspective homography**. Since the squares are all
the same size and orientation but at unknown positions, the script solves for the
homography *and* each square's in-plane translation jointly (SciPy refinement, square 0
pinned at the origin). With <3 squares it falls back to a px/mm scale field (linear in
image position when ≥5 squares, else constant).

**3. Measure.** Landmarks are pushed through `H⁻¹` into millimetres and the 20 bone
lengths are computed there. Bones tilted more than `--max-tilt-deg` (default 22°) out of
the paper plane — read from MediaPipe's world landmarks — are dropped, because
out-of-plane bones are foreshortened and would bias every length *downward*.

**4. Aggregate.** Per bone: `n_used`, mean, median, **std, variance**, CV%, MAD, SEM,
95% CI, p5/p95, IQR, and count of MAD-rejected outliers (modified z > 3.5).

---

## Using your multi-height shots — this is the important one

Your hand rests **on top of** the paper, so the back of the hand sits ~20 mm above the
marker plane and is magnified by `D / (D − h)` where `D` is the camera-to-paper
distance. At a 300 mm shooting distance that is a **7 % overestimate on every bone** — far
larger than the ~1 % measurement noise. This is the dominant error in the whole setup.

Shots from different heights let you solve for it. With `--parallax`, the script:

- estimates `D` per frame from EXIF focal length and the marker scale (`D = f_px / s`),
- fits `L_measured = L_true · D/(D − h)` over all frames, with **one global `h`** and one
  `L_true` per bone (grid search over `h`, closed-form `L_true` at each step),
- writes `parallax_corrected_mm` into `bone_length_stats.csv` and reports `h_mm`.

It warns when your camera-distance range is under 15 %, because `h` is then weakly
identified. **Shoot deliberately: some frames close (~20 cm), some far (~60 cm).** If EXIF
focal length is missing, pass `--focal-px`, or skip the fit and apply a fixed correction
with `--hand-thickness-mm 20`.

---

## Outputs

| File | Contents |
|---|---|
| `bone_length_stats.csv` | per bone: mean, median, **std, var**, CV%, MAD, SEM, CI95, p5/p95, IQR, `normalized` (÷ Middle MCP–PIP), `parallax_corrected_mm` |
| `per_frame_bone_lengths_mm.csv` | one row per (frame, bone) with tilt and validity |
| `frame_report.csv` | per frame: marker count, px/mm, mapping method, reprojection RMS, status |
| `bone_lengths.json` | final lengths in **mm and metres**, plus normalized — drop straight into FK |
| `summary.json` | config, frame counts, mapping-method histogram, mean CV, parallax result |
| `bone_length_distributions.png`, `bone_length_cv.png` | with `--plots`, in project finger colours |
| `debug/*.jpg` | with `--debug`, overlays of detected squares + skeleton |

`bone_lengths.json` keys follow the project's 20-bone order with
`Middle_MCP_PIP` at index 9 as the normalization reference.

---

## What I verified on your four photos

Detection and geometry were validated directly on the images you uploaded:

- **Filled squares: 6/6 detected in both photos.** Side-length spread 0.98 % and 1.13 %
  (that spread *is* the perspective gradient across the sheet, which the homography removes).
- **Homography solve: reprojection RMS ≈ 2.0–2.4 px** on ~163 px squares (~1.3 %) —
  consistent with hand-drawn squares not being geometrically perfect.
- **Homography vs scale-field paths agree to 0.27 % / 0.60 %** on a synthetic hand
  projected through the real detected geometry, so perspective across the hand region
  costs well under 1 %.
- **Outline squares: 2 detected per photo**, sides consistent to 0.15 % and 0.65 %,
  giving 8.84 and 8.68 px/mm.
- **Sub-pixel refinement matters:** raw `approxPolyDP` corners gave 158.4 px per square,
  the edge-line refinement gives 163.1 px — a **3 % scale shift**, i.e. 3 % on every bone.

### Expect fewer outline squares than filled ones
Only the squares that are fully visible *and* unoccluded are usable — in your border
photos that is 2 of 6 (the rest are under the hand or clipped by the frame). Two squares
give a solid scale but not a homography, so those runs fall back to a constant scale
field. Either move the squares further out so 4+ stay clear of the hand, or prefer the
filled-marker set, which is markedly more robust.

---

## Tuning

**Filled squares** — if detection fails, run `--debug` and check the overlays first.
Purple lives around OpenCV hue 105–165; widen with `--hue-lo/--hue-hi`, drop
`--sat-min` for washed-out lighting.

**Outline squares** — `--dark-thr` defaults to sweeping `[18,21,24,27,30]` and keeping
the most self-consistent set; 24 was best on your photos. `--stroke-px` auto-sizes from
the image (7 px at 3072×4080); raise it if the printed grid survives, lower it if your
pen strokes get erased. `--min-markers 1` is reasonable here.

**Systematic bias to be aware of:** for the filled squares, the measured edge is the
*outer* boundary of the ink, which may sit slightly outside the 20 mm box you drew. Any
error here scales every bone linearly. `--edge-bias-mm` subtracts a fixed amount if you
measure the discrepancy with calipers. The outline squares don't have this problem — the
mid-line fit targets the drawn line's centre.

**Experimental flags:** `--grid-check` and `--grid-field` try to use the printed 1 mm
graph paper as an independent ruler via FFT pitch estimation. It gave 8.147 px on one
photo (matching the marker scale to 0.04 %) but locked onto unrelated peaks on the
others, likely confounded by JPEG 8×8 block artifacts. Both are off by default; treat any
number they produce as unverified.

---

## Comparing the two runs

Both scripts write the same schema, so cross-validating the 20 mm and 30 mm references
is a few lines:

```python
import pandas as pd
a = pd.read_csv('calib_filled/bone_length_stats.csv').set_index('bone')
b = pd.read_csv('calib_border/bone_length_stats.csv').set_index('bone')
cmp = pd.DataFrame({'filled_mm': a.median_mm, 'border_mm': b.median_mm})
cmp['diff_pct'] = 100 * (cmp.border_mm - cmp.filled_mm) / cmp.filled_mm
print(cmp.round(2))
```

A consistent offset across *all* bones is a scale error in one of the two references
(check `--square-mm` and `--edge-bias-mm`). Scatter that varies bone-by-bone is
MediaPipe noise, and the per-bone `std` columns tell you how much.
