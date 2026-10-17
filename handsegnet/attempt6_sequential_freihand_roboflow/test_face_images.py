"""
Test the Attempt6 final model (webcam_test_final model) on:
  1. Test_Images_with_face (20 images)
  2. test_Images_without_face (8 images)

Generates visualizations (green mask overlay + bounding box) and binary masks
for manual inspection. No ground truth needed.
"""

import torch
import numpy as np
import cv2
import os
import json
from datetime import datetime
from tqdm import tqdm
from model_handsegnet_cpm_fast import HandSegNetCPMFast

# ─── Configuration ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "phase2_roboflow_20260313_220707",
    "final_model.pth"
)

WITH_FACE_DIR    = "C:/Users/gunem/studying-2dto3d/alldata/Test_Images_with_face"
WITHOUT_FACE_DIR = "C:/Users/gunem/studying-2dto3d/alldata/test_Images_without_face"

MASK_THRESHOLD = 0.5
MIN_AREA = 120
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def collect_images(folder):
    return sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])


def load_image_unicode(path):
    """Handle unicode filenames that cv2.imread may fail on."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image_unicode(path, image):
    """Handle unicode filenames for saving."""
    ext = os.path.splitext(path)[1] if os.path.splitext(path)[1] else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        with open(path, "wb") as f:
            f.write(encoded.tobytes())


def sanitize_filename(name):
    ascii_name = name.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in ascii_name)
    return cleaned if cleaned else "image"


def get_bounding_box(mask_np):
    m = (mask_np > MASK_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None
    return cv2.boundingRect(largest)


# ═══════════════════════════════════════════════════════════
#  Process a split (with_face / without_face)
# ═══════════════════════════════════════════════════════════

def process_split(model, split_name, input_dir, output_root):
    vis_dir  = os.path.join(output_root, split_name, "visualizations")
    mask_dir = os.path.join(output_root, split_name, "masks")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_files = collect_images(input_dir)
    results = []
    detected_count = 0

    print(f"\nProcessing {split_name}: {len(image_files)} images from {input_dir}")

    with torch.no_grad():
        for idx, fname in enumerate(tqdm(image_files, desc=split_name)):
            img_path = os.path.join(input_dir, fname)
            img_bgr = load_image_unicode(img_path)
            if img_bgr is None:
                print(f"  WARNING: could not load {fname}")
                results.append({"file": fname, "status": "load_failed", "detected": False})
                continue

            h, w = img_bgr.shape[:2]

            # preprocess
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # inference
            _, _, out = model(inp)
            pred = torch.sigmoid(out).squeeze().cpu().numpy()
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

            # visualisation
            vis = img_bgr.copy()
            overlay = np.zeros_like(img_bgr)
            overlay[pred > MASK_THRESHOLD] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # bounding box
            bbox = get_bounding_box(pred)
            detected = bbox is not None
            if detected:
                x, y, bw, bh = bbox
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
                label_y = max(28, y - 10)
                cv2.putText(vis, "Hand Detected", (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                detected_count += 1
            else:
                cv2.putText(vis, "No Hand Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # label on bottom
            cv2.putText(vis, f"Attempt6 Final | {split_name}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # save
            safe_stem = sanitize_filename(os.path.splitext(fname)[0])
            vis_path  = os.path.join(vis_dir, f"{idx:04d}_{safe_stem}.jpg")
            mask_path = os.path.join(mask_dir, f"{idx:04d}_{safe_stem}.png")

            save_image_unicode(vis_path, vis)
            mask_binary = (pred > MASK_THRESHOLD).astype(np.uint8) * 255
            save_image_unicode(mask_path, mask_binary)

            bbox_dict = None
            if bbox:
                x, y, bw, bh = bbox
                bbox_dict = {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}

            results.append({
                "file": fname,
                "status": "ok",
                "detected": detected,
                "bbox": bbox_dict,
                "vis": vis_path,
                "mask": mask_path,
            })

    total = len(image_files)
    rate = 100.0 * detected_count / total if total else 0
    print(f"  Detection rate: {detected_count}/{total} ({rate:.1f}%)")

    return {
        "split": split_name,
        "input_dir": input_dir,
        "total": total,
        "detected": detected_count,
        "not_detected": total - detected_count,
        "detection_rate": rate,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Loading Attempt6 final model (webcam_test_final model)...")
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(
        os.path.dirname(__file__),
        f"test_face_results_attempt6_{timestamp}"
    )

    # ── with face ──
    with_face_stats = process_split(model, "with_face", WITH_FACE_DIR, output_root)

    # ── without face ──
    without_face_stats = process_split(model, "without_face", WITHOUT_FACE_DIR, output_root)

    # ── save summary ──
    summary = {
        "model": MODEL_PATH,
        "device": str(DEVICE),
        "threshold": MASK_THRESHOLD,
        "min_area": MIN_AREA,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": [with_face_stats, without_face_stats],
    }

    summary_json = os.path.join(output_root, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    summary_txt = os.path.join(output_root, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"ATTEMPT6 FINAL MODEL - FACE/NO-FACE TEST RESULTS\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"{'='*60}\n\n")
        for s in [with_face_stats, without_face_stats]:
            f.write(f"Split: {s['split']}\n")
            f.write(f"{'─'*60}\n")
            f.write(f"  Input dir     : {s['input_dir']}\n")
            f.write(f"  Total images  : {s['total']}\n")
            f.write(f"  Detected      : {s['detected']}\n")
            f.write(f"  Not detected  : {s['not_detected']}\n")
            f.write(f"  Detection rate: {s['detection_rate']:.1f}%\n\n")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {os.path.abspath(output_root)}")
    print(f"  with_face/    : {with_face_stats['detected']}/{with_face_stats['total']} detected ({with_face_stats['detection_rate']:.1f}%)")
    print(f"  without_face/ : {without_face_stats['detected']}/{without_face_stats['total']} detected ({without_face_stats['detection_rate']:.1f}%)")
    print(f"Summary: {summary_txt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
