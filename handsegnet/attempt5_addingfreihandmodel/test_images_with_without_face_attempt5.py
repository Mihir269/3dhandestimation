import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "trained_models"
DEFAULT_WITH_FACE_DIR = REPO_ROOT / "alldata" / "Test_Images_with_face"
DEFAULT_WITHOUT_FACE_DIR = REPO_ROOT / "alldata" / "test_Images_without_face"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_results_attempt5_face_split"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attempt5 best_model.pth on with-face and without-face test folders.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit path to best_model.pth. If omitted, latest attempt5 best_model.pth is used.",
    )
    parser.add_argument("--with-face-dir", type=Path, default=DEFAULT_WITH_FACE_DIR)
    parser.add_argument("--without-face-dir", type=Path, default=DEFAULT_WITHOUT_FACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold for detection.")
    parser.add_argument("--min-area", type=float, default=120.0, help="Minimum contour area for hand box.")
    return parser.parse_args()


def find_latest_best_model(model_root: Path) -> Path:
    candidates = [p for p in model_root.glob("**/best_model.pth") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No best_model.pth found under: {model_root}. "
            "Train attempt5 first or pass --model-path."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_model_path(model_path: Optional[Path]) -> Path:
    if model_path is not None:
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path
    return find_latest_best_model(DEFAULT_MODEL_ROOT)


def load_model(model_path: Path) -> HandSegNetCPMFast:
    model = HandSegNetCPMFast().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_images(folder: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def load_image_unicode(image_path: Path) -> Optional[np.ndarray]:
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def save_image_unicode(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ext = image_path.suffix if image_path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for saving: {image_path}")
    image_path.write_bytes(encoded.tobytes())


def sanitize_filename(name: str) -> str:
    ascii_name = name.encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in ascii_name)
    return cleaned if cleaned else "image"


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def predict_mask(model: HandSegNetCPMFast, image_tensor: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
    _, _, logits = model(image_tensor)
    mask = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    target_h, target_w = output_size
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return mask


def get_bounding_box(mask: np.ndarray, threshold: float, min_area: float) -> Optional[Tuple[int, int, int, int]]:
    mask_uint8 = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def bbox_to_dict(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Dict[str, int]]:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def annotate_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    threshold: float,
    split_name: str,
) -> np.ndarray:
    overlay = np.zeros_like(image_bgr)
    overlay[mask > threshold] = (0, 255, 0)
    vis = cv2.addWeighted(image_bgr, 0.75, overlay, 0.25, 0.0)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)
        label_y = max(28, y - 10)
        cv2.putText(vis, "Hand detected", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(vis, "No hand detected", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(vis, f"Split: {split_name}", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return vis


def process_split(
    model: HandSegNetCPMFast,
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    min_area: float,
) -> Dict[str, object]:
    split_output = output_dir / split_name
    vis_dir = split_output / "visualizations"
    mask_dir = split_output / "masks"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    stats: Dict[str, object] = {
        "split": split_name,
        "input_dir": str(input_dir),
        "total_images": len(image_paths),
        "processed": 0,
        "failed": 0,
        "detected": 0,
        "not_detected": 0,
        "results": [],
    }

    for idx, image_path in enumerate(tqdm(image_paths, desc=f"Processing {split_name}")):
        image_bgr = load_image_unicode(image_path)
        if image_bgr is None:
            stats["failed"] += 1
            stats["results"].append(
                {
                    "index": idx,
                    "file_name": image_path.name,
                    "status": "load_failed",
                    "detected": False,
                    "bbox_xywh": None,
                }
            )
            continue

        image_tensor = preprocess_image(image_bgr).to(DEVICE)
        with torch.inference_mode():
            mask = predict_mask(model, image_tensor, image_bgr.shape[:2])

        bbox = get_bounding_box(mask, threshold, min_area)
        detected = bbox is not None

        if detected:
            stats["detected"] += 1
        else:
            stats["not_detected"] += 1
        stats["processed"] += 1

        vis = annotate_image(image_bgr, mask, bbox, threshold, split_name)
        mask_binary = ((mask > threshold).astype(np.uint8) * 255)

        safe_stem = sanitize_filename(image_path.stem)
        vis_path = vis_dir / f"{idx:04d}_{safe_stem}.jpg"
        mask_path = mask_dir / f"{idx:04d}_{safe_stem}.png"

        save_image_unicode(vis_path, vis)
        save_image_unicode(mask_path, mask_binary)

        stats["results"].append(
            {
                "index": idx,
                "file_name": image_path.name,
                "status": "ok",
                "detected": detected,
                "bbox_xywh": bbox_to_dict(bbox),
                "visualization": str(vis_path),
                "mask": str(mask_path),
            }
        )

    processed = int(stats["processed"])
    detected = int(stats["detected"])
    stats["detection_rate_percent"] = 100.0 * detected / processed if processed else 0.0
    return stats


def save_summary(output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_txt = output_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("ATTEMPT5 TEST RESULTS (WITH FACE / WITHOUT FACE)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Model: {summary['model_path']}\n")
        f.write(f"Device: {summary['device']}\n")
        f.write(f"Threshold: {summary['threshold']}\n")
        f.write(f"Min area: {summary['min_area']}\n")
        f.write(f"Created at: {summary['created_at']}\n\n")

        for split_stats in summary["splits"]:
            f.write(f"Split: {split_stats['split']}\n")
            f.write("-" * 72 + "\n")
            f.write(f"Input dir: {split_stats['input_dir']}\n")
            f.write(f"Total images: {split_stats['total_images']}\n")
            f.write(f"Processed: {split_stats['processed']}\n")
            f.write(f"Failed: {split_stats['failed']}\n")
            f.write(f"Detected: {split_stats['detected']}\n")
            f.write(f"Not detected: {split_stats['not_detected']}\n")
            f.write(f"Detection rate: {split_stats['detection_rate_percent']:.2f}%\n\n")


def main() -> None:
    args = parse_args()

    with_face_dir = args.with_face_dir.expanduser().resolve()
    without_face_dir = args.without_face_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not with_face_dir.is_dir():
        raise FileNotFoundError(f"With-face directory not found: {with_face_dir}")
    if not without_face_dir.is_dir():
        raise FileNotFoundError(f"Without-face directory not found: {without_face_dir}")

    model_path = resolve_model_path(args.model_path)
    print(f"Device: {DEVICE}")
    print(f"Using model: {model_path}")
    print(f"With-face dir: {with_face_dir}")
    print(f"Without-face dir: {without_face_dir}")
    print(f"Output dir: {output_dir}")

    model = load_model(model_path)

    split_with_face = process_split(
        model=model,
        split_name="with_face",
        input_dir=with_face_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        min_area=args.min_area,
    )
    split_without_face = process_split(
        model=model,
        split_name="without_face",
        input_dir=without_face_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        min_area=args.min_area,
    )

    summary: Dict[str, object] = {
        "model_path": str(model_path),
        "device": str(DEVICE),
        "threshold": args.threshold,
        "min_area": args.min_area,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": [split_with_face, split_without_face],
    }

    save_summary(output_dir, summary)

    print("\nSummary")
    print("-" * 72)
    for split_stats in summary["splits"]:
        print(
            f"{split_stats['split']}: "
            f"processed={split_stats['processed']}, "
            f"failed={split_stats['failed']}, "
            f"detected={split_stats['detected']}, "
            f"rate={split_stats['detection_rate_percent']:.2f}%"
        )
    print(f"Summary files: {output_dir / 'summary.txt'} and {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
