import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "trained_models"
DEFAULT_SCREENSHOT_DIR = SCRIPT_DIR / "webcam_screenshots_attempt5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open webcam and draw a hand bounding box using attempt5 best_model.pth.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit path to best_model.pth. If omitted, latest attempt5 best_model.pth is used.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=480, help="Requested webcam height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested webcam FPS.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask threshold for detection.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=120.0,
        help="Minimum contour area before reporting a hand box.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIR,
        help="Directory where raw webcam screenshots are saved when 's' is pressed.",
    )
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


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def predict_mask(model: HandSegNetCPMFast, frame_tensor: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
    _, _, logits = model(frame_tensor)
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


def annotate_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    threshold: float,
    fps_value: float,
    frame_count: int,
    detection_count: int,
    model_name: str,
) -> np.ndarray:
    overlay = np.zeros_like(frame)
    overlay[mask > threshold] = (0, 255, 0)
    vis = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0.0)

    detection_rate = 100.0 * detection_count / frame_count if frame_count else 0.0

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 62), (24, 24, 24), -1)
    cv2.putText(vis, f"Attempt5 model: {model_name}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(
        vis,
        f"FPS {fps_value:.1f} | Detection rate {detection_rate:.1f}% | Frames {frame_count}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
    )

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 3)
        text_y = max(76, y - 10)
        cv2.putText(vis, "Hand detected", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        cv2.putText(vis, "No hand detected", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(
        vis,
        "q/ESC: quit | s: save screenshot",
        (12, vis.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
    )
    return vis


def save_frame(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"webcam_{timestamp}.jpg"
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame for saving.")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)

    print(f"Device: {DEVICE}")
    print(f"Using model: {model_path}")

    model = load_model(model_path)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with camera id {args.camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_w}x{actual_h}")
    print("Controls: q/ESC to exit, s to save screenshot")

    frame_count = 0
    detection_count = 0
    saved_count = 0
    start = time.perf_counter()

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from webcam.")
                    break

                frame_count += 1
                elapsed = max(time.perf_counter() - start, 1e-6)
                fps_value = frame_count / elapsed

                frame_tensor = preprocess_frame(frame).to(DEVICE)
                mask = predict_mask(model, frame_tensor, frame.shape[:2])
                bbox = get_bounding_box(mask, args.threshold, args.min_area)
                if bbox is not None:
                    detection_count += 1

                vis = annotate_frame(
                    frame,
                    mask,
                    bbox,
                    args.threshold,
                    fps_value,
                    frame_count,
                    detection_count,
                    model_path.name,
                )

                cv2.imshow("Attempt5 Webcam Hand BBox Test", vis)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    saved_path = save_frame(frame, args.save_dir)
                    saved_count += 1
                    print(f"Saved screenshot: {saved_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    detection_rate = 100.0 * detection_count / frame_count if frame_count else 0.0
    print("\nSession summary")
    print("-" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Detections: {detection_count} ({detection_rate:.2f}%)")
    print(f"Screenshots saved: {saved_count}")
    print(f"Screenshot directory: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()
