import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from model_handsegnet_cpm_fast import HandSegNetCPMFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCREENSHOT_DIR = SCRIPT_DIR / "webcam_screenshots"
MODEL_PATHS = {
    "best": SCRIPT_DIR / "trained_models" / "transfer_learning_20260309_014131" / "best_model.pth",
    "final": SCRIPT_DIR / "trained_models" / "transfer_learning_20260309_014131" / "final_model.pth",
}
MODEL_COLORS = {
    "best": (255, 0, 0),
    "final": (0, 165, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run webcam hand segmentation using the transfer-learning checkpoints.",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final", "both"),
        default="both",
        help="Checkpoint to run. 'both' shows the best and final checkpoints side by side.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested webcam width.")
    parser.add_argument("--height", type=int, default=480, help="Requested webcam height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested webcam FPS.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask threshold used for overlay and bounding box extraction.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area required before a hand box is reported.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIR,
        help="Directory where raw webcam screenshots are saved when 's' is pressed.",
    )
    return parser.parse_args()


def resolve_model_names(selection: str) -> List[str]:
    if selection == "both":
        return ["best", "final"]
    return [selection]


def load_model(model_path: Path) -> HandSegNetCPMFast:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = HandSegNetCPMFast().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 255.0


def predict_mask(model: HandSegNetCPMFast, frame_tensor: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
    _, _, logits = model(frame_tensor)
    mask = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

    target_height, target_width = output_size
    if mask.shape != (target_height, target_width):
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return mask


def get_bounding_box(mask: np.ndarray, threshold: float, min_area: float) -> Optional[Tuple[int, int, int, int]]:
    mask_uint8 = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return None

    x, y, width, height = cv2.boundingRect(largest_contour)
    return x, y, width, height


def annotate_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    title: str,
    bbox_color: Tuple[int, int, int],
    threshold: float,
    frame_count: int,
    detection_count: int,
    fps_value: float,
) -> np.ndarray:
    overlay = np.zeros_like(frame)
    overlay[mask > threshold] = (0, 255, 0)

    annotated = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0.0)
    detection_rate = 100.0 * detection_count / frame_count if frame_count else 0.0

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (24, 24, 24), -1)
    cv2.putText(annotated, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        annotated,
        f"FPS {fps_value:.1f} | Detections {detection_rate:.1f}% | Frames {frame_count}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )

    if bbox is not None:
        x, y, width, height = bbox
        cv2.rectangle(annotated, (x, y), (x + width, y + height), bbox_color, 3)
        label_top = max(62, y - 10)
        cv2.putText(
            annotated,
            "Hand detected",
            (x, label_top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            bbox_color,
            2,
        )
    else:
        cv2.putText(
            annotated,
            "No hand detected",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        annotated,
        "q/ESC: quit | s: save screenshot",
        (12, annotated.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
    )
    return annotated


def save_frame(frame: np.ndarray, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"webcam_{timestamp}.jpg"
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode webcam frame for saving.")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def run_webcam_test(args: argparse.Namespace) -> None:
    model_names = resolve_model_names(args.model)
    models: Dict[str, HandSegNetCPMFast] = {}

    print(f"Device: {DEVICE}")
    for model_name in model_names:
        model_path = MODEL_PATHS[model_name]
        print(f"Loading {model_name} checkpoint: {model_path}")
        models[model_name] = load_model(model_path)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with camera id {args.camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_width}x{actual_height}")
    print("Controls: q/ESC to exit, s to save the current raw frame.")

    frame_count = 0
    saved_count = 0
    detection_counts = {model_name: 0 for model_name in model_names}
    start_time = time.perf_counter()
    window_title = "Transfer-Learning Webcam Test"

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from the webcam.")
                    break

                frame_count += 1
                elapsed = max(time.perf_counter() - start_time, 1e-6)
                fps_value = frame_count / elapsed

                frame_tensor = preprocess_frame(frame).to(DEVICE)
                displays = []

                for model_name in model_names:
                    mask = predict_mask(models[model_name], frame_tensor, frame.shape[:2])
                    bbox = get_bounding_box(mask, args.threshold, args.min_area)
                    if bbox is not None:
                        detection_counts[model_name] += 1

                    title = f"{model_name.upper()} checkpoint"
                    displays.append(
                        annotate_frame(
                            frame,
                            mask,
                            bbox,
                            title,
                            MODEL_COLORS[model_name],
                            args.threshold,
                            frame_count,
                            detection_counts[model_name],
                            fps_value,
                        )
                    )

                display_frame = displays[0] if len(displays) == 1 else cv2.hconcat(displays)
                cv2.imshow(window_title, display_frame)

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

    print("\nSession summary")
    print("-" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Screenshots saved: {saved_count}")
    for model_name in model_names:
        detections = detection_counts[model_name]
        detection_rate = 100.0 * detections / frame_count if frame_count else 0.0
        print(f"{model_name}: {detections} detections ({detection_rate:.2f}%)")
    print(f"Screenshot directory: {args.save_dir.resolve()}")


if __name__ == "__main__":
    run_webcam_test(parse_args())