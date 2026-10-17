import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "webcam_screenshots"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "webcam_screenshot_bboxes"
MODEL_PATHS = {
    "best": SCRIPT_DIR / "trained_models" / "transfer_learning_20260309_014131" / "best_model.pth",
    "final": SCRIPT_DIR / "trained_models" / "transfer_learning_20260309_014131" / "final_model.pth",
}
MODEL_COLORS = {
    "best": (255, 0, 0),
    "final": (0, 165, 255),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hand bounding boxes for images stored in webcam_screenshots.",
    )
    parser.add_argument(
        "--model",
        choices=("best", "final", "both"),
        default="both",
        help="Checkpoint to run. 'both' writes a separate result set for best and final.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the webcam screenshots to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotated images, masks, and summaries are written.",
    )
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
    return parser.parse_args()


def resolve_model_names(selection: str) -> List[str]:
    if selection == "both":
        return ["best", "final"]
    return [selection]


def find_image_paths(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        return []

    image_paths = [
        image_path
        for image_path in input_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def load_model(model_path: Path) -> HandSegNetCPMFast:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = HandSegNetCPMFast().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def read_image(image_path: Path) -> Optional[np.ndarray]:
    file_bytes = image_path.read_bytes()
    if not file_bytes:
        return None

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def save_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    extension = image_path.suffix.lower() or ".png"
    if extension not in IMAGE_EXTENSIONS and extension != ".png":
        extension = ".png"

    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for writing: {image_path}")
    image_path.write_bytes(encoded.tobytes())


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 255.0


def predict_mask(model: HandSegNetCPMFast, image_tensor: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
    _, _, logits = model(image_tensor)
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


def annotate_image(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    threshold: float,
    title: str,
    bbox_color: Tuple[int, int, int],
) -> np.ndarray:
    overlay = np.zeros_like(image)
    overlay[mask > threshold] = (0, 255, 0)

    annotated = cv2.addWeighted(image, 0.75, overlay, 0.25, 0.0)
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (24, 24, 24), -1)
    cv2.putText(annotated, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if bbox is not None:
        x, y, width, height = bbox
        cv2.rectangle(annotated, (x, y), (x + width, y + height), bbox_color, 3)
        label_top = max(54, y - 10)
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
            (12, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return annotated


def bbox_to_dict(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Dict[str, int]]:
    if bbox is None:
        return None

    x, y, width, height = bbox
    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }


def process_images_for_model(
    model_name: str,
    image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    threshold: float,
    min_area: float,
) -> None:
    model_path = MODEL_PATHS[model_name]
    print(f"Loading {model_name} checkpoint: {model_path}")
    model = load_model(model_path)

    model_output_dir = output_dir / model_name
    annotated_dir = model_output_dir / "annotated"
    masks_dir = model_output_dir / "masks"
    results = []
    detected_count = 0

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{model_name}] {index}/{len(image_paths)} {image_path.name}")
            image = read_image(image_path)
            if image is None:
                results.append(
                    {
                        "image": str(image_path.relative_to(input_dir)).replace("\\", "/"),
                        "detected": False,
                        "bbox_xywh": None,
                        "error": "Could not read image.",
                    }
                )
                continue

            image_tensor = preprocess_image(image).to(DEVICE)
            mask = predict_mask(model, image_tensor, image.shape[:2])
            bbox = get_bounding_box(mask, threshold, min_area)
            if bbox is not None:
                detected_count += 1

            relative_image_path = image_path.relative_to(input_dir)
            annotated_output_path = annotated_dir / relative_image_path
            mask_output_path = (masks_dir / relative_image_path).with_suffix(".png")

            annotated_image = annotate_image(
                image,
                mask,
                bbox,
                threshold,
                f"{model_name.upper()} checkpoint",
                MODEL_COLORS[model_name],
            )
            binary_mask = (mask > threshold).astype(np.uint8) * 255

            save_image(annotated_output_path, annotated_image)
            save_image(mask_output_path, binary_mask)

            results.append(
                {
                    "image": str(relative_image_path).replace("\\", "/"),
                    "detected": bbox is not None,
                    "bbox_xywh": bbox_to_dict(bbox),
                    "annotated_output": str(annotated_output_path.relative_to(output_dir)).replace("\\", "/"),
                    "mask_output": str(mask_output_path.relative_to(output_dir)).replace("\\", "/"),
                }
            )

    summary = {
        "model": model_name,
        "model_path": str(model_path),
        "input_dir": str(input_dir),
        "output_dir": str(model_output_dir),
        "threshold": threshold,
        "min_area": min_area,
        "images_processed": len(image_paths),
        "detections": detected_count,
        "not_detected": len(image_paths) - detected_count,
        "results": results,
    }

    model_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    detection_rate = 100.0 * detected_count / len(image_paths) if image_paths else 0.0
    print("-" * 60)
    print(f"Model: {model_name}")
    print(f"Processed: {len(image_paths)} images")
    print(f"Detected: {detected_count} ({detection_rate:.2f}%)")
    print(f"Annotated images: {annotated_dir.resolve()}")
    print(f"Masks: {masks_dir.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


def main() -> None:
    args = parse_args()
    image_paths = find_image_paths(args.input_dir)

    if not image_paths:
        print(f"No images found in: {args.input_dir.resolve()}")
        return

    print(f"Device: {DEVICE}")
    print(f"Found {len(image_paths)} images in {args.input_dir.resolve()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in resolve_model_names(args.model):
        process_images_for_model(
            model_name=model_name,
            image_paths=image_paths,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_area=args.min_area,
        )


if __name__ == "__main__":
    main()