"""
MediaPipe Hand Detection on test_images folder
Detects hands and saves annotated images.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from pathlib import Path
import urllib.request


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded: {MODEL_PATH}")


def create_detector(min_detection_confidence=0.5):
    download_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(image, results):
    if not results.hand_landmarks:
        return image

    h, w = image.shape[:2]

    for idx, hand_landmarks in enumerate(results.hand_landmarks):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

        for i, point in enumerate(points):
            if i == 0:
                color = (0, 0, 255)  # Wrist - red
            elif i in [4, 8, 12, 16, 20]:
                color = (255, 0, 0)  # Fingertips - blue
            else:
                color = (0, 255, 255)  # Others - yellow
            cv2.circle(image, point, 5, color, -1)

        if results.handedness and idx < len(results.handedness):
            label = results.handedness[idx][0].category_name
            score = results.handedness[idx][0].score
            cv2.putText(image, f"{label}: {score:.2f}",
                        (points[0][0], points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return image


def process_test_images(input_dir="alldata/test_images", output_dir="output_test_images"):
    """Process all images in test_images folder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_path, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in image_extensions]

    print("=" * 60)
    print("MediaPipe Hand Detection - test_images")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(image_files)} images")
    print("-" * 60)

    detector = create_detector(min_detection_confidence=0.5)

    total_hands = 0
    detected_images = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read: {img_path.name}")
            continue

        # Detect
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        total_hands += num_hands

        if num_hands > 0:
            detected_images += 1

        # Draw landmarks
        annotated = image.copy()
        draw_landmarks(annotated, results)

        # Add info text
        cv2.putText(annotated, f"Hands: {num_hands}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save
        output_file = output_path / f"detected_{img_path.name}"
        cv2.imwrite(str(output_file), annotated)

        print(f"{img_path.name}: {num_hands} hand(s)")

    print("-" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Images with hands: {detected_images}")
    print(f"Total hands detected: {total_hands}")
    print(f"Output saved to: {output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Hand detection on test_images')
    parser.add_argument('--input', type=str, default='alldata/test_images',
                        help='Input directory')
    parser.add_argument('--output', type=str, default='output_test_images',
                        help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold')

    args = parser.parse_args()
    process_test_images(args.input, args.output)


if __name__ == "__main__":
    main()
