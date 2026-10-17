"""
MediaPipe Hand Detection - 3D Keypoint Estimation
Detects 21 hand landmarks from images.
Saves annotated images with hand landmarks drawn.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from pathlib import Path
import urllib.request


class HandDetector:
    """MediaPipe-based hand detector for 21 3D keypoints."""

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

    def __init__(self, max_num_hands=2, min_detection_confidence=0.5):
        self._download_model()

        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _download_model(self):
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading hand landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded: {self.MODEL_PATH}")

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.detector.detect(mp_image)

    def draw_landmarks(self, image, results):
        if not results.hand_landmarks:
            return image

        h, w = image.shape[:2]

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

            for i, point in enumerate(points):
                color = (0, 0, 255) if i == 0 else (255, 0, 0)
                cv2.circle(image, point, 5, color, -1)

            if results.handedness and idx < len(results.handedness):
                label = results.handedness[idx][0].category_name
                score = results.handedness[idx][0].score
                cv2.putText(image, f"{label}: {score:.2f}", (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return image

    def close(self):
        pass


def process_image(image_path, detector, output_dir=None):
    """Process a single image and save annotated result."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    results = detector.detect(image)
    annotated_image = image.copy()
    detector.draw_landmarks(annotated_image, results)

    num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
    cv2.putText(annotated_image, f"Hands: {num_hands}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = Path(image_path).name
        output_path = Path(output_dir) / f"detected_{filename}"
        cv2.imwrite(str(output_path), annotated_image)

    return num_hands


def process_folder(folder_path, detector, output_dir=None):
    """Process all images in a folder."""
    folder = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    image_files = [f for f in folder.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\nProcessing {len(image_files)} images from: {folder}")
    print("-" * 50)

    total_hands = 0
    for img_path in image_files:
        num_hands = process_image(img_path, detector, output_dir)
        if num_hands is not None:
            total_hands += num_hands
            print(f"{img_path.name}: {num_hands} hand(s)")

    return len(image_files), total_hands


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MediaPipe Hand Detection')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'image', 'folder'],
                        help='Detection mode')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image or folder path')
    parser.add_argument('--output', type=str, default='output_hands',
                        help='Output directory for annotated images')
    parser.add_argument('--max-hands', type=int, default=2,
                        help='Maximum number of hands to detect')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum detection confidence')

    args = parser.parse_args()

    detector = HandDetector(
        max_num_hands=args.max_hands,
        min_detection_confidence=args.confidence
    )

    if args.mode == 'test':
        with_face_dir = "alldata/Test_Images_with_face"
        if os.path.exists(with_face_dir):
            print("\n" + "=" * 60)
            print("TESTING: Images WITH Face")
            print("=" * 60)
            num_imgs, total_hands = process_folder(
                with_face_dir, detector,
                output_dir=f"{args.output}/with_face"
            )
            print(f"\nWith face: {num_imgs} images, {total_hands} hands detected")

        without_face_dir = "alldata/test_Images_without_face"
        if os.path.exists(without_face_dir):
            print("\n" + "=" * 60)
            print("TESTING: Images WITHOUT Face")
            print("=" * 60)
            num_imgs, total_hands = process_folder(
                without_face_dir, detector,
                output_dir=f"{args.output}/without_face"
            )
            print(f"\nWithout face: {num_imgs} images, {total_hands} hands detected")

        print(f"\nOutput saved to: {args.output}/")

    elif args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        process_image(args.input, detector, args.output)
        print(f"Output saved to: {args.output}/")

    elif args.mode == 'folder':
        if not args.input:
            print("Error: --input required for folder mode")
            return
        process_folder(args.input, detector, args.output)
        print(f"Output saved to: {args.output}/")

    detector.close()


if __name__ == "__main__":
    main()
