"""
MediaPipe Webcam Hand Detection
Live hand landmark detection on webcam feed.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
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

LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]


def download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded: {MODEL_PATH}")


def draw_landmarks(frame, results):
    """Draw hand landmarks on the frame."""
    if not results.hand_landmarks:
        return frame

    h, w = frame.shape[:2]

    for idx, hand_landmarks in enumerate(results.hand_landmarks):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        # Draw landmarks
        for i, point in enumerate(points):
            # Wrist is red, fingertips are blue, others are yellow
            if i == 0:
                color = (0, 0, 255)  # Red - wrist
            elif i in [4, 8, 12, 16, 20]:
                color = (255, 0, 0)  # Blue - fingertips
            else:
                color = (0, 255, 255)  # Yellow - others
            cv2.circle(frame, point, 5, color, -1)

        # Draw handedness label
        if results.handedness and idx < len(results.handedness):
            label = results.handedness[idx][0].category_name
            score = results.handedness[idx][0].score
            cv2.putText(frame, f"{label}: {score:.2f}",
                        (points[0][0], points[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame


SCREENSHOT_DIR = "webcam_screenshots"


def run_webcam(max_num_hands=2, min_detection_confidence=0.5, camera_id=0):
    """
    Run live hand detection on webcam.

    Args:
        max_num_hands: Maximum number of hands to detect
        min_detection_confidence: Minimum confidence threshold
        camera_id: Camera device ID (0 for default webcam)
    """
    download_model()

    # Create screenshot directory
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Create detector with VIDEO mode for better tracking
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (camera_id={camera_id})")
        return

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print(f"Screenshots will be saved to: {SCREENSHOT_DIR}/")
    print("Press 'q' to quit, 's' to save screenshot")
    print("-" * 50)

    frame_timestamp_ms = 0
    screenshot_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            continue

        # Flip horizontally for selfie-view (mirror effect)
        frame = cv2.flip(frame, 1)
        frame_timestamp_ms += int(1000 / max(fps, 30))

        # Convert to MediaPipe format and detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw landmarks
        draw_landmarks(frame, results)

        # Display info
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        cv2.putText(frame, f"Hands detected: {num_hands}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display 3D world coordinates for each hand
        if results.hand_world_landmarks:
            for i, world_landmarks in enumerate(results.hand_world_landmarks):
                wrist = world_landmarks[0]
                label = results.handedness[i][0].category_name if results.handedness else "?"
                text = f"{label} wrist: ({wrist.x:.3f}, {wrist.y:.3f}, {wrist.z:.3f}) m"
                cv2.putText(frame, text, (10, 60 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show FPS
        cv2.putText(frame, "Press Q to quit | S to save", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        cv2.imshow('MediaPipe Hand Detection - Live', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{screenshot_count}.jpg")
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MediaPipe Webcam Hand Detection')
    parser.add_argument('--max-hands', type=int, default=2,
                        help='Maximum number of hands to detect (default: 2)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum detection confidence (default: 0.5)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')

    args = parser.parse_args()

    print("=" * 50)
    print("MediaPipe Webcam Hand Detection")
    print("=" * 50)
    print(f"Max hands: {args.max_hands}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Camera ID: {args.camera}")
    print("=" * 50)

    run_webcam(
        max_num_hands=args.max_hands,
        min_detection_confidence=args.confidence,
        camera_id=args.camera
    )


if __name__ == "__main__":
    main()
