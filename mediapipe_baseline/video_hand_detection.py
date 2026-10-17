"""
Video Hand Detection - 21 2D Keypoints
Detects hand landmarks from MP4 video files and saves annotated output.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
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

LANDMARK_NAMES = [
    'WRIST',
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded: {MODEL_PATH}")


def create_detector():
    download_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, results):
    """Draw 21 2D keypoints on frame."""
    if not results.hand_landmarks:
        return frame, []

    h, w = frame.shape[:2]
    all_keypoints = []

    for idx, hand_landmarks in enumerate(results.hand_landmarks):
        # Get 2D pixel coordinates for all 21 keypoints
        keypoints_2d = []
        points = []
        for lm in hand_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))
            keypoints_2d.append({'x': px, 'y': py})

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        # Draw keypoints
        for i, point in enumerate(points):
            if i == 0:
                color = (0, 0, 255)  # Wrist - red
            elif i in [4, 8, 12, 16, 20]:
                color = (255, 0, 0)  # Fingertips - blue
            else:
                color = (0, 255, 255)  # Others - yellow
            cv2.circle(frame, point, 5, color, -1)
            # Draw keypoint ID
            cv2.putText(frame, str(i), (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw handedness label
        if results.handedness and idx < len(results.handedness):
            label = results.handedness[idx][0].category_name
            score = results.handedness[idx][0].score
            cv2.putText(frame, f"{label}: {score:.2f}",
                        (points[0][0], points[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        all_keypoints.append({
            'handedness': label if results.handedness else 'Unknown',
            'keypoints': keypoints_2d
        })

    return frame, all_keypoints


def process_video(video_path, output_dir):
    """
    Process a video file and detect 21 2D keypoints.

    Args:
        video_path: Path to input MP4 file
        output_dir: Directory to save output

    Returns:
        dict with video info and keypoints per frame
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {video_path.name}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Setup output video
    output_video_name = f"detected_{video_path.stem}.mp4"
    output_video_path = output_dir / output_video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Create detector
    detector = create_detector()

    # Process frames
    frame_idx = 0
    frame_timestamp_ms = 0
    all_frame_keypoints = []
    frames_with_hands = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_timestamp_ms = int(frame_idx * 1000 / fps)

        # Detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw landmarks and get keypoints
        annotated_frame, keypoints = draw_landmarks(frame.copy(), results)

        # Add frame info
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        if num_hands > 0:
            frames_with_hands += 1

        cv2.putText(annotated_frame, f"Frame: {frame_idx} | Hands: {num_hands}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Write frame
        out.write(annotated_frame)

        # Store keypoints
        all_frame_keypoints.append({
            'frame': frame_idx,
            'timestamp_ms': frame_timestamp_ms,
            'num_hands': num_hands,
            'hands': keypoints
        })

        frame_idx += 1

        # Progress
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...", end='\r')

    cap.release()
    out.release()

    print(f"\n  Frames with hands: {frames_with_hands}/{total_frames}")
    print(f"  Output video: {output_video_path}")

    # Save keypoints to JSON
    keypoints_file = output_dir / f"keypoints_{video_path.stem}.json"
    video_data = {
        'video_name': video_path.name,
        'resolution': {'width': frame_width, 'height': frame_height},
        'fps': fps,
        'total_frames': total_frames,
        'frames_with_hands': frames_with_hands,
        'landmark_names': LANDMARK_NAMES,
        'frames': all_frame_keypoints
    }

    with open(keypoints_file, 'w') as f:
        json.dump(video_data, f, indent=2)

    print(f"  Keypoints saved: {keypoints_file}")

    return video_data


def process_all_videos(input_dir="alldata/test_video", output_dir="output_video_detection"):
    """Process all MP4 videos in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    video_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.MP4"))

    print("=" * 60)
    print("Video Hand Detection - 21 2D Keypoints")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(video_files)} video(s)")
    print("=" * 60)

    for video_path in video_files:
        process_video(video_path, output_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"All outputs saved to: {output_dir}/")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Video Hand Detection - 21 2D Keypoints')
    parser.add_argument('--input', type=str, default='alldata/test_video',
                        help='Input directory with MP4 files')
    parser.add_argument('--output', type=str, default='output_video_detection',
                        help='Output directory')

    args = parser.parse_args()
    process_all_videos(args.input, args.output)


if __name__ == "__main__":
    main()
