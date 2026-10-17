"""
Webcam test for the FreiHAND fine-tuned model (Phase 1 output of Attempt 6).
Shows real-time hand segmentation with bounding box overlay.

Controls:
  q / ESC  — quit
  s        — save screenshot
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from model_handsegnet_cpm_fast import HandSegNetCPMFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "trained_models_20260313_155930",
    "model_after_freihand.pth"
)

CAMERA_ID = 0
MASK_THRESHOLD = 0.5


def get_bounding_box(mask):
    m = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    return cv2.boundingRect(largest)


def main():
    print("Loading model...")
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {w}x{h}")
    print("Press 'q'/ESC to quit, 's' to save screenshot\n")

    save_dir = os.path.join(os.path.dirname(__file__), "webcam_freihand_test")
    os.makedirs(save_dir, exist_ok=True)
    shot_count = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Inference
            _, _, out = model(inp)

            # Mask
            pred = torch.sigmoid(out).squeeze().cpu().numpy()
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

            # Visualisation
            display = frame.copy()

            # Green mask overlay
            overlay = np.zeros_like(frame)
            overlay[pred > MASK_THRESHOLD] = [0, 255, 0]
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

            # Bounding box
            bbox = get_bounding_box(pred)
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(display, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
                cv2.rectangle(display, (x, y - 30), (x + 160, y), (255, 0, 0), -1)
                cv2.putText(display, "Hand Detected", (x + 5, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(display, "No Hand Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(display, "Attempt6 - FreiHAND Model", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Attempt6 FreiHAND Model - Webcam Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                shot_count += 1
                path = os.path.join(save_dir, f"screenshot_{shot_count:04d}.jpg")
                cv2.imwrite(path, display)
                mask_path = os.path.join(save_dir, f"mask_{shot_count:04d}.png")
                cv2.imwrite(mask_path, (pred > MASK_THRESHOLD).astype(np.uint8) * 255)
                print(f"Saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
