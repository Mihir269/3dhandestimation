"""
Webcam test for the Combined 40/40/20 final model.
Uses the best_model.pth from train_combined_40_40_20 training.

Controls:
  q / ESC  — quit
  s        — save screenshot
  m        — toggle between best_model and final_model
"""

import torch
import numpy as np
import cv2
import os
import time
from model_handsegnet_cpm_fast import HandSegNetCPMFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMBINED_DIR = os.path.join(
    os.path.dirname(__file__),
    "final_combined_20260314_020453"
)
BEST_MODEL_PATH = os.path.join(COMBINED_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(COMBINED_DIR, "final_model.pth")

CAMERA_ID = 0
MASK_THRESHOLD = 0.5


def load_model(path):
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


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
    # Start with best model
    use_best = True
    current_path = BEST_MODEL_PATH
    print("Loading Combined 40/40/20 best model...")
    model = load_model(current_path)
    print(f"Model loaded: {current_path}")
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
    print("Controls: 'q'/ESC=quit, 's'=screenshot, 'm'=toggle best/final model\n")

    save_dir = os.path.join(os.path.dirname(__file__), "webcam_combined_test")
    os.makedirs(save_dir, exist_ok=True)
    shot_count = 0
    prev_time = time.time()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # Inference (use stage 3 output)
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

            # Info bar
            model_label = "BEST model (epoch 8)" if use_best else "FINAL model (epoch 10)"
            cv2.putText(display, f"Combined 40/40/20 | {model_label}", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(display, f"FPS: {fps:.1f} | 'm'=swap model, 's'=screenshot, 'q'=quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            cv2.imshow("Combined 40/40/20 Model - Webcam Test", display)

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
            elif key == ord('m'):
                use_best = not use_best
                current_path = BEST_MODEL_PATH if use_best else FINAL_MODEL_PATH
                label = "best" if use_best else "final"
                print(f"Switching to {label} model...")
                model = load_model(current_path)
                print(f"Loaded: {current_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
