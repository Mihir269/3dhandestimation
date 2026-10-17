import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGES_DIR = "test_images"
MODEL_PATH = r"C:\Users\gunem\studying-2dto3d\handsegnetnew\trained_models\transfer_learning_20260309_014131\final_model.pth"
OUTPUT_DIR = "test_results_cpm_roboflow_final"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)


def get_bounding_box(mask):
    """
    Extract bounding box from binary mask

    Args:
        mask: Binary mask (numpy array)

    Returns:
        (x, y, w, h) bounding box or None if no hand detected
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 100:  # Minimum area threshold
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def preprocess_image(img_path):
    """
    Load and preprocess image for model

    Args:
        img_path: Path to image

    Returns:
        Preprocessed image tensor (1, 3, H, W) and original image
    """
    try:
        # Use numpy to read the file to handle Unicode paths better on Windows
        with open(img_path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return None, None

        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        return img_tensor, original_img
    except Exception as e:
        return None, None


def sanitize_filename(filename):
    """
    Sanitize filename to remove problematic Unicode characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename with ASCII-compatible characters only
    """
    # Replace problematic Unicode characters with ASCII equivalents
    sanitized = filename.encode('ascii', 'replace').decode('ascii')
    # Remove any remaining non-alphanumeric characters (except dots and dashes)
    sanitized = ''.join(c if c.isalnum() or c in '._-' else '_' for c in sanitized)
    return sanitized


def infer_test_images():
    """Main inference function for test images"""

    print("Loading model...")
    model = HandSegNetCPMFast().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Get list of test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        test_images.extend(Path(TEST_IMAGES_DIR).glob(ext))

    test_images = sorted(test_images)
    print(f"Found {len(test_images)} test images")

    if len(test_images) == 0:
        print(f"No images found in {TEST_IMAGES_DIR}")
        return

    # Statistics
    detection_stats = {
        "detected": 0,
        "not_detected": 0
    }

    print(f"Processing test images...")

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(test_images, desc="Inferencing")):
            img_tensor, original_img = preprocess_image(str(img_path))

            if img_tensor is None:
                safe_name = sanitize_filename(img_path.stem)
                print(f"Failed to load image: {safe_name}")
                continue

            # Move to device and infer
            img_tensor = img_tensor.to(DEVICE)

            # Forward pass
            _, _, output = model(img_tensor)

            # Get segmentation mask
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

            # Interpolate to original image size
            if pred_mask.ndim == 2:
                h, w = original_img.shape[:2]
                pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                pred_mask_resized = pred_mask

            # Create visualization
            vis_img = original_img.copy()

            # Draw segmentation mask overlay
            pred_mask_3ch = np.zeros_like(original_img)
            pred_mask_uint8 = (pred_mask_resized > 0.5).astype(np.uint8)
            pred_mask_3ch[pred_mask_uint8 == 1] = [0, 255, 0]  # Green

            # Blend mask with original image
            vis_img = cv2.addWeighted(vis_img, 0.7, pred_mask_3ch, 0.3, 0)

            # Extract and draw bounding box
            bbox = get_bounding_box(pred_mask_resized)

            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(vis_img, "Hand Detected", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                detection_stats["detected"] += 1
            else:
                cv2.putText(vis_img, "No Hand Detected", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                detection_stats["not_detected"] += 1

            # Sanitize filename to handle Unicode characters
            safe_stem = sanitize_filename(img_path.stem)

            # Save visualization
            output_filename = f"result_{idx:04d}_{safe_stem}.jpg"
            output_path = os.path.join(OUTPUT_DIR, "visualizations", output_filename)
            cv2.imwrite(output_path, vis_img)

            # Save predicted mask
            mask_filename = f"mask_{idx:04d}_{safe_stem}.png"
            mask_output_path = os.path.join(OUTPUT_DIR, "masks", mask_filename)
            pred_mask_uint8_colored = (pred_mask_resized > 0.5).astype(np.uint8) * 255
            cv2.imwrite(mask_output_path, pred_mask_uint8_colored)

    # Print statistics
    print("\n" + "="*60)
    print("TEST IMAGES INFERENCE RESULTS - HandSegNet CPM")
    print("="*60)

    print(f"\nTotal images processed: {len(test_images)}\n")

    print("HAND DETECTION STATISTICS:")
    print("-" * 60)
    total_processed = detection_stats["detected"] + detection_stats["not_detected"]
    if total_processed > 0:
        detection_rate = 100 * detection_stats["detected"] / total_processed
        print(f"Hands Detected: {detection_stats['detected']}/{total_processed} ({detection_rate:.2f}%)")
        print(f"No Hand Detected: {detection_stats['not_detected']}/{total_processed}")
    else:
        print("No images were processed")

    print("\nOUTPUT FILES:")
    print("-" * 60)
    print(f"Visualizations saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'visualizations'))}")
    print(f"Predicted masks saved to: {os.path.abspath(os.path.join(OUTPUT_DIR, 'masks'))}")

    # Save summary report
    summary_file = os.path.join(OUTPUT_DIR, "inference_summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TEST IMAGES INFERENCE RESULTS - HandSegNet CPM\n")
        f.write("="*60 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Input directory: {TEST_IMAGES_DIR}\n")
        f.write(f"Total images processed: {len(test_images)}\n\n")

        f.write("HAND DETECTION STATISTICS:\n")
        f.write("-" * 60 + "\n")
        if total_processed > 0:
            f.write(f"Hands Detected: {detection_stats['detected']}/{total_processed} ({detection_rate:.2f}%)\n")
            f.write(f"No Hand Detected: {detection_stats['not_detected']}/{total_processed}\n\n")
        else:
            f.write("No images were processed\n\n")

        f.write("PROCESSED IMAGES:\n")
        f.write("-" * 60 + "\n")
        for idx, img_path in enumerate(test_images):
            try:
                f.write(f"{idx+1}. {img_path.name}\n")
            except UnicodeEncodeError:
                # Fallback for filenames with Unicode characters
                safe_name = sanitize_filename(img_path.stem) + img_path.suffix
                f.write(f"{idx+1}. {safe_name}\n")

    print(f"Summary report saved to: {os.path.abspath(summary_file)}\n")
    print("="*60)
    print("Inference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    infer_test_images()
