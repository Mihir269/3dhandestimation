import cv2
import torch
import numpy as np
import torch.nn.functional as F
from model_handsegnet_cpm_fast import HandSegNetCPMFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "handsegnet_cpm_rhd.pth"

MIN_CONTOUR_AREA = 2000


# -----------------------
# Load Model
# -----------------------

model = HandSegNetCPMFast().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# -----------------------
# Start Webcam
# -----------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        _, _, output = model(img)

        output = F.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        mask = torch.sigmoid(output)
        mask = (mask > 0.5).float()

    mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > MIN_CONTOUR_AREA:

            x, y, bw, bh = cv2.boundingRect(largest)

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

    cv2.imshow("HandSegNet", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()