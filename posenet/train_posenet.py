import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.posenet import PoseNet
from datasets.posenet_dataset import PoseNetDataset


def main():
    # ---------------- CONFIG ----------------
    DATA_ROOT = "cropped_dataset"
    BATCH_SIZE = 48
    EPOCHS = 20
    LR = 1e-3
    NUM_WORKERS = 0  # Windows safe

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)

    # ---------------- DATA ----------------
    dataset = PoseNetDataset(DATA_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"✅ Training samples: {len(dataset)}")

    # ---------------- MODEL ----------------
    model = PoseNet().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, heatmaps in loader:
            imgs = imgs.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            preds = model(imgs)

            preds = F.interpolate(
                preds,
                size=heatmaps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss = criterion(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "posenet_freihand.pth")
    print("✅ PoseNet saved")


if __name__ == "__main__":
    main()