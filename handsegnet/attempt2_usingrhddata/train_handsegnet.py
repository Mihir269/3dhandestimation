import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from handsegnet.attempt2_usingrhddata.dataset_rhd import RHDSegmentationDataset
from handsegnet.attempt3_usingcpmandrhddata.model_handsegnet_cpm_fast import HandSegNetCPMFast

torch.backends.cudnn.benchmark = True
DATA_ROOT = "data/RHD_published_v2/training"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("\n==============================")
    print("Initializing Training")
    print("==============================")

    print(f"Device being used: {DEVICE}")

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory:",
              torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    print("\nLoading dataset...")

    dataset = RHDSegmentationDataset(DATA_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    print("Dataset loaded successfully")
    print(f"Total training samples: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("==============================\n")

    print("Initializing model...")

    model = HandSegNetCPMFast().to(DEVICE)

    print("Model loaded")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Optimizer: Adam")
    print("Learning rate:", LR)
    print("Loss function: BCEWithLogitsLoss")
    print("==============================\n")

    for epoch in range(EPOCHS):

        model.train()

        epoch_start = time.time()

        total_loss = 0

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        for batch_idx, (imgs, masks) in enumerate(loader):

            batch_start = time.time()

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            s1, s2, s3 = model(imgs)

            s1 = torch.nn.functional.interpolate(
                s1, size=masks.shape[-2:])
            s2 = torch.nn.functional.interpolate(
                s2, size=masks.shape[-2:])
            s3 = torch.nn.functional.interpolate(
                s3, size=masks.shape[-2:])

            loss1 = criterion(s1, masks)
            loss2 = criterion(s2, masks)
            loss3 = criterion(s3, masks)

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_time = time.time() - batch_start

            # print every 50 batches
            if batch_idx % 50 == 0:

                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.5f} "
                    f"Batch time: {batch_time:.3f}s"
                )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1} completed "
            f"| Avg Loss: {total_loss/len(loader):.5f} "
            f"| Time: {epoch_time:.2f}s"
        )

    print("\nTraining completed")

    torch.save(model.state_dict(), "handsegnet_cpm_rhd.pth")

    print("Model saved as handsegnet_cpm_rhd.pth")


if __name__ == "__main__":
    main()