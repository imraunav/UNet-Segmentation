import torch
from torch import nn
from torch.optim import AdamW
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import os

from model import UNet

from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_pred_as_imgs,
    get_loaders,
)

# Hyperparameters
LR = 3e-4  # Karpathy's recommendation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_EPOCH = 50
NUM_WORKERS = 2
IMG_HEIGHT = 161
IMG_WEIGHT = 161
PIN_MEMORY = True
TRAIN_IMG_DIR = "./data/train"
TRAIN_MASK_DIR = "./data/train_masks"
VAL_IMG_DIR = "./data/val"
VAL_MASK_DIR = "./data/val_masks"
CKPT_DIR = "./checkpoint/checkpoint.pt"

if os.path.exists("./checkpoint") == False:
    os.makedirs("./checkpoint")


def train_step(model, optimizer, dataloader, crit, scaler):
    model.train()
    loop = tqdm(dataloader)
    epoch_losses = []
    for datas, targets in loop:
        datas = datas.to(device)
        targets = targets.unsqueeze(1).to(device)  # add a channel dim

        # forward pass
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            preds = model(datas)
            loss = crit(preds, targets)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())
        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WEIGHT),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0, 0, 0],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WEIGHT),
            A.Normalize(
                mean=[0, 0, 0],
                std=[0.5, 0.5, 0.5],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    crit = nn.BCEWithLogitsLoss().to(device)
    optimizer = AdamW(model.parameters(), LR)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    start = 0
    train_loss = []
    val_loss = []
    if os.path.exists(CKPT_DIR):
        print("Checkpoint found...")
        start, train_loss, val_loss = load_checkpoint(CKPT_DIR, model, optimizer)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start, MAX_EPOCH):
        print(f"Epoch : {epoch+1} | ", end="")
        train_loss.append(train_step(model, optimizer, train_loader, crit, scaler))
        print(f"Train loss : {train_loss[-1]}")
        check_accuracy(val_loader, model, device)

        save_checkpoint(
            model.state_dict(), optimizer.state_dict(), train_loss, val_loss, epoch + 1
        )
        save_pred_as_imgs(val_loader, model, device=device)


if __name__ == "__main__":
    main()
