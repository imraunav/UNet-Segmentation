import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os


def save_checkpoint(
    model_state,
    optimizer_state,
    train_loss,
    val_loss,
    epoch=100_000_000,
    filename="checkpoint.pt",
):
    states = {
        "model": model_state,
        "optimizer": optimizer_state,
        "train loss": train_loss,
        "val loss": val_loss,
        "epoch": epoch,
    }
    torch.save(states, filename)
    print("! Checkpoint saved.")


def load_checkpoint(filename, model, optimizer):
    print("Loading checkpoint...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Checkpoint loaded.")
    return checkpoint["epoch"], checkpoint["train loss"], checkpoint["val loss"]


def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = CarvanaDataset(train_dir, train_mask_dir, train_transform)
    val_ds = CarvanaDataset(val_dir, val_mask_dir, train_transform)

    train_loader = DataLoader(
        train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size, False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader


@torch.no_grad()
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x).sigmoid()
        preds = (preds > 0.5).float()
        num_correct += (preds == y).sum()
        num_pixels += torch.numel(preds)  # number of elements
        dice_score += (2 * (preds * y)).sum() / ((preds + y).sum() + 1e-8)

    print(
        f"Accuracy : {num_correct/num_pixels * 100:.2f}%, Dice : {dice_score/len(loader)}"
    )


@torch.no_grad()
def save_pred_as_imgs(loader, model, folder="saved_imgs", device="cuda"):
    model.eval()
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        preds = model(x).sigmoid()
        preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, os.path.join(folder, f"pred_{i}.png"))
        torchvision.utils.save_image(
            y.unsqueeze(1), os.path.join(folder, f"true_{i}.png")
        )
