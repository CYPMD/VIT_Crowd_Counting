from pathlib import Path
import math
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF

import timm


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    "/content/drive/MyDrive/ShanghaiTech/part_B_final"
)

TRAIN_DIR = DATASET_ROOT / "train_data"
TEST_DIR = DATASET_ROOT / "test_data"

CROP_SIZE = 512
PATCH_SIZE = 32

# Explicit ImageNet-1k pretrained ViT-B/32
MODEL_NAME = "vit_base_patch32_224.augreg_in1k"

BATCH_SIZE = 4
NUM_WORKERS = 2

EPOCHS = 50

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Density values are tiny (~1e-3 or smaller).
# Scaling both prediction and target in the loss gives healthier
# numerical magnitudes without changing the optimum.
DENSITY_SCALE = 1000.0

VAL_FRACTION = 0.10

SEED = 42


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# Dataset
# ============================================================

class ShanghaiTechDensityDataset(Dataset):
    """
    Loads:
        images/IMG_X.jpg
        density_maps/IMG_X.npy

    Training:
        random 512x512 crop
        random horizontal flip

    Validation:
        center 512x512 crop

    Important:
        The exact same spatial operations are applied to the image
        and density map.
    """

    def __init__(
        self,
        split_dir,
        image_files=None,
        crop_size=512,
        train=True,
    ):
        super().__init__()

        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.density_dir = self.split_dir / "density_maps"

        self.crop_size = crop_size
        self.train = train

        if image_files is None:
            self.image_files = sorted(
                self.image_dir.glob("*.jpg")
            )
        else:
            self.image_files = list(image_files)

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        density_path = (
            self.density_dir /
            f"{image_path.stem}.npy"
        )

        if not density_path.exists():
            raise FileNotFoundError(
                f"Density map not found: {density_path}"
            )

        # ------------------------------------------------------
        # Load image
        # ------------------------------------------------------
        image = Image.open(image_path).convert("RGB")

        # [3, H, W], range [0, 1]
        image = TF.pil_to_tensor(image).float() / 255.0

        # ------------------------------------------------------
        # Load density map
        # ------------------------------------------------------
        density = np.load(density_path)

        if density.ndim != 2:
            raise ValueError(
                f"Expected 2D density map, got "
                f"{density.shape} for {density_path}"
            )

        density = torch.from_numpy(
            density.astype(np.float32)
        ).unsqueeze(0)

        # [1, H, W]

        # ------------------------------------------------------
        # Check spatial alignment
        # ------------------------------------------------------
        _, image_h, image_w = image.shape
        _, density_h, density_w = density.shape

        if (image_h, image_w) != (density_h, density_w):
            raise ValueError(
                f"Image/density size mismatch:\n"
                f"{image_path}: {(image_h, image_w)}\n"
                f"{density_path}: {(density_h, density_w)}"
            )

        # ------------------------------------------------------
        # Pad if image is smaller than crop
        # Normally not needed for ShanghaiTech Part B,
        # but makes the loader robust.
        # ------------------------------------------------------
        pad_h = max(0, self.crop_size - image_h)
        pad_w = max(0, self.crop_size - image_w)

        if pad_h > 0 or pad_w > 0:

            # pad format:
            # (left, right, top, bottom)
            image = F.pad(
                image,
                (0, pad_w, 0, pad_h),
                value=0.0,
            )

            density = F.pad(
                density,
                (0, pad_w, 0, pad_h),
                value=0.0,
            )

        _, h, w = image.shape

        # ------------------------------------------------------
        # Crop
        # ------------------------------------------------------
        if self.train:

            top = random.randint(
                0,
                h - self.crop_size
            )

            left = random.randint(
                0,
                w - self.crop_size
            )

        else:

            top = (h - self.crop_size) // 2
            left = (w - self.crop_size) // 2

        bottom = top + self.crop_size
        right = left + self.crop_size

        image = image[
            :,
            top:bottom,
            left:right,
        ]

        density = density[
            :,
            top:bottom,
            left:right,
        ]

        # ------------------------------------------------------
        # Horizontal flip
        # ------------------------------------------------------
        if self.train and random.random() < 0.5:

            image = torch.flip(
                image,
                dims=[2],
            )

            density = torch.flip(
                density,
                dims=[2],
            )

        # ------------------------------------------------------
        # ImageNet normalization
        # ------------------------------------------------------
        image = TF.normalize(
            image,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return image, density


# ============================================================
# Train/validation split
# ============================================================

def create_train_val_datasets(
    train_dir,
    val_fraction=0.1,
    seed=42,
):

    image_dir = Path(train_dir) / "images"

    image_files = sorted(
        image_dir.glob("*.jpg")
    )

    if len(image_files) == 0:
        raise RuntimeError(
            f"No training images found in {image_dir}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(image_files),
        generator=generator,
    ).tolist()

    num_val = max(
        1,
        int(len(image_files) * val_fraction),
    )

    val_indices = permutation[:num_val]
    train_indices = permutation[num_val:]

    train_files = [
        image_files[i]
        for i in train_indices
    ]

    val_files = [
        image_files[i]
        for i in val_indices
    ]

    train_dataset = ShanghaiTechDensityDataset(
        split_dir=train_dir,
        image_files=train_files,
        crop_size=CROP_SIZE,
        train=True,
    )

    val_dataset = ShanghaiTechDensityDataset(
        split_dir=train_dir,
        image_files=val_files,
        crop_size=CROP_SIZE,
        train=False,
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    return train_dataset, val_dataset


# ============================================================
# ViT density model
# ============================================================

class ViTDensityEstimator(nn.Module):
    """
    Architecture:

        512x512 RGB
             |
             v
        ViT patch embedding
        patch size = 32x32
             |
             v
        16x16 = 256 patch tokens
             |
             v
        Transformer blocks
             |
             v
        Linear(768 -> 1)
             |
             v
        16x16 coarse density map
             |
             v
        Bilinear interpolation
             |
             v
        512x512 density map
    """

    def __init__(
        self,
        model_name=MODEL_NAME,
        patch_size=32,
        preserve_mass=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.preserve_mass = preserve_mass

        # ------------------------------------------------------
        # Pretrained ImageNet ViT
        # ------------------------------------------------------
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,

            # Remove classifier
            num_classes=0,

            # Don't globally pool the tokens.
            global_pool="",

            # Allows 512x512 although pretrained model used 224x224.
            dynamic_img_size=True,
        )

        embedding_dim = self.backbone.num_features

        print(
            f"ViT embedding dimension: {embedding_dim}"
        )

        print(
            f"Prefix tokens: "
            f"{self.backbone.num_prefix_tokens}"
        )

        # ------------------------------------------------------
        # Density head
        #
        # Each spatial token -> one scalar
        # ------------------------------------------------------
        self.density_head = nn.Linear(
            embedding_dim,
            1,
        )

        # Small initialization for the newly-created regression head
        nn.init.normal_(
            self.density_head.weight,
            mean=0.0,
            std=0.001,
        )

        nn.init.zeros_(
            self.density_head.bias
        )

    def forward(self, x):

        batch_size, _, height, width = x.shape

        if (
            height % self.patch_size != 0
            or width % self.patch_size != 0
        ):
            raise ValueError(
                f"Input dimensions must be divisible by "
                f"{self.patch_size}. "
                f"Got {(height, width)}."
            )

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        # ------------------------------------------------------
        # ViT
        #
        # output:
        # [B, prefix_tokens + patch_tokens, embedding_dim]
        # ------------------------------------------------------
        tokens = self.backbone.forward_features(x)

        # ------------------------------------------------------
        # Remove CLS token / any other prefix tokens
        # ------------------------------------------------------
        num_prefix_tokens = (
            self.backbone.num_prefix_tokens
        )

        patch_tokens = tokens[
            :,
            num_prefix_tokens:,
            :,
        ]

        expected_tokens = grid_h * grid_w

        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} patch tokens, "
                f"but got {patch_tokens.shape[1]}"
            )

        # ------------------------------------------------------
        # One scalar per token
        #
        # [B, N, D] -> [B, N, 1]
        # ------------------------------------------------------
        coarse = self.density_head(
            patch_tokens
        )

        # ------------------------------------------------------
        # [B, N, 1]
        # ->
        # [B, 1, grid_h, grid_w]
        # ------------------------------------------------------
        coarse = coarse.transpose(1, 2)

        coarse = coarse.reshape(
            batch_size,
            1,
            grid_h,
            grid_w,
        )

        # ------------------------------------------------------
        # Bilinear upsample
        # ------------------------------------------------------
        density = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------
        # Density-mass correction
        #
        # Bilinear interpolation preserves approximately the
        # average pixel value, not the SUM.
        #
        # A 16x16 -> 512x512 upsampling increases the number
        # of spatial locations by 1024.
        #
        # Multiplying by:
        #
        #     16*16 / (512*512) = 1/1024
        #
        # lets coarse token values behave approximately like
        # "density mass per patch".
        # ------------------------------------------------------
        if self.preserve_mass:

            scale = (
                (grid_h * grid_w)
                / float(height * width)
            )

            density = density * scale

        return density


# ============================================================
# Loss
# ============================================================

def density_loss(
    prediction,
    target,
    scale=DENSITY_SCALE,
):
    """
    Scaling by 1000 only changes the numerical magnitude
    of the MSE. Predictions themselves remain in the original
    density-map units.
    """

    return F.mse_loss(
        prediction * scale,
        target * scale,
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
):

    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    amp_enabled = device.type == "cuda"

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ------------------------------------------------------
        # Forward
        # ------------------------------------------------------
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions = model(images)

            loss = density_loss(
                predictions,
                targets,
            )

        # ------------------------------------------------------
        # Backprop
        # ------------------------------------------------------
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        # ------------------------------------------------------
        # Count metric
        #
        # integral of density map = crowd count
        # ------------------------------------------------------
        with torch.no_grad():

            predicted_counts = predictions.sum(
                dim=(1, 2, 3)
            )

            target_counts = targets.sum(
                dim=(1, 2, 3)
            )

            mae = torch.abs(
                predicted_counts - target_counts
            ).sum()

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_mae += mae.item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    amp_enabled = device.type == "cuda"

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions = model(images)

            loss = density_loss(
                predictions,
                targets,
            )

        predicted_counts = predictions.sum(
            dim=(1, 2, 3)
        )

        target_counts = targets.sum(
            dim=(1, 2, 3)
        )

        errors = (
            predicted_counts - target_counts
        )

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_absolute_error += (
            errors.abs().sum().item()
        )

        total_squared_error += (
            (errors ** 2).sum().item()
        )

        total_samples += batch_size

    mae = (
        total_absolute_error
        / total_samples
    )

    rmse = math.sqrt(
        total_squared_error
        / total_samples
    )

    return {
        "loss": total_loss / total_samples,
        "mae": mae,
        "rmse": rmse,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ----------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------
    train_dataset, val_dataset = (
        create_train_val_datasets(
            TRAIN_DIR,
            val_fraction=VAL_FRACTION,
            seed=SEED,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    model = ViTDensityEstimator(
        model_name=MODEL_NAME,
        patch_size=PATCH_SIZE,
        preserve_mass=True,
    )

    model = model.to(device)

    # ----------------------------------------------------------
    # Separate LR:
    #
    # pretrained transformer = smaller LR
    # new density head       = larger LR
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.density_head.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    amp_enabled = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # ----------------------------------------------------------
    # Sanity check
    # ----------------------------------------------------------
    images, targets = next(
        iter(train_loader)
    )

    print("\nSanity check:")
    print("Images:", images.shape)
    print("Targets:", targets.shape)

    with torch.no_grad():

        output = model(
            images[:1].to(device)
        )

    print("Output:", output.shape)

    print(
        "Target count:",
        targets[0].sum().item(),
    )

    print(
        "Initial predicted count:",
        output[0].sum().item(),
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_MAE={train_metrics['mae']:.3f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_MAE={val_metrics['mae']:.3f} | "
            f"val_RMSE={val_metrics['rmse']:.3f}"
        )

        # ------------------------------------------------------
        # Save best model according to validation count MAE
        # ------------------------------------------------------
        if val_metrics["mae"] < best_mae:

            best_mae = val_metrics["mae"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "model_name": MODEL_NAME,
                "crop_size": CROP_SIZE,
                "patch_size": PATCH_SIZE,
            }

            torch.save(
                checkpoint,
                "best_vit_b32_density.pth",
            )

            print(
                f"Saved best model "
                f"(MAE={best_mae:.3f})"
            )


if __name__ == "__main__":
    main()