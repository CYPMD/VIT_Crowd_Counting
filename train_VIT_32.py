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

# Training crop size
CROP_SIZE = 512

# ViT-B/32
PATCH_SIZE = 32

MODEL_NAME = "vit_base_patch32_224.augreg_in1k"

# Training batch size
BATCH_SIZE = 4

# Full validation images can have different dimensions
VAL_BATCH_SIZE = 1

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
# Training dataset
# ============================================================

class ShanghaiTechTrainDataset(Dataset):
    """
    Training dataset.

    Training:
        - load full image
        - random 512x512 crop
        - random horizontal flip
        - ImageNet normalization

    The same spatial operations are applied to image
    and density map.
    """

    def __init__(
        self,
        split_dir,
        image_files,
        crop_size=512,
    ):
        super().__init__()

        self.split_dir = Path(split_dir)

        self.image_dir = self.split_dir / "images"
        self.density_dir = self.split_dir / "density_maps"

        self.image_files = list(image_files)

        self.crop_size = crop_size

        if len(self.image_files) == 0:
            raise RuntimeError(
                "Training dataset contains no images."
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        density_path = (
            self.density_dir
            / f"{image_path.stem}.npy"
        )

        if not density_path.exists():
            raise FileNotFoundError(
                f"Density map not found: {density_path}"
            )

        # --------------------------------------------------------
        # Load image
        # --------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = (
            TF.pil_to_tensor(image).float()
            / 255.0
        )

        # --------------------------------------------------------
        # Load density map
        # --------------------------------------------------------

        density = np.load(
            density_path
        ).astype(np.float32)

        if density.ndim != 2:
            raise ValueError(
                f"Expected 2D density map, "
                f"got {density.shape} "
                f"for {density_path}"
            )

        density = torch.from_numpy(
            density
        ).unsqueeze(0)

        # --------------------------------------------------------
        # Check alignment
        # --------------------------------------------------------

        _, image_h, image_w = image.shape
        _, density_h, density_w = density.shape

        if (
            image_h != density_h
            or image_w != density_w
        ):
            raise ValueError(
                f"Image/density size mismatch:\n"
                f"{image_path}: "
                f"{(image_h, image_w)}\n"
                f"{density_path}: "
                f"{(density_h, density_w)}"
            )

        # --------------------------------------------------------
        # Pad if image smaller than crop
        # --------------------------------------------------------

        pad_h = max(
            0,
            self.crop_size - image_h,
        )

        pad_w = max(
            0,
            self.crop_size - image_w,
        )

        if pad_h > 0 or pad_w > 0:

            image = F.pad(
                image,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
                value=0.0,
            )

            density = F.pad(
                density,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
                value=0.0,
            )

        # --------------------------------------------------------
        # Random crop
        # --------------------------------------------------------

        _, h, w = image.shape

        top = random.randint(
            0,
            h - self.crop_size,
        )

        left = random.randint(
            0,
            w - self.crop_size,
        )

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

        # --------------------------------------------------------
        # Random horizontal flip
        # --------------------------------------------------------

        if random.random() < 0.5:

            image = torch.flip(
                image,
                dims=[2],
            )

            density = torch.flip(
                density,
                dims=[2],
            )

        # --------------------------------------------------------
        # ImageNet normalization
        # --------------------------------------------------------

        image = TF.normalize(
            image,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return image, density


# ============================================================
# Full-image validation dataset
# ============================================================

class ShanghaiTechValidationDataset(Dataset):
    """
    Validation uses the COMPLETE image.

    No crop.
    No augmentation.

    Images are padded only when needed so height and width are
    divisible by PATCH_SIZE=32.

    The original dimensions are returned so padded pixels can
    be removed before calculating validation loss/counts.
    """

    def __init__(
        self,
        split_dir,
        image_files,
        patch_size=32,
    ):
        super().__init__()

        self.split_dir = Path(split_dir)

        self.image_dir = self.split_dir / "images"
        self.density_dir = self.split_dir / "density_maps"

        self.image_files = list(image_files)

        self.patch_size = patch_size

        if len(self.image_files) == 0:
            raise RuntimeError(
                "Validation dataset contains no images."
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        density_path = (
            self.density_dir
            / f"{image_path.stem}.npy"
        )

        if not density_path.exists():
            raise FileNotFoundError(
                f"Density map not found: {density_path}"
            )

        # --------------------------------------------------------
        # Load FULL image
        # --------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = (
            TF.pil_to_tensor(image).float()
            / 255.0
        )

        # --------------------------------------------------------
        # Load FULL density map
        # --------------------------------------------------------

        density = np.load(
            density_path
        ).astype(np.float32)

        if density.ndim != 2:
            raise ValueError(
                f"Expected 2D density map, "
                f"got {density.shape} "
                f"for {density_path}"
            )

        density = torch.from_numpy(
            density
        ).unsqueeze(0)

        # --------------------------------------------------------
        # Check alignment
        # --------------------------------------------------------

        _, image_h, image_w = image.shape
        _, density_h, density_w = density.shape

        if (
            image_h != density_h
            or image_w != density_w
        ):
            raise ValueError(
                f"Image/density size mismatch:\n"
                f"{image_path}: "
                f"{(image_h, image_w)}\n"
                f"{density_path}: "
                f"{(density_h, density_w)}"
            )

        original_h = image_h
        original_w = image_w

        # --------------------------------------------------------
        # ImageNet normalization
        # --------------------------------------------------------

        image = TF.normalize(
            image,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        # --------------------------------------------------------
        # Pad to nearest multiple of 32
        # --------------------------------------------------------

        padded_h = (
            math.ceil(
                original_h / self.patch_size
            )
            * self.patch_size
        )

        padded_w = (
            math.ceil(
                original_w / self.patch_size
            )
            * self.patch_size
        )

        pad_h = padded_h - original_h
        pad_w = padded_w - original_w

        if pad_h > 0 or pad_w > 0:

            image = F.pad(
                image,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
                value=0.0,
            )

            density = F.pad(
                density,
                (
                    0,
                    pad_w,
                    0,
                    pad_h,
                ),
                value=0.0,
            )

        return (
            image,
            density,
            original_h,
            original_w,
            image_path.name,
        )


# ============================================================
# Train / validation split
# ============================================================

def create_train_val_datasets(
    train_dir,
    val_fraction=0.1,
    seed=42,
):

    image_dir = (
        Path(train_dir)
        / "images"
    )

    image_files = sorted(
        image_dir.glob("*.jpg")
    )

    if len(image_files) == 0:
        raise RuntimeError(
            f"No training images found in "
            f"{image_dir}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(image_files),
        generator=generator,
    ).tolist()

    num_val = max(
        1,
        int(
            len(image_files)
            * val_fraction
        ),
    )

    val_indices = (
        permutation[:num_val]
    )

    train_indices = (
        permutation[num_val:]
    )

    train_files = [
        image_files[i]
        for i in train_indices
    ]

    val_files = [
        image_files[i]
        for i in val_indices
    ]

    # --------------------------------------------------------
    # Training:
    # random 512x512 crops
    # --------------------------------------------------------

    train_dataset = (
        ShanghaiTechTrainDataset(
            split_dir=train_dir,
            image_files=train_files,
            crop_size=CROP_SIZE,
        )
    )

    # --------------------------------------------------------
    # Validation:
    # full images
    # --------------------------------------------------------

    val_dataset = (
        ShanghaiTechValidationDataset(
            split_dir=train_dir,
            image_files=val_files,
            patch_size=PATCH_SIZE,
        )
    )

    print(
        f"Training images:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        "Training mode:     "
        "random 512x512 crops"
    )

    print(
        "Validation mode:   "
        "full images"
    )

    return (
        train_dataset,
        val_dataset,
    )


# ============================================================
# ViT density estimator
# ============================================================

class ViTDensityEstimator(nn.Module):
    """
    ViT-B/32 density estimator.

    For 512x512 training crops:

        512x512 RGB
            |
            v
        ViT patch embedding
        patch size = 32
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

        # --------------------------------------------------------
        # ImageNet pretrained ViT-B/32
        # --------------------------------------------------------

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )

        embedding_dim = (
            self.backbone.num_features
        )

        print(
            f"ViT embedding dimension: "
            f"{embedding_dim}"
        )

        print(
            f"Prefix tokens: "
            f"{self.backbone.num_prefix_tokens}"
        )

        # --------------------------------------------------------
        # Density head
        # --------------------------------------------------------

        self.density_head = nn.Linear(
            embedding_dim,
            1,
        )

        nn.init.normal_(
            self.density_head.weight,
            mean=0.0,
            std=0.001,
        )

        nn.init.zeros_(
            self.density_head.bias
        )

    def forward(self, x):

        (
            batch_size,
            _,
            height,
            width,
        ) = x.shape

        # --------------------------------------------------------
        # Dimensions must be divisible by 32
        # --------------------------------------------------------

        if (
            height % self.patch_size != 0
            or
            width % self.patch_size != 0
        ):
            raise ValueError(
                f"Input dimensions must be "
                f"divisible by {self.patch_size}. "
                f"Got {(height, width)}."
            )

        grid_h = (
            height
            // self.patch_size
        )

        grid_w = (
            width
            // self.patch_size
        )

        # --------------------------------------------------------
        # ViT feature extraction
        # --------------------------------------------------------

        tokens = (
            self.backbone.forward_features(
                x
            )
        )

        # --------------------------------------------------------
        # Remove CLS / prefix tokens
        # --------------------------------------------------------

        num_prefix_tokens = (
            self.backbone.num_prefix_tokens
        )

        patch_tokens = tokens[
            :,
            num_prefix_tokens:,
            :,
        ]

        expected_tokens = (
            grid_h * grid_w
        )

        if (
            patch_tokens.shape[1]
            != expected_tokens
        ):
            raise RuntimeError(
                f"Expected "
                f"{expected_tokens} patch tokens, "
                f"but got "
                f"{patch_tokens.shape[1]}"
            )

        # --------------------------------------------------------
        # One scalar per patch
        # --------------------------------------------------------

        coarse = self.density_head(
            patch_tokens
        )

        # [B, N, 1] -> [B, 1, N]
        coarse = coarse.transpose(
            1,
            2,
        )

        # [B, 1, N]
        # ->
        # [B, 1, grid_h, grid_w]
        coarse = coarse.reshape(
            batch_size,
            1,
            grid_h,
            grid_w,
        )

        # --------------------------------------------------------
        # Upsample to input resolution
        # --------------------------------------------------------

        density = F.interpolate(
            coarse,
            size=(
                height,
                width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        # --------------------------------------------------------
        # Preserve approximate density integral
        # --------------------------------------------------------
        #
        # For a 512x512 image:
        #
        #     grid = 16x16
        #
        # scale:
        #
        #     16*16 / (512*512)
        #     = 1/1024
        #
        # --------------------------------------------------------

        if self.preserve_mass:

            scale = (
                (grid_h * grid_w)
                / float(
                    height * width
                )
            )

            density = (
                density * scale
            )

        return density


# ============================================================
# Density loss
# ============================================================

def density_loss(
    prediction,
    target,
    scale=DENSITY_SCALE,
):

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

    amp_enabled = (
        device.type == "cuda"
    )

    for (
        images,
        targets,
    ) in loader:

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

        # --------------------------------------------------------
        # Forward
        # --------------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions = model(
                images
            )

            loss = density_loss(
                predictions,
                targets,
            )

        # --------------------------------------------------------
        # Backpropagation
        # --------------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        # --------------------------------------------------------
        # Training crop MAE
        # --------------------------------------------------------

        with torch.no_grad():

            predicted_counts = (
                predictions.sum(
                    dim=(1, 2, 3)
                )
            )

            target_counts = (
                targets.sum(
                    dim=(1, 2, 3)
                )
            )

            mae = torch.abs(
                predicted_counts
                - target_counts
            ).sum()

        batch_size = (
            images.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_mae += (
            mae.item()
        )

        total_samples += (
            batch_size
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "mae": (
            total_mae
            / total_samples
        ),
    }


# ============================================================
# Full-image validation
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

    amp_enabled = (
        device.type == "cuda"
    )

    print(
        "\nValidating on full images...\n"
    )

    for (
        images,
        targets,
        original_h,
        original_w,
        filenames,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        if images.shape[0] != 1:
            raise RuntimeError(
                "Full-image validation requires "
                "VAL_BATCH_SIZE = 1."
            )

        original_h = int(
            original_h[0].item()
        )

        original_w = int(
            original_w[0].item()
        )

        # --------------------------------------------------------
        # Forward on complete padded image
        # --------------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions = model(
                images
            )

        # --------------------------------------------------------
        # Remove padding
        # --------------------------------------------------------

        predictions = predictions[
            :,
            :,
            :original_h,
            :original_w,
        ]

        targets = targets[
            :,
            :,
            :original_h,
            :original_w,
        ]

        # --------------------------------------------------------
        # Full-image density loss
        # --------------------------------------------------------

        loss = density_loss(
            predictions,
            targets,
        )

        # --------------------------------------------------------
        # FULL IMAGE crowd count
        # --------------------------------------------------------

        predicted_counts = (
            predictions.sum(
                dim=(1, 2, 3)
            )
        )

        target_counts = (
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        errors = (
            predicted_counts
            - target_counts
        )

        absolute_errors = (
            errors.abs()
        )

        squared_errors = (
            errors ** 2
        )

        batch_size = (
            predictions.shape[0]
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_absolute_error += (
            absolute_errors
            .sum()
            .item()
        )

        total_squared_error += (
            squared_errors
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

        # --------------------------------------------------------
        # Individual validation image
        # --------------------------------------------------------

        print(
            f"{filenames[0]:15s} "
            f"GT={target_counts[0].item():8.2f} "
            f"Pred={predicted_counts[0].item():8.2f} "
            f"Error={errors[0].item():+8.2f}"
        )

    # ------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------

    mae = (
        total_absolute_error
        / total_samples
    )

    rmse = math.sqrt(
        total_squared_error
        / total_samples
    )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
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

    print(
        f"Device: {device}"
    )

    # ----------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------

    (
        train_dataset,
        val_dataset,
    ) = create_train_val_datasets(
        TRAIN_DIR,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    # ----------------------------------------------------------
    # Training loader
    # ----------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ----------------------------------------------------------
    # Validation loader
    #
    # Full images have different dimensions,
    # therefore batch size must be 1.
    # ----------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
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

    model = model.to(
        device
    )

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.backbone.parameters(),

                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    model.density_head.parameters(),

                "lr":
                    HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # ----------------------------------------------------------
    # Scheduler
    # ----------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    # ----------------------------------------------------------
    # Mixed precision
    # ----------------------------------------------------------

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # ----------------------------------------------------------
    # Training sanity check
    # ----------------------------------------------------------

    images, targets = next(
        iter(train_loader)
    )

    print(
        "\nTraining sanity check:"
    )

    print(
        "Images:",
        images.shape,
    )

    print(
        "Targets:",
        targets.shape,
    )

    with torch.no_grad():

        output = model(
            images[:1].to(device)
        )

    print(
        "Output:",
        output.shape,
    )

    print(
        "Target crop count:",
        targets[0].sum().item(),
    )

    print(
        "Initial predicted crop count:",
        output[0].sum().item(),
    )

    # ----------------------------------------------------------
    # Validation sanity check
    # ----------------------------------------------------------

    (
        val_image,
        val_target,
        val_h,
        val_w,
        val_filename,
    ) = next(
        iter(val_loader)
    )

    print(
        "\nFull-image validation sanity check:"
    )

    print(
        "Filename:",
        val_filename[0],
    )

    print(
        "Padded image shape:",
        val_image.shape,
    )

    print(
        "Padded target shape:",
        val_target.shape,
    )

    original_h = int(
        val_h[0].item()
    )

    original_w = int(
        val_w[0].item()
    )

    print(
        "Original image size:",
        (
            original_h,
            original_w,
        ),
    )

    print(
        "Full-image GT count:",
        val_target[
            :,
            :,
            :original_h,
            :original_w,
        ].sum().item(),
    )

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------

    best_mae = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{EPOCHS:03d}"
        )

        print(
            "=" * 70
        )

        # ------------------------------------------------------
        # Train
        # ------------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
        )

        # ------------------------------------------------------
        # Full-image validation
        # ------------------------------------------------------

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        # ------------------------------------------------------
        # Scheduler
        # ------------------------------------------------------

        scheduler.step()

        # ------------------------------------------------------
        # Epoch summary
        # ------------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Epoch "
            f"{epoch:03d}/"
            f"{EPOCHS:03d} | "
            f"train_loss="
            f"{train_metrics['loss']:.6f} | "
            f"train_crop_MAE="
            f"{train_metrics['mae']:.3f} | "
            f"val_loss="
            f"{val_metrics['loss']:.6f} | "
            f"val_full_MAE="
            f"{val_metrics['mae']:.3f} | "
            f"val_full_RMSE="
            f"{val_metrics['rmse']:.3f}"
        )

        print(
            "-" * 70
        )

        # ------------------------------------------------------
        # Save best model according to FULL-IMAGE validation MAE
        # ------------------------------------------------------

        if (
            val_metrics["mae"]
            < best_mae
        ):

            best_mae = (
                val_metrics["mae"]
            )

            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_mae":
                    val_metrics["mae"],

                "val_rmse":
                    val_metrics["rmse"],

                "model_name":
                    MODEL_NAME,

                "crop_size":
                    CROP_SIZE,

                "patch_size":
                    PATCH_SIZE,

                "validation_mode":
                    "full_image",
            }

            torch.save(
                checkpoint,
                "best_vit_b32_density.pth",
            )

            print(
                "\nSaved best model "
                f"(full-image validation "
                f"MAE={best_mae:.3f})"
            )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
