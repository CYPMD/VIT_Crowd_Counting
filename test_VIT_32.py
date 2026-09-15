from pathlib import Path
import math

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

TEST_DIR = DATASET_ROOT / "test_data"

CHECKPOINT_PATH = Path(
    "best_vit_b32_density.pth"
)

MODEL_NAME = "vit_base_patch32_224.augreg_in1k"

PATCH_SIZE = 32

# Full images are much larger than 512x512 crops.
# Batch size 1 is safest.
BATCH_SIZE = 1
NUM_WORKERS = 2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Dataset
# ============================================================

class ShanghaiTechTestDataset(Dataset):

    def __init__(self, split_dir):
        super().__init__()

        self.split_dir = Path(split_dir)

        self.image_dir = self.split_dir / "images"
        self.density_dir = self.split_dir / "density_maps"

        self.image_files = sorted(
            self.image_dir.glob("*.jpg")
        )

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No test images found in {self.image_dir}"
            )

        print(
            f"Number of test images: {len(self.image_files)}"
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
        # Image
        # ------------------------------------------------------
        image = Image.open(
            image_path
        ).convert("RGB")

        image = (
            TF.pil_to_tensor(image).float()
            / 255.0
        )

        # ------------------------------------------------------
        # Density map
        # ------------------------------------------------------
        density = np.load(
            density_path
        ).astype(np.float32)

        density = torch.from_numpy(
            density
        ).unsqueeze(0)

        # ------------------------------------------------------
        # Sanity check
        # ------------------------------------------------------
        if image.shape[-2:] != density.shape[-2:]:

            raise ValueError(
                f"Image and density map shapes differ:\n"
                f"Image: {image_path} {image.shape}\n"
                f"Density: {density_path} {density.shape}"
            )

        # ------------------------------------------------------
        # ImageNet normalization
        # ------------------------------------------------------
        image = TF.normalize(
            image,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return (
            image,
            density,
            image_path.name,
        )


# ============================================================
# ViT density estimator
# Must be the SAME architecture used during training
# ============================================================

class ViTDensityEstimator(nn.Module):

    def __init__(
        self,
        model_name=MODEL_NAME,
        patch_size=32,
        preserve_mass=True,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.preserve_mass = preserve_mass

        # pretrained=False because we will load
        # our trained checkpoint anyway.
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )

        embedding_dim = (
            self.backbone.num_features
        )

        self.density_head = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, x):

        batch_size, _, height, width = x.shape

        if (
            height % self.patch_size != 0
            or width % self.patch_size != 0
        ):
            raise ValueError(
                f"Image size {(height, width)} is not "
                f"divisible by patch size {self.patch_size}"
            )

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        # ------------------------------------------------------
        # ViT feature extraction
        # ------------------------------------------------------
        tokens = self.backbone.forward_features(
            x
        )

        # Remove CLS / prefix tokens
        num_prefix_tokens = (
            self.backbone.num_prefix_tokens
        )

        patch_tokens = tokens[
            :,
            num_prefix_tokens:,
            :,
        ]

        # ------------------------------------------------------
        # Density prediction for each patch
        # ------------------------------------------------------
        coarse = self.density_head(
            patch_tokens
        )

        # [B, N, 1] -> [B, 1, N]
        coarse = coarse.transpose(
            1,
            2,
        )

        # [B, 1, N] -> [B, 1, H/32, W/32]
        coarse = coarse.reshape(
            batch_size,
            1,
            grid_h,
            grid_w,
        )

        # ------------------------------------------------------
        # Upsample
        # ------------------------------------------------------
        density = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------
        # Preserve density integral
        # ------------------------------------------------------
        if self.preserve_mass:

            scale = (
                (grid_h * grid_w)
                / float(height * width)
            )

            density = density * scale

        return density


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    absolute_errors = []
    squared_errors = []

    predictions_all = []
    ground_truth_all = []

    amp_enabled = (
        device.type == "cuda"
    )

    print("\nEvaluating test set...\n")

    for i, (
        images,
        targets,
        filenames,
    ) in enumerate(loader, start=1):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # ------------------------------------------------------
        # Forward
        # ------------------------------------------------------
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions = model(
                images
            )

        # ------------------------------------------------------
        # Density integral = predicted number of people
        # ------------------------------------------------------
        pred_counts = predictions.sum(
            dim=(1, 2, 3)
        )

        gt_counts = targets.sum(
            dim=(1, 2, 3)
        )

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------
        errors = (
            pred_counts - gt_counts
        )

        abs_errors = (
            errors.abs()
        )

        sq_errors = (
            errors ** 2
        )

        absolute_errors.extend(
            abs_errors.cpu().numpy().tolist()
        )

        squared_errors.extend(
            sq_errors.cpu().numpy().tolist()
        )

        predictions_all.extend(
            pred_counts.cpu().numpy().tolist()
        )

        ground_truth_all.extend(
            gt_counts.cpu().numpy().tolist()
        )

        # ------------------------------------------------------
        # Print individual image result
        # ------------------------------------------------------
        for j in range(
            len(filenames)
        ):

            print(
                f"[{i:3d}/{len(loader):3d}] "
                f"{filenames[j]:15s} "
                f"GT={gt_counts[j].item():8.2f} "
                f"Pred={pred_counts[j].item():8.2f} "
                f"Error={errors[j].item():+8.2f}"
            )

    # ========================================================
    # Final metrics
    # ========================================================

    mae = np.mean(
        absolute_errors
    )

    mse_count = np.mean(
        squared_errors
    )

    rmse = math.sqrt(
        mse_count
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "predictions": predictions_all,
        "ground_truth": ground_truth_all,
        "absolute_errors": absolute_errors,
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
        f"Using device: {device}"
    )

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    test_dataset = (
        ShanghaiTechTestDataset(
            TEST_DIR
        )
    )

    test_loader = DataLoader(
        test_dataset,
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

    # ----------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------
    print(
        f"\nLoading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    if "epoch" in checkpoint:
        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_mae" in checkpoint:
        print(
            f"Checkpoint validation MAE: "
            f"{checkpoint['val_mae']:.3f}"
        )

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    results = evaluate(
        model,
        test_loader,
        device,
    )

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ShanghaiTech Part B Test Results")
    print("=" * 60)

    print(
        f"MAE  : {results['MAE']:.3f}"
    )

    print(
        f"RMSE : {results['RMSE']:.3f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()