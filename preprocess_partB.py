"""
Generate density maps for ShanghaiTech Crowd Counting Dataset - Part B.

Expected directory structure:

ShanghaiTech/
└── part_B_final/
    ├── train_data/
    │   ├── images/
    │   │   ├── IMG_1.jpg
    │   │   ├── IMG_2.jpg
    │   │   └── ...
    │   └── ground_truth/
    │       ├── GT_IMG_1.mat
    │       ├── GT_IMG_2.mat
    │       └── ...
    │
    └── test_data/
        ├── images/
        └── ground_truth/

Output:

ShanghaiTech/
└── part_B_final/
    ├── train_data/
    │   └── density_maps/
    │       ├── IMG_1.npy
    │       └── ...
    └── test_data/
        └── density_maps/
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


SIGMA = 15.0


def load_shanghaitech_points(mat_path):
    """
    Load head annotations from ShanghaiTech .mat ground-truth file.

    Returns:
        points: numpy array with shape (N, 2)
                points[:, 0] = x coordinate
                points[:, 1] = y coordinate
    """
    mat = loadmat(mat_path)

    # ShanghaiTech annotation structure:
    # image_info[0, 0][0, 0][0]
    points = mat["image_info"][0, 0][0, 0][0]

    return points.astype(np.float32)


def gaussian_kernel_2d(sigma):
    """
    Create a 2D Gaussian kernel.

    Radius = 3 * sigma, which covers >99% of Gaussian mass.
    """
    radius = int(3 * sigma)

    x = np.arange(-radius, radius + 1, dtype=np.float32)
    y = np.arange(-radius, radius + 1, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2.0 * sigma**2)
    )

    # Normalize full kernel.
    kernel /= kernel.sum()

    return kernel


def generate_density_map(height, width, points, sigma=15.0):
    """
    Generate a crowd density map using fixed Gaussian sigma.

    Each annotation contributes exactly 1 to the density-map integral,
    including annotations close to image boundaries.

    Args:
        height: image height
        width: image width
        points: Nx2 array containing (x, y) annotations
        sigma: fixed Gaussian sigma in pixels

    Returns:
        density_map: float32 array with shape (height, width)
    """
    density_map = np.zeros((height, width), dtype=np.float32)

    if len(points) == 0:
        return density_map

    kernel = gaussian_kernel_2d(sigma)

    radius = kernel.shape[0] // 2

    for point in points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        # Skip invalid annotations.
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        # Region in image.
        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)

        # Corresponding region in Gaussian kernel.
        kx1 = x1 - (x - radius)
        kx2 = kernel.shape[1] - ((x + radius + 1) - x2)

        ky1 = y1 - (y - radius)
        ky2 = kernel.shape[0] - ((y + radius + 1) - y2)

        cropped_kernel = kernel[ky1:ky2, kx1:kx2].copy()

        # Re-normalize because Gaussian may be cropped at image boundary.
        kernel_sum = cropped_kernel.sum()

        if kernel_sum > 0:
            cropped_kernel /= kernel_sum

        density_map[y1:y2, x1:x2] += cropped_kernel

    return density_map


def process_split(split_dir, sigma=15.0):
    """
    Process either train_data or test_data.
    """
    split_dir = Path(split_dir)

    image_dir = split_dir / "images"
    gt_dir = split_dir / "ground_truth"
    output_dir = split_dir / "density_maps"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))

    print(f"\nProcessing: {split_dir}")
    print(f"Number of images: {len(image_files)}")
    print(f"Sigma: {sigma}")

    for i, image_path in enumerate(image_files, start=1):

        # Example:
        # IMG_1.jpg -> GT_IMG_1.mat
        gt_path = gt_dir / f"GT_{image_path.stem}.mat"

        if not gt_path.exists():
            print(f"[WARNING] Ground truth missing: {gt_path}")
            continue

        # ---------------------------------------------------------
        # Get image dimensions
        # ---------------------------------------------------------
        with Image.open(image_path) as img:
            width, height = img.size

        # ---------------------------------------------------------
        # Load annotations
        # ---------------------------------------------------------
        points = load_shanghaitech_points(gt_path)

        # ---------------------------------------------------------
        # Generate density map
        # ---------------------------------------------------------
        density_map = generate_density_map(
            height=height,
            width=width,
            points=points,
            sigma=sigma,
        )

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------
        output_path = output_dir / f"{image_path.stem}.npy"

        np.save(output_path, density_map.astype(np.float32))

        # Useful sanity check
        gt_count = len(points)
        density_count = density_map.sum()

        print(
            f"[{i:4d}/{len(image_files):4d}] "
            f"{image_path.name:15s} "
            f"shape={density_map.shape} "
            f"GT={gt_count:4d} "
            f"density_sum={density_count:.3f}"
        )


def main():

    # CHANGE THIS PATH
    dataset_root = Path(
        "/content/drive/MyDrive/ShanghaiTech/part_B_final"
    )

    sigma = 15.0

    process_split(
        dataset_root / "train_data",
        sigma=sigma,
    )

    process_split(
        dataset_root / "test_data",
        sigma=sigma,
    )


if __name__ == "__main__":
    main()