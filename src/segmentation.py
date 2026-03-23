# utils/segmentation.py — Image Segmentation Pipeline (Watershed + Canny)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ══════════════════════════════════════════════════════════════════════════════
# Watershed Segmentation
# ══════════════════════════════════════════════════════════════════════════════

def apply_watershed(image: np.ndarray) -> dict:
    """
    Segment mammogram using the Watershed Algorithm.

    Steps:
        1. Threshold → binary mask
        2. Distance transform
        3. Local maxima → seeds/markers
        4. Watershed fill from markers

    Args:
        image: Grayscale uint8 image (enhanced preferred)

    Returns:
        dict with: binary, distance, markers, segmented, overlay
    """
    results = {}

    # 1. Otsu threshold → binary
    _, binary = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["binary"] = binary

    # 2. Distance transform
    dist_transform = ndi.distance_transform_edt(binary)
    results["distance"] = dist_transform

    # 3. Find local maxima as seeds
    coords = peak_local_max(dist_transform,
                            min_distance=WATERSHED_MARKERS * 10,
                            labels=binary)
    mask = np.zeros(dist_transform.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    results["markers"] = markers

    # 4. Apply watershed
    labels = watershed(-dist_transform, markers, mask=binary)
    results["labels"] = labels

    # 5. Colored overlay
    segmented = np.zeros((*image.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    np.random.seed(RANDOM_SEED)
    colors = {lbl: np.random.randint(80, 255, 3).tolist()
              for lbl in unique_labels if lbl != 0}
    for lbl, color in colors.items():
        segmented[labels == lbl] = color
    results["segmented"] = segmented

    # 6. Draw contours on original
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        region_mask = np.uint8(labels == lbl) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 100), 1)
    results["overlay"] = overlay

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Canny Edge Detection
# ══════════════════════════════════════════════════════════════════════════════

def apply_canny(image: np.ndarray,
                threshold1: int = CANNY_THRESHOLD1,
                threshold2: int = CANNY_THRESHOLD2) -> dict:
    """
    Detect tumor edges using multi-stage Canny Edge Detection.

    Steps:
        1. Gaussian blur (noise suppression)
        2. Canny edge detection
        3. Dilation of edges
        4. Overlay on original

    Args:
        image     : Grayscale uint8 image
        threshold1: Lower hysteresis threshold
        threshold2: Upper hysteresis threshold

    Returns:
        dict with: smoothed, edges, dilated, overlay
    """
    results = {}

    # 1. Gaussian blur
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    results["smoothed"] = smoothed

    # 2. Canny edges
    edges = cv2.Canny(smoothed, threshold1, threshold2)
    results["edges"] = edges

    # 3. Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    results["dilated"] = dilated

    # 4. Color overlay
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[dilated > 0] = [0, 200, 255]   # Cyan edge highlight
    results["overlay"] = overlay

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Prepare Segmented Images for CNN Input
# ══════════════════════════════════════════════════════════════════════════════

def prepare_watershed_for_cnn(watershed_results: dict,
                               target_size: tuple = MAMMOGRAM_SIZE) -> np.ndarray:
    """
    Convert watershed segmented output to CNN-ready grayscale tensor.

    Args:
        watershed_results: Output dict from apply_watershed()
        target_size      : (H, W) resize target

    Returns:
        float32 array of shape (H, W, 1) in [0, 1]
    """
    seg = watershed_results["segmented"]
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    return (gray.astype(np.float32) / 255.0)[..., np.newaxis]


def prepare_canny_for_cnn(canny_results: dict,
                           target_size: tuple = MAMMOGRAM_SIZE) -> np.ndarray:
    """
    Convert Canny edge output to CNN-ready tensor.

    Args:
        canny_results: Output dict from apply_canny()
        target_size  : (H, W) resize target

    Returns:
        float32 array of shape (H, W, 1) in [0, 1]
    """
    edges = canny_results["dilated"]
    resized = cv2.resize(edges, target_size)
    return (resized.astype(np.float32) / 255.0)[..., np.newaxis]


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_watershed_results(results: dict, save_path: str = None):
    """Plot watershed segmentation stages."""
    stages = [
        ("binary",    "Binary (Otsu)",        "gray"),
        ("distance",  "Distance Transform",   "inferno"),
        ("segmented", "Watershed Segments",   None),
        ("overlay",   "Contour Overlay",      None),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(18, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax, (key, title, cmap) in zip(axes, stages):
        img = results[key]
        if img.dtype == np.float64:
            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap or "gray")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle("Watershed Segmentation Pipeline", color="white",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_canny_results(results: dict, save_path: str = None):
    """Plot Canny edge detection stages."""
    stages = [
        ("smoothed", "Gaussian Blurred", "gray"),
        ("edges",    "Canny Edges",      "gray"),
        ("dilated",  "Dilated Edges",    "gray"),
        ("overlay",  "Edge Overlay",     None),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(18, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax, (key, title, cmap) in zip(axes, stages):
        img = results[key]
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle("Canny Edge Detection Pipeline", color="white",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_segmentation_comparison(original: np.ndarray,
                                  watershed_results: dict,
                                  canny_results: dict,
                                  save_path: str = None):
    """
    Side-by-side comparison: Original | Watershed | Canny
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    items = [
        (original,                     "gray", "Original (CLAHE)"),
        (watershed_results["overlay"], None,   "Watershed Segmentation"),
        (canny_results["overlay"],     None,   "Canny Edge Detection"),
    ]

    for ax, (img, cmap, title) in zip(axes, items):
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("Segmentation Method Comparison", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def _save_and_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        print(f"[Segmentation] Saved → {save_path}")
    plt.show()
