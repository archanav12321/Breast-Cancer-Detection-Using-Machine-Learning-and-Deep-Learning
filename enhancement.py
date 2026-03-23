# utils/enhancement.py — Image Enhancement Pipeline

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ══════════════════════════════════════════════════════════════════════════════
# Individual Enhancement Functions
# ══════════════════════════════════════════════════════════════════════════════

def extract_roi(image: np.ndarray) -> np.ndarray:
    """
    Extract Region of Interest (ROI) from mammogram.
    Removes background by thresholding and finding the largest contour.

    Args:
        image: Grayscale uint8 image

    Returns:
        ROI-masked image
    """
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    roi = cv2.bitwise_and(image, image, mask=mask)
    return roi


def apply_median_blur(image: np.ndarray, ksize: int = MEDIAN_BLUR_KSIZE) -> np.ndarray:
    """
    Apply Median Blur to remove salt-and-pepper noise.

    Args:
        image : Grayscale or color image
        ksize : Kernel size (must be odd)

    Returns:
        Blurred image
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def apply_sharpening(image: np.ndarray, strength: float = SHARPEN_STRENGTH) -> np.ndarray:
    """
    Sharpen image using an unsharp mask approach.

    Args:
        image    : Input image
        strength : Blend factor (1.0 = original, >1.0 = sharper)

    Returns:
        Sharpened image
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    blended = cv2.addWeighted(image, 1.0, sharpened, strength - 1.0, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray,
                clip_limit: float = CLAHE_CLIP_LIMIT,
                tile_grid: tuple = CLAHE_TILE_GRID) -> np.ndarray:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).
    Enhances local contrast while limiting noise amplification.

    Args:
        image      : Grayscale image
        clip_limit : Threshold for contrast limiting
        tile_grid  : Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


# ══════════════════════════════════════════════════════════════════════════════
# Full Enhancement Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def enhance_mammogram(image: np.ndarray) -> dict:
    """
    Run the complete enhancement pipeline on a mammogram image.

    Pipeline:
        1. ROI Extraction
        2. Median Blur (noise removal)
        3. Sharpening
        4. CLAHE (contrast enhancement)

    Args:
        image: Grayscale uint8 mammogram

    Returns:
        dict with keys: original, roi, blurred, sharpened, clahe
    """
    results = {"original": image.copy()}

    # Step 1: ROI
    roi = extract_roi(image)
    results["roi"] = roi

    # Step 2: Median Blur
    blurred = apply_median_blur(roi)
    results["blurred"] = blurred

    # Step 3: Sharpening
    sharpened = apply_sharpening(blurred)
    results["sharpened"] = sharpened

    # Step 4: CLAHE
    clahe_img = apply_clahe(sharpened)
    results["clahe"] = clahe_img

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_enhancement_pipeline(results: dict, save_path: str = None):
    """
    Plot all enhancement steps side-by-side.

    Args:
        results   : Output dict from enhance_mammogram()
        save_path : If provided, saves figure to this path
    """
    titles = {
        "original" : "Original Mammogram",
        "roi"      : "ROI Extraction",
        "blurred"  : "Median Blur",
        "sharpened": "Sharpened",
        "clahe"    : "CLAHE Enhanced"
    }

    fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax, (key, img) in zip(axes, results.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(titles.get(key, key), color="white", fontsize=11, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plt.suptitle("Mammogram Enhancement Pipeline", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        print(f"[Enhancement] Saved → {save_path}")
    plt.show()


def plot_intensity_histogram(images_dict: dict, save_path: str = None):
    """
    Plot pixel intensity histograms for each enhancement stage.

    Args:
        images_dict: Output dict from enhance_mammogram()
        save_path  : Optional save path
    """
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]
    labels = list(images_dict.keys())

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for i, (key, img) in enumerate(images_dict.items()):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        ax.plot(hist, color=colors[i % len(colors)],
                label=labels[i], linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Pixel Intensity", color="#8b949e")
    ax.set_ylabel("Frequency", color="#8b949e")
    ax.set_title("Pixel Intensity Histograms — Enhancement Stages",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d",
              labelcolor="white", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        print(f"[Enhancement] Histogram saved → {save_path}")
    plt.show()
