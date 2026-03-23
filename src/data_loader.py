# utils/data_loader.py — Dataset loading & preprocessing utilities

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ══════════════════════════════════════════════════════════════════════════════
# CBIS-DDSM Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_cbis_single_image(image_path: str) -> np.ndarray:
    """
    Load a single CBIS-DDSM mammogram image for enhancement & segmentation.
    Converts DICOM-like grayscale to uint8 and resizes.

    Args:
        image_path: Path to image file (PNG / JPEG / DICOM-converted)

    Returns:
        Grayscale numpy array of shape (H, W)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.resize(img, MAMMOGRAM_SIZE)
    return img


def load_cbis_dataset(csv_path: str, image_root: str) -> tuple:
    """
    Load CBIS-DDSM dataset from CSV metadata file.
    Expected CSV columns: 'image_file_path', 'pathology' (BENIGN / MALIGNANT).

    Args:
        csv_path  : Path to CBIS-DDSM CSV metadata file
        image_root: Root directory containing images

    Returns:
        images (np.ndarray), labels (np.ndarray), df (pd.DataFrame)
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading CBIS-DDSM"):
        img_path = os.path.join(image_root, str(row["image_file_path"]).strip())
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, MAMMOGRAM_SIZE)
        images.append(img)
        label = 1 if "MALIGNANT" in str(row["pathology"]).upper() else 0
        labels.append(label)

    images = np.array(images, dtype=np.float32) / 255.0
    images = images[..., np.newaxis]          # Add channel dim → (N, H, W, 1)
    labels = np.array(labels, dtype=np.int32)
    return images, labels, df


# ══════════════════════════════════════════════════════════════════════════════
# Breast Histopathology Images Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_histopathology_dataset(data_dir: str) -> tuple:
    """
    Load the Breast Histopathology Images dataset.

    Expected directory structure:
        data_dir/
            <patient_id>/
                0/   ← non-cancerous patches (IDC-)
                1/   ← cancerous patches     (IDC+)

    Args:
        data_dir: Root directory of the dataset

    Returns:
        images (np.ndarray), labels (np.ndarray)
    """
    images, labels = [], []
    data_path = Path(data_dir)

    all_files = list(data_path.rglob("*.png")) + list(data_path.rglob("*.jpg"))
    print(f"[DataLoader] Found {len(all_files)} histopathology patches.")

    for fpath in tqdm(all_files, desc="Loading Histopathology"):
        # Label is encoded in parent folder name: 0 or 1
        label_str = fpath.parent.name
        if label_str not in ("0", "1"):
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(int(label_str))

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)
    print(f"[DataLoader] Cancerous: {labels.sum()} | Non-Cancerous: {(labels == 0).sum()}")
    return images, labels


# ══════════════════════════════════════════════════════════════════════════════
# Train / Val / Test Splitter
# ══════════════════════════════════════════════════════════════════════════════

def split_dataset(images: np.ndarray, labels: np.ndarray,
                  val_split: float = VALIDATION_SPLIT,
                  test_split: float = TEST_SPLIT,
                  seed: int = RANDOM_SEED) -> dict:
    """
    Split dataset into train, validation, and test sets with stratification.

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    n_classes = len(np.unique(labels))
    y_cat = to_categorical(labels, num_classes=n_classes)

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, y_cat,
        test_size=test_split,
        stratify=labels,
        random_state=seed
    )
    val_ratio = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=seed
    )

    print(f"[Split] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "n_classes": n_classes
    }


# ══════════════════════════════════════════════════════════════════════════════
# Demo / Synthetic Data Generator (for testing without real dataset)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_histopathology(n_samples: int = 500,
                                       img_size: tuple = IMG_SIZE) -> tuple:
    """
    Generate synthetic histopathology-like patches for pipeline testing.
    Cancerous patches have brighter, more structured centers.
    """
    np.random.seed(RANDOM_SEED)
    images, labels = [], []

    for i in range(n_samples):
        label = i % 2
        img = np.random.rand(*img_size, 3).astype(np.float32)
        if label == 1:
            # Simulate denser cell clusters (brighter region in center)
            cx, cy = img_size[0] // 2, img_size[1] // 2
            r = img_size[0] // 4
            for x in range(img_size[0]):
                for y in range(img_size[1]):
                    if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                        img[x, y] = np.clip(img[x, y] + 0.4, 0, 1)
        images.append(img)
        labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def generate_synthetic_mammogram(size: tuple = MAMMOGRAM_SIZE) -> np.ndarray:
    """Generate a synthetic grayscale mammogram for pipeline testing."""
    np.random.seed(RANDOM_SEED)
    img = np.random.randint(30, 120, size, dtype=np.uint8)
    # Simulate a dense region (potential tumor)
    cx, cy = size[0] // 2, size[1] // 2
    cv2.circle(img, (cy, cx), 40, 200, -1)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    return img
