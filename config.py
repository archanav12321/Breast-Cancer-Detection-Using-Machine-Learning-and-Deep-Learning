# config.py — Central configuration for Breast Cancer Detection Project

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CBIS_DIR        = os.path.join(DATA_DIR, "cbis_ddsm")
HISTO_DIR       = os.path.join(DATA_DIR, "histopathology")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
ENHANCED_DIR    = os.path.join(OUTPUT_DIR, "enhanced")
SEGMENTED_DIR   = os.path.join(OUTPUT_DIR, "segmented")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")

# ─── Image Settings ───────────────────────────────────────────────────────────
IMG_SIZE        = (50, 50)          # Histopathology patch size
MAMMOGRAM_SIZE  = (224, 224)        # CBIS-DDSM mammogram resize

# ─── Enhancement Settings ─────────────────────────────────────────────────────
MEDIAN_BLUR_KSIZE   = 5             # Kernel size for median blur
SHARPEN_STRENGTH    = 1.5           # Sharpening alpha
CLAHE_CLIP_LIMIT    = 2.0           # CLAHE clip limit
CLAHE_TILE_GRID     = (8, 8)        # CLAHE tile grid size

# ─── Segmentation Settings ────────────────────────────────────────────────────
CANNY_THRESHOLD1    = 50
CANNY_THRESHOLD2    = 150
WATERSHED_MARKERS   = 2             # Minimum peak distance for watershed markers

# ─── CNN Training Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 25
LEARNING_RATE   = 1e-4
DROPOUT_RATE    = 0.5
VALIDATION_SPLIT = 0.2
TEST_SPLIT      = 0.1
RANDOM_SEED     = 42

# ─── Model File Names ─────────────────────────────────────────────────────────
HISTO_MODEL_NAME        = "cnn_histopathology.keras"
WATERSHED_MODEL_NAME    = "cnn_watershed.keras"
CANNY_MODEL_NAME        = "cnn_canny.keras"

# ─── Classes ──────────────────────────────────────────────────────────────────
HISTO_CLASSES   = ["Non-Cancerous (IDC-)", "Cancerous (IDC+)"]
CBIS_CLASSES    = ["Benign", "Malignant"]
