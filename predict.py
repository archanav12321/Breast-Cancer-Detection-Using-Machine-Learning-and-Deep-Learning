#!/usr/bin/env python3
"""
predict.py — Single Image Inference
=====================================
Run breast cancer classification on a new mammogram or histopathology patch.

Usage:
    # Histopathology patch
    python predict.py --image path/to/patch.png --type histo

    # Mammogram (runs enhancement + segmentation + classification)
    python predict.py --image path/to/mammogram.png --type mammogram

    # Demo with synthetic image
    python predict.py --demo --type histo
    python predict.py --demo --type mammogram
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import (
    enhance_mammogram, apply_watershed, apply_canny,
    prepare_watershed_for_cnn, prepare_canny_for_cnn,
    generate_synthetic_mammogram, generate_synthetic_histopathology,
    plot_enhancement_pipeline, plot_watershed_results, plot_canny_results,
)


DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
TEXT     = "#e6edf3"
GREEN    = "#3fb950"
RED      = "#f78166"
BLUE     = "#58a6ff"


# ══════════════════════════════════════════════════════════════════════════════
# Histopathology Inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_histopathology(image_path: str = None, demo: bool = False):
    """
    Classify a histopathology patch as Cancerous (IDC+) or Non-Cancerous (IDC-).

    Args:
        image_path: Path to 50×50 RGB patch image
        demo      : Use synthetic image if True
    """
    model_path = os.path.join(cfg.MODEL_DIR, cfg.HISTO_MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  → Run: python train.py --mode histo")
        return

    model = tf.keras.models.load_model(model_path)

    if demo:
        imgs, labels = generate_synthetic_histopathology(n_samples=2)
        img = imgs[0]
        true_label = cfg.HISTO_CLASSES[labels[0]]
    else:
        raw = cv2.imread(image_path)
        if raw is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = cv2.resize(raw, cfg.IMG_SIZE).astype(np.float32) / 255.0
        true_label = "Unknown"

    inp = img[np.newaxis, ...]
    probs = model.predict(inp, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = cfg.HISTO_CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    _show_histo_result(img, pred_label, confidence, probs, true_label)


# ══════════════════════════════════════════════════════════════════════════════
# Mammogram Inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_mammogram(image_path: str = None, demo: bool = False):
    """
    Full mammogram pipeline: enhance → segment → classify.
    Runs both Watershed and Canny models and compares predictions.

    Args:
        image_path: Path to grayscale mammogram image
        demo      : Use synthetic image if True
    """
    ws_model_path = os.path.join(cfg.MODEL_DIR, cfg.WATERSHED_MODEL_NAME)
    cn_model_path = os.path.join(cfg.MODEL_DIR, cfg.CANNY_MODEL_NAME)

    if not os.path.exists(ws_model_path) or not os.path.exists(cn_model_path):
        print("[ERROR] Mammogram models not found.")
        print("  → Run: python train.py --mode mammogram")
        return

    ws_model = tf.keras.models.load_model(ws_model_path)
    cn_model = tf.keras.models.load_model(cn_model_path)

    if demo:
        raw = generate_synthetic_mammogram()
    else:
        raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        raw = cv2.resize(raw, cfg.MAMMOGRAM_SIZE)

    # Pipeline
    enhanced = enhance_mammogram(raw)
    clahe_img = enhanced["clahe"]

    ws_results = apply_watershed(clahe_img)
    cn_results = apply_canny(clahe_img)

    ws_inp = prepare_watershed_for_cnn(ws_results)[np.newaxis, ...]
    cn_inp = prepare_canny_for_cnn(cn_results)[np.newaxis, ...]

    ws_probs = ws_model.predict(ws_inp, verbose=0)[0]
    cn_probs = cn_model.predict(cn_inp, verbose=0)[0]

    ws_pred = cfg.CBIS_CLASSES[np.argmax(ws_probs)]
    cn_pred = cfg.CBIS_CLASSES[np.argmax(cn_probs)]

    # Show pipeline & results
    plot_enhancement_pipeline(enhanced)
    plot_watershed_results(ws_results)
    plot_canny_results(cn_results)
    _show_mammogram_result(raw, clahe_img,
                           ws_pred, ws_probs,
                           cn_pred, cn_probs)


# ══════════════════════════════════════════════════════════════════════════════
# Result Display
# ══════════════════════════════════════════════════════════════════════════════

def _show_histo_result(img, pred_label, confidence, probs, true_label):
    is_cancer = "Cancerous" in pred_label
    accent = RED if is_cancer else GREEN

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Image
    ax_img.imshow(img)
    ax_img.set_title(f"Prediction: {pred_label}\n{confidence:.1f}% Confidence",
                     color=accent, fontsize=13, fontweight="bold", pad=10)
    for spine in ax_img.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(3)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_facecolor(PANEL_BG)

    if true_label != "Unknown":
        ax_img.set_xlabel(f"True Label: {true_label}", color=TEXT, fontsize=10)

    # Probability bars
    ax_bar.set_facecolor(PANEL_BG)
    colors = [RED if "Canc" in c else GREEN for c in cfg.HISTO_CLASSES]
    bars = ax_bar.barh(cfg.HISTO_CLASSES, probs * 100,
                       color=colors, edgecolor=DARK_BG, height=0.4)
    for bar, p in zip(bars, probs):
        ax_bar.text(p * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f"{p * 100:.1f}%", va="center", color=TEXT, fontsize=12)
    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Probability (%)", color="#8b949e")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#30363d")
    ax_bar.tick_params(colors=TEXT)
    ax_bar.set_title("Class Probabilities", color=TEXT,
                     fontsize=12, fontweight="bold")

    fig.suptitle("🔬 Histopathology Classification Result",
                 color=TEXT, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def _show_mammogram_result(original, clahe_img,
                            ws_pred, ws_probs,
                            cn_pred, cn_probs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Original
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Mammogram", color=TEXT, fontsize=11, pad=8)
    axes[0].axis("off")
    axes[0].set_facecolor(PANEL_BG)

    # Watershed result
    ws_accent = RED if "Malignant" in ws_pred else GREEN
    axes[1].imshow(clahe_img, cmap="gray")
    axes[1].set_title(f"Watershed Model\n→ {ws_pred} ({ws_probs.max()*100:.1f}%)",
                      color=ws_accent, fontsize=11, fontweight="bold", pad=8)
    for sp in axes[1].spines.values():
        sp.set_edgecolor(ws_accent)
        sp.set_linewidth(3)
    axes[1].axis("off")

    # Canny result
    cn_accent = RED if "Malignant" in cn_pred else GREEN
    axes[2].imshow(clahe_img, cmap="gray")
    axes[2].set_title(f"Canny Model\n→ {cn_pred} ({cn_probs.max()*100:.1f}%)",
                      color=cn_accent, fontsize=11, fontweight="bold", pad=8)
    for sp in axes[2].spines.values():
        sp.set_edgecolor(cn_accent)
        sp.set_linewidth(3)
    axes[2].axis("off")

    fig.suptitle("🩻 Mammogram Classification Results",
                 color=TEXT, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Breast Cancer — Inference")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--type", type=str, default="histo",
                        choices=["histo", "mammogram"],
                        help="Image type (default: histo)")
    parser.add_argument("--demo", action="store_true",
                        help="Run on synthetic image (no model needed for structure)")
    args = parser.parse_args()

    if not args.demo and args.image is None:
        print("[ERROR] Provide --image <path> or use --demo")
        return

    if args.type == "histo":
        predict_histopathology(image_path=args.image, demo=args.demo)
    else:
        predict_mammogram(image_path=args.image, demo=args.demo)


if __name__ == "__main__":
    main()
