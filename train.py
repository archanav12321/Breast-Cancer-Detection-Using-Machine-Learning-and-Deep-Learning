#!/usr/bin/env python3
"""
train.py — Main Training Script
================================
Breast Cancer Detection using Deep Learning

Trains three CNN models:
  1. Histopathology CNN  (cancerous vs non-cancerous patches)
  2. Watershed CNN       (CBIS-DDSM, watershed-segmented input)
  3. Canny CNN           (CBIS-DDSM, canny-edge-segmented input)

Usage:
    python train.py --mode all              # Train all models
    python train.py --mode histo            # Train histopathology model only
    python train.py --mode mammogram        # Train both mammogram models
    python train.py --mode demo             # Run on synthetic data (no dataset needed)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, TensorBoard)
from datetime import datetime

# ─── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from utils import (
    load_histopathology_dataset, load_cbis_single_image, load_cbis_dataset,
    generate_synthetic_histopathology, generate_synthetic_mammogram,
    split_dataset,
    enhance_mammogram,
    apply_watershed, apply_canny,
    prepare_watershed_for_cnn, prepare_canny_for_cnn,
    build_histopathology_cnn, build_mammogram_cnn, print_model_summary,
    plot_class_distribution, plot_training_history,
    plot_confusion_matrix, plot_roc_curve,
    print_classification_report, plot_accuracy_comparison,
    plot_predictions_grid,
)


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════
tf.random.set_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

os.makedirs(cfg.MODEL_DIR, exist_ok=True)
os.makedirs(cfg.PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks Factory
# ══════════════════════════════════════════════════════════════════════════════

def get_callbacks(model_name: str) -> list:
    """Return standard training callbacks for any model."""
    ckpt_path = os.path.join(cfg.MODEL_DIR, f"{model_name}_best.keras")
    log_dir   = os.path.join("logs", model_name,
                             datetime.now().strftime("%Y%m%d-%H%M%S"))
    return [
        ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Train Histopathology Model
# ══════════════════════════════════════════════════════════════════════════════

def train_histopathology(demo: bool = False) -> float:
    """
    Train CNN on Breast Histopathology Images (IDC- vs IDC+).

    Returns:
        Test accuracy
    """
    print("\n" + "═" * 60)
    print("  HISTOPATHOLOGY CNN TRAINING")
    print("═" * 60)

    # ── Load data ────────────────────────────────────────────────
    if demo:
        print("[Demo] Generating synthetic histopathology patches...")
        images, labels = generate_synthetic_histopathology(n_samples=2000)
    else:
        if not os.path.exists(cfg.HISTO_DIR):
            print(f"[ERROR] Dataset not found: {cfg.HISTO_DIR}")
            print("  → Download: https://www.kaggle.com/paultimothymooney/breast-histopathology-images")
            return 0.0
        images, labels = load_histopathology_dataset(cfg.HISTO_DIR)

    # ── Visualize distribution ────────────────────────────────────
    plot_class_distribution(
        labels, cfg.HISTO_CLASSES, "Histopathology Dataset Distribution",
        save_path=os.path.join(cfg.PLOTS_DIR, "histo_class_dist.png")
    )

    # ── Split ─────────────────────────────────────────────────────
    data = split_dataset(images, labels)

    # ── Build model ───────────────────────────────────────────────
    input_shape = images.shape[1:]
    model = build_histopathology_cnn(input_shape=input_shape,
                                      n_classes=data["n_classes"])
    print_model_summary(model)

    # ── Train ─────────────────────────────────────────────────────
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=get_callbacks("histo_cnn"),
        verbose=1,
        class_weight=_compute_class_weights(labels)
    )

    # ── Evaluate ──────────────────────────────────────────────────
    loss, acc, *_ = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\n  ✅ Test Accuracy : {acc * 100:.2f}%")
    print(f"  ✅ Test Loss     : {loss:.4f}")

    y_pred = model.predict(data["X_test"])

    # ── Plots ─────────────────────────────────────────────────────
    plot_training_history(
        history, "Histopathology CNN",
        save_path=os.path.join(cfg.PLOTS_DIR, "histo_history.png")
    )
    plot_confusion_matrix(
        data["y_test"], y_pred, cfg.HISTO_CLASSES, "Histopathology CNN",
        save_path=os.path.join(cfg.PLOTS_DIR, "histo_confusion.png")
    )
    plot_roc_curve(
        data["y_test"], y_pred, cfg.HISTO_CLASSES, "Histopathology CNN",
        save_path=os.path.join(cfg.PLOTS_DIR, "histo_roc.png")
    )
    plot_predictions_grid(
        data["X_test"], data["y_test"], y_pred, cfg.HISTO_CLASSES,
        save_path=os.path.join(cfg.PLOTS_DIR, "histo_predictions.png")
    )
    print_classification_report(data["y_test"], y_pred, cfg.HISTO_CLASSES)

    # ── Save final ────────────────────────────────────────────────
    model_path = os.path.join(cfg.MODEL_DIR, cfg.HISTO_MODEL_NAME)
    model.save(model_path)
    print(f"  💾 Model saved → {model_path}")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# Train Mammogram CNN (Watershed + Canny)
# ══════════════════════════════════════════════════════════════════════════════

def train_mammogram(demo: bool = False) -> dict:
    """
    Train two CNNs on CBIS-DDSM mammograms:
      - Watershed-segmented input model
      - Canny-edge-segmented input model

    Returns:
        {"watershed": acc, "canny": acc}
    """
    print("\n" + "═" * 60)
    print("  MAMMOGRAM CNN TRAINING  (Watershed + Canny)")
    print("═" * 60)

    results = {}

    if demo:
        print("[Demo] Generating synthetic mammogram dataset...")
        n = 400
        ws_images, cn_images, labels = [], [], []
        for i in range(n):
            label = i % 2
            raw = generate_synthetic_mammogram()
            enhanced = enhance_mammogram(raw)
            clahe_img = enhanced["clahe"]

            ws = apply_watershed(clahe_img)
            cn = apply_canny(clahe_img)

            ws_images.append(prepare_watershed_for_cnn(ws))
            cn_images.append(prepare_canny_for_cnn(cn))
            labels.append(label)

        ws_images = np.array(ws_images, dtype=np.float32)
        cn_images = np.array(cn_images, dtype=np.float32)
        labels    = np.array(labels, dtype=np.int32)
    else:
        csv_path = os.path.join(cfg.CBIS_DIR, "mass_case_description_train_set.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] CSV not found: {csv_path}")
            print("  → Download: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
            return {}

        raw_images, labels, df = load_cbis_dataset(csv_path, cfg.CBIS_DIR)

        from utils import plot_cbis_metadata
        plot_cbis_metadata(
            df, save_path=os.path.join(cfg.PLOTS_DIR, "cbis_meta.png")
        )

        print("[Preprocessing] Enhancing + segmenting mammograms...")
        ws_images, cn_images = [], []
        for img in raw_images:
            gray = (img.squeeze() * 255).astype(np.uint8)
            enhanced = enhance_mammogram(gray)
            clahe_img = enhanced["clahe"]
            ws = apply_watershed(clahe_img)
            cn = apply_canny(clahe_img)
            ws_images.append(prepare_watershed_for_cnn(ws))
            cn_images.append(prepare_canny_for_cnn(cn))

        ws_images = np.array(ws_images, dtype=np.float32)
        cn_images = np.array(cn_images, dtype=np.float32)

    plot_class_distribution(
        labels, cfg.CBIS_CLASSES, "CBIS-DDSM Class Distribution",
        save_path=os.path.join(cfg.PLOTS_DIR, "cbis_class_dist.png")
    )

    for seg_type, images in [("watershed", ws_images), ("canny", cn_images)]:
        print(f"\n  ── Training {seg_type.upper()} CNN ──")
        data   = split_dataset(images, labels)
        model  = build_mammogram_cnn(input_shape=images.shape[1:],
                                      n_classes=data["n_classes"])
        history = model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=get_callbacks(f"{seg_type}_cnn"),
            verbose=1,
        )

        loss, acc, *_ = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        print(f"\n  ✅ [{seg_type.upper()}] Test Accuracy: {acc * 100:.2f}%")
        results[seg_type] = acc

        y_pred = model.predict(data["X_test"])
        tag = seg_type

        plot_training_history(
            history, f"{seg_type.title()} CNN",
            save_path=os.path.join(cfg.PLOTS_DIR, f"{tag}_history.png")
        )
        plot_confusion_matrix(
            data["y_test"], y_pred, cfg.CBIS_CLASSES,
            f"{seg_type.title()} CNN",
            save_path=os.path.join(cfg.PLOTS_DIR, f"{tag}_confusion.png")
        )
        plot_roc_curve(
            data["y_test"], y_pred, cfg.CBIS_CLASSES,
            f"{seg_type.title()} CNN",
            save_path=os.path.join(cfg.PLOTS_DIR, f"{tag}_roc.png")
        )
        print_classification_report(data["y_test"], y_pred, cfg.CBIS_CLASSES)

        mname = cfg.WATERSHED_MODEL_NAME if seg_type == "watershed" \
                else cfg.CANNY_MODEL_NAME
        model_path = os.path.join(cfg.MODEL_DIR, mname)
        model.save(model_path)
        print(f"  💾 Model saved → {model_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summary Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_final_summary(accuracies: dict):
    plot_accuracy_comparison(
        accuracies,
        save_path=os.path.join(cfg.PLOTS_DIR, "accuracy_summary.png")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_class_weights(labels: np.ndarray) -> dict:
    """Compute inverse-frequency class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Breast Cancer Detection — Training Script"
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["all", "histo", "mammogram", "demo"],
        help="Training mode (default: demo)"
    )
    parser.add_argument(
        "--epochs", type=int, default=cfg.EPOCHS,
        help=f"Number of epochs (default: {cfg.EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg.BATCH_SIZE,
        help=f"Batch size (default: {cfg.BATCH_SIZE})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg.EPOCHS     = args.epochs
    cfg.BATCH_SIZE = args.batch_size

    demo = (args.mode == "demo")
    if demo:
        print("\n  🔬 DEMO MODE — Using synthetic data (no dataset required)")

    all_accuracies = {}

    if args.mode in ("all", "histo", "demo"):
        acc = train_histopathology(demo=demo)
        all_accuracies["Histopathology CNN"] = acc

    if args.mode in ("all", "mammogram", "demo"):
        mammo_acc = train_mammogram(demo=demo)
        for k, v in mammo_acc.items():
            all_accuracies[f"{k.title()} CNN"] = v

    if all_accuracies:
        print("\n" + "═" * 60)
        print("  FINAL RESULTS SUMMARY")
        print("═" * 60)
        for name, acc in all_accuracies.items():
            print(f"  {name:30s} → {acc * 100:.2f}%")
        plot_final_summary(all_accuracies)

    print("\n  ✅ Training complete!\n")


if __name__ == "__main__":
    main()
