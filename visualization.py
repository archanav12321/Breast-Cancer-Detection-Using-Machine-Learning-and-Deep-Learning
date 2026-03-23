# utils/visualization.py — Data Visualization & Results Plotting

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Dark theme defaults
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
COLORS    = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff"]


def _apply_dark_style(ax, title: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(labels: np.ndarray, class_names: list,
                             title: str = "Class Distribution",
                             save_path: str = None):
    """
    Bar chart + pie chart of class distribution side by side.
    """
    unique, counts = np.unique(labels, return_counts=True)
    names = [class_names[i] for i in unique]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Bar chart
    bars = ax1.bar(names, counts, color=COLORS[:len(names)],
                   edgecolor=BORDER, linewidth=0.8, width=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                 f"{count:,}", ha="center", va="bottom", color=TEXT, fontsize=11)
    ax1.set_ylabel("Sample Count", color=MUTED)
    _apply_dark_style(ax1, "Sample Count per Class")

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        counts, labels=names, autopct="%1.1f%%",
        colors=COLORS[:len(names)], startangle=140,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2}
    )
    for t in texts + autotexts:
        t.set_color(TEXT)
    ax2.set_title("Class Proportions", color=TEXT, fontsize=12,
                  fontweight="bold", pad=10)
    ax2.set_facecolor(PANEL_BG)

    fig.suptitle(title, color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path, "ClassDist")


def plot_cbis_metadata(df: pd.DataFrame, save_path: str = None):
    """
    Visualize CBIS-DDSM dataset metadata distributions:
    pathology, breast density, assessment scores.
    """
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor(DARK_BG)

    cols_to_plot = []
    for col in ["pathology", "breast_density", "assessment", "subtlety"]:
        if col in df.columns:
            cols_to_plot.append(col)

    if not cols_to_plot:
        print("[Visualization] No recognized CBIS-DDSM metadata columns found.")
        return

    axes = fig.subplots(1, len(cols_to_plot))
    if len(cols_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols_to_plot):
        vc = df[col].value_counts()
        ax.barh(vc.index.astype(str), vc.values,
                color=COLORS[:len(vc)], edgecolor=BORDER)
        for i, v in enumerate(vc.values):
            ax.text(v + max(vc.values) * 0.01, i, str(v),
                    va="center", color=TEXT, fontsize=9)
        _apply_dark_style(ax, col.replace("_", " ").title())
        ax.set_xlabel("Count", color=MUTED)

    fig.suptitle("CBIS-DDSM Dataset Analysis", color=TEXT,
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path, "CBIS_Meta")


# ══════════════════════════════════════════════════════════════════════════════
# Training History
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, model_name: str = "CNN",
                           save_path: str = None):
    """
    Plot accuracy and loss curves for training and validation.
    """
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Accuracy
    ax1.plot(epochs, hist["accuracy"], color=COLORS[0], linewidth=2,
             label="Train Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], color=COLORS[1], linewidth=2,
             linestyle="--", label="Val Accuracy")
    ax1.set_xlabel("Epoch", color=MUTED)
    ax1.set_ylabel("Accuracy", color=MUTED)
    ax1.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)
    _apply_dark_style(ax1, "Accuracy")

    # Loss
    ax2.plot(epochs, hist["loss"], color=COLORS[2], linewidth=2,
             label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], color=COLORS[3], linewidth=2,
             linestyle="--", label="Val Loss")
    ax2.set_xlabel("Epoch", color=MUTED)
    ax2.set_ylabel("Loss", color=MUTED)
    ax2.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)
    _apply_dark_style(ax2, "Loss")

    fig.suptitle(f"{model_name} — Training History",
                 color=TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path, f"{model_name}_history")


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, model_name: str = "CNN",
                           save_path: str = None):
    """Plot styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true.argmax(axis=1) if y_true.ndim > 1 else y_true,
                          y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor=BORDER,
                annot_kws={"size": 14, "color": TEXT},
                ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_xlabel("Predicted Label", color=MUTED, fontsize=11)
    ax.set_ylabel("True Label", color=MUTED, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=10)
    _apply_dark_style(ax, f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    _save(fig, save_path, f"{model_name}_confusion")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   class_names: list, model_name: str = "CNN",
                   save_path: str = None):
    """Plot ROC curves for each class (One-vs-Rest)."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    if y_true.ndim == 1:
        from tensorflow.keras.utils import to_categorical
        y_true = to_categorical(y_true, n_classes)

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color=BORDER, linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", color=MUTED)
    ax.set_ylabel("True Positive Rate", color=MUTED)
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)
    _apply_dark_style(ax, f"{model_name} — ROC Curve")
    plt.tight_layout()
    _save(fig, save_path, f"{model_name}_roc")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: list):
    """Print formatted sklearn classification report."""
    yt = y_true.argmax(axis=1) if y_true.ndim > 1 else y_true
    yp = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred
    print("\n" + "═" * 55)
    print(classification_report(yt, yp, target_names=class_names))
    print("═" * 55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Sample Predictions Grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_predictions_grid(images: np.ndarray, y_true: np.ndarray,
                           y_pred: np.ndarray, class_names: list,
                           n: int = 16, save_path: str = None):
    """
    Show a grid of sample predictions with true vs predicted labels.
    Correct = green border, Wrong = red border.
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    fig.patch.set_facecolor(DARK_BG)
    axes = axes.flatten()

    yt = y_true.argmax(axis=1) if y_true.ndim > 1 else y_true
    yp = y_pred.argmax(axis=1) if y_pred.ndim > 1 else y_pred

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        correct = yt[i] == yp[i]
        color = "#3fb950" if correct else "#f78166"
        label = f"T:{class_names[yt[i]]}\nP:{class_names[yp[i]]}"
        ax.set_title(label, color=color, fontsize=8, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_facecolor(PANEL_BG)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample Predictions  (🟢 Correct  🔴 Wrong)",
                 color=TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path, "predictions_grid")


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_comparison(results: dict, save_path: str = None):
    """
    Bar chart comparing accuracy of different models/methods.

    Args:
        results: {"Model Name": accuracy_float, ...}
    """
    names = list(results.keys())
    accs  = [v * 100 for v in results.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK_BG)

    bars = ax.bar(names, accs, color=COLORS[:len(names)],
                  edgecolor=BORDER, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom",
                color=TEXT, fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    ax.axhline(y=90, color=COLORS[3], linestyle="--",
               linewidth=1, alpha=0.5, label="90% baseline")
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT)
    _apply_dark_style(ax, "Model Accuracy Comparison")
    plt.tight_layout()
    _save(fig, save_path, "accuracy_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, save_path, name):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        path = save_path if save_path.endswith(".png") else \
               os.path.join(save_path, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"[Visualization] Saved → {path}")
    plt.show()
