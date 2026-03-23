# utils/model.py — CNN Model Architectures for Breast Cancer Classification

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


# ══════════════════════════════════════════════════════════════════════════════
# CNN for Histopathology Patches  (50×50 RGB input)
# ══════════════════════════════════════════════════════════════════════════════

def build_histopathology_cnn(input_shape: tuple = (*IMG_SIZE, 3),
                              n_classes: int = 2,
                              learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Lightweight CNN for 50×50 RGB histopathology patch classification.

    Architecture:
        Conv Block ×3 → GlobalAvgPool → Dense → Dropout → Softmax

    Args:
        input_shape   : (H, W, C) — default (50, 50, 3)
        n_classes     : Number of output classes (2: IDC- / IDC+)
        learning_rate : Adam learning rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name="histo_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           name="histo_output")(x)

    model = models.Model(inputs, outputs, name="HistopathologyCNN")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# CNN for Segmented Mammograms  (224×224 Grayscale input)
# ══════════════════════════════════════════════════════════════════════════════

def build_mammogram_cnn(input_shape: tuple = (*MAMMOGRAM_SIZE, 1),
                        n_classes: int = 2,
                        learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Deeper CNN for 224×224 grayscale segmented mammogram classification.
    Used for both Watershed-segmented and Canny-edge-segmented inputs.

    Architecture:
        Conv Block ×4 → GlobalAvgPool → Dense → Dropout → Softmax

    Args:
        input_shape   : (H, W, 1) — default (224, 224, 1)
        n_classes     : 2 (Benign / Malignant)
        learning_rate : Adam learning rate

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name="mammo_input")

    # Block 1
    x = _conv_block(inputs, 32)

    # Block 2
    x = _conv_block(x, 64)

    # Block 3
    x = _conv_block(x, 128)

    # Block 4
    x = _conv_block(x, 256)

    # Global head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           name="mammo_output")(x)

    model = models.Model(inputs, outputs, name="MammogramCNN")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


def _conv_block(x, filters: int) -> tf.Tensor:
    """
    Standard Conv → BN → ReLU → Conv → BN → ReLU → MaxPool block.
    """
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Transfer Learning Variant (Optional — MobileNetV2 Backbone)
# ══════════════════════════════════════════════════════════════════════════════

def build_transfer_learning_model(input_shape: tuple = (224, 224, 3),
                                   n_classes: int = 2,
                                   learning_rate: float = LEARNING_RATE,
                                   freeze_base: bool = True) -> tf.keras.Model:
    """
    MobileNetV2-based transfer learning model for RGB mammograms.
    Fine-tune by setting freeze_base=False after initial training.

    Args:
        input_shape  : Must have 3 channels for MobileNetV2
        n_classes    : Number of output classes
        learning_rate: Adam learning rate
        freeze_base  : Whether to freeze pretrained weights

    Returns:
        Compiled Keras model
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="TransferLearningCNN")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model Summary Helper
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model: tf.keras.Model):
    """Print model summary with parameter counts."""
    model.summary()
    total   = model.count_params()
    trainab = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainab:,}")
    print(f"  Non-trainable    : {total - trainab:,}\n")
