"""
model.py

Model architectures: baseline CNN (from scratch) and transfer learning
(MobileNetV2). Matches the architectures validated in notebooks 03 and 04.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_baseline_cnn(input_shape: tuple = (224, 224, 3)) -> models.Model:
    """
    CNN built from scratch - used as a baseline for comparison
    against transfer learning.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="baseline_cnn")

    return model


def build_transfer_model(input_shape: tuple = (224, 224, 3), fine_tune_base: bool = False) -> models.Model:
    """
    Transfer learning model using MobileNetV2 pretrained on ImageNet.
    Only the classification head is trained by default (fine_tune_base=False).
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = fine_tune_base

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    return model


if __name__ == "__main__":
    print("Baseline CNN summary:")
    build_baseline_cnn().summary()

    print("\n\nTransfer learning model summary:")
    build_transfer_model().summary()