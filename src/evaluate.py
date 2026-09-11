"""
evaluate.py

Full evaluation for the 3-class currency model: classification report,
confusion matrix, and per-class analysis.

Usage:
    python evaluate.py --model-path ../models/mobilenetv2_3class_currency.keras
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_NAMES = ["Fake Notes", "Not A Note", "Real Notes"]  # matches class_indices order


def evaluate_model(model_path: str, data_dir: str = "../data/processed"):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print("Loading validation data...")
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        f"{data_dir}/val",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )
    print("Class indices:", val_gen.class_indices)

    y_true = val_gen.classes
    predictions = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    overall_accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - 3-Class Model")
    plt.tight_layout()
    plt.savefig("../reports/figures/confusion_matrix_3class.png")
    print("Confusion matrix saved to reports/figures/confusion_matrix_3class.png")

    misclassified_idx = np.where(y_true != y_pred)[0]
    print(f"Misclassified: {len(misclassified_idx)} / {len(y_true)}")

    return {"accuracy": overall_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="../data/processed")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir)