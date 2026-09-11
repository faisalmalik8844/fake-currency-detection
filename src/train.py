"""
train.py

Training script tying preprocessing and model together.
Trains either the baseline CNN or the MobileNetV2 transfer model.

Usage:
    python train.py --model transfer --epochs 10
    python train.py --model baseline --epochs 10
"""

import argparse
from datetime import datetime
from pathlib import Path

from preprocessing import get_data_generators
from model import build_baseline_cnn, build_transfer_model


def train_model(model_type: str, data_dir: str, epochs: int, batch_size: int):
    print(f"Loading data from {data_dir}...")
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)
    print("Class indices:", train_gen.class_indices)

    print(f"Building {model_type} model...")
    if model_type == "baseline":
        model = build_baseline_cnn()
    else:
        model = build_transfer_model()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
    )

    Path("../models").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../models/{model_type}_model_{timestamp}.keras"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    parser.add_argument("--data-dir", default="../data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_model(args.model, args.data_dir, args.epochs, args.batch_size)