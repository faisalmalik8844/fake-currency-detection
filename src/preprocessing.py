"""
preprocessing.py

Data loading, splitting, and augmentation pipeline for the PKR
Fake Currency Detection project. Matches the pipeline validated
in notebooks/02_preprocessing.ipynb.
"""

import random
import shutil
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["Fake Notes", "Real Notes" , "Not A Note"]


def split_dataset(raw_dir: str, processed_dir: str, split_ratio: float = 0.8, seed: int = 42):
    """
    Splits raw images into train/val folders, preserving class balance.
    """
    random.seed(seed)
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    for class_folder in CLASS_NAMES:
        images = list((raw_path / class_folder).glob("*.*"))
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        val_images = images[split_point:]

        (processed_path / "train" / class_folder).mkdir(parents=True, exist_ok=True)
        (processed_path / "val" / class_folder).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, processed_path / "train" / class_folder / img.name)
        for img in val_images:
            shutil.copy(img, processed_path / "val" / class_folder / img.name)

        print(f"{class_folder}: {len(train_images)} train, {len(val_images)} val")


def get_data_generators(processed_dir: str, img_size: tuple = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """
    Creates train/validation data generators.
    Augmentation is applied ONLY to training data.
    No flips are used since currency notes have a fixed orientation.
    """
    processed_path = Path(processed_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        processed_path / "train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        processed_path / "val",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, val_generator


if __name__ == "__main__":
    split_dataset("../data/raw", "../data/processed")
    train_gen, val_gen = get_data_generators("../data/processed")
    print("Class indices:", train_gen.class_indices)