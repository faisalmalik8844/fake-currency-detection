"""
test_preprocessing.py

Basic sanity tests for the preprocessing module.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import get_data_generators, CLASS_NAMES


def test_class_names_defined_correctly():
    """Sanity check that class names match the dataset's actual folder names."""
    assert CLASS_NAMES == ["Fake Notes", "Real Notes"]


def test_get_data_generators_returns_correct_class_mapping():
    """
    Confirms get_data_generators returns train and val generators
    with the expected class mapping.

    NOTE: this test requires data/processed/train and data/processed/val
    to already exist (run split_dataset first, or run notebooks/02_preprocessing.ipynb).
    """
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    train_gen, val_gen = get_data_generators(str(processed_dir))

    assert train_gen.class_indices == {"Fake Notes": 0, "Real Notes": 1}
    assert val_gen.samples > 0
    assert train_gen.samples > 0