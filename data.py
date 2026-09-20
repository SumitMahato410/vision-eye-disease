"""Dataset loading for both models.

Expected layout on disk (created by ``scripts/prepare_odir.py`` or
``scripts/split_dataset.py``)::

    data/internal/train/diabetic_retinopathy/xxx.jpg
    data/internal/train/glaucoma/...
    data/internal/val/...
    data/internal/test/...

Class names are taken from the folder names and sorted alphabetically, which is
what ``image_dataset_from_directory`` does, so the index of a class is stable
between training, evaluation and inference.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from .config import SEED, TaskConfig

SPLITS = ("train", "val", "test")

# TensorFlow is imported inside load_split. The class-discovery and counting
# helpers below are pure filesystem work, and the preparation scripts and the
# web app both use them without wanting a multi-second TensorFlow import.


def discover_classes(task: TaskConfig) -> list[str]:
    train_dir = task.dataset_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"No training data at {train_dir}.\n"
            f"Run the preparation script for the '{task.key}' task first "
            f"(see README section 4)."
        )
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(
            f"Found {len(classes)} class folder(s) in {train_dir}; need at least 2."
        )
    return classes


def count_per_class(split_dir: Path, classes: list[str]) -> dict[str, int]:
    counts = {}
    for c in classes:
        d = split_dir / c
        counts[c] = sum(1 for p in d.glob("*") if p.is_file()) if d.is_dir() else 0
    return counts


def load_split(
    task: TaskConfig,
    split: str,
    classes: list[str],
    shuffle: bool | None = None,
    augment: bool = False,
):
    """Return a batched tf.data.Dataset of (images[0-255], one_hot_labels)."""
    import tensorflow as tf
    from tensorflow import keras

    from .models import build_augmenter

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    directory = task.dataset_dir / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing split directory: {directory}")

    if shuffle is None:
        shuffle = split == "train"

    ds = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=(task.image_size, task.image_size),
        batch_size=task.batch_size,
        shuffle=shuffle,
        seed=SEED,
        interpolation="bilinear",
    )

    if augment:
        augmenter = build_augmenter(SEED)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(task: TaskConfig, classes: list[str]) -> dict[int, float]:
    """Inverse-frequency weights.

    ODIR-5K is heavily imbalanced — 'normal' and diabetes-related images vastly
    outnumber AMD. Without this the model learns to answer 'normal' for
    everything and still scores ~50% accuracy, which looks fine on paper and is
    useless in practice. This is the imbalance problem Bhati et al. [1] address.
    """
    counts = count_per_class(task.dataset_dir / "train", classes)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No training images found.")
    n = len(classes)
    weights = {}
    for i, c in enumerate(classes):
        weights[i] = (total / (n * counts[c])) if counts[c] else 1.0
    return weights


def dataset_summary(task: TaskConfig, classes: list[str]) -> dict:
    return {
        "task": task.key,
        "classes": classes,
        "counts": {s: count_per_class(task.dataset_dir / s, classes) for s in SPLITS},
    }


def save_classes(task: TaskConfig, classes: list[str]) -> None:
    task.classes_path.write_text(json.dumps(classes, indent=2))


def load_classes(task: TaskConfig) -> list[str]:
    if not task.classes_path.exists():
        raise FileNotFoundError(
            f"Class list missing: {task.classes_path}. Train the model first."
        )
    return json.loads(task.classes_path.read_text())


def labels_of(ds) -> np.ndarray:
    """Collect integer labels from a non-shuffled dataset, in order."""
    out = []
    for _, y in ds:
        out.append(np.argmax(y.numpy(), axis=1))
    return np.concatenate(out) if out else np.array([], dtype=int)
