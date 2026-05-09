# src/dataset.py
import tensorflow as tf

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
)


def load_train_val_test_datasets():
    """
    Load train/validation/test datasets from data/processed directory.

    Expected folder structure:
    data/processed/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, test_ds, class_names


def optimize_dataset(dataset):
    """
    Improve input pipeline performance using cache and prefetch.
    """

    AUTOTUNE = tf.data.AUTOTUNE

    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset