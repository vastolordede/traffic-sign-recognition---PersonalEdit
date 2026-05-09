from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEED


RAW_GTSRB_DIR = RAW_DATA_DIR / "gtsrb"

TRAIN_CSV_PATH = RAW_GTSRB_DIR / "Train.csv"
TEST_CSV_PATH = RAW_GTSRB_DIR / "Test.csv"

PROCESSED_TRAIN_DIR = PROCESSED_DATA_DIR / "train"
PROCESSED_VAL_DIR = PROCESSED_DATA_DIR / "val"
PROCESSED_TEST_DIR = PROCESSED_DATA_DIR / "test"


def reset_processed_dir():
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    PROCESSED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEST_DIR.mkdir(parents=True, exist_ok=True)


def copy_image(source_path: Path, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def prepare_train_val_split(val_size: float = 0.2):
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_rows, val_rows = train_test_split(
        train_df,
        test_size=val_size,
        random_state=SEED,
        stratify=train_df["ClassId"],
    )

    print("Copying train images...")

    for _, row in train_rows.iterrows():
        class_id = str(row["ClassId"])
        source_path = RAW_GTSRB_DIR / row["Path"]
        target_path = PROCESSED_TRAIN_DIR / class_id / source_path.name

        if source_path.exists():
            copy_image(source_path, target_path)
        else:
            print("Missing train image:", source_path)

    print("Copying validation images...")

    for _, row in val_rows.iterrows():
        class_id = str(row["ClassId"])
        source_path = RAW_GTSRB_DIR / row["Path"]
        target_path = PROCESSED_VAL_DIR / class_id / source_path.name

        if source_path.exists():
            copy_image(source_path, target_path)
        else:
            print("Missing validation image:", source_path)

    print("Train images:", len(train_rows))
    print("Validation images:", len(val_rows))


def prepare_test_set():
    test_df = pd.read_csv(TEST_CSV_PATH)

    print("Copying test images...")

    for _, row in test_df.iterrows():
        class_id = str(row["ClassId"])
        source_path = RAW_GTSRB_DIR / row["Path"]
        target_path = PROCESSED_TEST_DIR / class_id / source_path.name

        if source_path.exists():
            copy_image(source_path, target_path)
        else:
            print("Missing test image:", source_path)

    print("Test images:", len(test_df))


def prepare_gtsrb_dataset():
    print("Preparing GTSRB dataset...")
    print("Raw folder:", RAW_GTSRB_DIR)
    print("Processed folder:", PROCESSED_DATA_DIR)

    if not RAW_GTSRB_DIR.exists():
        raise FileNotFoundError(f"Raw GTSRB folder not found: {RAW_GTSRB_DIR}")

    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(f"Train.csv not found: {TRAIN_CSV_PATH}")

    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Test.csv not found: {TEST_CSV_PATH}")

    reset_processed_dir()
    prepare_train_val_split(val_size=0.2)
    prepare_test_set()

    print("Done preparing GTSRB dataset.")


if __name__ == "__main__":
    prepare_gtsrb_dataset()