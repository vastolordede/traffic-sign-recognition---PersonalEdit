# src/utils.py
from pathlib import Path


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"]


def count_images_by_class(data_dir: Path):
    """
    Count images in each class folder.
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    class_counts = {}

    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            image_files = [
                file
                for file in class_dir.glob("*")
                if file.suffix.lower() in IMAGE_EXTENSIONS
            ]

            class_counts[class_dir.name] = len(image_files)

    return class_counts


def print_class_distribution(data_dir: Path):
    """
    Print number of images for each class.
    """

    class_counts = count_images_by_class(data_dir)

    print(f"Dataset folder: {data_dir}")
    print("-" * 50)

    total = 0

    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
        total += count

    print("-" * 50)
    print(f"Total images: {total}")

    return class_counts


def compute_class_weight_from_train_dir(data_dir: Path, class_names):
    """
    Compute class weights from the training folder to handle class imbalance.

    Important:
    class_names must come from train_ds.class_names because
    image_dataset_from_directory maps label indices based on sorted folder names.

    Formula:
    weight_c = total_images / (num_classes * image_count_of_class_c)
    """

    class_counts = count_images_by_class(data_dir)

    total_images = sum(class_counts.values())
    num_classes = len(class_names)

    if total_images == 0:
        raise ValueError(f"No training images found in: {data_dir}")

    class_weight = {}

    for class_index, class_name in enumerate(class_names):
        class_name = str(class_name)
        count = class_counts.get(class_name, 0)

        if count == 0:
            raise ValueError(f"Class {class_name} has 0 images.")

        class_weight[class_index] = total_images / (num_classes * count)

    return class_weight