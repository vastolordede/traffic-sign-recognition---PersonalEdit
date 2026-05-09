# src/utils.py
from pathlib import Path


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
            image_files = list(class_dir.glob("*"))
            image_files = [
                file
                for file in image_files
                if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
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