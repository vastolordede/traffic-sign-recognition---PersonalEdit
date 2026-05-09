import sys

print("=" * 60)
print("PYTHON VERSION")
print("=" * 60)
print(sys.version)

packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("tensorflow", "tensorflow"),
    ("opencv-python", "cv2"),
    ("pillow", "PIL"),
    ("streamlit", "streamlit"),
    ("gradio", "gradio"),
]

print("\nMAIN LIBRARIES")
print("=" * 60)

for package_name, import_name in packages:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "version unknown")
        print(f"OK: {package_name} - {version}")
    except Exception as e:
        print(f"ERROR: {package_name} - {e}")

print("\nSKLEARN METRICS")
print("=" * 60)

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )
    print("OK: sklearn.metrics")
except Exception as e:
    print(f"ERROR: sklearn.metrics - {e}")

print("\nKERAS IMAGE PIPELINE")
print("=" * 60)

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import image_dataset_from_directory
    print("OK: ImageDataGenerator")
    print("OK: image_dataset_from_directory")
except Exception as e:
    print(f"ERROR: Keras image pipeline - {e}")

print("\nTENSORFLOW DEVICE")
print("=" * 60)

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print(f"ERROR: TensorFlow device check - {e}")