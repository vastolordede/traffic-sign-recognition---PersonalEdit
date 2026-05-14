# src/config.py
from pathlib import Path

# Root project
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Output paths
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Image config
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training config
BATCH_SIZE = 32
SEED = 42

# CNN sample baseline - used by 02_cnn_baseline.ipynb
CNN_SAMPLE_MODEL_PATH = MODELS_DIR / "cnn_baseline_sample.keras"
CNN_SAMPLE_HISTORY_FIGURE_PATH = FIGURES_DIR / "cnn_baseline_sample_accuracy_loss.png"
CNN_SAMPLE_REPORT_PATH = REPORTS_DIR / "cnn_baseline_sample_classification_report.txt"
CNN_SAMPLE_CM_PATH = FIGURES_DIR / "cnn_baseline_sample_confusion_matrix.png"

# CNN full training - used by 04_cnn_full_training_demo.ipynb
CNN_FULL_MODEL_PATH = MODELS_DIR / "cnn_full_gtsrb.keras"
CNN_FULL_HISTORY_FIGURE_PATH = FIGURES_DIR / "cnn_full_gtsrb_accuracy_loss.png"
CNN_FULL_RESULT_PATH = REPORTS_DIR / "cnn_full_gtsrb_result.txt"
CNN_FULL_REPORT_PATH = REPORTS_DIR / "cnn_full_gtsrb_classification_report.txt"
CNN_FULL_CM_PATH = FIGURES_DIR / "cnn_full_gtsrb_confusion_matrix.png"

# Transfer learning baseline - used by 03_transfer_baseline.ipynb
TRANSFER_MODEL_PATH = MODELS_DIR / "transfer_mobilenetv2.keras"
TRANSFER_HISTORY_FIGURE_PATH = FIGURES_DIR / "transfer_mobilenetv2_accuracy_loss.png"
TRANSFER_REPORT_PATH = REPORTS_DIR / "transfer_mobilenetv2_classification_report.txt"
TRANSFER_CM_PATH = FIGURES_DIR / "transfer_mobilenetv2_confusion_matrix.png"

# Backward-compatible aliases
# Old code that imports CNN_MODEL_PATH will still work.
CNN_MODEL_PATH = CNN_FULL_MODEL_PATH
CNN_HISTORY_FIGURE_PATH = CNN_FULL_HISTORY_FIGURE_PATH
CNN_FINAL_RESULT_PATH = CNN_FULL_RESULT_PATH

# Report path
CNN_METHODOLOGY_PATH = ROOT_DIR / "report" / "cnn_methodology.md"