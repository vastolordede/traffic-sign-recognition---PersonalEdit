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

# Model config
CNN_MODEL_PATH = MODELS_DIR / "cnn_baseline.keras"
CNN_HISTORY_FIGURE_PATH = FIGURES_DIR / "cnn_accuracy_loss.png"
CNN_FINAL_RESULT_PATH = REPORTS_DIR / "cnn_final_result.txt"
CNN_METHODOLOGY_PATH = ROOT_DIR / "report" / "cnn_methodology.md"