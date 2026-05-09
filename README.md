# Traffic Sign Recognition

Dự án xây dựng hệ thống nhận diện biển báo giao thông bằng Computer Vision và Deep Learning.

Mục tiêu của project là xây dựng một pipeline hoàn chỉnh:

```text
dataset -> preprocessing -> training -> evaluation -> demo
```

Project sử dụng dataset GTSRB, TensorFlow/Keras để xây dựng và huấn luyện mô hình CNN, scikit-learn để đánh giá kết quả, và Streamlit/Gradio cho phần demo.

---

## 1. Project Overview

Bài toán của project là phân loại ảnh biển báo giao thông vào 43 lớp khác nhau.

Dataset sau khi xử lý được chia thành 3 tập:

| Dataset | Số ảnh | Số class |
|---|---:|---:|
| Train | 31,367 | 43 |
| Validation | 7,842 | 43 |
| Test | 12,630 | 43 |

Mỗi class được lưu trong một thư mục riêng theo định dạng:

```text
data/processed/train/0/
data/processed/train/1/
...
data/processed/train/42/
```

---

## 2. Main Technologies

Project sử dụng các thư viện chính:

```text
Python
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Scikit-learn
OpenCV
Pillow
Jupyter Notebook
Streamlit
Gradio
```

Khuyến nghị dùng Python 3.10 hoặc Python 3.11.

Không nên dùng Python 3.13 vì TensorFlow có thể gặp lỗi tương thích.

---

## 3. Project Structure

```text
traffic-sign-recognition/
│
├── app/
│   └── .gitkeep
│
├── data/
│   ├── raw/
│   │   └── gtsrb/
│   │       ├── Train.csv
│   │       ├── Test.csv
│   │       └── ảnh gốc của dataset
│   │
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── sample_images/
│
├── models/
│   └── cnn_baseline.keras
│
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_baseline.ipynb
│   ├── 03_experiment_optimization.ipynb
│   └── 04_final_evaluation_demo.ipynb
│
├── report/
│   └── cnn_methodology.md
│
├── results/
│   ├── figures/
│   │   └── cnn_accuracy_loss.png
│   ├── predictions/
│   └── reports/
│       └── cnn_final_result.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── prepare_gtsrb.py
│   ├── preprocessing.py
│   ├── train_cnn.py
│   └── utils.py
│
├── check_env.py
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
└── README.md
```

---

## 4. Environment Setup

### 4.1. Create virtual environment

Trên Linux/WSL:

```bash
python3.10 -m venv tf-gpu-venv
source tf-gpu-venv/bin/activate
```

Hoặc nếu máy đã dùng đúng Python version:

```bash
python -m venv tf-gpu-venv
source tf-gpu-venv/bin/activate
```

Trên Windows PowerShell:

```powershell
python -m venv tf-gpu-venv
.\tf-gpu-venv\Scripts\Activate.ps1
```

---

### 4.2. Install dependencies

Cài các thư viện chính:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu muốn cài đúng toàn bộ version đã khóa:

```bash
pip install -r requirements-lock.txt
```

---

### 4.3. Check environment

Chạy file kiểm tra môi trường:

```bash
python check_env.py
```

File này kiểm tra các thư viện chính:

```text
numpy
pandas
matplotlib
scikit-learn
tensorflow
opencv-python
pillow
streamlit
gradio
```

Đồng thời kiểm tra TensorFlow và GPU:

```python
tf.config.list_physical_devices("GPU")
```

Nếu GPU được nhận, output sẽ có dạng:

```text
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Nếu không có GPU, project vẫn có thể chạy bằng CPU nhưng thời gian train sẽ lâu hơn.

---

### 4.4. Add Jupyter kernel

Nếu dùng VSCode Notebook hoặc Jupyter Notebook, cài kernel cho virtual environment:

```bash
python -m ipykernel install --user --name tf-gpu-venv --display-name "tf-gpu-venv"
```

Sau đó mở notebook và chọn kernel:

```text
tf-gpu-venv
```

---

## 5. Dataset Preparation

Dataset gốc GTSRB được đặt tại:

```text
data/raw/gtsrb/
```

Thư mục này cần chứa:

```text
Train.csv
Test.csv
ảnh gốc của dataset
```

Để tạo dữ liệu đã xử lý, chạy:

```bash
python -m src.prepare_gtsrb
```

Script này sẽ:

```text
1. Đọc Train.csv và Test.csv
2. Chia tập train thành train/validation theo tỉ lệ 80/20
3. Copy ảnh vào từng thư mục class tương ứng
4. Tạo cấu trúc data/processed/train, data/processed/val, data/processed/test
```

Kết quả sau khi xử lý:

```text
data/processed/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── ...
│
├── val/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   └── ...
│
└── test/
    ├── 0/
    ├── 1/
    ├── 2/
    └── ...
```

---

## 6. Configuration

Các cấu hình chính nằm trong:

```text
src/config.py
```

Một số cấu hình quan trọng:

```python
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
SEED = 42

CNN_MODEL_PATH = MODELS_DIR / "cnn_baseline.keras"
CNN_HISTORY_FIGURE_PATH = FIGURES_DIR / "cnn_accuracy_loss.png"
CNN_FINAL_RESULT_PATH = REPORTS_DIR / "cnn_final_result.txt"
CNN_METHODOLOGY_PATH = ROOT_DIR / "report" / "cnn_methodology.md"
```

Ý nghĩa:

```text
IMG_HEIGHT, IMG_WIDTH:
- Resize toàn bộ ảnh về kích thước 96x96.

BATCH_SIZE:
- Số ảnh trong mỗi batch khi train/evaluate.

SEED:
- Giúp chia dữ liệu và shuffle có tính ổn định.

CNN_MODEL_PATH:
- Nơi lưu best model bằng ModelCheckpoint.

CNN_HISTORY_FIGURE_PATH:
- Nơi lưu biểu đồ accuracy/loss sau khi train.

CNN_FINAL_RESULT_PATH:
- Nơi lưu kết quả đánh giá cuối cùng.
```

---

## 7. Notebook Workflow

Project được chia thành nhiều notebook theo từng giai đoạn.

---

### 7.1. 00_environment_check.ipynb

Notebook kiểm tra môi trường chạy project.

Nội dung chính:

```text
- Kiểm tra Python version
- Kiểm tra TensorFlow version
- Kiểm tra GPU
- Kiểm tra các thư viện chính
```

Ví dụ:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image

print("Environment OK")
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
```

---

### 7.2. 01_data_exploration.ipynb

Notebook khám phá dữ liệu.

Nội dung chính:

```text
- Kiểm tra đường dẫn train/val/test
- Đếm số lượng ảnh theo từng class
- Kiểm tra số lượng class
- Hiển thị ảnh mẫu
```

Kết quả hiện tại:

```text
Train images: 31,367
Validation images: 7,842
Test images: 12,630
Number of classes: 43
```

---

### 7.3. 02_cnn_baseline.ipynb

Notebook xây dựng và train mẫu CNN baseline.

Vai trò của notebook này là kiểm tra pipeline huấn luyện ban đầu, chưa phải model final chính thức.

Nội dung chính:

```text
- Load train/validation/test dataset
- Optimize dataset bằng cache và prefetch
- Build CNN baseline model
- Kiểm tra input/output shape
- Train thử bằng model.fit()
- Train baseline model với callbacks
- Lưu best model
- Vẽ biểu đồ accuracy/loss
- Evaluate nhanh trên small test set
```

Callbacks được sử dụng:

```text
EarlyStopping
ModelCheckpoint
ReduceLROnPlateau
```

Output chính:

```text
models/cnn_baseline.keras
results/figures/cnn_accuracy_loss.png
```

Lưu ý:

```text
Notebook 02 chỉ được xem là baseline và training prototype.
Kết quả trong notebook này dùng để chứng minh pipeline hoạt động đúng.
Model final chính thức sẽ được train ở notebook 03.
```

---

### 7.4. 03_experiment_optimization.ipynb

Notebook dành cho experiment và optimization.

Vai trò:

```text
- Thử nghiệm các cấu hình CNN
- So sánh learning rate, dropout, class_weight
- So sánh train subset và full dataset
- Chọn cấu hình tốt nhất
- Train final optimized model
```

Output dự kiến:

```text
models/cnn_final_optimized.keras
results/figures/final_training_curve.png
experiment_summary.md
```

Notebook này là nơi train model chính thức của project.

---

### 7.5. 04_final_evaluation_demo.ipynb

Notebook dành cho final evaluation và demo.

Vai trò:

```text
- Load model tốt nhất từ notebook 03
- Evaluate full test set
- Tạo classification_report
- Tạo confusion_matrix
- Test ảnh ngoài
- Chuẩn bị demo bằng Streamlit hoặc Gradio
```

Output dự kiến:

```text
results/reports/classification_report.txt
results/figures/confusion_matrix.png
demo final
```

Notebook 04 không nên train lại model từ đầu. Notebook này chỉ load model đã train, đánh giá và demo.

---

## 8. CNN Baseline Model

Model CNN baseline được định nghĩa trong:

```text
src/train_cnn.py
```

Kiến trúc chính:

```text
Input image: 96x96x3

Data Augmentation
Rescaling

Conv Block 32 filters
Conv Block 64 filters
Conv Block 128 filters
Conv Block 256 filters

GlobalAveragePooling2D

Dense 256
BatchNormalization
Dropout 0.40

Dense 128
Dropout 0.30

Dense num_classes
Softmax
```

Loss function:

```text
sparse_categorical_crossentropy
```

Optimizer:

```text
Adam learning_rate=0.001
```

Metric:

```text
accuracy
```

---

## 9. Data Preprocessing

Tiền xử lý ảnh được định nghĩa trong:

```text
src/preprocessing.py
```

Các bước chính:

```text
1. Rescaling pixel từ [0, 255] về [0, 1]
2. Data augmentation để tăng khả năng tổng quát của model
```

Data augmentation được sử dụng:

```text
RandomRotation
RandomZoom
RandomTranslation
RandomContrast
```

Không sử dụng horizontal flip vì một số biển báo có hướng trái/phải. Nếu lật ngang ảnh, ý nghĩa của biển báo có thể bị thay đổi.

---

## 10. Source Code Modules

### 10.1. src/config.py

Chứa toàn bộ đường dẫn và cấu hình chính của project.

Bao gồm:

```text
Dataset paths
Output paths
Image size
Batch size
Seed
Model path
Report path
```

---

### 10.2. src/dataset.py

Chứa hàm load dataset:

```python
load_train_val_test_datasets()
```

Hàm này sử dụng:

```python
tf.keras.utils.image_dataset_from_directory
```

để load dữ liệu từ:

```text
data/processed/train
data/processed/val
data/processed/test
```

Ngoài ra có hàm:

```python
optimize_dataset()
```

dùng:

```text
cache()
prefetch()
```

để tăng hiệu năng input pipeline.

---

### 10.3. src/preprocessing.py

Chứa các layer tiền xử lý ảnh:

```text
Rescaling
Data augmentation
```

Các augmentation được dùng:

```text
RandomRotation
RandomZoom
RandomTranslation
RandomContrast
```

---

### 10.4. src/train_cnn.py

Chứa toàn bộ logic xây dựng và train CNN:

```text
conv_block()
build_cnn_baseline()
get_cnn_callbacks()
plot_training_history()
train_cnn_model()
```

File này chịu trách nhiệm:

```text
- Tạo model CNN
- Compile model
- Tạo callbacks
- Train model
- Lưu best model
- Lưu biểu đồ accuracy/loss
```

---

### 10.5. src/evaluate.py

Chứa các hàm đánh giá model:

```text
load_cnn_model()
evaluate_model()
save_cnn_final_result()
evaluate_cnn_final()
```

Output đánh giá được lưu tại:

```text
results/reports/cnn_final_result.txt
```

---

### 10.6. src/prepare_gtsrb.py

Chứa script chuẩn bị dataset GTSRB.

Script này:

```text
- Đọc Train.csv
- Chia train/validation bằng train_test_split
- Stratify theo ClassId
- Đọc Test.csv
- Copy ảnh vào data/processed
```

---

### 10.7. src/utils.py

Chứa hàm hỗ trợ kiểm tra dữ liệu:

```text
count_images_by_class()
print_class_distribution()
```

---

## 11. Training Pipeline

Pipeline train model cơ bản:

```text
1. Load dataset từ data/processed
2. Resize ảnh về 96x96
3. Batch ảnh với batch size = 32
4. Cache và prefetch dataset
5. Build CNN model
6. Compile model bằng Adam optimizer
7. Train bằng model.fit()
8. Dùng callbacks để kiểm soát quá trình train
9. Lưu best model bằng ModelCheckpoint
10. Vẽ accuracy/loss curve
```

Callbacks:

```text
EarlyStopping:
- Dừng train nếu val_loss không cải thiện sau một số epoch.
- restore_best_weights=True để lấy lại weights tốt nhất.

ModelCheckpoint:
- Lưu model tốt nhất dựa trên val_accuracy.

ReduceLROnPlateau:
- Giảm learning rate nếu val_loss không cải thiện.
```

---

## 12. Evaluation Pipeline

Pipeline evaluate model:

```text
1. Load model đã train từ models/
2. Load test dataset từ data/processed/test
3. Evaluate model bằng model.evaluate()
4. Ghi test loss và test accuracy
5. Tạo classification report
6. Tạo confusion matrix
7. Lưu kết quả vào results/
```

Hiện tại `src/evaluate.py` đã có các hàm cơ bản:

```text
load_cnn_model()
evaluate_model()
save_cnn_final_result()
evaluate_cnn_final()
```

Các phần classification report và confusion matrix sẽ được hoàn thiện ở notebook 04.

---

## 13. Current Team Task Division

### Week 2 - Training + Evaluation

| Member | Feature | Main Task | Deliverable | Status |
|---|---|---|---|---|
| Đặng | CNN Baseline + Training Prototype | Train mẫu CNN baseline và kiểm tra pipeline training | `02_cnn_baseline.ipynb`, `cnn_baseline.keras`, `cnn_accuracy_loss.png` | Đã hoàn thành |
| Sang | CNN Optimization Experiments | Thử nghiệm các cấu hình CNN | `03_experiment_optimization.ipynb`, `experiment_summary.md` | Chưa hoàn thành |
| Thảo | Evaluation Functions | Hoàn thiện hàm đánh giá model | evaluation utilities, report template | Chưa hoàn thành |

---

### Week 3 - Final Result + Paper

| Member | Feature | Main Task | Deliverable | Status |
|---|---|---|---|---|
| Đặng | Baseline Cleanup + Handoff | Dọn notebook 02 và bàn giao baseline training pipeline | cleaned `02_cnn_baseline.ipynb` | Đã hoàn thành |
| Sang | Final Optimized CNN Training + Methodology | Train final model và viết methodology | final model, experiment table, methodology section | Chưa hoàn thành |
| Thảo | Final Evaluation + Demo | Evaluate model cuối và hoàn thiện demo | classification report, confusion matrix, demo final | Chưa hoàn thành |

---

## 14. Git Ignore Policy

Project không push dataset thật, model lớn, cache hoặc file kết quả sinh tự động.

Các file/thư mục bị ignore:

```text
data/raw/**
data/processed/**
models/*.keras
models/*.h5
results/figures/*
results/reports/*
results/predictions/*
__pycache__/
.ipynb_checkpoints/
.venv/
venv/
env/
.vscode/
.idea/
```

Các thư mục vẫn được giữ bằng `.gitkeep`.

Lý do:

```text
- Dataset thường rất lớn
- Model .keras có thể nặng
- Result có thể sinh lại khi chạy notebook
- Virtual environment không nên push lên GitHub
- Cache Python và Jupyter không cần đưa lên repository
```

Nếu muốn push một ảnh kết quả cụ thể như `cnn_accuracy_loss.png`, có thể thêm exception vào `.gitignore`:

```gitignore
!results/figures/cnn_accuracy_loss.png
```

Nếu muốn push model `.keras` lên GitHub, cần bỏ hoặc sửa dòng:

```gitignore
models/*.keras
```

Tuy nhiên, khuyến nghị không push model lớn trực tiếp lên GitHub. Có thể lưu model trên Google Drive hoặc nền tảng lưu trữ riêng.

---

## 15. How to Run

### Step 1: Clone project

```bash
git clone <repository-url>
cd traffic-sign-recognition
```

---

### Step 2: Create environment

Trên Linux/WSL:

```bash
python3.10 -m venv tf-gpu-venv
source tf-gpu-venv/bin/activate
```

Trên Windows PowerShell:

```powershell
python -m venv tf-gpu-venv
.\tf-gpu-venv\Scripts\Activate.ps1
```

---

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Hoặc dùng lock file:

```bash
pip install -r requirements-lock.txt
```

---

### Step 4: Check environment

```bash
python check_env.py
```

---

### Step 5: Prepare dataset

Đặt dataset GTSRB vào:

```text
data/raw/gtsrb/
```

Sau đó chạy:

```bash
python -m src.prepare_gtsrb
```

---

### Step 6: Run notebooks

Mở project bằng VSCode hoặc Jupyter Notebook.

Chạy theo thứ tự:

```text
00_environment_check.ipynb
01_data_exploration.ipynb
02_cnn_baseline.ipynb
03_experiment_optimization.ipynb
04_final_evaluation_demo.ipynb
```

---

## 16. Current Progress

Đã hoàn thành:

```text
- Tạo cấu trúc project
- Cấu hình môi trường Python/TensorFlow
- Chuẩn bị dataset train/val/test
- Kiểm tra phân bố dữ liệu
- Xây dựng CNN baseline
- Train mẫu CNN baseline
- Lưu model baseline
- Lưu biểu đồ training accuracy/loss
```

Đang tiếp tục:

```text
- Experiment optimization
- Final model training
- Final evaluation
- Confusion matrix
- Classification report
- Demo nhận diện ảnh
- Paper/report
```

---

## 17. Notes

Notebook 02 chỉ được xem là baseline và training prototype.

Model final chính thức sẽ được train trong notebook 03.

Notebook 04 sẽ load model final từ notebook 03 để đánh giá trên full test set và làm demo nhận diện ảnh.

Dataset thật, model lớn, virtual environment và file cache không được push trực tiếp lên GitHub.

Các thư mục rỗng cần giữ trong repository được quản lý bằng `.gitkeep`.