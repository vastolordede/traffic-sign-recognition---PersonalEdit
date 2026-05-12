from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== Path =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_full_final.keras"

IMG_SIZE = (96, 96)

# Keras image_dataset_from_directory mặc định sort class theo string:
# ['0', '1', '10', '11', ..., '2', '20', ...]
CLASS_NAMES = sorted([str(i) for i in range(43)])
CLASS_LABELS = {
    "0": "Speed limit 20 km/h",
    "1": "Speed limit 30 km/h",
    "2": "Speed limit 50 km/h",
    "3": "Speed limit 60 km/h",
    "4": "Speed limit 70 km/h",
    "5": "Speed limit 80 km/h",
    "6": "End of speed limit 80 km/h",
    "7": "Speed limit 100 km/h",
    "8": "Speed limit 120 km/h",
    "9": "No passing",
    "10": "No passing for vehicles over 3.5 tons",
    "11": "Right-of-way at next intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Vehicles over 3.5 tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve left",
    "20": "Dangerous curve right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End of all speed and passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End of no passing by vehicles over 3.5 tons",
}

# ===== Load model =====
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


def predict_traffic_sign(image: Image.Image):
    if image is None:
        return "No image", {}

    # Convert RGB + resize giống lúc train
    image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE)

    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    pred_index = int(np.argmax(predictions))
    confidence = float(predictions[pred_index])

    class_id = CLASS_NAMES[pred_index]
    class_label = CLASS_LABELS.get(class_id, "Unknown class")

    # Top 5 kết quả
    top_indices = predictions.argsort()[-5:][::-1]
    top_predictions = {
        f"Class {CLASS_NAMES[i]} - {CLASS_LABELS.get(CLASS_NAMES[i], 'Unknown')}": float(predictions[i])
        for i in top_indices
    }

    result_text = (
        f"Predicted class: {class_id}\n"
        f"Label: {class_label}\n"
        f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)"
    )

    return result_text, top_predictions


demo = gr.Interface(
    fn=predict_traffic_sign,
    inputs=gr.Image(type="pil", label="Upload traffic sign image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Top 5 confidence"),
    ],
    title="Traffic Sign Recognition Demo",
    description="Upload a cropped traffic sign image. The CNN model will predict the traffic sign class.",
)

if __name__ == "__main__":
    demo.launch()