import tensorflow as tf

from src.config import CNN_MODEL_PATH, CNN_FINAL_RESULT_PATH, REPORTS_DIR


def load_cnn_model(model_path=CNN_MODEL_PATH):
    """
    Load the best CNN baseline model saved by ModelCheckpoint.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the CNN model first in Week 2."
        )

    model = tf.keras.models.load_model(model_path)

    return model


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.
    """

    test_loss, test_accuracy = model.evaluate(test_ds)

    return test_loss, test_accuracy


def save_cnn_final_result(test_loss, test_accuracy, save_path=CNN_FINAL_RESULT_PATH):
    """
    Save final CNN baseline result to text file.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    content = f"""CNN Baseline Final Result

Model:
- CNN Baseline using TensorFlow/Keras Sequential API

Saved model:
- models/cnn_baseline.keras

Final evaluation:
- Test loss: {test_loss:.4f}
- Test accuracy: {test_accuracy:.4f}
- Test accuracy percent: {test_accuracy * 100:.2f}%

Note:
- The model was saved using ModelCheckpoint during Week 2.
- This result is used as the CNN baseline for comparison with transfer learning models.
"""

    save_path.write_text(content, encoding="utf-8")

    print(f"Final result saved to: {save_path}")


def evaluate_cnn_final(test_ds):
    """
    Load best CNN model, evaluate on test dataset, and save final result.
    """

    model = load_cnn_model()
    test_loss, test_accuracy = evaluate_model(model, test_ds)
    save_cnn_final_result(test_loss, test_accuracy)

    return model, test_loss, test_accuracy