# src/evaluate.py
import tensorflow as tf

from src.config import (
    CNN_FULL_MODEL_PATH,
    CNN_FULL_RESULT_PATH,
    REPORTS_DIR,
)


def load_model_from_path(model_path=CNN_FULL_MODEL_PATH):
    """
    Load a saved Keras model from a given path.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the model first."
        )

    return tf.keras.models.load_model(model_path)


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.
    """

    test_loss, test_accuracy = model.evaluate(test_ds)

    return test_loss, test_accuracy


def save_evaluation_result(
    test_loss,
    test_accuracy,
    model_name="CNN Full GTSRB",
    model_path=CNN_FULL_MODEL_PATH,
    save_path=CNN_FULL_RESULT_PATH,
):
    """
    Save final model result to a text file.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""{model_name} Final Result

Model:
- {model_name}

Saved model:
- {model_path}

Final evaluation:
- Test loss: {test_loss:.4f}
- Test accuracy: {test_accuracy:.4f}
- Test accuracy percent: {test_accuracy * 100:.2f}%

Note:
- The model was saved using ModelCheckpoint.
- The result should be interpreted together with classification report and confusion matrix.
"""

    save_path.write_text(content, encoding="utf-8")

    print(f"Final result saved to: {save_path}")


def evaluate_cnn_final(test_ds):
    """
    Load best CNN full model, evaluate it on test dataset, and save result.
    """

    model = load_model_from_path(CNN_FULL_MODEL_PATH)
    test_loss, test_accuracy = evaluate_model(model, test_ds)

    save_evaluation_result(
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        model_name="CNN Full GTSRB",
        model_path=CNN_FULL_MODEL_PATH,
        save_path=CNN_FULL_RESULT_PATH,
    )

    return model, test_loss, test_accuracy