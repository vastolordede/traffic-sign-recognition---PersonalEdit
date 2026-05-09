import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CNN_MODEL_PATH,
    CNN_HISTORY_FIGURE_PATH,
    MODELS_DIR,
    FIGURES_DIR,
)
from src.preprocessing import get_rescaling_layer, get_data_augmentation_layer


def conv_block(filters: int, dropout_rate: float = 0.0):
    """
    CNN block: Conv2D -> BatchNorm -> Conv2D -> BatchNorm -> MaxPooling -> optional Dropout.
    """

    layers = [
        tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),
    ]

    if dropout_rate > 0:
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    return layers


def build_cnn_baseline(num_classes: int) -> tf.keras.Model:
    """
    Build an optimized CNN baseline model using Keras Sequential API.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

            get_data_augmentation_layer(),
            get_rescaling_layer(),

            *conv_block(32, dropout_rate=0.10),
            *conv_block(64, dropout_rate=0.15),
            *conv_block(128, dropout_rate=0.20),
            *conv_block(256, dropout_rate=0.25),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.40),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),

            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_baseline_optimized",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_cnn_callbacks():
    """
    Create callbacks for CNN training.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=CNN_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, model_checkpoint, reduce_lr]


def plot_training_history(history, save_path=CNN_HISTORY_FIGURE_PATH):
    """
    Plot training and validation accuracy/loss.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("CNN Baseline Optimized Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("CNN Baseline Optimized Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Training history figure saved to: {save_path}")


def train_cnn_model(train_ds, val_ds, num_classes: int, epochs: int = 30, class_weight=None):
    """
    Build and train optimized CNN baseline.
    """

    model = build_cnn_baseline(num_classes)
    callbacks = get_cnn_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    plot_training_history(history)

    return model, history


if __name__ == "__main__":
    from src.dataset import load_train_val_test_datasets

    train_ds, val_ds, test_ds, class_names = load_train_val_test_datasets()
    num_classes = len(class_names)

    model, history = train_cnn_model(
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        epochs=30,
    )

    print("Optimized CNN training completed.")
    print(f"Best model saved to: {CNN_MODEL_PATH}")