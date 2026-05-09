import tensorflow as tf


def get_rescaling_layer():
    """
    Normalize pixel values from [0, 255] to [0, 1].
    """

    return tf.keras.layers.Rescaling(1.0 / 255)


def get_data_augmentation_layer():
    """
    Data augmentation for traffic sign classification.

    Note:
    Do not use horizontal flip because some traffic signs are direction-sensitive.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )