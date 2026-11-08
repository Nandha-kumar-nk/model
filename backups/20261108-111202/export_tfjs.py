"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - TFJS MODEL EXPORTER
================================================================================
Converts trained Keras model (models/waste_classifier.keras) into 
browser-compatible TensorFlow.js format (models/model.json + shard weights).
================================================================================
"""

import os
import sys
import types


def _import_tensorflowjs():
    """Load the Keras converter without loading the optional TF-DF native ops."""
    if os.name == "nt":
        sys.modules.setdefault(
            "tensorflow_decision_forests",
            types.ModuleType("tensorflow_decision_forests"),
        )
    try:
        import tensorflowjs as tfjs
        return tfjs
    except (ImportError, OSError) as error:
        if "tensorflow_decision_forests" not in str(error):
            raise

        # tensorflowjs imports its SavedModel converter at module import time.
        # The Keras conversion path does not use TF-DF, whose native Windows
        # extension may be unavailable or incompatible with the installed TF.
        sys.modules.setdefault(
            "tensorflow_decision_forests",
            types.ModuleType("tensorflow_decision_forests"),
        )
        import tensorflowjs as tfjs
        return tfjs


def _browser_compatible_model(model):
    """Remove training-only augmentation layers unsupported by TensorFlow.js."""
    import tensorflow as tf

    augmentation = model.get_layer("data_augmentation")
    if "RandomFlip" not in repr(augmentation.layers):
        return model

    inputs = tf.keras.Input(shape=model.input_shape[1:], name="input_frame")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_rescaling")(inputs)
    base_model = model.get_layer("mobilenetv2_0.35_224")
    x = base_model(x, training=False)
    for layer_name in (
        "global_avg_pool",
        "batch_norm",
        "dropout_1",
        "dense_features",
        "dropout_2",
        "waste_probabilities",
    ):
        layer = model.get_layer(layer_name)
        x = layer(x, training=False)
    return tf.keras.Model(inputs, x, name="WasteClassifier_MobileNetV2_TFJS")


def convert_keras_to_tfjs(keras_model_path="models/waste_classifier.keras", output_dir="models"):
    if not os.path.exists(keras_model_path):
        print(f"[ERROR] Keras model '{keras_model_path}' not found.")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)

    try:
        tfjs = _import_tensorflowjs()
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow.js conversion dependencies are missing. "
            "Install tensorflowjs in the project Python environment."
        ) from error

    print(f"[INFO] Loading Keras model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    output_units = model.output_shape[-1]
    if output_units != 3:
        raise ValueError(f"Expected a three-class model, but the model output has {output_units} units.")
    export_model = _browser_compatible_model(model)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Converting and exporting TensorFlow.js model to: {output_dir}")
    tfjs.converters.save_keras_model(export_model, output_dir)

    print("[SUCCESS] TensorFlow.js model successfully saved!")
    print(f"Files generated:")
    print(f" - {os.path.join(output_dir, 'model.json')}")
    print(f" - {output_dir}/group1-shard*.bin")

if __name__ == "__main__":
    convert_keras_to_tfjs()
