"""
==============================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - TENSORFLOW.JS EXPORT SCRIPT
==============================================================================
Converts trained Keras model into TensorFlow.js web format (model.json + shards)
Enables 100% browser-side inference with zero backend and zero API keys.
==============================================================================
"""

import os
import argparse
import tensorflow as tf
import tensorflowjs as tfjs

def convert_keras_to_tfjs(keras_model_path="models/waste_classifier.keras",
                          output_dir="models"):
    """
    Converts trained Keras model to TensorFlow.js Layers or Graph format.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading Keras model from '{keras_model_path}'...")
    model = tf.keras.models.load_model(keras_model_path)
    output_units = model.output_shape[-1]
    if output_units != 3:
        raise ValueError(f"Expected a three-class model, but the model output has {output_units} units.")

    print(f"Exporting to TensorFlow.js format at '{output_dir}'...")
    tfjs.converters.save_keras_model(
        model,
        output_dir,
        quantization_dtype_map={'uint8': '*'}  # Optional 8-bit web weight quantization
    )
    print(f"\nTensorFlow.js model generated successfully!")
    print(f"Files in '{output_dir}':")
    for f in os.listdir(output_dir):
        size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024.0
        print(f"  - {f} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Keras Model to TensorFlow.js")
    parser.add_argument("--model", type=str, default="models/waste_classifier.keras", help="Path to trained Keras model")
    parser.add_argument("--output", type=str, default="models", help="Served TensorFlow.js model directory")
    args = parser.parse_args()

    convert_keras_to_tfjs(keras_model_path=args.model, output_dir=args.output)
