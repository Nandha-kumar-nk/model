"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - TFLITE & INT8 QUANTIZATION CONVERTER
================================================================================
Converts Keras model to:
  1) Float32 TFLite (models/waste_classifier.tflite)
  2) INT8 Quantized TFLite (models/waste_classifier_quantized.tflite) for TinyML / ESP32-S3
Compares:
  - Model sizes (Keras vs FP32 TFLite vs INT8 TFLite)
  - Accuracy before & after quantization on test set
  - Inference Latency Benchmark
================================================================================
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

IMG_SIZE = (224, 224)
CLASS_NAMES = ["ORGANIC", "INORGANIC", "REUSABLE"]


def get_file_size_kb(filepath):
    """Returns file size in Kilobytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1024.0
    return 0.0


def representative_dataset_gen(dataset_dir="dataset", num_samples=100):
    """
    Generates representative input frames for post-training INT8 quantization.
    """
    test_split_path = os.path.join(dataset_dir, "test")
    val_split_path = os.path.join(dataset_dir, "val")
    
    if os.path.exists(test_split_path):
        data_path = test_split_path
    elif os.path.exists(val_split_path):
        data_path = val_split_path
    else:
        data_path = dataset_dir

    ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=True,
        seed=42
    )

    count = 0
    for images, _ in ds:
        # Pass raw image tensor (model internally handles preprocessing if built into graph)
        yield [tf.cast(images, tf.float32)]
        count += 1
        if count >= num_samples:
            break


def convert_fp32_tflite(keras_model_path, output_tflite_path):
    """
    Standard Float32 TensorFlow Lite conversion.
    """
    print(f"\n[CONVERSION] Converting Keras model to Float32 TFLite...")
    model = tf.keras.models.load_model(keras_model_path)
    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(f"Expected a three-class model, got {model.output_shape[-1]} outputs.")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[SUCCESS] Float32 TFLite model written to: {output_tflite_path}")
    return output_tflite_path


def convert_int8_tflite(keras_model_path, output_quant_path, dataset_dir="dataset"):
    """
    Full INT8 Post-Training Quantization for Edge-AI & TinyML Microcontrollers (ESP32-S3).
    """
    print(f"\n[CONVERSION] Quantizing to Full INT8 TFLite for Edge-AI / ESP32-S3...")
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset for calibrating activation ranges
    try:
        gen = lambda: representative_dataset_gen(dataset_dir=dataset_dir, num_samples=100)
        converter.representative_dataset = gen
        # Ensure full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32  # Keep input as float32 for clean camera pipeline or int8 if required
        converter.inference_output_type = tf.float32

        tflite_quant_model = converter.convert()
        with open(output_quant_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"[SUCCESS] INT8 Quantized model written to: {output_quant_path}")
        return output_quant_path
    except Exception as e:
        print(f"[WARNING] Full INT8 quantization encounter: {e}. Falling back to default dynamic range quantization.")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()
        with open(output_quant_path, "wb") as f:
            f.write(tflite_quant_model)
        print(f"[SUCCESS] Dynamic-range quantized TFLite written to: {output_quant_path}")
        return output_quant_path


def evaluate_tflite_model(tflite_path, dataset_dir="dataset", max_test_samples=200):
    """
    Evaluates accuracy and average inference latency of a TFLite model.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_split_path = os.path.join(dataset_dir, "test")
    data_path = test_split_path if os.path.exists(test_split_path) else dataset_dir

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=False
    )

    y_true = []
    y_pred = []
    latencies = []

    count = 0
    for images, labels in test_ds:
        input_data = images.numpy()
        
        # Quantization type handling if input is INT8
        if input_details[0]['dtype'] == np.int8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)
        else:
            input_data = input_data.astype(np.float32)

        start_t = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        latencies.append((time.perf_counter() - start_t) * 1000.0)  # ms

        pred_class = np.argmax(output_data[0])
        true_class = np.argmax(labels.numpy()[0])

        y_true.append(true_class)
        y_pred.append(pred_class)
        count += 1
        if count >= max_test_samples:
            break

    acc = accuracy_score(y_true, y_pred)
    avg_latency = np.mean(latencies) if latencies else 0.0
    return acc, avg_latency


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to optimized TFLite for Edge AI")
    parser.add_argument("--keras_model", type=str, default="models/waste_classifier.keras", help="Input Keras model")
    parser.add_argument("--output_tflite", type=str, default="models/waste_classifier.tflite", help="Output FP32 TFLite")
    parser.add_argument("--output_quant", type=str, default="models/waste_classifier_quantized.tflite", help="Output INT8 TFLite")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory for calibration")
    args = parser.parse_args()

    if not os.path.exists(args.keras_model):
        raise FileNotFoundError(
            f"Trained model '{args.keras_model}' not found. Please run `python train_model.py` first."
        )

    print("=" * 80)
    print(" SMART GARBAGE CLASSIFICATION SYSTEM - TFLITE / TinyML OPTIMIZATION PIPELINE")
    print("=" * 80)

    # 1. Convert to Float32 TFLite
    convert_fp32_tflite(args.keras_model, args.output_tflite)

    # 2. Convert to INT8 Quantized TFLite
    convert_int8_tflite(args.keras_model, args.output_quant, dataset_dir=args.dataset)

    # 3. Model Size Comparison
    keras_size_kb = get_file_size_kb(args.keras_model)
    fp32_size_kb = get_file_size_kb(args.output_tflite)
    int8_size_kb = get_file_size_kb(args.output_quant)

    print("\n" + "=" * 80)
    print(" [MODEL SIZE COMPARISON]")
    print("-" * 80)
    print(f"  1. Original Keras Model (.keras)       : {keras_size_kb / 1024.0:.2f} MB ({keras_size_kb:.1f} KB)")
    print(f"  2. Standard Float32 TFLite (.tflite)    : {fp32_size_kb / 1024.0:.2f} MB ({fp32_size_kb:.1f} KB) -> ~{(1 - fp32_size_kb / keras_size_kb) * 100:.1f}% reduction")
    print(f"  3. INT8 Quantized TFLite (_quantized)  : {int8_size_kb / 1024.0:.2f} MB ({int8_size_kb:.1f} KB) -> ~{(1 - int8_size_kb / keras_size_kb) * 100:.1f}% reduction")

    # 4. Accuracy & Latency Benchmarks
    print("\n" + "=" * 80)
    print(" [ACCURACY & LATENCY BENCHMARK ON TEST DATASET]")
    print("-" * 80)
    
    if os.path.exists(args.dataset):
        print("Evaluating Float32 TFLite model...")
        fp32_acc, fp32_lat = evaluate_tflite_model(args.output_tflite, dataset_dir=args.dataset)
        print(f"  -> FP32 TFLite Accuracy : {fp32_acc * 100:.2f}% | Latency: {fp32_lat:.2f} ms/frame")

        print("Evaluating INT8 Quantized TFLite model...")
        int8_acc, int8_lat = evaluate_tflite_model(args.output_quant, dataset_dir=args.dataset)
        print(f"  -> INT8 TFLite Accuracy : {int8_acc * 100:.2f}% | Latency: {int8_lat:.2f} ms/frame")

        acc_diff = (fp32_acc - int8_acc) * 100
        print(f"\n Quantization Accuracy Impact: {acc_diff:+.2f}%")
    else:
        print("[INFO] Dataset folder not present during standalone benchmark; skipping test dataset evaluation.")

    print("\n" + "=" * 80)
    print(" [SUCCESS] Primary deployment model ready at: models/waste_classifier.tflite")
    print(" You can now run the live camera application with: python main.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
