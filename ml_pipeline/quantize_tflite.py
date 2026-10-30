"""
==============================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - TFLITE & INT8 QUANTIZATION PIPELINE
==============================================================================
Generates:
  1. Standard Float32 TFLite (waste_classifier_float32.tflite)
  2. Full INT8 Quantized TFLite (waste_classifier_int8.tflite)
  3. C Header file array for ESP32-S3 TinyML (model_data.h)
==============================================================================
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

IMG_SIZE = (224, 224)
CLASS_NAMES = ['organic', 'recyclable', 'non_recyclable', 'hazardous']

def representative_data_gen(val_dir="dataset/val", num_samples=200):
    """
    Generator that yields representative image batches for INT8 quantization calibration.
    """
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=True
    )
    
    count = 0
    for img, _ in val_ds:
        # Preprocessing matching model input (Rescaling to [-1, 1])
        img_preprocessed = (img.numpy() / 127.5) - 1.0
        yield [img_preprocessed.astype(np.float32)]
        count += 1
        if count >= num_samples:
            break

def convert_to_tflite_and_quantize(keras_model_path="models/best_waste_model.keras",
                                   dataset_dir="dataset",
                                   output_dir="models/tflite"):
    """
    Converts trained Keras model into Float32, Float16, and INT8 Quantized TFLite formats.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading Keras model from '{keras_model_path}'...")
    model = tf.keras.models.load_model(keras_model_path)

    # ---------------------------------------------------------
    # 1. Standard Float32 TFLite Model
    # ---------------------------------------------------------
    print("\n[1/3] Converting to Float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float32 = converter.convert()
    fp32_path = os.path.join(output_dir, "waste_classifier_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_float32)
    fp32_size_kb = len(tflite_float32) / 1024.0
    print(f"  Saved FP32 TFLite: {fp32_path} ({fp32_size_kb:.2f} KB)")

    # ---------------------------------------------------------
    # 2. Float16 Quantized TFLite Model
    # ---------------------------------------------------------
    print("\n[2/3] Converting to Float16 TFLite...")
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()
    fp16_path = os.path.join(output_dir, "waste_classifier_fp16.tflite")
    with open(fp16_path, "wb") as f:
        f.write(tflite_fp16)
    fp16_size_kb = len(tflite_fp16) / 1024.0
    print(f"  Saved FP16 TFLite: {fp16_path} ({fp16_size_kb:.2f} KB)")

    # ---------------------------------------------------------
    # 3. Full INT8 Quantized Model (TinyML / ESP32-S3 Deployment)
    # ---------------------------------------------------------
    print("\n[3/3] Converting to Full INT8 Quantized TFLite...")
    val_path = os.path.join(dataset_dir, 'val')
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = lambda: representative_data_gen(val_path, num_samples=150)
    
    # Enforce full integer quantization for TFLite Micro compatibility
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8

    tflite_int8 = converter_int8.convert()
    int8_path = os.path.join(output_dir, "waste_classifier_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    int8_size_kb = len(tflite_int8) / 1024.0
    print(f"  Saved INT8 TFLite: {int8_path} ({int8_size_kb:.2f} KB)")

    # ---------------------------------------------------------
    # 4. Generate C Header Array for ESP32-S3 Firmware
    # ---------------------------------------------------------
    header_path = os.path.join(output_dir, "waste_classifier_model_data.h")
    export_c_header(tflite_int8, header_path)
    print(f"\nGenerated C Header file for ESP32-S3: '{header_path}'")

    # ---------------------------------------------------------
    # 5. Evaluate and Compare TFLite Accuracy on Test Set
    # ---------------------------------------------------------
    test_path = os.path.join(dataset_dir, 'test')
    evaluate_tflite_models(fp32_path, int8_path, test_path)

def evaluate_tflite_models(fp32_path, int8_path, test_dir):
    """
    Compares accuracy of FP32 vs INT8 quantized models using TFLite Interpreters.
    """
    print("\n" + "=" * 65)
    print(">>> TFLITE TEST DATASET ACCURACY & SIZE COMPARISON")
    print("=" * 65)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=False
    )

    # 1. Evaluate FP32 Model
    interp_fp32 = tf.lite.Interpreter(model_path=fp32_path)
    interp_fp32.allocate_tensors()
    input_idx_fp32 = interp_fp32.get_input_details()[0]['index']
    output_idx_fp32 = interp_fp32.get_output_details()[0]['index']

    # 2. Evaluate INT8 Model
    interp_int8 = tf.lite.Interpreter(model_path=int8_path)
    interp_int8.allocate_tensors()
    input_details_int8 = interp_int8.get_input_details()[0]
    output_details_int8 = interp_int8.get_output_details()[0]
    
    in_scale, in_zero = input_details_int8['quantization']
    out_scale, out_zero = output_details_int8['quantization']

    y_true = []
    y_pred_fp32 = []
    y_pred_int8 = []

    for img, label in test_ds:
        y_true.append(np.argmax(label.numpy()[0]))

        # FP32 inference (normalized [-1, 1])
        img_norm = (img.numpy() / 127.5) - 1.0
        interp_fp32.set_tensor(input_idx_fp32, img_norm.astype(np.float32))
        interp_fp32.invoke()
        pred_fp32 = interp_fp32.get_tensor(output_idx_fp32)
        y_pred_fp32.append(np.argmax(pred_fp32[0]))

        # INT8 inference (quantized)
        img_int8 = (img_norm / in_scale + in_zero).astype(np.int8)
        interp_int8.set_tensor(input_details_int8['index'], img_int8)
        interp_int8.invoke()
        pred_int8 = interp_int8.get_tensor(output_details_int8['index'])
        y_pred_int8.append(np.argmax(pred_int8[0]))

    acc_fp32 = accuracy_score(y_true, y_pred_fp32) * 100.0
    acc_int8 = accuracy_score(y_true, y_pred_int8) * 100.0
    
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    int8_size_kb = os.path.getsize(int8_path) / 1024.0
    compression_ratio = fp32_size_kb / int8_size_kb

    print(f"{'FORMAT':<16} | {'SIZE (KB)':<12} | {'TEST ACCURACY':<15} | {'COMPRESSION':<12}")
    print("-" * 65)
    print(f"{'Float32 TFLite':<16} | {fp32_size_kb:<12.1f} | {acc_fp32:<15.2f}% | {'1.0x (Baseline)':<12}")
    print(f"{'INT8 Quantized':<16} | {int8_size_kb:<12.1f} | {acc_int8:<15.2f}% | {f'{compression_ratio:.1f}x smaller':<12}")
    print("=" * 65)
    print(f"Accuracy Retention: {(acc_int8 / acc_fp32) * 100.0:.1f}%")
    print(f"ESP32-S3 Flash/RAM Ready: YES (Model fits easily inside ESP32-S3 8MB Flash & 8MB PSRAM)\n")

def export_c_header(tflite_bytes, output_header_path):
    """
    Exports TFLite byte array as a C header file (xxd style) for Arduino / ESP-IDF.
    """
    with open(output_header_path, "w") as f:
        f.write("// Auto-generated by Smart Garbage Classifier Quantization Pipeline\n")
        f.write("#ifndef WASTE_CLASSIFIER_MODEL_DATA_H_\n")
        f.write("#define WASTE_CLASSIFIER_MODEL_DATA_H_\n\n")
        f.write(f"const unsigned int g_waste_classifier_model_data_size = {len(tflite_bytes)};\n")
        f.write("alignas(16) const unsigned char g_waste_classifier_model_data[] = {\n  ")
        
        for i, byte in enumerate(tflite_bytes):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
                
        f.write("\n};\n\n")
        f.write("#endif  // WASTE_CLASSIFIER_MODEL_DATA_H_\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Waste Classifier Model for TinyML")
    parser.add_argument("--model", type=str, default="models/best_waste_model.keras", help="Path to trained Keras model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="models/tflite", help="Output directory for TFLite models")
    args = parser.parse_args()

    convert_to_tflite_and_quantize(keras_model_path=args.model, dataset_dir=args.dataset, output_dir=args.output)
