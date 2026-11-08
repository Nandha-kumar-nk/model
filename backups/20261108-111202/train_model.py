"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - AI MODEL TRAINING PIPELINE
================================================================================
Model: MobileNetV2 (Transfer Learning for TinyML / Edge AI Optimization)
Classes (3): ORGANIC, INORGANIC, REUSABLE
Split: 70% Training | 15% Validation | 15% Testing
Output: models/waste_classifier.keras, results/accuracy.png, results/loss.png
================================================================================
"""

import os
import shutil
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Fix seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Default Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASS_NAMES = ["ORGANIC", "INORGANIC", "REUSABLE"]
SOURCE_TO_TARGET = {
    "organic": "organic",
    "recyclable": "reusable",
    "non_recyclable": "inorganic",
    "non-recyclable": "inorganic",
    "hazardous": "inorganic",
    "inorganic": "inorganic",
    "reusable": "reusable"
}
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5


def _image_files(path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        os.path.join(root, filename)
        for root, _, files in os.walk(path)
        for filename in files
        if os.path.splitext(filename)[1].lower() in extensions
    ]


def prepare_mapped_dataset(dataset_dir):
    """Materialize the four source folders into an exact three-class dataset."""
    mapped_dir = os.path.join(dataset_dir, ".mapped_three_class")
    source_splits = ["train", "val", "test"] if all(
        os.path.isdir(os.path.join(dataset_dir, split)) for split in ["train", "val", "test"]
    ) else [None]

    total_images = 0
    for split in source_splits:
        split_root = dataset_dir if split is None else os.path.join(dataset_dir, split)
        for source_name, target_name in SOURCE_TO_TARGET.items():
            source_dir = os.path.join(split_root, source_name)
            if os.path.isdir(source_dir):
                files = _image_files(source_dir)
                total_images += len(files)
                target_dir = os.path.join(mapped_dir, target_name) if split is None else os.path.join(mapped_dir, split, target_name)
                os.makedirs(target_dir, exist_ok=True)
                for source_file in files:
                    destination = os.path.join(target_dir, f"{source_name}_{os.path.basename(source_file)}")
                    if not os.path.exists(destination):
                        shutil.copy2(source_file, destination)

    if total_images == 0:
        raise ValueError(
            f"No training images were found under '{dataset_dir}'. Add images to organic/, "
            "recyclable/, non_recyclable/, and hazardous/ (hazardous and non_recyclable merge into inorganic)."
        )
    return mapped_dir, source_splits != [None]


def check_and_prepare_dataset(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Validates dataset structure and loads data using tf.keras.utils.image_dataset_from_directory.
    Supports either:
      1) dataset/ containing subfolders: organic/, recyclable/, non_recyclable/, hazardous/ (auto 70/15/15 split)
      2) dataset/ containing pre-split: train/, val/, test/
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' not found. "
            f"Please create '{dataset_dir}/' with organic, recyclable, non_recyclable, and hazardous folders."
        )

    dataset_dir, has_train_val_test = prepare_mapped_dataset(dataset_dir)

    if has_train_val_test:
        print("[INFO] Detected pre-split dataset (train/val/test).")
        train_dir = os.path.join(dataset_dir, 'train')
        val_dir = os.path.join(dataset_dir, 'val')
        test_dir = os.path.join(dataset_dir, 'test')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode='categorical',
            class_names=[c.lower() for c in CLASS_NAMES],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=SEED
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode='categorical',
            class_names=[c.lower() for c in CLASS_NAMES],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            label_mode='categorical',
            class_names=[c.lower() for c in CLASS_NAMES],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        # Load from single directory and perform 70% Train, 15% Val, 15% Test split
        print("[INFO] Detected single dataset directory. Performing 70% Train / 15% Val / 15% Test split...")
        
        # Verify subdirectories
        found_subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        normalized_classes = [c.lower() for c in CLASS_NAMES]
        
        # Allow case-insensitive directory names
        class_mapping = {}
        for d in found_subdirs:
            d_norm = d.lower().replace("-", "_").replace(" ", "_")
            if d_norm in normalized_classes:
                class_mapping[d] = d_norm

        if len(class_mapping) < NUM_CLASSES:
            print(f"[WARNING] Expected 3 mapped class folders ({normalized_classes}), but found: {found_subdirs}")
            print(f"[INFO] Looking for matching directory structures...")

        full_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            label_mode='categorical',
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=SEED
        )
        
        total_batches = tf.data.experimental.cardinality(full_ds).numpy()
        if total_batches == 0:
            raise ValueError(f"Dataset in '{dataset_dir}' contains 0 images. Please add waste images.")

        train_size = int(0.70 * total_batches)
        val_size = int(0.15 * total_batches)
        test_size = total_batches - train_size - val_size

        if train_size == 0:
            train_size = max(1, total_batches - 2)
            val_size = 1
            test_size = 1

        train_ds = full_ds.take(train_size)
        remaining = full_ds.skip(train_size)
        val_ds = remaining.take(val_size)
        test_ds = remaining.skip(val_size)

        print(f"[INFO] Batches split -> Train: {train_size}, Validation: {val_size}, Test: {test_size}")

    # Optimize data pipeline caching & prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, alpha=0.35):
    """
    Builds MobileNetV2 Transfer Learning model with integrated Data Augmentation.
    alpha=0.35/0.50/1.0: Depth multiplier. 0.35 is optimal for microcontrollers (ESP32-S3).
    """
    # 1. Data Augmentation Pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.2, seed=SEED),
        layers.RandomZoom(0.2, seed=SEED),
        layers.RandomTranslation(0.1, 0.1, seed=SEED),
        layers.RandomContrast(0.15, seed=SEED),
    ], name="data_augmentation")

    # 2. Input Layer
    inputs = tf.keras.Input(shape=input_shape, name="input_frame")
    
    # Apply data augmentation
    augmented = data_augmentation(inputs)
    
    # Preprocessing: Scales RGB [0, 255] to [-1, 1] as required by MobileNetV2
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)

    # 3. Pretrained MobileNetV2 Base (ImageNet weights, without top classifier)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    base_model.trainable = False  # Freeze for initial training phase

    # 4. Feature Extraction & Classification Head
    x = base_model(preprocessed, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation='relu', name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="waste_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="WasteClassifier_MobileNetV2")
    return model, base_model


def plot_and_save_history(history_phase1, history_phase2, results_dir="results"):
    """
    Combines training history and generates standard publication-ready plots.
    """
    os.makedirs(results_dir, exist_ok=True)

    acc = history_phase1.history.get('accuracy', []) + (history_phase2.history.get('accuracy', []) if history_phase2 else [])
    val_acc = history_phase1.history.get('val_accuracy', []) + (history_phase2.history.get('val_accuracy', []) if history_phase2 else [])
    loss = history_phase1.history.get('loss', []) + (history_phase2.history.get('loss', []) if history_phase2 else [])
    val_loss = history_phase1.history.get('val_loss', []) + (history_phase2.history.get('val_loss', []) if history_phase2 else [])

    epochs_range = range(1, len(acc) + 1)
    split_epoch = len(history_phase1.history.get('accuracy', []))

    # 1. Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, 'g--', label='Validation Accuracy', linewidth=2)
    if history_phase2 and split_epoch > 0:
        plt.axvline(x=split_epoch, color='r', linestyle=':', label='Fine-Tuning Start')
    plt.title('Waste Classification Model - Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    acc_path = os.path.join(results_dir, "accuracy.png")
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Accuracy curve saved to: {acc_path}")

    # 2. Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, 'r--', label='Validation Loss', linewidth=2)
    if history_phase2 and split_epoch > 0:
        plt.axvline(x=split_epoch, color='k', linestyle=':', label='Fine-Tuning Start')
    plt.title('Waste Classification Model - Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Categorical Crossentropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    loss_path = os.path.join(results_dir, "loss.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Loss curve saved to: {loss_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Smart Garbage Classification Deep Learning Model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save metric plots")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--initial_epochs", type=int, default=INITIAL_EPOCHS, help="Epochs for Phase 1 (frozen base)")
    parser.add_argument("--fine_tune_epochs", type=int, default=FINE_TUNE_EPOCHS, help="Epochs for Phase 2 (fine-tuning)")
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width multiplier (e.g. 0.35 for TinyML)")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 70)
    print(" SMART GARBAGE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    print(f" TensorFlow Version: {tf.__version__}")
    print(f" Target Classes (3): {CLASS_NAMES}")
    print(f" Input Resolution : {IMG_SIZE}")
    print(f" Architecture     : MobileNetV2 (alpha={args.alpha})")
    print("=" * 70)

    # 1. Load Data Pipelines
    train_ds, val_ds, test_ds = check_and_prepare_dataset(
        dataset_dir=args.dataset,
        img_size=IMG_SIZE,
        batch_size=args.batch_size
    )

    # 2. Build Model
    model, base_model = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES,
        alpha=args.alpha
    )
    model.summary()

    # Callbacks
    model_save_path = os.path.join(args.models_dir, "waste_classifier.keras")
    checkpoint_cb = callbacks.ModelCheckpoint(
        model_save_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # 3. PHASE 1: Train Classification Head (Transfer Learning)
    print("\n" + "=" * 70)
    print(f" [PHASE 1/2] Training classification head ({args.initial_epochs} epochs, lr={INITIAL_LR})...")
    print("=" * 70)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    # 4. PHASE 2: Fine-Tuning Top Layers of MobileNetV2
    history_phase2 = None
    if args.fine_tune_epochs > 0:
        print("\n" + "=" * 70)
        print(f" [PHASE 2/2] Fine-tuning MobileNetV2 top layers ({args.fine_tune_epochs} epochs, lr={FINE_TUNE_LR})...")
        print("=" * 70)

        base_model.trainable = True
        # Fine-tune from the top layers (keep first 100 layers frozen)
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.initial_epochs + args.fine_tune_epochs,
            initial_epoch=history_phase1.epoch[-1] + 1,
            callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
        )

    # 5. Plot and save training history
    plot_and_save_history(history_phase1, history_phase2, results_dir=args.results_dir)

    # 6. Evaluate on Test Dataset
    print("\n" + "=" * 70)
    print(" [EVALUATION] Testing best model on unseen test dataset...")
    print("=" * 70)

    # Load best saved weights
    best_model = tf.keras.models.load_model(model_save_path)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
    print(f"\n >>> FINAL TEST ACCURACY: {test_acc * 100:.2f}% | TEST LOSS: {test_loss:.4f} <<< \n")

    print(f"[SUCCESS] Trained model saved to: {model_save_path}")
    print("Next step: Run `python evaluate_model.py` for full confusion matrix and metrics.")
    print("Then run: `python convert_to_tflite.py` for TinyML quantization.")


if __name__ == "__main__":
    main()
