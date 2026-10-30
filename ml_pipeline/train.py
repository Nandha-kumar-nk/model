"""
==============================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - MODEL TRAINING & EVALUATION PIPELINE
==============================================================================
Architecture: MobileNetV2 (Transfer Learning for TinyML / Edge AI)
Target Output: 3 Classes (Organic, Inorganic, Reusable)
==============================================================================
"""

import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

# ---------------------------------------------------------
# Hyperparameters & Configurations
# ---------------------------------------------------------
IMG_SIZE = (224, 224)       # Input dimensions (224x224 or 160x160 for ESP32)
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASS_NAMES = ['organic', 'inorganic', 'reusable']
SOURCE_TO_TARGET = {
    'organic': 'organic',
    'non_recyclable': 'inorganic',
    'hazardous': 'inorganic',
    'recyclable': 'reusable',
}
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42

def build_data_pipelines(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Creates optimized tf.data input pipelines for train, val, and test splits.
    """
    dataset_dir = prepare_mapped_dataset(dataset_dir)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    print(f"Loading datasets from '{dataset_dir}'...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    # Performance optimization: Prefetch and Cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def prepare_mapped_dataset(dataset_dir):
    """Merge the legacy four source folders into the three model classes."""
    split_dirs = [os.path.join(dataset_dir, split) for split in ('train', 'val', 'test')]
    if all(os.path.isdir(os.path.join(split, target))
           for split in split_dirs for target in CLASS_NAMES):
        return dataset_dir

    mapped_root = os.path.join(dataset_dir, '.mapped_three_class')
    if os.path.isdir(mapped_root):
        shutil.rmtree(mapped_root)

    copied = 0
    for split in ('train', 'val', 'test'):
        for source, target in SOURCE_TO_TARGET.items():
            source_dir = os.path.join(dataset_dir, split, source)
            if not os.path.isdir(source_dir):
                continue
            target_dir = os.path.join(mapped_root, split, target)
            os.makedirs(target_dir, exist_ok=True)
            for name in os.listdir(source_dir):
                source_path = os.path.join(source_dir, name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, os.path.join(target_dir, f"{source}_{name}"))
                    copied += 1
    if copied == 0:
        raise FileNotFoundError(
            "No training images found. Add images under dataset/{train,val,test}/"
            "organic, recyclable, non_recyclable, or hazardous."
        )
    return mapped_root

def create_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, alpha=0.35):
    """
    Builds a Transfer Learning model using MobileNetV2 with data augmentation.
    Alpha=0.35 provides high accuracy with ultra-low memory suitable for ESP32-S3.
    """
    # 1. Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    # 2. Input and Preprocessing
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    # MobileNetV2 expects pixel values scaled in [-1, 1]
    x = layers.Rescaling(1./127.5, offset=-1.0, name="mobilenetv2_rescaling")(x)

    # 3. Base Feature Extractor (Pre-trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=alpha
    )
    base_model.trainable = False  # Freeze base during Phase 1

    x = base_model(x, training=False)

    # 4. Classification Head (Lightweight for TinyML)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"WasteClassifier_MobileNetV2_a{alpha}")
    return model, base_model

def train_waste_classifier(dataset_dir="dataset", output_dir="models", alpha=0.35):
    """
    Executes Phase 1 (Transfer Learning) and Phase 2 (Fine-Tuning) training.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_ds, val_ds, test_ds = build_data_pipelines(dataset_dir)

    model, base_model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES, alpha=alpha)
    model.summary()

    # ---------------------------------------------------------
    # PHASE 1: Feature Extraction Training (Base Frozen)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> PHASE 1: Training Classification Head (Base Model Frozen)")
    print("=" * 60)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    checkpoint_path = os.path.join(output_dir, "best_waste_model.keras")
    cb_list = [
        callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=cb_list
    )

    # ---------------------------------------------------------
    # PHASE 2: Fine-Tuning Top Layers of Base Model
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(">>> PHASE 2: Fine-Tuning Top Layers of MobileNetV2")
    print("=" * 60)

    base_model.trainable = True
    # Freeze bottom layers and only train the top ~30 layers
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history_phase1.epoch[-1] + 1,
        callbacks=cb_list
    )

    # Save final models in multiple formats
    model.save(os.path.join(output_dir, "waste_classifier_final.keras"))
    model.save(os.path.join(output_dir, "waste_classifier_final.h5"))
    print(f"\nTrained models successfully saved in '{output_dir}/'")

    # ---------------------------------------------------------
    # Evaluation on Unseen Test Dataset
    # ---------------------------------------------------------
    evaluate_model_performance(model, test_ds, output_dir)
    plot_training_curves(history_phase1, history_phase2, output_dir)

def evaluate_model_performance(model, test_ds, output_dir="models"):
    """
    Evaluates model on the test dataset and outputs:
    - Test Loss & Test Accuracy
    - Precision, Recall, F1-score per class
    - Confusion Matrix heatmap
    """
    print("\n" + "=" * 60)
    print(">>> EVALUATION ON UNSEEN TEST DATASET")
    print("=" * 60)

    # Extract true labels and model predictions
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # Calculate metrics
    eval_results = model.evaluate(test_ds, verbose=0)
    test_loss = eval_results[0]
    test_acc = eval_results[1]

    print(f"\n[TEST RESULTS]")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%\n")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("Classification Report:")
    print("-" * 60)
    print(report)
    print("-" * 60)

    # Save report to text file
    with open(os.path.join(output_dir, "test_evaluation_report.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[c.upper() for c in CLASS_NAMES],
                yticklabels=[c.upper() for c in CLASS_NAMES])
    plt.title('Confusion Matrix - Smart Garbage Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot saved to '{cm_path}'")

def plot_training_curves(h1, h2, output_dir="models"):
    """
    Plots training & validation accuracy and loss across both training phases.
    """
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#10b981', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#3b82f6', linewidth=2)
    plt.axvline(x=len(h1.history['accuracy'])-1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#ef4444', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#f59e0b', linewidth=2)
    plt.axvline(x=len(h1.history['loss'])-1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training history curves saved to '{plot_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smart Waste Classifier")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to split dataset directory")
    parser.add_argument("--output", type=str, default="models", help="Directory to save models and plots")
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width multiplier (0.35 for ESP32)")
    args = parser.parse_args()

    train_waste_classifier(dataset_dir=args.dataset, output_dir=args.output, alpha=args.alpha)
