"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - MODEL EVALUATION & METRICS
================================================================================
Evaluates trained Keras/TFLite model on the test dataset.
Generates:
  - Classification Report (Precision, Recall, F1-Score per class)
  - Confusion Matrix Plot -> saved to results/confusion_matrix.png
================================================================================
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["ORGANIC", "INORGANIC", "REUSABLE"]


def load_test_dataset(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Loads test split from dataset directory.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    from train_model import prepare_mapped_dataset
    dataset_dir, has_split = prepare_mapped_dataset(dataset_dir)
    test_split_path = os.path.join(dataset_dir, "test")
    if os.path.exists(test_split_path):
        print(f"[INFO] Loading dedicated test set from '{test_split_path}'...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_split_path,
            label_mode='categorical',
            class_names=[c.lower() for c in CLASS_NAMES],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        print(f"[INFO] Extracting 15% test split from '{dataset_dir}'...")
        full_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            label_mode='categorical',
            class_names=[c.lower() for c in CLASS_NAMES],
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )
        total_batches = tf.data.experimental.cardinality(full_ds).numpy()
        train_size = int(0.70 * total_batches)
        val_size = int(0.15 * total_batches)
        test_ds = full_ds.skip(train_size + val_size)

    return test_ds


def evaluate_keras_model(model_path, test_ds):
    """
    Runs full inference and returns true labels and predicted probabilities.
    """
    print(f"[INFO] Loading Keras model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels.numpy())

    y_true_one_hot = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return y_true, y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred, class_names, output_path="results/confusion_matrix.png"):
    """
    Plots and saves a professional confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # handle any zero rows

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor='white',
        annot_kws={"size": 13, "fontweight": "bold"}
    )
    plt.title("Smart Garbage Classifier - Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Ground Truth Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Smart Garbage Classifier Model")
    parser.add_argument("--model", type=str, default="models/waste_classifier.keras", help="Path to trained model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to save evaluation artifacts")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 75)
    print(" SMART GARBAGE CLASSIFICATION SYSTEM - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 75)

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file '{args.model}' not found. Please train the model first by running:\n"
            f"python train_model.py"
        )

    # 1. Load Test Dataset
    test_ds = load_test_dataset(dataset_dir=args.dataset)

    # 2. Run Predictions
    y_true, y_pred, y_probs = evaluate_keras_model(args.model, test_ds)

    # 3. Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 75)
    print(f" >>> OVERALL TEST ACCURACY: {acc * 100:.2f}% <<<")
    print("=" * 75)

    # 4. Detailed Classification Report
    print("\n[CLASSIFICATION REPORT - PRECISION / RECALL / F1-SCORE]")
    print("-" * 75)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    # 5. Confusion Matrix
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, output_path=cm_path)

    print("=" * 75)
    print("[EVALUATION COMPLETE] All metrics computed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    main()
