"""
==============================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - DATASET PREPARATION & SPLIT SCRIPT
==============================================================================
Maps raw public waste datasets into exactly 3 target classes:
  1. organic
  2. inorganic
  3. reusable

Performs stratified split:
  - 70% Training
  - 15% Validation
  - 15% Testing
==============================================================================
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_CLASSES = ['organic', 'inorganic', 'reusable']
SOURCE_TO_TARGET = {
    'organic': 'organic',
    'recyclable': 'reusable',
    'non_recyclable': 'inorganic',
    'hazardous': 'inorganic',
}

def create_dataset_directories(base_dir="dataset"):
    """Creates directory structure for raw and split datasets."""
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in TARGET_CLASSES:
            path = Path(base_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)
    print(f"Created dataset structure under '{base_dir}/'")

def split_and_copy_images(source_class_dir, target_base_dir, class_name, 
                          train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Splits images from a class folder into train, val, and test splits."""
    source_path = Path(source_class_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not images:
        print(f"Warning: No valid images found in {source_class_dir}")
        return

    random.shuffle(images)
    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    split_map = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    for split_name, img_list in split_map.items():
        dest_dir = Path(target_base_dir) / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for img in tqdm(img_list, desc=f"Copying {class_name} -> {split_name}"):
            dest_file = dest_dir / img.name
            if not dest_file.exists():
                shutil.copy2(img, dest_file)
                
    print(f"Class '{class_name}' split complete: Total={total} | Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

def print_dataset_summary(base_dir="dataset"):
    """Prints a detailed statistical table of the dataset splits."""
    print("\n" + "=" * 65)
    print(f"{'SPLIT':<12} | {'ORGANIC':<10} | {'INORGANIC':<12} | {'REUSABLE':<10} | {'TOTAL':<8}")
    print("=" * 65)
    
    grand_total = 0
    for split in ['train', 'val', 'test']:
        counts = {}
        for cls in TARGET_CLASSES:
            dir_path = Path(base_dir) / split / cls
            if dir_path.exists():
                counts[cls] = len([f for f in dir_path.iterdir() if f.is_file()])
            else:
                counts[cls] = 0
        split_total = sum(counts.values())
        grand_total += split_total
        print(f"{split.upper():<12} | {counts['organic']:<10} | {counts['inorganic']:<12} | {counts['reusable']:<10} | {split_total:<8}")
    
    print("=" * 65)
    print(f"GRAND TOTAL IMAGES: {grand_total}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    base_dataset_dir = "dataset"
    create_dataset_directories(base_dataset_dir)
    print("\nDataset preparation script initialized.")
    print("Place your categorized images in 'dataset/raw/<class_name>/' or download Kaggle waste datasets.")
    print_dataset_summary(base_dataset_dir)
