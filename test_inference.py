"""Run reproducible Keras predictions for one or more image paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

CLASS_NAMES = ("ORGANIC", "INORGANIC", "REUSABLE")
IMAGE_SIZE = (224, 224)


def predict(model: tf.keras.Model, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE)
        pixels = np.asarray(image, dtype=np.float32)[None, ...]
    return model(pixels, training=False).numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", default="models/waste_classifier.keras")
    parser.add_argument("--expected", nargs="*", choices=CLASS_NAMES)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if tuple(model.output_shape[1:]) != (3,):
        raise ValueError(f"Unexpected model output shape: {model.output_shape}")

    for index, image_path in enumerate(args.images):
        probabilities = predict(model, image_path)
        predicted = CLASS_NAMES[int(np.argmax(probabilities))]
        expected = args.expected[index] if index < len(args.expected) else "unknown"
        correct = expected == "unknown" or predicted == expected
        print(
            f"{image_path.name}\tactual={expected}\tpredicted={predicted}"
            f"\tORGANIC={probabilities[0]:.6f}"
            f"\tINORGANIC={probabilities[1]:.6f}"
            f"\tREUSABLE={probabilities[2]:.6f}"
            f"\tconfidence={probabilities.max():.6f}"
            f"\tcorrect={correct}"
        )


if __name__ == "__main__":
    main()
