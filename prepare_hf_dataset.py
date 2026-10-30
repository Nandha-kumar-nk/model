"""Prepare huaweilin/waste-classification as a reproducible three-class dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from PIL import Image

TARGET_CLASSES = ("organic", "inorganic", "reusable")
DEFAULT_SOURCE_TO_TARGET = {
    "organic": "organic",
    "non_recyclable": "inorganic",
    "hazardous": "inorganic",
    "recyclable": "reusable",
}


def image_bytes(value) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    if isinstance(value, Image.Image):
        output = io.BytesIO()
        value.convert("RGB").save(output, format="PNG")
        return output.getvalue()
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def normalize(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def label_name(example, feature) -> str:
    value = example
    names = getattr(feature, "names", None)
    if names:
        value = names[int(value)]
    return normalize(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="huaweilin/waste-classification")
    parser.add_argument("--image-field", default=None)
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--source-to-target", type=Path, default=None)
    args = parser.parse_args()

    mapping = DEFAULT_SOURCE_TO_TARGET
    if args.source_to_target:
        mapping = json.loads(args.source_to_target.read_text(encoding="utf-8"))
    mapping = {normalize(k): normalize(v) for k, v in mapping.items()}

    ds = load_dataset(args.dataset)
    print(ds)
    print("Columns:", ds.column_names)

    rows = []
    seen = set()
    for split_name, split in ds.items():
        columns = split.column_names
        print("Features:", split.features)
        image_field = args.image_field or next(
            (name for name in columns if name == "image"), None
        )
        label_field = args.label_field or next(
            (name for name in columns if name in {"label", "labels", "class", "category"}), None
        )
        if image_field is None or label_field is None:
            raise RuntimeError(f"Could not identify image/label fields in {split_name}: {columns}")
        print(f"{split_name}: image_field={image_field}, label_field={label_field}")
        feature = split.features[label_field]
        for index, item in enumerate(split):
            source = label_name(item[label_field], feature)
            target = mapping.get(source)
            if target not in TARGET_CLASSES:
                raise RuntimeError(
                    f"Dataset label {source!r} has no explicit mapping. "
                    f"Known labels: {sorted(mapping)}"
                )
            raw = image_bytes(item[image_field])
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in seen:
                seen.add(digest)
                rows.append((digest, target, raw, split_name, index))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for split in ("train", "val", "test"):
        shutil.rmtree(args.output / split, ignore_errors=True)

    for target in TARGET_CLASSES:
        class_rows = [row for row in rows if row[1] == target]
        train_end = int(len(class_rows) * 0.70)
        val_end = train_end + int(len(class_rows) * 0.15)
        partitions = (
            ("train", class_rows[:train_end]),
            ("val", class_rows[train_end:val_end]),
            ("test", class_rows[val_end:]),
        )
        for split, partition in partitions:
            destination = args.output / split / target
            destination.mkdir(parents=True, exist_ok=True)
            for digest, _, raw, _, _ in partition:
                Image.open(io.BytesIO(raw)).convert("RGB").save(
                    destination / f"{digest}.jpg", format="JPEG", quality=95
                )
            print(f"{split:5} {target:10} {len(partition):5}")
    print(f"Unique images written: {len(rows)}")


if __name__ == "__main__":
    main()
