import json
import random
import shutil
from pathlib import Path


# ============================================================
# AeroSentinel - AU-AIR to YOLO Dataset Preparation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "04_AUAIR_multimodal_uav"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"

YOLO_DIR = PROJECT_ROOT / "data" / "yolo"

TRAIN_IMAGES = YOLO_DIR / "images" / "train"
VAL_IMAGES = YOLO_DIR / "images" / "val"

TRAIN_LABELS = YOLO_DIR / "labels" / "train"
VAL_LABELS = YOLO_DIR / "labels" / "val"


# ============================================================
# Configuration
# ============================================================

VAL_RATIO = 0.20
RANDOM_SEED = 42


# ============================================================
# Create directories
# ============================================================

for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load annotations
# ============================================================

print("=" * 70)
print("AeroSentinel - AU-AIR → YOLO Dataset Preparation")
print("=" * 70)

print("\nLoading annotations...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]
categories = data["categories"]

print(f"Total annotation records: {len(annotations)}")
print(f"Classes: {categories}")


# ============================================================
# Keep only records whose images actually exist
# ============================================================

print("\nFinding available image/annotation pairs...")

valid_records = []

for record in annotations:

    image_name = record.get("image_name")

    if not image_name:
        continue

    image_path = IMAGES_DIR / image_name

    if image_path.exists():
        valid_records.append(record)


print(f"Usable image/annotation pairs: {len(valid_records)}")


# ============================================================
# Remove duplicate image names
# ============================================================

records_by_image = {}

for record in valid_records:
    records_by_image[record["image_name"]] = record

valid_records = list(records_by_image.values())

print(f"Unique usable images: {len(valid_records)}")


# ============================================================
# Shuffle
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(valid_records)


# ============================================================
# Train / validation split
# ============================================================

split_index = int(len(valid_records) * (1 - VAL_RATIO))

train_records = valid_records[:split_index]
val_records = valid_records[split_index:]

print("\nDataset split:")
print(f"  Training images:   {len(train_records)}")
print(f"  Validation images: {len(val_records)}")


# ============================================================
# Convert one record to YOLO
# ============================================================

def convert_record(record, image_destination, label_destination):

    image_name = record["image_name"]

    source_image = IMAGES_DIR / image_name
    destination_image = image_destination / image_name

    shutil.copy2(source_image, destination_image)

    image_width = float(record.get("image_width:", 1920))
    image_height = float(record.get("image_height", 1080))

    label_lines = []

    for box in record.get("bbox", []):

        class_id = int(box["class"])

        left = float(box["left"])
        top = float(box["top"])
        width = float(box["width"])
        height = float(box["height"])

        right = left + width
        bottom = top + height

        # Skip invalid boxes
        if (
            width <= 0
            or height <= 0
            or left < 0
            or top < 0
            or right > image_width
            or bottom > image_height
        ):
            continue

        # Convert to center coordinates
        center_x = left + width / 2
        center_y = top + height / 2

        # Normalize to 0-1
        center_x /= image_width
        center_y /= image_height
        width /= image_width
        height /= image_height

        label_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_name = Path(image_name).stem + ".txt"

    label_path = label_destination / label_name

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))


# ============================================================
# Process training set
# ============================================================

print("\nPreparing training dataset...")

for index, record in enumerate(train_records):

    convert_record(
        record,
        TRAIN_IMAGES,
        TRAIN_LABELS,
    )

    if (index + 1) % 500 == 0:
        print(f"  Processed {index + 1}/{len(train_records)}")


# ============================================================
# Process validation set
# ============================================================

print("\nPreparing validation dataset...")

for index, record in enumerate(val_records):

    convert_record(
        record,
        VAL_IMAGES,
        VAL_LABELS,
    )

    if (index + 1) % 500 == 0:
        print(f"  Processed {index + 1}/{len(val_records)}")


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("YOLO DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"""
Training images:    {len(train_records)}
Validation images:  {len(val_records)}

Output directory:
{YOLO_DIR}

Classes:
""")

for index, name in enumerate(categories):
    print(f"  {index}: {name}")

print("\nNext step: verify the generated YOLO labels.")
print("=" * 70)