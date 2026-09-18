import json
from pathlib import Path

import cv2


# ============================================================
# AeroSentinel - Visual Annotation Check
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "04_AUAIR_multimodal_uav"
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"
IMAGES_DIR = DATASET_DIR / "images"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotation_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load annotations
# ============================================================

print("Loading annotations...")

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]
categories = data["categories"]

print(f"Total annotation records: {len(annotations)}")
print(f"Classes: {categories}")


# ============================================================
# Find annotation records whose images actually exist
# ============================================================

valid_records = []

for record in annotations:

    image_name = record.get("image_name")

    if not image_name:
        continue

    image_path = IMAGES_DIR / image_name

    if image_path.exists():
        valid_records.append(record)


print(f"Images with matching annotations: {len(valid_records)}")


# ============================================================
# Select a few images
# ============================================================

# We deliberately use the first few matching records
# so the result is reproducible.

samples = valid_records[:5]


# ============================================================
# Draw bounding boxes
# ============================================================

for index, record in enumerate(samples):

    image_name = record["image_name"]
    image_path = IMAGES_DIR / image_name

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_name}")
        continue

    image_height, image_width = image.shape[:2]

    boxes_drawn = 0

    for box in record.get("bbox", []):

        class_id = box.get("class")

        left = int(box.get("left", 0))
        top = int(box.get("top", 0))
        width = int(box.get("width", 0))
        height = int(box.get("height", 0))

        right = left + width
        bottom = top + height

        # Skip invalid boxes
        if (
            left < 0
            or top < 0
            or width <= 0
            or height <= 0
            or right > image_width
            or bottom > image_height
        ):
            continue

        if isinstance(class_id, int) and 0 <= class_id < len(categories):
            class_name = categories[class_id]
        else:
            class_name = "Unknown"

        # Draw rectangle
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            3,
        )

        # Draw label
        label = f"{class_name} ({class_id})"

        cv2.putText(
            image,
            label,
            (left, max(top - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        boxes_drawn += 1

    # Add image name at top
    cv2.putText(
        image,
        image_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    output_path = OUTPUT_DIR / f"sample_{index + 1}.jpg"

    cv2.imwrite(str(output_path), image)

    print(
        f"Created: {output_path.name} "
        f"| boxes drawn: {boxes_drawn}"
    )


print("\nVisual annotation check complete.")
print(f"Open the images in: {OUTPUT_DIR}")