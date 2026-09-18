import json
from pathlib import Path
from collections import Counter


# ============================================================
# AeroSentinel - AU-AIR Dataset Inspector
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "04_AUAIR_multimodal_uav"

ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"
IMAGES_DIR = DATASET_DIR / "images"


print("=" * 70)
print("AeroSentinel - AU-AIR Dataset Inspector")
print("=" * 70)

print(f"\nDataset folder: {DATASET_DIR}")
print(f"Annotations:    {ANNOTATIONS_FILE}")
print(f"Images folder:  {IMAGES_DIR}")


# ============================================================
# 1. Check paths
# ============================================================

print("\n" + "=" * 70)
print("PATH CHECK")
print("=" * 70)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(f"annotations.json not found: {ANNOTATIONS_FILE}")

if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"images folder not found: {IMAGES_DIR}")

print("Dataset folder: OK")
print("annotations.json: OK")
print("images folder: OK")


# ============================================================
# 2. Count images
# ============================================================

image_files = list(IMAGES_DIR.glob("*.jpg"))

print("\n" + "=" * 70)
print("IMAGE INSPECTION")
print("=" * 70)

print(f"Actual JPG images found: {len(image_files)}")


# ============================================================
# 3. Load JSON
# ============================================================

print("\n" + "=" * 70)
print("LOADING ANNOTATIONS")
print("=" * 70)

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("annotations.json loaded successfully.")
print(f"Top-level type: {type(data).__name__}")
print(f"Top-level keys: {list(data.keys())}")


# ============================================================
# 4. Extract annotations and categories
# ============================================================

annotations = data.get("annotations", [])
categories = data.get("categories", [])

print(f"\nAnnotation records: {len(annotations)}")
print(f"Categories: {categories}")


# ============================================================
# 5. Class inspection
# ============================================================

print("\n" + "=" * 70)
print("OBJECT CLASS INSPECTION")
print("=" * 70)

class_counter = Counter()
total_boxes = 0

for record in annotations:

    bboxes = record.get("bbox", [])

    for box in bboxes:
        class_id = box.get("class")

        if class_id is not None:
            class_counter[class_id] += 1
            total_boxes += 1

print(f"Total bounding boxes: {total_boxes}")

print("\nClass distribution:")

for class_id, count in sorted(class_counter.items()):

    if isinstance(class_id, int) and class_id < len(categories):
        class_name = categories[class_id]
    else:
        class_name = "Unknown"

    print(f"  {class_id}: {class_name:<12} {count:,}")


# ============================================================
# 6. Annotation ↔ image matching
# ============================================================

print("\n" + "=" * 70)
print("IMAGE / ANNOTATION MATCHING")
print("=" * 70)

image_names_on_disk = {
    image.name for image in image_files
}

annotation_image_names = {
    record.get("image_name")
    for record in annotations
    if record.get("image_name")
}

matched = image_names_on_disk & annotation_image_names
missing_images = annotation_image_names - image_names_on_disk
extra_images = image_names_on_disk - annotation_image_names

print(f"Images on disk:             {len(image_names_on_disk):,}")
print(f"Images referenced by JSON:  {len(annotation_image_names):,}")
print(f"Matched images:             {len(matched):,}")
print(f"Referenced but missing:     {len(missing_images):,}")
print(f"Images without annotation:  {len(extra_images):,}")


# ============================================================
# 7. Bounding-box validation
# ============================================================

print("\n" + "=" * 70)
print("BOUNDING BOX VALIDATION")
print("=" * 70)

invalid_boxes = 0
valid_boxes = 0

for record in annotations:

    image_width = record.get("image_width:", 1920)
    image_height = record.get("image_height", 1080)

    for box in record.get("bbox", []):

        left = box.get("left", 0)
        top = box.get("top", 0)
        width = box.get("width", 0)
        height = box.get("height", 0)

        right = left + width
        bottom = top + height

        valid = (
            width > 0
            and height > 0
            and left >= 0
            and top >= 0
            and right <= image_width
            and bottom <= image_height
        )

        if valid:
            valid_boxes += 1
        else:
            invalid_boxes += 1

print(f"Valid bounding boxes:   {valid_boxes:,}")
print(f"Invalid bounding boxes: {invalid_boxes:,}")


# ============================================================
# 8. Telemetry validation
# ============================================================

print("\n" + "=" * 70)
print("TELEMETRY VALIDATION")
print("=" * 70)

telemetry_fields = [
    "longtitude",
    "latitude",
    "altitude",
    "linear_x",
    "linear_y",
    "linear_z",
    "angle_phi",
    "angle_theta",
    "angle_psi",
    "time",
]

records_with_missing_telemetry = 0

for record in annotations:

    missing = False

    for field in telemetry_fields:
        if field not in record:
            missing = True
            break

    if missing:
        records_with_missing_telemetry += 1

print(f"Records checked: {len(annotations):,}")
print(
    f"Records with missing telemetry: "
    f"{records_with_missing_telemetry:,}"
)


# ============================================================
# 9. Dataset summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"""
Images available:           {len(image_files):,}
Annotation records:         {len(annotations):,}
Total bounding boxes:       {total_boxes:,}
Object classes:             {len(categories)}
Matched images:             {len(matched):,}
Missing referenced images:  {len(missing_images):,}
Invalid bounding boxes:     {invalid_boxes:,}
Missing telemetry records:  {records_with_missing_telemetry:,}
""")

print("=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)