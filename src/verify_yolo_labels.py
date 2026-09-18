from pathlib import Path
import cv2

# YOLO dataset locations
IMAGE_DIR = Path("data/yolo/images/val")
LABEL_DIR = Path("data/yolo/labels/val")

CLASS_NAMES = [
    "Human",
    "Car",
    "Truck",
    "Van",
    "Motorbike",
    "Bicycle",
    "Bus",
    "Trailer"
]

# Output folder
OUTPUT_DIR = Path("data/yolo_verification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Take the first 5 validation images
images = sorted(IMAGE_DIR.glob("*.jpg"))[:5]

print("=" * 60)
print("AeroSentinel - YOLO Label Verification")
print("=" * 60)

for image_path in images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"Missing label: {label_path.name}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    box_count = 0

    for line in lines:

        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        center_x = float(values[1])
        center_y = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized YOLO coordinates
        # back to pixel coordinates.
        x_center = center_x * width
        y_center = center_y * height
        w = box_width * width
        h = box_height * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Keep coordinates inside the image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Class name
        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"Class {class_id}"

        cv2.putText(
            image,
            class_name,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        box_count += 1

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    print(f"{image_path.name}: {box_count} boxes")

print()
print("Verification images saved to:")
print(OUTPUT_DIR)
print("=" * 60)