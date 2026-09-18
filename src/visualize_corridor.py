from pathlib import Path

import cv2
import pandas as pd

from flight_corridor import calculate_corridor, draw_corridor


# ---------------------------------------------------------
# AeroSentinel - Corridor Visualization
# ---------------------------------------------------------

TELEMETRY_FILE = Path("outputs/telemetry/uav_motion.csv")

IMAGE_DIR = Path("data/04_AUAIR_multimodal_uav/images")

OUTPUT_DIR = Path("outputs/corridor")


def main():

    print("=" * 60)
    print("AeroSentinel - Corridor Visualization")
    print("=" * 60)

    # Load telemetry
    df = pd.read_csv(TELEMETRY_FILE)

    row = df.iloc[0]

    image_name = row["image_name"]

    image_path = IMAGE_DIR / image_name

    print()
    print(f"Image: {image_name}")

    if not image_path.exists():

        print()
        print("ERROR: Image not found:")
        print(image_path)
        return

    # Load image
    image = cv2.imread(str(image_path))

    if image is None:

        print()
        print("ERROR: Could not read image.")
        return

    height, width = image.shape[:2]

    print(f"Image size: {width} x {height}")

    # Calculate corridor
    polygon = calculate_corridor(
        image_width=width,
        image_height=height,
        linear_x=row["linear_x"],
        linear_y=row["linear_y"],
    )

    # Draw corridor
    output = draw_corridor(
        image,
        polygon
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "corridor_overlay.jpg"

    cv2.imwrite(
        str(output_file),
        output
    )

    print()
    print("Corridor overlay saved to:")
    print(output_file)

    print()
    print("=" * 60)
    print("Visualization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()