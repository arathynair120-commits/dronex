import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# AeroSentinel - Projected Flight Corridor
# ---------------------------------------------------------

TELEMETRY_FILE = Path("outputs/telemetry/uav_motion.csv")
OUTPUT_DIR = Path("outputs/corridor")


def normalize(value, minimum, maximum):
    """Normalize a value to 0-1."""

    if maximum == minimum:
        return 0.0

    return (value - minimum) / (maximum - minimum)


def calculate_corridor(
    image_width,
    image_height,
    linear_x,
    linear_y,
    corridor_width_ratio=0.18,
    corridor_length_ratio=0.45,
):
    """
    Create a simple image-space projected flight corridor.

    The corridor is centered around the image center and
    oriented according to the UAV's horizontal motion.

    This is an engineering visualization, not a true 3D
    trajectory reconstruction.
    """

    center_x = image_width / 2
    center_y = image_height * 0.72

    speed = math.sqrt(
        linear_x ** 2 +
        linear_y ** 2
    )

    # Avoid unstable direction when UAV is almost stationary.
    if speed < 0.01:
        linear_x = 1.0
        linear_y = 0.0
        speed = 1.0

    # Normalize motion direction.
    direction_x = linear_x / speed
    direction_y = linear_y / speed

    # Image-space corridor dimensions.
    corridor_width = image_width * corridor_width_ratio
    corridor_length = image_height * corridor_length_ratio

    # Perpendicular vector.
    perpendicular_x = -direction_y
    perpendicular_y = direction_x

    half_width = corridor_width / 2

    # Start point near lower-center of image.
    start_x = center_x
    start_y = center_y

    # End point in projected direction.
    end_x = start_x + direction_x * corridor_length
    end_y = start_y - direction_y * corridor_length

    # Four corners.
    p1 = (
        int(start_x + perpendicular_x * half_width),
        int(start_y + perpendicular_y * half_width),
    )

    p2 = (
        int(start_x - perpendicular_x * half_width),
        int(start_y - perpendicular_y * half_width),
    )

    p3 = (
        int(end_x - perpendicular_x * half_width),
        int(end_y - perpendicular_y * half_width),
    )

    p4 = (
        int(end_x + perpendicular_x * half_width),
        int(end_y + perpendicular_y * half_width),
    )

    polygon = np.array(
        [p1, p2, p3, p4],
        dtype=np.int32
    )

    return polygon


def draw_corridor(image, polygon):

    output = image.copy()

    # Semi-transparent corridor overlay.
    overlay = output.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 255, 255)
    )

    output = cv2.addWeighted(
        overlay,
        0.20,
        output,
        0.80,
        0
    )

    # Corridor boundary.
    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 255),
        3
    )

    # Label.
    x, y = polygon[0]

    cv2.putText(
        output,
        "PROJECTED FLIGHT CORRIDOR",
        (max(10, x), max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def main():

    print("=" * 60)
    print("AeroSentinel - Flight Corridor")
    print("=" * 60)

    if not TELEMETRY_FILE.exists():

        print()
        print("ERROR: UAV motion file not found.")
        print(f"Expected: {TELEMETRY_FILE}")
        return

    print()
    print("Loading UAV motion data...")

    df = pd.read_csv(TELEMETRY_FILE)

    print(f"Loaded {len(df)} records.")

    # Use the first telemetry record for the demo.
    row = df.iloc[0]

    image_width = int(row["image_width"])
    image_height = int(row["image_height"])

    polygon = calculate_corridor(
        image_width=image_width,
        image_height=image_height,
        linear_x=row["linear_x"],
        linear_y=row["linear_y"],
    )

    print()
    print("Sample UAV motion:")
    print(f"  linear_x: {row['linear_x']:.4f}")
    print(f"  linear_y: {row['linear_y']:.4f}")
    print(f"  heading:  {row['motion_heading_deg']:.2f} degrees")

    print()
    print("Image dimensions:")
    print(f"  Width : {image_width}")
    print(f"  Height: {image_height}")

    print()
    print("Corridor polygon:")
    print(polygon)

    # Create a blank image for visualization.
    image = np.zeros(
        (image_height, image_width, 3),
        dtype=np.uint8
    )

    output = draw_corridor(
        image,
        polygon
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = OUTPUT_DIR / "sample_corridor.jpg"

    cv2.imwrite(
        str(output_file),
        output
    )

    print()
    print("Sample corridor saved to:")
    print(output_file)

    print()
    print("=" * 60)
    print("Flight corridor test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()