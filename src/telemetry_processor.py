import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# AeroSentinel - Telemetry Processor
# ---------------------------------------------------------

# Dataset location
DATASET_DIR = Path("data/04_AUAIR_multimodal_uav")
ANNOTATIONS_FILE = DATASET_DIR / "annotations.json"


def load_annotations():
    """Load AU-AIR annotations JSON."""
    print("Loading annotations...")

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Annotations loaded.")

    return data


def extract_telemetry(data):
    """
    Extract telemetry information from AU-AIR annotation records.
    """

    records = []

    annotations = data.get("annotations", [])

    print(f"Total records found: {len(annotations)}")

    for item in annotations:

        record = {
            "image_name": item.get("image_name"),
            "image_width": item.get("image_width:"),
            "image_height": item.get("image_height"),

            "platform": item.get("platform"),
            "time": item.get("time"),

            "longitude": item.get("longtitude"),
            "latitude": item.get("latitude"),
            "altitude": item.get("altitude"),

            "linear_x": item.get("linear_x"),
            "linear_y": item.get("linear_y"),
            "linear_z": item.get("linear_z"),

            "angle_phi": item.get("angle_phi"),
            "angle_theta": item.get("angle_theta"),
            "angle_psi": item.get("angle_psi"),

        }

        records.append(record)

    return pd.DataFrame(records)


def main():

    print("=" * 60)
    print("AeroSentinel - Telemetry Processor")
    print("=" * 60)

    # Check dataset
    if not ANNOTATIONS_FILE.exists():
        print()
        print("ERROR: annotations.json not found.")
        print(f"Expected location: {ANNOTATIONS_FILE}")
        return

    # Load
    data = load_annotations()

    # Extract
    df = extract_telemetry(data)

    print()
    print("Telemetry dataframe created.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print("First 5 records:")
    print(df.head().to_string())

    print()
    print("Missing values:")
    print(df.isnull().sum().to_string())

    # Save
    output_dir = Path("outputs/telemetry")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "telemetry.csv"

    df.to_csv(output_file, index=False)

    print()
    print(f"Telemetry saved to:")
    print(output_file)

    print()
    print("=" * 60)
    print("Telemetry processing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()