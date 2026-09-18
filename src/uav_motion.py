from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# AeroSentinel - UAV Motion Processor
# ---------------------------------------------------------

INPUT_FILE = Path("outputs/telemetry/telemetry.csv")
OUTPUT_DIR = Path("outputs/telemetry")
OUTPUT_FILE = OUTPUT_DIR / "uav_motion.csv"


def calculate_motion(df):

    # -----------------------------------------------------
    # Horizontal and total velocity magnitude
    # -----------------------------------------------------

    df["horizontal_speed"] = np.sqrt(
        df["linear_x"] ** 2 +
        df["linear_y"] ** 2
    )

    df["total_speed"] = np.sqrt(
        df["linear_x"] ** 2 +
        df["linear_y"] ** 2 +
        df["linear_z"] ** 2
    )

    # -----------------------------------------------------
    # Vertical motion
    # -----------------------------------------------------

    df["vertical_speed"] = df["linear_z"]

    # -----------------------------------------------------
    # Heading
    #
    # We use atan2 to estimate the direction of horizontal
    # movement from linear_x and linear_y.
    # -----------------------------------------------------

    df["motion_heading_rad"] = np.arctan2(
        df["linear_y"],
        df["linear_x"]
    )

    df["motion_heading_deg"] = np.degrees(
        df["motion_heading_rad"]
    )

    # Convert negative angles to 0-360 degrees
    df["motion_heading_deg"] = (
        df["motion_heading_deg"] + 360
    ) % 360

    return df


def print_summary(df):

    print()
    print("=" * 60)
    print("UAV MOTION SUMMARY")
    print("=" * 60)

    print()
    print("Horizontal speed:")
    print(f"  Mean : {df['horizontal_speed'].mean():.4f}")
    print(f"  Min  : {df['horizontal_speed'].min():.4f}")
    print(f"  Max  : {df['horizontal_speed'].max():.4f}")

    print()
    print("Total speed:")
    print(f"  Mean : {df['total_speed'].mean():.4f}")
    print(f"  Min  : {df['total_speed'].min():.4f}")
    print(f"  Max  : {df['total_speed'].max():.4f}")

    print()
    print("Vertical motion:")
    print(f"  Mean : {df['vertical_speed'].mean():.4f}")
    print(f"  Min  : {df['vertical_speed'].min():.4f}")
    print(f"  Max  : {df['vertical_speed'].max():.4f}")

    print()
    print("Motion heading:")
    print(f"  Mean : {df['motion_heading_deg'].mean():.2f} degrees")
    print(f"  Min  : {df['motion_heading_deg'].min():.2f} degrees")
    print(f"  Max  : {df['motion_heading_deg'].max():.2f} degrees")


def main():

    print("=" * 60)
    print("AeroSentinel - UAV Motion Processor")
    print("=" * 60)

    if not INPUT_FILE.exists():

        print()
        print("ERROR: Telemetry file not found.")
        print(f"Expected: {INPUT_FILE}")
        return

    print()
    print("Loading telemetry...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} telemetry records.")

    print()
    print("Calculating UAV motion features...")

    df = calculate_motion(df)

    print_summary(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("Motion data saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 60)
    print("UAV motion processing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()