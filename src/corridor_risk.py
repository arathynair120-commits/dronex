from pathlib import Path
import numpy as np
import pandas as pd


MOTION_FILE = Path("outputs/tracking/object_motion.csv")
TELEMETRY_FILE = Path("outputs/telemetry/uav_motion.csv")
OUTPUT_FILE = Path("outputs/risk/object_risk.csv")


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def angle_difference(a, b):
    """
    Smallest absolute difference between two angles.
    Result is between 0 and 180 degrees.
    """
    difference = abs(a - b) % 360
    return min(difference, 360 - difference)


def calculate_risk(
    corridor_proximity,
    motion_alignment,
    persistence,
    bbox_growth
):

    score = (
        0.35 * corridor_proximity
        + 0.25 * motion_alignment
        + 0.20 * persistence
        + 0.20 * bbox_growth
    )

    score = clamp(score)

    if score >= 0.70:
        level = "HIGH"
    elif score >= 0.40:
        level = "CAUTION"
    else:
        level = "LOW"

    return score, level


def main():

    print("=" * 60)
    print("AeroSentinel - Flight-Aware Risk Analysis")
    print("=" * 60)

    if not MOTION_FILE.exists():
        print("\nERROR: Object motion file not found.")
        print(MOTION_FILE)
        return

    if not TELEMETRY_FILE.exists():
        print("\nERROR: UAV telemetry file not found.")
        print(TELEMETRY_FILE)
        return

    objects = pd.read_csv(MOTION_FILE)
    telemetry = pd.read_csv(TELEMETRY_FILE)

    print(f"\nObjects loaded   : {len(objects)}")
    print(f"Telemetry records: {len(telemetry)}")

    if len(telemetry) == 0:
        print("\nERROR: No telemetry records.")
        return

    # Use the first telemetry record for this prototype
    uav_heading = float(
        telemetry.iloc[0]["motion_heading_deg"]
    )

    print(
        f"\nUAV motion heading: "
        f"{uav_heading:.2f} degrees"
    )

    risk_records = []

    for _, obj in objects.iterrows():

        object_direction = float(
            obj["motion_direction_deg"]
        )

        # --------------------------------------------------
        # 1. MOTION ALIGNMENT
        # --------------------------------------------------

        difference = angle_difference(
            object_direction,
            uav_heading
        )

        # Same direction = 1
        # Opposite direction = 0
        motion_alignment = (
            1.0 - difference / 180.0
        )

        motion_alignment = clamp(
            motion_alignment
        )

        # --------------------------------------------------
        # 2. PERSISTENCE
        # --------------------------------------------------

        persistence = clamp(
            float(obj["persistence_score"])
        )

        # --------------------------------------------------
        # 3. BOUNDING-BOX GROWTH
        # --------------------------------------------------

        bbox_growth = clamp(
            float(obj["bbox_growth_score"])
        )

        # --------------------------------------------------
        # 4. CORRIDOR PROXIMITY
        # --------------------------------------------------
        #
        # For the MVP we use the object's final
        # normalized image position as a proxy.
        #
        # This is NOT true 3D geometry.
        #

        x = float(obj["end_x"])
        y = float(obj["end_y"])

        # AU-AIR images are 1920 x 1080
        image_width = 1920
        image_height = 1080

        normalized_x = x / image_width
        normalized_y = y / image_height

        # Approximate projected corridor around
        # the forward/central image region.
        center_x = 0.50
        center_y = 0.55

        distance = np.sqrt(
            (normalized_x - center_x) ** 2
            + (normalized_y - center_y) ** 2
        )

        # Convert distance into proximity.
        corridor_proximity = 1.0 - (
            distance / 0.70
        )

        corridor_proximity = clamp(
            corridor_proximity
        )

        # --------------------------------------------------
        # RISK
        # --------------------------------------------------

        score, level = calculate_risk(
            corridor_proximity,
            motion_alignment,
            persistence,
            bbox_growth
        )

        reasons = []

        if corridor_proximity >= 0.70:
            reasons.append(
                "near projected flight corridor"
            )

        if motion_alignment >= 0.70:
            reasons.append(
                "motion direction aligned with UAV"
            )

        if persistence >= 0.70:
            reasons.append(
                "persistent across frames"
            )

        if bbox_growth >= 0.70:
            reasons.append(
                "bounding box increasing"
            )

        if not reasons:
            reasons.append(
                "limited flight-path relevance"
            )

        risk_records.append({

            "track_id": int(
                obj["track_id"]
            ),

            "class_name":
                obj["class_name"],

            "corridor_proximity":
                corridor_proximity,

            "motion_alignment":
                motion_alignment,

            "persistence":
                persistence,

            "bbox_growth":
                bbox_growth,

            "risk_score":
                score,

            "risk_level":
                level,

            "reason":
                "; ".join(reasons)
        })

    risk_df = pd.DataFrame(
        risk_records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    risk_df = risk_df.sort_values(
        "risk_score",
        ascending=False
    )

    risk_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("RISK RESULTS")
    print("=" * 60)

    print()

    print(
        risk_df.to_string(
            index=False
        )
    )

    print("\nRisk distribution:")

    print(
        risk_df["risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("Flight-aware risk analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()