from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TRACKING_FILE = Path("outputs/tracking/tracking_results.csv")
TELEMETRY_FILE = Path("outputs/telemetry/uav_motion.csv")
OUTPUT_FILE = Path("outputs/risk/object_risk_v2.csv")


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def angle_difference(a, b):
    diff = abs(float(a) - float(b)) % 360
    return min(diff, 360 - diff)


def build_corridor(linear_x, linear_y):
    """
    Build the same heuristic image-space projected
    flight corridor used by the AeroSentinel prototype.
    """

    # Image-space origin / starting point
    center_x = IMAGE_WIDTH / 2
    center_y = IMAGE_HEIGHT * 0.72

    velocity = np.array([
        float(linear_x),
        float(linear_y)
    ])

    magnitude = np.linalg.norm(velocity)

    # If UAV is almost stationary, point forward.
    if magnitude < 1e-6:
        direction = np.array([0.0, -1.0])
    else:
        direction = velocity / magnitude

        # Map horizontal UAV motion into image space.
        # The negative Y keeps forward motion visually upward.
        direction = np.array([
            direction[0],
            -direction[1]
        ])

        direction = direction / np.linalg.norm(direction)

    perpendicular = np.array([
        -direction[1],
        direction[0]
    ])

    corridor_length = IMAGE_HEIGHT * 0.45
    corridor_half_width = IMAGE_WIDTH * 0.18 / 2

    start = np.array([
        center_x,
        center_y
    ])

    end = start + direction * corridor_length

    p1 = start + perpendicular * corridor_half_width
    p2 = start - perpendicular * corridor_half_width
    p3 = end - perpendicular * corridor_half_width
    p4 = end + perpendicular * corridor_half_width

    polygon = np.array([
        p1,
        p2,
        p3,
        p4
    ], dtype=np.float32)

    return polygon


def corridor_proximity(point_x, point_y, polygon):
    """
    Returns:
        1.0 = inside corridor
        values between 0 and 1 = near corridor
        0.0 = sufficiently far away
    """

    point = (
        float(point_x),
        float(point_y)
    )

    # Positive if inside, negative if outside.
    signed_distance = cv2.pointPolygonTest(
        polygon,
        point,
        True
    )

    # Inside corridor
    if signed_distance >= 0:
        return 1.0

    # Outside corridor:
    # decay proximity with distance.
    proximity_distance = IMAGE_WIDTH * 0.12

    score = 1.0 - (
        abs(signed_distance) /
        proximity_distance
    )

    return clamp(score)


def motion_alignment(object_direction, uav_direction):
    difference = angle_difference(
        object_direction,
        uav_direction
    )

    return clamp(
        1.0 - difference / 180.0
    )


def calculate_risk(
    corridor,
    alignment,
    persistence,
    growth
):

    score = (
        0.35 * corridor
        + 0.25 * alignment
        + 0.20 * persistence
        + 0.20 * growth
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
    print("AeroSentinel - Flight-Aware Risk Engine v2")
    print("=" * 60)

    if not TRACKING_FILE.exists():
        print("\nERROR: tracking_results.csv not found.")
        print(TRACKING_FILE)
        return

    if not TELEMETRY_FILE.exists():
        print("\nERROR: uav_motion.csv not found.")
        print(TELEMETRY_FILE)
        return

    tracking = pd.read_csv(TRACKING_FILE)
    telemetry = pd.read_csv(TELEMETRY_FILE)

    print(f"\nTracking records : {len(tracking)}")
    print(f"Telemetry records: {len(telemetry)}")

    # --------------------------------------------------
    # Match telemetry to image names
    # --------------------------------------------------

    telemetry_lookup = telemetry.set_index(
        "image_name"
    )

    risk_records = []

    # --------------------------------------------------
    # Process every tracked object
    # --------------------------------------------------

    for track_id, group in tracking.groupby("track_id"):

        if track_id == -1:
            continue

        group = group.sort_values("frame").reset_index(
            drop=True
        )

        if len(group) < 2:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        last_image = last["image_name"]

        # ----------------------------------------------
        # Find telemetry corresponding to final frame
        # ----------------------------------------------

        if last_image not in telemetry_lookup.index:

            print(
                f"WARNING: No telemetry for {last_image}"
            )

            continue

        telemetry_row = telemetry_lookup.loc[last_image]

        # ----------------------------------------------
        # UAV motion
        # ----------------------------------------------

        uav_heading = float(
            telemetry_row["motion_heading_deg"]
        )

        linear_x = float(
            telemetry_row["linear_x"]
        )

        linear_y = float(
            telemetry_row["linear_y"]
        )

        # ----------------------------------------------
        # Object motion
        # ----------------------------------------------

        dx = (
            float(last["center_x"])
            - float(first["center_x"])
        )

        dy = (
            float(last["center_y"])
            - float(first["center_y"])
        )

        object_direction = np.degrees(
            np.arctan2(dy, dx)
        )

        if object_direction < 0:
            object_direction += 360

        displacement = np.sqrt(
            dx ** 2 + dy ** 2
        )

        # ----------------------------------------------
        # Persistence
        # ----------------------------------------------

        frames_tracked = len(group)

        persistence = clamp(
            frames_tracked / 10.0
        )

        # ----------------------------------------------
        # Bounding box growth
        # ----------------------------------------------

        first_area = (
            float(first["width"])
            * float(first["height"])
        )

        last_area = (
            float(last["width"])
            * float(last["height"])
        )

        if first_area > 0:

            bbox_growth = (
                last_area - first_area
            ) / first_area

        else:

            bbox_growth = 0.0

        growth_score = clamp(
            bbox_growth
        )

        # ----------------------------------------------
        # Flight corridor
        # ----------------------------------------------

        polygon = build_corridor(
            linear_x,
            linear_y
        )

        final_x = float(last["center_x"])
        final_y = float(last["center_y"])

        corridor_score = corridor_proximity(
            final_x,
            final_y,
            polygon
        )

        # ----------------------------------------------
        # Motion alignment
        # ----------------------------------------------

        alignment_score = motion_alignment(
            object_direction,
            uav_heading
        )

        # ----------------------------------------------
        # Overall risk
        # ----------------------------------------------

        risk_score, risk_level = calculate_risk(
            corridor_score,
            alignment_score,
            persistence,
            growth_score
        )

        # ----------------------------------------------
        # Explainability
        # ----------------------------------------------

        reasons = []

        if corridor_score >= 0.70:
            reasons.append(
                "inside/near projected flight corridor"
            )

        if alignment_score >= 0.70:
            reasons.append(
                "motion aligned with UAV"
            )

        if persistence >= 0.70:
            reasons.append(
                "persistent across frames"
            )

        if growth_score >= 0.70:
            reasons.append(
                "object appears to be approaching"
            )

        if not reasons:
            reasons.append(
                "limited short-term flight-path relevance"
            )

        risk_records.append({

            "track_id": int(track_id),

            "class_name":
                first["class_name"],

            "last_image":
                last_image,

            "frames_tracked":
                frames_tracked,

            "object_displacement_px":
                displacement,

            "object_direction_deg":
                object_direction,

            "uav_heading_deg":
                uav_heading,

            "corridor_proximity":
                corridor_score,

            "motion_alignment":
                alignment_score,

            "persistence":
                persistence,

            "bbox_growth":
                bbox_growth,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "reason":
                "; ".join(reasons)
        })

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    risk_df = pd.DataFrame(risk_records)

    if risk_df.empty:

        print("\nERROR: No risk records generated.")
        return

    risk_df = risk_df.sort_values(
        "risk_score",
        ascending=False
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    risk_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Display
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FLIGHT-AWARE RISK RESULTS")
    print("=" * 60)

    columns = [
        "track_id",
        "class_name",
        "frames_tracked",
        "corridor_proximity",
        "motion_alignment",
        "persistence",
        "bbox_growth",
        "risk_score",
        "risk_level"
    ]

    print(
        risk_df[columns].to_string(
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
    print("Flight-aware risk engine complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()