from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("outputs/tracking/tracking_results.csv")
OUTPUT_FILE = Path("outputs/tracking/object_motion.csv")


def main():

    print("=" * 60)
    print("AeroSentinel - Object Motion Analysis")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print("\nERROR: Tracking results not found.")
        print(INPUT_FILE)
        print("\nRun object_tracking.py first.")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTracking records loaded: {len(df)}")

    motion_records = []

    for track_id, group in df.groupby("track_id"):

        # Ignore detections where ByteTrack did not assign an ID
        if track_id == -1:
            continue

        group = group.sort_values("frame").reset_index(drop=True)

        if len(group) < 2:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        dx = last["center_x"] - first["center_x"]
        dy = last["center_y"] - first["center_y"]

        displacement = np.sqrt(
            dx ** 2 + dy ** 2
        )

        frame_span = last["frame"] - first["frame"]

        if frame_span > 0:
            motion_speed = displacement / frame_span
        else:
            motion_speed = 0.0

        direction = np.degrees(
            np.arctan2(dy, dx)
        )

        if direction < 0:
            direction += 360

        initial_area = (
            first["width"] *
            first["height"]
        )

        final_area = (
            last["width"] *
            last["height"]
        )

        if initial_area > 0:

            bbox_growth = (
                final_area - initial_area
            ) / initial_area

        else:

            bbox_growth = 0.0

        # Convert growth to 0-1 range for risk engine
        growth_score = np.clip(
            bbox_growth / 1.0,
            0,
            1
        )

        persistence = np.clip(
            len(group) / 10.0,
            0,
            1
        )

        motion_records.append({

            "track_id": int(track_id),

            "class_name": first["class_name"],

            "frames_tracked": len(group),

            "start_frame": int(first["frame"]),

            "end_frame": int(last["frame"]),

            "displacement_pixels": float(
                displacement
            ),

            "motion_speed_pixels_per_frame": float(
                motion_speed
            ),

            "motion_direction_deg": float(
                direction
            ),

            "bbox_growth": float(
                bbox_growth
            ),

            "bbox_growth_score": float(
                growth_score
            ),

            "persistence_score": float(
                persistence
            ),

            "start_x": float(
                first["center_x"]
            ),

            "start_y": float(
                first["center_y"]
            ),

            "end_x": float(
                last["center_x"]
            ),

            "end_y": float(
                last["center_y"]
            ),

        })

    motion_df = pd.DataFrame(
        motion_records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    motion_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nTracks with motion data: "
        f"{len(motion_df)}"
    )

    if len(motion_df) > 0:

        print("\nMotion summary:")

        print(
            motion_df[
                [
                    "track_id",
                    "class_name",
                    "frames_tracked",
                    "displacement_pixels",
                    "motion_speed_pixels_per_frame",
                    "motion_direction_deg",
                    "bbox_growth",
                    "persistence_score"
                ]
            ].to_string(index=False)
        )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("Object motion analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()