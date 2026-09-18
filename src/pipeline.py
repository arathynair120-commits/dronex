from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# AeroSentinel Unified Backend Pipeline
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "04_AUAIR_multimodal_uav" / "images"
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "outputs"
    / "aerosentinel_yolo_test-2"
    / "weights"
    / "best.pt"
)

TELEMETRY_FILE = BASE_DIR / "outputs" / "telemetry" / "uav_motion.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_FILE = OUTPUT_DIR / "tracking_results.csv"
MOTION_FILE = OUTPUT_DIR / "object_motion.csv"
RISK_FILE = OUTPUT_DIR / "object_risk.csv"
RESULT_FILE = OUTPUT_DIR / "aerosentinel_result.json"
DEMO_FILE = OUTPUT_DIR / "aerosentinel_demo.jpg"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_FRAMES = 30
IMAGE_SIZE = 640
CONFIDENCE = 0.25

CLASS_NAMES = {
    0: "Human",
    1: "Car",
    2: "Truck",
    3: "Van",
    4: "Motorbike",
    5: "Bicycle",
    6: "Bus",
    7: "Trailer",
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def calculate_angle(dx, dy):
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360


def angle_difference(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def build_corridor(image_width, image_height, linear_x, linear_y):
    """
    Build a heuristic image-space projected flight corridor.

    This is NOT a true 3D trajectory.
    It represents the short-term projected flight direction
    using observed horizontal UAV motion.
    """

    cx = image_width / 2
    cy = image_height * 0.72

    speed = math.sqrt(linear_x ** 2 + linear_y ** 2)

    if speed < 0.001:
        dx = 0
        dy = -1
    else:
        # Camera/image-space approximation.
        dx = linear_y / speed
        dy = -linear_x / speed

    length = image_height * 0.45
    width = image_width * 0.18

    end_x = cx + dx * length
    end_y = cy + dy * length

    perp_x = -dy
    perp_y = dx

    p1 = (
        int(cx + perp_x * width / 2),
        int(cy + perp_y * width / 2),
    )

    p2 = (
        int(cx - perp_x * width / 2),
        int(cy - perp_y * width / 2),
    )

    p3 = (
        int(end_x - perp_x * width / 2),
        int(end_y - perp_y * width / 2),
    )

    p4 = (
        int(end_x + perp_x * width / 2),
        int(end_y + perp_y * width / 2),
    )

    return np.array([p1, p2, p3, p4], dtype=np.int32)


def corridor_proximity(point, polygon):
    """
    Convert distance to the projected corridor into a
    normalized proximity score.

    1 = very close / inside corridor
    0 = far from corridor
    """

    distance = abs(cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(point[0]), float(point[1])),
        True,
    ))

    polygon_area = abs(cv2.contourArea(polygon.astype(np.float32)))

    if polygon_area <= 0:
        return 0.0

    scale = math.sqrt(polygon_area)

    score = 1.0 - min(distance / (scale * 0.5), 1.0)

    inside = cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(point[0]), float(point[1])),
        False,
    )

    if inside >= 0:
        score = max(score, 0.75)

    return float(np.clip(score, 0.0, 1.0))


def motion_alignment(object_heading, uav_heading):
    difference = angle_difference(object_heading, uav_heading)

    # Same direction = 1
    # Opposite direction = 0
    return float((math.cos(math.radians(difference)) + 1) / 2)


def risk_level(score):
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "CAUTION"
    return "LOW"


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main():

    print("=" * 65)
    print("AeroSentinel - Unified Flight-Aware Hazard Pipeline")
    print("=" * 65)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    print("\n[1/8] Checking files...")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    if not TELEMETRY_FILE.exists():
        raise FileNotFoundError(
            f"Telemetry file not found:\n{TELEMETRY_FILE}"
        )

    print("Image directory :", IMAGE_DIR)
    print("YOLO model      :", MODEL_PATH)
    print("Telemetry       :", TELEMETRY_FILE)

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    print("\n[2/8] Loading telemetry...")

    telemetry = pd.read_csv(TELEMETRY_FILE)

    telemetry["image_name"] = telemetry["image_name"].astype(str)

    telemetry_lookup = telemetry.set_index("image_name").to_dict(
        orient="index"
    )

    print("Telemetry rows:", len(telemetry))

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------

    print("\n[3/8] Selecting frames...")

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    if len(image_files) == 0:
        image_files = sorted(IMAGE_DIR.glob("*.jpeg"))

    if len(image_files) == 0:
        raise RuntimeError("No JPG/JPEG images found.")

    image_files = image_files[:MAX_FRAMES]

    print("Frames selected:", len(image_files))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\n[4/8] Loading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Detection + tracking
    # --------------------------------------------------------

    print("\n[5/8] Running YOLO + ByteTrack...")

    tracking_records = []

    last_frame = None
    last_result = None

    for frame_number, image_path in enumerate(image_files, start=1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Skipping unreadable:", image_path.name)
            continue

        last_frame = frame.copy()

        result_list = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
            device="cpu",
        )

        result = result_list[0]
        last_result = result

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            print(
                f"Frame {frame_number:02d}/{len(image_files)} "
                f"- 0 detections"
            )
            continue

        ids = boxes.id

        if ids is None:
            print(
                f"Frame {frame_number:02d}/{len(image_files)} "
                f"- detections without track IDs"
            )
            continue

        ids = ids.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        print(
            f"Frame {frame_number:02d}/{len(image_files)} "
            f"- {len(ids)} tracked objects"
        )

        for box, track_id, confidence, class_id in zip(
            xyxy,
            ids,
            confs,
            classes,
        ):

            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            tracking_records.append({
                "frame": frame_number,
                "image_name": image_path.name,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES.get(
                    int(class_id),
                    str(class_id)
                ),
                "confidence": float(confidence),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "center_x": float(center_x),
                "center_y": float(center_y),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            })

    tracking_df = pd.DataFrame(tracking_records)

    if tracking_df.empty:
        raise RuntimeError(
            "No tracked objects were produced by YOLO."
        )

    tracking_df.to_csv(TRACKING_FILE, index=False)

    print("\nTracking records:", len(tracking_df))
    print("Saved:", TRACKING_FILE)

    # --------------------------------------------------------
    # Object motion
    # --------------------------------------------------------

    print("\n[6/8] Computing object motion...")

    motion_records = []

    for track_id, group in tracking_df.groupby("track_id"):

        group = group.sort_values("frame")

        start = group.iloc[0]
        end = group.iloc[-1]

        dx = float(end["center_x"] - start["center_x"])
        dy = float(end["center_y"] - start["center_y"])

        displacement = math.sqrt(dx ** 2 + dy ** 2)

        frame_difference = max(
            int(end["frame"] - start["frame"]),
            1,
        )

        speed = displacement / frame_difference

        heading = calculate_angle(dx, dy)

        start_area = (
            float(start["width"]) *
            float(start["height"])
        )

        end_area = (
            float(end["width"]) *
            float(end["height"])
        )

        if start_area > 0:
            bbox_growth = (
                (end_area - start_area) /
                start_area
            )
        else:
            bbox_growth = 0.0

        persistence = min(
            len(group) / len(image_files),
            1.0,
        )

        motion_records.append({
            "track_id": int(track_id),
            "class_id": int(end["class_id"]),
            "class_name": end["class_name"],
            "frames": int(len(group)),
            "start_frame": int(start["frame"]),
            "end_frame": int(end["frame"]),
            "start_image": start["image_name"],
            "end_image": end["image_name"],
            "start_x": float(start["center_x"]),
            "start_y": float(start["center_y"]),
            "end_x": float(end["center_x"]),
            "end_y": float(end["center_y"]),
            "displacement_px": float(displacement),
            "speed_px_per_frame": float(speed),
            "direction_deg": float(heading),
            "bbox_growth": float(bbox_growth),
            "persistence": float(persistence),
        })

    motion_df = pd.DataFrame(motion_records)

    motion_df.to_csv(MOTION_FILE, index=False)

    print("Tracks with motion:", len(motion_df))
    print("Saved:", MOTION_FILE)

    # --------------------------------------------------------
    # Flight-aware risk
    # --------------------------------------------------------

    print("\n[7/8] Computing flight-aware risk...")

    risk_records = []

    image_width = 1920
    image_height = 1080

    for _, row in motion_df.iterrows():

        telemetry_row = telemetry_lookup.get(
            row["end_image"]
        )

        if telemetry_row is None:
            # Fall back to the first telemetry row.
            telemetry_row = telemetry.iloc[0].to_dict()

        linear_x = float(
            telemetry_row.get("linear_x", 0.0)
        )

        linear_y = float(
            telemetry_row.get("linear_y", 0.0)
        )

        uav_heading = calculate_angle(
            linear_x,
            linear_y,
        )

        polygon = build_corridor(
            image_width,
            image_height,
            linear_x,
            linear_y,
        )

        point = (
            float(row["end_x"]),
            float(row["end_y"]),
        )

        proximity = corridor_proximity(
            point,
            polygon,
        )

        alignment = motion_alignment(
            float(row["direction_deg"]),
            uav_heading,
        )

        persistence = float(row["persistence"])

        bbox_growth = max(
            0.0,
            min(float(row["bbox_growth"]), 1.0),
        )

        # Illustrative engineering score.
        risk_score = (
            0.35 * proximity
            + 0.25 * alignment
            + 0.20 * persistence
            + 0.20 * bbox_growth
        )

        risk_records.append({
            "track_id": int(row["track_id"]),
            "class_name": row["class_name"],
            "frames": int(row["frames"]),
            "end_image": row["end_image"],
            "corridor_proximity": round(
                proximity,
                6,
            ),
            "motion_alignment": round(
                alignment,
                6,
            ),
            "persistence": round(
                persistence,
                6,
            ),
            "bbox_growth": round(
                bbox_growth,
                6,
            ),
            "risk_score": round(
                risk_score,
                6,
            ),
            "risk_level": risk_level(
                risk_score
            ),
            "uav_heading_deg": round(
                uav_heading,
                4,
            ),
        })

    risk_df = pd.DataFrame(risk_records)

    risk_df = risk_df.sort_values(
        "risk_score",
        ascending=False,
    )

    risk_df.to_csv(RISK_FILE, index=False)

    print("\nRisk results:")

    for _, row in risk_df.iterrows():
        print(
            f"Track {int(row['track_id'])} "
            f"{row['class_name']}: "
            f"{row['risk_score']:.3f} "
            f"{row['risk_level']}"
        )

    print("Saved:", RISK_FILE)

    # --------------------------------------------------------
    # Create final visualization
    # --------------------------------------------------------

    print("\n[8/8] Creating dashboard image...")

    if last_frame is None:
        raise RuntimeError("No valid final frame.")

    demo = last_frame.copy()

    # Use the telemetry corresponding to the final image
    final_image_name = image_files[-1].name

    final_telemetry = telemetry_lookup.get(
        final_image_name
    )

    if final_telemetry is None:
        final_telemetry = telemetry.iloc[0].to_dict()

    linear_x = float(
        final_telemetry.get("linear_x", 0.0)
    )

    linear_y = float(
        final_telemetry.get("linear_y", 0.0)
    )

    h, w = demo.shape[:2]

    polygon = build_corridor(
        w,
        h,
        linear_x,
        linear_y,
    )

    overlay = demo.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 255, 255),
    )

    demo = cv2.addWeighted(
        overlay,
        0.20,
        demo,
        0.80,
        0,
    )

    cv2.polylines(
        demo,
        [polygon],
        True,
        (0, 255, 255),
        4,
    )

    # Draw tracked objects from the final frame.
    final_tracking = tracking_df[
        tracking_df["frame"] == len(image_files)
    ]

    for _, obj in final_tracking.iterrows():

        x1 = int(obj["x1"])
        y1 = int(obj["y1"])
        x2 = int(obj["x2"])
        y2 = int(obj["y2"])

        track_id = int(obj["track_id"])
        class_name = obj["class_name"]
        confidence = float(obj["confidence"])

        cv2.rectangle(
            demo,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        label = (
            f"{class_name} #{track_id} "
            f"{confidence:.0%}"
        )

        cv2.putText(
            demo,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Header
    cv2.rectangle(
        demo,
        (0, 0),
        (w, 90),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        demo,
        "AEROSENTINEL",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        demo,
        "FLIGHT-AWARE UAV HAZARD INTELLIGENCE",
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Footer
    cv2.rectangle(
        demo,
        (0, h - 70),
        (w, h),
        (0, 0, 0),
        -1,
    )

    highest_risk = (
        risk_df.iloc[0]["risk_level"]
        if len(risk_df) > 0
        else "LOW"
    )

    footer = (
        f"TRACKED OBJECTS: {len(tracking_df['track_id'].unique())}   "
        f"HIGHEST RISK: {highest_risk}   "
        f"FRAME: {final_image_name}"
    )

    cv2.putText(
        demo,
        footer,
        (30, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(DEMO_FILE),
        demo,
    )

    print("Saved:", DEMO_FILE)

    # --------------------------------------------------------
    # JSON API result
    # --------------------------------------------------------

    highest = (
        risk_df.iloc[0].to_dict()
        if len(risk_df) > 0
        else {}
    )

    objects = []

    for _, row in risk_df.iterrows():
        objects.append({
            "track_id": int(row["track_id"]),
            "class_name": row["class_name"],
            "risk_score": float(row["risk_score"]),
            "risk_level": row["risk_level"],
            "corridor_proximity": float(
                row["corridor_proximity"]
            ),
            "motion_alignment": float(
                row["motion_alignment"]
            ),
            "persistence": float(
                row["persistence"]
            ),
            "bbox_growth": float(
                row["bbox_growth"]
            ),
        })

    result = {
        "system": "AeroSentinel",
        "status": "complete",
        "frames_processed": len(image_files),
        "tracking_records": len(tracking_df),
        "unique_tracks": int(
            tracking_df["track_id"].nunique()
        ),
        "highest_risk": highest,
        "objects": objects,
        "demo_image": DEMO_FILE.name,
        "note": (
            "Risk score is an illustrative engineering "
            "score for short-term flight-corridor hazard "
            "relevance. It is not a calibrated collision "
            "probability."
        ),
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("Saved:", RESULT_FILE)

    print("\n" + "=" * 65)
    print("AeroSentinel pipeline completed successfully.")
    print("=" * 65)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()