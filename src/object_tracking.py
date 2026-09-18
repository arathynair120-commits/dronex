from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = Path("runs/detect/outputs/aerosentinel_yolo_test-2/weights/best.pt")
IMAGE_DIR = Path("data/04_AUAIR_multimodal_uav/images")
OUTPUT_DIR = Path("outputs/tracking")


def main():

    print("=" * 60)
    print("AeroSentinel - Object Tracking")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("ERROR: Trained YOLO model not found.")
        print(MODEL_PATH)
        return

    images = sorted(IMAGE_DIR.glob("*.jpg"))

    # Smoke-test sequence
    images = images[:30]

    print(f"\nImages selected for tracking: {len(images)}")

    print("\nLoading trained YOLO model...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    print("\nStarting ByteTrack...\n")

    for frame_number, image_path in enumerate(images, start=1):

        frame = cv2.imread(str(image_path))

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        detection_count = 0

        if result.boxes is not None:

            detection_count = len(result.boxes)

            boxes = result.boxes.xyxy.cpu().numpy()

            confidences = result.boxes.conf.cpu().numpy()

            classes = result.boxes.cls.cpu().numpy().astype(int)

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * detection_count

            for i in range(detection_count):

                x1, y1, x2, y2 = boxes[i]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                width = x2 - x1
                height = y2 - y1

                class_id = classes[i]

                class_name = model.names[class_id]

                track_id = int(track_ids[i])

                records.append({
                    "frame": frame_number,
                    "image_name": image_path.name,
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidences[i]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                })

        annotated_frame = result.plot()

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        output_path = OUTPUT_DIR / f"track_{frame_number:03d}.jpg"

        cv2.imwrite(
            str(output_path),
            annotated_frame
        )

        print(
            f"Frame {frame_number:02d}/{len(images)} "
            f"| detections: {detection_count}"
        )

    # Save tracking data
    tracking_df = pd.DataFrame(records)

    csv_path = OUTPUT_DIR / "tracking_results.csv"

    tracking_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("TRACKING SUMMARY")
    print("=" * 60)

    print(f"\nFrames processed       : {len(images)}")
    print(f"Total detections       : {len(records)}")

    if len(records) > 0:

        print(
            f"Unique track IDs      : "
            f"{tracking_df['track_id'].nunique()}"
        )

        print("\nObjects by class:")

        print(
            tracking_df["class_name"]
            .value_counts()
            .to_string()
        )

    print("\nTracking CSV saved to:")
    print(csv_path)

    print("\nTracking images saved to:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Tracking complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()