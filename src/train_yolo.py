from ultralytics import YOLO

# Load a small pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train
model.train(
    data="data/yolo/data.yaml",
    epochs=3,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=2,
    project="outputs",
    name="aerosentinel_yolo_test",
)

print("Training test completed.")