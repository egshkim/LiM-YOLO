from ultralytics import YOLO

# Load a trained model
model = YOLO("path/to/best.pt")

# Run inference
results = model.predict(
    source="path/to/images",     # Image file, directory, or glob pattern
    imgsz=1024,
    conf=0.25,                   # Confidence threshold
    iou=0.7,                     # NMS IoU threshold
    save=True,                   # Save annotated images
    save_txt=True,               # Save results in txt format
)
