from ultralytics import YOLO

model = YOLO("path/to/best.pt")

metrics = model.val(
    data="your_dataset.yaml",
    imgsz=1024,
    batch=2,
    save_json=True,
)
