from ultralytics import YOLO

# Initialize model from the LiM-YOLO config
model = YOLO("ultralytics/cfg/models/v9/lim-yolo.yaml")

# Train
results = model.train(
    data="your_dataset.yaml",        # Path to your dataset config
    epochs=100,
    imgsz=1024,
    batch=2,                          # Adjust based on GPU memory
    device=0,                         # GPU device id (e.g., 0, [0,1] for multi-GPU)
    workers=16,
    optimizer="Adam",
    lr0=0.001,
    lrf=0.0001,
    seed=0,
    pretrained=False,
    single_cls=False,                 # Set True for single-class datasets
    # Disable augmentations (as used in the paper)
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    augment=False,
    plots=True,
    save_json=True,
    name="lim-yolo-experiment",
)
