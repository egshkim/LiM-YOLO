# LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection

[![arXiv](https://img.shields.io/badge/arXiv-2512.09700-b31b1b.svg)](https://doi.org/10.48550/arXiv.2512.09700)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository is the **official implementation** of the paper:
> **LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection in Optical Remote Sensing Imagery**
> *Seon-Hoon Kim, Hyeji Sim, Youeyun Jung, Okchul Jung, and Yerin Kim*
> Available on arXiv: [https://doi.org/10.48550/arXiv.2512.09700](https://doi.org/10.48550/arXiv.2512.09700)

## Introduction

**LiM-YOLO** is a specialized object detector designed for maritime targets in high-resolution satellite imagery. By analyzing the scale distribution of ships across four major benchmarks, we propose a **Pyramid Level Shift (P2–P4)** strategy that introduces a high-resolution P2 head while pruning the redundant P5 layer, resolving spatial feature dilution for small vessels. Additionally, we introduce **GN-CBLinear**, which appends Group Normalization after the 1×1 convolution in the auxiliary branch of YOLOv9's Programmable Gradient Information (PGI) framework, stabilizing training under micro-batch constraints imposed by high-resolution satellite inputs.

## Datasets

We evaluate LiM-YOLO across four remote sensing datasets covering diverse satellite platforms and resolutions. All annotations are in **Oriented Bounding Box (OBB)** format.

<table>
  <tr>
    <th>Dataset</th>
    <th>Source Platform</th>
    <th>Resolution (GSD)</th>
    <th># Train Images</th>
    <th># Train Instances</th>
    <th># Val Images</th>
    <th># Val Instances</th>
  </tr>
  <tr>
    <td>SODA-A</td>
    <td>Google Earth</td>
    <td>0.5m – 0.8m</td>
    <td>1,030</td>
    <td>37,971</td>
    <td>323</td>
    <td>21,908</td>
  </tr>
  <tr>
    <td>DOTA-v1.5</td>
    <td>Google Earth, GF-2, JL-1</td>
    <td>0.3m – 0.6m</td>
    <td>2,657</td>
    <td>56,313</td>
    <td>572</td>
    <td>11,474</td>
  </tr>
  <tr>
    <td>FAIR1M-v2.0</td>
    <td>Gaofen Series, Google Earth</td>
    <td>0.3m – 0.8m</td>
    <td>6,413</td>
    <td>37,997</td>
    <td>2,932</td>
    <td>27,703</td>
  </tr>
  <tr>
    <td>ShipRSImageNet</td>
    <td>WorldView-3, GF-2, JL-1</td>
    <td>0.12m – 6.0m</td>
    <td>2,709</td>
    <td>11,834</td>
    <td>692</td>
    <td>3,459</td>
  </tr>
  <tr>
    <td><strong>Total</strong></td>
    <td>–</td>
    <td>–</td>
    <td><strong>12,809</strong></td>
    <td><strong>144,115</strong></td>
    <td><strong>4,519</strong></td>
    <td><strong>64,544</strong></td>
  </tr>
</table>

<p align="center"><em>Table III from the paper: Details of the preprocessed datasets used in experiments.</em></p>

## Model Configuration

The core configuration file for the proposed **LiM-YOLO** model can be found at:

```
ultralytics/cfg/models/v9/lim-yolo.yaml
```

This configuration includes:
* **P2–P4 Head Structure**: Optimized for small ship detection (Stride 4, 8, 16).
* **Pruned P5**: Removed stride-32 backbone and head to eliminate receptive field redundancy.
* **GN-CBLinear**: Group Normalization appended after the 1×1 convolution in the PGI auxiliary branch, following the standard Conv→GN ordering.

## Installation

```bash
git clone https://github.com/egshkim/LiM-YOLO.git
cd LiM-YOLO
pip install -e .
```

## Training

LiM-YOLO is built on top of the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. You can train the model using either the Python API or the CLI.

### Python API

```python
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
```

### CLI

```bash
yolo obb train \
    model=ultralytics/cfg/models/v9/lim-yolo.yaml \
    data=your_dataset.yaml \
    epochs=100 \
    imgsz=1024 \
    batch=2 \
    device=0 \
    optimizer=Adam \
    lr0=0.001 \
    lrf=0.0001 \
    seed=0 \
    pretrained=False \
    hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 \
    augment=False \
    name=lim-yolo-experiment
```

### Dataset Configuration

Prepare a YAML file for your dataset following the [Ultralytics OBB format](https://docs.ultralytics.com/datasets/obb/). For example, a DOTA-v1.5 config would look like:

```yaml
path: /path/to/DOTAv1.5       # Dataset root directory
train: images/train            # Train images (relative to 'path')
val: images/val                # Val images (relative to 'path')
test: images/test              # Test images (optional)

names:
  0: plane
  1: ship
  2: storage tank
  # ... (see ultralytics/cfg/datasets/DOTAv1.5.yaml for the full class list)
```

### Training Notes

- **Batch size**: We used `batch=2` with 1024×1024 input resolution on a single NVIDIA RTX A6000 (48 GB). Adjust according to your GPU memory.
- **Augmentations**: Color augmentations (`hsv_h`, `hsv_s`, `hsv_v`) and additional augmentations (`augment`) are disabled by default to match the paper's settings.
- **Multi-GPU**: Pass a list of device ids (e.g., `device=[0,1]`) for data-parallel training.
- **Optimizer**: Adam with an initial learning rate of `0.001` decayed to `0.0001` over 100 epochs via cosine annealing.

## Inference

### Predict on Images

```python
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
```

### CLI

```bash
yolo obb predict \
    model=path/to/best.pt \
    source=path/to/images \
    imgsz=1024 \
    conf=0.25 \
    iou=0.7 \
    save=True
```

### Validation

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

metrics = model.val(
    data="your_dataset.yaml",
    imgsz=1024,
    batch=2,
    save_json=True,
)
```

```bash
yolo obb val \
    model=path/to/best.pt \
    data=your_dataset.yaml \
    imgsz=1024 \
    batch=2
```

### Export

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
model.export(format="onnx", imgsz=1024)
```

## Results

### Comparison with State-of-the-Art Models

Evaluated on the Integrated Ship Detection Dataset (all four datasets combined). The best results are highlighted in **bold**.

| Model | Params (M) | GFLOPs | Speed (ms/img) | F1 | Precision | Recall | mAP<sup>50</sup> | mAP<sup>50-95</sup> |
|:------|:----------:|:------:|:---------------:|:---:|:---------:|:------:|:-----------------:|:--------------------:|
| YOLOv8x | 69.47 | 263.9 | 17.8 | 0.777 | 0.825 | 0.734 | 0.816 | 0.566 |
| YOLOv10x | 30.78 | 166.9 | 18.0 | 0.756 | 0.811 | 0.708 | 0.796 | 0.543 |
| YOLO11x | 58.78 | 203.8 | 18.6 | 0.764 | 0.822 | 0.713 | 0.805 | 0.554 |
| YOLOv12x | 61.02 | 208.1 | 36.7 | 0.721 | 0.793 | 0.662 | 0.748 | 0.494 |
| RT-DETR-X | 70.38 | 278.2 | 19.8 | 0.755 | 0.819 | 0.699 | 0.793 | 0.545 |
| **LiM-YOLO (Ours)** | **21.16** | **189.4** | **26.7** | **0.791** | **0.839** | **0.748** | **0.832** | **0.600** |

<p align="center"><em>Table VIII from the paper: Comparison with state-of-the-art models on the Integrated Ship Detection Dataset.</em></p>

### Qualitative Results

<table>
  <tr>
    <th>Dataset</th>
    <th>Baseline (YOLOv9-E)</th>
    <th>LiM-YOLO (Ours)</th>
    <th>Ground Truth</th>
  </tr>
  <tr>
    <td><strong>SODA-A</strong></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/Baseline_01780_tile0007.jpg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/LiM-YOLO_01780_tile0007.jpg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/GT_01780_tile0007.jpg" width="250"></td>
  </tr>
  <tr>
    <td><strong>DOTA-v1.5</strong></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/Baseline_P2726_1024_1024.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/LiM-YOLO_P2726_1024_1024.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/GT_P2726_1024_1024.jpeg" width="250"></td>
  </tr>
  <tr>
    <td><strong>FAIR1M-v2.0</strong></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/Baseline_257_pad.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/LiM-YOLO_257_pad.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/GT_257_pad.jpeg" width="250"></td>
  </tr>
  <tr>
    <td><strong>ShipRSImageNet</strong></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/Baseline_001839_x0_y0.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/LiM-YOLO_001839_x0_y0.jpeg" width="250"></td>
    <td><img src="https://arxiv.org/html/2512.09700v1/results/GT_001839_x0_y0.jpeg" width="250"></td>
  </tr>
</table>

<p align="center"><em>Fig. 6 from the paper: Qualitative comparison of detection results across four remote sensing datasets. OBBs are overlaid on the images. For single-class datasets, class labels and confidence scores are omitted for clarity.</em></p>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{kim2025limyolo,
  title={LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection in Optical Remote Sensing Imagery},
  author={Kim, Seon-Hoon and Sim, Hyeji and Jung, Youeyun and Jung, Okchul and Kim, Yerin},
  journal={arXiv preprint arXiv:2512.09700},
  year={2025}
}
```

## License

This project is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0), consistent with the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework.

## Acknowledgments

This research was supported by Korea Institute of Marine Science & Technology Promotion (KIMST) funded by the Korea Coast Guard (RS-2023-00238652). This implementation is built on top of [Ultralytics](https://github.com/ultralytics/ultralytics).
