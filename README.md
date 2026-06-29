# LiM-YOLO: Less is More with Pyramid Level Shift for Ship Detection in Optical Remote Sensing

[![arXiv](https://img.shields.io/badge/arXiv-2512.09700-b31b1b.svg)](https://doi.org/10.48550/arXiv.2512.09700)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository is the **official implementation** of the following paper.
> **LiM-YOLO: Less is More with Pyramid Level Shift for Ship Detection in Optical Remote Sensing**
> *Seon-Hoon Kim, Yerin Kim, Hyeji Sim, Youeyun Jung, Ok-Chul Jung, Daewon Chung*
> [arXiv 2512.09700](https://doi.org/10.48550/arXiv.2512.09700)

## Introduction

**LiM-YOLO** is an oriented ship detector for high-resolution optical satellite imagery, built on YOLOv9-E. Ships are small and high-aspect-ratio, and the stride-32 pyramid level (P₅) compresses narrow vessels into sub-pixel features. LiM-YOLO applies a **Pyramid Level Shift** that moves the detection heads from P₃P₄P₅ (strides 8/16/32) to P₂P₃P₄ (strides 4/8/16). This adds a high-resolution P₂ head and removes the P₅ head and backbone stage. It also adds a **GN-CBLinear** module that places Group Normalization after the 1×1 convolution of YOLOv9-E's composite-backbone projection (CBLinear), which stabilizes training under the micro-batch regime required by 1024×1024 satellite tiles.

This repository also provides **LiM-YOLO-RB** (Reversible Branch) and **LiM-YOLO-RB-W** (Reversible Branch, Wide), two extensions that restore YOLOv9's auxiliary reversible branch (see [Models](#models)).

## Models

| Config | Description |
|:--|:--|
| `ultralytics/cfg/models/v9/lim-yolo.yaml` | **LiM-YOLO** adds Group Normalization to the composite-backbone projection (GN-CBLinear). |
| `ultralytics/cfg/models/v9/lim-yolo-rb.yaml` | **LiM-YOLO-RB** (Reversible Branch) restores YOLOv9's auxiliary reversible branch on top of LiM-YOLO. |
| `ultralytics/cfg/models/v9/lim-yolo-rb-w.yaml` | **LiM-YOLO-RB-W** (Reversible Branch, Wide) widens the P₄ stage to 1024 channels on top of LiM-YOLO-RB. |

Ultralytics' YOLOv9-E port omits the auxiliary reversible branch of the original [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9). LiM-YOLO-RB restores it.

> **Terminology.** CBLinear and CBFuse form a CBNet-style composite backbone ([Liang et al., IEEE TIP 2022](https://doi.org/10.1109/TIP.2022.3216771)) inside YOLOv9-E. CBLinear is a 1×1 linear projection and CBFuse is a parameter-free resample-and-sum fusion. They are retained at inference, unlike PGI's multi-level auxiliary information, which is used only during training. GN-CBLinear adds Group Normalization to the CBLinear projection.

## Installation

```bash
git clone https://github.com/egshkim/LiM-YOLO.git
cd LiM-YOLO
pip install -e .
```

## Usage

LiM-YOLO uses the standard Ultralytics API. Example scripts are under `ultralytics/examples/`. Edit the data and weights paths inside before running.

- `train.py` trains a model from a config.
- `predict.py` runs detection on images.
- `inference.py` validates a trained model.

## Results

Inference time is the average per-image forward time on a single NVIDIA RTX A6000, excluding preprocessing and postprocessing. At inference, the auxiliary branch of LiM-YOLO-RB and LiM-YOLO-RB-W is omitted.

### Per-dataset ablation

#### SODA-A
| Configuration | Params (M) | GFLOPs | Time (ms) | F1 | Prec. | Rec. | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| P₃ P₄ P₅ | 58.99 | 196.4 | 24.1 | 0.828 | 0.906 | 0.763 | 0.849 | 0.637 |
| P₂ P₃ P₄ P₅ | 57.41 | 230.2 | 29.9 | 0.833 | 0.909 | 0.769 | 0.855 | 0.656 |
| P₂ P₃ P₄ | 21.16 | 189.4 | 25.9 | 0.836 | 0.907 | 0.775 | 0.856 | 0.660 |
| P₂ P₃ | 16.15 | 173.4 | 24.5 | 0.832 | 0.907 | 0.769 | 0.860 | 0.660 |
| **LiM-YOLO** (P₂ P₃ P₄ + GN-CBLinear) | 21.16 | 189.4 | 26.9 | 0.829 | 0.905 | 0.765 | 0.861 | 0.662 |
| **LiM-YOLO-RB** | 21.16 | 189.4 | 26.9 | 0.835 | 0.909 | 0.771 | 0.861 | 0.678 |
| **LiM-YOLO-RB-W** | 22.84 | 194.8 | 27.5 | 0.836 | 0.911 | 0.773 | 0.863 | 0.682 |

#### DOTA-v1.5
| Configuration | Params (M) | GFLOPs | Time (ms) | F1 | Prec. | Rec. | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| P₃ P₄ P₅ | 58.99 | 196.4 | 24.6 | 0.883 | 0.942 | 0.831 | 0.913 | 0.736 |
| P₂ P₃ P₄ P₅ | 57.41 | 230.2 | 29.9 | 0.883 | 0.936 | 0.836 | 0.915 | 0.738 |
| P₂ P₃ P₄ | 21.16 | 189.4 | 25.8 | 0.891 | 0.940 | 0.847 | 0.923 | 0.744 |
| P₂ P₃ | 16.15 | 173.4 | 24.8 | 0.889 | 0.936 | 0.846 | 0.921 | 0.740 |
| **LiM-YOLO** (P₂ P₃ P₄ + GN-CBLinear) | 21.16 | 189.4 | 27.2 | 0.892 | 0.933 | 0.853 | 0.925 | 0.750 |
| **LiM-YOLO-RB** | 21.16 | 189.4 | 27.2 | 0.887 | 0.932 | 0.846 | 0.921 | 0.757 |
| **LiM-YOLO-RB-W** | 22.84 | 194.8 | 27.8 | 0.888 | 0.938 | 0.843 | 0.924 | 0.762 |

#### FAIR1M
| Configuration | Params (M) | GFLOPs | Time (ms) | F1 | Prec. | Rec. | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| P₃ P₄ P₅ | 59.00 | 196.4 | 24.4 | 0.422 | 0.388 | 0.463 | 0.395 | 0.285 |
| P₂ P₃ P₄ P₅ | 57.41 | 230.3 | 30.8 | 0.421 | 0.381 | 0.471 | 0.392 | 0.284 |
| P₂ P₃ P₄ | 21.16 | 189.5 | 25.8 | 0.437 | 0.404 | 0.477 | 0.402 | 0.290 |
| P₂ P₃ | 16.15 | 173.5 | 24.9 | 0.441 | 0.406 | 0.483 | 0.414 | 0.301 |
| **LiM-YOLO** (P₂ P₃ P₄ + GN-CBLinear) | 21.16 | 189.5 | 26.7 | 0.447 | 0.416 | 0.482 | 0.418 | 0.302 |
| **LiM-YOLO-RB** | 21.16 | 189.5 | 26.7 | 0.441 | 0.406 | 0.482 | 0.412 | 0.307 |
| **LiM-YOLO-RB-W** | 22.84 | 194.9 | 27.3 | 0.437 | 0.393 | 0.491 | 0.410 | 0.307 |

#### ShipRSImageNet
| Configuration | Params (M) | GFLOPs | Time (ms) | F1 | Prec. | Rec. | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| P₃ P₄ P₅ | 59.01 | 196.5 | 24.6 | 0.527 | 0.514 | 0.541 | 0.516 | 0.414 |
| P₂ P₃ P₄ P₅ | 57.42 | 230.4 | 30.0 | 0.514 | 0.496 | 0.534 | 0.526 | 0.415 |
| P₂ P₃ P₄ | 21.17 | 189.6 | 26.1 | 0.536 | 0.515 | 0.558 | 0.534 | 0.428 |
| P₂ P₃ | 16.16 | 173.6 | 25.2 | 0.515 | 0.499 | 0.532 | 0.524 | 0.325 |
| **LiM-YOLO** (P₂ P₃ P₄ + GN-CBLinear) | 21.17 | 189.6 | 26.9 | 0.574 | 0.548 | 0.601 | 0.578 | 0.448 |
| **LiM-YOLO-RB** | 21.17 | 189.6 | 26.9 | 0.552 | 0.554 | 0.549 | 0.570 | 0.470 |
| **LiM-YOLO-RB-W** | 22.85 | 195.0 | 27.5 | 0.589 | 0.593 | 0.585 | 0.580 | 0.482 |

### Comparison with state-of-the-art models

Evaluated on the Integrated Ship Detection Dataset (all four datasets combined).

| Model | Params (M) | GFLOPs | Time (ms) | F1 | Prec. | Rec. | mAP<sub>50</sub> | mAP<sub>50-95</sub> |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| YOLOv8x | 69.47 | 263.9 | 17.8 | 0.777 | 0.825 | 0.734 | 0.816 | 0.566 |
| YOLOv10x | 30.78 | 166.9 | 18.0 | 0.756 | 0.811 | 0.708 | 0.796 | 0.543 |
| YOLO11x | 58.78 | 203.8 | 18.6 | 0.764 | 0.822 | 0.713 | 0.805 | 0.554 |
| YOLOv12x | 61.02 | 208.1 | 36.7 | 0.721 | 0.793 | 0.662 | 0.748 | 0.494 |
| RT-DETR-X | 70.38 | 278.2 | 19.8 | 0.755 | 0.819 | 0.699 | 0.793 | 0.545 |
| **LiM-YOLO** | 21.16 | 189.4 | 26.7 | 0.791 | 0.839 | 0.748 | 0.832 | 0.600 |
| **LiM-YOLO-RB** | 21.16 | 189.4 | 26.7 | 0.798 | 0.846 | 0.755 | 0.838 | 0.631 |
| **LiM-YOLO-RB-W** | 22.84 | 194.8 | 27.3 | 0.797 | 0.847 | 0.753 | 0.838 | 0.631 |

### Qualitative Results

<table>
  <tr>
    <th>Dataset</th>
    <th>Baseline (YOLOv9-E)</th>
    <th>LiM-YOLO</th>
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

## Citation

```bibtex
@article{kim2025lim,
  title={LiM-YOLO: Less is More with Pyramid Level Shift for Ship Detection in Optical Remote Sensing},
  author={Kim, Seon-Hoon and Kim, Yerin and Sim, Hyeji and Jung, Youeyun and Jung, Ok-Chul and Chung, Daewon},
  journal={arXiv preprint arXiv:2512.09700},
  year={2025}
}
```

## License

This project is licensed under the [AGPL-3.0 License](https://www.gnu.org/licenses/agpl-3.0), consistent with the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework.

## Acknowledgments

This research was supported by Korea Institute of Marine Science & Technology Promotion (KIMST) funded by the Korea Coast Guard (RS-2023-00238652). This implementation is built on top of [Ultralytics](https://github.com/ultralytics/ultralytics).
