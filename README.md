# LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection

[![arXiv](https://img.shields.io/badge/arXiv-2512.09700-b31b1b.svg)](https://doi.org/10.48550/arXiv.2512.09700)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the **official implementation** of the paper:
> **LiM-YOLO: Less is More with Pyramid Level Shift and Normalized Auxiliary Branch for Ship Detection in Optical Remote Sensing Imagery**
> *Seon-Hoon Kim, Hyeji Sim, Youeyun Jung, Okchul Jung, and Yerin Kim*
> Available on arXiv: [https://doi.org/10.48550/arXiv.2512.09700](https://doi.org/10.48550/arXiv.2512.09700)

## 🚀 Introduction

**LiM-YOLO** is a specialized object detector designed for maritime targets in high-resolution satellite imagery. By analyzing the scale distribution of ships, we propose a **Pyramid Level Shift (P2-P4)** strategy to resolve feature dilution of small vessels and remove the redundant P5 layer. Additionally, we introduce a **Group Normalized Auxiliary Branch (GN-CBLinear)** to stabilize training under micro-batch settings.

## 📂 Model Configuration

The core configuration file for the proposed **LiM-YOLO** model can be found at:

`ultralytics/cfg/models/v9/lim-yolo.yaml`

This configuration includes:
* **P2-P4 Head Structure**: Optimized for small ship detection (Stride 4, 8, 16).
* **Pruned P5**: Removed stride-32 layers to reduce computational redundancy.
* **GN-CBLinear**: Group Normalization added to the programmable gradient information (PGI) branch.

## 🛠️ Usage

### Installation
```bash
git clone [https://github.com/egshkim/LiM-YOLO.git](https://github.com/egshkim/LiM-YOLO.git)
cd LiM-YOLO
