# ASPNet: Adaptive Spatial-frequency Perception for Small Object Detection in UAV Imagery

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

## 📌 Overview

ASPNet (Adaptive Spatial-frequency Perception Network) is a novel approach for **small object detection in UAV (Unmanned Aerial Vehicle) imagery**. It adaptively integrates spatial and frequency domain information to improve detection accuracy of small targets in aerial images.

## ✨ Key Features

- **Adaptive Spatial-Frequency Perception**: Dynamically fuses spatial and frequency features
- **Small Object Enhancement**: Specialized design for detecting small-scale targets
- **UAV Image Optimized**: Tailored for aerial photography characteristics
- **Real-time Performance**: Efficient architecture suitable for UAV deployment

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch (with CUDA support)
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install in development mode
pip install -e .

Usage
Command Line
bash
# Detect small objects in an image
python detect.py --source path/to/image.jpg --weights best.pt

# Run on UAV video stream
python detect.py --source video.mp4 --conf 0.25
