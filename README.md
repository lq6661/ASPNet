# ASPNet: Adaptive Spatial-frequency Perception for Small Object Detection in UAV Imagery

## Documentation

See below for a quickstart install and usage examples.

## Install

Pip install the ASPNet package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.

```bash
pip install -r requirements.txt
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .

ASPNet may be used directly in the Command Line Interface (CLI) with a command:
python detect.py model=best.pt source='path/to/image.jpg'
