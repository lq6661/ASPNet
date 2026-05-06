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

```bash
python detect.py model=best.pt source='path/to/image.jpg'

ASPNet may also be used directly in a Python environment:

from aspnet import ASPNet

# Load a model
model = ASPNet("best.pt")

# Perform object detection on an image
results = model("path/to/image.jpg")
results.show()

# Train the model
train_results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    device="cpu",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")
