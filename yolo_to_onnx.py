import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

# Download the YOLOv5m model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Set the model to evaluation mode
model.eval()

# An example input tensor (with dimensions expected by the model)
dummy_input = torch.randn(1, 3, 640, 640)

# Convert the model to ONNX format
onnx_model_path = "yolov5m.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, opset_version=11)

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Check model
onnx.checker.check_model(onnx_model)

# Create a model session with ONNX Runtime
ort_session = ort.InferenceSession(onnx_model_path)

#  Get the input names expected by the model
input_names = [input.name for input in ort_session.get_inputs()]
print("Model Inputs:", input_names)

#make a prediction using the same input tensor
outputs = ort_session.run(None, {input_names[0]: dummy_input.numpy()})

# Examine the outputs
print(outputs)
