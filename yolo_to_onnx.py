import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

# YOLOv5m modelini indir ve yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Modeli değerlendirme moduna al
model.eval()

# Örnek bir giriş tensörü (modelin beklediği boyutlarda)
dummy_input = torch.randn(1, 3, 640, 640)

# Modeli ONNX formatına dönüştür
onnx_model_path = "yolov5m.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, opset_version=11)

# ONNX modelini yükle
onnx_model = onnx.load(onnx_model_path)

# Modeli doğrula
onnx.checker.check_model(onnx_model)

# ONNX Runtime ile model oturumunu oluştur
ort_session = ort.InferenceSession(onnx_model_path)

# Modelin beklediği girdi adlarını al
input_names = [input.name for input in ort_session.get_inputs()]
print("Model Inputs:", input_names)

# Aynı giriş tensörünü kullanarak tahmin yap
outputs = ort_session.run(None, {input_names[0]: dummy_input.numpy()})

# Çıktıları inceleyin
print(outputs)
