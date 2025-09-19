# export_model.py
from ultralytics import YOLO

print("Exporting yolov8n.pt to ONNX format...")
model = YOLO('models/yolov8n.pt')
model.export(format='onnx')
print("Export complete: models/yolov8n.onnx")
