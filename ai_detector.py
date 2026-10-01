# ai_detector.py
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def run_yolo_inference(image_path: str) -> dict:
    results = model(image_path, imgsz=320, device="cpu", verbose=False)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            detected_objects.append({
                "label": model.names[int(box.cls[0].item())],
                "confidence": round(float(box.conf[0].item()), 2)
            })
    return {
        "primary_defect": detected_objects[0]["label"] if detected_objects else "General Road Defect",
        "severity": "Critical" if detected_objects and detected_objects[0]["confidence"] > 0.8 else "Medium",
        "detections": detected_objects
    }