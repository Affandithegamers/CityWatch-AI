import os
import gc
from PIL import Image
import torch
from ultralytics import YOLO

# Hardware computation device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 deep learning model weights
model = YOLO("yolov8n.pt")
model.to(DEVICE)

# Non-civic COCO classes that should NEVER be labeled as municipal defects
IRRELEVANT_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "person", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "remote", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}


def optimize_and_save_image(upload_file, target_path: str):
    """Downsamples smartphone photographs to prevent memory spikes on servers."""
    try:
        image = Image.open(upload_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        image.save(target_path, "JPEG", quality=80, optimize=True)
    finally:
        upload_file.file.close()


def run_yolo_multi_hazard_triage(raw_img_path: str, annotated_img_path: str, selected_category: str) -> dict:
    """Executes YOLOv8 object detection, discards non-civic objects (like birds), and scores hazard severity."""
    try:
        with torch.inference_mode():
            results = model(raw_img_path, imgsz=320, device=DEVICE, verbose=False)

        # Plot detection overlays
        plot_bgr = results[0].plot()
        plot_rgb = plot_bgr[..., ::-1]
        annotated_img = Image.fromarray(plot_rgb)
        annotated_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        annotated_img.save(annotated_img_path, "JPEG", quality=80, optimize=True)

        valid_detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id].lower()
            conf = float(box.conf[0].item())

            # Skip animals and domestic objects (e.g. water puddles flagged as 'bird')
            if class_name in IRRELEVANT_CLASSES:
                continue

            valid_detections.append({"label": class_name, "confidence": round(conf, 2)})

        cat_lower = selected_category.lower()
        if any(k in cat_lower for k in ["fallen tree", "vegetation", "manhole", "drain", "electrical", "wire"]):
            severity = "Critical"
        elif "pothole" in cat_lower or "surface" in cat_lower:
            severity = "Medium"
        else:
            severity = "Low"

        del results
        gc.collect()

        return {
            "hazard_type": selected_category,
            "severity": severity,
            "detections": valid_detections
        }
    except Exception as e:
        print(f"[AI PIPELINE ERROR] {e}")
        Image.open(raw_img_path).save(annotated_img_path, "JPEG", quality=80)
        return {"hazard_type": selected_category, "severity": "Medium", "detections": []}