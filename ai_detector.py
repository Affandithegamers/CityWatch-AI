import gc
import torch
from PIL import Image
from ultralytics import YOLO

# Hardware compute selection (CUDA GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] CityWatch AI Model initialized on device: {DEVICE}")

# Load YOLOv8 Nano vision model
model = YOLO("yolov8n.pt")
model.to(DEVICE)

# Non-civic COCO classes that should never be labeled as municipal defects
IRRELEVANT_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "person", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "remote", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}

def run_yolo_multi_hazard_triage(raw_img_path: str, annotated_img_path: str, selected_category: str) -> dict:
    """
    Executes YOLOv8 deep learning vision inference:
    1. Feeds the image tensor through convolutional layers.
    2. Draws visual bounding box predictions directly onto the saved photo.
    3. Filters out false-positive non-civic classes (animals, indoor furniture).
    4. Categorizes municipal severity level (Critical, Medium, Low).
    """
    try:
        with torch.inference_mode():
            results = model(raw_img_path, imgsz=320, device=DEVICE, verbose=False)

        # Draw AI bounding boxes onto the image
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

            if class_name in IRRELEVANT_CLASSES:
                continue

            valid_detections.append({
                "label": class_name,
                "confidence": round(conf, 2)
            })

        # Urgency triage logic based on road hazard risk
        cat_lower = selected_category.lower()
        if any(k in cat_lower for k in ["fallen tree", "vegetation", "manhole", "drain cover"]):
            severity = "Critical"
            primary_hazard = selected_category
        elif "traffic light" in cat_lower or "street light" in cat_lower:
            severity = "Critical" if "traffic light" in cat_lower else "Medium"
            primary_hazard = selected_category
        elif "pothole" in cat_lower or "surface" in cat_lower or "sinkhole" in cat_lower:
            severity = "Critical" if any(v["confidence"] > 0.70 for v in valid_detections) else "Medium"
            primary_hazard = selected_category
        elif "illegal waste" in cat_lower or "dumping" in cat_lower:
            severity = "Low"
            primary_hazard = selected_category
        else:
            primary_hazard = selected_category
            severity = "Medium"

        # Enrich hazard label if secondary civic objects are detected
        if valid_detections:
            top_obj = valid_detections[0]
            if top_obj["label"] in ["car", "truck", "bus"]:
                primary_hazard = f"{selected_category} (Vehicle Impact Zone)"
            elif top_obj["label"] in ["traffic light", "stop sign"]:
                primary_hazard = f"{selected_category} (AI Confirmed: {top_obj['label'].title()})"

        del results
        gc.collect()

        return {
            "hazard_type": primary_hazard,
            "severity": severity,
            "detections": valid_detections
        }

    except Exception as e:
        print(f"[AI MODULE ERROR] {e}")
        Image.open(raw_img_path).save(annotated_img_path, "JPEG", quality=80)
        return {
            "hazard_type": selected_category,
            "severity": "Medium",
            "detections": []
        }