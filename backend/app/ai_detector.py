import os
from ultralytics import YOLO

# 1. Load the pre-trained lightweight YOLOv8 nano model
# This automatically downloads 'yolov8n.pt' (~6MB) on your first run
model = YOLO("yolov8n.pt")

def detect_hazard(image_path: str) -> list:
    """
    Takes a file path or URL to an image, runs YOLOv8 object detection,
    and returns a structured list of detected objects with confidence scores.
    """
    # 2. Run inference on the image
    results = model(image_path)

    detected_items = []

    # 3. Loop through results and extract bounding box metadata
    for result in results:
        for box in result.boxes:
            # Get class ID (e.g., 0, 1) and map to class name (e.g., 'person', 'car', 'pothole')
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
            
            # Confidence score between 0.00 and 1.00
            confidence = float(box.conf[0].item())
            
            # Bounding box pixel coordinates: [x_min, y_min, x_max, y_max]
            coords = [round(c, 1) for c in box.xyxy[0].tolist()]

            detected_items.append({
                "label": label,
                "confidence": round(confidence, 2),
                "box": coords
            })

    return detected_items

# --- Standalone Test Execution ---
if __name__ == "__main__":
    # Test using a standard online test image
    test_image_url = "https://ultralytics.com/images/bus.jpg"
    print("Running AI detection test...")
    
    predictions = detect_hazard(test_image_url)
    
    print("\n--- Detection Results ---")
    for item in predictions:
        print(f"Found: {item['label']} | Confidence: {item['confidence']} | Box: {item['box']}")