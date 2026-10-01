import os
import math
import uuid
import gc
from datetime import datetime
from typing import Optional, List
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO

# 1. Initialize FastAPI Application
app = FastAPI(
    title="CityWatch AI",
    description="Municipal Hazard Reporting System with YOLOv8 Vision & 10m Spatial Deduplication",
    version="1.0.0"
)

# 2. CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Create Storage Directories
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount Uploads Directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 4. Device Detection (CPU for Render, CUDA for local RTX laptop)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] CityWatch AI is starting up on compute device: {DEVICE}")

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")
model.to(DEVICE)

# In-memory storage for active reports
REPORTS_DATABASE: List[dict] = []


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance between two GPS coordinates in meters."""
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def optimize_and_save_image(upload_file: UploadFile, target_path: str):
    """
    Downsamples large phone camera images to max 800px to prevent
    Render 512MB RAM Out-Of-Memory (OOM) crashes.
    """
    try:
        image = Image.open(upload_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        image.save(target_path, "JPEG", quality=80, optimize=True)
    finally:
        upload_file.file.close()


def run_yolo_inference(image_path: str) -> dict:
    """Runs YOLOv8 object detection using the dynamically selected device."""
    try:
        with torch.inference_mode():
            # imgsz=320 reduces RAM usage on Render's free tier
            results = model(image_path, imgsz=320, device=DEVICE, verbose=False)

        detected_objects = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                conf = float(box.conf[0].item())
                coordinates = [round(val, 1) for val in box.xyxy[0].tolist()]

                detected_objects.append({
                    "label": class_name,
                    "confidence": round(conf, 2),
                    "box": coordinates
                })

        if detected_objects:
            top_prediction = detected_objects[0]["label"].lower()
            top_conf = detected_objects[0]["confidence"]

            critical_keywords = ["hole", "crack", "fire", "wire", "traffic light", "stop sign"]
            if any(k in top_prediction for k in critical_keywords) or top_conf > 0.80:
                severity = "Critical"
            else:
                severity = "Medium"
            primary_defect = detected_objects[0]["label"]
        else:
            primary_defect = "General Road Defect"
            severity = "Low"

        # Explicit garbage collection to release RAM
        del results
        gc.collect()

        return {
            "primary_defect": primary_defect,
            "severity": severity,
            "detections": detected_objects
        }
    except Exception as e:
        print(f"[AI ERROR] Inference error: {e}")
        return {
            "primary_defect": "General Road Defect",
            "severity": "Low",
            "detections": []
        }


# --- REST API Endpoints ---

@app.get("/api/v1/health")
def health_status():
    return {
        "status": "online",
        "service": "CityWatch AI",
        "device": DEVICE,
        "reports_count": len(REPORTS_DATABASE)
    }


@app.get("/api/v1/reports")
def get_reports():
    return {
        "count": len(REPORTS_DATABASE),
        "data": REPORTS_DATABASE
    }


@app.post("/api/v1/report")
async def create_hazard_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        unique_filename = f"{uuid.uuid4().hex[:10]}.jpg"
        saved_filepath = os.path.join(UPLOAD_DIR, unique_filename)
        optimize_and_save_image(image, saved_filepath)

        # Run AI detection
        ai_output = run_yolo_inference(saved_filepath)

        # 10-meter spatial proximity deduplication check
        is_duplicate = False
        target_parent_id = None
        for existing in REPORTS_DATABASE:
            if existing["status"] != "Resolved":
                distance = calculate_haversine_distance(
                    latitude, longitude,
                    existing["latitude"], existing["longitude"]
                )
                if distance <= 10.0:
                    is_duplicate = True
                    target_parent_id = existing["report_id"]
                    existing["duplicate_count"] += 1
                    break

        # Register ticket record
        report_record = {
            "report_id": f"REP-{uuid.uuid4().hex[:6].upper()}",
            "latitude": latitude,
            "longitude": longitude,
            "hazard_type": ai_output["primary_defect"],
            "severity": ai_output["severity"],
            "status": "Merged (Duplicate)" if is_duplicate else "AI Verified",
            "notes": notes if notes else "No additional remarks.",
            "image_url": f"/uploads/{unique_filename}",
            "is_duplicate": is_duplicate,
            "parent_report_id": target_parent_id,
            "duplicate_count": 1 if not is_duplicate else 0,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if not is_duplicate:
            REPORTS_DATABASE.append(report_record)

        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "message": f"Merged into parent ticket {target_parent_id}" if is_duplicate else "Report registered and AI verified.",
            "ticket": report_record
        }

    except Exception as e:
        print(f"[SERVER ERROR] Error processing report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/v1/report/{report_id}/status")
def update_status(report_id: str, new_status: str):
    for report in REPORTS_DATABASE:
        if report["report_id"] == report_id:
            report["status"] = new_status
            return {"success": True, "report_id": report_id, "updated_status": new_status}
    raise HTTPException(status_code=404, detail="Report ID not found.")


# Mount static website root
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")