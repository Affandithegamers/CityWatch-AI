import os
import math
import shutil
import uuid
from datetime import datetime
from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from ultralytics import YOLO

# Paste your direct connection string from Supabase
DATABASE_URL = "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres"

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def check_and_save_hazard_report(lat: float, lon: float, hazard_type: str, severity: str, notes: str, image_url: str):
    conn = get_db_connection()
    cur = conn.cursor()

    # Native PostGIS 10-meter deduplication using spatial indexing
    # ST_DWithin calculates distance over spheroids using geography casts
    query_dedup = """
        SELECT id, report_code, duplicate_count
        FROM hazard_reports
        WHERE ST_DWithin(
            geom::geography,
            ST_SetSRID(ST_Point(%s, %s), 4326)::geography,
            10.0
        )
        AND status != 'Resolved'
        LIMIT 1;
    """
    cur.execute(query_dedup, (lon, lat))
    match = cur.fetchone()

    if match:
        # Update existing report counter
        cur.execute(
            "UPDATE hazard_reports SET duplicate_count = duplicate_count + 1 WHERE id = %s RETURNING *;",
            (match["id"],)
        )
        updated = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return {"is_duplicate": True, "ticket": updated}

    # Insert brand-new record with PostGIS Point geometry
    report_code = f"REP-{uuid.uuid4().hex[:6].upper()}"
    insert_query = """
        INSERT INTO hazard_reports (report_code, latitude, longitude, geom, hazard_type, severity, notes, image_url)
        VALUES (%s, %s, %s, ST_SetSRID(ST_Point(%s, %s), 4326), %s, %s, %s, %s)
        RETURNING *;
    """
    cur.execute(insert_query, (report_code, lat, lon, lon, lat, hazard_type, severity, notes, image_url))
    new_ticket = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {"is_duplicate": False, "ticket": new_ticket}

# 1. Initialize FastAPI Application
app = FastAPI(
    title="CityWatch AI",
    description="Municipal Hazard Reporting System with YOLOv8 Vision & 10m Spatial Deduplication",
    version="1.0.0"
)

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Storage Directories
UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 4. Mount Uploads and Frontend Static Assets
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 5. Device Configuration (RTX GPU acceleration if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] CityWatch AI is running on computation device: {DEVICE}")

# Load YOLOv8 nano model (automatically downloads on first execution)
model = YOLO("yolov8n.pt")
model.to(DEVICE)

# In-memory database storing all submitted civic reports
REPORTS_DATABASE: List[dict] = []


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two GPS coordinates in meters.
    Used for the 10-meter spatial deduplication check.
    """
    earth_radius_m = 6371000.0  # Earth's mean radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return earth_radius_m * c


def run_yolo_inference(image_path: str) -> dict:
    """
    Executes YOLOv8 object detection on the uploaded image and extracts
    defect classes, confidence metrics, and preliminary risk severity.
    """
    results = model(image_path, device=DEVICE)
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

    # Heuristic hazard severity categorization
    if detected_objects:
        top_prediction = detected_objects[0]["label"].lower()
        top_conf = detected_objects[0]["confidence"]

        # Critical severity triggers
        critical_keywords = ["hole", "crack", "fire", "wire", "traffic light", "stop sign"]
        if any(keyword in top_prediction for keyword in critical_keywords) or top_conf > 0.85:
            severity = "Critical"
        else:
            severity = "Medium"
        primary_defect = detected_objects[0]["label"]
    else:
        primary_defect = "Unclassified Hazard"
        severity = "Low"

    return {
        "primary_defect": primary_defect,
        "severity": severity,
        "detections": detected_objects
    }


# --- API Routes ---

@app.get("/api/v1/health")
def health_status():
    return {
        "status": "online",
        "service": "CityWatch AI Core",
        "device": DEVICE,
        "total_active_reports": len(REPORTS_DATABASE)
    }


@app.get("/api/v1/reports")
def retrieve_all_reports():
    """
    Returns all reports for mapping and admin dashboard visualization.
    """
    return {
        "count": len(REPORTS_DATABASE),
        "data": REPORTS_DATABASE
    }


@app.post("/api/v1/report")
async def submit_hazard_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    notes: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    """
    Processes citizen photo submissions: saves file, runs YOLOv8 detection,
    and runs a 10m spatial radius check to deduplicate incoming tickets.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    # 1. Store uploaded photo to local disk
    extension = image.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4().hex[:10]}.{extension}"
    saved_filepath = os.path.join(UPLOAD_DIR, unique_filename)

    with open(saved_filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # 2. Run Computer Vision Inference
    ai_output = run_yolo_inference(saved_filepath)

    # 3. Geospatial Deduplication (10-meter proximity threshold)
    is_duplicate = False
    target_parent_id = None

    for existing_ticket in REPORTS_DATABASE:
        if existing_ticket["status"] != "Resolved":
            distance = calculate_haversine_distance(
                latitude, longitude,
                existing_ticket["latitude"], existing_ticket["longitude"]
            )
            # If within 10 meters, mark as duplicate of existing ticket
            if distance <= 10.0:
                is_duplicate = True
                target_parent_id = existing_ticket["report_id"]
                existing_ticket["duplicate_count"] += 1
                break

    # 4. Construct Report Record
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
        "detections": ai_output["detections"],
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if not is_duplicate:
        REPORTS_DATABASE.append(report_record)

    return {
        "success": True,
        "is_duplicate": is_duplicate,
        "message": "Report merged with existing ticket within 10-meter radius." if is_duplicate else "New report registered and AI verified.",
        "ticket": report_record
    }


@app.patch("/api/v1/report/{report_id}/status")
def update_report_status(report_id: str, new_status: str):
    """
    Allows municipal administrators to progress ticket state:
    ('AI Verified' -> 'In Progress' -> 'Resolved')
    """
    valid_states = ["AI Verified", "In Progress", "Resolved"]
    if new_status not in valid_states:
        raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of: {valid_states}")

    for report in REPORTS_DATABASE:
        if report["report_id"] == report_id:
            report["status"] = new_status
            return {"success": True, "report_id": report_id, "updated_status": new_status}

    raise HTTPException(status_code=404, detail="Report ID not found.")


# Mount static root frontend at the end to avoid intercepting API endpoints
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")