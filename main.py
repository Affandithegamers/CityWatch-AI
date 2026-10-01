# main.py
import os
import math
import uuid
from datetime import datetime
from typing import Optional
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import (
    init_db, register_user, authenticate_user, insert_report,
    get_all_reports, get_report_by_id, update_report_status,
    upvote_report, resolve_report_with_image, increment_duplicate, get_database_stats
)
from ai_detector import run_yolo_multi_hazard_triage

app = FastAPI(
    title="CityWatch AI",
    description="Intelligent Municipal Multi-Hazard Triage & Spatial Deduplication Platform",
    version="2.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
STATIC_DIR = "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize database and migrations
init_db()

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates spatial distance between two GPS coordinates in meters."""
    earth_radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    return earth_radius_m * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))

def optimize_and_save_image(upload_file: UploadFile, target_path: str):
    """Downsamples uploads to max 800px to maintain stability on free-tier RAM."""
    try:
        image = Image.open(upload_file.file)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        image.save(target_path, "JPEG", quality=80, optimize=True)
    finally:
        upload_file.file.close()

# --- Authentication Endpoints ---

class RegisterPayload(BaseModel):
    email: str
    password: str
    full_name: str
    role: Optional[str] = "citizen"

class LoginPayload(BaseModel):
    email: str
    password: str

@app.post("/api/v1/auth/register")
def register(payload: RegisterPayload):
    try:
        user = register_user(payload.email, payload.password, payload.full_name, payload.role)
        return {"success": True, "message": "Account created successfully.", "user": user}
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/auth/login")
def login(payload: LoginPayload):
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"success": True, "user": user}

# --- Incident & Operations Endpoints ---

@app.get("/api/v1/stats")
def stats():
    return get_database_stats()

@app.get("/api/v1/reports")
def list_reports(user_email: Optional[str] = Query(None)):
    data = get_all_reports(user_email)
    return {"count": len(data), "data": data}

@app.get("/api/v1/report/{report_id}")
def single_report(report_id: str):
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Ticket not found in database.")
    return report

@app.post("/api/v1/report")
async def create_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    category: str = Form("Road Surface (Pothole / Sinkhole / Crack)"),
    notes: Optional[str] = Form(None),
    reported_by: str = Form("anonymous@citizen.my"),
    image: UploadFile = File(...)
):
    try:
        file_id = uuid.uuid4().hex[:10]
        raw_path = os.path.join(UPLOAD_DIR, f"raw_{file_id}.jpg")
        annotated_path = os.path.join(UPLOAD_DIR, f"ai_{file_id}.jpg")

        optimize_and_save_image(image, raw_path)
        ai_triage = run_yolo_multi_hazard_triage(raw_path, annotated_path, category)

        # 10m Spatial Proximity Clustering Check
        existing_reports = get_all_reports()
        is_duplicate = False
        target_parent_id = None

        for existing in existing_reports:
            if existing["status"] != "Resolved":
                dist = calculate_haversine_distance(latitude, longitude, existing["latitude"], existing["longitude"])
                if dist <= 10.0:
                    is_duplicate = True
                    target_parent_id = existing["report_id"]
                    increment_duplicate(target_parent_id)
                    break

        ticket = {
            "report_id": f"REP-{uuid.uuid4().hex[:6].upper()}",
            "latitude": latitude,
            "longitude": longitude,
            "hazard_type": ai_triage["hazard_type"],
            "severity": ai_triage["severity"],
            "status": "Merged (Duplicate)" if is_duplicate else "AI Verified",
            "notes": notes if notes else "No remarks provided.",
            "image_url": f"/uploads/ai_{file_id}.jpg",
            "reported_by": reported_by.strip().lower(),
            "is_duplicate": is_duplicate,
            "parent_report_id": target_parent_id,
            "duplicate_count": 1 if not is_duplicate else 0,
            "upvote_count": 0,
            "resolved_image_url": None,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if not is_duplicate:
            insert_report(ticket)

        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "message": f"Merged into ticket {target_parent_id}" if is_duplicate else "Registered and AI verified.",
            "ticket": ticket
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/report/{report_id}/upvote")
def upvote(report_id: str):
    """Community Endorsement: Increments upvotes for verified map pins."""
    new_votes = upvote_report(report_id)
    return {"success": True, "report_id": report_id, "upvotes": new_votes}

@app.post("/api/v1/report/{report_id}/resolve")
async def resolve_with_photo(
    report_id: str,
    after_image: UploadFile = File(...),
    role: str = Query("admin")
):
    """Contractor Verification: Uploads resolution proof and marks work order resolved."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can submit repair proof.")
    
    file_id = uuid.uuid4().hex[:10]
    target_path = os.path.join(UPLOAD_DIR, f"resolved_{file_id}.jpg")
    optimize_and_save_image(after_image, target_path)
    resolved_url = f"/uploads/resolved_{file_id}.jpg"
    
    success = resolve_report_with_image(report_id, resolved_url)
    if not success:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return {"success": True, "report_id": report_id, "resolved_image_url": resolved_url}

@app.patch("/api/v1/report/{report_id}/status")
def patch_status(report_id: str, new_status: str, role: Optional[str] = Query("admin")):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can update work orders.")
    success = update_report_status(report_id, new_status)
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "report_id": report_id, "updated_status": new_status}

# Mount static site root
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)