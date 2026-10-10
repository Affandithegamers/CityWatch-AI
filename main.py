"""
===============================================================================
CITYWATCH AI - FASTAPI BACKEND GATEWAY & SPATIAL DEDUPLICATION ENGINE
===============================================================================
Module: main.py
Version: 3.6.1
Description:
  - Connects frontend requests to the database and computer vision pipeline.
  - Implements the Haversine trigonometric formula for 20m spatial deduplication.
  - Generates unique ticket codes (e.g., REP-A1B2C3).
  - Enforces municipal administration access control for closing/deleting reports.
===============================================================================
"""

import os
import math
import uuid
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Database Data Access Objects (DAO)
from database import (
    init_db, register_user, authenticate_user, insert_report,
    get_all_reports, get_report_by_id, update_report_status,
    mark_report_resolved_with_image, toggle_upvote_status,
    add_endorsement_on_duplicate, get_user_upvoted_ids, get_database_stats,
    delete_report
)

# Computer Vision Inference Pipeline
from ai_detector import optimize_and_save_image, run_yolo_multi_hazard_triage

# -----------------------------------------------------------------------------
# 1. APPLICATION SETUP & MIDDLEWARE
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CityWatch AI",
    description="Municipal Multi-Hazard Triage & Spatial Deduplication Platform",
    version="3.6.1"
)

# Enable Cross-Origin Resource Sharing (CORS) for external browser clients
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

# Mount `/uploads` so annotated photos can be viewed directly in the browser
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize database schema on startup
init_db()

# Spatial buffer threshold (absorbs mobile GPS drift; set to 20 meters)
DUPLICATE_RADIUS_METERS = 20.0


# -----------------------------------------------------------------------------
# 2. MATHEMATICAL & SECURITY UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in meters.
    Formula:
      a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
      c = 2 * atan2(√a, √(1−a))
      d = R * c  (where R = 6,371,000 meters)
    """
    earth_radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    return earth_radius_m * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def compute_file_hash(upload_file: UploadFile) -> str:
    """
    Generates an MD5 checksum of uploaded photo bytes.
    Used to catch identical duplicate photo uploads instantly.
    """
    content = upload_file.file.read()
    upload_file.file.seek(0)  # Reset read pointer so file can be saved later
    return hashlib.md5(content).hexdigest()


# -----------------------------------------------------------------------------
# 3. AUTHENTICATION ENDPOINTS
# -----------------------------------------------------------------------------
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
    """Registers a new citizen or municipal authority account."""
    try:
        user = register_user(payload.email, payload.password, payload.full_name, payload.role)
        return {"success": True, "user": user}
    except Exception as e:
        if "UNIQUE constraint failed" in str(e) or "duplicate key" in str(e):
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/auth/login")
def login(payload: LoginPayload):
    """Authenticates credentials and returns user session metadata."""
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"success": True, "user": user}


# -----------------------------------------------------------------------------
# 4. DATA RETRIEVAL & TRACKING ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/api/v1/stats")
def stats():
    """Returns municipal KPIs for the operations dashboard."""
    return get_database_stats()


@app.get("/api/v1/reports")
def list_reports(user_email: Optional[str] = Query(None)):
    """Fetches all tickets, with personalized upvote flags for the active user."""
    data = get_all_reports()
    user_upvotes = set(get_user_upvoted_ids(user_email)) if user_email else set()
    for report in data:
        report["has_upvoted"] = report["report_id"] in user_upvotes
    return {"count": len(data), "data": data}


@app.get("/api/v1/report/{report_id}")
def single_report(report_id: str):
    """Retrieves milestone tracking information for a specific ticket."""
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Incident ticket not found.")
    return report


# -----------------------------------------------------------------------------
# 5. COMMUNITY ENGAGEMENT (UPVOTES)
# -----------------------------------------------------------------------------
@app.post("/api/v1/report/{report_id}/upvote")
def upvote_ticket(report_id: str, user_email: Optional[str] = Query(None)):
    """Allows logged-in citizens to register a '👍 I see this too!' endorsement."""
    if not user_email or user_email.strip().lower() in ["anonymous", "guest_user", "guest", "null", "undefined", ""]:
        raise HTTPException(
            status_code=401,
            detail="Authentication required: Please sign in or register to upvote reports."
        )
    result = toggle_upvote_status(report_id, user_email)
    return {"success": True, **result}


# -----------------------------------------------------------------------------
# 6. MUNICIPAL DISPATCH & WORK-ORDER MANAGEMENT
# -----------------------------------------------------------------------------
@app.post("/api/v1/report/{report_id}/resolve")
def resolve_ticket(report_id: str, role: str = Query(...), after_image: UploadFile = File(...)):
    """Closes a ticket and attaches photographic proof of contractor repair."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can resolve work orders.")

    file_id = uuid.uuid4().hex[:10]
    target_path = os.path.join(UPLOAD_DIR, f"resolved_{file_id}.jpg")
    optimize_and_save_image(after_image, target_path)

    success = mark_report_resolved_with_image(report_id, f"/uploads/resolved_{file_id}.jpg")
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "report_id": report_id}


@app.patch("/api/v1/report/{report_id}/status")
def patch_status(report_id: str, new_status: str, role: Optional[str] = Query("admin")):
    """Updates ticket lifecycle state (AI Verified -> Dispatched -> Resolved)."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can modify tickets.")
    success = update_report_status(report_id, new_status)
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "report_id": report_id, "updated_status": new_status}


@app.delete("/api/v1/report/{report_id}")
def remove_report(report_id: str, role: Optional[str] = Query("admin")):
    """Permanently deletes an invalid or prank report from the database."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can delete reports.")
    success = delete_report(report_id)
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "message": f"Report {report_id} permanently deleted."}


# -----------------------------------------------------------------------------
# 7. MAIN INCIDENT INTAKE & COMPOUND DEDUPLICATION PIPELINE
# -----------------------------------------------------------------------------

@app.post("/api/v1/report")
async def create_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    category: str = Form(...),
    notes: Optional[str] = Form(None),
    reported_by: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    try:
        # 1. Enforce user authentication
        if not reported_by or reported_by.strip().lower() in ["anonymous@citizen.my", "guest", "null", "undefined", ""]:
            raise HTTPException(status_code=401, detail="Authentication required: Guests cannot submit hazard photographs.")

        # 2. Compute image MD5 checksum
        img_hash = compute_file_hash(image)

        # 3. Create unique file paths and downscale photo
        file_id = uuid.uuid4().hex[:10]
        raw_path = os.path.join(UPLOAD_DIR, f"raw_{file_id}.jpg")
        annotated_path = os.path.join(UPLOAD_DIR, f"ai_{file_id}.jpg")

        optimize_and_save_image(image, raw_path)

        # 4. Run Computer Vision Inference
        ai_triage = run_yolo_multi_hazard_triage(raw_path, annotated_path, category)

        # 5. INITIALIZE ALL DEDUPLICATION VARIABLES FIRST
        existing_reports = get_all_reports()
        is_duplicate = False
        target_parent_id = None
        duplicate_reason = ""
        is_spam = ai_triage.get("is_spam", False)

        # Only check spatial deduplication for genuine civic defects
        if not is_spam:
            for existing in existing_reports:
                same_photo = bool(img_hash and existing.get("image_hash") == img_hash)
                dist = calculate_haversine_distance(latitude, longitude, existing["latitude"], existing["longitude"])
                same_spot = (dist <= DUPLICATE_RADIUS_METERS)

                existing_hazard = existing.get("hazard_type", "").lower()
                new_hazard = ai_triage["hazard_type"].lower()
                same_category = (new_hazard in existing_hazard or existing_hazard in new_hazard)
                is_active = existing.get("status") not in ["Resolved", "Closed"]

                # Match A: Exact same photo uploaded again
                if same_photo:
                    is_duplicate = True
                    target_parent_id = existing.get("parent_report_id") or existing["report_id"]
                    duplicate_reason = "Identical photo already on file"
                    add_endorsement_on_duplicate(target_parent_id, reported_by)
                    break

                # Match B: Within 20m, matching category, actively open
                if same_spot and same_category and is_active:
                    is_duplicate = True
                    target_parent_id = existing.get("parent_report_id") or existing["report_id"]
                    duplicate_reason = f"Nearby report within {round(dist, 1)}m matching hazard type"
                    add_endorsement_on_duplicate(target_parent_id, reported_by)
                    break

        # 6. ASSEMBLE TICKET
        ticket = {
            "report_id": f"REP-{uuid.uuid4().hex[:6].upper()}",
            "latitude": float(latitude),
            "longitude": float(longitude),
            "hazard_type": ai_triage["hazard_type"],
            "severity": ai_triage["severity"],
            "status": "Merged (Duplicate)" if is_duplicate else ai_triage["status"],
            "notes": notes if notes else "No remarks provided.",
            "image_url": f"/uploads/ai_{file_id}.jpg",
            "image_hash": img_hash,
            "resolved_image_url": None,
            "reported_by": reported_by.strip().lower(),
            "is_duplicate": is_duplicate,
            "parent_report_id": target_parent_id,
            "duplicate_count": 1 if not is_duplicate else 0,
            "upvote_count": 0,
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 7. Commit new unique tickets to the database
        if not is_duplicate:
            insert_report(ticket)

        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "target_parent_id": target_parent_id,
            "message": f"Merged into master ticket {target_parent_id} ({duplicate_reason})" if is_duplicate else "Registered and AI verified.",
            "ticket": ticket
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# -----------------------------------------------------------------------------
# 8. SERVE FRONTEND USER INTERFACE
# -----------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)