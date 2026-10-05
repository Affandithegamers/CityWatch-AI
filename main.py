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

from database import (
    init_db, register_user, authenticate_user, insert_report,
    get_all_reports, get_report_by_id, update_report_status,
    mark_report_resolved_with_image, toggle_upvote_status,
    add_endorsement_on_duplicate, get_user_upvoted_ids, get_database_stats
)
from ai_detector import optimize_and_save_image, run_yolo_multi_hazard_triage

app = FastAPI(title="CityWatch AI", version="3.5.0")

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

init_db()

# Operational radius in meters (incorporates GPS horizontal drift tolerance)
DUPLICATE_RADIUS_METERS = 20.0


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates spatial great-circle distance between two GPS coordinates in meters."""
    earth_radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    return earth_radius_m * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def compute_file_hash(upload_file: UploadFile) -> str:
    """Computes MD5 hash of image bytes for exact duplicate photo detection."""
    content = upload_file.file.read()
    upload_file.file.seek(0)
    return hashlib.md5(content).hexdigest()


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
        return {"success": True, "user": user}
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


@app.get("/api/v1/stats")
def stats():
    return get_database_stats()


@app.get("/api/v1/reports")
def list_reports(user_email: Optional[str] = Query(None)):
    data = get_all_reports()
    user_upvotes = set(get_user_upvoted_ids(user_email)) if user_email else set()
    for report in data:
        report["has_upvoted"] = report["report_id"] in user_upvotes
    return {"count": len(data), "data": data}


@app.get("/api/v1/report/{report_id}")
def single_report(report_id: str):
    report = get_report_by_id(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Incident ticket not found.")
    return report


@app.post("/api/v1/report/{report_id}/upvote")
def upvote_ticket(report_id: str, user_email: Optional[str] = Query(None)):
    if not user_email or user_email.strip().lower() in ["anonymous", "guest_user", "guest", "null", "undefined", ""]:
        raise HTTPException(
            status_code=401,
            detail="Authentication required: Guests cannot upvote reports to prevent spam or pranks. Please sign in or register."
        )
    result = toggle_upvote_status(report_id, user_email)
    return {"success": True, **result}


@app.post("/api/v1/report/{report_id}/resolve")
def resolve_ticket(report_id: str, role: str = Query(...), after_image: UploadFile = File(...)):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can resolve work orders.")

    file_id = uuid.uuid4().hex[:10]
    target_path = os.path.join(UPLOAD_DIR, f"resolved_{file_id}.jpg")
    optimize_and_save_image(after_image, target_path)

    success = mark_report_resolved_with_image(report_id, f"/uploads/resolved_{file_id}.jpg")
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "report_id": report_id}


@app.post("/api/v1/report")
async def create_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    category: str = Form(...),
    notes: Optional[str] = Form(None),
    reported_by: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    if not reported_by or reported_by.strip().lower() in ["anonymous@citizen.my", "guest", "null", "undefined", ""]:
        raise HTTPException(status_code=401, detail="Authentication required: Guests cannot submit hazard photographs.")

    img_hash = compute_file_hash(image)

    file_id = uuid.uuid4().hex[:10]
    raw_path = os.path.join(UPLOAD_DIR, f"raw_{file_id}.jpg")
    annotated_path = os.path.join(UPLOAD_DIR, f"ai_{file_id}.jpg")

    optimize_and_save_image(image, raw_path)
    ai_triage = run_yolo_multi_hazard_triage(raw_path, annotated_path, category)

    existing_reports = get_all_reports()
    is_duplicate = False
    target_parent_id = None
    duplicate_reason = ""

    for existing in existing_reports:
        # Check A: Exact same photo uploaded
        same_photo = (img_hash and existing.get("image_hash") == img_hash)

        # Check B: Same road vicinity within tolerance threshold (20m)
        dist = calculate_haversine_distance(latitude, longitude, existing["latitude"], existing["longitude"])
        same_spot = (dist <= DUPLICATE_RADIUS_METERS)

        # Check C: Category matches
        same_category = (category.lower() in existing["hazard_type"].lower() or existing["hazard_type"].lower() in category.lower())

        if same_photo:
            is_duplicate = True
            target_parent_id = existing["report_id"]
            duplicate_reason = "Identical photo already on file"
            add_endorsement_on_duplicate(target_parent_id, reported_by)
            break

        if same_spot and same_category:
            is_duplicate = True
            target_parent_id = existing["report_id"]
            duplicate_reason = f"Nearby report within {round(dist, 1)}m matching hazard type"
            add_endorsement_on_duplicate(target_parent_id, reported_by)
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
        "image_hash": img_hash,
        "resolved_image_url": None,
        "reported_by": reported_by.strip().lower(),
        "is_duplicate": is_duplicate,
        "parent_report_id": target_parent_id,
        "duplicate_count": 1 if not is_duplicate else 0,
        "upvote_count": 0,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if not is_duplicate:
        insert_report(ticket)

    return {
        "success": True,
        "is_duplicate": is_duplicate,
        "target_parent_id": target_parent_id,
        "message": f"Merged into master ticket {target_parent_id} ({duplicate_reason})" if is_duplicate else "Registered and AI verified.",
        "ticket": ticket
    }


@app.patch("/api/v1/report/{report_id}/status")
def patch_status(report_id: str, new_status: str, role: Optional[str] = Query("admin")):
    if role != "admin":
        raise HTTPException(status_code=403, detail="Unauthorized: Only Municipal Authority accounts can modify tickets.")
    success = update_report_status(report_id, new_status)
    if not success:
        raise HTTPException(status_code=404, detail="Report ID not found.")
    return {"success": True, "report_id": report_id, "updated_status": new_status}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)