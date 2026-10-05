import os
import uuid
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role key for backend uploads

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_image_to_supabase(image_bytes: bytes, filename: str) -> str:
    """Uploads downsampled image directly to Supabase Storage and returns the public URL."""
    file_path = f"reports/{uuid.uuid4().hex[:8]}_{filename}"
    
    # Upload to public bucket
    supabase.storage.from_("hazard-images").upload(
        path=file_path,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )
    
    # Get permanent public URL
    return supabase.storage.from_("hazard-images").get_public_url(file_path)

def save_report_to_db(ticket_data: dict):
    """Inserts a new report row permanently into Supabase PostgreSQL."""
    response = supabase.table("hazard_reports").insert(ticket_data).execute()
    return response.data

def fetch_all_reports():
    """Fetches all active and historical incidents for the map."""
    response = supabase.table("hazard_reports").select("*").order("created_at", desc=True).execute()
    return response.data