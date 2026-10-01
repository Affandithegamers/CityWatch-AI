# database.py
import sqlite3
import hashlib
from typing import Optional, List, Dict

DB_PATH = "citywatch.db"

def hash_password(password: str) -> str:
    """Hashes passwords using SHA-256 before database storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables for Users and Multi-Hazard Incident Reports."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table (Role-Based Access Control)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'citizen',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Incident Reports Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        report_id TEXT PRIMARY KEY,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        hazard_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        image_url TEXT,
        reported_by TEXT NOT NULL,
        is_duplicate INTEGER DEFAULT 0,
        parent_report_id TEXT,
        duplicate_count INTEGER DEFAULT 1,
        submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Seed default municipal admin if not already present
    cursor.execute("SELECT id FROM users WHERE email = 'admin@dbkl.gov.my'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            ("admin@dbkl.gov.my", hash_password("admin123"), "En. Razak (DBKL Admin)", "admin")
        )

    conn.commit()
    conn.close()
    print("[*] SQLite Database initialized with Users & Reports tables.")

# --- User Management ---

def register_user(email: str, password: str, full_name: str, role: str = "citizen") -> Dict:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        pw_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
            (email.strip().lower(), pw_hash, full_name.strip(), role.strip().lower())
        )
        conn.commit()
        return {
            "email": email.strip().lower(),
            "full_name": full_name.strip(),
            "role": role.strip().lower()
        }
    finally:
        conn.close()

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    pw_hash = hash_password(password)
    cursor.execute(
        "SELECT email, full_name, role FROM users WHERE email = ? AND password_hash = ?",
        (email.strip().lower(), pw_hash)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"email": row["email"], "full_name": row["full_name"], "role": row["role"]}
    return None

# --- Incident Management ---

def insert_report(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO reports (
        report_id, latitude, longitude, hazard_type, severity,
        status, notes, image_url, reported_by, is_duplicate,
        parent_report_id, duplicate_count, submitted_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["report_id"], data["latitude"], data["longitude"], data["hazard_type"],
        data["severity"], data["status"], data["notes"], data["image_url"],
        data["reported_by"], int(data["is_duplicate"]), data["parent_report_id"],
        data["duplicate_count"], data["submitted_at"]
    ))
    conn.commit()
    conn.close()

def get_all_reports(user_email: Optional[str] = None) -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    if user_email:
        cursor.execute("SELECT * FROM reports WHERE reported_by = ? ORDER BY submitted_at DESC", (user_email.strip().lower(),))
    else:
        cursor.execute("SELECT * FROM reports ORDER BY submitted_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_report_by_id(report_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE report_id = ?", (report_id.strip().upper(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def update_report_status(report_id: str, new_status: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET status = ? WHERE report_id = ?", (new_status, report_id.strip().upper()))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated

def increment_duplicate(parent_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET duplicate_count = duplicate_count + 1 WHERE report_id = ?", (parent_id,))
    conn.commit()
    conn.close()

def get_database_stats() -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM reports")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reports WHERE severity = 'Critical'")
    critical = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reports WHERE status = 'Resolved'")
    resolved = cursor.fetchone()[0]

    cursor.execute("SELECT COALESCE(SUM(duplicate_count - 1), 0) FROM reports WHERE is_duplicate = 0")
    dups = cursor.fetchone()[0]
    conn.close()

    return {
        "total_reports": total,
        "critical_count": critical,
        "resolved_count": resolved,
        "duplicates_prevented": max(0, dups)
    }