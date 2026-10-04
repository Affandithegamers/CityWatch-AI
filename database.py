import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = "citywatch.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'citizen',
            created_at TEXT NOT NULL
        )
    """)

    # 2. Reports Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            report_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            hazard_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT,
            image_url TEXT NOT NULL,
            resolved_image_url TEXT,
            reported_by TEXT NOT NULL,
            is_duplicate INTEGER DEFAULT 0,
            parent_report_id TEXT,
            duplicate_count INTEGER DEFAULT 1,
            upvote_count INTEGER DEFAULT 0,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (reported_by) REFERENCES users (email)
        )
    """)

    # 3. Upvotes Tracking Table (Enforces 1 vote per user & enables undo/downvote)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_upvotes (
            report_id TEXT NOT NULL,
            user_email TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (report_id, user_email)
        )
    """)

    # Automatic schema migrations
    cursor.execute("PRAGMA table_info(reports)")
    columns = [col["name"] for col in cursor.fetchall()]
    if "upvote_count" not in columns:
        cursor.execute("ALTER TABLE reports ADD COLUMN upvote_count INTEGER DEFAULT 0")
    if "resolved_image_url" not in columns:
        cursor.execute("ALTER TABLE reports ADD COLUMN resolved_image_url TEXT")

    conn.commit()

    # Pre-seed default academic accounts
    seed_users = [
        ("citizen@citywatch.my", "user123", "Muhammad Affandi (Citizen)", "citizen"),
        ("admin@dbkl.gov.my", "admin123", "En. Razak (DBKL Operations)", "admin")
    ]
    for email, pwd, name, role in seed_users:
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (email, password_hash, full_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (email, hash_password(pwd), name, role, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )

    conn.commit()
    conn.close()


def register_user(email: str, password: str, full_name: str, role: str = "citizen") -> Dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_email = email.strip().lower()
    pw_hash = hash_password(password)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO users (email, password_hash, full_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (clean_email, pw_hash, full_name.strip(), role.strip().lower(), now_str)
    )
    conn.commit()
    conn.close()
    return {"email": clean_email, "full_name": full_name, "role": role}


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT email, full_name, role FROM users WHERE email = ? AND password_hash = ?",
        (email.strip().lower(), hash_password(password))
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"email": row["email"], "full_name": row["full_name"], "role": row["role"]}
    return None


def insert_report(report: Dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reports (
            report_id, latitude, longitude, hazard_type, severity,
            status, notes, image_url, resolved_image_url, reported_by,
            is_duplicate, parent_report_id, duplicate_count, upvote_count, submitted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report["report_id"], report["latitude"], report["longitude"],
        report["hazard_type"], report["severity"], report["status"],
        report["notes"], report["image_url"], report.get("resolved_image_url"),
        report["reported_by"], 1 if report["is_duplicate"] else 0,
        report["parent_report_id"], report["duplicate_count"],
        report.get("upvote_count", 0), report["submitted_at"]
    ))
    conn.commit()
    conn.close()


def get_all_reports(user_email: Optional[str] = None) -> List[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_email:
        cursor.execute("SELECT * FROM reports WHERE reported_by = ? ORDER BY submitted_at DESC", (user_email.strip().lower(),))
    else:
        cursor.execute("SELECT * FROM reports WHERE is_duplicate = 0 ORDER BY submitted_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_report_by_id(report_id: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE report_id = ?", (report_id.strip().upper(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def toggle_upvote_status(report_id: str, user_email: str) -> Dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_id = report_id.strip().upper()
    clean_email = user_email.strip().lower()

    cursor.execute(
        "SELECT 1 FROM report_upvotes WHERE report_id = ? AND user_email = ?",
        (clean_id, clean_email)
    )
    already_voted = cursor.fetchone() is not None

    if already_voted:
        # CANCEL / UNDO (-1)
        cursor.execute(
            "DELETE FROM report_upvotes WHERE report_id = ? AND user_email = ?",
            (clean_id, clean_email)
        )
        cursor.execute(
            "UPDATE reports SET upvote_count = MAX(0, upvote_count - 1) WHERE report_id = ?",
            (clean_id,)
        )
        has_upvoted = False
    else:
        # ADD UPVOTE (+1)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO report_upvotes (report_id, user_email, created_at) VALUES (?, ?, ?)",
            (clean_id, clean_email, now_str)
        )
        cursor.execute(
            "UPDATE reports SET upvote_count = upvote_count + 1 WHERE report_id = ?",
            (clean_id,)
        )
        has_upvoted = True

    conn.commit()

    cursor.execute("SELECT upvote_count FROM reports WHERE report_id = ?", (clean_id,))
    row = cursor.fetchone()
    count = row["upvote_count"] if row else 0
    conn.close()

    return {"report_id": clean_id, "has_upvoted": has_upvoted, "upvote_count": count}


def get_user_upvoted_ids(user_email: str) -> List[str]:
    if not user_email:
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT report_id FROM report_upvotes WHERE user_email = ?", (user_email.strip().lower(),))
    rows = cursor.fetchall()
    conn.close()
    return [r["report_id"] for r in rows]


def update_report_status(report_id: str, new_status: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reports SET status = ? WHERE report_id = ?", (new_status, report_id))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def mark_report_resolved_with_image(report_id: str, resolved_image_url: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE reports SET status = 'Resolved', resolved_image_url = ? WHERE report_id = ?",
        (resolved_image_url, report_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def get_database_stats() -> Dict:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS total FROM reports WHERE is_duplicate = 0")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) AS critical FROM reports WHERE severity = 'Critical' AND is_duplicate = 0")
    critical = cursor.fetchone()["critical"]

    cursor.execute("SELECT COUNT(*) AS resolved FROM reports WHERE status = 'Resolved' AND is_duplicate = 0")
    resolved = cursor.fetchone()["resolved"]

    cursor.execute("SELECT SUM(upvote_count) AS total_upvotes FROM reports WHERE is_duplicate = 0")
    upvotes_row = cursor.fetchone()["total_upvotes"]
    total_upvotes = upvotes_row if upvotes_row else 0

    conn.close()
    return {
        "total_reports": total,
        "critical_count": critical,
        "resolved_count": resolved,
        "duplicates_prevented": total_upvotes
    }