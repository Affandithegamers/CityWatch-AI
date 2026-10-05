import os
import sqlite3
import hashlib
from typing import Optional, List, Dict, Any

DATABASE_URL = os.getenv("DATABASE_URL")
LOCAL_DB_PATH = "citywatch.db"


def hash_password(password: str) -> str:
    """Computes a SHA-256 hexadecimal hash for secure password storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_connection():
    """Returns a PostgreSQL connection if DATABASE_URL is set; otherwise falls back to SQLite."""
    if DATABASE_URL:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def init_db():
    """Initializes schema, applies missing column migrations, and seeds demo accounts."""
    conn = get_connection()
    cur = conn.cursor()

    try:
        if DATABASE_URL:
            # 1. Supabase PostgreSQL Schema
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT DEFAULT 'citizen',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_id TEXT UNIQUE NOT NULL,
                    latitude DOUBLE PRECISION NOT NULL,
                    longitude DOUBLE PRECISION NOT NULL,
                    geom GEOMETRY(Point, 4326),
                    hazard_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'AI Verified',
                    notes TEXT,
                    image_url TEXT,
                    image_hash TEXT,
                    resolved_image_url TEXT,
                    reported_by TEXT NOT NULL,
                    is_duplicate BOOLEAN DEFAULT FALSE,
                    parent_report_id TEXT,
                    duplicate_count INTEGER DEFAULT 1,
                    upvote_count INTEGER DEFAULT 0,
                    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS upvotes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    report_id TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(report_id, user_email)
                );
            """)
            conn.commit()

            # Safe column migrations for PostgreSQL
            for col_sql in [
                "ALTER TABLE reports ADD COLUMN IF NOT EXISTS image_hash TEXT;",
                "ALTER TABLE reports ADD COLUMN IF NOT EXISTS duplicate_count INTEGER DEFAULT 1;",
                "ALTER TABLE reports ADD COLUMN IF NOT EXISTS upvote_count INTEGER DEFAULT 0;",
                "ALTER TABLE reports ADD COLUMN IF NOT EXISTS parent_report_id TEXT;",
                "ALTER TABLE reports ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE;"
            ]:
                try:
                    cur.execute(col_sql)
                    conn.commit()
                except Exception:
                    conn.rollback()

            # Seed default test accounts on Supabase
            cur.execute("""
                INSERT INTO users (email, password_hash, full_name, role)
                VALUES 
                    ('citizen@citywatch.my', %s, 'Citizen Tester', 'citizen'),
                    ('admin@dbkl.gov.my', %s, 'Council Admin', 'admin')
                ON CONFLICT (email) DO NOTHING;
            """, (hash_password("user123"), hash_password("admin123")))
            conn.commit()

        else:
            # 2. Local SQLite Schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    role TEXT DEFAULT 'citizen',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    hazard_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    image_url TEXT,
                    image_hash TEXT,
                    resolved_image_url TEXT,
                    reported_by TEXT NOT NULL,
                    is_duplicate INTEGER DEFAULT 0,
                    parent_report_id TEXT,
                    duplicate_count INTEGER DEFAULT 1,
                    upvote_count INTEGER DEFAULT 0,
                    submitted_at TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS upvotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    UNIQUE(report_id, user_email)
                );
            """)
            conn.commit()

            # Safe column migrations for SQLite
            for col_def in [
                "image_hash TEXT",
                "duplicate_count INTEGER DEFAULT 1",
                "upvote_count INTEGER DEFAULT 0",
                "parent_report_id TEXT",
                "is_duplicate INTEGER DEFAULT 0"
            ]:
                try:
                    cur.execute(f"ALTER TABLE reports ADD COLUMN {col_def};")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass

            # Seed default test accounts on SQLite
            cur.execute("SELECT id FROM users WHERE email = 'citizen@citywatch.my';")
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?);",
                    ("citizen@citywatch.my", hash_password("user123"), "Citizen Tester", "citizen")
                )

            cur.execute("SELECT id FROM users WHERE email = 'admin@dbkl.gov.my';")
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?)",
                    ("admin@dbkl.gov.my", hash_password("admin123"), "Council Admin", "admin")
                )
            conn.commit()

    finally:
        cur.close()
        conn.close()


def register_user(email: str, password: str, full_name: str, role: str = "citizen") -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    clean_email = email.strip().lower()
    pw_hash = hash_password(password)

    try:
        if DATABASE_URL:
            cur.execute(
                "INSERT INTO users (email, password_hash, full_name, role) VALUES (%s, %s, %s, %s);",
                (clean_email, pw_hash, full_name.strip(), role.strip().lower())
            )
        else:
            cur.execute(
                "INSERT INTO users (email, password_hash, full_name, role) VALUES (?, ?, ?, ?);",
                (clean_email, pw_hash, full_name.strip(), role.strip().lower())
            )
        conn.commit()
        return {"email": clean_email, "full_name": full_name.strip(), "role": role.strip().lower()}
    finally:
        cur.close()
        conn.close()


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    clean_email = email.strip().lower()
    pw_hash = hash_password(password)

    try:
        if DATABASE_URL:
            cur.execute(
                "SELECT email, full_name, role FROM users WHERE email = %s AND password_hash = %s;",
                (clean_email, pw_hash)
            )
        else:
            cur.execute(
                "SELECT email, full_name, role FROM users WHERE email = ? AND password_hash = ?;",
                (clean_email, pw_hash)
            )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        cur.close()
        conn.close()


def insert_report(ticket: Dict[str, Any]):
    conn = get_connection()
    cur = conn.cursor()

    try:
        if DATABASE_URL:
            cur.execute("""
                INSERT INTO reports (
                    report_id, latitude, longitude, geom, hazard_type, severity,
                    status, notes, image_url, image_hash, resolved_image_url,
                    reported_by, is_duplicate, parent_report_id, duplicate_count,
                    upvote_count, submitted_at
                ) VALUES (
                    %s, %s, %s, ST_SetSRID(ST_Point(%s, %s), 4326), %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                );
            """, (
                ticket["report_id"], ticket["latitude"], ticket["longitude"],
                ticket["longitude"], ticket["latitude"], ticket["hazard_type"],
                ticket["severity"], ticket["status"], ticket.get("notes", ""),
                ticket.get("image_url", ""), ticket.get("image_hash", ""),
                ticket.get("resolved_image_url"), ticket["reported_by"],
                bool(ticket.get("is_duplicate", False)), ticket.get("parent_report_id"),
                ticket.get("duplicate_count", 1), ticket.get("upvote_count", 0)
            ))
        else:
            cur.execute("""
                INSERT INTO reports (
                    report_id, latitude, longitude, hazard_type, severity,
                    status, notes, image_url, image_hash, resolved_image_url,
                    reported_by, is_duplicate, parent_report_id, duplicate_count,
                    upvote_count, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                ticket["report_id"], ticket["latitude"], ticket["longitude"],
                ticket["hazard_type"], ticket["severity"], ticket["status"],
                ticket.get("notes", ""), ticket.get("image_url", ""),
                ticket.get("image_hash", ""), ticket.get("resolved_image_url"),
                ticket["reported_by"], 1 if ticket.get("is_duplicate") else 0,
                ticket.get("parent_report_id"), ticket.get("duplicate_count", 1),
                ticket.get("upvote_count", 0), ticket["submitted_at"]
            ))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def get_all_reports(user_email: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    try:
        if DATABASE_URL:
            if user_email:
                cur.execute(
                    "SELECT * FROM reports WHERE reported_by = %s ORDER BY submitted_at DESC;",
                    (user_email.strip().lower(),)
                )
            else:
                cur.execute("SELECT * FROM reports ORDER BY submitted_at DESC;")
        else:
            if user_email:
                cur.execute(
                    "SELECT * FROM reports WHERE reported_by = ? ORDER BY id DESC;",
                    (user_email.strip().lower(),)
                )
            else:
                cur.execute("SELECT * FROM reports ORDER BY id DESC;")

        rows = cur.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["is_duplicate"] = bool(d.get("is_duplicate"))
            results.append(d)
        return results
    finally:
        cur.close()
        conn.close()


def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()

    try:
        clean_id = report_id.strip()
        if DATABASE_URL:
            cur.execute("SELECT * FROM reports WHERE UPPER(report_id) = UPPER(%s);", (clean_id,))
        else:
            cur.execute("SELECT * FROM reports WHERE UPPER(report_id) = UPPER(?);", (clean_id,))

        row = cur.fetchone()
        if row:
            d = dict(row)
            d["is_duplicate"] = bool(d.get("is_duplicate"))
            return d
        return None
    finally:
        cur.close()
        conn.close()


def update_report_status(report_id: str, new_status: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    clean_id = report_id.strip()

    try:
        if DATABASE_URL:
            cur.execute("UPDATE reports SET status = %s WHERE UPPER(report_id) = UPPER(%s);", (new_status, clean_id))
        else:
            cur.execute("UPDATE reports SET status = ? WHERE UPPER(report_id) = UPPER(?);", (new_status, clean_id))

        updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        cur.close()
        conn.close()


def mark_report_resolved_with_image(report_id: str, resolved_image_url: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    clean_id = report_id.strip()

    try:
        if DATABASE_URL:
            cur.execute(
                "UPDATE reports SET status = 'Resolved', resolved_image_url = %s WHERE UPPER(report_id) = UPPER(%s);",
                (resolved_image_url, clean_id)
            )
        else:
            cur.execute(
                "UPDATE reports SET status = 'Resolved', resolved_image_url = ? WHERE UPPER(report_id) = UPPER(?);",
                (resolved_image_url, clean_id)
            )

        updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        cur.close()
        conn.close()


def delete_report(report_id: str) -> bool:
    """Permanently deletes a report and its associated upvotes from the database."""
    conn = get_connection()
    cur = conn.cursor()
    clean_id = report_id.strip()

    try:
        if DATABASE_URL:
            cur.execute("DELETE FROM upvotes WHERE UPPER(report_id) = UPPER(%s);", (clean_id,))
            cur.execute("DELETE FROM reports WHERE UPPER(report_id) = UPPER(%s);", (clean_id,))
        else:
            cur.execute("DELETE FROM upvotes WHERE UPPER(report_id) = UPPER(?);", (clean_id,))
            cur.execute("DELETE FROM reports WHERE UPPER(report_id) = UPPER(?);", (clean_id,))

        deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        cur.close()
        conn.close()


def add_endorsement_on_duplicate(report_id: str, user_email: Optional[str]):
    conn = get_connection()
    cur = conn.cursor()
    clean_id = report_id.strip()

    try:
        if DATABASE_URL:
            cur.execute(
                "UPDATE reports SET duplicate_count = duplicate_count + 1 WHERE UPPER(report_id) = UPPER(%s);",
                (clean_id,)
            )
            if user_email and user_email.strip().lower() not in ["anonymous", "guest", "null", "undefined", ""]:
                norm_email = user_email.strip().lower()
                cur.execute("""
                    INSERT INTO upvotes (report_id, user_email) 
                    VALUES (%s, %s) 
                    ON CONFLICT (report_id, user_email) DO NOTHING;
                """, (clean_id, norm_email))
                cur.execute("SELECT COUNT(*) as count FROM upvotes WHERE UPPER(report_id) = UPPER(%s);", (clean_id,))
                new_count = cur.fetchone()["count"]
                cur.execute(
                    "UPDATE reports SET upvote_count = %s WHERE UPPER(report_id) = UPPER(%s);",
                    (new_count, clean_id)
                )
        else:
            cur.execute(
                "UPDATE reports SET duplicate_count = duplicate_count + 1 WHERE UPPER(report_id) = UPPER(?);",
                (clean_id,)
            )
            if user_email and user_email.strip().lower() not in ["anonymous", "guest", "null", "undefined", ""]:
                norm_email = user_email.strip().lower()
                cur.execute(
                    "INSERT OR IGNORE INTO upvotes (report_id, user_email) VALUES (?, ?);",
                    (clean_id, norm_email)
                )
                cur.execute("SELECT COUNT(*) FROM upvotes WHERE UPPER(report_id) = UPPER(?);", (clean_id,))
                new_count = cur.fetchone()[0]
                cur.execute(
                    "UPDATE reports SET upvote_count = ? WHERE UPPER(report_id) = UPPER(?);",
                    (new_count, clean_id)
                )

        conn.commit()
    finally:
        cur.close()
        conn.close()


def toggle_upvote_status(report_id: str, user_email: str) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    norm_email = user_email.strip().lower()
    clean_id = report_id.strip()

    try:
        if DATABASE_URL:
            cur.execute(
                "SELECT id FROM upvotes WHERE UPPER(report_id) = UPPER(%s) AND user_email = %s;",
                (clean_id, norm_email)
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    "DELETE FROM upvotes WHERE UPPER(report_id) = UPPER(%s) AND user_email = %s;",
                    (clean_id, norm_email)
                )
                has_upvoted = False
            else:
                cur.execute(
                    "INSERT INTO upvotes (report_id, user_email) VALUES (%s, %s);",
                    (clean_id, norm_email)
                )
                has_upvoted = True

            cur.execute("SELECT COUNT(*) as count FROM upvotes WHERE UPPER(report_id) = UPPER(%s);", (clean_id,))
            new_count = cur.fetchone()["count"]
            cur.execute(
                "UPDATE reports SET upvote_count = %s WHERE UPPER(report_id) = UPPER(%s);",
                (new_count, clean_id)
            )
        else:
            cur.execute(
                "SELECT id FROM upvotes WHERE UPPER(report_id) = UPPER(?) AND user_email = ?;",
                (clean_id, norm_email)
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    "DELETE FROM upvotes WHERE UPPER(report_id) = UPPER(?) AND user_email = ?;",
                    (clean_id, norm_email)
                )
                has_upvoted = False
            else:
                cur.execute(
                    "INSERT INTO upvotes (report_id, user_email) VALUES (?, ?);",
                    (clean_id, norm_email)
                )
                has_upvoted = True

            cur.execute("SELECT COUNT(*) FROM upvotes WHERE UPPER(report_id) = UPPER(?);", (clean_id,))
            new_count = cur.fetchone()[0]
            cur.execute(
                "UPDATE reports SET upvote_count = ? WHERE UPPER(report_id) = UPPER(?);",
                (new_count, clean_id)
            )

        conn.commit()
        return {"report_id": report_id, "upvotes": new_count, "has_upvoted": has_upvoted}
    finally:
        cur.close()
        conn.close()


def get_user_upvoted_ids(user_email: Optional[str]) -> List[str]:
    if not user_email:
        return []
    conn = get_connection()
    cur = conn.cursor()
    norm_email = user_email.strip().lower()

    try:
        if DATABASE_URL:
            cur.execute("SELECT report_id FROM upvotes WHERE user_email = %s;", (norm_email,))
            rows = cur.fetchall()
            return [r["report_id"] for r in rows]
        else:
            cur.execute("SELECT report_id FROM upvotes WHERE user_email = ?;", (norm_email,))
            rows = cur.fetchall()
            return [r["report_id"] for r in rows]
    finally:
        cur.close()
        conn.close()


def get_database_stats() -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()

    try:
        if DATABASE_URL:
            cur.execute("SELECT COUNT(*) as count FROM reports;")
            total = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) as count FROM reports WHERE severity = 'Critical';")
            critical = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) as count FROM reports WHERE status = 'Resolved';")
            resolved = cur.fetchone()["count"]
            cur.execute("SELECT COALESCE(SUM(duplicate_count), 0) as count FROM reports;")
            duplicates = cur.fetchone()["count"]
        else:
            cur.execute("SELECT COUNT(*) FROM reports;")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reports WHERE severity = 'Critical';")
            critical = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reports WHERE status = 'Resolved';")
            resolved = cur.fetchone()[0]
            cur.execute("SELECT COALESCE(SUM(duplicate_count), 0) FROM reports;")
            duplicates = cur.fetchone()[0]

        return {
            "total_reports": total,
            "critical_count": critical,
            "resolved_count": resolved,
            "duplicates_prevented": max(0, duplicates - total) if duplicates > total else 0
        }
    finally:
        cur.close()
        conn.close()