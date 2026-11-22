from ..connection import get_conn
from datetime import datetime
from typing import Optional
import re


def _normalize_phone(raw: str) -> str:
    """Digits only, then keep just the last 10 -- absorbs formatting
    differences (dashes, parens, spaces) AND a leading country code
    (+1 555 123 4567 and 555-123-4567 must be treated as the same
    number). Assumes US-style 10-digit numbers, matching this schema's
    seed data; adjust the trailing-digit count if you support other
    country formats."""
    digits = re.sub(r"\D", "", raw or "")
    return digits[-10:] if len(digits) >= 10 else digits

def fetch_patient(patient_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None

    

def search_patients(name: str = None, dob: str = None, contact_info: str = None, limit: int = 50) -> list[dict]:
    """Search patients by optional filters. Staff-only"""
    query = "SELECT patient_id, name, dob, contact_info, created_at FROM patients WHERE 1=1"
    params = []

    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if dob:
        query += " AND dob = ?"
        params.append(dob)
    if contact_info:
        query += " AND contact_info LIKE ?"
        params.append(f"%{contact_info}%")

    query += " ORDER BY name ASC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    
def update_patient_contact(patient_id: str, phone: str = None, email: str = None,
                            address: str = None, insurance_provider: str = None) -> dict:
    """Update only contact-related fields for a patient."""
    fields = {
        "email":email,
        "phone": phone,
        "address": address,
        "insurance_provider": insurance_provider
    }
    updates = {k:v for k, v in fields.items() if v is not None}
    if not updates:
        raise ValueError("No fields provided to update.")
    set_clause = ", ".join(f"{col} = ?" for col in updates)
    params = list(updates.values())
    params.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    params.append(patient_id)

    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE patients SET {set_clause}, updated_at = ? WHERE patient_id = ?",
            params,
        )
        # conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"No patient found with id {patient_id}")
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return dict(row)

def find_patient_for_claim(
        last_name: str,
        dob: str,
        phone: str,
        first_name: Optional[str] = None
) -> Optional[dict]:
    """
    Three-factor identity lookup: last_name (case-insensitive) + dob
    (exact) + phone (normalized, digits-only). first_name is accepted but
    NOT filtered on here -- nicknames and minor spelling differences
    shouldn't block a real patient, and the other three facts together
    are already a strong enough check."""
    with get_conn() as conn:
        rows = conn.execute("""
        SELECT patient_id, first_name, last_name, dob, phone
        FROM patients
        WHERE lower(last_name) = lower(?)
          AND dob = ?
        """,
        (last_name.strip(), dob)
        ).fetchall()
    normalized_input_phone = _normalize_phone(phone)
    matches = [r for r in rows if _normalize_phone(r["phone"]) == normalized_input_phone]
    if len(matches) > 1:
        return None
    return dict(matches[0])

def patient_already_claimed(patient_id: str) -> bool:
    """True if a users row already points at this patient_id."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 from users where patient_id = ?",
            (patient_id,)
        ).fetchone()
    return row is not None

