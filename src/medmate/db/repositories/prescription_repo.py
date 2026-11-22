from typing import Optional
from ..connection import get_conn
from datetime import datetime

def fetch_prescriptions(patient_id: str, status: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        query = "SELECT * FROM prescriptions WHERE patient_id = ?"
        params = [patient_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY start_date DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

def get_prescription_by_id(prescription_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM prescriptions WHERE prescription_id = ?", (prescription_id,)
        ).fetchone()
        return dict(row) if row else None

def update_prescription(prescription_id: str, dosage: str = None, end_date: str = None, status: str = None) -> dict:
    fields = {
        "dosage": dosage,
        "end_date": end_date,
        "status": status,
    }
    updates = {k:v for k, v in fields.items() if v is not None}
    if not updates:
        raise ValueError("No fields provided to update.")
    set_clause = ", ".join(f"{col} = ?" for col in updates)
    params = list(updates.values())
    params.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    params.append(prescription_id)
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE prescriptions SET {set_clause}, updated_at = ? WHERE prescription_id = ?",
            params,
        )
        conn.commit()
        if cur.rowcount == 0:
            raise ValueError(f"No prescription found")
        
        return get_prescription_by_id(prescription_id)

    
