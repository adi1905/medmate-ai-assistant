from ..connection import get_conn
import uuid
from ..utils.datetime_utils import now_ist_str

def fetch_appointments(patient_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE patient_id = ? ORDER BY scheduled_time DESC",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]

def get_appointment_by_id(appointment_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
        ).fetchone()
        return dict(row) if row else None


def insert_appointment(patient_id: str, provider_id: str, scheduled_time: str, reason: str) -> str:
    with get_conn() as conn:
        appt_id = f"AP-{uuid.uuid4().hex[:6]}"
        now_ist = now_ist_str()
        conn.execute(
            """INSERT INTO appointments (appointment_id, patient_id, provider_id, scheduled_time,
               reason, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?)""",
            (appt_id, patient_id, provider_id, scheduled_time, reason, now_ist, now_ist),
        )
        return appt_id

def update_appointment_status(appointment_id: str, status: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE appointments SET status = ?, updated_at = datetime('now') WHERE appointment_id = ?",
            (status, appointment_id),
        )
        return cur.rowcount > 0
    
def update_appointment_fields(appointment_id: str, scheduled_time: str | None = None,
                              provider_id: str | None = None, reason: str | None = None)-> bool:
    """Partial update — only non-None fields are written."""
    fields = {
        "scheduled_time": scheduled_time,
        "provider_id": provider_id,
        "reason": reason,
    }
    updates = {k: v for k, v in fields.items() if v is not None}
    if not updates:
        return None
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values())
    now_ist = now_ist_str()

    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE appointments SET {set_clause}, updated_at = ? WHERE appointment_id = ?",
            values + [now_ist, appointment_id],
        )
        return cur.rowcount > 0
    

def cancel_appointment_db(appointment_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            """UPDATE appointments SET status = 'cancelled', updated_at = ?
               WHERE appointment_id = ? AND status = 'scheduled'""",
            (now_ist_str(), appointment_id),
        )
        return cur.rowcount > 0