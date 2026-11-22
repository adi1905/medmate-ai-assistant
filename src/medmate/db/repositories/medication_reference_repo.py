from typing import Optional
from ..connection import get_conn

def get_reference(medication_name: str) -> Optional[dict]:
    """Static curated table — known_interactions is a JSON list you seed yourself,
    never LLM-generated."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM medications_reference WHERE LOWER(medication_name) = LOWER(?)",
            (medication_name,),
        ).fetchone()
        return dict(row) if row else None