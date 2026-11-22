from ..connection import get_conn
from datetime import datetime
from typing import List


ALLOWED_TABLES = {
    "patients", "providers", "appointments",
    "prescriptions", "medications_reference", "lab_results"
}

def get_no_show_rate(start: str, end: str) -> dict:
    """Breakdown of appointment statuses in a period, with computed no-show rate."""
    query = """
        SELECT status, COUNT(*) as cnt
        FROM appointments
        WHERE scheduled_time BETWEEN ? AND ?
        GROUP BY status
    """
    with get_conn() as conn:
        rows = conn.execute(query, (start, end,)).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}

    total = sum(counts.values())
    no_shows = counts.get("no_show",0)
    return {
        "start": start,
        "end": end,
        "total_appointments": total,
        "no_show_count": no_shows,
        "no_show_rate": round(no_shows / total, 4) if total else 0.0,
        "breakdown": counts,
    }

def get_appointment_volume_trend(start: str, end: str, bucket: str = "day") -> list[dict]:
    """Appointment counts bucketed by day/week/month."""
    bucker_expr = {
        "day":"date(scheduled_time)",
        "week":"strftime('%Y-W%W', scheduled_time)",
        "month":"strftime('%Y-%m', scheduled_time)",
    }.get(bucket)
    if not bucker_expr:
        raise ValueError(f"Invalid bucket: {bucket}. Use day, week, or month.")
    query = f"""
        SELECT {bucker_expr} as period, COUNT(*) as cnt
        FROM appointments
        WHERE scheduled_time BETWEEN ? AND ?
        GROUP by period,
        ORDER by period ASC    
    """
    with get_conn() as conn:
        rows = conn.execute(query, (start, end, )).fetchall()
        return [dict[r] for r in rows]
    
def get_top_medications(limit: int = 10) -> list[dict]:
    """Most prescribed active medications, by patient count."""
    query = """
        SELECT medication_name, COUNT(*) as cnt
        FROM prescriptions
        WHERE status = 'active'
        GROUP by medication_name
        ORDER BY cnt DESC
        LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]
    
def get_flagged_lab_summary(start: str, end: str) -> dict:
    """Count and breakdown of flagged lab results in a period."""
    query = """
        SELECT test_name, COUNT(*) as cnt
        FROM lab_results
        WHERE flagged = 1 AND recorded_at BETWEEN ? AND ?
        GROUP BY test_name
        ORDER BY cnt DESC
    """
    with get_conn() as conn:
        rows = conn.execute(query, (start, end)).fetchall()
        breakdown = [dict(r) for r in rows]
    return {
        "start": start,
        "end": end,
        "total_flagged": sum(r["cnt"] for r in breakdown),
        "breakdown": breakdown,
    }

def get_provider_utilization(provider_id: str, start: str, end: str) -> dict:
    """Appointment status breakdown for a single provider in a period."""
    query = """
        SELECT status, COUNT(*) as cnt
        FROM appointments
        WHERE provider_id = ? AND scheduled_time BETWEEN ? AND ?
        GROUP BY status
    """
    with get_conn() as conn:
        rows = conn.execute(query, (provider_id, start, end)).fetchall()
        counts = {r["status"]: r["cnt"] for r in rows}
    return {
        "provider_id": provider_id,
        "start": start,
        "end": end,
        "total_appointments": sum(counts.values()),
        "breakdown": counts,
    }





OPTIONAL_AUDIT_COLUMNS = {"created_at", "updated_at"}


def get_schema_snapshot(include_samples: bool = True) -> str:
    """
    Introspects sqlite_master + PRAGMA table_info for allowed tables.
    Returns a formatted string used to ground the text-to-SQL LLM prompt.
    Includes columns, types, FKs, CHECK-constraint enums, and 2-3 sample rows.
    """

    ENUMS = {
        ("appointments", "status"): [
            "scheduled",
            "completed",
            "cancelled",
            "no_show",
        ],
        ("prescriptions", "status"): [
            "active",
            "completed",
            "discontinued",
        ],
        ("patients", "sex"): [
            "M",
            "F",
        ],
        ("lab_results", "flagged"): [
            "0",
            "1",
        ],
    }

    lines: List[str] = []

    with get_conn() as conn:

        for table in sorted(ALLOWED_TABLES):

            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()

            fk_lookup = {
                fk["from"]: f"{fk['table']}.{fk['to']}"
                for fk in fks
            }

            column_strings = []

            for col in cols:

                name = col["name"]

                # Remove noisy audit columns if desired
                if name in OPTIONAL_AUDIT_COLUMNS:
                    continue

                text = name

                if col["pk"]:
                    text += " PK"

                if name in fk_lookup:
                    text += f" -> {fk_lookup[name]}"

                enum = ENUMS.get((table, name))
                if enum:
                    text += " {" + "|".join(enum) + "}"

                column_strings.append(text)

            lines.append(
                f"{table}({', '.join(column_strings)})"
            )

            if include_samples:
                row = conn.execute(
                    f"SELECT * FROM {table} LIMIT 1"
                ).fetchone()

                if row:

                    sample = {}

                    for k, v in dict(row).items():

                        if k in OPTIONAL_AUDIT_COLUMNS:
                            continue

                        if v is None:
                            continue

                        # Skip very long strings (addresses, notes, etc.)
                        if isinstance(v, str) and len(v) > 30:
                            continue

                        sample[k] = v

                    if sample:
                        sample_text = ", ".join(
                            f"{k}={v}"
                            for k, v in sample.items()
                        )
                        lines.append(f"Example: {sample_text}")

            lines.append("")

    return "\n".join(lines)

def run_readonly_query(sql: str, params: tuple = ()) -> tuple[list[str], list[dict]]:
    """
    Executes LLM-generated SQL against a read-only connection.
    This is the ONLY function in the codebase that runs dynamic/unvalidated SQL —
    callers must run sql_guard.validate_sql() before calling this.
    """
    with get_conn(readonly=True) as conn:
        cur = conn.execute(sql, params)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(r) for r in cur.fetchall()]
        return columns, rows


# schema = get_schema_snapshot()
# print(schema)