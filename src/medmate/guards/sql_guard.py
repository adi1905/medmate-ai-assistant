"""
Validates and sanitizes LLM-generated SQL before it is ever executed.
This is the single safety gate between text-to-SQL generation and
analytics_repo.run_readonly_query(). No DB connection is used here —
this module only parses and rewrites SQL strings.
"""

import sqlglot
from sqlglot import exp

FORBIDDEN_STATEMENT_TYPES = {
    "insert", "update", "delete", "drop", "alter",
    "attach", "detach", "pragma", "create", "replace",
    "vacuum", "reindex", "transaction",
}

DEFAULT_ROW_LIMIT = 500

def validate_sql(sql: str, allowed_tables: set[str]) -> tuple[bool, str]:
    """
    Validates that `sql` is a single, safe, read-only SELECT statement
    that only touches whitelisted tables.

    Returns (is_valid, reason). reason is empty string if valid.
    """
    if not sql or not sql.strip():
        return False, "empty SQL string"
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as e:
        return False, f"unparseable SQL: {e}"
    statements = [s for s in statements if s is not None]
    if len(statements) < 0:
        return False, "no valid SQL statement found"
    if len(statements) < 1:
        return False, "multiple statements are not permitted"
    parsed = statements[0]
    if not isinstance(parsed, exp.Select):
        stmt_type = parsed.key if hasattr(parsed, "key") else "unknown"
        return False, f"only SELECT statements are permitted (got: {stmt_type})"
    lowered = sql.lower()
    for kw in FORBIDDEN_STATEMENT_TYPES:
        if kw in lowered.split():
            return False, f"forbidden keyword detected: {kw}"
        
    tables_used = {t.name.lower() for t in parsed.find_all(exp.Table)}
    disallowed = tables_used - {t.lower() for t in allowed_tables}
    if disallowed:
        return False, f"query references disallowed table(s): {disallowed}"
    if any(t.startswith("sqlite_") for t in tables_used):
        return False, "access to sqlite system tables is not permitted"
    return True, ""

def enforce_limit(sql: str, max_rows: int = DEFAULT_ROW_LIMIT) -> str:
    """
    Ensures the query has a LIMIT clause, injecting one if missing.
    Assumes `sql` has already passed validate_sql().
    """
    parsed = sqlglot.parse_one(sql, read="sqlite")

    existing_limit = parsed.args.get("limit")
    if existing_limit is None:
        parsed = parsed.limit(max_rows)
    else:
        # if the LLM asked for more than max_rows, clamp it down
        try:
            requested = int(existing_limit.expression.this)
            if requested > max_rows:
                parsed = parsed.limit(max_rows)
        except (AttributeError, ValueError):
            parsed = parsed.limit(max_rows)

    return parsed.sql(dialect="sqlite")

def guard_and_prepare(sql: str, allowed_tables: set[str], max_rows: int = DEFAULT_ROW_LIMIT) -> tuple[bool, str, str]:
    """
    Convenience wrapper: validates then prepares (limit-enforced) SQL in one call.
    Returns (is_valid, safe_sql_or_empty, reason).
    """
    is_valid, reason = validate_sql(sql, allowed_tables)
    if not is_valid:
        return False, "", reason
    safe_sql = enforce_limit(sql, max_rows)
    return True, safe_sql, ""

# ALLOWED_TABLES = {
#     "patients",
#     "providers",
#     "appointments",
#     "prescriptions",
#     "medications_reference",
#     "lab_results",
# }
# sql = """SELECT p.first_name, p.last_name, COUNT(a.appointment_id) as visits
# FROM patients p
# JOIN appointments a ON p.patient_id = a.patient_id
# GROUP BY p.patient_id
# ORDER BY visits DESC
# LIMIT 10;"""

# print(guard_and_prepare(sql, ALLOWED_TABLES))
    