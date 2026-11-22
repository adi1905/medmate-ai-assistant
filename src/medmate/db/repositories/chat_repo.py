from datetime import datetime
from typing import Optional
from ..utils.datetime_utils import now_ist_str
from ..connection import get_conn


def create_chat_message(message_id: str, thread_id: str, role: str, content: str, created_at: str) -> None:
    """
    Insert a chat message. Returns nothing -- every column here is
    supplied by the caller (no autoincrement/default/trigger-generated
    fields), so a follow-up SELECT to read the row back would just be
    fetching values the caller already has. routers/chat.py already
    builds its own MessageOut from these same inputs.
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (message_id, thread_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, thread_id, role, content, created_at),
        )


def create_chat_thread(thread_id: str, user_id: str, title: str, created_at: str) -> None:
    """Same reasoning as create_chat_message -- no generated columns, no
    need to read the row back after inserting it."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO chat_threads (thread_id, user_id, title, created_at, last_message_at)
               VALUES (?, ?, ?, ?, ?)""",
            (thread_id, user_id, title, created_at, created_at),
        )


def update_chat_thread_last_message(thread_id: str, last_message_at: str) -> None:
    """
    Touches last_message_at. Still raises if the thread doesn't exist
    (that check has real value -- it catches a bad thread_id), but no
    longer re-SELECTs the row afterward; the caller (_touch_thread in
    chat.py) discarded that return value anyway.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE chat_threads SET last_message_at = ? WHERE thread_id = ?",
            (last_message_at, thread_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No chat thread found with id {thread_id}")


def create_request_log(
    log_id: str,
    thread_id: str,
    user_id: str,
    role: str,
    event_type: str,
    summary: str,
    created_at: str, 
    trace_id: Optional[str] = None,
    tool_call_count: int = 0,
    latency_ms: Optional[float] = None,
    hitl_outcome: Optional[str] = None,
) -> None:
    """
    Write-only, observability log -- nothing reads this back in the same
    request, so no SELECT here either.

    NOTE: check this against your actual request_logs schema -- if the
    column was created as `redacted_summary` (per the PII-redaction
    design), this INSERT's `summary` column name won't match and will
    raise `no such column`. Rename one side to match the other before
    this runs for real.
    """
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO request_logs
               (log_id, thread_id, user_id, role, event_type, redacted_summary,
                trace_id, tool_call_count, latency_ms, hitl_outcome, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log_id, thread_id, user_id, role, event_type, summary,
             trace_id, tool_call_count, latency_ms, hitl_outcome, created_at),
        )


def fetch_chat_thread_owner(thread_id: str) -> dict | None:
    """Fetch the owner of a chat thread."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return dict(row) if row else None


def list_chat_threads(user_id: str) -> list[dict]:
    """List all chat threads for a user, most recent first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT thread_id, title, created_at, last_message_at
               FROM chat_threads WHERE user_id = ? ORDER BY last_message_at DESC""",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def fetch_chat_messages(thread_id: str) -> list[dict]:
    """Fetch all messages for a chat thread in chronological order."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT message_id, thread_id, role, content, created_at
               FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC""",
            (thread_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def create_pending_action(
    action_id: str, thread_id: str, tool_name: str, tool_args_json: str,
    summary: str, created_at: str,
) -> None:
    """
    Records a HITL confirmation as durably 'pending'. Expires any prior
    pending row for this thread first -- only one confirmation can be
    outstanding per thread at a time (matches the old dict's overwrite
    semantics), but now it's explicit and auditable instead of just
    silently clobbered in memory. Both statements run in one connection
    context, so this is atomic -- no window where a thread briefly has
    two 'pending' rows.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE pending_actions SET status = 'expired' WHERE thread_id = ? AND status = 'pending'",
            (thread_id,),
        )
        conn.execute(
            """INSERT INTO pending_actions
               (action_id, thread_id, tool_name, tool_args, summary, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
            (action_id, thread_id, tool_name, tool_args_json, summary, created_at),
        )
 
 
def get_pending_action(action_id: str) -> dict | None:
    """Look up a pending action by its id -- used to validate a /resume
    request before touching the graph at all."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
        ).fetchone()
        return dict(row) if row else None
 
 
def resolve_pending_action(action_id: str, status: str, resolved_at: str) -> bool:
    """
    Atomically flips a pending action to approved/rejected -- the WHERE
    status = 'pending' clause is a compare-and-swap: if two /resume
    requests for the same action_id race, only the first UPDATE affects
    a row; the second gets rowcount 0 and should treat that as "already
    resolved" rather than actually re-running the graph resume. Returns
    True if this call was the one that resolved it, False if it was
    already resolved/expired by the time this ran.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE pending_actions SET status = ?, resolved_at = ? WHERE action_id = ? AND status = 'pending'",
            (status, resolved_at, action_id),
        )
        return cur.rowcount > 0