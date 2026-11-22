from ..connection import get_conn


def find_user_by_email(email: str) -> dict | None:
    """
    Look up a login account by email for the /auth/login flow. Returns
    the full row (including hashed_password) as a dict, or None if no
    active account matches -- inactive accounts (is_active = 0) are
    treated the same as "not found" so a deactivated login can't be
    probed for existence.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT user_id, email, hashed_password, role, patient_id,
                   provider_id, display_name, is_active
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
        if not row or not row["is_active"]:
            return None
        return dict(row)


def get_user_by_id(user_id: str) -> dict | None:
    """Look up a login account by user_id -- used by /auth/me and anywhere
    else that already has a verified JWT sub claim and needs the rest of
    the profile (display_name, etc)."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT user_id, email, role, patient_id, provider_id,
                   display_name, is_active
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row and row["is_active"] else None


def email_exists(email: str) -> bool:
    """Cheap existence check for registration routes -- avoids pulling
    hashed_password just to test for a 409 conflict."""
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        return row is not None


def insert_user(
    user_id: str,
    email: str,
    hashed_password: str,
    role: str,
    display_name: str,
    created_at: str,
    patient_id: str | None = None,
    provider_id: str | None = None,
) -> dict:
    """Insert a new login account. Caller is responsible for having
    already checked email_exists(...) and, for role='patient', that
    patient_id isn't already claimed -- this function does the write,
    not the validation, same division of labor as create_appointment
    etc. in the tools layer."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO users
                (user_id, email, hashed_password, role, patient_id,
                 provider_id, display_name, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (user_id, email, hashed_password, role, patient_id,
             provider_id, display_name, created_at, created_at),
        )
        conn.commit()
        return {
            "user_id": user_id, "email": email, "role": role,
            "patient_id": patient_id, "provider_id": provider_id,
            "display_name": display_name,
        }

def patient_already_claimed(patient_id: str) -> bool:
    """Return True if a user account already exists for the given patient."""

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM users
            WHERE patient_id = ?
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()

        return row is not None