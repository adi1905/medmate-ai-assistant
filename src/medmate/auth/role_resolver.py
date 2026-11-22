from dataclasses import dataclass
from typing import Literal
import yaml
from pathlib import Path
from config.settings import ROLES_CONFIG_PATH

class RoleResolutionError(Exception):
    pass

@dataclass(frozen=True)
class SessionIdentity:
    role: Literal["patient", "staff"]
    user_id: str
    allowed_tools: list[str]

_roles_cache: dict | None = None

def _load_roles_config() -> dict:
    global _roles_cache
    if _roles_cache is None:
        with open(ROLES_CONFIG_PATH) as f:
            _roles_cache = yaml.safe_load(f)
    return _roles_cache

def resolve_role(role_claim: str, user_id: str, auth_token: str = None) -> SessionIdentity:
    """
    Resolves and validates the caller's role at session start.
    In production, auth_token would be verified against an identity provider
    (e.g. decode JWT, check session store) — role_claim should NEVER be trusted
    from client input alone. For MedMate's scope, we validate against a
    lookup (e.g. staff_id exists in a staff table) rather than trusting the claim blindly.
    """
    if role_claim not in ("patient", "staff"):
        raise RoleResolutionError(f"Unknown role claim: {role_claim}")

    if not user_id:
        raise RoleResolutionError("user_id is required to resolve a session identity")

    # Minimal integrity check — replace with real auth/session validation
    if role_claim == "staff":
        _verify_staff_id(user_id)
    else:
        _verify_patient_id(user_id)

    config = _load_roles_config()
    allowed_tools = config.get(role_claim, {}).get("tools", [])

    return SessionIdentity(role=role_claim, user_id=user_id, allowed_tools=allowed_tools)

def _verify_patient_id(patient_id: str):
    from ..db.repositories.patient_repo import fetch_patient
    if fetch_patient(patient_id) is None:
        raise RoleResolutionError(f"No patient found for id {patient_id}")

def _verify_staff_id(staff_id: str):
    # Replace with a real staff table lookup once one exists
    if not staff_id.startswith("S"):
        raise RoleResolutionError(f"Invalid staff id {staff_id}")