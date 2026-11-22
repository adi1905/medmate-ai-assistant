from langchain_core.tools import tool
from typing import Optional, Annotated
from langgraph.prebuilt import InjectedState
from ...auth import rbac
from datetime import datetime
import re
from ...db.repositories import patient_repo
from ...db.repositories import provider_repo
from ...db.repositories import appointment_repo
from .schemas import (GetPatientInput, UpdateAppointmentInput, CancelAppointmentInput,
                      CreateAppointmentInput, SearchPatientInput, 
                      FindProviderInput, FindAppointmentInput, UpdatePatientContactInfoInput,
                      GetAppointmentsInput)
from ...graph.state import MedMateState
from rapidfuzz import fuzz, process
from ..common.tools import _current_identity, ToolResult




def _clean_name_query(name_query: str) -> str:
    return re.sub(r"^\s*(dr\.?|doctor)\s+", "",name_query, flags=re.IGNORECASE).strip()

def _reason_matches(query: str, reason: str, threshold: int = 60) -> bool:
    return fuzz.partial_ratio(query.lower(), reason.lower()) >= threshold


def _normalize_datetime(value: str) -> str:
    """Accepts ISO 8601 (with or without 'T', with or without timezone) and
    normalizes to 'YYYY-MM-DD HH:MM:SS' to match the rest of the schema."""
    if value is None:
        return value
    cleaned = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


@tool(args_schema=GetPatientInput)
def get_patient(patient_id: str, state: Annotated[MedMateState, InjectedState],):
    """Look up a patient's record. Patients can only view their own record."""
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
        record = patient_repo.fetch_patient(patient_id)
        if not record:
            return ToolResult(success=False, error="Patient not found")
        if role == "patient":
            record = rbac.redact_for_patient(record)
        return ToolResult(success=True, data=record).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    
@tool(args_schema=SearchPatientInput)
def search_patients(state: Annotated[MedMateState, InjectedState],
    name: str = None, dob: str = None, contact_info: str = None) -> dict:
    """Search patients by filters. STAFF ONLY."""
    role, _ = _current_identity(state)
    try:
        rbac.require_staff(role)
        if not any([name, dob, contact_info]):
            return ToolResult(
                success=False,
                error="Provide at least one search filter: name, dob, or contact_info."
            ).model_dump()
        results = patient_repo.search_patients(name, dob, contact_info)
        if not results:
            return ToolResult(success=True, data={"results": [], "count": 0}).model_dump()
        return ToolResult(
            success=True,
            data={"results": results, "count": len(results)}
        ).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()


@tool(args_schema=CreateAppointmentInput)
def create_appointment(patient_id: str, provider_id: str, scheduled_time: str, reason: str, state: Annotated[MedMateState, InjectedState],) -> dict:
    """Propose creating a new appointment. Requires an exact provider_id — call find_provider first
     if the user only gave a name. REQUIRES HUMAN CONFIRMATION before executing."""
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
        provider = provider_repo.get_provider_by_id(provider_id)
        if not provider:
            return ToolResult(success=False, error=f"No provider with id {provider_id}.").model_dump()

        pending = {
            "action":"create_appointment",
            "params": {
                "patient_id": patient_id,
                "provider_id": provider_id,
                "scheduled_time": str(scheduled_time),
                "reason": reason,
            },
        }
        return ToolResult(
            success=True,
            requires_confirmation=True,
            pending_action=pending,
            data={"summary": f"New appointment with {provider['name']} ({provider['specialty']}) at {scheduled_time}"},
        ).model_dump()
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    

@tool(args_schema=FindProviderInput)
def find_provider(name_query: str) -> dict:
    """Look up provider_id(s) matching a name the user typed. Call this BEFORE create_appointment
    whenever the user gives a provider by name rather than an exact provider_id."""
    cleaned = _clean_name_query(name_query)
    matches = provider_repo.find_providers_by_name(cleaned)
    if not matches:
        return ToolResult(success=False, error=f"No provider found matching '{name_query}'.").model_dump()
    if len(matches) > 1:
        return ToolResult(
            success=True,
            data={"matches": matches, "ambiguous": True},
        ).model_dump()
    return ToolResult(success=True, data = {"matches":matches, "ambiguous":False}).model_dump()

@tool(args_schema=FindAppointmentInput)
def find_appointment(patient_id: str, 
                     state: Annotated[MedMateState, InjectedState],
                     status: Optional[str] = "scheduled",
                      provider_name: Optional[str] = None,
                      reason_keyword: Optional[str] = None,
                      on_or_after: Optional[str] = None) -> dict:
    """Look up a patient's appointments to resolve which appointment_id the user means, and
    retrieve its current details (scheduled_time, provider_id, reason) so partial updates
    (e.g. 'move to 5pm' without a date) can be merged with the existing scheduled_time.
    Call this BEFORE update_appointment or cancel_appointment whenever the user refers to
    an appointment by description — by provider name (provider_name), by what it's for
    (reason_keyword, e.g. 'thyroid', 'headache'), or by date — rather than an exact
    appointment_id. If multiple matches come back, ask the user to disambiguate before
    calling update_appointment or cancel_appointment."""
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    all_appts = appointment_repo.fetch_appointments(patient_id)
    filtered = all_appts

    if status:
        filtered = [a for a in filtered if a["status"] == status]
    if on_or_after:
        filtered = [a for a in filtered if a["scheduled_time"] >= on_or_after]
    if reason_keyword:
        filtered = [a for a in filtered if _reason_matches(reason_keyword, a["reason"])]
    provider_ids_for_name = None
    if provider_name:
        cleaned = _clean_name_query(provider_name)
        prov_matches = provider_repo.find_providers_by_name(cleaned)
        if not prov_matches:
            return ToolResult(success=False, error=f"No provider found matching '{provider_name}'.").model_dump()
        provider_ids_for_name = {m["provider_id"] for m in prov_matches}
        filtered = [a for a in filtered if a["provider_id"] in provider_ids_for_name]
    if not filtered:
        other_status = []
        if provider_ids_for_name:
            other_status = [a for a in all_appts if a["provider_id"] in provider_ids_for_name]
        elif reason_keyword:
            other_status = [a for a in all_appts if _reason_matches(reason_keyword, a["reason"])]

        if other_status:
            a = sorted(other_status, key=lambda x: x["scheduled_time"], reverse=True)[0]
            return ToolResult(
                success=False,
                error=(f"Found a matching appointment on {a['scheduled_time']}, "
                    f"but its status is '{a['status']}' — not eligible for update.")
            ).model_dump()
        return ToolResult(success=False, error="No matching scheduled appointments found.").model_dump()
    if len(filtered) > 1:
        summaries = [
            {
                "appointment_id": a["appointment_id"],
                "provider_id": a["provider_id"],
                "scheduled_time": a["scheduled_time"],
                "reason": a["reason"],
                "status": a["status"],
            }
            for a in filtered
        ]
        return ToolResult(success=True, data={"matches": summaries, "ambiguous": True}).model_dump()
    a = filtered[0]
    return ToolResult(
        success=True,
        data={"matches": [a], "ambiguous": False, "appointment": a},
    ).model_dump()


@tool(args_schema=UpdateAppointmentInput)
def update_appointment(appointment_id: str, 
                       state: Annotated[MedMateState, InjectedState],
                       scheduled_time: Optional[str] = None,
                        provider_id: Optional[str] = None,
                        reason: Optional[str] = None) -> dict:
    """Propose updating an existing appointment's time, provider, or reason.
    Requires an exact appointment_id — call find_appointment first if the user only
    described the appointment. If only changing the time-of-day, combine it with the
    existing scheduled_time's date (returned by find_appointment) to form a full ISO
    datetime — never pass a bare time. If reassigning provider, requires an exact
    provider_id — call find_provider first if the user gave a name.
    REQUIRES HUMAN CONFIRMATION before executing."""
    role, requester_id = _current_identity(state)
    existing = appointment_repo.get_appointment_by_id(appointment_id)
    if not existing:
        return ToolResult(success=False, error=f"No appointment with id {appointment_id}.").model_dump()
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, existing["patient_id"])
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    if existing["status"] in ("cancelled", "completed", "no_show"):
        return ToolResult(
            success=False,
            error=f"Cannot update an appointment with status '{existing['status']}'."
        ).model_dump()
    if provider_id:
        provider = provider_repo.get_provider_by_id(provider_id)
        if not provider:
            return ToolResult(success=False, error=f"No provider with id {provider_id}.").model_dump()
    changes = {}
    if scheduled_time and scheduled_time != existing["scheduled_time"]:
        scheduled_time = _normalize_datetime(scheduled_time)
        changes["scheduled_time"] = {"old": existing["scheduled_time"], "new": scheduled_time}
    if provider_id and provider_id != existing["provider_id"]:
        changes["provider_id"] = {"old": existing["provider_id"], "new": provider_id}
    if reason and reason != existing["reason"]:
        changes["reason"] = {"old": existing["reason"], "new": reason}
    
    if not changes:
        return ToolResult(success=False, error="No changes provided.").model_dump()

    pending_actions = {
        "action": "update_appointment",
        "params": {
            "appointment_id": appointment_id,
            "scheduled_time": scheduled_time,
            "provider_id": provider_id,
            "reason": reason,
        },
    }
    return ToolResult(
        success=True,
        requires_confirmation=True,
        pending_action=pending_actions,
        data = {"summary": "Update appointment", "changes": changes},
    ).model_dump()


@tool(args_schema=CancelAppointmentInput)
def cancel_appointment(appointment_id: str, 
                       state: Annotated[MedMateState, InjectedState],
                       reason: Optional[str] = None) -> dict:
    """Propose cancelling an existing appointment. Requires an exact appointment_id —
    call find_appointment first if the user only described the appointment.
    REQUIRES HUMAN CONFIRMATION before executing. This action cannot be undone —
    the user would need to create a new appointment if they change their mind."""
    role, requester_id = _current_identity(state)
    existing = appointment_repo.get_appointment_by_id(appointment_id)
    if not existing:
        return ToolResult(success=False, error=f"No appointment with id {appointment_id}.").model_dump()
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, existing["patient_id"])
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    if existing["status"] != "scheduled":
        return ToolResult(
            success=False,
            error=f"Cannot cancel — appointment is already '{existing['status']}'."
        ).model_dump()
    pending = {
        "action": "cancel_appointment",
        "params": {"appointment_id": appointment_id, "reason": reason},
    }
    return ToolResult(
        success=True,
        requires_confirmation=True,
        pending_action=pending,
        data={
            "summary": f"Cancel appointment on {existing['scheduled_time']} "
                       f"with provider {existing['provider_id']}",
            "changes": {"status": {"old": "scheduled", "new": "cancelled"}},
        },
    ).model_dump()

@tool(args_schema=GetAppointmentsInput)
def get_appointment_history(patient_id: str,
                            state: Annotated[MedMateState, InjectedState],
                            status: Optional[str] = None, 
                            limit: Optional[int] = 50) -> dict:
    """
    Get all the appointment history which is associated with the provided patient_id
    """
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    all_appts = appointment_repo.fetch_appointments(patient_id)
    if status:
        all_appts = [a for a in all_appts if a["status"] == status]
    if limit:
        all_appts = all_appts[:limit]
    return ToolResult(
        success=True,
        data={"appointments":all_appts, "count":len(all_appts)}
    ).model_dump()

@tool(args_schema=UpdatePatientContactInfoInput)
def update_patient_contact_info(patient_id: str, 
                                state: Annotated[MedMateState, InjectedState],
                                email: Optional[str] = None, 
                                phone: Optional[str] = None,
                                address: Optional[str] = None,
                                insurance_provider: Optional[str] = None) -> dict:
    """Propose updating a patient's contact info (phone, email, address, insurance).
    Does NOT update name, DOB, or sex — those require staff correction through a
    separate process. REQUIRES HUMAN CONFIRMATION before executing."""

    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error = str(e)).model_dump()
    existing = patient_repo.fetch_patient(patient_id)
    if not existing:
        return ToolResult(success=False, error="Patient not found").model_dump()
    fields = {
        "email":email,
        "phone": phone,
        "address": address,
        "insurance_provider": insurance_provider
    }
    changes = {
        field: {"old": existing.get(field), "new": new_val}
        for field, new_val in fields.items()
        if new_val is not None and new_val != existing.get(field)
    }
    if not changes:
        return ToolResult(success=False, error ="No changes provided.").model_dump()
    pending_actions = {
        "action":"update_patient_contact_info",
        "params":{"patient_id": patient_id, **{k: v["new"] for k, v in changes.items()}},
    }
    return ToolResult(
        success=True,
        requires_confirmation=True,
        pending_action=pending_actions,
        data={"summary": f"Update contact info for {patient_id}", "changes": changes},
    ).model_dump()
    

