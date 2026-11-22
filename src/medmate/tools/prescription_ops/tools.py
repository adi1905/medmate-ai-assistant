from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Optional, Annotated
from ...auth import rbac
from ...db.repositories import prescription_repo, medication_reference_repo
from ..common.tools import ToolResult, _current_identity
from .schemas import (GetCurrentMedicationsInput, FindPrescriptionInput, 
                      CheckDrugInteractionFlagsInput, LogMedicationChangeInput)
from ...graph.state import MedMateState
from rapidfuzz import fuzz

def _name_matches(query: str, name: str, threshold: int = 70) -> bool:
    return fuzz.partial_ratio(query.lower(), name.lower()) >= threshold

@tool(args_schema=GetCurrentMedicationsInput)
def get_current_medications(patient_id: str, state: Annotated[MedMateState, InjectedState],
                            status: Optional[str] = "active") -> dict:
    """Look up a patient's prescriptions. Patients can only view their own."""
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error = str(e))
    meds = prescription_repo.fetch_prescriptions(patient_id, status=status)
    return ToolResult(success=True, data={"medications": meds, "count": len(meds)})

@tool(args_schema=FindPrescriptionInput)
def find_prescription(patient_id: str, state: Annotated[MedMateState, InjectedState],
                      medication_name: Optional[str] = None,
                      status: Optional[str] = "active") -> dict:
    """Look up a patient's prescriptions to resolve which prescription_id the user means.
    Call this BEFORE log_medication_change whenever the user refers to a medication by
    name rather than an exact prescription_id. If multiple matches come back, ask the
    user to disambiguate before calling log_medication_change."""
    role, requester_id = _current_identity(state)
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, patient_id)
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()
    all_meds = prescription_repo.fetch_prescriptions(patient_id)
    filtered = all_meds
    if status:
        filtered = [m for m in filtered if m["status"] == status]
    if medication_name:
        filtered = [m for m in filtered if _name_matches(medication_name, m["medication_name"])]
    if not filtered:
        return ToolResult(success=False, error="No matching prescriptions found.").model_dump()
    if len(filtered) > 1:
        return ToolResult(
            success=True,
            data={"matches": filtered, "ambiguous": True},
        ).model_dump()
    m = filtered[0]
    return ToolResult(
        success=True,
        data={"matches": [m], "ambiguous": False, "prescription": m},
    ).model_dump()

@tool(args_schema=CheckDrugInteractionFlagsInput)
def check_drug_interaction_flags(med_list: list[str]) -> dict:
    """Pure rule-based lookup against a curated interaction reference table.
    Returns known interaction flags for the given medication list. This tool is the
    ONLY source of truth for interactions — never assert an interaction the tool
    didn't return, and never omit one it did."""
    flags = []
    unknown = []
    for med in med_list:
        ref = medication_reference_repo.get_reference(med)
        if not ref:
            unknown.append(med)
            continue
        known_interactions = ref.get("known_interactions") or []
        for other in med_list:
            if other.lower() == med.lower():
                continue
            if any(other.lower() == interacting.lower() for interacting in known_interactions):
                flags.append({"medication": med, "interacts_with": other})
    return ToolResult(
        success=True,
        data={
            "flags": flags,
            "flagged": len(flags) > 0,
            "unknown_medications": unknown,
        },
    ).model_dump()

@tool(args_schema=LogMedicationChangeInput)
def log_medication_change(prescription_id: str, state: Annotated[MedMateState, InjectedState],
                          dosage: Optional[str] = None,
                          end_date: Optional[str] = None,
                          status: Optional[str] = None) -> dict:
    """Propose a change to an existing prescription's dosage, end date, or status.
    Requires an exact prescription_id — call find_prescription first if the user only
    named the medication. Automatically checks interaction flags against the new state
    before proposing. REQUIRES HUMAN CONFIRMATION before executing."""
    role, requester_id = _current_identity(state)
    existing = prescription_repo.get_prescription_by_id(prescription_id)
    if not existing:
        return ToolResult(success=False, error=f"No prescription with id {prescription_id}.").model_dump()
    try:
        rbac.enforce_own_record_or_staff(role, requester_id, existing["patient_id"])
    except rbac.PermissionError_ as e:
        return ToolResult(success=False, error=str(e)).model_dump()

    if existing["status"] in ("completed", "discontinued") and status != "active":
        return ToolResult(
            success=False,
            error=f"Cannot modify a prescription with status '{existing['status']}'."
        ).model_dump()
    changes = {}
    if dosage and dosage != existing["dosage"]:
        changes["dosage"] = {"old": existing["dosage"], "new": dosage}
    if end_date and end_date != existing["end_date"]:
        changes["end_date"] = {"old": existing["end_date"], "new": end_date}
    if status and status != existing["status"]:
        changes["status"] = {"old": existing["status"], "new": status}
    if not changes:
        return ToolResult(success=False, error="No changes provided.").model_dump()
    active_meds = prescription_repo.fetch_prescriptions(existing["patient_id"], status="active")
    med_names = [m["medication_name"] for m in active_meds if m["prescription_id"] != prescription_id]
    med_names.append(existing["medication_name"])
    interaction_result = check_drug_interaction_flags.invoke({"med_list": med_names})
    interaction_flags = interaction_result.get("data", {}).get("flags", [])
    pending = {
        "action": "log_medication_change",
        "params": {
            "prescription_id": prescription_id,
            "dosage": dosage,
            "end_date": end_date,
            "status": status,
        },
    }
    summary = f"Update prescription for {existing['medication_name']}"
    if interaction_flags:
        summary += f" — WARNING: {len(interaction_flags)} known interaction(s) flagged"
    return ToolResult(
        success=True,
        requires_confirmation=True,
        pending_action=pending,
        data={
            "summary": summary,
            "changes": changes,
            "interaction_flags": interaction_flags,
        },
    ).model_dump()
