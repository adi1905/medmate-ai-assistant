class PermissionError_(Exception):
    pass

def enforce_own_record_or_staff(role: str, requester_id: str, target_patient_id: str):
    """Patients can only access their own records. Staff can access any."""
    if role == "staff":
        return
    elif role == "patient" and requester_id == target_patient_id:
        return
    raise PermissionError_(
        f"Access denied: patient {requester_id} cannot access records for {target_patient_id}"
    )

def require_staff(role: str):
    if role != "staff":
        raise PermissionError_("This action requires staff privileges")
    

def redact_for_patient(patient_record: dict) -> dict:
    """Field-level redaction — e.g. hide internal notes patients shouldn't see."""
    allowed = {"patient_id", "first_name", "last_name", "dob", "address"}
    return {k: v for k, v in patient_record.items() if k in allowed}


    
