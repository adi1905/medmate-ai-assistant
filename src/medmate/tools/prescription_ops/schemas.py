from pydantic import BaseModel, Field
from typing import Optional, Literal

class GetCurrentMedicationsInput(BaseModel):
    patient_id: str
    status: Optional[Literal["active", "completed", "discontinued"]] = "active"

class FindPrescriptionInput(BaseModel):
    patient_id: str
    medication_name: Optional[str] = Field(
        None, description="Free-text medication name to match, e.g. 'lisinopril'."
    )
    status: Optional[Literal["active", "completed", "discontinued"]] = "active"

class CheckDrugInteractionFlagsInput(BaseModel):
    med_list: list[str] = Field(
        ..., description="List of medication names to check against each other and the reference table."
    )

class LogMedicationChangeInput(BaseModel):
    prescription_id: str
    dosage: Optional[str] = Field(None, description="New dosage, e.g. '10mg daily'.")
    end_date: Optional[str] = Field(None, description="New end date, ISO format.")
    status: Optional[Literal["active", "completed", "discontinued"]] = None