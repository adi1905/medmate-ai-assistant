from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class GetPatientInput(BaseModel):
    patient_id: str = Field(..., description="Opaque patient identifier string, e.g. 'PT-0006'. "
                    "Copy verbatim from context — never convert to a number.")

class SearchPatientInput(BaseModel):
    name: Optional[str] = None
    dob: Optional[str] = None
    contact_info: Optional[str] = None

class CreateAppointmentInput(BaseModel):
    patient_id: str
    provider_id: str = Field(..., description="Exact provider_id, e.g. 'PR-001'. Never a name — "
                                               "call find_provider first to resolve a name to an ID.")
    scheduled_time: datetime
    reason: str = Field(..., description="Reason for the visit")


class FindProviderInput(BaseModel):
    name_query: str = Field(..., description="Provider name as the user said it, e.g. 'Dr. Sarah Lee' or 'Sarah Lee'")

class FindAppointmentInput(BaseModel):
    patient_id: str
    status: Optional[str] = "scheduled"
    provider_name: Optional[str]  = None
    reason_keyword: Optional[str] = None
    on_or_after: Optional[str] = None

class UpdateAppointmentInput(BaseModel):
    appointment_id: str
    scheduled_time: Optional[str] = None  # full ISO datetime, not just a time-of-day
    provider_id: Optional[str] = None
    reason: Optional[str] = None

class CancelAppointmentInput(BaseModel):
    appointment_id: str
    reason: Optional[str] = None

class GetAppointmentsInput(BaseModel):
    patient_id: str 
    status: Optional[str] = Field(None, description="Filter by status: scheduled/completed/cancelled/no_show. Omit for all.")
    limit: Optional[str] = Field(None, description="Max records to return, most recent first")

class UpdatePatientContactInfoInput(BaseModel):
    patient_id: str
    phone: Optional[str] = Field(None, description="New phone number, if changing")
    email: Optional[str] = Field(None, description="New email address, if changing")
    address: Optional[str] = Field(None, description="New mailing address, if changing")
    insurance_provider: Optional[str] = Field(None, description="New insurance provider, if changing")

