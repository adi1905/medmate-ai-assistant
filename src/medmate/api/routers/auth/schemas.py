from pydantic import BaseModel, EmailStr, field_validator
from ...core.security import Role
from typing import Optional

def _validate_password_length(value: str) -> str:
    if len(value) < 10:
        raise ValueError("Password must be at least 10 characters")
    if not any(c.isdigit() for c in value):
        raise ValueError("Password must contain at least one digit")
    if not any(c.isupper() for c in value):
        raise ValueError("Password must contain at least one uppercase letter")
    return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    pass

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int 

class UserPublic(BaseModel):
    user_id: str
    name: str
    role: Role
    patient_id: Optional[str] = None

class PatientClaimRequest(BaseModel):
    last_name: str
    phone: str
    dob: str
    first_name: Optional[str] = None
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def check_password(cls, v : str)  -> str:
        return _validate_password_length(v)

class RegisterResponse(BaseModel):
    user: UserPublic
    tokens: TokenResponse

class StaffRegisterRequest(BaseModel):
    display_name: str
    email: EmailStr
    password: str
    role: str

    @field_validator("password")
    @classmethod
    def check_password(cls, v: str) -> str:
        return _validate_password_length(v)

    @field_validator("role")
    @classmethod
    def check_role(cls, v: Role) -> Role:
        if v == Role.PATIENT:
            raise ValueError("Use /auth/register or /auth/register/claim for patient accounts")
        return v

