import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, Request, status

from ...core.security import (
    CurrentUser,
    Role,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    require_roles,
    verify_password,
    set_refresh_cookie,
    clear_refresh_cookie,
    get_refresh_cookie,
)
from .schemas import (
    LoginRequest,
    PatientClaimRequest,
    RegisterResponse,
    StaffRegisterRequest,
    TokenResponse,
    UserPublic,
)
from ....db.repositories import users_repo
from ....db.repositories import patient_repo
from ....db.utils.datetime_utils import now_ist_str
from config.settings import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["auth"])


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def _issue_tokens(response: Response, user_id: str, role: Role, patient_id: str | None) -> TokenResponse:
    access_token = create_access_token(user_id, role, patient_id)
    refresh_token = create_refresh_token(user_id, role, patient_id)
    set_refresh_cookie(response, refresh_token)
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response):
    user = users_repo.find_user_by_email(body.email)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    role = Role(user["role"])
    return _issue_tokens(response, user["user_id"], role, user["patient_id"])


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: Request, response: Response):
    refresh_token = get_refresh_cookie(request)
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token")
    payload = decode_token(refresh_token)
    if payload.type != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expected refresh token")
    return _issue_tokens(response, payload.sub, payload.role, payload.patient_id)


@router.post("/register/claim", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def claim_existing_patient(body: PatientClaimRequest, response: Response):
    patient = patient_repo.find_patient_for_claim(body.last_name, body.dob, body.phone, body.first_name)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching patient record")
    already_claimed = users_repo.patient_already_claimed(patient["patient_id"])
    if already_claimed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This patient record already has an account. Try logging in instead.",
        )
    if users_repo.email_exists(body.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user_id = _new_id("USR")
    hashed = hash_password(body.password)
    display_name = f"{patient['first_name']} {patient['last_name']}"
    users_repo.insert_user(
        user_id=user_id, email=body.email, hashed_password=hashed,
        role=Role.PATIENT.value, display_name=display_name,
        created_at=now_ist_str(), patient_id=patient["patient_id"],
    )
    tokens = _issue_tokens(response, user_id, Role.PATIENT, patient["patient_id"])
    return RegisterResponse(
        user=UserPublic(user_id=user_id, name=display_name, role=Role.PATIENT, patient_id=patient["patient_id"]),
        tokens=tokens,
    )


@router.post(
    "/register-staff",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(Role.ADMIN))],
)
def register_staff(body: StaffRegisterRequest):
    exists = users_repo.email_exists(body.email)
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user_id = _new_id("USR")
    hashed = hash_password(body.password)
    users_repo.insert_user(user_id=user_id, email=body.email, hashed_password=hashed,
                           role=Role.STAFF.value, display_name=body.display_name,
                           created_at=now_ist_str(), patient_id=None, provider_id=body.provider_id)
    return UserPublic(user_id=user_id, name=body.display_name, role=Role.STAFF, patient_id=None)


@router.get("/me", response_model=UserPublic)
def me(user: CurrentUser = Depends(get_current_user)):
    record = users_repo.get_user_by_id(user.user_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer active")
    return UserPublic(user_id=record["user_id"], name=record["display_name"],
                       role=Role(record["role"]), patient_id=record["patient_id"])


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, user: CurrentUser = Depends(get_current_user)):
    clear_refresh_cookie(response)
    return None