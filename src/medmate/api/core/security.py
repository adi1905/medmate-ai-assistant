from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
from config.settings import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
bearer_scheme = HTTPBearer()

REFRESH_COOKIE_NAME = "medmate_refresh_token"
REFRESH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days



class Role(str, Enum):
    PATIENT = "patient"
    STAFF = "staff"
    ADMIN = "admin"  

class TokenPayload(BaseModel):
    sub: str
    role: str
    patient_id: Optional[str] = None
    exp: int
    type: str

class CurrentUser(BaseModel):
    user_id: str
    role: str
    patient_id: Optional[str] = None

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def _create_token(sub: str, role: Role, patient_id: Optional[str], expires_delta: timedelta, token_type: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "role": role,
        "patient_id": patient_id,
        "exp": now + expires_delta,
        "iat": now,
        "type": token_type,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
 
 
def create_access_token(sub: str, role: Role, patient_id: Optional[str] = None) -> str:
    return _create_token(
        sub, role, patient_id,
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "access",
    )
 
 
def create_refresh_token(sub: str, role: Role, patient_id: Optional[str] = None) -> str:
    return _create_token(
        sub, role, patient_id,
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "refresh",
    )
 
 
def decode_token(token: str) -> TokenPayload:
    try:
        raw = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**raw)
    except (JWTError, ValidationError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=True,       # requires HTTPS in prod; use secure=False only for plain-http localhost testing
        samesite="strict",
        max_age=REFRESH_COOKIE_MAX_AGE,
        path="/auth",
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/auth")

def get_refresh_cookie(request: Request) -> str | None:
    return request.cookies.get(REFRESH_COOKIE_NAME)
 
def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> CurrentUser:
    payload = decode_token(creds.credentials)
    if payload.type != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Expected access token")
    return CurrentUser(user_id=payload.sub, role=payload.role, patient_id=payload.patient_id)
 
 
def require_roles(*allowed: Role):
    """Dependency factory: require_roles(Role.STAFF, Role.ADMIN)"""
    def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role for this route")
        return user
    return _checker
 
 
def resolve_patient_scope(requested_patient_id: str, user: CurrentUser) -> str:
    """
    Enforces: patients can only ever act on their own patient_id.
    Staff/admin may pass any patient_id. Use this in every records/chat
    route that touches a specific patient's data.
    """
    if user.role == Role.PATIENT:
        if user.patient_id != requested_patient_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access another patient's data")
    return requested_patient_id