from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None

class PaginationParams(BaseModel):
    limit: int = 20
    offset: int = 0

class TraceMeta(BaseModel):
    """Attached to every agent-originated response so the frontend can
    deep-link into LangSmith / your observability store."""
    trace_id: Optional[str] = None
    latency_ms: Optional[float] = None
    tool_calls: Optional[int] = None
    node_path: Optional[list[str]] = None

class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp : datetime