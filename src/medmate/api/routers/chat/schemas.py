from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from ...core.common_schemas import TraceMeta

# ---------- Threads ----------

class ThreadCreateRequest(BaseModel):
    # Nothing required from client -- role/patient scope comes from the JWT.
    # Optional title lets the frontend show a nicer sidebar entry.
    title: Optional[str] = None


class ThreadOut(BaseModel):
    thread_id: str
    title: Optional[str] = None
    created_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int = 0


class ThreadListResponse(BaseModel):
    threads: list[ThreadOut]


# ---------- Messages ----------

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageOut(BaseModel):
    message_id: str
    role: MessageRole
    content: str
    created_at: datetime


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageOut]


class ChatMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)


# ---------- Human-in-the-loop ----------

class PendingAction(BaseModel):
    """
    Mirrors a LangGraph `interrupt()` payload. Sent back to the frontend
    whenever the graph pauses on a write tool (create/update/cancel
    appointment, update_patient_contact_info, log_medication_change, etc).
    The frontend renders this as a confirm dialog -- it must NOT let the
    user free-type past it; only approve/edit/reject are valid next actions.
    """
    action_id: str
    tool_name: str
    tool_args: dict[str, Any]
    human_readable_summary: str  # e.g. "Cancel appointment #482 for Jane Doe on Aug 3, 10:00 AM"


class ChatMessageResponse(BaseModel):
    thread_id: str
    message: MessageOut
    pending_action: Optional[PendingAction] = None  # non-null => graph is interrupted, awaiting resume
    refused: bool = False                            # true when risk classifier hard-refused
    trace: Optional[TraceMeta] = None


class HITLDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT_AND_APPROVE = "edit_and_approve"


class HITLResumeRequest(BaseModel):
    action_id: str
    decision: HITLDecision
    edited_args: Optional[dict[str, Any]] = None  # required if decision == edit_and_approve


class HITLResumeResponse(BaseModel):
    thread_id: str
    message: MessageOut          # the assistant's follow-up after resuming
    executed: bool                # false if rejected
    trace: Optional[TraceMeta] = None