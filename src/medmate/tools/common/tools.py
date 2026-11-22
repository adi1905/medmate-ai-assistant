from pydantic import BaseModel, Field
from typing import Optional, Literal
from ...graph.state import MedMateState
from typing import Optional, Annotated
from langgraph.prebuilt import InjectedState
from langchain.tools import tool

class ToolResult(BaseModel):
    """Every tool returns this envelope. Non-negotiable."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    requires_confirmation: bool = False
    pending_action: Optional[dict] = None


class RememberPreferenceInput(BaseModel):
    fact_type: Literal["preference", "recurring_symptom", "logistics", "identity", "critical"] = Field(
        ..., description=(
            "Category of durable fact. Use 'identity' for name/nickname/age/DOB-type "
            "facts about who the patient is. Use 'critical' for safety-relevant facts "
            "like allergies. Use 'preference'/'logistics' for scheduling-style habits."
        )
    )
    content: str = Field(..., description="Short natural-language fact, e.g. 'nickname is Aditya', 'age is 27'.")
    sub_key: Optional[str] = Field(
        default=None,
        description=(
            "Required for identity/critical facts that should be single-valued per user, "
            "e.g. 'nickname', 'age', 'dob', 'allergy:penicillin'. Lets future updates "
            "overwrite instead of duplicating. Omit for preference/logistics/recurring_symptom."
        ),
    )


def _current_identity(state: MedMateState) -> tuple[str, str]:
    """Pull role/user_id injected at session start — never trust LLM-provided role."""
    return state.get("role", "unknown"), state.get("user_id","unknown")

@tool(args_schema=RememberPreferenceInput)
def remember_preference(fact_type: str, content: str, state: Annotated[MedMateState, InjectedState], sub_key: Optional[str] = None) -> dict:
    """Call this when the user states a standing preference, identity fact (name/
    nickname/age), critical fact (allergy), recurring symptom, or logistics note.
    For identity/critical facts, always pass sub_key (e.g. 'nickname', 'age',
    'allergy:penicillin') so updates overwrite instead of duplicating.
    Does not require confirmation; this only stores a note."""
    _, user_id = _current_identity(state)
    fact_payload = {
        "user_id": user_id,
        "fact_type": fact_type,
        "content": content,
        "confidence": "stated",
        "sub_key": sub_key,
    }
    return ToolResult(
        success=True,
        data={"summary": f"Noted {fact_type}: {content}"},
        pending_action={"action": "remember_preference", "params": fact_payload},
    ).model_dump()