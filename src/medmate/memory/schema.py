from datetime import datetime, timezone
from pydantic import BaseModel, Field

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ns_semantic(uid: str) -> tuple: return (uid, "semantic")
def ns_episodic(uid: str) -> tuple: return (uid, "episodic")

PATIENT_SLOTS = {
    "preferred_time_of_day": "morning / afternoon / evening for appointments",
    "preferred_provider":    "provider the user prefers to see",
    "preferred_language":    "language to use",
    "reminder_channel":      "sms / email / call",
    "answer_style":          "short vs detailed, simple wording",
    "accessibility_need":    "logistics only, e.g. large text, interpreter",
}
STAFF_SLOTS = {
    "default_analytics_period": "weekly / monthly / quarterly",
    "preferred_chart_type":     "bar / line",
    "answer_style":             "short vs detailed",
    "preferred_language":       "language to use",
}

def slots_for(role: str) -> dict:
    return STAFF_SLOTS if role == "staff" else PATIENT_SLOTS

class ExtractedPref(BaseModel):
    slot: str
    value: str = Field(max_length=120)
    confidence: float = Field(ge=0, le=1)
    explicit: bool          # user directly stated it about themselves

class ExtractionResult(BaseModel):
    preferences: list[ExtractedPref] = []