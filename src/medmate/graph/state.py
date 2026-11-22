from typing import Optional, Literal, Annotated
from langgraph.graph import MessagesState

CLEAR_EVENTS = [{"_clear": True}]

def merge_memory_events(old, new):
    """Reducer: append events; [{'_clear': True}] empties the queue."""
    new = new or []
    if new and new[0].get("_clear"):
        return []
    return (old or []) + new

class MedMateState(MessagesState):
    # identity: set by your auth layer, never by the LLM
    role: Literal["patient", "staff"]
    user_id: str

    # HITL (per-turn)
    pending_action: Optional[dict]
    confirmed_action: Optional[dict]

    # routing (per-turn)
    plan_agents: Optional[list[str]]
    plan_index: int
    plan_reasoning: Optional[str]
    refused: bool

    # short-term (thread-scoped)
    conv_summary: str

    # long-term, read side
    memory_loaded: bool                 # profile loaded once per thread
    memory_profile: dict[str, str]
    memory_recent: list[str]
    memory_relevant: list[str]          # per-turn, reset by start_turn

    # long-term, write queue
    memory_events: Annotated[list[dict], merge_memory_events]