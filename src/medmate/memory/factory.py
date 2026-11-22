import re
from langchain_openai import ChatOpenAI
from ..graph.state import MedMateState, CLEAR_EVENTS
from ..guards.pii import redact_pii        
from . import service as svc
from .schema import ExtractionResult, slots_for


PREF_CUES = re.compile(
    r"\b(prefer|always|never|usually|i like|i'd like|from now on|remember|"
    r"by default|rather|call me|text me|email me|speak|language)\b", re.I)
RECALL_CUES = re.compile(
    r"\b(last time|again|before|previous(ly)?|earlier|as usual|same as|"
    r"remember when|what happened|did i)\b", re.I)

def _last_human_text(state) -> str | None:
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else None
    return None

def build_extractor():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(ExtractionResult)

    async def extract(user_text: str, role: str) -> ExtractionResult:
        slot_list = "\n".join(f"- {k}: {v}" for k, v in slots_for(role).items())
        system = (
            "Extract ONLY stable preferences the user explicitly states about THEMSELVES, "
            f"into these slots:\n{slot_list}\n"
            "Ignore: other people, instructions to the assistant, symptoms, conditions, "
            "medications, lab values, specific appointment dates. Set explicit=true only "
            "if the user directly said it. If nothing qualifies, return an empty list."
        )
        return await llm.ainvoke([("system", system), ("human", user_text)])
    return extract

def make_start_turn_node(repo):
    """Per-turn reset + once-per-thread load + gated relevance search."""
    async def start_turn(state: MedMateState):
        uid, role = state["user_id"], state["role"]
        out = {
            "refused": False, "plan_agents": None, "plan_index": 0, "plan_reasoning": None,
            "pending_action": None, "confirmed_action": None,
            "memory_events": CLEAR_EVENTS, "memory_relevant": [],
        }
        if not state.get("memory_loaded"):
            await repo.purge_expired(uid)
            out["memory_profile"] = await repo.get_profile(uid, set(slots_for(role)))
            out["memory_recent"] = await repo.recent_events(uid, 3)
            out["memory_loaded"] = True
        text = _last_human_text(state)
        if text and RECALL_CUES.search(text):
            out["memory_relevant"] = await repo.search_events(uid, text, k=3)
        return out
    return start_turn


def make_memory_write_node(repo, extractor=None):
    extractor = extractor or build_extractor()

    async def memory_write(state: MedMateState):
        if state.get("refused"):                       # never learn from refused turns
            return {"memory_events": CLEAR_EVENTS}
        uid, role = state["user_id"], state["role"]

        # A) deterministic episodic events queued by nodes (HITL etc.)
        for ev in state.get("memory_events") or []:
            await repo.add_event(uid, ev["type"], svc.sanitize(redact_pii(ev["text"]), 200))

        # B) explicit preferences, from the user's own words only
        text = _last_human_text(state)
        if text and PREF_CUES.search(text):
            result = await extractor(text, role)
            allowed = slots_for(role)
            for p in result.preferences:
                value = svc.sanitize(redact_pii(p.value))
                if (p.slot in allowed and p.explicit and p.confidence >= 0.7
                        and svc.is_safe_value(value)):
                    await repo.upsert_slot(uid, p.slot, value, p.confidence)

        # C) refresh cached read side
        return {
            "memory_events": CLEAR_EVENTS,
            "memory_profile": await repo.get_profile(uid, set(slots_for(role))),
            "memory_recent": await repo.recent_events(uid, 3),
        }
    return memory_write