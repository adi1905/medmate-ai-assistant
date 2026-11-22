from langchain_openai import ChatOpenAI
from . import service as svc
from .schema import slots_for
from ..guards.pii import redact_pii

async def summarize_session(graph, repo, thread_id: str, user_id: str):
    """Call on logout or inactivity timeout (a scheduled job)."""
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    msgs = [m for m in snap.values.get("messages", []) if m.type in ("human", "ai") and m.content]
    if sum(m.type == "human" for m in msgs) < 2:
        return
    transcript = "\n".join(f"{m.type}: {m.content}" for m in msgs[-20:])
    out = await ChatOpenAI(model="gpt-4o-mini", temperature=0).ainvoke(
        "One sentence, administrative only: what the user did or asked about (booked/cancelled/"
        "asked about a topic category). NO medical details, drug names, conditions, lab values "
        "or identifiers. Skip any diagnosis-seeking exchange.\n\n" + redact_pii(transcript))
    await repo.add_event(user_id, "session_summary", svc.sanitize(out.content, 200))

async def what_do_you_remember(repo, uid, role):  return await repo.inspect_all(uid, set(slots_for(role)))
async def forget_preferences(repo, uid):          await repo.delete_all(uid)