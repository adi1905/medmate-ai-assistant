from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from ..graph.state import MedMateState
from ..guards.pii import redact_pii

MAX_TOKENS = 3000
KEEP_TURNS = 3
STUB_OVER_CHARS = 400

def make_context_manager(llm=None):
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def context_manager(state: MedMateState):
        msgs = state["messages"]
        starts = [i for i, m in enumerate(msgs) if m.type == "human"]
        if not starts:
            return {}
        cur = starts[-1]
        updates, new_summary, cut = [], None, 0

        # 1) summarize older turns, cutting only at a human-message boundary
        if count_tokens_approximately(msgs) > MAX_TOKENS and len(starts) > KEEP_TURNS:
            cut = starts[-KEEP_TURNS]
            old = msgs[:cut]
            transcript = "\n".join(f"{m.type}: {m.content}" for m in old
                                   if m.type in ("human", "ai") and m.content)
            out = await llm.ainvoke(
                "Update the running summary of this conversation. Keep: IDs mentioned, what was "
                "asked, actions taken or declined, unresolved items. Max 120 words.\n\n"
                f"Previous summary: {state.get('conv_summary', '') or '(none)'}\n\n"
                f"New messages:\n{redact_pii(transcript)}")
            new_summary = out.content
            updates += [RemoveMessage(id=m.id) for m in old]

        # 2) stub big tool outputs from earlier turns (current turn untouched)
        for m in msgs[cut:cur]:
            if m.type == "tool" and len(str(m.content)) > STUB_OVER_CHARS:
                updates.append(ToolMessage(id=m.id, tool_call_id=m.tool_call_id, name=m.name,
                    content=f"[{m.name} output trimmed, {len(str(m.content))} chars]"))

        result = {"messages": updates} if updates else {}
        if new_summary is not None:
            result["conv_summary"] = new_summary
        return result
    return context_manager