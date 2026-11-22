from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

from ..graph.state import MedMateState
from ..tools.prescription_ops.tools import (get_current_medications, find_prescription,
                                            check_drug_interaction_flags, log_medication_change)
from ..tools.common.tools import remember_preference
from ..graph.hitl import hitl_confirm_node, execute_write_node, route_after_confirm


ALL_PRESCRIPTION_OPS_TOOLS = [
    get_current_medications, find_prescription, check_drug_interaction_flags,
    log_medication_change, remember_preference,
]

SYSTEM_PROMPT = """You are the Prescription Ops assistant for MedMate.
You help patients and staff view current medications and propose changes to
existing prescriptions. You never diagnose, prescribe, or give treatment
advice — redirect those questions to the patient's doctor or pharmacist.

INTERACTION FLAGS — CRITICAL:
Interaction flags come ONLY from check_drug_interaction_flags. Never assert
that two medications interact from your own knowledge, and never omit or
soften a flag the tool returned. log_medication_change already runs this
check automatically and includes results in its proposal — always mention
any returned interaction_flags to the user plainly before they confirm.

GENERAL RULES FOR WRITE TOOLS (log_medication_change):
- Before calling a write tool, confirm you have every argument it needs,
  using only what the user has explicitly stated. Never invent or default a
  dosage, end date, or status.
- If something required is missing, ask the user directly for exactly that.
- "approve"/"yes"/"confirm" only approves something you just proposed. It
  never supplies a missing argument.
- Once every required argument is known with certainty, call the tool
  immediately — do not ask "is that correct?" first, the tool itself
  triggers the confirmation step shown to the user.
- Only pass fields that are actually changing.

RESOLVING MEDICATIONS TO IDs:
When the user names a medication by free text instead of an exact
prescription_id, call find_prescription first.
- Zero matches: relay the error as-is; don't call log_medication_change.
- Multiple matches: list them briefly and ask the user to pick one.
- Exactly one match: use it directly, no need to confirm the lookup itself.

REMEMBERING PREFERENCES:
If the user states a standing preference, recurring symptom, or logistics
note relevant to future interactions (e.g. "I always fill prescriptions at
the same pharmacy", "I keep getting headaches with this medication"), call
remember_preference with:
- fact_type: "preference", "recurring_symptom", or "logistics" as appropriate
- content: a short natural-language description
Not for one-off requests."""


def _build_memory_context(state: MedMateState) -> str:
    facts = state.get("semantic_facts")
    if not facts:
        return ""
    lines = []
    for f in facts:
        if isinstance(f, dict):
            lines.append(f"- ({f.get('fact_type', 'fact')}) {f.get('content', f)}")
        else:
            lines.append(f"- {f}")
    facts_str = "\n".join(lines)
    return f"""

KNOWN PATIENT CONTEXT (from memory — treat as durable context; the user does
not need to restate these):
{facts_str}"""


def _build_system_prompt(role: str, user_id: str, state: MedMateState) -> str:
    identity_line = f"""

CURRENT_USER_ROLE: {role}
CURRENT_USER_ID: {user_id}

CRITICAL: CURRENT_USER_ID is an opaque string identifier, not a number.
Copy it EXACTLY as written above, character-for-character, whenever you call
a tool that takes patient_id. Never ask the user for their own ID."""
    memory_line = _build_memory_context(state)
    return SYSTEM_PROMPT + identity_line + memory_line


def _tools_for_role(state: MedMateState) -> list:
    """Both roles allowed; RBAC on individual records is enforced inside each tool."""
    return ALL_PRESCRIPTION_OPS_TOOLS


def call_model(state: MedMateState) -> dict:
    role = state.get("role", "unknown")
    user_id = state.get("user_id", "unknown")

    tools_for_role = _tools_for_role(state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools_for_role)

    system_prompt = _build_system_prompt(role, user_id, state)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        messages = [SystemMessage(content=system_prompt)] + list(messages[1:])

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _parse_tool_content(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
    return content if isinstance(content, dict) else None


def route_after_tools(state: MedMateState) -> str:
    last_message = state["messages"][-1]
    tool_messages = [
        m for m in state["messages"][-len(getattr(last_message, "tool_calls", [1])):]
        if getattr(m, "type", None) == "tool"
    ] or [last_message]
    for tm in tool_messages:
        content = _parse_tool_content(tm.content)
        if isinstance(content, dict) and content.get("requires_confirmation"):
            return "hitl_confirm"
    return END


def extract_durable_fact(state: MedMateState) -> dict:
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) != "tool":
            break
        content = _parse_tool_content(m.content)
        if not content:
            continue
        pending = content.get("pending_action")
        if isinstance(pending, dict) and pending.get("action") == "remember_preference":
            return {"durable_fact": pending["params"]}
    return {}


def build_prescription_ops_subgraph():
    builder = StateGraph(MedMateState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_PRESCRIPTION_OPS_TOOLS))
    builder.add_node("extract_durable_fact", extract_durable_fact)
    builder.add_node("hitl_confirm", hitl_confirm_node)
    builder.add_node("execute_write", execute_write_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "extract_durable_fact")
    builder.add_conditional_edges(
        "extract_durable_fact", route_after_tools, {"hitl_confirm": "hitl_confirm", END: "agent"}
    )
    builder.add_conditional_edges(
        "hitl_confirm",
        route_after_confirm,
        {"execute_write": "execute_write", "agent": "agent"},
    )
    builder.add_edge("execute_write", "agent")

    return builder.compile()