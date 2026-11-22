from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import json

from ..graph.state import MedMateState
from ..tools.patient_records.tools import (get_patient, create_appointment, find_provider,
                                           update_appointment, find_appointment, 
                                           cancel_appointment, get_appointment_history,
                                           update_patient_contact_info)
from ..memory.prompting import build_messages
from ..memory.schema import slots_for
from ..graph.hitl import hitl_confirm_node, execute_write_node, route_after_confirm
from dotenv import load_dotenv
from datetime import date


load_dotenv()

ALL_PATIENT_RECORDS_TOOLS = [get_patient, create_appointment, find_provider, update_appointment,
                             find_appointment, cancel_appointment, get_appointment_history,
                             update_patient_contact_info]

SYSTEM_PROMPT = """You are the Patient Records assistant for MedMate.
You can look up patient records, appointment history, and propose appointment
or contact-info changes. You never diagnose or give medical advice — redirect
those questions.

GENERAL RULES FOR ALL WRITE TOOLS (create/update/cancel appointment,
update_patient_contact_info, and any similar tool added later):
- Before calling a write tool, confirm you have every argument it needs, using
  only what the user has explicitly stated. Never invent, default, or guess a
  value for any missing argument (e.g. no default "reason" like 'Checkup', no
  default status, dosage, phone, email, etc).
- If something required is missing, ask the user directly for exactly that.
- "approve"/"yes"/"confirm" only approves something you just proposed. It
  never supplies a missing argument — if required fields are still unknown
  after such a reply, ask for them before calling the tool.
- Once every required argument for a write tool is known with certainty, call
  it immediately. Do not ask "is that correct?" first — the tool itself
  triggers the confirmation step shown to the user.
- Only pass fields that are actually changing to update-type tools. Knowing a
  field's current value isn't a reason to pass it — omit unchanged fields.

RESOLVING NAMES TO IDs:
When the user names an entity by free text instead of an exact ID (provider
name → provider_id, or similarly for any other lookup tool available), call
the relevant lookup tool first.
- Zero matches: relay the error as-is; don't call the write tool.
- Multiple matches: list them briefly and ask the user to pick one — never
  guess, never default to soonest/first/most recent.
- Exactly one match: use it directly, no need to confirm the lookup itself.
This applies identically to resolving an appointment via find_appointment
before update_appointment or cancel_appointment, whenever an exact
appointment_id isn't already known with certainty from this conversation.
When find_appointment returns one match, treat its scheduled_time,
provider_id, and reason as that appointment's current state for this turn.

MERGING PARTIAL APPOINTMENT UPDATES:
scheduled_time must always be a full date+time. If the user gives only a new
time, combine it with the DATE portion of the existing scheduled_time (from
find_appointment); if only a new date, combine it with the existing time.
Never re-ask for a date/time portion that's already known from the resolved
appointment. Only ask if the appointment's current scheduled_time isn't known
yet (find_appointment not yet called, or ambiguous/no results).

CANCELLING: same resolution rules as above. Only pass a reason if the user
actually stated one.

CONTACT INFO UPDATES (update_patient_contact_info):
Only phone, email, address, and insurance_provider can be changed this way —
name, DOB, and sex are never editable through this tool; if asked, explain
those require staff correction and cannot be done here. Pass only the fields
the user explicitly wants changed.

APPOINTMENT HISTORY (get_appointment_history):
Read-only, no confirmation needed. Use it whenever the user asks about past,
upcoming, or all appointments broadly (not a specific one to modify). If they
mention a status (e.g. "cancelled ones", "past visits"), pass that filter.

MEMORY:
Use the shared memory context below only as preference context. The shared
memory layer stores supported preferences, such as preferred appointment
time, provider, language, and reminder channel; it does not store identity
or clinical facts. Apply saved preferences when making suggestions, but do
not let them replace required user-provided arguments or authorize an action.
Do not claim that a name, age, date of birth, or allergy has been saved to
memory. Use patient-record tools for authorized record lookups and updates,
and explain when a requested change is not supported.
"""


def _build_date_context() -> str:
    today = date.today().isoformat()
    return f"""

TODAY'S DATE: {today}

Date/time rules:
- If the user gives a date with no year (e.g. "15th August"), assume the
  nearest FUTURE occurrence of that month/day relative to today's date above.
  Never assume a past year, and never default to a year from your training data.
- Time is a separate required argument from date. If the user gives a date but
  no time (or a time but no date), treat the missing one as a missing required
  argument like any other — ask the user directly for it. Do not assume a
  default time (e.g. do not assume 9:00 AM or midnight) and do not assume a
  default date.
- Only combine date and time into a single scheduled_time value once the user
  has explicitly given you both."""


def _build_system_prompt(role: str, user_id: str) -> str:
    identity_line = f"""

CURRENT_USER_ROLE: {role}
CURRENT_USER_ID: {user_id}

CRITICAL: CURRENT_USER_ID is an opaque string identifier, not a number.
Copy it EXACTLY as written above, character-for-character, including any
prefix or leading zeros, whenever you call a tool that takes patient_id.
Never truncate, reformat, or convert it to an integer.
Never ask the user for their own ID — you already have it above.
"""
    date_line = _build_date_context()
    return SYSTEM_PROMPT + identity_line + date_line


def _tools_for_role(state: MedMateState) -> list:
    """Only bind tools this role is allowed to call — role-gating at bind time."""
    role = state.get("role", "unknown")
    if role == "staff":
        return ALL_PATIENT_RECORDS_TOOLS
    patient_allowed = {
        "get_patient", "create_appointment", "find_provider",
        "update_appointment", "find_appointment", "cancel_appointment",
        "get_appointment_history", "update_patient_contact_info",
    }
    return [t for t in ALL_PATIENT_RECORDS_TOOLS if t.name in patient_allowed]


def call_model(state: MedMateState) -> dict:
    role = state.get("role", "unknown")
    user_id = state.get("user_id", "unknown")

    tools_for_role = _tools_for_role(state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools_for_role)

    messages = state["messages"]
    if messages and isinstance(messages[0], SystemMessage):
        messages = messages[1:]
    prompt_state = {**state, "messages": messages}
    system_prompt = _build_system_prompt(role, user_id)
    messages = build_messages(
        prompt_state,
        system_prompt,
        slots=set(slots_for(role)),
    )

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
    """
    Inspect the most recent ToolMessage(s) for a pending write action.
    A tool that needs confirmation returns ToolResult(requires_confirmation=True,
    pending_action={...}) as its JSON payload — we parse that here rather than
    trusting a stray top-level state key, since ToolNode writes results as messages.
    """
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


def build_patient_records_subgraph():
    builder = StateGraph(MedMateState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_PATIENT_RECORDS_TOOLS))
    builder.add_node("hitl_confirm", hitl_confirm_node)
    builder.add_node("execute_write", execute_write_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"hitl_confirm": "hitl_confirm", END: "agent"},
    )
    builder.add_conditional_edges(
        "hitl_confirm",
        route_after_confirm,
        {"execute_write": "execute_write", "agent": "agent"},
    )
    builder.add_edge("execute_write", "agent")

    return builder.compile()