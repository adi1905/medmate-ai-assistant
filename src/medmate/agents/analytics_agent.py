from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
load_dotenv()

from ..graph.state import MedMateState
from ..tools.analytics.tools import (no_show_rate_by_period, appointment_volume_trend, 
                                   top_medications,flagged_lab_summary, 
                                   provider_utilization, 
                                   answer_custom_query, generate_chart)
from ..tools.common.tools import remember_preference
from datetime import date


ALL_ANALYTICS_TOOLS = [
    no_show_rate_by_period, appointment_volume_trend, top_medications,
    flagged_lab_summary, provider_utilization, answer_custom_query,
    generate_chart, remember_preference,
]

SYSTEM_PROMPT = """You are the Analytics assistant for MedMate. STAFF ONLY.
You answer clinic operations questions using the tools available — no_show
rates, appointment volume trends, top medications, flagged lab summaries,
and provider utilization. You never diagnose or give medical advice, and you
never discuss a specific patient's clinical details — this agent is for
aggregate/operational metrics only.

TOOL SELECTION:
- Always prefer a specific metric tool (no_show_rate_by_period,
  appointment_volume_trend, top_medications, flagged_lab_summary,
  provider_utilization) when the question matches what it does.
- Use answer_custom_query ONLY when none of the specific tools fit the
  question. Don't default to it just because it's more flexible.

DATE RANGES:
- If the user gives a relative range ("last month", "this quarter", "YTD"),
  resolve it to explicit start/end ISO dates using today's date below before
  calling any tool. Never pass a relative phrase directly as start/end.
- If a tool requires a date range and the user gave none, ask for one — don't
  assume a default window.

CHARTS:
- Call generate_chart only when the user's request signals they want a
  visual — words like "chart", "graph", "plot", "visualize", "show me a
  breakdown", or similar. A plain factual question ("what's the no-show
  rate") should get a text answer only — do not call generate_chart
  unprompted.
- Always call the relevant metric tool (or answer_custom_query) FIRST to get
  the data, then pass that data into generate_chart. Never call
  generate_chart on data you haven't actually retrieved.
- Pick chart_type based on the data shape: 'line' for trends over time
  (volume trend), 'bar' for rankings/comparisons (top medications, provider
  breakdowns), 'pie' only for status/category proportions of a single whole
  (e.g. appointment status distribution). Default to 'bar' if unsure.
- Set x_key/y_key to match the actual keys in the tool result you received —
  don't guess generic names.
- After generating a chart, still give a short plain-language summary of
  what it shows; don't let the chart replace the explanation entirely.

PRESENTING RESULTS:
- Summarize the returned data in plain language; don't just dump raw numbers
  or tool JSON at the user.
- If answer_custom_query returns an error, relay it as-is and suggest
  rephrasing — don't guess at what the user meant and retry silently.

REMEMBERING PREFERENCES:
If the staff user states a standing preference relevant to future analytics
interactions (e.g. "always show me bar charts", "I usually want the
quarterly view by default"), call remember_preference with:
- fact_type: "preference" or "logistics" as appropriate
- content: a short natural-language description
This is scoped to the staff user's own working preferences only — never
record anything about a specific patient here. Not for one-off requests."""


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

KNOWN STAFF PREFERENCES (from memory — treat as durable context; the user
does not need to restate these):
{facts_str}"""


def _build_system_prompt(role: str, state: MedMateState) -> str:
    today = date.today().isoformat()

    identity_line = f"""

CURRENT_USER_ROLE: {role}
TODAY'S DATE: {today}

Resolve any relative date range against today's date above before calling a
tool. Never assume a default range if the user hasn't specified one."""
    memory_line = _build_memory_context(state)
    return SYSTEM_PROMPT + identity_line + memory_line


def _tools_for_role(state: MedMateState) -> list:
    """Only bind tools this role is allowed to call — role-gating at bind time.
    In practice the router shouldn't even reach this subgraph for non-staff,
    but we gate here too as defense-in-depth."""
    role = state.get("role", "unknown")
    return ALL_ANALYTICS_TOOLS if role == "staff" else []

def call_model(state: MedMateState) -> dict:
    tools_for_role = _tools_for_role(state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools_for_role)
    system_prompt = _build_system_prompt(state.get("role", "unknown"), state)
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


def build_analytics_subgraph():
    builder = StateGraph(MedMateState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_ANALYTICS_TOOLS))
    builder.add_node("extract_durable_fact", extract_durable_fact)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools":"tools", END: END})
    builder.add_edge("tools", "extract_durable_fact")
    builder.add_edge("extract_durable_fact", "agent")
    return builder.compile()