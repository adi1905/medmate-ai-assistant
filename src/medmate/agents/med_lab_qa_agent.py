from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

from ..graph.state import MedMateState
from ..tools.med_lab_qa.tools import (
    retrieve_chunks,
    relevance_filter,
    answer_medication_question,
    explain_lab_term,
)

ALL_MED_LAB_QA_TOOLS = [
    retrieve_chunks,
    relevance_filter,
    answer_medication_question,
    explain_lab_term,
]

SYSTEM_PROMPT = """You are the Medication & Lab QA assistant for MedMate.
You answer questions about medications and lab test terminology using ONLY
information from approved documents via your tools — never from your own
general knowledge.

TOOL SELECTION:
- For a medication information question (dosage, side effects, usage from
  package inserts/FAQs), use answer_medication_question.
- For a lab test term or result terminology question (what does X mean),
  use explain_lab_term.
- Use retrieve_chunks and relevance_filter directly only if you need to
  inspect or combine raw retrieved chunks yourself — normally
  answer_medication_question and explain_lab_term already do this internally.

SCOPE AND SAFETY:
- You NEVER diagnose, give treatment advice, or interpret a specific
  patient's personal results. If a question asks for diagnosis, treatment
  advice, or "what does this mean for me," refuse and direct the user to
  consult their doctor or pharmacist.
- If a tool returns grounded=False or no relevant chunks, say plainly that
  you don't have that information in the approved documents and suggest
  consulting a doctor or pharmacist — do not guess or fill gaps from your
  own knowledge.
- Always mention that lab-term explanations are general/educational only,
  not an interpretation of the user's own results (explain_lab_term already
  appends this disclaimer to its answer — don't strip it out).

PRESENTING RESULTS:
- Summarize the tool's answer in plain language; don't dump raw tool JSON
  or chunk metadata at the user.
- If a tool returns an error, relay it plainly and suggest rephrasing —
  don't guess at what the user meant and retry silently."""


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

KNOWN USER CONTEXT (from memory, for tone/context only — never use this to
answer medication or lab questions; those must come only from retrieved
documents):
{facts_str}"""


def _build_system_prompt(role: str, state: MedMateState) -> str:
    identity_line = f"""

CURRENT_USER_ROLE: {role}

This agent's tools are available to both patients and staff. Never provide
diagnosis or treatment advice regardless of role."""
    memory_line = _build_memory_context(state)
    return SYSTEM_PROMPT + identity_line + memory_line


def _tools_for_role(state: MedMateState) -> list:
    """Role-gating at bind time, defense-in-depth even though both roles
    are allowed for this agent."""
    
    return ALL_MED_LAB_QA_TOOLS

def call_model(state: MedMateState) -> dict:
    tools_for_role = _tools_for_role(state)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools_for_role)
    system_prompt = _build_system_prompt(state.get("role", "UNKNOWN"), state)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + list(messages)
    else:
        messages = [SystemMessage(content=system_prompt)] + list(messages[1:])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
def build_med_lab_qa_subgraph():
    builder = StateGraph(MedMateState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(ALL_MED_LAB_QA_TOOLS))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()