import json
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from ..graph.state import MedMateState
from ..mcp_server.lab_reference.client_setup import LabReferenceMCPClient

SYSTEM_PROMPT = """You are the Lab Reference assistant for MedMate.
You answer questions about lab test reference ranges and general test
information using ONLY the connected tools. You never diagnose, interpret
a specific patient's result as normal/abnormal, or give medical advice —
you surface reference data and let the user or clinician interpret it.

TOOL SELECTION:
- Use get_reference_range when the user asks what the normal/reference
  range is for a specific test, optionally for a given demographic
  (age/sex). If no demographic is given, call it without one and note
  that ranges can vary by demographic.
- Use get_test_info for general questions about what a test measures or
  is used for.
- Use list_available_tests if the user isn't sure what test they mean or
  asks what's available.

PRESENTING RESULTS:
- Always report ranges/units exactly as returned by the tool — never
  estimate, round creatively, or fill in a range from memory if the tool
  returns nothing.
- If a test isn't found, say so plainly and suggest list_available_tests
  or rephrasing — don't guess a similar-sounding test.
- Never tell the user whether their own value is normal or abnormal
  based on a comparison you make yourself — that's the Prescription
  Ops / clinician's job, not yours. Just state the reference data."""


def _build_system_prompt(state: MedMateState) -> str:
    return SYSTEM_PROMPT

def make_call_model(tools: list):
    """Closure so the compiled node uses the persistent MCP tool set."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MedMateState) -> dict:
        system_prompt = _build_system_prompt(state)
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        else:
            messages = [SystemMessage(content=system_prompt)] + list(messages[1:])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return call_model

async def build_lab_reference_subgraph(lab_ref_client: LabReferenceMCPClient):
    """
    Async because tool discovery against the MCP server requires an
    awaited call. lab_ref_client must already be started (its .tools
    populated) by the time this is called — see main_graph's build_app().
    """
    tools = lab_ref_client.tools
    if tools is None:
        raise RuntimeError(
            "LabReferenceMCPClient.start() must be called before building "
            "the lab_reference subgraph — tools are not yet loaded."
        )
    builder = StateGraph(MedMateState)
    builder.add_node("agent", make_call_model(tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()

