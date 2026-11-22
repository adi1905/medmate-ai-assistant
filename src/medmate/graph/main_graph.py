from langgraph.graph import START, END, StateGraph
from .state import MedMateState

from ..router.router import route_query,next_agent_condition, advance_plan
from ..router.risk_classifier import classify_intent_risk, refusal_condition
from ..memory.factory import make_start_turn_node, make_memory_write_node
from ..memory.context import make_context_manager

from ..agents.analytics_agent import build_analytics_subgraph
from ..agents.patient_records_agent import build_patient_records_subgraph
from ..agents.med_lab_qa_agent import build_med_lab_qa_subgraph
from ..agents.prescription_ops_agent import build_prescription_ops_subgraph
from ..agents.lab_reference_agent import build_lab_reference_subgraph

from ..mcp_server.lab_reference.client_setup import LabReferenceMCPClient
from ..agents.generic_agent import general_node



AGENTS = ("patient_records", "med_lab_qa", "prescription_ops", "analytics", "lab_reference", "general")
ROUTES = {**{a: a for a in AGENTS}, "end": "memory_write"}

async def build_main_graph(checkpointer, repo):

    lab_ref_client = LabReferenceMCPClient()
    await lab_ref_client.start()

    b = StateGraph(MedMateState)
    b.add_node("start_turn", make_start_turn_node(repo))
    b.add_node("context_manager", make_context_manager())
    b.add_node("risk_classifier", classify_intent_risk)
    b.add_node("router", route_query)
    b.add_node("patient_records", build_patient_records_subgraph())
    b.add_node("med_lab_qa", build_med_lab_qa_subgraph())
    b.add_node("prescription_ops", build_prescription_ops_subgraph())
    b.add_node("analytics", build_analytics_subgraph())
    b.add_node("lab_reference", await build_lab_reference_subgraph(lab_ref_client))
    b.add_node("general", general_node)
    b.add_node("advance", advance_plan)
    b.add_node("memory_write", make_memory_write_node(repo))

    b.add_edge(START, "start_turn")
    b.add_edge("start_turn", "context_manager")
    b.add_edge("context_manager", "risk_classifier")
    b.add_conditional_edges("risk_classifier", refusal_condition, {"router": "router", "end": END})
    b.add_conditional_edges("router", next_agent_condition, ROUTES)
    for a in AGENTS:
        b.add_edge(a, "advance")
    b.add_conditional_edges("advance", next_agent_condition, ROUTES)
    b.add_edge("memory_write", END)

    

    compiled = b.compile(checkpointer=checkpointer, store=repo.store)
    compiled._lab_ref_client = lab_ref_client
    return compiled


async def shutdown_main_graph(compiled_graph):
    client = getattr(compiled_graph, "_lab_ref_client", None)
    if client:
        await client.stop()