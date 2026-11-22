"""
scripts/test_main_graph_manual.py

Interactive console test for the FULL main graph: routing across
sub-agents, memory load/write, AND human-in-the-loop confirmation for
write tools. Run this end-to-end BEFORE trusting the FastAPI /chat
routes -- the API's interrupt detection is built on the exact same
underlying mechanism this script exercises directly. If a write-
triggering message doesn't pause here for confirmation, it will not
behave correctly through the API either.

CHANGED vs the InMemoryStore/MemorySaver version:

  1. Checkpointer is now AsyncSqliteSaver (SQLITE_CHECKPOINTER_PATH),
     matching main.py, instead of MemorySaver. This is opened/closed
     as an async context manager -- mirrors the lifespan() pattern in
     main.py so this script exercises the exact same code path that's
     502-ing in the API.

  2. Store is now the real HybridMemoryStore (Qdrant + sqlite episodic
     events via SQLITE_EPISODIC_PATH / MEMORY_QDRANT_PATH), not
     InMemoryStore. This means it hits the real embed_text() call
     (OpenAI API) and real Qdrant collection -- if the 502 is coming
     from an embeddings/Qdrant issue rather than the graph logic
     itself, this script will surface it directly instead of masking
     it behind an in-memory store that never touches either.

  3. dump_memory() no longer uses store.search((user_id, "semantic"))
     -- that's the langgraph BaseStore API, which HybridMemoryStore
     does NOT implement (see hybrid_store.py docstring: "purpose-built
     interface instead of generic get/put/search"). Rewritten to call
     get_always_loaded_facts() + search_semantic_facts() directly.

  4. Because setup/teardown of the checkpointer and store now involves
     real async context managers / async close(), main() wraps
     everything in a try/finally that mirrors main.py's lifespan
     shutdown ordering: graph -> checkpointer -> store -> qdrant.

Same two fixes vs the standalone subgraph script (test_patient_records_manual.py)
still apply here:

  1. role/user_id go in `inputs` (the graph STATE), not
     config.configurable. call_model() in patient_records_agent.py reads
     state.get("role")/state.get("user_id") -- putting them only in
     config (as the subgraph-only script does) means they silently
     default to "unknown", which _tools_for_role() treats as
     patient-level access no matter what role you intended to test.

  2. `allowed_tools` from the subgraph script's config is dropped here --
     it isn't read anywhere in call_model()/_tools_for_role() as shown,
     so keeping it would just be dead config. Add it back (and actually
     read it in the graph) if you want tool restriction finer-grained
     than the role split.
"""
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from qdrant_client import QdrantClient

from config.settings import SQLITE_CHECKPOINTER_PATH, SQLITE_EPISODIC_PATH, MEMORY_QDRANT_PATH
from ..memory.hybrid_store import HybridMemoryStore
from ..graph.main_graph import build_main_graph, shutdown_main_graph

graph = None  # built inside main() -- build_main_graph is async, can't run at import time
store = None  # built inside main() -- needs qdrant + setup(), also async


async def dump_memory(user_id):
    print("\n--- STORE DUMP ---")
    always_loaded = await store.get_always_loaded_facts(user_id)
    print("always-loaded (critical/identity):", always_loaded)

    # search_semantic_facts needs a query to embed against -- there's no
    # "list everything" method on HybridMemoryStore, so this is the closest
    # equivalent: a broad query with a low score threshold to surface most
    # of what's stored for this user.
    recalled = await store.search_semantic_facts(user_id, query=user_id, k=20, min_score=0.0)
    print("semantic (broad recall):", recalled)
    print("------------------\n")


async def stream_turn(inputs, config):
    """Streams node-by-node execution: AI thoughts, tool calls/results,
    memory events, plan changes. Unchanged from the original script --
    also used to stream the resume turn after a confirmation."""
    seen_ids = set()
    last_plan = None

    async for ns, chunk in graph.astream(inputs, config=config, stream_mode="values", subgraphs=True):
        semantic_facts = chunk.get("semantic_facts")
        if semantic_facts:
            print(f"    [MEMORY LOADED] {semantic_facts}")

        durable_fact = chunk.get("durable_fact")
        if durable_fact:
            print(f"    [MEMORY WRITE PENDING] {durable_fact}")

        episodic_event = chunk.get("episodic_event")
        if episodic_event:
            print(f"    [EPISODIC EVENT PENDING] {episodic_event}")

        plan_agents = chunk.get("plan_agents")
        plan_index = chunk.get("plan_index")
        plan_key = (tuple(plan_agents) if plan_agents else None, plan_index)
        if plan_agents is not None and plan_key != last_plan:
            print(f"    [PLAN] agents={plan_agents} index={plan_index}")
            last_plan = plan_key

        messages = chunk.get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        msg_id = getattr(last, "id", None) or id(last)
        if msg_id in seen_ids:
            continue
        seen_ids.add(msg_id)

        msg_type = getattr(last, "type", type(last).__name__)
        if msg_type == "ai":
            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    print(f"[AI -> TOOL CALL] {tc['name']}({tc['args']})")
            if last.content:
                print(f"[AI] {last.content}")
        elif msg_type == "tool":
            print(f"[TOOL RESULT: {last.name}] {last.content}")


async def handle_pending_interrupt(config):
    """
    Checks whether the TOP-LEVEL graph is paused on interrupt() -- this
    fires when patient_records (or any subgraph node) calls a write tool.
    Loops in case a single turn triggers more than one confirmation.

    IMPORTANT -- this is exactly what you need to verify before trusting
    routers/chat.py's interrupt detection: that route checks
    `result.get("__interrupt__")` on the return value of `graph.ainvoke(...)`,
    while this script instead calls `graph.aget_state(config)` and reads
    `state.tasks[0].interrupts[0].value`. Both SHOULD surface the same
    underlying interrupt for your installed langgraph version, since
    interrupts propagate up through subgraph boundaries to whatever the
    outermost stream/invoke call is -- but confirm that here first. If
    `result.get("__interrupt__")` turns out empty/missing in your version
    while this get_state check finds a pending task, switch chat.py's
    send_message to use `await graph.aget_state(config)` instead of
    relying on ainvoke's return value.
    """
    state = await graph.aget_state(config)
    while state.next:
        interrupt_payload = state.tasks[0].interrupts[0].value
        print(f"\n[CONFIRMATION NEEDED] {interrupt_payload}")
        user_decision = input("Type 'approve' or 'reject' (or 'exit' to leave it pending): ").strip().lower()

        if user_decision == "exit":
            print("Exiting without resolving the pending confirmation.")
            return

        if user_decision not in ("approve", "reject"):
            print("Please type exactly 'approve', 'reject', or 'exit'.")
            state = await graph.aget_state(config)
            continue

        # NOTE: sends a bare string, same as your subgraph script. If
        # hitl_confirm_node only checks `decision == "approve"` (string
        # equality), this works as-is. routers/chat.py's /resume route
        # sends a DICT ({"decision": "approve", "edited_args": ...})
        # instead -- confirm hitl_confirm_node actually handles both
        # shapes (see the earlier fix to hitl.py) before assuming the API
        # route behaves the same as this console test.
        await stream_turn(Command(resume=user_decision), config)
        state = await graph.aget_state(config)


async def main():
    global graph, store

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(SQLITE_CHECKPOINTER_PATH)
    checkpointer = None
    qdrant = None

    try:
        checkpointer = await checkpointer_cm.__aenter__()
        # AsyncSqliteSaver has no separate .setup() call -- schema is created on connect,
        # same as main.py.

        qdrant = QdrantClient(path=MEMORY_QDRANT_PATH)
        store = HybridMemoryStore(qdrant, SQLITE_EPISODIC_PATH)
        await store.setup()

        graph = await build_main_graph(checkpointer=checkpointer, store=store)

        role = input("[role: patient/staff] ").strip() or "patient"
        user_id = input("[user_id] ").strip() or "PT-00488"
        thread_id = input("[thread_id] ").strip() or "test-main-graph-1"
        config = {"configurable": {"thread_id": thread_id}}
        print("=== MedMate Main Graph -- interactive test (with HITL, sqlite + qdrant) ===\n")
        print("(commands: 'mem' to dump store, 'newsession' to switch thread_id, 'exit' to quit)\n")

        while True:
            user_input = input("[YOU] ").strip()

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.lower() == "mem":
                await dump_memory(user_id)
                continue

            if user_input.lower() == "newsession":
                new_thread_id = input("[new thread_id] ").strip()
                if new_thread_id:
                    config = {"configurable": {"thread_id": new_thread_id}}
                    print(f"    [SWITCHED TO THREAD: {new_thread_id}] (same user_id, same store)\n")
                continue

            if not user_input:
                continue

            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "role": role,
                "user_id": user_id,
            }
            await stream_turn(inputs, config)
            await handle_pending_interrupt(config)
    finally:
        if graph is not None:
            await shutdown_main_graph(graph)

        if checkpointer is not None:
            await checkpointer_cm.__aexit__(None, None, None)

        if store is not None:
            await store.close()

        if qdrant is not None:
            qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())