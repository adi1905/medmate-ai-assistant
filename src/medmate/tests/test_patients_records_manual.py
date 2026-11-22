# scripts/test_patient_records_manual.py

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from ..agents.patient_records_agent import build_patient_records_subgraph

subgraph = build_patient_records_subgraph(checkpointer=MemorySaver())


config = {"configurable": {"thread_id": "test-thread-1"}}

# These now travel in the STATE (the `inputs` dict), not config.
TEST_ROLE = "patient"
TEST_USER_ID = "PT-00089"


def stream_turn(subgraph, inputs, config):
    """
    Streams node-by-node execution: shows AI thoughts (tool_calls + content),
    tool outputs, and interrupts as they happen.
    """
    for chunk in subgraph.stream(inputs, config=config, stream_mode="values"):
        messages = chunk.get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        msg_type = getattr(last, "type", type(last).__name__)

        if msg_type == "ai":
            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    print(f"[AI -> TOOL CALL] {tc['name']}({tc['args']})")
            if last.content:
                print(f"[AI] {last.content}")

        elif msg_type == "tool":
            print(f"[TOOL RESULT: {last.name}] {last.content}")

        elif msg_type == "human":
            print(f"[HUMAN] {last.content}")


def handle_pending_interrupt(subgraph, config):
    state = subgraph.get_state(config)
    while state.next:
        interrupt_payload = state.tasks[0].interrupts[0].value
        print(f"\n[CONFIRMATION NEEDED] {interrupt_payload}")
        user_decision = input("Type 'approve' or 'reject': ").strip().lower()

        if user_decision == "exit":
            print("Exiting without resolving the pending confirmation.")
            return  # leave the graph paused; don't call Command(resume=...) at all

        if user_decision not in ("approve", "reject"):
            print("Please type exactly 'approve', 'reject', or 'exit'.")
            continue

        stream_turn(subgraph, Command(resume=user_decision), config)
        state = subgraph.get_state(config)


def main():
    print("=== MedMate Patient Records — interactive test ===")
    print(f"(testing as role={TEST_ROLE!r}, user_id={TEST_USER_ID!r} -- edit the "
          f"constants above to test as staff or a different patient)")
    print("Type your message and press Enter. Type 'exit' to quit.\n")

    while True:
        user_input = input("[YOU] ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break

        if not user_input:
            continue

        # role/user_id passed here, every turn -- this is what
        # call_model()'s state.get("role")/state.get("user_id") actually
        # reads. Harmless to resend the same values each turn; the
        # checkpointer just overwrites them with the same value.
        stream_turn(
            subgraph,
            {
                "messages": [HumanMessage(content=user_input)],
                "role": TEST_ROLE,
                "user_id": TEST_USER_ID,
            },
            config,
        )

        handle_pending_interrupt(subgraph, config)


if __name__ == "__main__":
    main()