from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ..agents.med_lab_qa_agent import build_med_lab_qa_subgraph

subgraph = build_med_lab_qa_subgraph(checkpointer=MemorySaver())
config = {
    "configurable": {
        "thread_id": "test-thread-med-lab-qa-1",
        "role": "patient",
        "user_id": "PT-001",
        "allowed_tools": [
            "retrieve_chunks",
            "relevance_filter",
            "answer_medication_question",
            "explain_lab_term",
        ],
    }
}


def stream_turn(subgraph, inputs, config):
    """
    Streams node-by-node execution: shows AI thoughts (tool_calls + content)
    and tool outputs as they happen.
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


def main():
    print("=== MedMate Medication & Lab QA — interactive test ===")
    print("Type your message and press Enter. Type 'exit' to quit.\n")

    while True:
        user_input = input("[YOU] ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break

        if not user_input:
            continue

        stream_turn(
            subgraph,
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )


if __name__ == "__main__":
    main()