from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from ..agents.analytics_agent import build_analytics_subgraph

subgraph = build_analytics_subgraph(checkpointer=MemorySaver())
config = {
    "configurable": {
        "thread_id": "test-thread-analytics-1",
        "role": "staff",
        "user_id": "PR-001",
        "allowed_tools": [
            "no_show_rate_by_period",
            "appointment_volume_trend",
            "top_medications",
            "flagged_lab_summary",
            "provider_utilization",
            "answer_custom_query",
            "generate_chart"
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

        elif msg_type == "tool" and last.name == "generate_chart":
            import json, base64
            content = json.loads(last.content) if isinstance(last.content, str) else last.content
            img_b64 = content.get("data", {}).get("image_base64")
            if img_b64:
                with open("last_chart.png", "wb") as f:
                    f.write(base64.b64decode(img_b64))
                print(f"[TOOL RESULT: generate_chart] saved to last_chart.png")
            else:
                print(f"[TOOL RESULT: generate_chart] {last.content}")

        elif msg_type == "tool":
            print(f"[TOOL RESULT: {last.name}] {last.content}")

        elif msg_type == "human":
            print(f"[HUMAN] {last.content}")


def main():
    print("=== MedMate Analytics — interactive test ===")
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
