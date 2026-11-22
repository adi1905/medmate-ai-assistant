import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from medmate.agents import patient_records_agent
from medmate.memory.repo import MemoryRepo


def test_patient_records_agent_uses_shared_memory_context(monkeypatch) -> None:
    captured_messages = []

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(content="ok")

    monkeypatch.setattr(
        patient_records_agent,
        "ChatOpenAI",
        lambda **kwargs: FakeModel(),
    )

    patient_records_agent.call_model(
        {
            "role": "patient",
            "user_id": "PT-123",
            "messages": [HumanMessage(content="Book my next visit.")],
            "memory_profile": {"preferred_time_of_day": "morning"},
            "memory_recent": ["Prefers morning appointments"],
            "memory_relevant": [],
            "conv_summary": "",
            "semantic_facts": ["legacy fact that should not be used"],
        }
    )

    system_message = next(
        message for message in captured_messages
        if isinstance(message, SystemMessage)
    )
    assert "preferred time of day: morning" in system_message.content
    assert "legacy fact that should not be used" not in system_message.content
    assert "KNOWN PATIENT PREFERENCES" not in system_message.content


def test_patient_records_tools_do_not_include_legacy_memory_tool() -> None:
    assert all(
        tool.name != "remember_preference"
        for tool in patient_records_agent.ALL_PATIENT_RECORDS_TOOLS
    )


def test_memory_profile_returns_only_allowed_slots() -> None:
    class FakeStore:
        async def asearch(self, namespace, limit):
            return [
                SimpleNamespace(
                    key="preferred_time_of_day",
                    value={"value": "morning"},
                ),
                SimpleNamespace(
                    key="preferred_language",
                    value={"value": "English"},
                ),
            ]

    profile = asyncio.run(
        MemoryRepo(FakeStore()).get_profile(
            "PT-123",
            {"preferred_time_of_day"},
        )
    )

    assert profile == {"preferred_time_of_day": "morning"}
