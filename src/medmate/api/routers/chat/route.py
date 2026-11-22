import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import json
from ...core.security import (CurrentUser, get_current_user, Role)
from ...core.dependencies import get_graph

from ...routers.chat.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    HITLResumeRequest,
    HITLResumeResponse,
    MessageOut,
    MessageRole,
    PendingAction,
    ThreadCreateRequest,
    ThreadHistoryResponse,
    ThreadListResponse,
    ThreadOut
)

from ...core.common_schemas import TraceMeta
from ....db.repositories import chat_repo
from ....db.utils.datetime_utils import now_ist_str
from ....guards.pii import redact_pii

import logging

logger = logging.getLogger("medmate.chat")




router = APIRouter(prefix="/chat", tags = ["chat"])


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"
 
 


def _graph_role(role: Role) -> str:
    """
    Your graph's _tools_for_role() only branches on the literal string
    'staff' -- everything else gets the (more restricted) patient tool
    set. Role.ADMIN must be mapped to 'staff' explicitly here, or an
    admin user would silently get patient-level tool access inside the
    graph despite passing every API-layer permission check.
    """
    return "staff" if role in (Role.STAFF, Role.ADMIN) else "patient"

async def _run_graph_with_tool_logging(graph, inputs, config) -> dict:
    """
    Streams the graph turn (same as test_main_graph_manual.py's stream_turn)
    so tool calls/results are logged, then returns the final state dict --
    equivalent to what graph.ainvoke(...) would have returned, so the rest
    of send_message() doesn't need to change.
    """
    seen_ids = set()
    final_chunk = None

    async for ns, chunk in graph.astream(inputs, config=config, stream_mode="values", subgraphs=True):
        final_chunk = chunk  # last value wins; astream in "values" mode yields full state each step

        messages = chunk.get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        msg_id = getattr(last, "id", None) or id(last)
        if msg_id in seen_ids:
            continue
        seen_ids.add(msg_id)

        msg_type = getattr(last, "type", type(last).__name__)
        if msg_type == "ai" and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                logger.info("TOOL CALL ns=%s name=%s args=%s", ns, tc["name"], tc["args"])
        elif msg_type == "tool":
            logger.info("TOOL RESULT ns=%s name=%s output=%s", ns, last.name, last.content)

    # __interrupt__ isn't part of stream_mode="values" chunks the same way
    # ainvoke's return dict exposes it -- pull it from state instead.
    state = await graph.aget_state(config)
    result = dict(final_chunk or {})
    if state.next:
        result["__interrupt__"] = [t.interrupts[0] for t in state.tasks if t.interrupts]
    return result


def _memory_identity(user: CurrentUser) -> str:
    return user.patient_id if user.role == Role.PATIENT else user.user_id


def _persist_message(thread_id: str, role: MessageRole, content: str) -> MessageOut:
    message_id = _new_id("MSG")
    created_at = now_ist_str()

    chat_repo.create_chat_message(message_id, thread_id, role, content, created_at)
    return MessageOut(message_id=message_id, role = role, content = content, created_at=datetime.fromisoformat(created_at))

def _touch_thread(thread_id: str) -> None:
    last_message_at = now_ist_str()
    chat_repo.update_chat_thread_last_message(thread_id=thread_id, last_message_at=last_message_at)
    return None

def _log_turn(thread_id: str, user_id: str, role: str, event_type: str,
              raw_summary_text: str = "", trace_id: Optional[str] = None,
              tool_call_count: int = 0, latency_ms: Optional[float] = None,
              hitl_outcome: Optional[str] = None) -> None:
    redacted = redact_pii(raw_summary_text)
    log_id = _new_id("LOG")
    created_at = now_ist_str()
    chat_repo.create_request_log(log_id=log_id, thread_id=thread_id, user_id=user_id,
                                 role=role, event_type=event_type, summary=redacted, created_at=created_at,
                                 trace_id=trace_id, tool_call_count=tool_call_count, latency_ms=latency_ms,
                                 hitl_outcome=hitl_outcome)
    return None


def _assert_owns_thread(thread_id: str, user: CurrentUser) -> None:
    owner = chat_repo.fetch_chat_thread_owner(thread_id)

    if owner is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    if owner["user_id"] != user.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return None

@router.post("/threads", response_model = ThreadOut, status_code=status.HTTP_201_CREATED)
def create_thread(body: ThreadCreateRequest, user: CurrentUser = Depends(get_current_user)):
    thread_id = _new_id("THR")
    created_at = now_ist_str()
    chat_repo.create_chat_thread(thread_id=thread_id, user_id = user.user_id, title = body.title,
                                 created_at=created_at)
    return ThreadOut(thread_id=thread_id, title = body.title, created_at=created_at,
                     message_count=0)


@router.get("/threads", response_model=ThreadListResponse)
def list_threads(user: CurrentUser = Depends(get_current_user)):
    threads = chat_repo.list_chat_threads(user.user_id)
    return ThreadListResponse(threads=[
        ThreadOut(
            thread_id=thread["thread_id"],
                title=thread["title"],
                created_at=thread["created_at"],
                last_message_at=(
                    thread["last_message_at"]
                    if thread["last_message_at"]
                    else None
                ),
                message_count=0
        )
        for thread in threads
    ])

@router.get("/threads/{thread_id}/messages", response_model=ThreadHistoryResponse)
def get_thread_history(thread_id: str, user: CurrentUser = Depends(get_current_user)):
    _assert_owns_thread(thread_id, user)
    messages = chat_repo.fetch_chat_messages(thread_id=thread_id)
    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=[
            MessageOut(
                message_id=message["message_id"],
                role=MessageRole(message["role"]),
                content=message["content"],
                created_at=message["created_at"],
            )
            for message in messages
        ],
    )

@router.post("/threads/{thread_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    thread_id: str,
    body: ChatMessageRequest,
    user: CurrentUser = Depends(get_current_user),
    graph=Depends(get_graph),
):
    _assert_owns_thread(thread_id, user)
    _persist_message(thread_id, MessageRole.USER, body.content)
    config = {"configurable":{"thread_id": thread_id}}
    inputs = {
        "messages":[HumanMessage(content=body.content)],
        "role":_graph_role(user.role),
        "user_id":_memory_identity(user)
    }
    start = time.perf_counter()
    try:
        result = await _run_graph_with_tool_logging(graph, inputs, config)
        print("result:", result)
    except Exception as exc:  # noqa: BLE001 -- error-tracking boundary
        latency_ms = (time.perf_counter() - start) * 1000
        _log_turn(thread_id, user.user_id, user.role, "error",
                   raw_summary_text=str(exc),latency_ms=latency_ms)
        raise HTTPException(status_code=502, detail="Agent failed to process the message") from exc
    latency_ms = (time.perf_counter() - start) * 1000
    if result.get("__interrupt__"):
        intr = result["__interrupt__"][0].value
        action = intr["action"]
        action_id = _new_id("ACT")
        summary = intr.get("question", action.get("summary", "Confirm this action"))
        chat_repo.create_pending_action(
            action_id=action_id, thread_id=thread_id, tool_name=action["action"],
            tool_args_json=json.dumps(action["params"]), summary=summary,
            created_at=now_ist_str(),
        )
        assistant_msg = _persist_message(thread_id, MessageRole.ASSISTANT, summary)
        _touch_thread(thread_id)
        _log_turn(thread_id, user.user_id, user.role, "hitl_prompt", raw_summary_text=summary,
                  tool_call_count=1, latency_ms = latency_ms)
        return ChatMessageResponse(
            thread_id=thread_id,
            message=assistant_msg,
            pending_action=PendingAction(
                action_id=action_id,
                tool_name=action["action"],
                tool_args=action["params"],
                human_readable_summary=summary,
            ),
            trace=TraceMeta(latency_ms=latency_ms)
        )

    reply_text = result["messages"][-1].content
    assistant_msg = _persist_message(thread_id, MessageRole.ASSISTANT, reply_text)
    _touch_thread(thread_id=thread_id)
    _log_turn(thread_id, user.user_id, user.role, "message", 
              raw_summary_text=reply_text, latency_ms=latency_ms)
    return ChatMessageResponse(thread_id=thread_id, message=assistant_msg, trace=TraceMeta(latency_ms=latency_ms))


@router.post("/threads/{thread_id}/resume", response_model=HITLResumeResponse)
async def resume_pending_action(
    thread_id: str,
    body: HITLResumeRequest,
    user: CurrentUser = Depends(get_current_user),
    graph=Depends(get_graph),
):
    _assert_owns_thread(thread_id, user)
    pending = chat_repo.get_pending_action(body.action_id)
    if pending is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such confirmation")
    if pending["thread_id"] != thread_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Confirmation does not belong to this thread")
    if pending["status"] != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                             detail=f"This confirmation is already {pending['status']}")
    config = {"configurable":{"thread_id":thread_id}}
    if body.decision.value == "reject":
        resume_value = {"decision":"rejected"}
        outcome = "rejected"
    else:
        resume_value = {"decision": "approve", "edited_args": body.edited_args}
        outcome = "edited" if body.decision.value == "edit_and_approve" else "approved"
    resolved = chat_repo.resolve_pending_action(body.action_id, outcome, now_ist_str())
    if not resolved:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This confirmation was just resolved elsewhere")
    start = time.perf_counter()
    result = await _run_graph_with_tool_logging(graph, Command(resume=resume_value), config)
    latency_ms = (time.perf_counter() - start) * 1000
    reply_text = result["messages"][-1].content
    assistant_msg = _persist_message(thread_id, MessageRole.ASSISTANT, reply_text)
    _touch_thread(thread_id)
    _log_turn(thread_id, user.user_id, user.role, "hitl_resolved",
              raw_summary_text=reply_text, hitl_outcome=outcome, latency_ms=latency_ms)
    return HITLResumeResponse(thread_id=thread_id, message=assistant_msg,
                               executed=outcome != "rejected",
                               trace=TraceMeta(latency_ms=latency_ms))

