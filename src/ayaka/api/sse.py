"""SSE event mapping — LangGraph astream_events -> SSE frames.

Emits: token / tool_call / tool_result / done / error.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE frame: `event: <name>\ndata: <json>\n\n`."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_chat(
    agent: CompiledStateGraph,
    device_sn: str,
    contact_id: str,
    session_id: str,
    message: str,
) -> AsyncIterator[str]:
    """Run one chat turn and yield SSE frames.

    Event mapping:
      on_chat_model_stream -> event: token        {data: <delta text>}
      on_tool_start         -> event: tool_call   {toolName}
      on_tool_end           -> event: tool_result  {toolName, data}
      final                 -> event: done        {sessionId}
      exception             -> event: error       {message}
    """
    config = {"configurable": {"thread_id": session_id}}
    state_in: dict[str, Any] = {
        "messages": [HumanMessage(content=message)],
        "device_sn": device_sn,
        "contact_id": contact_id,
        "session_id": session_id,
    }

    try:
        async for event in agent.astream_events(
            state_in,
            config=config,
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = _extract_text(chunk)
                if text:
                    yield _sse("token", {"type": "token", "data": text})

            elif kind == "on_tool_start":
                yield _sse(
                    "tool_call",
                    {"type": "tool_call", "toolName": event["name"]},
                )

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                yield _sse(
                    "tool_result",
                    {
                        "type": "tool_result",
                        "toolName": event["name"],
                        "data": _stringify(output),
                    },
                )

        yield _sse("done", {"type": "done", "sessionId": session_id})

    except Exception as exc:  # noqa: BLE001 — SSE needs a frame, not a traceback
        yield _sse("error", {"type": "error", "message": str(exc)})


def _extract_text(chunk: Any) -> str:
    """Pull plain text out of a chat-model stream chunk."""
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _stringify(value: Any) -> str:
    """Best-effort text form of a tool output for the SSE frame."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
