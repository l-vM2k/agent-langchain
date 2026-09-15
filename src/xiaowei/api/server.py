"""FastAPI server — chat endpoints (SSE stream + sync), mirroring AgentChatController."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from xiaowei.agent import AgentManager
from xiaowei.api.sse import stream_chat
from xiaowei.config import ConfigLoader
from xiaowei.intent import (
    IntentClassifier,
    IntentRule,
    IntentRuleSet,
    IntentShortcutEnhancer,
    IntentShortcutToolRegistry,
)
from xiaowei.memory.anonymous import derive_anonymous_contact_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="langchain-xiaowei", version="0.1.0")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_config = ConfigLoader(_PROJECT_ROOT / "config")
_agents = AgentManager(_config, _PROJECT_ROOT / ".data")

# 意图快捷通道 — 对应 IntentShortcutEnhancer 装配
_intent_rules = IntentRuleSet(
    enabled=_config.intent.enabled,
    rules=[
        IntentRule(
            intent_type=r["intentType"],
            pattern=r["pattern"],
            priority=int(r.get("priority", 100)),
            enhance_template=r.get("enhanceTemplate", ""),
        )
        for r in _config.intent.rules
    ],
)
_intent_enhancer = IntentShortcutEnhancer(
    IntentClassifier(_intent_rules), IntentShortcutToolRegistry()
)


class ChatRequest(BaseModel):
    device_sn: str = Field(default="default")
    message: str
    session_id: str | None = None
    contact_id: str | None = None


def _resolve_ids(request: ChatRequest) -> tuple[str, str]:
    """解析 session_id / contact_id(匿名派生)— 对应 Controller 入口兜底。"""
    session_id = request.session_id or str(uuid.uuid4())
    contact_id = request.contact_id or derive_anonymous_contact_id(
        request.device_sn, session_id
    )
    return session_id, contact_id


@app.post("/agent/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE streaming chat, mirrors AgentChatController.chatStream."""
    session_id, contact_id = _resolve_ids(request)
    try:
        agent = _agents.get_agent(request.device_sn)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # 意图快捷增强(命中则预调工具注入,未命中原样)
    message = _intent_enhancer.enhance(request.message)

    return StreamingResponse(
        stream_chat(
            agent,
            device_sn=request.device_sn,
            contact_id=contact_id,
            session_id=session_id,
            message=message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


@app.post("/agent/chat/sync")
async def chat_sync(request: ChatRequest) -> dict:
    """Blocking chat, mirrors AgentChatController.chatSync."""
    session_id, contact_id = _resolve_ids(request)
    try:
        agent = _agents.get_agent(request.device_sn)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    message = _intent_enhancer.enhance(request.message)
    config = {"configurable": {"thread_id": session_id}}
    result = await agent.ainvoke(
        {
            "messages": [("user", message)],
            "device_sn": request.device_sn,
            "contact_id": contact_id,
            "session_id": session_id,
        },
        config=config,
    )
    last = result["messages"][-1]
    return {
        "sessionId": session_id,
        "content": getattr(last, "content", str(last)),
    }


@app.get("/agents")
async def list_agents() -> dict:
    """List configured agents (debug helper)."""
    return {"agents": _config.list_agents()}


@app.get("/soul/state")
async def soul_state(device_sn: str = "default", contact_id: str = "") -> dict:
    """Debug: 查看灵魂状态(状态机转移结果)。"""
    if contact_id:
        agent = _agents.get_agent(device_sn)
        # 从 manager 侧的 orchestrator store 读状态
        # (agent 构建时 bundle 了 state_store;此处通过 data 文件读)
        import json as _json
        from pathlib import Path as _Path

        state_file = _PROJECT_ROOT / ".data" / "soul_state.json"
        if state_file.exists():
            data = _json.loads(state_file.read_text(encoding="utf-8"))
            state = data.get(device_sn, {}).get(contact_id)
            if state:
                return {"device_sn": device_sn, "contact_id": contact_id, "state": state}
        return {"device_sn": device_sn, "contact_id": contact_id, "state": None}
    return {"device_sn": device_sn, "contact_id": contact_id, "state": None}
