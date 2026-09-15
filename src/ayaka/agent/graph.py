"""LangGraph StateGraph definition — the core agent pipeline.

The agent runs as a 6-node graph:

    START -> prepare_soul -> retrieve_reme -> inject_soul_prompt
            -> react_agent -> record_reme -> finalize_soul -> END

- prepare_soul:     personality/emotion/behavior/memory/prompt preparation
- retrieve_reme:    ReMe retrieve -> inject as a SystemMessage
- inject_soul_prompt: soul prompt -> SystemMessage (per-turn dynamic)
- react_agent:      create_react_agent subgraph (LLM <-> tools)
- record_reme:      ReMe record (this turn's user/assistant text)
- finalize_soul:    state machine transition + memory extraction
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from ayaka.memory.reme_store import ReMeMemoryStore
from ayaka.soul.models import (
    PersonalityProfile,
    SoulContext,
    SoulTurnTrace,
)
from ayaka.soul.orchestrator import SoulOrchestrator


class AgentState(TypedDict):
    """Graph state — the per-turn context bundle."""

    # identity / routing keys (set once per request)
    device_sn: str
    contact_id: str
    session_id: str

    # conversation history (reducer: append)
    messages: Annotated[list[BaseMessage], add_messages]

    # soul pipeline (set by prepare_soul, consumed downstream)
    soul_context: Optional[SoulContext]

    # final assistant text (set after react_agent for record/finalize)
    assistant_text: str

    # user text of this turn (for record/finalize)
    user_text: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _make_prepare_soul(orchestrator: SoulOrchestrator, personality: PersonalityProfile):
    """prepare_soul 节点"""

    async def prepare_soul(state: AgentState) -> dict:
        user_text = _last_user_text(state["messages"])
        context = orchestrator.prepare(
            personality=personality,
            device_sn=state["device_sn"],
            contact_id=state["contact_id"],
            session_id=state["session_id"],
            user_text=user_text,
        )
        return {"soul_context": context, "user_text": user_text}

    return prepare_soul


def _make_retrieve_reme(reme_store: ReMeMemoryStore):
    """retrieve_reme 节点

    检索长期记忆,命中则注入为一条 SystemMessage(而非 ToolMessage,
    因为没有 preceding AI tool_call,OpenAI 协议会拒绝裸 ToolMessage)。
    """

    async def retrieve_reme(state: AgentState) -> dict:
        context: SoulContext | None = state.get("soul_context")
        if context is None or not context.enabled:
            return {}
        user_text = state.get("user_text") or _last_user_text(state["messages"])
        memory_text = await reme_store.retrieve(
            state["device_sn"], state["contact_id"], user_text
        )
        if not memory_text:
            return {}
        from langchain_core.messages import SystemMessage

        return {"messages": [SystemMessage(content=memory_text)]}

    return retrieve_reme


def _make_react_agent(llm: BaseChatModel, tools: list[BaseTool], sys_prompt: str):
    """react_agent 节点 — create_react_agent 子图。

    系统提示 = agent sys_prompt + 灵魂 prompt(本轮人格上下文)。
    灵魂 prompt 由 prepare_soul 节点写入 state,这里在调用前拼到
    system message 里,让 LLM 感知人格/情绪/记忆。
    """
    return create_react_agent(model=llm, tools=tools, prompt=sys_prompt or None)


def _make_inject_soul_prompt():
    """inject_soul_prompt 节点 — 把灵魂 prompt 拼到 messages 头部。

    create_react_agent 的 prompt 参数是静态的(构建时绑定),无法在
    每轮动态注入。本节点在 react_agent 之前把灵魂 prompt 作为一条
    SystemMessage 插入 messages,让 LLM 感知本轮人格上下文。
    """

    async def inject_soul_prompt(state: AgentState) -> dict:
        context: SoulContext | None = state.get("soul_context")
        if context is None or not context.enabled or not context.prompt:
            return {}
        from langchain_core.messages import SystemMessage

        return {"messages": [SystemMessage(content=context.prompt)]}

    return inject_soul_prompt


def _make_record_reme(reme_store: ReMeMemoryStore):
    """record_reme 节点(本轮对话写入)"""

    async def record_reme(state: AgentState) -> dict:
        assistant_text = _last_assistant_text(state["messages"])
        user_text = state.get("user_text") or _last_user_text(state["messages"])
        await reme_store.record(
            state["device_sn"], state["contact_id"], user_text, assistant_text
        )
        return {"assistant_text": assistant_text}

    return record_reme


def _make_finalize_soul(orchestrator: SoulOrchestrator):
    """finalize_soul 节点(状态机 + 记忆摄取)"""

    async def finalize_soul(state: AgentState) -> dict:
        context: SoulContext | None = state.get("soul_context")
        if context is None or not context.enabled:
            return {}
        assistant_text = state.get("assistant_text") or _last_assistant_text(
            state["messages"]
        )
        user_text = state.get("user_text") or _last_user_text(state["messages"])
        trace = _build_turn_trace(state["messages"])
        # 收尾失败不阻塞(orchestrator.finalize 内部已吞异常)
        orchestrator.finalize(context, user_text, assistant_text, trace)
        return {}

    return finalize_soul


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_user_text(messages: list[BaseMessage]) -> str:
    """取最后一条 user 消息文本"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
    return ""


def _last_assistant_text(messages: list[BaseMessage]) -> str:
    """取最后一条 assistant 消息文本(排除 tool message)。"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai":
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
    return ""


def _build_turn_trace(messages: list[BaseMessage]) -> SoulTurnTrace:
    """从消息流统计工具调用轨迹。"""
    tool_call_count = 0
    tool_success_count = 0
    tool_failure_count = 0
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            tool_call_count += len(tool_calls)
        elif msg_type == "tool":
            content = msg.content if isinstance(msg.content, str) else ""
            if content.startswith(("计算失败", "REST 调用失败", "错误", "Error", "error")):
                tool_failure_count += 1
            else:
                tool_success_count += 1
    assistant_text = _last_assistant_text(messages)
    return SoulTurnTrace(
        tool_call_count=tool_call_count,
        tool_success_count=tool_success_count,
        tool_failure_count=tool_failure_count,
        error_occurred=False,
        assistant_text_length=len(assistant_text),
    )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    sys_prompt: str,
    orchestrator: SoulOrchestrator,
    reme_store: ReMeMemoryStore,
    personality: PersonalityProfile,
) -> CompiledStateGraph:
    """Build the full 6-node agent graph.

    Structure:
        START -> prepare_soul -> retrieve_reme -> inject_soul_prompt
                -> react_agent -> record_reme -> finalize_soul -> END

    - prepare_soul:     loads personality/state, perceives emotion, generates
                        behavior signal, recalls memories, composes soul prompt
    - retrieve_reme:    retrieves long-term memory -> SystemMessage
    - inject_soul_prompt: soul prompt -> SystemMessage (per-turn dynamic)
    - react_agent:      create_react_agent subgraph (LLM <-> tools), with the
                        static sys_prompt as its base system prompt
    - record_reme:      writes this turn's user/assistant text to ReMe workspace
    - finalize_soul:    state machine transition + memory extraction
    """
    graph = StateGraph(AgentState)

    graph.add_node("prepare_soul", _make_prepare_soul(orchestrator, personality))
    graph.add_node("retrieve_reme", _make_retrieve_reme(reme_store))
    graph.add_node("inject_soul_prompt", _make_inject_soul_prompt())
    graph.add_node("react_agent", _make_react_agent(llm, tools, sys_prompt))
    graph.add_node("record_reme", _make_record_reme(reme_store))
    graph.add_node("finalize_soul", _make_finalize_soul(orchestrator))

    graph.add_edge(START, "prepare_soul")
    graph.add_edge("prepare_soul", "retrieve_reme")
    graph.add_edge("retrieve_reme", "inject_soul_prompt")
    graph.add_edge("inject_soul_prompt", "react_agent")
    graph.add_edge("react_agent", "record_reme")
    graph.add_edge("record_reme", "finalize_soul")
    graph.add_edge("finalize_soul", END)

    return graph.compile(checkpointer=None)


__all__ = [
    "AgentState",
    "build_agent_graph",
]
