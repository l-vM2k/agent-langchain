"""Fake LLM for offline testing — streams canned responses without any network.

Set XIAOWEI_FAKE_LLM=1 (or pass fake=True) to use this instead of a real provider.
Useful for verifying the SSE pipeline end-to-end without an API key.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

CANNED_REPLY = (
    "你好呀,我是小薇。这是一条来自 FakeLLM 的离线回复:"
    "SSE 流式链路已经跑通了。等你填上真实的 API Key,"
    "把 XIAOWEI_FAKE_LLM 环境变量去掉,我就会变成真正的大模型。"
)


class FakeChatModel(BaseChatModel):
    """A deterministic chat model that streams a canned reply token by token.

    Implements the same BaseChatModel interface as ChatOpenAI, so the graph
    and SSE layer cannot tell the difference. Supports:
      - sync/async generate
      - sync/async streaming
      - bind_tools (returns self — the ReAct loop gets no tool calls)
    """

    chunk_size: int = 8
    reply: str = CANNED_REPLY

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[Union[dict[str, Any], type, Any]],
        **kwargs: Any,
    ) -> Runnable:
        """Accept tool binding but never emit tool calls (canned reply only)."""
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        await asyncio.sleep(0)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))]
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for i in range(0, len(self.reply), self.chunk_size):
            piece = self.reply[i : i + self.chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for i in range(0, len(self.reply), self.chunk_size):
            piece = self.reply[i : i + self.chunk_size]
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            await asyncio.sleep(0.02)  # small delay so SSE feels like a real stream

