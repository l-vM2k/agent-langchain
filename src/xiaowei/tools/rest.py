"""REST tools — mirrors RestClientWrapper.java + RestToolDefinition.java.

Builds a StructuredTool per REST tool config: the LLM fills the args,
we render the URL template, call the endpoint, return the body text.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class RestCallArgs(BaseModel):
    """LLM 提供的调用参数(动态字段由 description 引导)。"""

    input: str = Field(default="", description="请求参数或查询内容")


def create_rest_tool(
    name: str,
    description: str,
    url: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 10.0,
) -> StructuredTool:
    """创建一个 REST 工具 — 对应 RestToolDefinition + RestClientWrapper。

    简化版: LLM 只传一个 input 字符串,作为 JSON body 的 `input` 字段发出。
    """

    async def _call(input: str = "") -> str:
        payload: dict[str, Any] = {"input": input}
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                if method.upper() in ("GET",):
                    resp = await client.request(
                        method.upper(), url, params={"input": input}, headers=headers or {}
                    )
                else:
                    resp = await client.request(
                        method.upper(), url, json=payload, headers=headers or {}
                    )
                resp.raise_for_status()
                return resp.text
        except httpx.HTTPError as exc:
            return f"REST 调用失败: {exc}"

    return StructuredTool.from_function(
        coroutine=_call,
        name=name,
        description=description,
        args_schema=RestCallArgs,
    )
