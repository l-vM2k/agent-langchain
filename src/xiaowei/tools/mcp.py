"""MCP tools — mirrors McpClientBuilder (LOCAL stdio / REMOTE streamable_http).

Uses langchain-mcp-adapters when installed; falls back to a no-op with a
clear log message so the rest of the system works without the extra dep.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def create_mcp_tools(server_configs: dict[str, dict[str, Any]]) -> list[Any]:
    """按配置创建 MCP 工具集 — 对应 registerLocalTool/registerRemoteTool。

    server_configs: {name: {command/args/env | url, transport}}
    Returns a list of LangChain tools (possibly empty if the adapter is absent).
    """
    if not server_configs:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters 未安装,跳过 MCP 工具注册。"
            "安装: pip install 'xiaowei[mcp]'"
        )
        return []

    client = MultiServerMCPClient(server_configs)
    try:
        tools = await client.get_tools()
        logger.info("MCP 工具注册完成: count=%d", len(tools))
        return tools
    except Exception as exc:  # noqa: BLE001 — MCP 不可用不阻塞 Agent 启动
        logger.warning("MCP 工具注册失败(降级为无 MCP): %s", exc)
        return []
