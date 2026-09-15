"""Tool factory.

Assembles the tool list for an agent from config:
  - builtin tools (current_time / calculator) filtered by allow/deny
  - REST tools from config/tools/rest.yaml
  - MCP tools from config/tools/mcp_*.yaml (async, optional dep)
  - knowledge search from config/knowledge/base.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from ayaka.config import AgentConfig, ConfigLoader

logger = logging.getLogger(__name__)


class ToolFactory:
    """按 AgentConfig 组装工具集。"""

    def __init__(self, config_loader: ConfigLoader) -> None:
        self._config = config_loader

    def build_sync_tools(self, agent_cfg: AgentConfig) -> list[BaseTool]:
        """构建同步可用工具(builtin + REST + knowledge)。

        MCP 工具是异步的,单独走 build_mcp_tools。
        """
        tools: list[BaseTool] = []
        tools.extend(self._builtin_tools(agent_cfg))
        tools.extend(self._rest_tools(agent_cfg))
        tool = self._knowledge_tool(agent_cfg)
        if tool is not None:
            tools.append(tool)
        return self._apply_filter(tools, agent_cfg)

    async def build_mcp_tools(self, agent_cfg: AgentConfig) -> list[BaseTool]:
        """构建 MCP 工具(LOCAL stdio / REMOTE http)。"""
        from ayaka.tools.mcp import create_mcp_tools

        server_configs: dict[str, dict[str, Any]] = {}
        for tool_cfg in self._config.list_tools():
            if tool_cfg.type not in ("LOCAL", "REMOTE"):
                continue
            if tool_cfg.type == "LOCAL":
                server_configs[tool_cfg.name] = {
                    "command": tool_cfg.config.get("command", ""),
                    "args": tool_cfg.config.get("args", []),
                    "env": tool_cfg.config.get("env", {}),
                    "transport": "stdio",
                }
            else:
                server_configs[tool_cfg.name] = {
                    "url": tool_cfg.config.get("baseUrl", ""),
                    "transport": tool_cfg.config.get("transport", "streamable_http"),
                }
        if not server_configs:
            return []
        tools = await create_mcp_tools(server_configs)
        return self._apply_filter(tools, agent_cfg)

    # ---------- 内部 ----------

    def _builtin_tools(self, agent_cfg: AgentConfig) -> list[BaseTool]:
        """内置工具: current_time + calculator。"""
        from ayaka.tools.builtin import calculator, current_time

        return [current_time, calculator]

    def _rest_tools(self, agent_cfg: AgentConfig) -> list[BaseTool]:
        """REST 工具。"""
        from ayaka.tools.rest import create_rest_tool

        tools: list[BaseTool] = []
        for tool_cfg in self._config.list_tools():
            if tool_cfg.type != "REST":
                continue
            tools.append(
                create_rest_tool(
                    name=tool_cfg.name,
                    description=tool_cfg.config.get("description", tool_cfg.name),
                    url=tool_cfg.config.get("url", ""),
                    method=tool_cfg.config.get("method", "POST"),
                    headers=tool_cfg.config.get("headers"),
                    timeout_seconds=float(tool_cfg.config.get("timeout", 10)),
                )
            )
        return tools

    def _knowledge_tool(self, agent_cfg: AgentConfig) -> BaseTool | None:
        """知识库检索工具。"""
        if not agent_cfg.knowledge_enabled:
            return None
        knowledge_path = self._config.config_dir / "knowledge" / "base.json"
        if not knowledge_path.exists():
            return None
        from ayaka.tools.knowledge import create_knowledge_tool

        return create_knowledge_tool(knowledge_path)

    @staticmethod
    def _apply_filter(tools: list[BaseTool], agent_cfg: AgentConfig) -> list[BaseTool]:
        """tools_allow / tools_deny 过滤。"""
        result = tools
        if agent_cfg.tools_allow:
            allow = set(agent_cfg.tools_allow)
            result = [t for t in result if t.name in allow]
        if agent_cfg.tools_deny:
            deny = set(agent_cfg.tools_deny)
            result = [t for t in result if t.name not in deny]
        return result
