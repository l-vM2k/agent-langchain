"""Configuration loading — YAML files under config/, mirroring the DB-backed config of the Java side."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_env(value: str) -> str:
    """Expand ${VAR} placeholders from environment variables."""
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env(raw)


def _expand_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _resolve_env(node)
    return node


@dataclass
class ModelConfig:
    """One chat model definition (mirrors builder_chat_model row)."""

    model_id: int
    name: str
    provider_code: str
    api_host: str
    api_key: str
    model_name: str
    stream: bool = True
    temperature: float = 0.7
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """One tool definition (mirrors builder_mcp_tool row)."""

    name: str
    type: str  # LOCAL | REMOTE | REST | BUILTIN
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """One agent definition (mirrors builder_agent row)."""

    device_sn: str
    name: str
    model_id: int
    sys_prompt: str
    max_iters: int = 10
    tools_allow: list[str] = field(default_factory=list)
    tools_deny: list[str] = field(default_factory=list)
    knowledge_enabled: bool = True
    # 人格
    persona_type: str = "WARM_COMPANION"
    personality_dimensions: dict[str, float] = field(default_factory=dict)
    # 灵魂链路开关
    soul_enabled: bool = True
    # ReMe 长期记忆开关
    reme_enabled: bool = True


@dataclass
class IntentConfig:
    """意图快捷通道配置(mirrors application.yml intent 节点)。"""

    enabled: bool = True
    rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PersonalityConfig:
    """人格配置聚合(strategies + prompt_rules)。"""

    strategies: dict[str, Any] = field(default_factory=dict)
    prompt_rules: dict[str, str] = field(default_factory=dict)


class ConfigLoader:
    """Loads YAML config from config/ directory.

    Layout:
      config/models/*.yaml        -> ModelConfig (one per file)
      config/agents/*.yaml       -> AgentConfig (one per file)
      config/tools/*.yaml        -> ToolConfig (list per file)
      config/intent.yaml         -> IntentConfig
      config/personality/*.yaml  -> PersonalityConfig parts
    """

    def __init__(self, config_dir: str | Path = "config") -> None:
        self.config_dir = Path(config_dir)
        self._models: dict[int, ModelConfig] = {}
        self._agents: dict[str, AgentConfig] = {}
        self._tools: list[ToolConfig] = []
        self._intent: IntentConfig = IntentConfig()
        self._personality: PersonalityConfig = PersonalityConfig()
        self.reload()

    def reload(self) -> None:
        self._models.clear()
        self._agents.clear()
        self._tools.clear()

        # models
        models_dir = self.config_dir / "models"
        if models_dir.is_dir():
            for path in sorted(models_dir.glob("*.yaml")):
                data = _load_yaml(path)
                cfg = ModelConfig(
                    model_id=int(data["model_id"]),
                    name=data.get("name", path.stem),
                    provider_code=data.get("provider_code", "openai"),
                    api_host=data.get("api_host", ""),
                    api_key=data.get("api_key", ""),
                    model_name=data.get("model_name", ""),
                    stream=bool(data.get("stream", True)),
                    temperature=float(data.get("temperature", 0.7)),
                    extra_body=data.get("model_param_json") or {},
                )
                self._models[cfg.model_id] = cfg

        # agents
        agents_dir = self.config_dir / "agents"
        if agents_dir.is_dir():
            for path in sorted(agents_dir.glob("*.yaml")):
                data = _load_yaml(path)
                cfg = AgentConfig(
                    device_sn=data["device_sn"],
                    name=data.get("name", path.stem),
                    model_id=int(data["model_id"]),
                    sys_prompt=data.get("sys_prompt", ""),
                    max_iters=int(data.get("max_iters", 10)),
                    tools_allow=list(data.get("tools_allow", [])),
                    tools_deny=list(data.get("tools_deny", [])),
                    knowledge_enabled=bool(data.get("knowledge_enabled", True)),
                    persona_type=data.get("persona_type", "WARM_COMPANION"),
                    personality_dimensions=dict(data.get("personality_dimensions", {})),
                    soul_enabled=bool(data.get("soul_enabled", True)),
                    reme_enabled=bool(data.get("reme_enabled", True)),
                )
                self._agents[cfg.device_sn] = cfg

        # tools
        tools_dir = self.config_dir / "tools"
        if tools_dir.is_dir():
            for path in sorted(tools_dir.glob("*.yaml")):
                data = _load_yaml(path)
                for entry in data.get("tools", []):
                    self._tools.append(
                        ToolConfig(
                            name=entry["name"],
                            type=entry.get("type", "BUILTIN"),
                            config=dict(entry.get("config", {})),
                        )
                    )

        # intent
        intent_path = self.config_dir / "intent.yaml"
        if intent_path.exists():
            data = _load_yaml(intent_path)
            self._intent = IntentConfig(
                enabled=bool(data.get("enabled", True)),
                rules=list(data.get("rules", [])),
            )

        # personality
        personality_dir = self.config_dir / "personality"
        strategies: dict[str, Any] = {}
        prompt_rules: dict[str, str] = {}
        if personality_dir.is_dir():
            strategies_path = personality_dir / "strategies.yaml"
            if strategies_path.exists():
                strategies = _load_yaml(strategies_path) or {}
            rules_path = personality_dir / "prompt_rules.yaml"
            if rules_path.exists():
                rules_data = _load_yaml(rules_path) or {}
                for rule in rules_data.get("rules", []):
                    key = f"{rule.get('dimension')}:{rule.get('band')}"
                    prompt_rules[key] = rule.get("prompt_text", "")
        self._personality = PersonalityConfig(
            strategies=strategies, prompt_rules=prompt_rules
        )

    # ---------- accessors ----------

    def get_model(self, model_id: int) -> ModelConfig:
        if model_id not in self._models:
            raise KeyError(f"model_id={model_id} not found in config/models/")
        return self._models[model_id]

    def get_agent(self, device_sn: str) -> AgentConfig:
        if device_sn not in self._agents:
            raise KeyError(f"device_sn={device_sn} not found in config/agents/")
        return self._agents[device_sn]

    def list_agents(self) -> list[str]:
        return list(self._agents)

    def list_tools(self) -> list[ToolConfig]:
        return list(self._tools)

    @property
    def intent(self) -> IntentConfig:
        return self._intent

    @property
    def personality(self) -> PersonalityConfig:
        return self._personality
