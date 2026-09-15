"""Agent lifecycle manager — lazy-load per device_sn."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from ayaka.config import AgentConfig, ConfigLoader
from ayaka.llm import create_llm
from ayaka.memory.reme_store import ReMeMemoryStore
from ayaka.soul.models import PersonalityDimensions, PersonalityProfile, PersonaType
from ayaka.soul.orchestrator import (
    SoulOrchestrator,
    SoulTurnFinalizer,
    SoulTurnPreparer,
)
from ayaka.soul.behavior import BehaviorSignalGenerator
from ayaka.soul.emotion import EmotionAnalyzer
from ayaka.soul.memory_selector import (
    RuleBasedSoulMemoryExtractor,
    SoulMemorySelector,
)
from ayaka.soul.memory_store import SoulMemoryStore
from ayaka.soul.prompt_composer import SoulPromptComposer
from ayaka.soul.state_machine import (
    DEFAULT_STATE_STRATEGIES,
    SoulStateMachine,
)
from ayaka.soul.state_store import SoulStateStore
from ayaka.tools.factory import ToolFactory

logger = logging.getLogger(__name__)


class AgentManager:
    """Registry of compiled agent graphs, keyed by device_sn.

    Lazy creation, one instance per device. Each agent bundles its own soul
    pipeline (state store / memory store / ReMe workspace) so contacts on the
    same device stay isolated.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        data_dir: str | Path = ".data",
    ) -> None:
        self._config = config_loader
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, CompiledStateGraph] = {}

    def get_agent(self, device_sn: str) -> CompiledStateGraph:
        """Return the compiled graph for device_sn, building it on first use."""
        if device_sn not in self._registry:
            self._registry[device_sn] = self._build(device_sn)
        return self._registry[device_sn]

    def _build(self, device_sn: str) -> CompiledStateGraph:
        """构建一个设备的完整 Agent(图 + 灵魂链 + 记忆 + 工具)。"""
        from ayaka.agent.graph import build_agent_graph

        agent_cfg: AgentConfig = self._config.get_agent(device_sn)
        model_cfg = self._config.get_model(agent_cfg.model_id)
        llm: BaseChatModel = create_llm(model_cfg)

        # ---- 工具 ----
        tool_factory = ToolFactory(self._config)
        tools: list[BaseTool] = tool_factory.build_sync_tools(agent_cfg)
        # MCP 工具是异步的,在同步构建路径里跳过(启动时预取会阻塞);
        # 需要时可在 server 启动钩子里 await build_mcp_tools 后重建 agent。

        # ---- 灵魂链 ----
        orchestrator = self._build_orchestrator(agent_cfg, device_sn)

        # ---- ReMe ----
        reme_store = ReMeMemoryStore(self._data_dir / "reme")

        # ---- 人格 ----
        personality = self._build_personality(agent_cfg, device_sn)

        graph = build_agent_graph(
            llm=llm,
            tools=tools,
            sys_prompt=agent_cfg.sys_prompt,
            orchestrator=orchestrator,
            reme_store=reme_store,
            personality=personality,
        )
        logger.info(
            "Agent 构建完成: device=%s model=%s tools=%d soul=%s",
            device_sn, model_cfg.model_name, len(tools), agent_cfg.soul_enabled,
        )
        return graph

    def _build_personality(self, agent_cfg: AgentConfig, device_sn: str) -> PersonalityProfile:
        """从 agent 配置解析人格(含种子复制)。"""
        overrides = agent_cfg.personality_dimensions or {}
        dimensions = PersonalityDimensions(**{
            k: v for k, v in overrides.items() if k in PersonalityDimensions.model_fields
        }) if overrides else PersonalityDimensions()
        return PersonalityProfile(
            device_sn=device_sn,
            enabled=agent_cfg.soul_enabled,
            persona_type=PersonaType(agent_cfg.persona_type),
            dimensions=dimensions,
            sys_prompt=agent_cfg.sys_prompt,
        )

    def _build_orchestrator(self, agent_cfg: AgentConfig, device_sn: str) -> SoulOrchestrator:
        """组装灵魂链(Preparer + Finalizer)。"""
        personality_cfg = self._config.personality

        # 行为信号权重(按 personaType 从 strategies.yaml 读)
        strategies = personality_cfg.strategies or {}
        behavior_weights = {
            persona: data.get("behavior_weights", {})
            for persona, data in strategies.items()
        }
        behavior_generator = BehaviorSignalGenerator(behavior_weights)
        # Prompt 档位文案
        prompt_composer = SoulPromptComposer(personality_cfg.prompt_rules)

        # 状态机(按 personaType 的 state 参数)
        state_strategies = {}
        for persona, data in strategies.items():
            state = data.get("state", {})
            state_strategies[persona] = DEFAULT_STATE_STRATEGIES[persona].__class__(
                **state
            ) if state else DEFAULT_STATE_STRATEGIES[persona]
        state_machine = SoulStateMachine(state_strategies)

        # 存储
        state_store = SoulStateStore(self._data_dir / "soul_state.json")
        memory_store = SoulMemoryStore(self._data_dir / "soul_memory.json")

        preparer = SoulTurnPreparer(
            state_store=state_store,
            emotion_analyzer=EmotionAnalyzer(),
            behavior_generator=behavior_generator,
            memory_store=memory_store,
            memory_selector=SoulMemorySelector(),
            prompt_composer=prompt_composer,
        )
        finalizer = SoulTurnFinalizer(
            state_store=state_store,
            state_machine=state_machine,
            memory_store=memory_store,
            memory_extractor=RuleBasedSoulMemoryExtractor(),
        )
        return SoulOrchestrator(preparer, finalizer)

    def drop(self, device_sn: str) -> None:
        """Forget a cached agent so the next get_agent rebuilds it."""
        self._registry.pop(device_sn, None)
