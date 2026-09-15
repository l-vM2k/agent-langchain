"""Soul orchestrator — mirrors SoulOrchestrator + SoulTurnPreparer + SoulTurnFinalizer.

prepare():  load personality + state -> perceive emotion -> generate behavior
            signal -> recall memories -> compose prompt -> assemble SoulContext.
finalize(): state machine transition + memory extraction/ingest + persist.
"""

from __future__ import annotations

from datetime import datetime

from xiaowei.soul.behavior import BehaviorSignalGenerator
from xiaowei.soul.emotion import EmotionAnalyzer
from xiaowei.soul.memory_selector import (
    RuleBasedSoulMemoryExtractor,
    SoulMemorySelector,
)
from xiaowei.soul.memory_store import SoulMemoryStore
from xiaowei.soul.models import (
    ContactProfile,
    PersonalityProfile,
    RelationshipTone,
    SoulContext,
    SoulPerception,
    SoulState,
    SoulTurnTrace,
)
from xiaowei.soul.prompt_composer import SoulPromptComposer
from xiaowei.soul.state_machine import SoulStateMachine
from xiaowei.soul.state_store import SoulStateStore


class SoulTurnPreparer:
    """本轮数据准备器 — SoulTurnPreparer.java"""

    def __init__(
        self,
        state_store: SoulStateStore,
        emotion_analyzer: EmotionAnalyzer,
        behavior_generator: BehaviorSignalGenerator,
        memory_store: SoulMemoryStore,
        memory_selector: SoulMemorySelector,
        prompt_composer: SoulPromptComposer,
    ) -> None:
        self._state_store = state_store
        self._emotion = emotion_analyzer
        self._behavior = behavior_generator
        self._memory_store = memory_store
        self._memory_selector = memory_selector
        self._prompt_composer = prompt_composer

    def prepare(
        self,
        personality: PersonalityProfile,
        device_sn: str,
        contact_id: str,
        session_id: str,
        user_text: str,
        contact: ContactProfile | None = None,
    ) -> SoulContext:
        """准备本轮灵魂上下文 — SoulTurnPreparer.prepare

        1. loadSoulState(无则 defaults 初始化)
        2. emotionPerceiver.perceive(user_text) → SoulPerception
        3. behaviorPlanner.planExpression(加权计算 6 维信号)
        4. 记忆召回(ACTIVE 候选 + SoulMemorySelector 门槛选择)
        5. promptComposer.compose(拼装人格/情绪/记忆段)
        6. 返回 SoulContext
        """
        # 1. 灵魂状态(懒初始化)
        state = self._state_store.get(device_sn, contact_id)
        if state is None:
            state = SoulState.defaults(device_sn, contact_id, session_id)

        # 2. 情绪感知
        perception: SoulPerception = self._emotion.analyze(user_text)

        # 3. 行为信号
        signal = self._behavior.generate(personality, state, perception.emotion_type)

        # 4. 记忆召回
        candidates = self._memory_store.list_active(device_sn, contact_id)
        selected = self._memory_selector.select(candidates, user_text)

        # 5. Prompt 组装
        prompt = self._prompt_composer.compose(
            config=personality,
            state=state,
            behavior_signal=signal,
            relationship_tone=signal.relationship_tone,
            contact=contact,
            memories=selected,
        )

        return SoulContext(
            device_sn=device_sn,
            contact_id=contact_id,
            session_id=session_id,
            personality_config=personality,
            soul_state=state,
            perception=perception,
            behavior_signal=signal,
            relationship_tone=signal.relationship_tone,
            care_success_score=0.5,
            selected_memories=selected,
            prompt=prompt,
            enabled=True,
            contact_profile=contact,
        )


class SoulTurnFinalizer:
    """本轮收尾器 — SoulTurnFinalizer.java(简化版: 状态机 + 记忆摄取)"""

    def __init__(
        self,
        state_store: SoulStateStore,
        state_machine: SoulStateMachine,
        memory_store: SoulMemoryStore,
        memory_extractor: RuleBasedSoulMemoryExtractor,
    ) -> None:
        self._state_store = state_store
        self._state_machine = state_machine
        self._memory_store = memory_store
        self._memory_extractor = memory_extractor

    def finalize(
        self,
        context: SoulContext,
        user_text: str,
        assistant_text: str,
        trace: SoulTurnTrace,
    ) -> None:
        """收尾本轮 — SoulTurnFinalizer.finalize

        1. 状态机转移(next) + soul_state upsert
        2. 记忆抽取(extract) + 摄取(ingest)
        失败静默降级,不阻塞对话主流程。
        """
        if not context.enabled:
            return
        try:
            # 1. 状态机转移
            next_state = self._state_machine.next(
                current=context.soul_state,
                perception=context.perception,
                trace=trace,
                now=datetime.now(),
                persona_type=context.personality_config.persona_type,
            )
            self._state_store.put(next_state)

            # 2. 记忆抽取(从本轮用户文本)
            extracted = self._memory_extractor.extract(
                context.device_sn, context.contact_id, user_text
            )
            if extracted:
                self._memory_store.add_all(extracted)
        except Exception:  # noqa: BLE001 — 收尾失败不阻塞对话
            pass


class SoulOrchestrator:
    """灵魂链路编排器 — SoulOrchestrator.java

    对话主链路统一入口,编排 Preparer(准备) + Finalizer(收尾)。
    """

    def __init__(self, preparer: SoulTurnPreparer, finalizer: SoulTurnFinalizer) -> None:
        self.preparer = preparer
        self.finalizer = finalizer

    def prepare(
        self,
        personality: PersonalityProfile,
        device_sn: str,
        contact_id: str,
        session_id: str,
        user_text: str,
        contact: ContactProfile | None = None,
    ) -> SoulContext:
        """准备本轮灵魂上下文(链A 入口)。"""
        return self.preparer.prepare(
            personality, device_sn, contact_id, session_id, user_text, contact
        )

    def finalize(
        self,
        context: SoulContext,
        user_text: str,
        assistant_text: str,
        trace: SoulTurnTrace,
    ) -> None:
        """收尾本轮(onTurnEnd 副作用编排)。"""
        self.finalizer.finalize(context, user_text, assistant_text, trace)
