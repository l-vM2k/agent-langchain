"""Soul state machine — mirrors SoulStateMachine.java.

Pure-function state transition: decay PAD toward baseline, apply perception
deltas, apply tool-success/failure trust adjustments. No side effects.
"""

from __future__ import annotations

from datetime import datetime

from xiaowei.soul.models import PersonaType, SoulPerception, SoulState, SoulTurnTrace


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class StateStrategyParams:
    """State 策略参数 — StateStrategy.java(从 strategies.yaml 加载)"""

    def __init__(
        self,
        decay_rate_min: float = 0.05,
        decay_rate_max: float = 0.35,
        decay_rate_slope: float = 0.30,
        decay_rate_minutes_divisor: float = 240.0,
        tool_success_trust_delta: float = 0.03,
        attachment_increment: float = 0.01,
    ) -> None:
        self.decay_rate_min = decay_rate_min
        self.decay_rate_max = decay_rate_max
        self.decay_rate_slope = decay_rate_slope
        self.decay_rate_minutes_divisor = decay_rate_minutes_divisor
        self.tool_success_trust_delta = tool_success_trust_delta
        self.attachment_increment = attachment_increment

    def decay_rate(self, minutes: float) -> float:
        """clamp(min + minutes/divisor * slope, min, max) — StateStrategy.decayRate"""
        if minutes <= 0:
            return self.decay_rate_min
        raw = self.decay_rate_min + (minutes / self.decay_rate_minutes_divisor) * self.decay_rate_slope
        return max(self.decay_rate_min, min(self.decay_rate_max, raw))


# 三底座默认参数 — WarmCompanion/RationalAssistant/QuietGuardian StateStrategy
DEFAULT_STATE_STRATEGIES: dict[str, StateStrategyParams] = {
    PersonaType.WARM_COMPANION.value: StateStrategyParams(
        decay_rate_slope=0.30, tool_success_trust_delta=0.03, attachment_increment=0.01,
    ),
    PersonaType.RATIONAL_ASSISTANT.value: StateStrategyParams(
        decay_rate_slope=0.40, tool_success_trust_delta=0.04, attachment_increment=0.005,
    ),
    PersonaType.QUIET_GUARDIAN.value: StateStrategyParams(
        decay_rate_slope=0.20, tool_success_trust_delta=0.02, attachment_increment=0.008,
    ),
}


class SoulStateMachine:
    """动态灵魂状态机 — SoulStateMachine.java

    衰减率/工具 trust delta/attachment 递增系数按设备 personaType 解析。
    """

    def __init__(self, strategies: dict[str, StateStrategyParams] | None = None) -> None:
        self._strategies = strategies or DEFAULT_STATE_STRATEGIES

    def next(
        self,
        current: SoulState,
        perception: SoulPerception,
        trace: SoulTurnTrace,
        now: datetime,
        persona_type: PersonaType = PersonaType.WARM_COMPANION,
    ) -> SoulState:
        """状态转移 — SoulStateMachine.next

        1. PAD 三维 + frustration: 先衰减回 baseline,再叠加本轮信号 delta
        2. attachment: 当前值 + 信号 delta + 每轮自然递增
        3. trust: 当前值 + 信号 delta + 工具成功/失败修正
        4. boredom: 短回复无工具时 +0.03
        """
        strategy = self._strategies.get(
            persona_type.value if hasattr(persona_type, "value") else str(persona_type),
            DEFAULT_STATE_STRATEGIES[PersonaType.WARM_COMPANION.value],
        )
        effective_now = now or datetime.now()
        signal = perception or SoulPerception.neutral()
        turn_trace = trace or SoulTurnTrace.empty()

        decay = self._decay_rate(current.last_interaction_at, effective_now, strategy)
        valence = self._decay_toward(current.valence, 0.15, decay) + signal.valence_delta
        arousal = self._decay_toward(current.arousal, 0.0, decay) + signal.arousal_delta
        # PAD dominance: 先衰减回 0,再叠加本轮情绪信号 delta
        dominance = self._decay_toward(current.dominance, 0.0, decay) + signal.dominance_delta
        # frustration 同样先衰减回 0 再叠加信号,避免工具失败 +0.15 后永远卡高点
        frustration = self._decay_toward(current.frustration, 0.0, decay) + signal.frustration_delta
        attachment = current.attachment + signal.attachment_delta + strategy.attachment_increment
        trust = current.trust + signal.trust_delta
        boredom = current.boredom

        if turn_trace.tool_success_count > 0:
            trust += strategy.tool_success_trust_delta * turn_trace.tool_success_count
            frustration -= 0.05 * turn_trace.tool_success_count
        if turn_trace.tool_failure_count > 0 or turn_trace.error_occurred:
            failures = max(1, turn_trace.tool_failure_count)
            trust -= 0.03 * failures
            frustration += 0.15 * failures
        if turn_trace.assistant_text_length < 8 and turn_trace.tool_call_count == 0:
            boredom += 0.03

        return SoulState(
            device_sn=current.device_sn,
            contact_id=current.contact_id,
            session_id=current.session_id,
            valence=_clamp(valence, -1, 1),
            arousal=_clamp(arousal, -1, 1),
            dominance=_clamp(dominance, -1, 1),
            frustration=_clamp(frustration, 0, 1),
            attachment=_clamp(attachment, 0, 1),
            trust=_clamp(trust, 0, 1),
            boredom=_clamp(boredom, 0, 1),
            turn_count=current.turn_count + 1,
            last_user_emotion=signal.emotion_type,
            last_summary=current.last_summary,
            last_interaction_at=effective_now,
        )

    # ---------- 内部 ----------

    def _decay_rate(self, previous: datetime, now: datetime, strategy: StateStrategyParams) -> float:
        """距上次交互的分钟数 → 衰减率 — SoulStateMachine.decayRate"""
        if previous is None or now is None or not now > previous:
            return strategy.decay_rate(0)
        minutes = (now - previous).total_seconds() / 60.0
        return strategy.decay_rate(minutes)

    @staticmethod
    def _decay_toward(value: float, baseline: float, rate: float) -> float:
        """向 baseline 衰减 — SoulStateMachine.decayToward"""
        return value + (baseline - value) * rate
