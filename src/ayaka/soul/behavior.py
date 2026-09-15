"""Behavior signal generation.

Turns (personality dimensions + soul state + emotion) into 6 behavior signals
via per-persona weight matrices. Weights come from the strategy config
(config/personality/strategies.yaml).
"""

from __future__ import annotations

from ayaka.soul.models import (
    BehaviorSignal,
    EmotionType,
    PersonalityProfile,
    RelationshipTone,
    SoulState,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize_signed(value: float) -> float:
    """-1~1 → 0~1"""
    return _clamp((value + 1) / 2)


class BehaviorSignalGenerator:
    """将静态人格、内驱力和动态灵魂状态转换为本轮行为倾向。

    权重矩阵从 strategies 配置读取(按 personaType 切换)。
    """

    def __init__(self, strategies: dict[str, "PersonaStrategies | dict"]) -> None:
        """strategies: {persona_type: behavior_weights dict},来自 config/personality/strategies.yaml"""
        self._strategies = strategies

    def generate(
        self,
        profile: PersonalityProfile,
        state: SoulState,
        emotion: EmotionType,
    ) -> BehaviorSignal:
        """生成 6 维行为信号"""
        personality = profile or PersonalityProfile()
        soul = state or SoulState.defaults(personality.device_sn, "", "")
        user_emotion = emotion or EmotionType.NEUTRAL

        behavior = self._behavior_weights(personality.persona_type)
        dims = personality.dimensions

        # warmth = empathy*0.30 + attachment*0.25 + trust*0.20 + driveConnection*0.25
        warmth = self._weighted(
            (dims.empathy, behavior["warmth"].get("empathy", 0.0)),
            (soul.attachment, behavior["warmth"].get("attachment", 0.0)),
            (soul.trust, behavior["warmth"].get("trust", 0.0)),
            (dims.drive_connection, behavior["warmth"].get("driveConnection", 0.0)),
        )

        # initiative = dominance*0.35 + driveConnection*0.25 + boredom*0.15 + (1-driveSafety)*0.25
        initiative = self._weighted(
            (dims.dominance, behavior["initiative"].get("dominance", 0.0)),
            (dims.drive_connection, behavior["initiative"].get("driveConnection", 0.0)),
            (soul.boredom, behavior["initiative"].get("boredom", 0.0)),
            (1 - dims.drive_safety, behavior["initiative"].get("inverseDriveSafety", 0.0)),
        )
        if user_emotion in (EmotionType.ANGRY, EmotionType.NEGATIVE_FEEDBACK):
            initiative -= 0.20

        # curiosity = curiosity*0.60 + driveNovelty*0.40
        curiosity = self._weighted(
            (dims.curiosity, behavior["curiosity"].get("curiosity", 0.0)),
            (dims.drive_novelty, behavior["curiosity"].get("driveNovelty", 0.0)),
        )

        # restraint = driveSafety*0.45 + formality*0.10 + frustration*0.35 + (1-trust)*0.10
        restraint = self._weighted(
            (dims.drive_safety, behavior["restraint"].get("driveSafety", 0.0)),
            (dims.formality, behavior["restraint"].get("formality", 0.0)),
            (soul.frustration, behavior["restraint"].get("frustration", 0.0)),
            (1 - soul.trust, behavior["restraint"].get("inverseTrust", 0.0)),
        )
        if user_emotion in (EmotionType.ANGRY, EmotionType.NEGATIVE_FEEDBACK):
            restraint += 0.08

        # playfulness = humor*0.25 + drivePlay*0.45 + normalize(valence)*0.30
        playfulness = self._weighted(
            (dims.humor, behavior["playfulness"].get("humor", 0.0)),
            (dims.drive_play, behavior["playfulness"].get("drivePlay", 0.0)),
            (_normalize_signed(soul.valence), behavior["playfulness"].get("valenceNormalized", 0.0)),
        )
        if user_emotion in (EmotionType.SAD, EmotionType.ANXIOUS,
                            EmotionType.ANGRY, EmotionType.NEGATIVE_FEEDBACK):
            playfulness -= 0.25

        # actionDesire = driveExpression*0.50 + normalize(arousal)*0.25
        #               + playfulness*0.20 + driveSafety*0.10
        action_desire = self._weighted(
            (dims.drive_expression, behavior["actionDesire"].get("driveExpression", 0.0)),
            (_normalize_signed(soul.arousal), behavior["actionDesire"].get("arousalNormalized", 0.0)),
            (playfulness, behavior["actionDesire"].get("playfulness", 0.0)),
            (dims.drive_safety, behavior["actionDesire"].get("driveSafety", 0.0)),
        )
        if restraint > 0.85 or soul.frustration > 0.70:
            action_desire -= 0.20

        return BehaviorSignal(
            warmth=_clamp(warmth),
            initiative=_clamp(initiative),
            curiosity=_clamp(curiosity),
            restraint=_clamp(restraint),
            playfulness=_clamp(playfulness),
            action_desire=_clamp(action_desire),
            relationship_tone=self._relationship_tone(soul),
        )

    # ---------- 关系档位推断 ----------

    def _relationship_tone(self, state: SoulState) -> RelationshipTone:
        """由动态 SoulState 推断关系档位。

        signal 推断上限 INTIMATE;FAMILY_COMPANION 第 5 档仅由
        profile.relationshipLevel 在 assembler 层决定,这里推不出。
        """
        if state.trust >= 0.88 and state.attachment >= 0.82 and state.turn_count >= 30:
            return RelationshipTone.INTIMATE
        if state.trust >= 0.78 and state.attachment >= 0.68:
            return RelationshipTone.TRUSTED
        if state.turn_count >= 3 or state.attachment >= 0.35 or state.trust >= 0.65:
            return RelationshipTone.FAMILIAR
        return RelationshipTone.STRANGER

    # ---------- 工具 ----------

    def _behavior_weights(self, persona_type) -> dict[str, dict[str, float]]:
        """按 personaType 取行为权重矩阵(缺失兜底 WARM_COMPANION)。"""
        from ayaka.soul.models import PersonaType

        key = persona_type.value if hasattr(persona_type, "value") else str(persona_type)
        weights = self._strategies.get(key)
        if weights is None:
            weights = self._strategies.get(PersonaType.WARM_COMPANION.value, {})
        # 兼容两种传参: {persona: weights_dict} 或 {persona: {"behavior_weights": {...}}}
        if isinstance(weights, dict) and "behavior_weights" in weights:
            weights = weights["behavior_weights"]
        return weights or {}

    @staticmethod
    def _weighted(*pairs: tuple[float, float]) -> float:
        """加权求和"""
        return sum(value * weight for value, weight in pairs)
