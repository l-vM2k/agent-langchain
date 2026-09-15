"""Soul data models — mirrors of the Java records/enums in org.ruoyi.agent.soul.core.

Field names and defaults are kept behavior-equivalent to the Java side so the
strategy tables (weights, deltas, thresholds) transfer 1:1.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PersonaType(str, Enum):
    """人格底座枚举 — PersonaType.java"""

    WARM_COMPANION = "WARM_COMPANION"
    RATIONAL_ASSISTANT = "RATIONAL_ASSISTANT"
    QUIET_GUARDIAN = "QUIET_GUARDIAN"


class EmotionType(str, Enum):
    """用户情绪类型 — EmotionType.java"""

    POSITIVE = "POSITIVE"
    SAD = "SAD"
    ANXIOUS = "ANXIOUS"
    ANGRY = "ANGRY"
    GRATEFUL = "GRATEFUL"
    NEGATIVE_FEEDBACK = "NEGATIVE_FEEDBACK"
    CONFIDING = "CONFIDING"
    NEUTRAL = "NEUTRAL"


class RelationshipTone(str, Enum):
    """关系档位,5 档按亲密度递增 — RelationshipTone.java"""

    STRANGER = "STRANGER"
    FAMILIAR = "FAMILIAR"
    TRUSTED = "TRUSTED"
    INTIMATE = "INTIMATE"
    FAMILY_COMPANION = "FAMILY_COMPANION"


class PersonalityDimensions(BaseModel):
    """人格 14 维度,取值 [0,1] — PersonalityDimensions.java

    Field order matches the Java record; defaults are the WARM_COMPANION
    warmth baseline (PersonalityDimensions.warmthDefault()).
    """

    empathy: float = 0.55
    dominance: float = 0.45
    formality: float = 0.30
    curiosity: float = 0.50
    relationship_distance: float = 0.45
    humor: float = 0.40
    drive_connection: float = 0.55
    drive_novelty: float = 0.50
    drive_expression: float = 0.50
    drive_safety: float = 0.70
    drive_play: float = 0.35
    baseline_valence: float = 0.15
    baseline_arousal: float = 0.0
    baseline_dominance: float = 0.0


class PersonalityProfile(BaseModel):
    """设备静态人格配置 — PersonalityProfile.java"""

    device_sn: str = ""
    enabled: bool = True
    persona_type: PersonaType = PersonaType.WARM_COMPANION
    dimensions: PersonalityDimensions = Field(default_factory=PersonalityDimensions)
    sys_prompt: str = ""


class SoulState(BaseModel):
    """会话维度动态灵魂状态 — SoulState.java

    PAD 三维 [-1,1];frustration/attachment/trust/boredom [0,1]。
    """

    device_sn: str = ""
    contact_id: str = ""
    session_id: str = ""
    valence: float = 0.15
    arousal: float = 0.0
    dominance: float = 0.0
    frustration: float = 0.0
    attachment: float = 0.1
    trust: float = 0.5
    boredom: float = 0.0
    turn_count: int = 0
    last_user_emotion: EmotionType = EmotionType.NEUTRAL
    last_summary: Optional[str] = None
    last_interaction_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def defaults(cls, device_sn: str, contact_id: str, session_id: str) -> "SoulState":
        """SoulState.defaults() — 初始状态工厂。"""
        return cls(
            device_sn=device_sn,
            contact_id=contact_id,
            session_id=session_id,
        )


class BehaviorSignal(BaseModel):
    """行为信号 6 维,[0,1] — BehaviorSignal.java"""

    warmth: float = 0.5
    initiative: float = 0.45
    curiosity: float = 0.5
    restraint: float = 0.6
    playfulness: float = 0.4
    action_desire: float = 0.35
    relationship_tone: RelationshipTone = RelationshipTone.STRANGER

    @classmethod
    def neutral(cls) -> "BehaviorSignal":
        """BehaviorSignal.neutral() — 中性信号。"""
        return cls()


class SoulPerception(BaseModel):
    """本轮感知结果 — SoulPerception.java"""

    emotion_type: EmotionType = EmotionType.NEUTRAL
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    dominance_delta: float = 0.0
    trust_delta: float = 0.0
    attachment_delta: float = 0.0
    frustration_delta: float = 0.0
    confiding: bool = False
    care_needed: bool = False
    restraint_hint: bool = False

    @classmethod
    def neutral(cls) -> "SoulPerception":
        """SoulPerception.neutral() — 中性感知。"""
        return cls()


class SoulTurnTrace(BaseModel):
    """本轮执行轨迹 — SoulTurnTrace.java(工具调用统计,供状态机消费)"""

    tool_call_count: int = 0
    tool_success_count: int = 0
    tool_failure_count: int = 0
    error_occurred: bool = False
    assistant_text_length: int = 0

    @classmethod
    def empty(cls) -> "SoulTurnTrace":
        return cls()


class SoulMemory(BaseModel):
    """一条灵魂长期记忆 — AgentSoulMemoryBo.java"""

    id: Optional[int] = None
    device_sn: str = ""
    contact_id: str = ""
    memory_type: str = "PREFERENCE"
    content: str = ""
    confidence: float = 0.5
    importance: float = 0.5
    privacy_level: str = "STANDARD"
    status: str = "ACTIVE"
    expire_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ContactProfile(BaseModel):
    """联系人档案 — AgentContactProfile.java(Task 11 prompt 注入用)"""

    contact_id: str = ""
    nick_name: str = ""
    hobbies: str = ""
    personality_desc: str = ""
    medical_history: str = ""
    background_desc: str = ""

    def is_empty(self) -> bool:
        return not any(
            [self.nick_name, self.hobbies, self.personality_desc,
             self.medical_history, self.background_desc]
        )


class SoulContext(BaseModel):
    """灵魂上下文 — 本轮对话的完整上下文 — SoulContext.java"""

    device_sn: str
    contact_id: str
    session_id: str
    personality_config: PersonalityProfile = Field(default_factory=PersonalityProfile)
    soul_state: SoulState = Field(default_factory=SoulState)
    perception: SoulPerception = Field(default_factory=SoulPerception)
    behavior_signal: BehaviorSignal = Field(default_factory=BehaviorSignal)
    relationship_tone: RelationshipTone = RelationshipTone.STRANGER
    care_success_score: float = 0.5
    selected_memories: list[SoulMemory] = Field(default_factory=list)
    prompt: str = ""
    enabled: bool = True
    contact_profile: Optional[ContactProfile] = None

    @classmethod
    def disabled(cls, device_sn: str, contact_id: str, session_id: str) -> "SoulContext":
        """SoulContext.disabled() — 灵魂链路未启用时的降级上下文。"""
        return cls(
            device_sn=device_sn,
            contact_id=contact_id,
            session_id=session_id,
            soul_state=SoulState.defaults(device_sn, contact_id, session_id),
            enabled=False,
        )
