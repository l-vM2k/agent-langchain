"""Soul package — personality, emotion, state machine, memory, prompt composition.

Mirrors the Java `org.ruoyi.agent.soul` package (the most valuable part of the
original system): a personality layer that runs before/after each chat turn.
"""

from xiaowei.soul.models import (
    BehaviorSignal,
    ContactProfile,
    EmotionType,
    PersonalityDimensions,
    PersonalityProfile,
    RelationshipTone,
    SoulContext,
    SoulMemory,
    SoulPerception,
    SoulState,
    SoulTurnTrace,
)
from xiaowei.soul.orchestrator import SoulOrchestrator

__all__ = [
    "BehaviorSignal",
    "ContactProfile",
    "EmotionType",
    "PersonalityDimensions",
    "PersonalityProfile",
    "RelationshipTone",
    "SoulContext",
    "SoulMemory",
    "SoulPerception",
    "SoulState",
    "SoulTurnTrace",
    "SoulOrchestrator",
]
