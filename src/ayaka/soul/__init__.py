"""Soul package — personality, emotion, state machine, memory, prompt composition.

A personality layer that runs before/after each chat turn.
"""

from ayaka.soul.models import (
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
from ayaka.soul.orchestrator import SoulOrchestrator

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
