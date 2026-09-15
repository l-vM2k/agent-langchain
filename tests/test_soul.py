"""Tests for the soul pipeline: emotion, behavior, state machine, prompt, memory."""

from __future__ import annotations

from datetime import datetime, timedelta

from ayaka.soul.behavior import BehaviorSignalGenerator
from ayaka.soul.emotion import EmotionAnalyzer
from ayaka.soul.memory_selector import (
    RuleBasedSoulMemoryExtractor,
    SoulMemorySelector,
)
from ayaka.soul.memory_store import SoulMemoryStore
from ayaka.soul.models import (
    EmotionType,
    PersonalityDimensions,
    PersonalityProfile,
    RelationshipTone,
    SoulMemory,
    SoulState,
)
from ayaka.soul.prompt_composer import SoulPromptComposer
from ayaka.soul.state_machine import SoulStateMachine
from ayaka.soul.state_store import SoulStateStore


# ---------- EmotionAnalyzer ----------

def test_emotion_neutral():
    analyzer = EmotionAnalyzer()
    p = analyzer.analyze("今天天气不错")
    # "不错" 前是"天"非否定 → POSITIVE 命中
    assert p.emotion_type == EmotionType.POSITIVE, f"got {p.emotion_type}"


def test_emotion_negation():
    analyzer = EmotionAnalyzer()
    # "不生气" → "生气"前 1 字为"不" → 不命中 ANGRY
    p = analyzer.analyze("我不生气")
    assert p.emotion_type != EmotionType.ANGRY, f"negation failed: {p.emotion_type}"


def test_emotion_priority():
    analyzer = EmotionAnalyzer()
    # ANGRY 优先级最高
    p = analyzer.analyze("我很生气也很焦虑")
    assert p.emotion_type == EmotionType.ANGRY


def test_emotion_sad_confiding():
    analyzer = EmotionAnalyzer()
    p = analyzer.analyze("我最近很难过,一个人待着,心里空空的,感觉撑不住了")
    assert p.emotion_type == EmotionType.SAD
    assert p.confiding is True
    assert p.care_needed is True


def test_emotion_deltas():
    analyzer = EmotionAnalyzer()
    p = analyzer.analyze("谢谢您")
    assert p.emotion_type == EmotionType.GRATEFUL
    assert abs(p.valence_delta - 0.24) < 1e-9
    assert abs(p.trust_delta - 0.08) < 1e-9


# ---------- BehaviorSignalGenerator ----------

def _behavior_gen() -> BehaviorSignalGenerator:
    from ayaka.soul.models import PersonaType
    weights = {
        PersonaType.WARM_COMPANION.value: {
            "warmth": {"empathy": 0.30, "attachment": 0.25, "trust": 0.20, "driveConnection": 0.25},
            "initiative": {"dominance": 0.35, "driveConnection": 0.25, "boredom": 0.15, "inverseDriveSafety": 0.25},
            "curiosity": {"curiosity": 0.60, "driveNovelty": 0.40},
            "restraint": {"driveSafety": 0.45, "formality": 0.10, "frustration": 0.35, "inverseTrust": 0.10},
            "playfulness": {"humor": 0.25, "drivePlay": 0.45, "valenceNormalized": 0.30},
            "actionDesire": {"driveExpression": 0.50, "arousalNormalized": 0.25, "playfulness": 0.20, "driveSafety": 0.10},
        }
    }
    return BehaviorSignalGenerator(weights)


def test_behavior_signal_ranges():
    gen = _behavior_gen()
    profile = PersonalityProfile()
    state = SoulState.defaults("d1", "c1", "s1")
    signal = gen.generate(profile, state, EmotionType.NEUTRAL)
    for field in ("warmth", "initiative", "curiosity", "restraint", "playfulness", "action_desire"):
        value = getattr(signal, field)
        assert 0.0 <= value <= 1.0, f"{field} out of range: {value}"


def test_behavior_angry_reduces_initiative():
    gen = _behavior_gen()
    profile = PersonalityProfile()
    state = SoulState.defaults("d1", "c1", "s1")
    neutral = gen.generate(profile, state, EmotionType.NEUTRAL)
    angry = gen.generate(profile, state, EmotionType.ANGRY)
    assert angry.initiative <= neutral.initiative
    assert angry.restraint >= neutral.restraint


def test_behavior_relationship_tone():
    gen = _behavior_gen()
    profile = PersonalityProfile()
    # turn_count >= 3 → FAMILIAR
    state = SoulState.defaults("d1", "c1", "s1")
    state.turn_count = 5
    signal = gen.generate(profile, state, EmotionType.NEUTRAL)
    assert signal.relationship_tone == RelationshipTone.FAMILIAR


# ---------- SoulStateMachine ----------

def test_state_machine_turn_increment():
    sm = SoulStateMachine()
    state = SoulState.defaults("d1", "c1", "s1")
    from ayaka.soul.models import SoulPerception, SoulTurnTrace

    nxt = sm.next(state, SoulPerception.neutral(), SoulTurnTrace.empty(), datetime.now())
    assert nxt.turn_count == state.turn_count + 1


def test_state_machine_tool_success_boosts_trust():
    sm = SoulStateMachine()
    state = SoulState.defaults("d1", "c1", "s1")
    from ayaka.soul.models import SoulPerception, SoulTurnTrace

    trace = SoulTurnTrace(tool_call_count=1, tool_success_count=1)
    nxt = sm.next(state, SoulPerception.neutral(), trace, datetime.now())
    # trust = 0.5 + 0.03(工具成功) ≈ 0.53
    assert nxt.trust > state.trust


def test_state_machine_decay():
    sm = SoulStateMachine()
    from ayaka.soul.models import SoulPerception, SoulTurnTrace

    state = SoulState.defaults("d1", "c1", "s1")
    state.valence = 0.9
    # 240 分钟前的交互 → 衰减率显著
    state.last_interaction_at = datetime.now() - timedelta(minutes=240)
    nxt = sm.next(state, SoulPerception.neutral(), SoulTurnTrace.empty(), datetime.now())
    # valence 衰减回 0.15 方向
    assert nxt.valence < 0.9


# ---------- SoulPromptComposer ----------

def test_prompt_composer_sections():
    composer = SoulPromptComposer()
    profile = PersonalityProfile()
    state = SoulState.defaults("d1", "c1", "s1")
    from ayaka.soul.models import BehaviorSignal

    signal = BehaviorSignal.neutral()
    prompt = composer.compose(profile, state, signal, signal.relationship_tone)
    assert "本轮表达倾向" in prompt
    assert "此刻情绪底色" in prompt
    assert "行为要求" in prompt
    assert len(prompt) < 6000


def test_prompt_composer_memory_section():
    composer = SoulPromptComposer()
    profile = PersonalityProfile()
    state = SoulState.defaults("d1", "c1", "s1")
    from ayaka.soul.models import BehaviorSignal

    signal = BehaviorSignal.neutral()
    memories = [
        SoulMemory(device_sn="d1", contact_id="c1", memory_type="PREFERENCE",
                   content="用户喜欢听老歌", confidence=0.8, importance=0.7),
    ]
    prompt = composer.compose(profile, state, signal, signal.relationship_tone,
                               memories=memories)
    assert "【长期记忆】" in prompt
    assert "用户喜欢听老歌" in prompt


# ---------- Memory selector / extractor ----------

def test_memory_selector_thresholds():
    selector = SoulMemorySelector()
    now = datetime.now()
    good = SoulMemory(content="喜欢听老歌", confidence=0.8, importance=0.7)
    low_conf = SoulMemory(content="喜欢听老歌", confidence=0.3, importance=0.7)
    expired = SoulMemory(content="喜欢听老歌", confidence=0.8, importance=0.7,
                         expire_at=now - timedelta(days=1))
    selected = selector.select([good, low_conf, expired], "")
    assert len(selected) == 1
    assert selected[0] is good or selected[0].content == good.content


def test_memory_extractor_types():
    extractor = RuleBasedSoulMemoryExtractor()
    memories = extractor.extract("d1", "c1", "我妈妈叫王芳,我喜欢听老歌")
    types = {m.memory_type for m in memories}
    assert "FAMILY_MEMBER" in types
    assert "PREFERENCE" in types


def test_memory_store_roundtrip(tmp_path):
    store = SoulMemoryStore(tmp_path / "mem.json")
    mem = SoulMemory(device_sn="d1", contact_id="c1", memory_type="PREFERENCE",
                     content="喜欢听老歌", confidence=0.8, importance=0.7)
    store.add(mem)
    loaded = store.list_active("d1", "c1")
    assert len(loaded) == 1
    assert loaded[0].content == "喜欢听老歌"


def test_state_store_roundtrip(tmp_path):
    store = SoulStateStore(tmp_path / "state.json")
    state = SoulState.defaults("d1", "c1", "s1")
    state.trust = 0.77
    store.put(state)
    loaded = store.get("d1", "c1")
    assert loaded is not None
    assert abs(loaded.trust - 0.77) < 1e-9
