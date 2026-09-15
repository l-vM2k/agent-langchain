"""Emotion perception — mirrors EmotionAnalyzer.java + TextEmotionPerceiver.java.

Rule-based keyword matching with negation-prefix detection, producing a
SoulPerception with 6-dimension deltas. No LLM call — keeps latency low.
"""

from __future__ import annotations

from xiaowei.soul.models import EmotionType, SoulPerception

# 否定前缀 — EmotionAnalyzer.NEGATION_PREFIXES_1/2
NEGATION_PREFIXES_1 = {"不", "没", "别", "未"}
NEGATION_PREFIXES_2 = {"有没"}

# 情绪词表 — WarmCompanionStrategies.EMOTION_LEXICONS(三底座共用)
EMOTION_LEXICONS: dict[EmotionType, list[str]] = {
    EmotionType.GRATEFUL: ["谢谢", "感谢", "辛苦", "帮了我"],
    EmotionType.ANGRY: ["生气", "气死", "垃圾", "烦", "太差"],
    EmotionType.NEGATIVE_FEEDBACK: ["不对", "错了", "没用", "不行", "失败"],
    EmotionType.SAD: [
        "难过", "伤心", "委屈", "失落", "哭",
        # 疲惫倦怠型 / 空虚孤独型 / 自我否定型 / 受挫型 / 低落型(2026-07-20 扩充)
        "累", "疲惫", "撑不住", "心累", "心里空", "空虚", "空空的",
        "孤独", "一个人", "不够好", "受挫", "被批", "被骂",
        "没意思", "抑郁", "想哭", "绝望", "emo",
    ],
    EmotionType.ANXIOUS: [
        "焦虑", "担心", "害怕", "压力", "怎么办",
        # 焦虑躯体化型(避开含「烦」的词,会被 ANGRY 截胡)
        "失眠", "睡不着", "不安", "紧张", "心慌", "喘不过气", "崩溃",
    ],
    EmotionType.POSITIVE: ["开心", "喜欢", "太好了", "棒", "哈哈", "不错"],
}

# 6 维 deltas — WarmCompanionStrategies.EMOTION_DELTAS
# 字段顺序: valence, arousal, dominance, trust, attachment, frustration
EMOTION_DELTAS: dict[EmotionType, dict[str, float]] = {
    EmotionType.ANGRY: {"valence": -0.35, "arousal": 0.25, "dominance": -0.15,
                        "trust": -0.08, "attachment": 0.0, "frustration": 0.18},
    EmotionType.NEGATIVE_FEEDBACK: {"valence": -0.25, "arousal": 0.10,
                                    "dominance": -0.12, "trust": -0.06,
                                    "attachment": 0.0, "frustration": 0.14},
    EmotionType.GRATEFUL: {"valence": 0.24, "arousal": 0.05, "dominance": 0.10,
                           "trust": 0.08, "attachment": 0.03, "frustration": -0.03},
    EmotionType.SAD: {"valence": -0.32, "arousal": -0.08, "dominance": 0.0,
                      "trust": 0.0, "attachment": 0.06, "frustration": 0.02},
    EmotionType.ANXIOUS: {"valence": -0.24, "arousal": 0.20, "dominance": 0.0,
                          "trust": 0.0, "attachment": 0.04, "frustration": 0.05},
    EmotionType.POSITIVE: {"valence": 0.26, "arousal": 0.12, "dominance": 0.08,
                           "trust": 0.03, "attachment": 0.02, "frustration": -0.02},
    EmotionType.CONFIDING: {"valence": -0.08, "arousal": -0.02, "dominance": 0.0,
                            "trust": 0.01, "attachment": 0.05, "frustration": 0.0},
    EmotionType.NEUTRAL: {},
}

# 命中优先级 — EmotionAnalyzer.detect()
DETECT_ORDER: list[EmotionType] = [
    EmotionType.ANGRY,
    EmotionType.NEGATIVE_FEEDBACK,
    EmotionType.GRATEFUL,
    EmotionType.SAD,
    EmotionType.ANXIOUS,
    EmotionType.POSITIVE,
]

# CONFIDING 判定: 长文本(>=36 字)且含 SAD/ANXIOUS 词
LONG_CONFIDING_MIN_LEN = 36


class EmotionAnalyzer:
    """轻量规则情绪识别器 — EmotionAnalyzer.java

    关键词命中 + 否定前缀检测,不调 LLM。
    """

    def analyze(self, text: str | None) -> SoulPerception:
        """分析文本情绪,返回感知结果(含 6 维 deltas + care/restraint 派生)。"""
        normalized = (text or "").strip()
        if not normalized:
            return SoulPerception.neutral()

        emotion, confiding = self._detect(normalized)
        return self._to_perception(emotion, confiding)

    def signal_of_type(self, emotion: EmotionType) -> tuple[EmotionType, dict[str, float], bool]:
        """按已知情绪类型构造 (type, deltas, confiding) — EmotionAnalyzer.signalOfType"""
        if emotion is None or emotion == EmotionType.NEUTRAL:
            return EmotionType.NEUTRAL, {}, False
        confiding = emotion in (EmotionType.SAD, EmotionType.CONFIDING)
        return emotion, EMOTION_DELTAS.get(emotion, {}), confiding

    # ---------- 内部 ----------

    def _detect(self, text: str) -> tuple[EmotionType, bool]:
        """按优先级检测情绪 — EmotionAnalyzer.detect()

        优先级: ANGRY > NEGATIVE_FEEDBACK > GRATEFUL > SAD > ANXIOUS > POSITIVE
                 > CONFIDING(long) > NEUTRAL
        """
        long_confiding = (
            len(text) >= LONG_CONFIDING_MIN_LEN
            and (self._contains_any(text, EMOTION_LEXICONS[EmotionType.SAD])
                 or self._contains_any(text, EMOTION_LEXICONS[EmotionType.ANXIOUS]))
        )

        for emotion in DETECT_ORDER:
            if self._contains_any(text, EMOTION_LEXICONS[emotion]):
                # confiding 透传规则(对齐 Java detect):
                #   SAD → true(固定)
                #   ANGRY / NEGATIVE_FEEDBACK / ANXIOUS → longConfiding
                #   GRATEFUL / POSITIVE → false
                if emotion == EmotionType.SAD:
                    confiding = True
                elif emotion in (EmotionType.ANGRY, EmotionType.NEGATIVE_FEEDBACK,
                                 EmotionType.ANXIOUS):
                    confiding = long_confiding
                else:
                    confiding = False
                return emotion, confiding

        if long_confiding:
            return EmotionType.CONFIDING, True
        return EmotionType.NEUTRAL, False

    def _to_perception(self, emotion: EmotionType, confiding: bool) -> SoulPerception:
        """UserEmotionSignal → SoulPerception — TextEmotionPerceiver.toPerception()

        care_needed = SAD/ANXIOUS/CONFIDING(关怀类)
        restraint_hint = ANGRY/NEGATIVE_FEEDBACK(克制类)
        """
        deltas = EMOTION_DELTAS.get(emotion, {})
        care_needed = emotion in (EmotionType.SAD, EmotionType.ANXIOUS, EmotionType.CONFIDING)
        restraint_hint = emotion in (EmotionType.ANGRY, EmotionType.NEGATIVE_FEEDBACK)
        return SoulPerception(
            emotion_type=emotion,
            valence_delta=deltas.get("valence", 0.0),
            arousal_delta=deltas.get("arousal", 0.0),
            dominance_delta=deltas.get("dominance", 0.0),
            trust_delta=deltas.get("trust", 0.0),
            attachment_delta=deltas.get("attachment", 0.0),
            frustration_delta=deltas.get("frustration", 0.0),
            confiding=confiding,
            care_needed=care_needed,
            restraint_hint=restraint_hint,
        )

    def _contains_any(self, text: str, words: list[str]) -> bool:
        """是否命中任一关键词(含否定前缀检测) — EmotionAnalyzer.containsAny"""
        for word in words:
            if self._has_affirmative_occurrence(text, word):
                return True
        return False

    def _has_affirmative_occurrence(self, text: str, word: str) -> bool:
        """检查 word 在 text 中是否存在「未被否定」的出现位置。

        "我不生气"中"生气"前 1 字为"不" → 不命中;
        "感觉不错"中"不错"前 1 字"觉"非否定 → 命中。
        """
        if not word:
            return False
        from_index = 0
        while True:
            idx = text.find(word, from_index)
            if idx < 0:
                return False
            if not self._is_negated(text, idx):
                return True
            from_index = idx + 1  # 该位置被否定,继续找下一次出现

    def _is_negated(self, text: str, idx: int) -> bool:
        """判断 keyword 在 idx 位置是否被紧邻前缀否定 — EmotionAnalyzer.isNegated"""
        if idx >= 1:
            prev1 = text[idx - 1 : idx]
            if prev1 in NEGATION_PREFIXES_1:
                return True
            if idx >= 2:
                prev2 = text[idx - 2 : idx]
                if prev2 in NEGATION_PREFIXES_2:
                    return True
        return False
