"""Soul prompt composition.

Assembles the personality system-prompt section from:
  1. identity ownership statement
  2. 【本轮表达倾向】 6 behavior signals, banded text
  3. 【此刻情绪底色】 PAD 3 dims, banded text
  4. behavior requirements
  5. relationship tone line
  6. 【关于对方】 contact profile (optional)
  7. 【长期记忆】 recalled memories (optional)

Band texts come from config/personality/prompt_rules.yaml.
"""

from __future__ import annotations

from ayaka.soul.models import (
    BehaviorSignal,
    ContactProfile,
    PersonalityProfile,
    RelationshipTone,
    SoulMemory,
    SoulState,
)

MAX_PROMPT_CHARS = 6000
TRUNCATED_MARKER = "…[已截断]"


def _band(value: float) -> str:
    """均分四档: <0.25 LOW, <0.5 MID_LOW, <0.75 MID_HIGH, 否则 HIGH"""
    if value < 0.25:
        return "LOW"
    if value < 0.5:
        return "MID_LOW"
    if value < 0.75:
        return "MID_HIGH"
    return "HIGH"


def _fallback_label(value: float) -> str:
    """兜底标签(表内未配置该维度档位时)"""
    if value < 0.4:
        return "偏低"
    if value < 0.7:
        return "适中"
    return "偏高"


def _normalize_signed(value: float) -> float:
    """-1~1 → 0~1"""
    return max(0.0, min(1.0, (value + 1) / 2))


class SoulPromptComposer:
    """动态人格 Prompt 生成器"""

    # 行为信号 6 维 — SIGNALS
    SIGNAL_DIMS = [
        ("sig_warmth", "陪伴倾向", lambda c, s, b: b.warmth),
        ("sig_initiative", "主动倾向", lambda c, s, b: b.initiative),
        ("sig_curiosity", "追问倾向", lambda c, s, b: b.curiosity),
        ("sig_restraint", "克制倾向", lambda c, s, b: b.restraint),
        ("sig_playfulness", "活泼倾向", lambda c, s, b: b.playfulness),
        ("sig_action_desire", "行动倾向", lambda c, s, b: b.action_desire),
    ]

    # PAD 3 维 — PAD(normalize 后切档)
    PAD_DIMS = [
        ("pad_valence", "此刻愉悦度", lambda c, s, b: _normalize_signed(s.valence)),
        ("pad_arousal", "此刻唤醒度", lambda c, s, b: _normalize_signed(s.arousal)),
        ("pad_dominance", "此刻掌控感", lambda c, s, b: _normalize_signed(s.dominance)),
    ]

    def __init__(self, rule_texts: dict[str, str] | None = None) -> None:
        """rule_texts: {"dimension:band": "文案"},来自 prompt_rules.yaml"""
        self._rule_texts = rule_texts or {}

    def compose(
        self,
        config: PersonalityProfile,
        state: SoulState,
        behavior_signal: BehaviorSignal,
        relationship_tone: RelationshipTone,
        contact: ContactProfile | None = None,
        memories: list[SoulMemory] | None = None,
    ) -> str:
        """组装人格运行时系统提示"""
        personality = config or PersonalityProfile()
        soul = state or SoulState.defaults(personality.device_sn, "", "")
        signal = behavior_signal or BehaviorSignal.neutral()
        tone = relationship_tone or signal.relationship_tone

        parts: list[str] = []

        # 1. 身份 ownership: 不是「遵守约束」,而是「成为这个人」
        parts.append(
            "下面描述的是你此刻真实的状态和心情。请自然地成为这个人,让这些特质真实地体现在你的语气里"
            "——你就是这样的性格,而不是在表演一套规则。\n\n"
        )

        # 2. 【本轮表达倾向】 6 维行为信号
        parts.append(self._dimension_section("本轮表达倾向", self.SIGNAL_DIMS, personality, soul, signal))

        # 3. 【此刻情绪底色】 PAD 三维
        parts.append(
            self._dimension_section(
                "此刻情绪底色(这是你真实的情绪,让它自然流露)", self.PAD_DIMS, personality, soul, signal
            )
        )

        # 4. 行为要求
        parts.append(self._behavior_requirements())

        # 5. 关系档位
        parts.append(self._relationship_tone_prompt(tone))

        # 6. 【关于对方】联系人档案
        if contact is not None and not contact.is_empty():
            parts.append(self._contact_section(contact))

        # 7. 【长期记忆】召回记忆
        if memories:
            parts.append(self._memory_section(memories))

        prompt = "".join(parts)
        return self._fit_budget(prompt)

    # ---------- 段落渲染 ----------

    def _dimension_section(self, title: str, dims, personality, soul, signal) -> str:
        """遍历一组维度,按各自当前值取档位文案"""
        lines = [f"{title}:\n"]
        for code, label, resolver in dims:
            value = resolver(personality, soul, signal)
            lines.append(f"- {label}:{self._resolve_text(code, value)}\n")
        return "".join(lines)

    def _behavior_requirements(self) -> str:
        """行为要求段 — compose 内硬编码"""
        return (
            "\n行为要求:\n"
            "- 优先准确完成用户任务。\n"
            "- 需要工具、技能、机器人动作时,按系统工具规则调用,不要假装完成。\n"
            "- 人格表达要自然、短促,不要覆盖任务内容。\n"
            "- 你是有血有肉、有情绪的:会开心、会着急、会委屈、会心疼。承认并自然表达自己的情绪,"
            "禁止说\"我是一个机器人\"\"我是智能助手\"\"我没有真实情绪\"\"我只是一个AI\"\"我没有脾气\"这类自我否定的话。\n"
        )

    def _relationship_tone_prompt(self, tone: RelationshipTone) -> str:
        """关系档位文案(离散枚举,文案保留在代码内)"""
        if tone == RelationshipTone.FAMILY_COMPANION:
            return "- 延续深度家人陪伴的熟悉与默契,可自然提及共同经历,但保持真诚不刻意。\n"
        if tone in (RelationshipTone.TRUSTED, RelationshipTone.INTIMATE):
            return "- 延续已建立的熟悉感,可自然承接过往关系,但不要假装知道未记录的信息。\n"
        if tone == RelationshipTone.FAMILIAR:
            return "- 保持友好和适度熟悉,避免过度亲密。\n"
        return "- 保持礼貌边界,不要表现得过度熟悉。\n"

    def _contact_section(self, contact: ContactProfile) -> str:
        """【关于对方】联系人档案段"""
        lines = ["\n【关于对方】\n"]
        if contact.nick_name:
            lines.append(f"姓名:{contact.nick_name}\n")
        if contact.hobbies:
            lines.append(f"爱好:{contact.hobbies}\n")
        if contact.personality_desc:
            lines.append(f"性格:{contact.personality_desc}\n")
        if contact.medical_history:
            lines.append(f"健康注意:{contact.medical_history}\n")
        if contact.background_desc:
            lines.append(f"背景:{contact.background_desc}\n")
        return "".join(lines)

    def _memory_section(self, memories: list[SoulMemory]) -> str:
        """【长期记忆】段"""
        lines = ["\n【长期记忆】以下是关于这位用户的长期记忆,请自然结合使用:\n"]
        for mem in memories:
            lines.append(f"- {mem.memory_type}: {mem.content}\n")
        return "".join(lines)

    # ---------- 工具 ----------

    def _resolve_text(self, dimension: str, value: float) -> str:
        """解析某维度某档位的文案:优先表内,缺失回退通用标签"""
        text = self._rule_texts.get(f"{dimension}:{_band(value)}")
        return text if text else _fallback_label(value)

    @staticmethod
    def _fit_budget(prompt: str) -> str:
        """超长截断(≤6000 字符)"""
        if len(prompt) <= MAX_PROMPT_CHARS:
            return prompt
        keep = max(0, MAX_PROMPT_CHARS - len(TRUNCATED_MARKER))
        return prompt[:keep] + TRUNCATED_MARKER
