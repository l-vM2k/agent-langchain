"""Soul memory selection + extraction — mirrors SoulMemorySelector.java +
RuleBasedSoulMemoryExtractor.java.

Selection: filter ACTIVE + usable (confidence/importance thresholds) + relevant
to query, sort by importance desc then confidence desc.

Extraction: rule-based keyword hits over the turn text, producing typed
memories (BOUNDARY / FAMILY_MEMBER / ROUTINE / PREFERENCE / IMPORTANT_EVENT /
CARE_NEED / SHARED_EXPERIENCE).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from xiaowei.soul.models import SoulMemory

MIN_CONFIDENCE = 0.55
MIN_IMPORTANCE = 0.35

# 相关性关键词 — SoulMemorySelector.isRelevant
RELEVANCE_KEYWORDS = ["睡前", "妈妈", "爸爸", "家人", "喜欢", "不喜欢", "故事"]

# 抽取默认有效期 — RuleBasedSoulMemoryExtractor(+180 天)
DEFAULT_EXPIRE_DAYS = 180


class SoulMemorySelector:
    """灵魂长期记忆选择器 — SoulMemorySelector.java"""

    def select(self, memories: list[SoulMemory], query: str) -> list[SoulMemory]:
        """过滤 + 排序 — SoulMemorySelector.select

        1. ACTIVE 状态
        2. 未过期 + confidence >= 0.55 + importance >= 0.35
        3. 与 query 相关(或 query 为空时全通过)
        4. importance 降序 → confidence 降序
        """
        if not memories:
            return []
        now = datetime.now()
        selected = [
            m for m in memories
            if m is not None
            and m.status == "ACTIVE"
            and self._is_usable(m, now)
            and self._is_relevant(m, query)
        ]
        selected.sort(key=lambda m: (m.importance, m.confidence), reverse=True)
        return selected

    def _is_usable(self, memory: SoulMemory, now: datetime) -> bool:
        """未过期 + 双门槛 — SoulMemorySelector.isUsable"""
        if memory.expire_at is not None and memory.expire_at < now:
            return False
        return memory.confidence >= MIN_CONFIDENCE and memory.importance >= MIN_IMPORTANCE

    @staticmethod
    def _is_relevant(memory: SoulMemory, query: str) -> bool:
        """相关性判定 — SoulMemorySelector.isRelevant"""
        if not query or not query.strip():
            return True
        content = memory.content or ""
        if not content.strip():
            return False
        for keyword in RELEVANCE_KEYWORDS:
            if keyword in query and keyword in content:
                return True
        return content in query or query in content


class RuleBasedSoulMemoryExtractor:
    """规则版灵魂长期记忆抽取器 — RuleBasedSoulMemoryExtractor.java

    关键词命中即抽取,并列判断非互斥。
    """

    def extract(
        self,
        device_sn: str,
        contact_id: str,
        text: str,
    ) -> list[SoulMemory]:
        """从本轮对话文本抽取记忆 — extract"""
        if not text or not text.strip():
            return []
        memories: list[SoulMemory] = []

        # 边界: 用户明确不希望的称呼/行为(不过期)
        if self._contains_any(text, "不喜欢被叫", "不要叫我", "别叫我"):
            memories.append(self._memory(device_sn, contact_id, "BOUNDARY", text,
                                         0.86, 0.9, "STANDARD", never_expire=True))
        # 家庭成员称谓(不过期,私密)
        if "妈妈叫" in text or "爸爸叫" in text or "家人叫" in text:
            memories.append(self._memory(device_sn, contact_id, "FAMILY_MEMBER", text,
                                         0.82, 0.85, "PRIVATE", never_expire=True))
        # 作息 vs 偏好: 互斥(睡前+偏好落 ROUTINE,否则泛"喜欢"落 PREFERENCE)
        if "睡前" in text and self._contains_any(text, "喜欢", "习惯", "想听"):
            memories.append(self._memory(device_sn, contact_id, "ROUTINE", text,
                                         0.8, 0.78, "STANDARD", never_expire=False))
        elif "喜欢" in text:
            memories.append(self._memory(device_sn, contact_id, "PREFERENCE", text,
                                         0.76, 0.68, "STANDARD", never_expire=False))
        # 重要事件: 生日/纪念日/考试/毕业/搬家/入职/退休等里程碑
        if self._contains_any(text, "纪念日", "生日", "结婚", "考试", "毕业", "搬家", "入职", "退休"):
            memories.append(self._memory(device_sn, contact_id, "IMPORTANT_EVENT", text,
                                         0.85, 0.88, "STANDARD", never_expire=False))
        # 关怀需求: 吃药/提醒/按时/复查等健康嘱托(私密)
        if self._contains_any(text, "吃药", "按时", "提醒我", "照顾", "注意身体", "别熬夜", "复查"):
            memories.append(self._memory(device_sn, contact_id, "CARE_NEED", text,
                                         0.80, 0.82, "PRIVATE", never_expire=False))
        # 共同经历: 我们一起/昨天我们/全家等陪伴记忆
        if self._contains_any(text, "我们一起", "昨天我们", "上次我们", "全家", "一起去看"):
            memories.append(self._memory(device_sn, contact_id, "SHARED_EXPERIENCE", text,
                                         0.78, 0.75, "STANDARD", never_expire=False))
        return memories

    # ---------- 内部 ----------

    @staticmethod
    def _contains_any(text: str, *keywords: str) -> bool:
        return any(k in text for k in keywords)

    @staticmethod
    def _memory(
        device_sn: str,
        contact_id: str,
        memory_type: str,
        content: str,
        confidence: float,
        importance: float,
        privacy_level: str,
        never_expire: bool,
    ) -> SoulMemory:
        """记忆工厂 — RuleBasedSoulMemoryExtractor.memory"""
        return SoulMemory(
            device_sn=device_sn,
            contact_id=contact_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            importance=importance,
            privacy_level=privacy_level,
            status="ACTIVE",
            expire_at=None if never_expire else datetime.now() + timedelta(days=DEFAULT_EXPIRE_DAYS),
        )
