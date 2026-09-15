"""ReMe long-term memory store.

Workspace isolation: (device_sn, contact_id) -> one workspace.
Anonymous contacts (anon-*) skip both retrieve and record.
Record failures are swallowed silently — never poison the chat flow.

Stage-1 backend: JSON file per workspace with keyword-based retrieval.
The interface is async so a future FAISS/remote-ReMe backend drops in
without touching the graph nodes.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

from ayaka.memory.anonymous import is_anonymous

# 检索注入的包装模板
WRAP_TEMPLATE = """
以下内容来自当前用户的长期记忆,请结合当前问题使用:
<long_term_memory>
{content}
</long_term_memory>
"""

# 记录条数上限(防单 workspace 无限膨胀)
MAX_ENTRIES_PER_WORKSPACE = 500


class ReMeMemoryStore:
    """长期记忆存储,按 (device_sn, contact_id) 隔离 workspace。"""

    def __init__(self, base_dir: str | Path = ".data/reme") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._thread_lock = threading.Lock()

    # ---------- workspace ----------

    def _workspace_id(self, device_sn: str, contact_id: str) -> str:
        """workspace_id = deviceSn_contactId"""
        effective_contact = contact_id if contact_id else device_sn
        return f"{device_sn}_{effective_contact}"

    def _workspace_path(self, workspace_id: str) -> Path:
        return self.base_dir / f"{workspace_id}.json"

    def _get_lock(self, workspace_id: str) -> asyncio.Lock:
        with self._thread_lock:
            if workspace_id not in self._locks:
                self._locks[workspace_id] = asyncio.Lock()
            return self._locks[workspace_id]

    # ---------- 检索 ----------

    async def retrieve(
        self,
        device_sn: str,
        contact_id: str,
        query: str,
        top_k: int = 3,
    ) -> Optional[str]:
        """检索长期记忆,返回包装后的注入文本;无命中/匿名/关闭时返回 None。

        阶段1 用关键词重叠评分;阶段2 可换 FAISS 向量检索。
        """
        if is_anonymous(contact_id):
            return None
        if not query or not query.strip():
            return None

        workspace_id = self._workspace_id(device_sn, contact_id)
        async with self._get_lock(workspace_id):
            entries = await asyncio.to_thread(self._load, workspace_id)

        scored = self._score(entries, query)
        if not scored:
            return None
        top = scored[:top_k]
        combined = "\n".join(f"- {entry}" for entry, _score in top)
        return WRAP_TEMPLATE.format(content=combined)

    # ---------- 写入 ----------

    async def record(
        self,
        device_sn: str,
        contact_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """写入一轮记忆。匿名跳过;失败静默。"""
        if is_anonymous(contact_id):
            return
        if not user_text and not assistant_text:
            return

        workspace_id = self._workspace_id(device_sn, contact_id)
        async with self._get_lock(workspace_id):
            def _write() -> None:
                entries = self._load(workspace_id)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if user_text:
                    entries.append(f"[{timestamp}] 用户说: {user_text}")
                if assistant_text:
                    entries.append(f"[{timestamp}] 助手答: {assistant_text}")
                # 超限裁剪(保留最近的)
                if len(entries) > MAX_ENTRIES_PER_WORKSPACE:
                    entries = entries[-MAX_ENTRIES_PER_WORKSPACE:]
                self._save(workspace_id, entries)

            try:
                await asyncio.to_thread(_write)
            except OSError:
                # 写失败静默吞掉,不毒化对话主流程
                pass

    # ---------- 文件 IO ----------

    def _load(self, workspace_id: str) -> list[str]:
        path = self._workspace_path(workspace_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("entries", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, workspace_id: str, entries: list[str]) -> None:
        path = self._workspace_path(workspace_id)
        path.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    # ---------- 评分 ----------

    @staticmethod
    def _score(entries: list[str], query: str) -> list[tuple[str, float]]:
        """关键词重叠评分: 按查询词在条目中的命中数排序。

        用 2-3 字滑窗抽取中文子串(而非贪婪整句匹配),让"张三是谁"
        能拆成"张三"/"是谁"等短词,命中"我叫张三"的条目。
        """
        # 抽取查询中的有效词: 2-3 字中文滑窗 + 英文单词
        query_terms: set[str] = set()
        # 中文 2 字滑窗
        chinese_chars = re.findall(r"[\u4e00-\u9fa5]+", query)
        for segment in chinese_chars:
            for length in (2, 3):
                for i in range(len(segment) - length + 1):
                    query_terms.add(segment[i : i + length])
        # 英文单词(>= 2 字符)
        for word in re.findall(r"[A-Za-z]{2,}", query):
            query_terms.add(word)
        if not query_terms:
            return []
        scored: list[tuple[str, float]] = []
        for entry in entries:
            hits = sum(1 for term in query_terms if term in entry)
            if hits > 0:
                scored.append((entry, float(hits)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
