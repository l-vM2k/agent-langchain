"""Knowledge search tool — mirrors KnowledgeSearchTool.java.

Stage-1 backend: keyword search over a local JSON knowledge file
(config/knowledge/base.json). Interface mirrors a vector store so a
FAISS backend can replace it later without touching the graph.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(description="检索查询词")


class KnowledgeBase:
    """本地知识库(JSON 文件,关键词匹配)。"""

    def __init__(self, knowledge_path: str | Path) -> None:
        self._path = Path(knowledge_path)
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    self._entries = json.loads(self._path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    self._entries = []
            else:
                self._entries = []

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """关键词重叠评分检索。"""
        with self._lock:
            if not query.strip() or not self._entries:
                return []
            scored = []
            for entry in self._entries:
                text = entry.get("content", "")
                score = sum(1 for ch in query if ch in text) + sum(
                    1 for term in query.split() if term in text
                )
                if score > 0:
                    scored.append((text, score))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            return [text for text, _ in scored[:top_k]]


def create_knowledge_tool(knowledge_path: str | Path):
    """创建知识库检索工具 — 对应 KnowledgeSearchTool。"""
    base = KnowledgeBase(knowledge_path)

    async def _search(query: str) -> str:
        results = base.search(query)
        if not results:
            return "知识库中没有找到相关内容。"
        return "\n---\n".join(results)

    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        coroutine=_search,
        name="knowledge_search",
        description="在知识库中检索相关信息。当用户的问题可能涉及知识库内容(产品文档/FAQ/资料)时调用。",
        args_schema=KnowledgeSearchArgs,
    )
