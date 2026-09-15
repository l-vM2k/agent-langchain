"""Soul memory persistence — in-memory store for extracted SoulMemory entries.

Mirrors agent_soul_memory table access (IAgentSoulMemoryService) in a
file-backed form suitable for the learning project.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from xiaowei.soul.models import SoulMemory


class SoulMemoryStore:
    """灵魂长期记忆存储 — 对应 agent_soul_memory 表。

    Keyed by (device_sn, contact_id); JSON-file persisted.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._memories: dict[str, dict[str, list[SoulMemory]]] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # ---------- 查询 ----------

    def list_active(self, device_sn: str, contact_id: str) -> list[SoulMemory]:
        """列出该 (device, contact) 的 ACTIVE 记忆。"""
        with self._lock:
            entries = self._memories.get(device_sn, {}).get(contact_id, [])
            return [m.model_copy(deep=True) for m in entries if m.status == "ACTIVE"]

    # ---------- 写入 ----------

    def add(self, memory: SoulMemory) -> SoulMemory:
        """插入一条记忆(分配 id)。"""
        with self._lock:
            memory.id = self._next_id
            self._next_id += 1
            self._memories.setdefault(memory.device_sn, {}).setdefault(memory.contact_id, []).append(
                memory.model_copy(deep=True)
            )
            self._persist()
            return memory

    def add_all(self, memories: list[SoulMemory]) -> list[SoulMemory]:
        return [self.add(m) for m in memories]

    # ---------- 持久化 ----------

    def _persist(self) -> None:
        if not self._persist_path:
            return
        payload = {
            "next_id": self._next_id,
            "memories": {
                device: {
                    contact: [m.model_dump(mode="json") for m in entries]
                    for contact, entries in contacts.items()
                }
                for device, contacts in self._memories.items()
            },
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._next_id = payload.get("next_id", 1)
            for device, contacts in payload.get("memories", {}).items():
                self._memories[device] = {
                    contact: [SoulMemory.model_validate(m) for m in entries]
                    for contact, entries in contacts.items()
                }
        except (json.JSONDecodeError, OSError, ValueError):
            self._memories = {}
            self._next_id = 1

    def reset(self, device_sn: Optional[str] = None, contact_id: Optional[str] = None) -> None:
        """清空记忆(测试用)。"""
        with self._lock:
            if device_sn is None:
                self._memories.clear()
                self._next_id = 1
            elif contact_id is None:
                self._memories.pop(device_sn, None)
            else:
                self._memories.get(device_sn, {}).pop(contact_id, None)
            self._persist()
