"""Soul state persistence — mirrors AgentStateStore (in-memory / SQLite).

Stage 1-3: in-memory dict keyed by (device_sn, contact_id).
Stage 4+: optional SQLite backend (same interface).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from xiaowei.soul.models import SoulState


class SoulStateStore:
    """灵魂状态存储 — 对应 Java AgentStateStore。

    In-memory implementation with optional JSON-file persistence so state
    survives restarts on the 2C2G server (file is tiny).
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._states: dict[str, dict[str, SoulState]] = {}
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # ---------- 查询 ----------

    def get(self, device_sn: str, contact_id: str) -> Optional[SoulState]:
        """按 (device_sn, contact_id) 取状态,无则 None。"""
        with self._lock:
            state = self._states.get(device_sn, {}).get(contact_id)
            return state.model_copy(deep=True) if state else None

    # ---------- 写入 ----------

    def put(self, state: SoulState) -> None:
        """upsert 一条状态。"""
        with self._lock:
            self._states.setdefault(state.device_sn, {})[state.contact_id] = state.model_copy(deep=True)
            self._persist()

    def delete(self, device_sn: str, contact_id: str) -> None:
        with self._lock:
            device_map = self._states.get(device_sn)
            if device_map:
                device_map.pop(contact_id, None)
                self._persist()

    # ---------- 持久化 ----------

    def _persist(self) -> None:
        if not self._persist_path:
            return
        payload = {
            device: {contact: state.model_dump(mode="json") for contact, state in contacts.items()}
            for device, contacts in self._states.items()
        }
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def _load_from_disk(self) -> None:
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for device, contacts in payload.items():
                self._states[device] = {
                    contact: SoulState.model_validate(data) for contact, data in contacts.items()
                }
        except (json.JSONDecodeError, OSError, ValueError):
            # 损坏文件: 静默重置(状态可由对话重建)
            self._states = {}
