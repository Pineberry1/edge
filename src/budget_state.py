"""Thread-safe per-stream window budget shared with the uplink listener."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class BudgetState:
    def __init__(self, default_windows: int = 10) -> None:
        self._lock = threading.Lock()
        self._windows_per_decision = max(1, int(default_windows))
        self._version = 0
        self._last_reason: Optional[str] = None
        self._last_update_wall: float = time.time()
        self._changed = False
        self._force_close: Optional[Dict[str, Any]] = None

    def set(self, version: int, windows_per_decision: int, reason: str) -> bool:
        with self._lock:
            version = int(version)
            if version <= self._version:
                return False
            new_windows = max(1, int(windows_per_decision))
            changed = new_windows != self._windows_per_decision
            self._version = version
            self._windows_per_decision = new_windows
            self._last_reason = reason
            self._last_update_wall = time.time()
            self._changed = self._changed or changed
            return changed

    def request_force_close(
        self,
        *,
        reason: str,
        decision_id: Optional[int] = None,
        stream_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            event: Dict[str, Any] = {
                "reason": reason,
                "at_wall": time.time(),
            }
            if decision_id is not None:
                event["decision_id"] = int(decision_id)
            if stream_id:
                event["stream_id"] = str(stream_id)
            self._force_close = event

    def consume_force_close(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            event = self._force_close
            self._force_close = None
            return dict(event) if event is not None else None

    def consume_window(self, count: int) -> bool:
        with self._lock:
            return int(count) >= self._windows_per_decision

    @property
    def windows_per_decision(self) -> int:
        with self._lock:
            return self._windows_per_decision

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def changed_since_last_check(self) -> bool:
        with self._lock:
            changed = self._changed
            self._changed = False
            return changed

    def reset_decision(self) -> None:
        # The edge loop owns the per-decision counter; this method is kept as
        # a named hook for callers that want to mark that boundary explicitly.
        return None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "version": self._version,
                "windows_per_decision": self._windows_per_decision,
                "last_reason": self._last_reason,
                "last_update_wall": self._last_update_wall,
                "force_close_pending": self._force_close is not None,
            }
