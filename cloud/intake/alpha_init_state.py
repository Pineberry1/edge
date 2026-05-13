"""Thread-safe per-stream alpha_init cache.

alpha_init is content-derived by the vLLM merger and returned as a side-channel
hint. Intake applies it to the next window by composing it with the controller's
global pressure factor.
"""

from __future__ import annotations

import threading
from typing import Optional


class AlphaInitState:
    """Per-stream alpha_init cache. One instance per StreamSession."""

    def __init__(self, default: float = 1.0, alpha_min: float = 0.3) -> None:
        self._lock = threading.Lock()
        self._alpha_min = float(alpha_min)
        self._alpha_init = self._clamp(default)
        self._version = 0
        self._last_window_id: Optional[int] = None

    def _clamp(self, value: float) -> float:
        return max(self._alpha_min, min(1.0, float(value)))

    def update_from_response(self, hint: Optional[float], window_id: int) -> None:
        if hint is None:
            return
        value = self._clamp(float(hint))
        with self._lock:
            if int(window_id) == self._last_window_id:
                return
            self._alpha_init = value
            self._last_window_id = int(window_id)
            self._version += 1

    def get(self) -> tuple[float, int]:
        with self._lock:
            return self._alpha_init, self._version

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "alpha_init": self._alpha_init,
                "version": self._version,
                "last_window_id": self._last_window_id,
                "alpha_min": self._alpha_min,
            }
