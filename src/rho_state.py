"""Thread-safe mutable ρ shared between the packet pipeline and the uplink
reverse-channel listener.

The uplink thread receives cloud→edge `rho_update` messages and writes here;
the GOP buffer reads here every time it finalises a GOP. A plain lock is
enough — updates are rare (controller tick, not per-packet) and reads are
cheap.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class RhoState:
    def __init__(self, initial: float, lo: float = 0.02, hi: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._value = self._clamp(initial, lo, hi)
        self._lo = lo
        self._hi = hi
        self._last_update_wall: float = time.time()
        self._update_count: int = 0
        self._last_reason: Optional[str] = None

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    @property
    def current(self) -> float:
        with self._lock:
            return self._value

    def set(self, new_value: float, reason: Optional[str] = None) -> float:
        with self._lock:
            self._value = self._clamp(new_value, self._lo, self._hi)
            self._last_update_wall = time.time()
            self._update_count += 1
            self._last_reason = reason
            return self._value

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "rho": self._value,
                "lo": self._lo,
                "hi": self._hi,
                "updates": self._update_count,
                "last_reason": self._last_reason,
                "last_update_wall": self._last_update_wall,
            }
