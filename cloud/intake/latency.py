"""Per-stream rolling window latency tracker.

The controller in `controller.py` only uses local queue/KV signals (M5); this
module tracks *end-to-end* per-window latency and exposes rolling P50/P95 for
post-hoc analysis of M3 / Fig.C plots. Not part of the control loop itself.

We record three stage latencies per window:

  * `append_ms`   — sum of synchronous /append call durations for the window
  * `pre_stream_end_ms` — wall time before the edge decision boundary reaches
                   intake; mostly simulated camera collection / parked session
  * `stream_end_to_result_ms` — cloud work after intake receives stream_end
  * `e2e_ms`     — wall time from window_open on the edge to intake logging
                   the vLLM `finished=True` poll result

A window is only considered for P95 once the vLLM poll has returned; empty
output windows are still counted.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

log = logging.getLogger("intake.latency")


@dataclass
class WindowLatencySample:
    stream_id: str
    window_id: int
    at_wall: float
    frames: int
    append_ms: float
    e2e_ms: float
    output_len: int
    stream_end_to_result_ms: float = 0.0
    pre_stream_end_ms: float = 0.0


@dataclass
class LatencyTracker:
    cap: int = 512

    _samples_by_stream: Dict[str, Deque[WindowLatencySample]] = field(default_factory=dict)
    _all_samples: Deque[WindowLatencySample] = field(default_factory=lambda: deque(maxlen=4096))

    def record(self, sample: WindowLatencySample) -> None:
        d = self._samples_by_stream.setdefault(
            sample.stream_id, deque(maxlen=self.cap)
        )
        d.append(sample)
        self._all_samples.append(sample)

    def percentile(self, stream_id: Optional[str], field_name: str, q: float) -> Optional[float]:
        samples: List[float] = []
        src = (
            list(self._samples_by_stream.get(stream_id, []))
            if stream_id is not None
            else list(self._all_samples)
        )
        for s in src:
            samples.append(getattr(s, field_name))
        if not samples:
            return None
        samples.sort()
        k = max(0, min(len(samples) - 1, int(round((q / 100.0) * (len(samples) - 1)))))
        return samples[k]

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for sid, buf in self._samples_by_stream.items():
            if not buf:
                continue
            append = [s.append_ms for s in buf]
            e2e = [s.e2e_ms for s in buf]
            final = [s.stream_end_to_result_ms for s in buf]
            pre_end = [s.pre_stream_end_ms for s in buf]
            out[sid] = {
                "n": float(len(buf)),
                "append_p50": _p(append, 50),
                "append_p95": _p(append, 95),
                "pre_stream_end_p50": _p(pre_end, 50),
                "pre_stream_end_p95": _p(pre_end, 95),
                "stream_end_to_result_p50": _p(final, 50),
                "stream_end_to_result_p95": _p(final, 95),
                "e2e_p50": _p(e2e, 50),
                "e2e_p95": _p(e2e, 95),
            }
        # aggregate
        if self._all_samples:
            append = [s.append_ms for s in self._all_samples]
            e2e = [s.e2e_ms for s in self._all_samples]
            final = [s.stream_end_to_result_ms for s in self._all_samples]
            pre_end = [s.pre_stream_end_ms for s in self._all_samples]
            out["__all__"] = {
                "n": float(len(self._all_samples)),
                "append_p50": _p(append, 50),
                "append_p95": _p(append, 95),
                "pre_stream_end_p50": _p(pre_end, 50),
                "pre_stream_end_p95": _p(pre_end, 95),
                "stream_end_to_result_p50": _p(final, 50),
                "stream_end_to_result_p95": _p(final, 95),
                "e2e_p50": _p(e2e, 50),
                "e2e_p95": _p(e2e, 95),
            }
        return out


def _p(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[k]
