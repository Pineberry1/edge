"""Decision-window budget allocator for active streams.

The default policy keeps the camera decision horizon fixed. BAVA's rho/alpha
controller owns normal pressure control; the budget channel only shortens a
decision window after vLLM's online-prefill early-finalizer has actually fired.
The old fair-share shortening policy remains available as an explicit ablation.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Dict, Iterable


log = logging.getLogger("intake.budget")


@dataclass(frozen=True)
class StreamCostMeta:
    stream_id: str
    frame_height: int
    frame_width: int
    frames_per_window: int
    rho: float
    prompt_tokens: int
    max_tokens: int
    alpha: float = 1.0
    live_kv_tokens: int = 0
    memory_tokens_per_window: int = 0
    early_finalized_recent: bool = False
    last_early_finalized_windows: int = 0


@dataclass(frozen=True)
class GlobalCapacity:
    kv_cap_tokens: int
    safety_factor: float = 0.85
    min_windows: int = 2
    max_windows: int = 32
    block_overhead_tokens: int = 64
    policy: str = "ef_guard"
    target_windows: int = 10
    ef_windows: int = 6
    ef_dynamic: bool = True
    ef_margin_windows: int = 1


@dataclass(frozen=True)
class StreamBudget:
    stream_id: str
    windows_per_decision: int
    est_kv_tokens: int
    version: int


def tokens_per_frame(h: int, w: int) -> int:
    # Qwen3-VL: each visual patch is 28 px; clamp to 1.
    return max(1, math.ceil(h / 28) * math.ceil(w / 28))


def per_window_tokens(meta: StreamCostMeta) -> int:
    eff_frames = max(1, math.ceil(meta.rho * meta.frames_per_window))
    visual_tokens = eff_frames * tokens_per_frame(meta.frame_height, meta.frame_width)
    folded = max(1, math.ceil(visual_tokens * max(0.0, min(1.0, meta.alpha))))
    return folded + max(0, int(meta.memory_tokens_per_window))


def per_decision_tokens(meta: StreamCostMeta, windows: int, cap_overhead: int) -> int:
    return (
        windows * per_window_tokens(meta)
        + meta.prompt_tokens
        + meta.max_tokens
        + cap_overhead
        + max(0, int(meta.live_kv_tokens))
    )


class BudgetAllocator:
    def __init__(self, cap: GlobalCapacity) -> None:
        self.cap = cap
        self._version = 0

    def _next_version(self) -> int:
        self._version += 1
        return self._version

    def _clamp_windows(self, windows: int) -> int:
        cap = self.cap
        lo = max(1, int(cap.min_windows))
        hi = max(lo, int(cap.max_windows))
        return max(lo, min(hi, int(windows)))

    def _fair_share_windows(self, meta: StreamCostMeta, n_streams: int) -> int:
        cap = self.cap
        share = (cap.kv_cap_tokens * cap.safety_factor) / max(1, int(n_streams))
        per_window = per_window_tokens(meta)
        headroom = (
            share
            - meta.prompt_tokens
            - meta.max_tokens
            - cap.block_overhead_tokens
            - max(0, int(meta.live_kv_tokens))
        )
        return self._clamp_windows(max(1, int(headroom // per_window)))

    def _ef_guard_windows(
        self,
        meta: StreamCostMeta,
        *,
        target_windows: int,
        ef_floor_windows: int,
    ) -> int:
        if not bool(meta.early_finalized_recent):
            return target_windows
        if not bool(self.cap.ef_dynamic):
            return ef_floor_windows
        observed = max(0, int(meta.last_early_finalized_windows or 0))
        if observed <= 0:
            return ef_floor_windows
        margin = max(0, int(self.cap.ef_margin_windows or 0))
        return self._clamp_windows(
            min(target_windows, max(ef_floor_windows, observed + margin))
        )

    def allocate(self, metas: Iterable[StreamCostMeta]) -> Dict[str, StreamBudget]:
        meta_list = list(metas)
        if not meta_list:
            return {}
        cap = self.cap
        policy = str(cap.policy or "ef_guard").strip().lower()
        target_windows = self._clamp_windows(int(cap.target_windows or 10))
        ef_windows = min(
            target_windows,
            self._clamp_windows(int(cap.ef_windows or target_windows)),
        )
        out: Dict[str, StreamBudget] = {}
        for meta in meta_list:
            if policy in {"fair", "fair_share", "equal_share", "legacy"}:
                w = self._fair_share_windows(meta, len(meta_list))
            elif policy in {"ef", "ef_guard", "early_finalizer_guard"}:
                w = self._ef_guard_windows(
                    meta,
                    target_windows=target_windows,
                    ef_floor_windows=ef_windows,
                )
            elif policy in {"fixed", "fixed_window", "target", "target_window"}:
                w = target_windows
            else:
                log.warning("unknown BAVA budget policy %r; using ef_guard", policy)
                w = target_windows
            out[meta.stream_id] = StreamBudget(
                stream_id=meta.stream_id,
                windows_per_decision=w,
                est_kv_tokens=per_decision_tokens(meta, w, cap.block_overhead_tokens),
                version=self._next_version(),
            )
        return out
