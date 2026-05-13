"""Stable edge visual send-window search.

This module targets long-lived camera deployments. It learns a stable
per-engine send allowance below the vLLM early-finalizer boundary, then maps
that allowance to per-stream rho updates. The semantic decision window remains
separate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional, Set, Tuple


SendFn = Callable[[str, Dict[str, object]], Awaitable[None]]
ApplyFn = Callable[[str, float, str], None]
CurrentRhoFn = Callable[[str], Optional[float]]


@dataclass
class StableSendWindowConfig:
    lo: float = 0.05
    hi: float = 1.0
    initial: float = 0.0
    increase_step: float = 0.02
    stable_results_required: int = 20
    probe_interval_s: float = 120.0
    kv_probe_max: float = 0.85
    ef_reduce_factor: float = 0.80
    failure_reduce_factor: float = 0.50
    operating_margin: float = 0.85
    ef_cooldown_s: float = 90.0
    min_update_interval_s: float = 1.0


@dataclass
class EngineSendWindowState:
    engine_index: int
    window: float
    unsafe_window: Optional[float] = None
    ef_count: int = 0
    unsafe_count: int = 0
    stable_results: int = 0
    last_ef_wall: float = 0.0
    last_adjust_wall: float = 0.0
    last_probe_wall: float = 0.0
    last_reason: str = "init"


@dataclass
class StreamSendState:
    stream_id: str
    engine_index: int
    rho_content: float


class StableSendWindowManager:
    def __init__(
        self,
        cfg: Optional[StableSendWindowConfig] = None,
        send_fn: Optional[SendFn] = None,
        apply_fn: Optional[ApplyFn] = None,
        current_rho_fn: Optional[CurrentRhoFn] = None,
    ) -> None:
        self.cfg = cfg or StableSendWindowConfig()
        self.send_fn = send_fn
        self.apply_fn = apply_fn
        self.current_rho_fn = current_rho_fn
        self._engines: Dict[int, EngineSendWindowState] = {}
        self._streams: Dict[str, StreamSendState] = {}
        self._seen_ef_decisions: Set[Tuple[str, int]] = set()

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def _initial_window(self, rho_content: float) -> float:
        value = self.cfg.initial if self.cfg.initial > 0 else rho_content
        return self._clamp(value, self.cfg.lo, self.cfg.hi)

    def register_stream(self, stream_id: str, engine_index: int, rho_content: float) -> None:
        engine_index = int(engine_index)
        rho_content = self._clamp(rho_content, self.cfg.lo, self.cfg.hi)
        self._streams[stream_id] = StreamSendState(
            stream_id=stream_id,
            engine_index=engine_index,
            rho_content=rho_content,
        )
        self._engines.setdefault(
            engine_index,
            EngineSendWindowState(
                engine_index=engine_index,
                window=self._initial_window(rho_content),
            ),
        )

    def unregister_stream(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)

    def rho_limit_for_stream(self, stream_id: str) -> Optional[float]:
        stream = self._streams.get(stream_id)
        if stream is None:
            return None
        engine = self._engines.get(stream.engine_index)
        if engine is None:
            return stream.rho_content
        return min(stream.rho_content, engine.window)

    def snapshot(self) -> Dict[str, object]:
        return {
            "config": {
                "lo": self.cfg.lo,
                "hi": self.cfg.hi,
                "increase_step": self.cfg.increase_step,
                "stable_results_required": self.cfg.stable_results_required,
                "probe_interval_s": self.cfg.probe_interval_s,
                "kv_probe_max": self.cfg.kv_probe_max,
                "ef_reduce_factor": self.cfg.ef_reduce_factor,
                "operating_margin": self.cfg.operating_margin,
                "ef_cooldown_s": self.cfg.ef_cooldown_s,
            },
            "engines": {
                str(idx): {
                    "window": st.window,
                    "unsafe_window": st.unsafe_window,
                    "ef_count": st.ef_count,
                    "unsafe_count": st.unsafe_count,
                    "stable_results": st.stable_results,
                    "last_ef_wall": st.last_ef_wall,
                    "last_adjust_wall": st.last_adjust_wall,
                    "last_probe_wall": st.last_probe_wall,
                    "last_reason": st.last_reason,
                }
                for idx, st in sorted(self._engines.items())
            },
            "streams": {
                sid: {
                    "engine_index": st.engine_index,
                    "rho_content": st.rho_content,
                    "rho_limit": self.rho_limit_for_stream(sid),
                }
                for sid, st in sorted(self._streams.items())
            },
        }

    async def observe_result(
        self,
        *,
        stream_id: str,
        engine_index: int,
        early_finalized: bool,
        kv_usage: Optional[float] = None,
        decision_id: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        now = time.time() if now is None else float(now)
        engine = self._engines.setdefault(
            int(engine_index),
            EngineSendWindowState(
                engine_index=int(engine_index),
                window=self._clamp(self.cfg.initial or self.cfg.hi, self.cfg.lo, self.cfg.hi),
            ),
        )
        if early_finalized:
            if decision_id is not None:
                key = (stream_id, int(decision_id))
                if key in self._seen_ef_decisions:
                    return None
                self._seen_ef_decisions.add(key)
            return await self._on_ef(engine, stream_id=stream_id, now=now)
        return await self._on_stable(engine, stream_id=stream_id, kv_usage=kv_usage, now=now)

    async def _on_ef(
        self,
        engine: EngineSendWindowState,
        *,
        stream_id: str,
        now: float,
    ) -> Dict[str, object]:
        old = engine.window
        engine.ef_count += 1
        engine.unsafe_count += 1
        engine.stable_results = 0
        if engine.last_ef_wall > 0 and now - engine.last_ef_wall < self.cfg.ef_cooldown_s:
            reason = (
                f"send_window_ef_cooldown stream={stream_id} window={old:.3f} "
                f"ef_count={engine.ef_count}"
            )
            engine.last_reason = reason
            return {
                "action": "ef_cooldown",
                "engine_index": engine.engine_index,
                "old_window": old,
                "new_window": old,
                "reason": reason,
            }
        engine.last_ef_wall = now
        effective = self._effective_window(engine)
        engine.unsafe_window = (
            effective if engine.unsafe_window is None else min(engine.unsafe_window, effective)
        )
        target = effective * min(self.cfg.ef_reduce_factor, self.cfg.operating_margin)
        new = self._clamp(target, self.cfg.lo, self.cfg.hi)
        reason = (
            f"send_window_ef stream={stream_id} old={old:.3f} "
            f"effective={effective:.3f} new={new:.3f} "
            f"unsafe={engine.unsafe_window:.3f} ef_count={engine.ef_count}"
        )
        if old - new > 1e-6:
            engine.window = new
            engine.last_adjust_wall = now
            engine.last_reason = reason
            await self._push_engine(engine, reason, allow_increase=False)
        else:
            engine.last_reason = reason
        return {
            "action": "ef_reduce",
            "engine_index": engine.engine_index,
            "old_window": old,
            "new_window": engine.window,
            "reason": reason,
        }

    async def observe_engine_pressure(
        self,
        *,
        engine_index: int,
        kind: str,
        kv_usage: Optional[float] = None,
        severe: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, object]:
        now = time.time() if now is None else float(now)
        engine = self._engines.setdefault(
            int(engine_index),
            EngineSendWindowState(
                engine_index=int(engine_index),
                window=self._clamp(self.cfg.initial or self.cfg.hi, self.cfg.lo, self.cfg.hi),
            ),
        )
        old = engine.window
        engine.unsafe_count += 1
        engine.stable_results = 0
        if engine.last_ef_wall > 0 and now - engine.last_ef_wall < self.cfg.ef_cooldown_s:
            reason = (
                f"send_window_{kind}_cooldown engine={engine.engine_index} "
                f"window={old:.3f} unsafe_count={engine.unsafe_count}"
            )
            engine.last_reason = reason
            return {
                "action": f"{kind}_cooldown",
                "engine_index": engine.engine_index,
                "old_window": old,
                "new_window": old,
                "reason": reason,
            }
        engine.last_ef_wall = now
        effective = self._effective_window(engine)
        engine.unsafe_window = (
            effective if engine.unsafe_window is None else min(engine.unsafe_window, effective)
        )
        factor = self.cfg.failure_reduce_factor if severe else min(
            self.cfg.ef_reduce_factor,
            self.cfg.operating_margin,
        )
        new = self._clamp(effective * factor, self.cfg.lo, self.cfg.hi)
        reason = (
            f"send_window_{kind} engine={engine.engine_index} old={old:.3f} "
            f"effective={effective:.3f} new={new:.3f} unsafe={engine.unsafe_window:.3f} "
            f"kv={kv_usage if kv_usage is not None else -1:.3f} "
            f"unsafe_count={engine.unsafe_count}"
        )
        if old - new > 1e-6:
            engine.window = new
            engine.last_adjust_wall = now
            engine.last_reason = reason
            await self._push_engine(engine, reason, allow_increase=False)
        else:
            engine.last_reason = reason
        return {
            "action": f"{kind}_reduce",
            "engine_index": engine.engine_index,
            "old_window": old,
            "new_window": engine.window,
            "reason": reason,
        }

    def _effective_window(self, engine: EngineSendWindowState) -> float:
        values = [
            min(st.rho_content, engine.window)
            for st in self._streams.values()
            if st.engine_index == engine.engine_index
        ]
        if not values:
            return engine.window
        return max(values)

    async def _on_stable(
        self,
        engine: EngineSendWindowState,
        *,
        stream_id: str,
        kv_usage: Optional[float],
        now: float,
    ) -> Optional[Dict[str, object]]:
        if engine.last_ef_wall > 0 and now - engine.last_ef_wall < self.cfg.ef_cooldown_s:
            return None
        if kv_usage is not None and kv_usage > self.cfg.kv_probe_max:
            engine.stable_results = 0
            return None
        engine.stable_results += 1
        if engine.stable_results < max(1, int(self.cfg.stable_results_required)):
            return None
        if now - engine.last_probe_wall < self.cfg.probe_interval_s:
            return None
        target_hi = self.cfg.hi
        if engine.unsafe_window is not None:
            target_hi = min(target_hi, engine.unsafe_window * self.cfg.operating_margin)
        old = engine.window
        new = self._clamp(old + self.cfg.increase_step, self.cfg.lo, target_hi)
        if new <= old + 1e-6:
            engine.stable_results = 0
            return None
        reason = (
            f"send_window_probe stream={stream_id} old={old:.3f} new={new:.3f} "
            f"kv={kv_usage if kv_usage is not None else -1:.3f}"
        )
        engine.window = new
        engine.stable_results = 0
        engine.last_probe_wall = now
        engine.last_adjust_wall = now
        engine.last_reason = reason
        await self._push_engine(engine, reason, old_window=old, allow_increase=True)
        return {
            "action": "stable_probe",
            "engine_index": engine.engine_index,
            "old_window": old,
            "new_window": new,
            "reason": reason,
        }

    def _current_rho(self, stream_id: str) -> Optional[float]:
        if self.current_rho_fn is None:
            return None
        try:
            value = self.current_rho_fn(stream_id)
        except Exception:
            return None
        if value is None:
            return None
        return self._clamp(float(value), self.cfg.lo, self.cfg.hi)

    async def _push_engine(
        self,
        engine: EngineSendWindowState,
        reason: str,
        *,
        old_window: Optional[float] = None,
        allow_increase: bool = False,
    ) -> None:
        if self.send_fn is None:
            return
        for stream in list(self._streams.values()):
            if stream.engine_index != engine.engine_index:
                continue
            rho_limit = min(stream.rho_content, engine.window)
            current = self._current_rho(stream.stream_id)
            if current is not None:
                if allow_increase:
                    old_limit = min(
                        stream.rho_content,
                        engine.window if old_window is None else float(old_window),
                    )
                    if current < old_limit - 1e-4:
                        continue
                    rho = rho_limit
                else:
                    rho = min(current, rho_limit)
                if abs(rho - current) <= 1e-6:
                    continue
            else:
                rho = rho_limit
            msg: Dict[str, object] = {
                "kind": "rho_update",
                "rho": rho,
                "send_window": engine.window,
                "engine_index": engine.engine_index,
                "reason": reason,
            }
            await self.send_fn(stream.stream_id, msg)
            if self.apply_fn is not None:
                self.apply_fn(stream.stream_id, rho, reason)
