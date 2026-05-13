"""BAVA closed-loop controller — M4/M5 with asymmetric recovery + per-stream weighting.

Implements the v10 dual-valve split:

    rho   = edge/network ingress valve, reduced only by network or
            pre-alpha ingress pressure.
    alpha = cloud-side KV admission valve, reduced by compute / KV pressure.

The important separation is that vLLM's global waiting count is not allowed to
drive rho in the default controller mode. In online-prefill long-session runs
that metric can track open sessions rather than real backlog, which would make
every multi-camera run look overloaded. The default rho policy keeps the
content-side stream rho unless network / decode / vision pressure is observed.

    alpha_used = clamp(alpha_init * eta_press(KV, flow), alpha_min, 1.0)

alpha is split into a content side-channel (alpha_init, produced by the
merger) and a global pressure factor (eta_press, computed from compute / KV
pressure).

Extensions added to the plain §5 form
-------------------------------------

1. **Asymmetric recovery.** When ingress pressure sits under target for
   several consecutive ticks, rho climbs back toward the stream's content
   default. It does not climb above that default unless explicitly configured.

2. **Per-stream weighting.** vLLM `/metrics` is global, but intake knows how
   many of those waiting requests each stream contributed via its own
   `inflight_appends` counter. The global Δρ / Δα is weighted by each
   stream's share of intake-side load, clamped to [0.5, 2.0] so idle streams
   still receive *some* update and busy streams pay extra during overload.

3. **Emergency preempt brake.** Unchanged: on sustained
   `num_preemptions_total` rate, cut ρ harder.

4. **Diagnostics.** The log keeps legacy Q/KV fields for visibility, but rho
   decisions are tagged with network/vision/legacy reasons.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Tuple

from .metrics import VllmSnapshot

log = logging.getLogger("intake.controller")

SendFn = Callable[[str, Dict[str, object]], Awaitable[None]]
# stream_id → (inflight_appends, last_append_ms_or_None, inflight_windows)
LoadLookup = Callable[[str], Tuple[int, Optional[float], int]]
AlphaInitLookup = Callable[[str], Tuple[float, int]]
FlowLookup = Callable[[], Optional[Any]]
RhoLimitLookup = Callable[[str], Optional[float]]


class MetricsScraper(Protocol):
    async def scrape(self) -> VllmSnapshot:
        ...


@dataclass
class ControllerConfig:
    tick_s: float = 0.5

    # Setpoints (M5 reference targets).
    q_target: float = 2.0         # target vllm:num_requests_waiting
    kv_target: float = 0.6        # target vllm:kv_cache_usage_perc

    # Step sizes — kept small for the stability lemma in §5.
    mu_rho: float = 0.05
    mu_alpha: float = 0.05

    # Dead-band around setpoints — no cutoff update while inside this band.
    q_deadband: float = 1.0
    kv_deadband: float = 0.05

    # Per-stream throttle so we don't spam the edge.
    min_update_interval_s: float = 1.0

    # Bounds.
    rho_lo: float = 0.05
    rho_hi: float = 1.0
    alpha_lo: float = 0.3
    alpha_hi: float = 1.0

    # Emergency brake: on sustained preemptions, cut rho harder.
    preempt_rate_panic: float = 0.5
    preempt_extra_step: float = 0.1

    # Asymmetric climb-back — slower than cutoff.
    climb_back_enabled: bool = True
    climb_back_ticks_required: int = 6     # consecutive under-target ticks
    climb_back_step_rho: float = 0.02
    climb_back_q_slack: float = 0.5        # must be below Q* - slack
    climb_back_kv_slack: float = 0.1       # must be below KV* - slack

    # Per-stream weighting.
    per_stream_weighting: bool = True
    per_stream_weight_min: float = 0.5
    per_stream_weight_max: float = 2.0

    # v10 token-flow pressure model. A zero capacity disables that pressure.
    # rho_control_mode:
    #   network (default): rho only reacts to network / pre-alpha ingress.
    #   queue: legacy behavior, vLLM waiting queue can reduce rho.
    #   off: controller never changes rho, but alpha still updates.
    rho_control_mode: str = "network"
    net_capacity_bytes_s: float = 0.0
    net_target_util: float = 0.90
    net_deadband: float = 0.05
    net_backlog_target_s: float = 2.0
    net_send_wait_target_ms: float = 250.0
    edge_capacity_tokens_s: float = 0.0  # pre-alpha decode / vision token capacity
    edge_target_util: float = 0.90
    edge_deadband: float = 0.05
    prefill_capacity_tokens_s: float = 0.0
    kv_horizon_s: float = 10.0
    kv_margin_ratio: float = 0.10
    eta_floor: float = 0.5


@dataclass
class ControllerStreamState:
    rho: float
    alpha: float
    rho_content: float = 1.0
    alpha_init: float = 1.0
    alpha_init_version: int = 0
    eta_press: float = 1.0
    last_update_wall: float = 0.0
    last_push_reason: Optional[str] = None


class BavaController:
    """Closed-loop controller over an aggregated vLLM metrics view.

    When per-stream weighting is enabled and `load_lookup` is provided, each
    stream receives a weighted share of the global Δρ/Δα proportional to its
    contribution to intake-side load (`inflight_appends + inflight_windows`).
    """

    def __init__(
        self,
        scraper: MetricsScraper,
        config: Optional[ControllerConfig] = None,
        log_path: Optional[Path] = None,
        load_lookup: Optional[LoadLookup] = None,
        alpha_init_lookup: Optional[AlphaInitLookup] = None,
        flow_lookup: Optional[FlowLookup] = None,
        rho_limit_lookup: Optional[RhoLimitLookup] = None,
    ) -> None:
        self.scraper = scraper
        self.cfg = config or ControllerConfig()
        self.log_path = log_path
        self.load_lookup = load_lookup
        self.alpha_init_lookup = alpha_init_lookup
        self.flow_lookup = flow_lookup
        self.rho_limit_lookup = rho_limit_lookup
        self._streams: Dict[str, ControllerStreamState] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._send_fn: Optional[SendFn] = None
        self._snapshot: Optional[VllmSnapshot] = None
        self._last_push_per_stream: Dict[str, Dict[str, object]] = {}
        self._ticks_since_heartbeat: int = 0
        # Under-target streaks for asymmetric recovery.
        self._q_under_streak: int = 0
        self._kv_under_streak: int = 0
        self._rho_under_streak: int = 0

    # -------- stream lifecycle --------

    def track(self, stream_id: str, initial_rho: float, initial_alpha: float = 1.0) -> None:
        rho = _clamp(float(initial_rho), self.cfg.rho_lo, self.cfg.rho_hi)
        self._streams[stream_id] = ControllerStreamState(
            rho=rho,
            alpha=float(initial_alpha),
            rho_content=rho,
        )

    def untrack(self, stream_id: str) -> None:
        self._streams.pop(stream_id, None)
        self._last_push_per_stream.pop(stream_id, None)

    def state(self, stream_id: str) -> Optional[ControllerStreamState]:
        return self._streams.get(stream_id)

    def snapshot(self) -> Optional[VllmSnapshot]:
        return self._snapshot

    @staticmethod
    def _compute_eta_press(
        snap: Optional[VllmSnapshot],
        flow: Optional[Any] = None,
        cfg: Optional[ControllerConfig] = None,
    ) -> float:
        return _compute_eta_press(snap, flow, cfg)

    def _flow_snapshot(self) -> Optional[Any]:
        if self.flow_lookup is None:
            return None
        try:
            return self.flow_lookup()
        except Exception:
            return None

    def _alpha_init_for_stream(self, stream_id: str) -> Tuple[float, int]:
        if self.alpha_init_lookup is None:
            return (1.0, 0)
        try:
            alpha_init, version = self.alpha_init_lookup(stream_id)
        except Exception:
            return (1.0, 0)
        return (_clamp(float(alpha_init), self.cfg.alpha_lo, 1.0), int(version))

    def _next_alpha_for_stream(
        self,
        stream_id: str,
        snap: Optional[VllmSnapshot],
        flow: Optional[Any] = None,
    ) -> Tuple[float, float, int, float]:
        alpha_init, version = self._alpha_init_for_stream(stream_id)
        eta = self._compute_eta_press(snap, flow, self.cfg)
        alpha_used = _clamp(alpha_init * eta, self.cfg.alpha_lo, self.cfg.alpha_hi)
        return alpha_used, alpha_init, version, eta

    def alpha_for_stream(self, stream_id: str) -> float:
        alpha_used, alpha_init, version, eta = self._next_alpha_for_stream(
            stream_id,
            self._snapshot,
            self._flow_snapshot(),
        )
        st = self._streams.get(stream_id)
        if st is not None:
            st.alpha = alpha_used
            st.alpha_init = alpha_init
            st.alpha_init_version = version
            st.eta_press = eta
        return alpha_used

    # -------- task lifecycle --------

    def start(self, send_fn: SendFn) -> None:
        if self._task is not None:
            return
        self._send_fn = send_fn
        self._task = asyncio.create_task(self._loop(), name="bava-controller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                self._task.cancel()

    # -------- control loop --------

    async def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    snap = await self.scraper.scrape()
                    self._snapshot = snap
                    await self._on_snapshot(snap)
                except Exception as e:  # noqa: BLE001
                    log.warning("controller tick error: %s", e)
                remaining = max(0.0, self.cfg.tick_s - (time.time() - t0))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass
        finally:
            pass

    # -------- core update --------

    def _compute_stream_weights(self) -> Dict[str, float]:
        """Per-stream weight for the global Δρ/Δα update.

        Weight is proportional to stream_load / mean_load, clamped to
        [per_stream_weight_min, per_stream_weight_max]. Falls back to uniform
        (1.0 for each) if no load lookup is available or no one has load.
        """
        streams = list(self._streams.keys())
        n = len(streams)
        if n == 0:
            return {}
        if not self.cfg.per_stream_weighting or self.load_lookup is None:
            return {sid: 1.0 for sid in streams}
        loads: Dict[str, float] = {}
        for sid in streams:
            try:
                inflight_app, last_ms, inflight_win = self.load_lookup(sid)
            except Exception:
                inflight_app, last_ms, inflight_win = 0, None, 0
            loads[sid] = float(inflight_app) + 0.5 * float(inflight_win)
        total = sum(loads.values())
        if total <= 1e-9:
            return {sid: 1.0 for sid in streams}
        mean = total / n
        weights: Dict[str, float] = {}
        for sid, l in loads.items():
            raw = l / mean if mean > 0 else 1.0
            weights[sid] = max(
                self.cfg.per_stream_weight_min,
                min(self.cfg.per_stream_weight_max, raw),
            )
        return weights

    async def _on_snapshot(self, snap: VllmSnapshot) -> None:
        self._snapshot = snap
        flow = self._flow_snapshot()
        flow_diag = _flow_diagnostics(snap, flow, self.cfg)
        q_err = snap.num_requests_waiting - self.cfg.q_target
        kv_err = snap.kv_cache_usage_perc - self.cfg.kv_target
        lyapunov = 0.5 * (q_err * q_err + kv_err * kv_err)

        # Rho is an ingress valve. In the default v10 mode, only network or
        # pre-alpha decode/vision pressure can reduce rho. Raw vLLM waiting
        # queue is kept for diagnostics and for an explicit legacy mode only.
        rho_mode = (self.cfg.rho_control_mode or "network").strip().lower()
        net_pressure = float(flow_diag.get("net_pressure", 0.0) or 0.0)
        vision_pressure = float(flow_diag.get("vision_pressure", 0.0) or 0.0)
        rho_pressure = max(net_pressure, vision_pressure)
        rho_pressure_name = "network" if net_pressure >= vision_pressure else "vision"
        rho_delta = 0.0
        rho_cutoff = False
        if rho_mode == "queue" and q_err > self.cfg.q_deadband:
            rho_delta = -self.cfg.mu_rho * q_err           # shrink
            rho_pressure_name = "legacy_queue"
            rho_cutoff = True
        elif rho_mode not in ("off", "fixed", "none"):
            rho_deadband = max(self.cfg.net_deadband, self.cfg.edge_deadband)
            if rho_pressure > 1.0 + rho_deadband:
                rho_delta = -self.cfg.mu_rho * (rho_pressure - 1.0)
                rho_cutoff = True

        # Asymmetric recovery (active when below target for long enough).
        q_under = (q_err < -self.cfg.climb_back_q_slack)
        kv_under = (kv_err < -self.cfg.climb_back_kv_slack)
        self._q_under_streak = self._q_under_streak + 1 if q_under else 0
        self._kv_under_streak = self._kv_under_streak + 1 if kv_under else 0
        rho_under = (rho_mode not in ("off", "fixed", "none")) and (rho_pressure < 1.0 - max(self.cfg.net_deadband, self.cfg.edge_deadband))
        self._rho_under_streak = self._rho_under_streak + 1 if rho_under else 0
        climb_back_triggered = False
        if self.cfg.climb_back_enabled:
            if (
                rho_under
                and rho_delta == 0.0
                and self._rho_under_streak >= self.cfg.climb_back_ticks_required
            ):
                rho_delta = self.cfg.climb_back_step_rho   # grow
                climb_back_triggered = True

        # Legacy emergency preempt brake. In v10 network-rho mode, cloud
        # preemptions are handled by alpha/admission/emergency-finalize, not by
        # dropping edge frames.
        if rho_mode == "queue" and snap.preemption_rate >= self.cfg.preempt_rate_panic:
            rho_delta -= self.cfg.preempt_extra_step
            rho_cutoff = True

        weights = self._compute_stream_weights()

        now = time.time()
        for stream_id, st in list(self._streams.items()):
            if now - st.last_update_wall < self.cfg.min_update_interval_s:
                continue
            # Weight cutoffs (overload response) by stream load; keep
            # climb-back uniform so every stream recovers together.
            if climb_back_triggered and rho_delta >= 0:
                w = 1.0
            else:
                w = weights.get(stream_id, 1.0)
            rho_hi = self.cfg.rho_hi
            if rho_mode not in ("queue", "legacy"):
                rho_hi = min(rho_hi, st.rho_content)
            if self.rho_limit_lookup is not None:
                try:
                    rho_limit = self.rho_limit_lookup(stream_id)
                except Exception:
                    rho_limit = None
                if rho_limit is not None:
                    rho_hi = min(rho_hi, float(rho_limit))
            new_rho = _clamp(st.rho + rho_delta * w, self.cfg.rho_lo, rho_hi)
            new_alpha, alpha_init, alpha_init_version, eta_press = self._next_alpha_for_stream(
                stream_id,
                snap,
                flow,
            )
            rho_changed = abs(new_rho - st.rho) > 1e-4
            alpha_changed = abs(new_alpha - st.alpha) > 1e-4
            alpha_init_changed = alpha_init_version != st.alpha_init_version
            if not (rho_changed or alpha_changed or alpha_init_changed):
                continue
            if climb_back_triggered:
                tag = f"{rho_pressure_name}_recovery"
            elif rho_cutoff:
                tag = f"{rho_pressure_name}_cutoff"
            else:
                tag = "alpha_update"
            reason = (
                f"{tag} Q={snap.num_requests_waiting:.1f}(err={q_err:+.1f}) "
                f"KV={snap.kv_cache_usage_perc:.2f}(err={kv_err:+.2f}) "
                f"preempt/s={snap.preemption_rate:.2f} w={w:.2f} "
                f"netP={flow_diag.get('net_pressure', 0.0):.2f} "
                f"visionP={flow_diag.get('vision_pressure', 0.0):.2f} "
                f"kvRateP={flow_diag.get('kv_rate_pressure', 0.0):.2f} "
                f"futureP={flow_diag.get('future_pressure', 0.0):.2f} "
                f"alpha_init={alpha_init:.3f} eta={eta_press:.3f} V={lyapunov:.3f}"
            )
            push_msg = {
                "kind": "rho_update",
                "rho": new_rho,
                "alpha": new_alpha,
                "alpha_init": alpha_init,
                "eta_press": eta_press,
                "reason": reason,
            }
            if self._send_fn is not None:
                try:
                    await self._send_fn(stream_id, push_msg)
                except Exception as e:
                    log.debug("send rho_update to %s failed: %s", stream_id, e)
                    continue
            st.rho = new_rho
            st.alpha = new_alpha
            st.alpha_init = alpha_init
            st.alpha_init_version = alpha_init_version
            st.eta_press = eta_press
            st.last_update_wall = now
            st.last_push_reason = reason
            self._last_push_per_stream[stream_id] = push_msg
            log.info(
                "stream=%s rho %.3f α %.3f (%s)",
                stream_id,
                new_rho,
                new_alpha,
                reason,
            )

        setattr(self._snapshot, "alpha_used", {s: st.alpha for s, st in self._streams.items()})
        setattr(self._snapshot, "alpha_init", {s: st.alpha_init for s, st in self._streams.items()})
        setattr(self._snapshot, "eta_press", self._compute_eta_press(snap, flow, self.cfg))
        setattr(self._snapshot, "token_flow", flow_diag)

        self._write_log_line(snap, q_err, kv_err, lyapunov, weights, flow_diag)

        # Heartbeat every ~10 ticks so operators see liveness even when idle.
        self._ticks_since_heartbeat += 1
        if self._ticks_since_heartbeat >= 20 and self._streams:
            self._ticks_since_heartbeat = 0
            log.info(
                "tick Q=%.1f KV=%.2f running=%.0f streams=%d V=%.3f q_under=%d kv_under=%d",
                snap.num_requests_waiting,
                snap.kv_cache_usage_perc,
                snap.num_requests_running,
                len(self._streams),
                lyapunov,
                self._q_under_streak,
                self._kv_under_streak,
            )

    # -------- diagnostics log --------

    def _write_log_line(
        self,
        snap: VllmSnapshot,
        q_err: float,
        kv_err: float,
        V: float,
        weights: Dict[str, float],
        flow_diag: Dict[str, Any],
    ) -> None:
        if self.log_path is None:
            return
        try:
            line = {
                "t": snap.at_wall,
                "Q": snap.num_requests_waiting,
                "Q_err": q_err,
                "KV": snap.kv_cache_usage_perc,
                "KV_err": kv_err,
                "running": snap.num_requests_running,
                "prompt_rate": snap.prompt_token_rate,
                "gen_rate": snap.generation_token_rate,
                "preempt_rate": snap.preemption_rate,
                "V": V,
                "q_under_streak": self._q_under_streak,
                "kv_under_streak": self._kv_under_streak,
                "rho_under_streak": self._rho_under_streak,
                "flow": flow_diag,
                "streams": {
                    s: {
                        "rho": st.rho,
                        "rho_content": st.rho_content,
                        "alpha": st.alpha,
                        "alpha_init": st.alpha_init,
                        "alpha_init_version": st.alpha_init_version,
                        "eta_press": st.eta_press,
                        "w": weights.get(s, 1.0),
                    }
                    for s, st in self._streams.items()
                },
            }
            with self.log_path.open("a") as f:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
        except Exception as e:  # noqa: BLE001
            log.debug("controller log write failed: %s", e)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _flow_diagnostics(
    snap: Optional[VllmSnapshot],
    flow: Optional[Any],
    cfg: ControllerConfig,
) -> Dict[str, Any]:
    lambda_edge = float(getattr(flow, "lambda_edge_total", 0.0) or 0.0)
    lambda_kv = float(getattr(flow, "lambda_kv_total", 0.0) or 0.0)
    lambda_free = float(getattr(flow, "lambda_free_total", 0.0) or 0.0)
    live_kv_tokens = float(getattr(flow, "live_kv_tokens_total", 0.0) or 0.0)
    d_k_dt = float(getattr(flow, "d_k_dt_total", 0.0) or 0.0)
    lambda_net_recv_bytes = float(getattr(flow, "lambda_net_recv_bytes_total", 0.0) or 0.0)
    lambda_net_offer_bytes = float(getattr(flow, "lambda_net_offer_bytes_total", 0.0) or 0.0)
    lambda_net_full_bytes = float(getattr(flow, "lambda_net_full_bytes_total", 0.0) or 0.0)
    net_backlog_bytes = float(getattr(flow, "net_backlog_bytes_total", 0.0) or 0.0)
    net_send_wait_ms = float(getattr(flow, "net_send_wait_ms_max", 0.0) or 0.0)

    net_pressure = 0.0
    net_util_pressure = 0.0
    net_backlog_pressure = 0.0
    net_send_wait_pressure = 0.0
    net_offer_gap_pressure = 0.0
    if cfg.net_capacity_bytes_s > 0:
        stable_bytes_s = max(1e-6, cfg.net_capacity_bytes_s * cfg.net_target_util)
        offered_or_recv = max(lambda_net_offer_bytes, lambda_net_recv_bytes)
        net_util_pressure = offered_or_recv / stable_bytes_s
        backlog_target = stable_bytes_s * max(1e-3, cfg.net_backlog_target_s)
        net_backlog_pressure = net_backlog_bytes / max(1e-6, backlog_target)
    if cfg.net_send_wait_target_ms > 0:
        net_send_wait_pressure = net_send_wait_ms / max(1e-6, cfg.net_send_wait_target_ms)
    if lambda_net_offer_bytes > lambda_net_recv_bytes and (
        net_backlog_bytes > 0 or net_send_wait_pressure > 0.5
    ):
        net_offer_gap_pressure = lambda_net_offer_bytes / max(1.0, lambda_net_recv_bytes)
    net_pressure = max(
        net_util_pressure,
        net_backlog_pressure,
        net_send_wait_pressure,
        net_offer_gap_pressure,
    )

    vision_pressure = 0.0
    if cfg.edge_capacity_tokens_s > 0:
        vision_pressure = lambda_edge / max(1e-6, cfg.edge_capacity_tokens_s * cfg.edge_target_util)

    prompt_rate = float(getattr(snap, "prompt_token_rate", 0.0) or 0.0) if snap is not None else 0.0
    mu_prefill = cfg.prefill_capacity_tokens_s if cfg.prefill_capacity_tokens_s > 0 else prompt_rate
    kv_rate_pressure = lambda_kv / mu_prefill if mu_prefill > 1e-6 else 0.0

    k_cap = int(getattr(snap, "kv_total_tokens", 0) or 0) if snap is not None else 0
    kv_water = float(getattr(snap, "kv_cache_usage_perc", 0.0) or 0.0) if snap is not None else 0.0
    k_used = kv_water * float(k_cap) if k_cap > 0 else live_kv_tokens
    projected_growth = max(0.0, lambda_kv - lambda_free) * max(0.0, cfg.kv_horizon_s)
    k_future = k_used + projected_growth
    margin = float(k_cap) * cfg.kv_margin_ratio if k_cap > 0 else 0.0
    future_limit = max(1.0, float(k_cap) - margin) if k_cap > 0 else 0.0
    future_pressure = k_future / future_limit if future_limit > 0 else 0.0

    state = "normal"
    if kv_water >= 0.95 or future_pressure >= 1.0:
        state = "critical"
    elif net_pressure > 1.0:
        state = "network_pressure"
    elif vision_pressure > 1.0:
        state = "vision_pressure"
    elif kv_water >= 0.88 or future_pressure >= 0.9:
        state = "kv_pressure"
    elif kv_rate_pressure > 1.0:
        state = "cloud_pressure"

    raw = flow.as_dict() if hasattr(flow, "as_dict") else None
    return {
        "state": state,
        "lambda_edge": lambda_edge,
        "lambda_kv": lambda_kv,
        "lambda_free": lambda_free,
        "d_k_dt": d_k_dt,
        "live_kv_tokens": live_kv_tokens,
        "lambda_net_recv_bytes": lambda_net_recv_bytes,
        "lambda_net_offer_bytes": lambda_net_offer_bytes,
        "lambda_net_full_bytes": lambda_net_full_bytes,
        "net_backlog_bytes": net_backlog_bytes,
        "net_send_wait_ms": net_send_wait_ms,
        "net_capacity_bytes_s": cfg.net_capacity_bytes_s,
        "net_pressure": net_pressure,
        "net_util_pressure": net_util_pressure,
        "net_backlog_pressure": net_backlog_pressure,
        "net_send_wait_pressure": net_send_wait_pressure,
        "net_offer_gap_pressure": net_offer_gap_pressure,
        "edge_capacity_tokens_s": cfg.edge_capacity_tokens_s,
        "edge_pressure": vision_pressure,  # backward-compatible alias
        "vision_pressure": vision_pressure,
        "mu_prefill_tokens_s": mu_prefill,
        "kv_rate_pressure": kv_rate_pressure,
        "k_used_tokens": k_used,
        "k_future_tokens": k_future,
        "k_cap_tokens": k_cap,
        "kv_margin_tokens": margin,
        "future_pressure": future_pressure,
        "raw": raw,
    }


def _compute_eta_press(
    snap: Optional[VllmSnapshot],
    flow: Optional[Any] = None,
    cfg: Optional[ControllerConfig] = None,
) -> float:
    """KV-pressure modulation in (0, 1]."""
    if snap is None or getattr(snap, "kv_cache_usage_perc", None) is None:
        return 1.0
    floor = 0.5 if cfg is None else max(0.01, min(1.0, cfg.eta_floor))
    kv = float(snap.kv_cache_usage_perc)
    theta_warn = 0.78
    theta_panic = 0.95
    if kv <= theta_warn:
        eta_kv = 1.0
    elif kv >= theta_panic:
        eta_kv = floor
    else:
        t = (kv - theta_warn) / (theta_panic - theta_warn)
        s = t * t * (3 - 2 * t)
        eta_kv = 1.0 - s * (1.0 - floor)
    if cfg is not None:
        preempt = float(getattr(snap, "preemption_rate", 0.0) or 0.0)
        if preempt >= cfg.preempt_rate_panic:
            eta_kv = min(eta_kv, floor)

    eta_flow = 1.0
    if cfg is not None and flow is not None:
        diag = _flow_diagnostics(snap, flow, cfg)
        for pressure_name in ("kv_rate_pressure", "future_pressure"):
            pressure = float(diag.get(pressure_name, 0.0) or 0.0)
            if pressure > 1.0:
                eta_flow = min(eta_flow, max(floor, 1.0 / pressure))
    return max(floor, min(1.0, eta_kv, eta_flow))


def controller_config_from_env() -> ControllerConfig:
    def _f(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, default))
        except Exception:
            return default

    def _i(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, default))
        except Exception:
            return default

    def _b(key: str, default: bool) -> bool:
        v = os.environ.get(key)
        if v is None:
            return default
        return v not in ("0", "", "false", "False", "no", "NO")

    def _s(key: str, default: str) -> str:
        v = os.environ.get(key)
        return default if v in (None, "") else str(v)

    return ControllerConfig(
        tick_s=_f("BAVA_TICK_S", 0.5),
        q_target=_f("BAVA_Q_TARGET", 2.0),
        kv_target=_f("BAVA_KV_TARGET", 0.6),
        mu_rho=_f("BAVA_MU_RHO", 0.05),
        mu_alpha=_f("BAVA_MU_ALPHA", 0.05),
        q_deadband=_f("BAVA_Q_DEADBAND", 1.0),
        kv_deadband=_f("BAVA_KV_DEADBAND", 0.05),
        min_update_interval_s=_f("BAVA_MIN_UPDATE_S", 1.0),
        rho_lo=_f("BAVA_RHO_LO", 0.05),
        rho_hi=_f("BAVA_RHO_HI", 1.0),
        alpha_lo=max(0.3, _f("BAVA_ALPHA_LO", 0.3)),
        alpha_hi=_f("BAVA_ALPHA_HI", 1.0),
        preempt_rate_panic=_f("BAVA_PREEMPT_PANIC", 0.5),
        preempt_extra_step=_f("BAVA_PREEMPT_STEP", 0.1),
        climb_back_enabled=_b("BAVA_CLIMB_BACK", True),
        climb_back_ticks_required=_i("BAVA_CLIMB_BACK_TICKS", 6),
        climb_back_step_rho=_f("BAVA_CLIMB_BACK_STEP_RHO", 0.02),
        climb_back_q_slack=_f("BAVA_CLIMB_BACK_Q_SLACK", 0.5),
        climb_back_kv_slack=_f("BAVA_CLIMB_BACK_KV_SLACK", 0.1),
        per_stream_weighting=_b("BAVA_PER_STREAM_WEIGHTING", True),
        per_stream_weight_min=_f("BAVA_STREAM_WEIGHT_MIN", 0.5),
        per_stream_weight_max=_f("BAVA_STREAM_WEIGHT_MAX", 2.0),
        rho_control_mode=_s("BAVA_RHO_CONTROL_MODE", "network"),
        net_capacity_bytes_s=_f("BAVA_NET_CAP_BYTES_S", 0.0),
        net_target_util=_f("BAVA_NET_TARGET_UTIL", 0.90),
        net_deadband=_f("BAVA_NET_DEADBAND", 0.05),
        net_backlog_target_s=_f("BAVA_NET_BACKLOG_TARGET_S", 2.0),
        net_send_wait_target_ms=_f("BAVA_NET_SEND_WAIT_TARGET_MS", 250.0),
        edge_capacity_tokens_s=_f("BAVA_EDGE_CAP_TOKENS_S", 0.0),
        edge_target_util=_f("BAVA_EDGE_TARGET_UTIL", 0.90),
        edge_deadband=_f("BAVA_EDGE_DEADBAND", 0.05),
        prefill_capacity_tokens_s=_f("BAVA_PREFILL_CAP_TOKENS_S", 0.0),
        kv_horizon_s=_f("BAVA_KV_HORIZON_S", 10.0),
        kv_margin_ratio=_f("BAVA_KV_MARGIN_RATIO", 0.10),
        eta_floor=_f("BAVA_ETA_FLOOR", 0.5),
    )
