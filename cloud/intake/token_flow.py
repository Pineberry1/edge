"""Runtime token-flow accounting for BAVA intake.

This module keeps the v10 token-flow quantities concrete:

* lambda_edge: visual-token arrival rate after edge rho filtering, before alpha.
* lambda_kv: visual-token injection rate after cloud alpha execution.
* lambda_free: estimated KV release rate at decision finalization / abort.
* live_kv_tokens: intake-side estimate of KV tokens currently held by a stream.
* lambda_net_recv_bytes: cloud-observed byte arrival rate after edge rho.
* lambda_net_offer_bytes: edge-offered byte rate after edge rho, when reported.
* net_backlog_bytes / net_send_wait_ms: edge-side network queue pressure.

All numbers are estimates. They are deliberately cheap and stable enough for
controller/admission decisions; vLLM metrics remain the source of truth for
global KV waterline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


PATCH_STRIDE = 28


def tokens_for_image_shape(height: int, width: int, patch_stride: int = PATCH_STRIDE) -> int:
    """Approximate Qwen3-VL visual token count for one processed image."""
    h = max(1, int(height))
    w = max(1, int(width))
    stride = max(1, int(patch_stride))
    return max(1, math.ceil(h / stride) * math.ceil(w / stride))


def _ewma(previous: float, sample: float, alpha: float) -> float:
    sample = float(sample)
    if previous == 0.0:
        return sample
    a = max(0.0, min(1.0, float(alpha)))
    return (1.0 - a) * previous + a * sample


@dataclass(frozen=True)
class StreamFlowSnapshot:
    stream_id: str
    lambda_edge: float
    lambda_kv: float
    lambda_free: float
    d_k_dt: float
    live_kv_tokens: int
    lambda_net_recv_bytes: float
    lambda_net_offer_bytes: float
    lambda_net_full_bytes: float
    net_backlog_bytes: int
    net_send_wait_ms: float
    last_raw_tokens: int
    last_kv_tokens: int
    last_recv_bytes: int
    last_offer_bytes: int
    last_full_bytes: int
    cumulative_edge_tokens: int
    cumulative_kv_tokens: int
    cumulative_released_tokens: int
    cumulative_recv_bytes: int
    cumulative_offer_bytes: int
    cumulative_full_bytes: int
    window_seconds: float
    decision_window_seconds: float
    last_update_wall: float

    def as_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "lambda_edge": self.lambda_edge,
            "lambda_kv": self.lambda_kv,
            "lambda_free": self.lambda_free,
            "d_k_dt": self.d_k_dt,
            "live_kv_tokens": self.live_kv_tokens,
            "lambda_net_recv_bytes": self.lambda_net_recv_bytes,
            "lambda_net_offer_bytes": self.lambda_net_offer_bytes,
            "lambda_net_full_bytes": self.lambda_net_full_bytes,
            "net_backlog_bytes": self.net_backlog_bytes,
            "net_send_wait_ms": self.net_send_wait_ms,
            "last_raw_tokens": self.last_raw_tokens,
            "last_kv_tokens": self.last_kv_tokens,
            "last_recv_bytes": self.last_recv_bytes,
            "last_offer_bytes": self.last_offer_bytes,
            "last_full_bytes": self.last_full_bytes,
            "cumulative_edge_tokens": self.cumulative_edge_tokens,
            "cumulative_kv_tokens": self.cumulative_kv_tokens,
            "cumulative_released_tokens": self.cumulative_released_tokens,
            "cumulative_recv_bytes": self.cumulative_recv_bytes,
            "cumulative_offer_bytes": self.cumulative_offer_bytes,
            "cumulative_full_bytes": self.cumulative_full_bytes,
            "window_seconds": self.window_seconds,
            "decision_window_seconds": self.decision_window_seconds,
            "last_update_wall": self.last_update_wall,
        }


@dataclass(frozen=True)
class FlowSnapshot:
    lambda_edge_total: float
    lambda_kv_total: float
    lambda_free_total: float
    d_k_dt_total: float
    live_kv_tokens_total: int
    lambda_net_recv_bytes_total: float
    lambda_net_offer_bytes_total: float
    lambda_net_full_bytes_total: float
    net_backlog_bytes_total: int
    net_send_wait_ms_max: float
    cumulative_recv_bytes_total: int
    cumulative_offer_bytes_total: int
    cumulative_full_bytes_total: int
    streams: dict[str, StreamFlowSnapshot]

    def as_dict(self) -> dict:
        return {
            "lambda_edge_total": self.lambda_edge_total,
            "lambda_kv_total": self.lambda_kv_total,
            "lambda_free_total": self.lambda_free_total,
            "d_k_dt_total": self.d_k_dt_total,
            "live_kv_tokens_total": self.live_kv_tokens_total,
            "lambda_net_recv_bytes_total": self.lambda_net_recv_bytes_total,
            "lambda_net_offer_bytes_total": self.lambda_net_offer_bytes_total,
            "lambda_net_full_bytes_total": self.lambda_net_full_bytes_total,
            "net_backlog_bytes_total": self.net_backlog_bytes_total,
            "net_send_wait_ms_max": self.net_send_wait_ms_max,
            "cumulative_recv_bytes_total": self.cumulative_recv_bytes_total,
            "cumulative_offer_bytes_total": self.cumulative_offer_bytes_total,
            "cumulative_full_bytes_total": self.cumulative_full_bytes_total,
            "streams": {sid: snap.as_dict() for sid, snap in self.streams.items()},
        }


class StreamFlowTracker:
    """Per-stream online estimator updated from intake events."""

    def __init__(
        self,
        stream_id: str,
        *,
        window_seconds: float = 1.0,
        decision_window_seconds: Optional[float] = None,
        ema_alpha: float = 0.2,
    ) -> None:
        self.stream_id = stream_id
        self.window_seconds = max(1e-3, float(window_seconds or 1.0))
        self.decision_window_seconds = max(
            self.window_seconds,
            float(decision_window_seconds or window_seconds or 1.0),
        )
        self.ema_alpha = max(0.0, min(1.0, float(ema_alpha)))

        self.lambda_edge = 0.0
        self.lambda_kv = 0.0
        self.lambda_free = 0.0
        self.d_k_dt = 0.0
        self.live_kv_tokens = 0
        self.lambda_net_recv_bytes = 0.0
        self.lambda_net_offer_bytes = 0.0
        self.lambda_net_full_bytes = 0.0
        self.net_backlog_bytes = 0
        self.net_send_wait_ms = 0.0
        self.last_raw_tokens = 0
        self.last_kv_tokens = 0
        self.last_recv_bytes = 0
        self.last_offer_bytes = 0
        self.last_full_bytes = 0
        self.cumulative_edge_tokens = 0
        self.cumulative_kv_tokens = 0
        self.cumulative_released_tokens = 0
        self.cumulative_recv_bytes = 0
        self.cumulative_offer_bytes = 0
        self.cumulative_full_bytes = 0
        self.last_update_wall = time.time()

    def _update_derivative(self, new_live_tokens: int, now: float) -> None:
        dt = max(1e-3, now - self.last_update_wall)
        self.d_k_dt = _ewma(
            self.d_k_dt,
            (float(new_live_tokens) - float(self.live_kv_tokens)) / dt,
            self.ema_alpha,
        )
        self.last_update_wall = now

    def observe_request_open(self, overhead_tokens: int, *, at_wall: Optional[float] = None) -> None:
        tokens = max(0, int(overhead_tokens))
        if tokens <= 0:
            return
        now = time.time() if at_wall is None else float(at_wall)
        new_live = self.live_kv_tokens + tokens
        self._update_derivative(new_live, now)
        self.live_kv_tokens = new_live
        self.cumulative_kv_tokens += tokens

    def observe_edge_window(
        self,
        *,
        raw_tokens: int,
        window_seconds: Optional[float] = None,
        at_wall: Optional[float] = None,
    ) -> None:
        raw = max(0, int(raw_tokens))
        seconds = max(1e-3, float(window_seconds or self.window_seconds))
        now = time.time() if at_wall is None else float(at_wall)
        self.lambda_edge = max(0.0, _ewma(self.lambda_edge, raw / seconds, self.ema_alpha))
        self.last_raw_tokens = raw
        self.cumulative_edge_tokens += raw
        self.last_update_wall = now

    def observe_network_window(
        self,
        *,
        recv_bytes: int,
        offer_bytes: Optional[int] = None,
        full_bytes: Optional[int] = None,
        edge_queue_bytes: Optional[int] = None,
        send_wait_ms: Optional[float] = None,
        window_seconds: Optional[float] = None,
        at_wall: Optional[float] = None,
    ) -> None:
        received = max(0, int(recv_bytes))
        offered = received if offer_bytes is None else max(0, int(offer_bytes))
        full = offered if full_bytes is None else max(0, int(full_bytes))
        seconds = max(1e-3, float(window_seconds or self.window_seconds))
        now = time.time() if at_wall is None else float(at_wall)
        self.lambda_net_recv_bytes = max(
            0.0,
            _ewma(self.lambda_net_recv_bytes, received / seconds, self.ema_alpha),
        )
        self.lambda_net_offer_bytes = max(
            0.0,
            _ewma(self.lambda_net_offer_bytes, offered / seconds, self.ema_alpha),
        )
        self.lambda_net_full_bytes = max(
            0.0,
            _ewma(self.lambda_net_full_bytes, full / seconds, self.ema_alpha),
        )
        if edge_queue_bytes is not None:
            self.net_backlog_bytes = max(0, int(edge_queue_bytes))
        if send_wait_ms is not None:
            self.net_send_wait_ms = max(
                0.0,
                _ewma(self.net_send_wait_ms, float(send_wait_ms), self.ema_alpha),
            )
        self.last_recv_bytes = received
        self.last_offer_bytes = offered
        self.last_full_bytes = full
        self.cumulative_recv_bytes += received
        self.cumulative_offer_bytes += offered
        self.cumulative_full_bytes += full
        self.last_update_wall = now

    def observe_kv_window(
        self,
        *,
        kv_tokens: int,
        window_seconds: Optional[float] = None,
        at_wall: Optional[float] = None,
    ) -> None:
        kv = max(0, int(kv_tokens))
        seconds = max(1e-3, float(window_seconds or self.window_seconds))
        now = time.time() if at_wall is None else float(at_wall)
        self.lambda_kv = max(0.0, _ewma(self.lambda_kv, kv / seconds, self.ema_alpha))
        new_live = self.live_kv_tokens + kv
        self._update_derivative(new_live, now)
        self.live_kv_tokens = new_live
        self.last_kv_tokens = kv
        self.cumulative_kv_tokens += kv

    def observe_release(
        self,
        *,
        decision_duration_s: Optional[float] = None,
        at_wall: Optional[float] = None,
    ) -> int:
        released = max(0, int(self.live_kv_tokens))
        now = time.time() if at_wall is None else float(at_wall)
        seconds = max(1e-3, float(decision_duration_s or self.decision_window_seconds))
        self.lambda_free = max(0.0, _ewma(self.lambda_free, released / seconds, self.ema_alpha))
        self._update_derivative(0, now)
        self.live_kv_tokens = 0
        self.cumulative_released_tokens += released
        return released

    def snapshot(self) -> StreamFlowSnapshot:
        return StreamFlowSnapshot(
            stream_id=self.stream_id,
            lambda_edge=float(self.lambda_edge),
            lambda_kv=float(self.lambda_kv),
            lambda_free=float(self.lambda_free),
            d_k_dt=float(self.d_k_dt),
            live_kv_tokens=int(self.live_kv_tokens),
            lambda_net_recv_bytes=float(self.lambda_net_recv_bytes),
            lambda_net_offer_bytes=float(self.lambda_net_offer_bytes),
            lambda_net_full_bytes=float(self.lambda_net_full_bytes),
            net_backlog_bytes=int(self.net_backlog_bytes),
            net_send_wait_ms=float(self.net_send_wait_ms),
            last_raw_tokens=int(self.last_raw_tokens),
            last_kv_tokens=int(self.last_kv_tokens),
            last_recv_bytes=int(self.last_recv_bytes),
            last_offer_bytes=int(self.last_offer_bytes),
            last_full_bytes=int(self.last_full_bytes),
            cumulative_edge_tokens=int(self.cumulative_edge_tokens),
            cumulative_kv_tokens=int(self.cumulative_kv_tokens),
            cumulative_released_tokens=int(self.cumulative_released_tokens),
            cumulative_recv_bytes=int(self.cumulative_recv_bytes),
            cumulative_offer_bytes=int(self.cumulative_offer_bytes),
            cumulative_full_bytes=int(self.cumulative_full_bytes),
            window_seconds=float(self.window_seconds),
            decision_window_seconds=float(self.decision_window_seconds),
            last_update_wall=float(self.last_update_wall),
        )


def aggregate_flow(streams: dict[str, StreamFlowSnapshot]) -> FlowSnapshot:
    return FlowSnapshot(
        lambda_edge_total=sum(s.lambda_edge for s in streams.values()),
        lambda_kv_total=sum(s.lambda_kv for s in streams.values()),
        lambda_free_total=sum(s.lambda_free for s in streams.values()),
        d_k_dt_total=sum(s.d_k_dt for s in streams.values()),
        live_kv_tokens_total=sum(s.live_kv_tokens for s in streams.values()),
        lambda_net_recv_bytes_total=sum(s.lambda_net_recv_bytes for s in streams.values()),
        lambda_net_offer_bytes_total=sum(s.lambda_net_offer_bytes for s in streams.values()),
        lambda_net_full_bytes_total=sum(s.lambda_net_full_bytes for s in streams.values()),
        net_backlog_bytes_total=sum(s.net_backlog_bytes for s in streams.values()),
        net_send_wait_ms_max=max((s.net_send_wait_ms for s in streams.values()), default=0.0),
        cumulative_recv_bytes_total=sum(s.cumulative_recv_bytes for s in streams.values()),
        cumulative_offer_bytes_total=sum(s.cumulative_offer_bytes for s in streams.values()),
        cumulative_full_bytes_total=sum(s.cumulative_full_bytes for s in streams.values()),
        streams=dict(streams),
    )
