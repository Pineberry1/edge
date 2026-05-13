"""KV-pressure admission decisions for intake online-prefill sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


SnapshotFn = Callable[[], Optional[Any]]
FlowSnapshotFn = Callable[[], Optional[Any]]


@dataclass(frozen=True)
class AdmissionConfig:
    kv_warn: float = 0.78
    kv_high: float = 0.88
    kv_panic: float = 0.95
    preempt_rate_panic: float = 0.5
    horizon_s: float = 10.0
    kv_margin_ratio: float = 0.10
    kv_margin_tokens: int = 0


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def admission_config_from_env() -> AdmissionConfig:
    return AdmissionConfig(
        kv_warn=_float_env("BAVA_ADMISSION_KV_WARN", AdmissionConfig.kv_warn),
        kv_high=_float_env("BAVA_ADMISSION_KV_HIGH", AdmissionConfig.kv_high),
        kv_panic=_float_env("BAVA_ADMISSION_KV_PANIC", AdmissionConfig.kv_panic),
        preempt_rate_panic=_float_env(
            "BAVA_ADMISSION_PREEMPT_PANIC",
            AdmissionConfig.preempt_rate_panic,
        ),
        horizon_s=_float_env("BAVA_ADMISSION_HORIZON_S", AdmissionConfig.horizon_s),
        kv_margin_ratio=_float_env(
            "BAVA_ADMISSION_KV_MARGIN_RATIO",
            AdmissionConfig.kv_margin_ratio,
        ),
        kv_margin_tokens=int(
            _float_env("BAVA_ADMISSION_KV_MARGIN_TOKENS", AdmissionConfig.kv_margin_tokens)
        ),
    )


class AdmissionGate:
    """Fail-open pressure gate backed by the controller's latest snapshot."""

    def __init__(
        self,
        snapshot_fn: SnapshotFn,
        cfg: AdmissionConfig,
        flow_snapshot_fn: Optional[FlowSnapshotFn] = None,
    ) -> None:
        self.snapshot_fn = snapshot_fn
        self.cfg = cfg
        self.flow_snapshot_fn = flow_snapshot_fn

    def _snapshot(self) -> Optional[Any]:
        try:
            return self.snapshot_fn()
        except Exception:
            return None

    @staticmethod
    def _field(snapshot: Any, name: str) -> float:
        try:
            return float(getattr(snapshot, name, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _flow_snapshot(self) -> Optional[Any]:
        if self.flow_snapshot_fn is None:
            return None
        try:
            return self.flow_snapshot_fn()
        except Exception:
            return None

    def _horizon_allows(self, snap: Any, delta_tokens: int) -> tuple[bool, str]:
        k_cap_raw = getattr(snap, "kv_total_tokens", None)
        try:
            k_cap = int(k_cap_raw or 0)
        except Exception:
            k_cap = 0
        if k_cap <= 0:
            return True, "no_kv_cap"
        kv = self._field(snap, "kv_cache_usage_perc")
        k_used = kv * float(k_cap)
        flow = self._flow_snapshot()
        lambda_kv = float(getattr(flow, "lambda_kv_total", 0.0) or 0.0)
        lambda_free = float(getattr(flow, "lambda_free_total", 0.0) or 0.0)
        projected_growth = max(0.0, lambda_kv - lambda_free) * max(0.0, self.cfg.horizon_s)
        margin = max(float(self.cfg.kv_margin_tokens), float(k_cap) * self.cfg.kv_margin_ratio)
        limit = float(k_cap) - margin
        k_future = k_used + projected_growth + max(0, int(delta_tokens))
        if k_future > limit:
            return False, "horizon_kv"
        return True, "horizon_admit"

    def decide_create(self, delta_tokens: int = 0) -> tuple[bool, str]:
        snap = self._snapshot()
        if snap is None:
            return True, "no_snapshot"
        kv = self._field(snap, "kv_cache_usage_perc")
        if kv >= self.cfg.kv_high:
            return False, "kv_high"
        preempt_rate = self._field(snap, "preemption_rate")
        if preempt_rate >= self.cfg.preempt_rate_panic:
            return False, "preempt_panic"
        ok, reason = self._horizon_allows(snap, delta_tokens)
        if not ok:
            return False, reason
        return True, "admit"

    def decide_append(self, delta_tokens: int = 0) -> tuple[bool, str]:
        snap = self._snapshot()
        if snap is None:
            return True, "no_snapshot"
        kv = self._field(snap, "kv_cache_usage_perc")
        if kv >= self.cfg.kv_panic:
            return False, "kv_panic"
        ok, reason = self._horizon_allows(snap, delta_tokens)
        if not ok:
            return False, reason
        return True, "admit"
