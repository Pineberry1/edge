from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("remote_intake_admission", ROOT / "admission.py")
admission = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


@pytest.mark.parametrize("kv", [0.5, 0.85, 0.92, 0.97])
@pytest.mark.parametrize("preempt_rate", [0.0, 0.7])
def test_admission_threshold_matrix(kv: float, preempt_rate: float) -> None:
    snap = SimpleNamespace(kv_cache_usage_perc=kv, preemption_rate=preempt_rate)
    gate = admission.AdmissionGate(lambda: snap, admission.AdmissionConfig())

    create_ok, _ = gate.decide_create()
    append_ok, _ = gate.decide_append()

    assert create_ok == (not (kv >= 0.88 or preempt_rate >= 0.5))
    assert append_ok == (not (kv >= 0.95))


def test_admission_fails_open_without_snapshot() -> None:
    gate = admission.AdmissionGate(lambda: None, admission.AdmissionConfig())

    assert gate.decide_create() == (True, "no_snapshot")
    assert gate.decide_append() == (True, "no_snapshot")


def test_admission_denies_when_horizon_exceeds_kv_limit() -> None:
    snap = SimpleNamespace(
        kv_cache_usage_perc=0.5,
        preemption_rate=0.0,
        kv_total_tokens=1000,
    )
    flow = SimpleNamespace(lambda_kv_total=100.0, lambda_free_total=0.0)
    gate = admission.AdmissionGate(
        lambda: snap,
        admission.AdmissionConfig(horizon_s=10.0, kv_margin_ratio=0.1),
        flow_snapshot_fn=lambda: flow,
    )

    assert gate.decide_create() == (False, "horizon_kv")
    assert gate.decide_append() == (False, "horizon_kv")
