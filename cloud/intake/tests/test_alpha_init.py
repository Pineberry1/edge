from __future__ import annotations

import concurrent.futures
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
PKG = types.ModuleType("remote_intake")
PKG.__path__ = [str(ROOT)]
sys.modules.setdefault("remote_intake", PKG)


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alpha_init_state = _load_module("remote_intake.alpha_init_state", "alpha_init_state.py")
metrics_stub = types.ModuleType("remote_intake.metrics")
metrics_stub.VllmSnapshot = object
sys.modules["remote_intake.metrics"] = metrics_stub
controller = _load_module("remote_intake.controller", "controller.py")


def test_idempotent_update() -> None:
    state = alpha_init_state.AlphaInitState(default=1.0)

    state.update_from_response(0.7, window_id=3)
    assert state.get() == (0.7, 1)

    state.update_from_response(0.5, window_id=3)
    assert state.get() == (0.7, 1)


def test_clamp() -> None:
    state = alpha_init_state.AlphaInitState(default=1.0, alpha_min=0.3)

    state.update_from_response(0.05, window_id=1)
    assert state.get() == (0.3, 1)

    state.update_from_response(1.5, window_id=2)
    assert state.get() == (1.0, 2)


def test_none_no_op() -> None:
    state = alpha_init_state.AlphaInitState(default=0.8)

    state.update_from_response(None, window_id=1)

    assert state.get() == (0.8, 0)


def test_thread_safety() -> None:
    state = alpha_init_state.AlphaInitState(default=1.0)

    def update(i: int) -> tuple[float, int]:
        state.update_from_response(0.3 + 0.001 * i, window_id=i)
        return state.get()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        reads = list(ex.map(update, range(1, 200)))

    alpha, version = state.get()
    assert 0.3 <= alpha <= 1.0
    assert 1 <= version <= 199
    assert all(0.3 <= a <= 1.0 and 0 <= v <= 199 for a, v in reads)


def test_eta_press_warn() -> None:
    assert controller._compute_eta_press(SimpleNamespace(kv_cache_usage_perc=0.5)) == 1.0
    assert controller._compute_eta_press(SimpleNamespace(kv_cache_usage_perc=0.78)) == 1.0


def test_eta_press_panic() -> None:
    assert controller._compute_eta_press(SimpleNamespace(kv_cache_usage_perc=0.95)) == 0.5


def test_eta_press_monotone() -> None:
    values = [
        controller._compute_eta_press(SimpleNamespace(kv_cache_usage_perc=kv))
        for kv in (0.78, 0.80, 0.84, 0.88, 0.92, 0.95)
    ]

    assert values == sorted(values, reverse=True)


def test_alpha_used_composition() -> None:
    snap = SimpleNamespace(kv_cache_usage_perc=0.875)
    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(alpha_lo=0.3),
        alpha_init_lookup=lambda _sid: (0.6, 1),
    )

    alpha_used, alpha_init, version, eta = ctrl._next_alpha_for_stream("s0", snap)

    assert alpha_init == 0.6
    assert version == 1
    assert alpha_used == pytest.approx(0.6 * eta)


def test_alpha_used_floor() -> None:
    snap = SimpleNamespace(kv_cache_usage_perc=0.95)
    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(alpha_lo=0.3),
        alpha_init_lookup=lambda _sid: (0.4, 1),
    )

    alpha_used, _, _, eta = ctrl._next_alpha_for_stream("s0", snap)

    assert eta == 0.5
    assert alpha_used == 0.3


def test_flow_pressure_reduces_eta() -> None:
    snap = SimpleNamespace(kv_cache_usage_perc=0.5, prompt_token_rate=0.0)
    flow = SimpleNamespace(
        lambda_edge_total=0.0,
        lambda_kv_total=120.0,
        lambda_free_total=0.0,
        d_k_dt_total=120.0,
        live_kv_tokens_total=0,
    )
    cfg = controller.ControllerConfig(
        alpha_lo=0.3,
        prefill_capacity_tokens_s=50.0,
        eta_floor=0.4,
    )
    ctrl = controller.BavaController(
        scraper=object(),
        config=cfg,
        alpha_init_lookup=lambda _sid: (1.0, 1),
    )

    alpha_used, _, _, eta = ctrl._next_alpha_for_stream("s0", snap, flow)

    assert eta == pytest.approx(1.0 / (120.0 / 50.0))
    assert alpha_used == pytest.approx(eta)


def _snap(**overrides):
    data = {
        "at_wall": 1.0,
        "num_requests_waiting": 100.0,
        "num_requests_running": 0.0,
        "kv_cache_usage_perc": 0.1,
        "num_preemptions_total": 0.0,
        "prompt_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "kv_total_tokens": 10000,
        "prompt_token_rate": 0.0,
        "generation_token_rate": 0.0,
        "preemption_rate": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _flow(**overrides):
    data = {
        "lambda_edge_total": 0.0,
        "lambda_kv_total": 0.0,
        "lambda_free_total": 0.0,
        "d_k_dt_total": 0.0,
        "live_kv_tokens_total": 0,
        "lambda_net_recv_bytes_total": 0.0,
        "lambda_net_offer_bytes_total": 0.0,
        "lambda_net_full_bytes_total": 0.0,
        "net_backlog_bytes_total": 0,
        "net_send_wait_ms_max": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_network_mode_ignores_raw_vllm_waiting_for_rho() -> None:
    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(rho_control_mode="network"),
        flow_lookup=lambda: _flow(),
    )
    ctrl.track("s0", initial_rho=0.5, initial_alpha=1.0)

    asyncio.run(ctrl._on_snapshot(_snap(num_requests_waiting=100.0)))

    assert ctrl.state("s0").rho == pytest.approx(0.5)


def test_rho_limit_lookup_caps_recovery() -> None:
    sent = []

    async def send(stream_id, msg):
        sent.append((stream_id, msg))

    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(
            rho_control_mode="network",
            net_capacity_bytes_s=1000.0,
            net_deadband=0.0,
            climb_back_ticks_required=1,
            climb_back_step_rho=0.05,
            min_update_interval_s=0.0,
        ),
        flow_lookup=lambda: _flow(lambda_net_recv_bytes_total=0.0),
        rho_limit_lookup=lambda _sid: 0.4,
    )
    ctrl._send_fn = send
    ctrl.track("s0", initial_rho=0.5, initial_alpha=1.0)

    asyncio.run(ctrl._on_snapshot(_snap(num_requests_waiting=0.0)))

    assert ctrl.state("s0").rho == pytest.approx(0.4)
    assert sent[-1][1]["rho"] == pytest.approx(0.4)


def test_network_pressure_reduces_rho() -> None:
    sent = []

    async def send(stream_id, msg):
        sent.append((stream_id, msg))

    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(
            rho_control_mode="network",
            net_capacity_bytes_s=100.0,
            net_target_util=1.0,
            net_deadband=0.0,
            mu_rho=0.1,
        ),
        flow_lookup=lambda: _flow(lambda_net_recv_bytes_total=200.0),
    )
    ctrl._send_fn = send
    ctrl.track("s0", initial_rho=0.5, initial_alpha=1.0)

    asyncio.run(ctrl._on_snapshot(_snap(num_requests_waiting=0.0)))

    assert ctrl.state("s0").rho == pytest.approx(0.4)
    assert sent and "network_cutoff" in sent[0][1]["reason"]


def test_network_backlog_reduces_rho() -> None:
    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(
            rho_control_mode="network",
            net_capacity_bytes_s=1000.0,
            net_target_util=1.0,
            net_backlog_target_s=1.0,
            net_deadband=0.0,
            mu_rho=0.1,
        ),
        flow_lookup=lambda: _flow(
            lambda_net_offer_bytes_total=500.0,
            lambda_net_recv_bytes_total=450.0,
            net_backlog_bytes_total=2000,
        ),
    )
    ctrl.track("s0", initial_rho=0.5, initial_alpha=1.0)

    asyncio.run(ctrl._on_snapshot(_snap(num_requests_waiting=0.0)))

    assert ctrl.state("s0").rho == pytest.approx(0.4)


def test_kv_pressure_reduces_alpha_not_rho_in_network_mode() -> None:
    ctrl = controller.BavaController(
        scraper=object(),
        config=controller.ControllerConfig(
            rho_control_mode="network",
            prefill_capacity_tokens_s=100.0,
            eta_floor=0.5,
        ),
        flow_lookup=lambda: _flow(lambda_kv_total=250.0),
    )
    ctrl.track("s0", initial_rho=0.5, initial_alpha=1.0)

    asyncio.run(ctrl._on_snapshot(_snap(num_requests_waiting=100.0)))

    assert ctrl.state("s0").rho == pytest.approx(0.5)
    assert ctrl.state("s0").alpha == pytest.approx(0.5)
