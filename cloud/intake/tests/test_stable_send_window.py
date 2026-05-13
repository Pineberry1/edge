from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remote_intake_stable_send_window",
    ROOT / "stable_send_window.py",
)
stable_send_window = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = stable_send_window
SPEC.loader.exec_module(stable_send_window)


def test_stable_probe_increases_without_ef() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.5,
        increase_step=0.05,
        stable_results_required=2,
        probe_interval_s=0.0,
        kv_probe_max=0.9,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)

    asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=False,
            kv_usage=0.5,
            now=10.0,
        )
    )
    assert not sent

    event = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=False,
            kv_usage=0.5,
            now=11.0,
        )
    )
    assert event["action"] == "stable_probe"
    assert sent[-1][1]["rho"] == 0.55
    assert mgr.rho_limit_for_stream("s0") == 0.55


def test_ef_reduces_only_affected_engine() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.8,
        ef_reduce_factor=0.8,
        operating_margin=0.85,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)
    mgr.register_stream("s1", engine_index=1, rho_content=1.0)

    event = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            kv_usage=0.95,
            now=10.0,
        )
    )
    assert event["action"] == "ef_reduce"
    assert mgr.rho_limit_for_stream("s0") == 0.6400000000000001
    assert mgr.rho_limit_for_stream("s1") == 0.8
    assert [sid for sid, _ in sent] == ["s0"]


def test_operating_margin_blocks_fast_reprobe_after_ef() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.8,
        increase_step=0.05,
        stable_results_required=1,
        probe_interval_s=0.0,
        ef_cooldown_s=5.0,
        ef_reduce_factor=0.8,
        operating_margin=0.85,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)

    asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            kv_usage=0.95,
            now=10.0,
        )
    )
    before = mgr.rho_limit_for_stream("s0")
    event = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=False,
            kv_usage=0.5,
            now=12.0,
        )
    )
    assert event is None
    assert mgr.rho_limit_for_stream("s0") == before


def test_probe_does_not_raise_stream_below_old_limit() -> None:
    current = {"s0": 0.3}
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))
        current[stream_id] = msg["rho"]

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.5,
        increase_step=0.05,
        stable_results_required=1,
        probe_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(
        cfg,
        send_fn=send_fn,
        current_rho_fn=lambda sid: current.get(sid),
    )
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)

    event = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=False,
            kv_usage=0.5,
            now=20.0,
        )
    )

    assert event["action"] == "stable_probe"
    assert mgr.rho_limit_for_stream("s0") == 0.55
    assert sent == []


def test_ef_decision_is_deduplicated() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.8,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)

    first = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            decision_id=7,
            now=10.0,
        )
    )
    second = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            decision_id=7,
            now=11.0,
        )
    )

    assert first["action"] == "ef_reduce"
    assert second is None
    assert len(sent) == 1


def test_engine_ef_wave_reduces_once_per_cooldown() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=0.8,
        ef_reduce_factor=0.8,
        ef_cooldown_s=30.0,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=1.0)
    mgr.register_stream("s1", engine_index=0, rho_content=1.0)

    first = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            decision_id=1,
            now=10.0,
        )
    )
    second = asyncio.run(
        mgr.observe_result(
            stream_id="s1",
            engine_index=0,
            early_finalized=True,
            decision_id=1,
            now=11.0,
        )
    )

    assert first["action"] == "ef_reduce"
    assert second["action"] == "ef_cooldown"
    assert mgr.rho_limit_for_stream("s0") == 0.6400000000000001
    assert len(sent) == 2


def test_ef_reduces_from_effective_content_window() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=1.0,
        ef_reduce_factor=0.8,
        operating_margin=0.85,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=0.5)

    event = asyncio.run(
        mgr.observe_result(
            stream_id="s0",
            engine_index=0,
            early_finalized=True,
            decision_id=1,
            now=10.0,
        )
    )

    assert event["action"] == "ef_reduce"
    assert mgr.rho_limit_for_stream("s0") == 0.4
    assert sent[-1][1]["rho"] == 0.4


def test_engine_unhealthy_feedback_reduces_severely() -> None:
    sent = []

    async def send_fn(stream_id, msg):
        sent.append((stream_id, msg))

    cfg = stable_send_window.StableSendWindowConfig(
        initial=1.0,
        failure_reduce_factor=0.5,
        min_update_interval_s=0.0,
    )
    mgr = stable_send_window.StableSendWindowManager(cfg, send_fn=send_fn)
    mgr.register_stream("s0", engine_index=0, rho_content=0.5)
    mgr.register_stream("s1", engine_index=1, rho_content=0.5)

    event = asyncio.run(
        mgr.observe_engine_pressure(
            engine_index=0,
            kind="engine_unhealthy",
            kv_usage=0.91,
            severe=True,
            now=10.0,
        )
    )

    assert event["action"] == "engine_unhealthy_reduce"
    assert mgr.rho_limit_for_stream("s0") == 0.25
    assert mgr.rho_limit_for_stream("s1") == 0.5
    assert [sid for sid, _ in sent] == ["s0"]
