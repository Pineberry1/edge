from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("remote_intake_token_flow", ROOT / "token_flow.py")
token_flow = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = token_flow
SPEC.loader.exec_module(token_flow)


def test_tokens_for_image_shape_uses_patch_grid() -> None:
    assert token_flow.tokens_for_image_shape(320, 240) == 12 * 9 == 108
    assert token_flow.tokens_for_image_shape(28, 28) == 1


def test_tracker_updates_flow_and_release() -> None:
    tracker = token_flow.StreamFlowTracker(
        "s0",
        window_seconds=2.0,
        decision_window_seconds=4.0,
        ema_alpha=1.0,
    )

    tracker.observe_edge_window(raw_tokens=200, window_seconds=2.0, at_wall=10.0)
    tracker.observe_network_window(
        recv_bytes=1000,
        offer_bytes=1500,
        full_bytes=3000,
        edge_queue_bytes=600,
        send_wait_ms=25.0,
        window_seconds=2.0,
        at_wall=11.0,
    )
    tracker.observe_kv_window(kv_tokens=80, window_seconds=2.0, at_wall=12.0)
    tracker.observe_request_open(20, at_wall=13.0)
    snap = tracker.snapshot()

    assert snap.lambda_edge == 100.0
    assert snap.lambda_net_recv_bytes == 500.0
    assert snap.lambda_net_offer_bytes == 750.0
    assert snap.lambda_net_full_bytes == 1500.0
    assert snap.net_backlog_bytes == 600
    assert snap.net_send_wait_ms == 25.0
    assert snap.lambda_kv == 40.0
    assert snap.last_recv_bytes == 1000
    assert snap.last_offer_bytes == 1500
    assert snap.last_full_bytes == 3000
    assert snap.live_kv_tokens == 100
    assert snap.d_k_dt == 20.0
    assert snap.cumulative_edge_tokens == 200
    assert snap.cumulative_recv_bytes == 1000
    assert snap.cumulative_offer_bytes == 1500
    assert snap.cumulative_full_bytes == 3000
    assert snap.cumulative_kv_tokens == 100

    released = tracker.observe_release(decision_duration_s=5.0, at_wall=18.0)
    snap2 = tracker.snapshot()

    assert released == 100
    assert snap2.lambda_free == 20.0
    assert snap2.live_kv_tokens == 0
    assert snap2.d_k_dt == -20.0
    assert snap2.cumulative_released_tokens == 100
