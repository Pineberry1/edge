"""Edge→cloud WS intake + vLLM online-prefill bridge.

Flow per client WebSocket:

    hello (text)         -> establish StreamSession bound to stream_id
    window_open (text)   -> create window state
    packet (binary)      -> buffer Annex-B payload
    window_close (text)  -> close one small online-prefill chunk and append it
                            to the active vLLM session
    stream_end (text)    -> edge-side decision boundary; end the active vLLM
                            session, poll for output, push `result` to edge
    bye (text)           -> close

Reverse channel: `BavaController` (§5 M4/M5) scrapes vLLM /metrics every
`BAVA_TICK_S`, computes ρ and α updates from queue / KV error, and pushes
`rho_update` text frames down each tracked stream.

The intake process talks to one or more vLLM engines on `${VLLM_API_BASE}`
or `${VLLM_API_BASE_LIST}` and does not modify vLLM itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .admission import AdmissionGate, admission_config_from_env
from .alpha_executor import AlphaPolicy
from .budget_allocator import BudgetAllocator, GlobalCapacity, StreamBudget, StreamCostMeta
from .controller import BavaController, controller_config_from_env
from .latency import LatencyTracker
from .metrics import MultiVllmMetricsScraper, VllmEngineSnapshot, VllmMetricsScraper, VllmSnapshot
from .stable_send_window import StableSendWindowConfig, StableSendWindowManager
from .token_flow import FlowSnapshot, aggregate_flow
from .vllm_client import VLLMOnlinePrefillClient
from .window_assembler import StreamSession
from .wire import (
    MSG_BYE,
    MSG_HELLO,
    MSG_PACKET,
    MSG_BUDGET_UPDATE,
    MSG_EDGE_STATS,
    MSG_EARLY_FINALIZE,
    MSG_STREAM_END,
    MSG_WINDOW_CLOSE,
    MSG_WINDOW_OPEN,
    unpack_binary,
)

logging.basicConfig(
    level=os.environ.get("BAVA_LOG", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("intake.server")


class IntakeState:
    def __init__(self) -> None:
        self.api_bases: List[str] = []
        self.vllm_list: List[VLLMOnlinePrefillClient] = []
        self.sessions: Dict[str, StreamSession] = {}
        self.sockets: Dict[str, WebSocket] = {}
        self.controller: Optional[BavaController] = None
        self.scraper: Optional[Any] = None
        self.admission_gate: Optional[AdmissionGate] = None
        self.allocator: Optional[BudgetAllocator] = None
        self.budget_task: Optional[asyncio.Task] = None
        self.send_window_task: Optional[asyncio.Task] = None
        self.budgets: Dict[str, StreamBudget] = {}
        self.send_window: Optional[StableSendWindowManager] = None
        self.latency: LatencyTracker = LatencyTracker()
        self.anchor_log_path: Optional[Path] = None
        self.alpha_policy_template: AlphaPolicy = AlphaPolicy(alpha=1.0)
        # vLLM request_ids created by each engine. Used at
        # shutdown / disconnect to DELETE residue on vLLM so the next intake
        # run doesn't see stale Q from our previous sessions.
        self.active_rids_by_engine: DefaultDict[int, set[str]] = defaultdict(set)


STATE = IntakeState()
app = FastAPI()


def _parse_api_bases(var_name: str, fallback: Optional[List[str]] = None) -> List[str]:
    raw = os.environ.get(var_name, "")
    bases = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if bases:
        return bases
    return list(fallback or [])


def _stable_engine_index(stream_id: str, engine_count: int) -> int:
    if engine_count <= 1:
        return 0
    digest = hashlib.blake2b(stream_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % engine_count


def _least_sessions_engine_index(stream_id: str, engine_count: int) -> int:
    if engine_count <= 1:
        return 0
    loads = {idx: 0 for idx in range(engine_count)}
    for sess in STATE.sessions.values():
        if 0 <= int(sess.engine_index) < engine_count:
            loads[int(sess.engine_index)] += 1
    min_load = min(loads.values())
    candidates = [idx for idx, load in loads.items() if load == min_load]
    if len(candidates) == 1:
        return candidates[0]
    digest = hashlib.blake2b(stream_id.encode("utf-8"), digest_size=8).digest()
    return candidates[int.from_bytes(digest, "big") % len(candidates)]


def _choose_engine_index(stream_id: str, engine_count: int) -> int:
    mode = os.environ.get("BAVA_ENGINE_ASSIGNMENT", "least_sessions").strip().lower()
    if mode in {"hash", "stable_hash"}:
        return _stable_engine_index(stream_id, engine_count)
    if mode not in {"least", "least_sessions", "least_loaded"}:
        log.warning("unknown BAVA_ENGINE_ASSIGNMENT=%r; using least_sessions", mode)
    return _least_sessions_engine_index(stream_id, engine_count)


def _track_rid(engine_index: int, request_id: str) -> None:
    STATE.active_rids_by_engine[int(engine_index)].add(request_id)


def _drop_rid(engine_index: int, request_id: str) -> None:
    STATE.active_rids_by_engine[int(engine_index)].discard(request_id)


def _engine_states() -> List[VllmEngineSnapshot]:
    if STATE.scraper is None:
        return []
    try:
        return STATE.scraper.engine_states()
    except Exception:
        return []


def _serialize_snapshot(snap: Optional[VllmSnapshot]) -> Optional[Dict[str, Any]]:
    if snap is None:
        return None
    out: Dict[str, Any] = {
        "Q": snap.num_requests_waiting,
        "KV": snap.kv_cache_usage_perc,
        "running": snap.num_requests_running,
        "prompt_rate": snap.prompt_token_rate,
        "gen_rate": snap.generation_token_rate,
        "preempt_rate": snap.preemption_rate,
    }
    if snap.kv_total_tokens is not None:
        out["kv_total_tokens"] = float(snap.kv_total_tokens)
    alpha_used = getattr(snap, "alpha_used", None)
    if alpha_used is not None:
        out["alpha_used"] = dict(alpha_used)
    alpha_init = getattr(snap, "alpha_init", None)
    if alpha_init is not None:
        out["alpha_init"] = dict(alpha_init)
    eta_press = getattr(snap, "eta_press", None)
    if eta_press is not None:
        out["eta_press"] = float(eta_press)
    token_flow = getattr(snap, "token_flow", None)
    if token_flow is not None:
        out["token_flow"] = token_flow
    return out


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_value(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _optional_id(value: Any) -> Optional[int]:
    parsed = _int_value(value, -1)
    return parsed if parsed >= 0 else None


async def _resolve_kv_cap_tokens() -> int:
    raw = os.environ.get("BAVA_KV_CAP_TOKENS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                log.info("budget allocator kv cap from BAVA_KV_CAP_TOKENS=%d", value)
                return value
        except ValueError:
            log.warning("invalid BAVA_KV_CAP_TOKENS=%r; trying metrics/default", raw)
    if STATE.scraper is not None:
        try:
            snap = await STATE.scraper.scrape()
            if snap.kv_total_tokens is not None and snap.kv_total_tokens > 0:
                log.info("budget allocator kv cap from metrics=%d", snap.kv_total_tokens)
                return int(snap.kv_total_tokens)
        except Exception as e:
            log.warning("budget allocator could not resolve kv cap from metrics: %s", e)
    fallback = 98304
    log.warning("budget allocator kv cap unavailable; defaulting to %d tokens", fallback)
    return fallback


def _budget_capacity_from_env(kv_cap_tokens: int) -> GlobalCapacity:
    return GlobalCapacity(
        kv_cap_tokens=int(kv_cap_tokens),
        safety_factor=_float_env("BAVA_KV_SAFETY", 0.85),
        min_windows=_int_env("BAVA_BUDGET_MIN_WINDOWS", 2),
        max_windows=_int_env("BAVA_BUDGET_MAX_WINDOWS", 32),
        block_overhead_tokens=_int_env("BAVA_BUDGET_BLOCK_OVERHEAD_TOKENS", 64),
        policy=os.environ.get("BAVA_BUDGET_POLICY", "ef_guard"),
        target_windows=_int_env(
            "BAVA_BUDGET_TARGET_WINDOWS",
            _int_env("BAVA_DEFAULT_WINDOWS_PER_DECISION", 10),
        ),
        ef_windows=_int_env("BAVA_BUDGET_EF_WINDOWS", 6),
        ef_dynamic=_bool_env("BAVA_BUDGET_EF_DYNAMIC", True),
        ef_margin_windows=_int_env("BAVA_BUDGET_EF_MARGIN_WINDOWS", 1),
    )


@app.on_event("startup")
async def _startup() -> None:
    api_bases = _parse_api_bases("VLLM_API_BASE_LIST")
    if not api_bases:
        api_bases = _parse_api_bases(
            "VLLM_API_BASE",
            fallback=["http://127.0.0.1:8000"],
        )
    STATE.api_bases = api_bases
    STATE.vllm_list = [VLLMOnlinePrefillClient(api_base=base) for base in api_bases]

    anchor_log_env = os.environ.get("BAVA_ANCHOR_LOG", "/tmp/bava_anchors.jsonl")
    STATE.anchor_log_path = Path(anchor_log_env) if anchor_log_env else None

    STATE.alpha_policy_template = AlphaPolicy(
        alpha=1.0,
        min_side=int(os.environ.get("BAVA_ALPHA_MIN_SIDE", "112")),
        max_side=int(os.environ.get("BAVA_ALPHA_MAX_SIDE", "1568")),
        align=int(os.environ.get("BAVA_ALPHA_ALIGN", "28")),
        skip_threshold=float(os.environ.get("BAVA_ALPHA_SKIP", "0.999")),
        score_power=float(os.environ.get("BAVA_ALPHA_SCORE_POWER", "1.0")),
        per_frame_floor=float(os.environ.get("BAVA_ALPHA_PER_FRAME_FLOOR", "0.15")),
    )

    metrics_bases = _parse_api_bases("BAVA_METRICS_BASE_LIST", fallback=api_bases)
    if not metrics_bases:
        metrics_base = os.environ.get("BAVA_METRICS_BASE", "")
        metrics_bases = [metrics_base.strip().rstrip("/")] if metrics_base.strip() else list(api_bases)
    if len(metrics_bases) == 1:
        STATE.scraper = VllmMetricsScraper(api_base=metrics_bases[0])
    else:
        STATE.scraper = MultiVllmMetricsScraper(api_bases=metrics_bases)

    controller_enabled = os.environ.get("BAVA_CONTROLLER_ENABLED", "1") != "0"
    send_window_enabled = _bool_env("BAVA_SEND_WINDOW_ENABLED", controller_enabled)
    if send_window_enabled:
        STATE.send_window = StableSendWindowManager(
            StableSendWindowConfig(
                lo=_float_env("BAVA_SEND_WINDOW_LO", _float_env("BAVA_RHO_LO", 0.05)),
                hi=_float_env("BAVA_SEND_WINDOW_HI", _float_env("BAVA_RHO_HI", 1.0)),
                initial=_float_env("BAVA_SEND_WINDOW_INIT", 0.0),
                increase_step=_float_env("BAVA_SEND_WINDOW_INCREASE_STEP", 0.02),
                stable_results_required=_int_env("BAVA_SEND_WINDOW_STABLE_RESULTS", 20),
                probe_interval_s=_float_env("BAVA_SEND_WINDOW_PROBE_INTERVAL_S", 120.0),
                kv_probe_max=_float_env("BAVA_SEND_WINDOW_KV_PROBE_MAX", 0.85),
                ef_reduce_factor=_float_env("BAVA_SEND_WINDOW_EF_REDUCE", 0.80),
                failure_reduce_factor=_float_env("BAVA_SEND_WINDOW_FAILURE_REDUCE", 0.50),
                operating_margin=_float_env("BAVA_SEND_WINDOW_MARGIN", 0.85),
                ef_cooldown_s=_float_env("BAVA_SEND_WINDOW_EF_COOLDOWN_S", 90.0),
                min_update_interval_s=_float_env("BAVA_SEND_WINDOW_MIN_UPDATE_S", 1.0),
            ),
            send_fn=_broadcast_to_stream,
            apply_fn=_apply_rho_to_controller_state,
            current_rho_fn=_lookup_controller_rho,
        )
    if controller_enabled:
        log_path_str = os.environ.get("BAVA_CONTROLLER_LOG", "/tmp/bava_controller.jsonl")
        STATE.controller = BavaController(
            scraper=STATE.scraper,
            config=controller_config_from_env(),
            log_path=Path(log_path_str) if log_path_str else None,
            load_lookup=_lookup_load,
            alpha_init_lookup=_lookup_alpha_init,
            flow_lookup=_lookup_flow,
            rho_limit_lookup=_lookup_send_window_rho_limit,
        )
        STATE.controller.start(_broadcast_to_stream)
    STATE.admission_gate = AdmissionGate(
        snapshot_fn=lambda: STATE.controller.snapshot() if STATE.controller else None,
        flow_snapshot_fn=_lookup_flow,
        cfg=admission_config_from_env(),
    )
    budget_enabled = os.environ.get("BAVA_BUDGET_ENABLED", "1") != "0"
    kv_cap_tokens = await _resolve_kv_cap_tokens() if budget_enabled else 0
    if budget_enabled:
        STATE.allocator = BudgetAllocator(_budget_capacity_from_env(kv_cap_tokens))
        budget_rebalance_s = _float_env("BAVA_BUDGET_REBALANCE_S", 2.0)
        if budget_rebalance_s > 0:
            STATE.budget_task = asyncio.create_task(
                _budget_rebalance_loop(budget_rebalance_s),
                name="bava-budget-rebalance",
            )
    if STATE.send_window is not None:
        monitor_s = _float_env("BAVA_SEND_WINDOW_ENGINE_MONITOR_S", 1.0)
        if monitor_s > 0:
            STATE.send_window_task = asyncio.create_task(
                _send_window_engine_monitor_loop(monitor_s),
                name="bava-send-window-engine-monitor",
            )
    else:
        STATE.allocator = None
    log.info(
        "intake startup: vllm=%s anchor_log=%s controller=%s budget=%s kv_cap_tokens=%d",
        ",".join(api_bases),
        STATE.anchor_log_path,
        controller_enabled,
        budget_enabled,
        kv_cap_tokens,
    )


async def _purge_active_rids() -> int:
    """DELETE every vLLM session this intake opened. Returns count aborted."""
    if not STATE.vllm_list:
        return 0
    grouped = {
        engine_index: list(rids)
        for engine_index, rids in STATE.active_rids_by_engine.items()
        if rids
    }
    if not grouped:
        return 0
    total = sum(len(rids) for rids in grouped.values())
    log.info("purge: aborting %d active vLLM sessions across %d engines", total, len(grouped))
    aborted = 0
    tasks = []
    meta = []
    for engine_index, rids in grouped.items():
        client = STATE.vllm_list[engine_index]
        for rid in rids:
            tasks.append(client.abort(rid))
            meta.append((engine_index, rid))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for (engine_index, rid), ok in zip(meta, results):
        if isinstance(ok, Exception):
            continue
        if ok:
            aborted += 1
        _drop_rid(engine_index, rid)
    log.info("purge: %d/%d sessions aborted on vLLM", aborted, total)
    return aborted


@app.on_event("shutdown")
async def _shutdown() -> None:
    # Order matters: stop controller first so it doesn't log snapshots of
    # a dying vllm_client; then purge sessions; then close transports.
    if STATE.budget_task is not None:
        STATE.budget_task.cancel()
        try:
            await STATE.budget_task
        except asyncio.CancelledError:
            pass
        STATE.budget_task = None
    if STATE.send_window_task is not None:
        STATE.send_window_task.cancel()
        try:
            await STATE.send_window_task
        except asyncio.CancelledError:
            pass
        STATE.send_window_task = None
    if STATE.controller is not None:
        await STATE.controller.stop()
    await _purge_active_rids()
    if STATE.scraper is not None:
        await STATE.scraper.aclose()
    if STATE.vllm_list:
        await asyncio.gather(
            *[client.aclose() for client in STATE.vllm_list],
            return_exceptions=True,
        )


async def _broadcast_to_stream(stream_id: str, msg: Dict[str, Any]) -> None:
    ws = STATE.sockets.get(stream_id)
    if ws is None:
        return
    await ws.send_text(json.dumps(msg, separators=(",", ":")))


def _apply_rho_to_controller_state(stream_id: str, rho: float, reason: str) -> None:
    if STATE.controller is None:
        return
    st = STATE.controller.state(stream_id)
    if st is None:
        return
    st.rho = float(rho)
    st.last_push_reason = reason
    st.last_update_wall = time.time()


def _lookup_send_window_rho_limit(stream_id: str) -> Optional[float]:
    if STATE.send_window is None:
        return None
    return STATE.send_window.rho_limit_for_stream(stream_id)


def _lookup_controller_rho(stream_id: str) -> Optional[float]:
    if STATE.controller is None:
        return None
    st = STATE.controller.state(stream_id)
    if st is None:
        return None
    return float(st.rho)


def _lookup_alpha(stream_id: str) -> float:
    """Pull alpha_used for a stream from the controller (falls back to 1.0)."""
    if STATE.controller is None:
        return 1.0
    return float(STATE.controller.alpha_for_stream(stream_id))


def _lookup_alpha_init(stream_id: str) -> tuple[float, int]:
    sess = STATE.sessions.get(stream_id)
    if sess is None:
        return (1.0, 0)
    return sess.alpha_init_state.get()


def _lookup_load(stream_id: str):
    """Controller → intake: report how much intake-side work this stream has.

    Returns (inflight_appends, last_append_ms_or_None, inflight_windows).
    Used by the controller to weight global Δρ/Δα by per-stream contribution.
    """
    sess = STATE.sessions.get(stream_id)
    if sess is None:
        return (0, None, 0)
    return (
        int(sess.inflight_appends),
        sess.last_append_ms,
        int(sess.inflight_windows),
    )


def _lookup_flow() -> Optional[FlowSnapshot]:
    snaps = {
        sid: sess.flow_snapshot()
        for sid, sess in STATE.sessions.items()
    }
    if not snaps:
        return None
    return aggregate_flow(snaps)


def _stream_meta(stream_id: str, sess: StreamSession) -> StreamCostMeta:
    st = STATE.controller.state(stream_id) if STATE.controller is not None else None
    rho = float(st.rho) if st is not None else float(sess.hello.get("rho") or 0.3)
    alpha = float(st.alpha) if st is not None else float(sess.hello.get("alpha") or 1.0)
    flow = sess.flow_snapshot()
    prompt = str(sess.hello.get("prompt") or "")
    ef_cooldown_s = _float_env("BAVA_BUDGET_EF_COOLDOWN_S", 60.0)
    return StreamCostMeta(
        stream_id=stream_id,
        frame_height=max(
            1,
            _int_value(sess.hello.get("frame_height"), _int_env("BAVA_FRAME_H", 320)),
        ),
        frame_width=max(
            1,
            _int_value(sess.hello.get("frame_width"), _int_env("BAVA_FRAME_W", 240)),
        ),
        frames_per_window=max(
            1,
            _int_value(
                sess.hello.get("frames_per_window"),
                _int_env("BAVA_MAX_FRAMES_PER_WINDOW", 8),
            ),
        ),
        rho=max(0.0, rho),
        prompt_tokens=max(1, _int_value(sess.hello.get("prompt_tokens"), max(8, len(prompt) // 3))),
        max_tokens=max(1, _int_value(sess.hello.get("max_tokens"), 24)),
        alpha=max(0.0, min(1.0, alpha)),
        live_kv_tokens=max(0, int(flow.live_kv_tokens)),
        memory_tokens_per_window=max(0, _int_env("BAVA_MEMORY_TOKENS_PER_WINDOW", 0)),
        early_finalized_recent=sess.early_finalized_recent(ef_cooldown_s),
        last_early_finalized_windows=max(0, int(sess.last_early_finalized_windows)),
    )


async def _rebalance_budget_and_broadcast(reason: str) -> None:
    if STATE.allocator is None:
        return
    metas = [
        _stream_meta(sid, sess)
        for sid, sess in sorted(STATE.sessions.items(), key=lambda item: item[0])
        if sess.inference_mode != "completion"
    ]
    if not metas:
        STATE.budgets = {}
        return
    budgets = STATE.allocator.allocate(metas)
    STATE.budgets = budgets
    for sid, budget in budgets.items():
        msg = {
            "kind": MSG_BUDGET_UPDATE,
            "stream_id": sid,
            "version": budget.version,
            "windows_per_decision": budget.windows_per_decision,
            "est_kv_tokens": budget.est_kv_tokens,
            "reason": reason,
        }
        try:
            await _broadcast_to_stream(sid, msg)
        except Exception as e:
            log.debug("budget broadcast failed stream=%s: %s", sid, e)
    log_fn = log.debug if reason == "periodic_flow_rebalance" else log.info
    log_fn(
        "budget rebalance reason=%s streams=%d windows=%s",
        reason,
        len(budgets),
        {sid: b.windows_per_decision for sid, b in budgets.items()},
    )


async def _budget_rebalance_loop(interval_s: float) -> None:
    interval = max(0.25, float(interval_s or 2.0))
    while True:
        await asyncio.sleep(interval)
        if not STATE.sessions:
            continue
        try:
            await _rebalance_budget_and_broadcast("periodic_flow_rebalance")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.debug("periodic budget rebalance failed: %s", e)


def _engine_has_streams(engine_index: int) -> bool:
    for sess in STATE.sessions.values():
        if int(sess.engine_index) == int(engine_index):
            return True
    return False


async def _send_window_engine_monitor_loop(interval_s: float) -> None:
    interval = max(0.25, float(interval_s or 1.0))
    kv_panic = _float_env("BAVA_SEND_WINDOW_KV_PANIC", 0.0)
    while True:
        await asyncio.sleep(interval)
        if STATE.send_window is None or not STATE.sessions:
            continue
        for state in _engine_states():
            if not _engine_has_streams(int(state.index)):
                continue
            snap = state.snapshot
            kv = (
                float(snap.kv_cache_usage_perc)
                if snap is not None and snap.kv_cache_usage_perc is not None
                else None
            )
            try:
                if not state.ok:
                    event = await STATE.send_window.observe_engine_pressure(
                        engine_index=int(state.index),
                        kind="engine_unhealthy",
                        kv_usage=kv,
                        severe=True,
                    )
                    log.warning(
                        "send-window engine unhealthy feedback engine=%d error=%s event=%s",
                        int(state.index),
                        state.error,
                        event,
                    )
                elif kv_panic > 0.0 and kv is not None and kv >= kv_panic:
                    event = await STATE.send_window.observe_engine_pressure(
                        engine_index=int(state.index),
                        kind="kv_panic",
                        kv_usage=kv,
                        severe=False,
                    )
                    log.warning(
                        "send-window kv panic feedback engine=%d kv=%.3f event=%s",
                        int(state.index),
                        kv,
                        event,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.debug("send-window engine monitor failed engine=%s: %s", state.index, e)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    snap = STATE.controller.snapshot() if STATE.controller else None
    if snap is None and STATE.scraper is not None:
        try:
            snap = await STATE.scraper.scrape()
        except Exception as e:
            log.debug("healthz metrics scrape failed: %s", e)
            try:
                snap = STATE.scraper.last_snapshot()
            except Exception:
                snap = None
    streams_state: Dict[str, Dict[str, Any]] = {}
    for sid, sess in list(STATE.sessions.items()):
        st = STATE.controller.state(sid) if STATE.controller is not None else None
        entry: Dict[str, Any] = {
            "rho": float(st.rho) if st is not None else float(sess.hello.get("rho") or 0.3),
            "alpha": float(st.alpha) if st is not None else float(sess.hello.get("alpha") or 1.0),
            "engine_index": int(sess.engine_index),
            "inference_mode": sess.inference_mode,
        }
        alpha_init, alpha_init_version = sess.alpha_init_state.get()
        entry["alpha_init"] = float(alpha_init)
        entry["alpha_init_version"] = int(alpha_init_version)
        if st is not None:
            entry["alpha_used"] = float(st.alpha)
            entry["eta_press"] = float(st.eta_press)
        if sess.engine_index < len(STATE.api_bases):
            entry["engine_api_base"] = STATE.api_bases[sess.engine_index]
        entry["inflight_appends"] = int(sess.inflight_appends)
        entry["inflight_windows"] = int(sess.inflight_windows)
        if sess.last_append_ms is not None:
            entry["last_append_ms"] = float(sess.last_append_ms)
        if sess.last_decode_ms is not None:
            entry["last_decode_ms"] = float(sess.last_decode_ms)
        entry["early_finalized_count"] = int(sess.early_finalized_count)
        entry["last_early_finalized_windows"] = int(sess.last_early_finalized_windows)
        entry["early_finalized_recent"] = bool(
            sess.early_finalized_recent(_float_env("BAVA_BUDGET_EF_COOLDOWN_S", 60.0))
        )
        if sess.last_early_finalized_wall > 0:
            entry["last_early_finalized_age_s"] = max(
                0.0,
                time.time() - float(sess.last_early_finalized_wall),
            )
        entry["flow"] = sess.flow_snapshot().as_dict()
        entry["visual_memory"] = sess.visual_memory_snapshot()
        budget = STATE.budgets.get(sid)
        if budget is not None:
            entry["windows_per_decision"] = int(budget.windows_per_decision)
            entry["budget_version"] = int(budget.version)
            entry["budget_est_kv_tokens"] = int(budget.est_kv_tokens)
        streams_state[sid] = entry
    flow = _lookup_flow()
    return {
        "ok": True,
        "active_streams": list(STATE.sessions.keys()),
        "controller_streams": streams_state,
        "token_flow": flow.as_dict() if flow is not None else None,
        "vllm_snapshot": _serialize_snapshot(snap),
        "vllm_engines": [
            {
                "index": state.index,
                "api_base": state.api_base,
                "ok": state.ok,
                "error": state.error,
                "snapshot": _serialize_snapshot(state.snapshot),
            }
            for state in _engine_states()
        ],
        "send_window": STATE.send_window.snapshot() if STATE.send_window is not None else None,
    }


def _engine_kv_usage(engine_index: int) -> Optional[float]:
    for state in _engine_states():
        if int(state.index) == int(engine_index) and state.snapshot is not None:
            return float(state.snapshot.kv_cache_usage_perc)
    snap = STATE.controller.snapshot() if STATE.controller is not None else None
    if snap is not None:
        return float(snap.kv_cache_usage_perc)
    return None


async def _on_session_result(msg: Dict[str, Any], sid: str) -> None:
    await _broadcast_to_stream(sid, msg)
    if msg.get("kind") != "result" or msg.get("mode") == "completion":
        return
    if STATE.send_window is None:
        return
    try:
        await STATE.send_window.observe_result(
            stream_id=sid,
            engine_index=int(msg.get("engine_index") or 0),
            early_finalized=bool(msg.get("early_finalized")),
            kv_usage=_engine_kv_usage(int(msg.get("engine_index") or 0)),
            decision_id=_optional_id(msg.get("window_id")) if bool(msg.get("early_finalized")) else None,
        )
    except Exception as e:
        log.debug("send-window observe result failed stream=%s: %s", sid, e)


async def _on_session_early_finalized(msg: Dict[str, Any], sid: str) -> None:
    if STATE.send_window is not None:
        try:
            await STATE.send_window.observe_result(
                stream_id=sid,
                engine_index=int(msg.get("engine_index") or 0),
                early_finalized=True,
                kv_usage=_engine_kv_usage(int(msg.get("engine_index") or 0)),
                decision_id=_optional_id(msg.get("decision_id")),
            )
        except Exception as e:
            log.debug("send-window EF observe failed stream=%s: %s", sid, e)
    try:
        await _rebalance_budget_and_broadcast("ef_guard")
    except Exception as e:
        log.debug("EF budget rebalance failed stream=%s: %s", sid, e)
    if not _bool_env("BAVA_EF_NOTIFY_EDGE", True):
        return
    budget = STATE.budgets.get(sid)
    edge_msg: Dict[str, Any] = {
        "kind": MSG_EARLY_FINALIZE,
        "stream_id": sid,
        "reason": str(msg.get("phase") or "early_finalize"),
        "decision_id": _optional_id(msg.get("decision_id")),
        "engine_index": int(msg.get("engine_index") or 0),
        "active_windows": _optional_id(msg.get("active_windows")),
    }
    if budget is not None:
        edge_msg.update(
            {
                "version": int(budget.version),
                "windows_per_decision": int(budget.windows_per_decision),
                "est_kv_tokens": int(budget.est_kv_tokens),
            }
        )
    try:
        await _broadcast_to_stream(sid, edge_msg)
        log.warning(
            "stream=%s sent edge early_finalize decision=%s windows=%s reason=%s",
            sid,
            edge_msg.get("decision_id"),
            edge_msg.get("windows_per_decision"),
            edge_msg.get("reason"),
        )
    except Exception as e:
        log.debug("edge early_finalize notify failed stream=%s: %s", sid, e)


@app.get("/stats/latency")
async def latency_stats() -> Dict[str, Any]:
    return STATE.latency.summary()


@app.post("/admin/purge_sessions")
async def admin_purge_sessions() -> Dict[str, Any]:
    """Manual hook to DELETE all active vLLM sessions this intake tracks.

    Used by the A/B harness between configs so we don't need to restart vLLM
    just to clear stale sessions. Idempotent.
    """
    aborted = await _purge_active_rids()
    return {"ok": True, "aborted": aborted}


@app.post("/admin/push_rho")
async def admin_push_rho(body: Dict[str, Any]) -> Dict[str, Any]:
    """Manual ρ push — useful for testing the reverse channel independently."""
    stream_id = str(body.get("stream_id") or "")
    rho = float(body.get("rho") or 0.3)
    alpha = body.get("alpha")
    reason = str(body.get("reason") or "admin")
    if stream_id not in STATE.sockets:
        return {"ok": False, "error": f"no active stream {stream_id!r}"}
    msg: Dict[str, Any] = {"kind": "rho_update", "rho": rho, "reason": reason}
    if alpha is not None:
        msg["alpha"] = float(alpha)
    await _broadcast_to_stream(stream_id, msg)
    if STATE.controller is not None:
        st = STATE.controller.state(stream_id)
        if st is not None:
            st.rho = rho
            if alpha is not None:
                st.alpha = float(alpha)
    return {"ok": True, "stream_id": stream_id, **msg}


@app.websocket("/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    stream_id: Optional[str] = None
    session: Optional[StreamSession] = None
    try:
        while True:
            msg = await ws.receive()
            kind = msg.get("type")
            if kind == "websocket.disconnect":
                break
            if "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    log.warning("bad text frame (ignored)")
                    continue
                result = await _handle_text(ws, payload, session, stream_id)
                if result is not None:
                    stream_id, session = result
            elif "bytes" in msg and msg["bytes"] is not None:
                if session is None:
                    log.warning("binary frame before hello — dropping")
                    continue
                try:
                    header, body = unpack_binary(msg["bytes"])
                except Exception as e:
                    log.warning("bad binary frame: %s", e)
                    continue
                if header.get("kind") != MSG_PACKET:
                    log.debug("binary frame non-packet kind=%r", header.get("kind"))
                session.on_packet(header, body)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("ws session error: %s", e)
    finally:
        if stream_id is not None:
            if session is not None:
                try:
                    await session.cleanup()
                except Exception as e:
                    log.debug("stream=%s cleanup failed: %s", stream_id, e)
            STATE.sockets.pop(stream_id, None)
            STATE.sessions.pop(stream_id, None)
            if STATE.send_window is not None:
                STATE.send_window.unregister_stream(stream_id)
            if STATE.controller is not None:
                STATE.controller.untrack(stream_id)
            log.info("stream=%s disconnected", stream_id)
            await _rebalance_budget_and_broadcast("rebalance_on_leave")


async def _handle_text(
    ws: WebSocket,
    payload: Dict[str, Any],
    session: Optional[StreamSession],
    stream_id: Optional[str],
):
    kind = payload.get("kind")
    if kind == MSG_HELLO:
        new_stream_id = str(payload.get("stream_id") or "anon")
        assert STATE.vllm_list
        engine_index = _choose_engine_index(new_stream_id, len(STATE.vllm_list))
        session = StreamSession(
            stream_id=new_stream_id,
            hello=payload,
            vllm=STATE.vllm_list[engine_index],
            engine_index=engine_index,
            anchor_log_path=STATE.anchor_log_path,
            on_result=lambda msg, sid=new_stream_id: _on_session_result(msg, sid),
            on_early_finalized=lambda msg, sid=new_stream_id: _on_session_early_finalized(msg, sid),
            latency_tracker=STATE.latency,
            alpha_lookup=_lookup_alpha if STATE.controller is not None else None,
            alpha_policy_template=STATE.alpha_policy_template,
            finalize_concurrency=int(os.environ.get("BAVA_STREAM_CONCURRENCY", "2")),
            rid_sink=lambda rid, idx=engine_index: _track_rid(idx, rid),
            rid_drop=lambda rid, idx=engine_index: _drop_rid(idx, rid),
            admission_gate=STATE.admission_gate,
        )
        STATE.sessions[new_stream_id] = session
        STATE.sockets[new_stream_id] = ws
        if STATE.controller is not None:
            STATE.controller.track(
                new_stream_id,
                initial_rho=float(payload.get("rho") or 0.3),
                initial_alpha=float(payload.get("alpha") or 1.0),
            )
        if STATE.send_window is not None:
            initial_rho = float(payload.get("rho") or 0.3)
            STATE.send_window.register_stream(
                new_stream_id,
                engine_index=engine_index,
                rho_content=initial_rho,
            )
            rho_limit = STATE.send_window.rho_limit_for_stream(new_stream_id)
            if rho_limit is not None and float(rho_limit) < initial_rho - 1e-6:
                reason = (
                    f"send_window_initial engine={engine_index} "
                    f"rho={float(rho_limit):.3f}"
                )
                await _broadcast_to_stream(
                    new_stream_id,
                    {
                        "kind": "rho_update",
                        "rho": float(rho_limit),
                        "send_window": float(rho_limit),
                        "engine_index": engine_index,
                        "reason": reason,
                    },
                )
                _apply_rho_to_controller_state(new_stream_id, float(rho_limit), reason)
        log.info(
            "stream=%s engine=%d api=%s mode=%s hello model=%r prompt=%r rho=%.3f window_s=%s",
            new_stream_id,
            engine_index,
            STATE.api_bases[engine_index],
            session.inference_mode,
            payload.get("model"),
            (payload.get("prompt") or "")[:80],
            float(payload.get("rho") or 0.0),
            payload.get("window_seconds"),
        )
        await _rebalance_budget_and_broadcast("rebalance_on_join")
        return new_stream_id, session
    if session is None:
        log.warning("text frame kind=%r before hello — ignored", kind)
        return None
    if kind == MSG_WINDOW_OPEN:
        session.on_window_open(payload)
    elif kind == MSG_WINDOW_CLOSE:
        st = session.mark_window_close(payload)
        if st is not None:
            session.schedule_ordered(lambda st=st: session.append_closed_window(st))
    elif kind == MSG_EDGE_STATS:
        session.on_edge_stats(payload)
    elif kind == MSG_STREAM_END:
        session.schedule_ordered(lambda payload=payload: session.on_stream_end(payload))
    elif kind == MSG_BYE:
        log.info("stream=%s bye", session.stream_id)
    else:
        log.debug("unknown text kind=%r", kind)
    return None


def main() -> None:
    import uvicorn

    host = os.environ.get("INTAKE_HOST", "0.0.0.0")
    port = int(os.environ.get("INTAKE_PORT", "9100"))
    uvicorn.run(
        "intake.server:app",
        host=host,
        port=port,
        log_level=os.environ.get("BAVA_LOG", "info").lower(),
    )


if __name__ == "__main__":
    main()
