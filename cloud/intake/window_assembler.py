"""Per-stream window assembly.

For each (stream_id, window_id), accumulate arriving binary packets' Annex-B
payloads in stream order (by seq). In the default `online_prefill` mode,
`window_close` is the small chunk boundary: when it arrives we:

  1. Concatenate payloads → PyAV decode → BGR frames.
  2. Preserve the frames the edge sent for this chunk. The edge owns temporal
     selection and decision-window boundaries; intake must not re-sample the
     native stream.
  3. Apply the α executor. The default path enforces the token budget by
     resizing each frame by √α; the tokenmerger experiment path keeps pixels
     unchanged and forwards α to vLLM for visual-token folding.
  4. JPEG-encode each prepared frame.
  5. Create/reuse the active vLLM online_prefill session and append frames.

In `completion` mode, steps 1-4 are identical but the frames are only cached.
`stream_end` sends the decision window as one native `/v1/chat/completions`
request. This is the fair completion baseline: the edge still uploads H.264
and the cloud intake performs decode/JPEG before calling vLLM.

`stream_end` is a separate edge-side decision boundary. In `online_prefill`
mode we send an empty append with `stream_end=True`, poll for the output, then
reset the active online-prefill session for the next decision window.

CPU-heavy steps (PyAV decode, cv2 resize, cv2 JPEG encode) run in a thread
via `asyncio.to_thread` so the FastAPI event loop and the WebSocket
read/write coroutines keep ticking for other streams. Without this,
concurrent N streams serialize on decode and the controller's metric
scraping starves.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set

from .admission import AdmissionGate
from .alpha_executor import AlphaPolicy, apply_alpha_uniform, apply_alpha_weighted
from .alpha_init_state import AlphaInitState
from .gop_decoder import (
    BgrFrame,
    decode_to_bgr,
    encode_bgr_to_jpeg_data_uri,
    subsample_uniform,
)
from .latency import LatencyTracker, WindowLatencySample
from .token_flow import StreamFlowSnapshot, StreamFlowTracker, tokens_for_image_shape
from .vllm_client import PollResult, VLLMOnlinePrefillClient

log = logging.getLogger("intake.window")


@dataclass
class PacketRec:
    seq: int
    gop_index: int
    gop_pos: int
    is_idr: bool
    payload: bytes
    header: Dict[str, Any]


@dataclass
class WindowState:
    stream_id: str
    window_id: int
    rho: float
    opened_at: float
    packets: List[PacketRec] = field(default_factory=list)
    closed: bool = False
    closed_at: Optional[float] = None
    request_id: Optional[str] = None
    decoded_frames: int = 0
    sent_frames: int = 0
    recv_bytes: int = 0
    alpha_used: float = 1.0
    avg_dim: Optional[int] = None  # geometric mean of sent frame sides
    raw_tokens_est: int = 0
    kv_tokens_est: int = 0
    output_text: Optional[str] = None
    prepared_frames: List[Dict[str, str]] = field(default_factory=list)
    decode_ms: float = 0.0
    appended: bool = False
    append_ms: float = 0.0
    alpha_ms: float = 0.0
    jpeg_ms: float = 0.0
    dropped_after_finalize: bool = False


@dataclass
class DecodePrefix:
    payload: bytes = b""
    frame_count: int = 0
    anchor_window_id: Optional[int] = None


AlphaLookup = Callable[[str], float]  # stream_id -> current alpha


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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip()


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class StreamSession:
    """One per WebSocket — one edge camera."""

    def __init__(
        self,
        stream_id: str,
        hello: Dict[str, Any],
        vllm: VLLMOnlinePrefillClient,
        engine_index: int,
        anchor_log_path: Optional[Path],
        on_result: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_early_finalized: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        alpha_lookup: Optional[AlphaLookup] = None,
        alpha_policy_template: Optional[AlphaPolicy] = None,
        finalize_concurrency: int = 2,
        rid_sink: Optional[Callable[[str], None]] = None,
        rid_drop: Optional[Callable[[str], None]] = None,
        admission_gate: Optional[AdmissionGate] = None,
    ) -> None:
        self.stream_id = stream_id
        self.hello = hello
        self.inference_mode = self._inference_mode_from_hello(hello)
        self.vllm = vllm
        self.engine_index = int(engine_index)
        self.anchor_log_path = anchor_log_path
        self.on_result = on_result
        self.on_early_finalized = on_early_finalized
        self.latency_tracker = latency_tracker
        self.alpha_lookup = alpha_lookup
        self.alpha_policy_template = alpha_policy_template or AlphaPolicy(alpha=1.0)
        self.rid_sink = rid_sink  # callback invoked on each vLLM session create
        self.rid_drop = rid_drop  # callback invoked when a session is confirmed finished
        self.admission_gate = admission_gate
        self.alpha_init_state = AlphaInitState(default=1.0, alpha_min=0.3)
        self.visual_memory_enabled = _bool_value(
            hello.get("visual_memory_merge"),
            False,
        )
        self.visual_memory_num_frames = max(1, _int_env("BAVA_VISUAL_MEMORY_NUM_FRAMES", 8))
        self.visual_memory_tokens_per_frame = max(
            1, _int_env("BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME", 32)
        )
        self.visual_memory_text_prefix = os.environ.get(
            "BAVA_VISUAL_MEMORY_TEXT_PREFIX", ""
        )
        self.visual_memory_id_prefix = os.environ.get(
            "BAVA_VISUAL_MEMORY_ID_PREFIX", "bava_mem"
        )
        self.visual_memory_strict = _bool_env("BAVA_VISUAL_MEMORY_STRICT", False)
        self.visual_memory_warm_prefix_cache = _bool_env(
            "BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE", True
        )
        self._visual_memory: Optional[Dict[str, Any]] = None
        self._visual_memory_version = 0
        self._visual_memory_last_request_id: Optional[str] = None
        self._visual_memory_last_decision_id: Optional[int] = None
        self._visual_memory_error: Optional[str] = None
        self._visual_memory_missing_warned = False
        self.flow = StreamFlowTracker(
            stream_id=stream_id,
            window_seconds=self._hello_float("window_seconds", 1.0),
            decision_window_seconds=self._hello_float(
                "decision_window_seconds",
                self._hello_float("window_seconds", 1.0),
            ),
            ema_alpha=float(os.environ.get("BAVA_FLOW_EMA_ALPHA", "0.2")),
        )
        self.windows: Dict[int, WindowState] = {}
        self.inflight_appends: int = 0
        self.inflight_windows: int = 0   # windows currently past the decode step
        self.last_append_ms: Optional[float] = None
        self.last_decode_ms: Optional[float] = None
        self.closed_ts = time.time()
        self._append_lock = asyncio.Lock()
        self._active_request_id: Optional[str] = None
        self._active_decision_id: int = 0
        self._active_opened_at: float = time.time()
        self._active_sent_frames: int = 0
        self._active_decode_ms: float = 0.0
        self._active_alpha_ms: float = 0.0
        self._active_jpeg_ms: float = 0.0
        self._active_append_ms: float = 0.0
        self._active_avg_dims: List[int] = []
        self._active_chunk_window_ids: List[int] = []
        self._active_early_finalized = False
        self.early_finalized_count = 0
        self.last_early_finalized_wall = 0.0
        self.last_early_finalized_decision_id: Optional[int] = None
        self.last_early_finalized_windows = 0
        self._sealed_window_id: Optional[int] = None
        self._last_create_admission_denied = False
        self._decode_prefix = DecodePrefix()
        self._op_tail: asyncio.Future = asyncio.get_running_loop().create_future()
        self._op_tail.set_result(None)
        self._ordered_tasks: Set[asyncio.Task] = set()
        self._closing = False
        # Back-pressure: only N windows of this stream may be processing vLLM
        # concurrently. The rest queue up in Semaphore order. Prevents a
        # runaway edge (high rho looped clip) from flooding the intake loop.
        self._finalize_sem = asyncio.Semaphore(max(1, finalize_concurrency))
        self.evicted_appended_windows: int = 0

    @staticmethod
    def _inference_mode_from_hello(hello: Dict[str, Any]) -> str:
        raw = (
            hello.get("inference_mode")
            or hello.get("mode")
            or os.environ.get("BAVA_INFERENCE_MODE")
            or "online_prefill"
        )
        mode = str(raw).strip().lower().replace("-", "_")
        if mode in {"completion", "chat_completion", "native_completion", "intake_completion"}:
            return "completion"
        return "online_prefill"

    def flow_snapshot(self) -> StreamFlowSnapshot:
        return self.flow.snapshot()

    def visual_memory_snapshot(self) -> Dict[str, Any]:
        mem = self._visual_memory if isinstance(self._visual_memory, dict) else None
        return {
            "enabled": bool(self.visual_memory_enabled),
            "warm_prefix_cache": bool(self.visual_memory_warm_prefix_cache),
            "version": int(self._visual_memory_version),
            "memory_id": str(mem.get("memory_id")) if mem and mem.get("memory_id") else None,
            "num_frames": int(mem.get("num_frames")) if mem and mem.get("num_frames") else None,
            "tokens_per_frame": (
                int(mem.get("tokens_per_frame"))
                if mem and mem.get("tokens_per_frame")
                else None
            ),
            "last_request_id": self._visual_memory_last_request_id,
            "last_decision_id": self._visual_memory_last_decision_id,
            "error": self._visual_memory_error,
        }

    def early_finalized_recent(self, cooldown_s: float) -> bool:
        if self.last_early_finalized_wall <= 0:
            return False
        if cooldown_s <= 0:
            return True
        return (time.time() - self.last_early_finalized_wall) <= float(cooldown_s)

    def schedule_ordered(self, op):
        """Run append/stream_end operations in control-frame arrival order."""
        if self._closing:
            fut = asyncio.get_running_loop().create_future()
            fut.set_result(None)
            log.debug("stream=%s ordered op ignored; session closing", self.stream_id)
            return fut
        prev = self._op_tail

        async def _runner():
            try:
                await prev
            except BaseException:
                pass
            if self._closing:
                return None
            return await op()

        task = asyncio.create_task(_runner())
        self._ordered_tasks.add(task)

        def _log_failure(done: asyncio.Future) -> None:
            self._ordered_tasks.discard(done)  # type: ignore[arg-type]
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                log.error(
                    "stream=%s ordered op failed",
                    self.stream_id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(_log_failure)
        self._op_tail = task
        return task

    def ensure_window(self, window_id: int, rho: float) -> WindowState:
        st = self.windows.get(window_id)
        if st is None:
            st = WindowState(
                stream_id=self.stream_id,
                window_id=window_id,
                rho=rho,
                opened_at=time.time(),
            )
            self.windows[window_id] = st
        return st

    def on_window_open(self, msg: Dict[str, Any]) -> None:
        wid = int(msg["window_id"])
        rho = float(msg.get("rho") or self.hello.get("rho") or 0.3)
        self.ensure_window(wid, rho)
        log.info("stream=%s window_open wid=%d rho=%.3f", self.stream_id, wid, rho)

    def on_packet(self, header: Dict[str, Any], payload: bytes) -> None:
        wid = int(header.get("window_id", -1))
        st = self.ensure_window(wid, float(header.get("rho") or 0.3))
        st.packets.append(
            PacketRec(
                seq=int(header["seq"]),
                gop_index=int(header.get("gop_index", -1)),
                gop_pos=int(header.get("gop_pos", 0)),
                is_idr=bool(header.get("is_idr")),
                payload=payload,
                header=header,
            )
        )
        self._maybe_log_anchor(header)

    def mark_window_close(self, msg: Dict[str, Any]) -> Optional[WindowState]:
        if self._closing:
            log.info(
                "stream=%s window_close ignored; session closing",
                self.stream_id,
            )
            return None
        wid = int(msg["window_id"])
        st = self.windows.get(wid)
        if st is None:
            log.warning("stream=%s window_close for unknown wid=%d", self.stream_id, wid)
            return None
        st.closed = True
        st.closed_at = time.time()
        recv_bytes = sum(len(p.payload) for p in st.packets)
        st.recv_bytes = recv_bytes
        edge_offer_bytes = _optional_int(msg.get("edge_offer_bytes"))
        edge_full_bytes = _optional_int(msg.get("edge_full_bytes"))
        edge_queue_bytes = _optional_int(
            msg.get("edge_queue_bytes", msg.get("edge_queue_size"))
        )
        edge_send_wait_ms = _optional_float(
            msg.get("edge_send_wait_ms_ewma", msg.get("edge_send_wait_ms"))
        )
        self.flow.observe_network_window(
            recv_bytes=recv_bytes,
            offer_bytes=edge_offer_bytes,
            full_bytes=edge_full_bytes,
            edge_queue_bytes=edge_queue_bytes,
            send_wait_ms=edge_send_wait_ms,
            window_seconds=self._window_seconds(),
        )
        log.info(
            "stream=%s window_close wid=%d packets=%d recv=%d offer=%s full=%s q=%s send_wait_ms=%s",
            self.stream_id,
            wid,
            len(st.packets),
            recv_bytes,
            edge_offer_bytes,
            edge_full_bytes,
            edge_queue_bytes,
            edge_send_wait_ms,
        )
        # Evict only windows that have already been appended to vLLM. Intake
        # must not change the native online-prefill input by dropping a closed
        # but unappended chunk; the edge/controller budget decides future
        # decision boundaries instead.
        max_queued = int(os.environ.get("BAVA_MAX_QUEUED_WINDOWS", "8"))
        if len(self.windows) > max_queued:
            old = sorted(self.windows.keys())[: len(self.windows) - max_queued]
            for oid in old:
                if oid == wid:
                    continue
                w = self.windows.get(oid)
                if w is not None and not w.closed:
                    continue
                if w is not None and w.appended:
                    # Already pushed to vLLM; only local metadata/payload cache
                    # is released.
                    self.windows.pop(oid, None)
                    self.evicted_appended_windows += 1
                    log.info(
                        "stream=%s evicted appended window metadata wid=%d (cache > %d)",
                        self.stream_id,
                        oid,
                        max_queued,
                    )
        if self._is_window_sealed(wid):
            st.dropped_after_finalize = True
            self.windows.pop(wid, None)
            log.info(
                "stream=%s window_close wid=%d ignored; already finalized_until=%d",
                self.stream_id,
                wid,
                int(self._sealed_window_id or -1),
            )
            return None
        return st

    def on_edge_stats(self, msg: Dict[str, Any]) -> None:
        """Optional edge-side network stats outside the normal window_close path."""
        recv_bytes = _optional_int(msg.get("cloud_recv_bytes")) or 0
        offer_bytes = _optional_int(msg.get("edge_offer_bytes"))
        full_bytes = _optional_int(msg.get("edge_full_bytes"))
        edge_queue_bytes = _optional_int(
            msg.get("edge_queue_bytes", msg.get("edge_queue_size"))
        )
        edge_send_wait_ms = _optional_float(
            msg.get("edge_send_wait_ms_ewma", msg.get("edge_send_wait_ms"))
        )
        window_seconds = _optional_float(msg.get("window_seconds")) or self._window_seconds()
        self.flow.observe_network_window(
            recv_bytes=recv_bytes,
            offer_bytes=offer_bytes,
            full_bytes=full_bytes,
            edge_queue_bytes=edge_queue_bytes,
            send_wait_ms=edge_send_wait_ms,
            window_seconds=window_seconds,
        )

    async def append_closed_window(self, st: WindowState) -> Optional[Dict[str, Any]]:
        if self._closing:
            st.dropped_after_finalize = True
            return None
        async with self._finalize_sem:
            self.inflight_windows += 1
            try:
                async with self._append_lock:
                    return await self._append_window_locked(st)
            finally:
                self.inflight_windows = max(0, self.inflight_windows - 1)

    async def on_stream_end(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._closing:
            log.info("stream=%s stream_end ignored; session closing", self.stream_id)
            return None
        stream_end_perf = time.perf_counter()
        edge_decision_id = int(msg.get("decision_id", self._active_decision_id))
        decision_id = max(edge_decision_id, int(self._active_decision_id))
        last_window_id_raw = msg.get("last_window_id")
        last_window_id = int(last_window_id_raw) if last_window_id_raw is not None else None
        if decision_id != edge_decision_id:
            log.info(
                "stream=%s stream_end stale decision=%d active_decision=%d; using active",
                self.stream_id,
                edge_decision_id,
                self._active_decision_id,
            )
        log.info(
            "stream=%s stream_end decision=%d last_window=%s",
            self.stream_id,
            decision_id,
            last_window_id,
        )
        async with self._finalize_sem:
            self.inflight_windows += 1
            try:
                async with self._append_lock:
                    pending = [
                        self.windows[wid]
                        for wid in sorted(self.windows)
                        if self.windows[wid].closed
                        and not self.windows[wid].appended
                        and not self.windows[wid].dropped_after_finalize
                        and (last_window_id is None or wid <= last_window_id)
                    ]
                    for st in pending:
                        await self._append_window_locked(st)
                    if self.inference_mode == "completion":
                        return await self._finish_completion_request_locked(
                            decision_id,
                            last_window_id,
                            stream_end_perf=stream_end_perf,
                        )
                    return await self._finish_active_request_locked(
                        decision_id,
                        last_window_id,
                        stream_end_perf=stream_end_perf,
                    )
            finally:
                self.inflight_windows = max(0, self.inflight_windows - 1)

    async def on_window_close(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        st = self.mark_window_close(msg)
        if st is None:
            return None
        return await self.append_closed_window(st)

    async def cleanup(self) -> None:
        self._closing = True
        drain_s = max(0.0, _float_env("BAVA_SESSION_CLEANUP_DRAIN_S", 2.0))
        if drain_s > 0:
            try:
                await asyncio.wait_for(asyncio.shield(self._op_tail), timeout=drain_s)
            except asyncio.TimeoutError:
                log.warning(
                    "stream=%s cleanup timed out waiting %.1fs for ordered ops",
                    self.stream_id,
                    drain_s,
                )
                await self._cancel_ordered_tasks_for_cleanup()
            except Exception as e:
                log.debug("stream=%s cleanup ordered op drain failed: %s", self.stream_id, e)

        lock_s = max(0.0, _float_env("BAVA_SESSION_CLEANUP_LOCK_TIMEOUT_S", 1.0))
        try:
            if lock_s > 0:
                await asyncio.wait_for(self._append_lock.acquire(), timeout=lock_s)
            else:
                await self._append_lock.acquire()
        except asyncio.TimeoutError:
            log.warning(
                "stream=%s cleanup skipped active abort; append lock busy",
                self.stream_id,
            )
            request_id = self._active_request_id
            if request_id is not None:
                await self._abort_request_best_effort(
                    request_id=request_id,
                    decision_id=self._active_decision_id,
                    phase="session_cleanup_lock_busy",
                )
            return
        try:
            request_id = self._active_request_id
            if request_id is None:
                return
            await self._abort_active_request_locked(
                request_id=request_id,
                decision_id=self._active_decision_id,
                last_window_id=(
                    self._active_chunk_window_ids[-1]
                    if self._active_chunk_window_ids
                    else self._sealed_window_id
                ),
                phase="session_cleanup",
            )
        finally:
            self._append_lock.release()

    # -------- internals --------

    async def _cancel_ordered_tasks_for_cleanup(self) -> None:
        tasks = [task for task in list(self._ordered_tasks) if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        cancel_s = max(0.0, _float_env("BAVA_SESSION_CLEANUP_CANCEL_S", 2.0))
        if cancel_s <= 0:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=cancel_s,
            )
        except asyncio.TimeoutError:
            log.warning(
                "stream=%s cleanup timed out waiting %.1fs for ordered op cancellation",
                self.stream_id,
                cancel_s,
            )

    def _current_alpha(self) -> float:
        if self.inference_mode == "completion":
            return 1.0
        if self.alpha_lookup is not None:
            try:
                return max(0.3, min(1.0, float(self.alpha_lookup(self.stream_id))))
            except Exception:
                pass
        return max(0.3, min(1.0, float(self.hello.get("alpha") or 1.0)))

    def _alpha_executor_mode(self) -> str:
        raw = _str_env("BAVA_ALPHA_EXECUTOR_MODE", "resize").lower()
        aliases = {
            "image": "resize",
            "pixel": "resize",
            "pixels": "resize",
            "none": "off",
            "disabled": "off",
            "disable": "off",
            "vllm": "tokenmerger",
            "token": "tokenmerger",
            "token_merger": "tokenmerger",
            "token-folding": "tokenmerger",
            "token_folding": "tokenmerger",
        }
        return aliases.get(raw, raw)

    def _use_tokenmerger_alpha(self) -> bool:
        return self.inference_mode != "completion" and self._alpha_executor_mode() in {
            "tokenmerger",
            "both",
        }

    def _tokenmerger_block_t(self) -> int:
        return max(
            1,
            _int_env(
                "BAVA_TOKEN_MERGER_BLOCK_T",
                _int_env("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_T", 1),
            ),
        )

    def _tokenmerger_block_hw(self) -> int:
        return max(
            1,
            _int_env(
                "BAVA_TOKEN_MERGER_BLOCK_HW",
                _int_env("VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_HW", 2),
            ),
        )

    def _hello_float(self, name: str, default: float) -> float:
        try:
            return float(self.hello.get(name) or default)
        except Exception:
            return float(default)

    def _window_seconds(self) -> float:
        return max(1e-3, self._hello_float("window_seconds", 1.0))

    def _result_poll_interval_s(self) -> float:
        return max(0.05, _float_env("BAVA_RESULT_POLL_INTERVAL_S", 0.25))

    def _result_timeout_s(self) -> float:
        if self._active_early_finalized:
            return max(0.1, _float_env("BAVA_EF_RESULT_TIMEOUT_S", 30.0))
        return max(0.1, _float_env("BAVA_RESULT_TIMEOUT_S", 120.0))

    def _request_overhead_tokens(self) -> int:
        prompt = str(self.hello.get("prompt") or "")
        try:
            prompt_tokens = int(self.hello.get("prompt_tokens") or max(8, len(prompt) // 3))
        except Exception:
            prompt_tokens = max(8, len(prompt) // 3)
        try:
            max_tokens = int(self.hello.get("max_tokens") or 24)
        except Exception:
            max_tokens = 24
        block_overhead = int(os.environ.get("BAVA_BUDGET_BLOCK_OVERHEAD_TOKENS", "64"))
        return max(0, prompt_tokens) + max(0, max_tokens) + max(0, block_overhead)

    def _is_window_sealed(self, window_id: int) -> bool:
        return self._sealed_window_id is not None and int(window_id) <= int(self._sealed_window_id)

    def _seal_windows_locked(self, last_window_id: Optional[int], *, reason: str) -> None:
        if last_window_id is None:
            return
        last = int(last_window_id)
        if self._sealed_window_id is None or last > self._sealed_window_id:
            self._sealed_window_id = last
        for wid in list(self.windows.keys()):
            if wid > last:
                continue
            st = self.windows[wid]
            if not st.appended:
                st.dropped_after_finalize = True
                log.info(
                    "stream=%s wid=%d dropped_after_finalize reason=%s finalized_until=%d",
                    self.stream_id,
                    wid,
                    reason,
                    int(self._sealed_window_id),
                )
            self.windows.pop(wid, None)

    async def _abort_active_request_locked(
        self,
        *,
        request_id: str,
        decision_id: int,
        last_window_id: Optional[int],
        phase: str,
    ) -> bool:
        log.warning(
            "stream=%s engine=%d decision=%d request=%s aborting active vLLM session phase=%s last_window=%s",
            self.stream_id,
            self.engine_index,
            decision_id,
            request_id,
            phase,
            last_window_id,
        )
        aborted = await self.vllm.abort(request_id)
        if aborted:
            released = self.flow.observe_release(
                decision_duration_s=max(1e-3, time.time() - self._active_opened_at)
            )
            if self.rid_drop is not None:
                try:
                    self.rid_drop(request_id)
                except Exception:
                    pass
            log.info(
                "stream=%s engine=%d decision=%d request=%s aborted active session phase=%s released_tokens_est=%d",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
                phase,
                released,
            )
        else:
            log.warning(
                "stream=%s engine=%d decision=%d request=%s abort failed phase=%s",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
                phase,
            )
        self._seal_windows_locked(last_window_id, reason=phase)
        self._reset_active_request_locked(next_decision_id=decision_id + 1)
        return bool(aborted)

    async def _abort_request_best_effort(
        self,
        *,
        request_id: str,
        decision_id: int,
        phase: str,
    ) -> bool:
        log.warning(
            "stream=%s engine=%d decision=%d request=%s best-effort abort phase=%s",
            self.stream_id,
            self.engine_index,
            decision_id,
            request_id,
            phase,
        )
        try:
            aborted = await self.vllm.abort(request_id)
        except Exception as e:
            log.warning(
                "stream=%s engine=%d decision=%d request=%s best-effort abort failed phase=%s error=%s",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
                phase,
                e,
            )
            return False
        if aborted and self.rid_drop is not None:
            try:
                self.rid_drop(request_id)
            except Exception:
                pass
        log.info(
            "stream=%s engine=%d decision=%d request=%s best-effort abort ok=%s phase=%s",
            self.stream_id,
            self.engine_index,
            decision_id,
            request_id,
            bool(aborted),
            phase,
        )
        return bool(aborted)

    @staticmethod
    def _alpha_init_hint_from_response(resp: Any) -> Optional[float]:
        if not isinstance(resp, dict):
            return None
        value = resp.get("alpha_init_hint")
        if value is None and isinstance(resp.get("metadata"), dict):
            value = resp["metadata"].get("alpha_init_hint")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _response_has_early_finalized(resp: Any) -> bool:
        if isinstance(resp, dict):
            if bool(resp.get("early_finalized")):
                return True
            status = str(resp.get("status") or "").strip().lower()
            if status == "early_finalized":
                return True
            return any(StreamSession._response_has_early_finalized(v) for v in resp.values())
        if isinstance(resp, (list, tuple)):
            return any(StreamSession._response_has_early_finalized(v) for v in resp)
        return False

    @staticmethod
    def _exception_mentions_early_finalized(exc: Exception) -> bool:
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                if StreamSession._response_has_early_finalized(resp.json()):
                    return True
            except Exception:
                pass
            try:
                text = str(getattr(resp, "text", "") or "")
                if "early_finalized" in text or "early finalized" in text.lower():
                    return True
            except Exception:
                pass
        return "early_finalized" in str(exc) or "early finalized" in str(exc).lower()

    def _mark_early_finalized(
        self,
        phase: str,
        decision_id: Optional[int] = None,
        active_windows: Optional[int] = None,
    ) -> None:
        did = int(self._active_decision_id if decision_id is None else decision_id)
        first_for_decision = not self._active_early_finalized
        active_window_count = max(
            0,
            int(
                len(self._active_chunk_window_ids)
                if active_windows is None
                else active_windows
            ),
        )
        if first_for_decision:
            self.early_finalized_count += 1
            self.last_early_finalized_windows = active_window_count
        self._active_early_finalized = True
        self.last_early_finalized_wall = time.time()
        self.last_early_finalized_decision_id = did
        log.warning(
            "stream=%s engine=%d decision=%d early_finalized phase=%s count=%d active_windows=%d",
            self.stream_id,
            self.engine_index,
            did,
            phase,
            self.early_finalized_count,
            active_window_count,
        )
        if first_for_decision and self.on_early_finalized is not None:
            payload = {
                "kind": "early_finalized",
                "stream_id": self.stream_id,
                "engine_index": self.engine_index,
                "decision_id": did,
                "phase": phase,
                "early_finalized_count": int(self.early_finalized_count),
                "active_windows": int(active_window_count),
            }
            try:
                asyncio.create_task(self.on_early_finalized(payload))
            except RuntimeError:
                pass

    def _update_alpha_init_from_response(self, resp: Any, window_id: Optional[int]) -> None:
        if window_id is None:
            return
        hint = self._alpha_init_hint_from_response(resp)
        before_alpha, before_version = self.alpha_init_state.get()
        self.alpha_init_state.update_from_response(hint, int(window_id))
        after_alpha, after_version = self.alpha_init_state.get()
        if after_version != before_version:
            log.info(
                "stream=%s alpha_init update wid=%d %.3f(v%d)->%.3f(v%d)",
                self.stream_id,
                int(window_id),
                before_alpha,
                before_version,
                after_alpha,
                after_version,
            )

    def _visual_memory_id_for_decision(self, decision_id: int) -> str:
        return f"{self.visual_memory_id_prefix}:{self.stream_id}:d{int(decision_id)}"

    @staticmethod
    def _extract_visual_memory_from_response(resp: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(resp, dict):
            return None
        raw = resp.get("visual_memory")
        if isinstance(raw, dict) and raw.get("data"):
            return raw
        return None

    def _update_visual_memory_from_response(
        self,
        resp: Any,
        *,
        decision_id: int,
        request_id: Optional[str],
    ) -> None:
        if not self.visual_memory_enabled:
            return

        mem = self._extract_visual_memory_from_response(resp)
        error = None
        if isinstance(resp, dict):
            error = resp.get("visual_memory_error")
        if mem is not None:
            self._visual_memory = mem
            self._visual_memory_version += 1
            self._visual_memory_last_request_id = request_id
            self._visual_memory_last_decision_id = int(decision_id)
            self._visual_memory_error = None
            self._visual_memory_missing_warned = False
            log.info(
                "stream=%s decision=%d visual memory updated memory_id=%s frames=%s tpf=%s",
                self.stream_id,
                int(decision_id),
                mem.get("memory_id"),
                mem.get("num_frames"),
                mem.get("tokens_per_frame"),
            )
            return

        if error:
            self._visual_memory_error = str(error)
            self._visual_memory = None
            self._visual_memory_last_request_id = request_id
            self._visual_memory_last_decision_id = int(decision_id)
            log.warning(
                "stream=%s decision=%d visual memory export error: %s",
                self.stream_id,
                int(decision_id),
                error,
            )
            return

        self._visual_memory = None
        self._visual_memory_error = None
        self._visual_memory_last_request_id = request_id
        self._visual_memory_last_decision_id = int(decision_id)
        if not self._visual_memory_missing_warned:
            self._visual_memory_missing_warned = True
            log.info(
                "stream=%s decision=%d visual memory missing in response; "
                "memory carry-over disabled until a memory export appears",
                self.stream_id,
                int(decision_id),
            )

    def _maybe_log_anchor(self, header: Dict[str, Any]) -> None:
        if self.anchor_log_path is None:
            return
        try:
            import json
            line = {
                "t": time.time(),
                "stream_id": self.stream_id,
                "window_id": header.get("window_id"),
                "seq": header.get("seq"),
                "is_idr": header.get("is_idr"),
                "score": header.get("score"),
                "anchor": header.get("anchor"),
            }
            with self.anchor_log_path.open("a") as f:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
        except Exception as e:
            log.debug("anchor log write failed: %s", e)

    @staticmethod
    def _ordered_packets(st: WindowState) -> List[PacketRec]:
        return sorted(st.packets, key=lambda p: p.seq)

    @staticmethod
    def _concat_packets(packets: Sequence[PacketRec]) -> bytes:
        return b"".join(p.payload for p in packets)

    def _concat_annexb(self, st: WindowState) -> bytes:
        return self._concat_packets(self._ordered_packets(st))

    async def _decode_payload(self, payload: bytes) -> tuple[List[BgrFrame], float]:
        return await asyncio.to_thread(decode_to_bgr, payload)

    async def _decode_window_bgr(self, st: WindowState) -> Optional[tuple[List[BgrFrame], float, str]]:
        ordered = self._ordered_packets(st)
        data = self._concat_packets(ordered)
        if not data:
            return None

        first_idr_idx: Optional[int] = None
        for i, pkt in enumerate(ordered):
            if pkt.is_idr:
                first_idr_idx = i
                break
        starts_with_idr = first_idr_idx == 0
        suffix_from_idr = (
            self._concat_packets(ordered[first_idr_idx:])
            if first_idr_idx is not None
            else b""
        )

        candidates: List[tuple[str, bytes, int]] = []
        if starts_with_idr:
            candidates.append(("raw_idr", data, 0))
            if self._decode_prefix.payload:
                candidates.append(
                    (
                        "prefix_then_idr",
                        self._decode_prefix.payload + data,
                        self._decode_prefix.frame_count,
                    )
                )
        else:
            if self._decode_prefix.payload:
                candidates.append(
                    (
                        "prefix",
                        self._decode_prefix.payload + data,
                        self._decode_prefix.frame_count,
                    )
                )
            if suffix_from_idr:
                candidates.append(("idr_suffix", suffix_from_idr, 0))
            candidates.append(("raw", data, 0))

        seen_payloads = set()
        last_error: Optional[Exception] = None
        for mode, payload, skip_frames in candidates:
            if not payload or payload in seen_payloads:
                continue
            seen_payloads.add(payload)
            try:
                decoded, decode_ms = await self._decode_payload(payload)
            except Exception as e:
                last_error = e
                log.debug(
                    "stream=%s wid=%d decode candidate=%s failed: %s",
                    self.stream_id,
                    st.window_id,
                    mode,
                    e,
                )
                continue
            if skip_frames > 0:
                current = decoded[skip_frames:] if len(decoded) > skip_frames else []
            else:
                current = decoded
            if not current:
                log.debug(
                    "stream=%s wid=%d decode candidate=%s produced no new frames "
                    "(decoded=%d skip=%d)",
                    self.stream_id,
                    st.window_id,
                    mode,
                    len(decoded),
                    skip_frames,
                )
                continue

            extra_decode_ms = await self._refresh_decode_prefix(
                st=st,
                first_idr_idx=first_idr_idx,
                suffix_from_idr=suffix_from_idr,
                successful_payload=payload,
                successful_frame_count=len(decoded),
                mode=mode,
            )
            current = [
                BgrFrame(index=i, pts_s=frame.pts_s, bgr=frame.bgr)
                for i, frame in enumerate(current)
            ]
            return current, decode_ms + extra_decode_ms, mode

        detail = f": {last_error}" if last_error is not None else ""
        log.warning(
            "stream=%s wid=%d decode failed candidates=%s has_prefix=%s has_idr=%s%s",
            self.stream_id,
            st.window_id,
            [c[0] for c in candidates],
            bool(self._decode_prefix.payload),
            first_idr_idx is not None,
            detail,
        )
        return None

    async def _refresh_decode_prefix(
        self,
        *,
        st: WindowState,
        first_idr_idx: Optional[int],
        suffix_from_idr: bytes,
        successful_payload: bytes,
        successful_frame_count: int,
        mode: str,
    ) -> float:
        max_bytes = int(os.environ.get("BAVA_DECODE_PREFIX_MAX_BYTES", str(128 * 1024 * 1024)))
        extra_ms = 0.0

        if first_idr_idx is not None and suffix_from_idr:
            if mode == "idr_suffix" or (first_idr_idx == 0 and mode in {"raw_idr", "raw"}):
                prefix = DecodePrefix(
                    payload=suffix_from_idr,
                    frame_count=successful_frame_count,
                    anchor_window_id=st.window_id,
                )
            else:
                try:
                    frames, extra_ms = await self._decode_payload(suffix_from_idr)
                    prefix = DecodePrefix(
                        payload=suffix_from_idr,
                        frame_count=len(frames),
                        anchor_window_id=st.window_id,
                    )
                except Exception as e:
                    log.debug(
                        "stream=%s wid=%d decode prefix refresh from IDR failed: %s; "
                        "keeping successful payload",
                        self.stream_id,
                        st.window_id,
                        e,
                    )
                    prefix = DecodePrefix(
                        payload=successful_payload,
                        frame_count=successful_frame_count,
                        anchor_window_id=self._decode_prefix.anchor_window_id,
                    )
        else:
            prefix = DecodePrefix(
                payload=successful_payload,
                frame_count=successful_frame_count,
                anchor_window_id=self._decode_prefix.anchor_window_id,
            )

        if max_bytes > 0 and len(prefix.payload) > max_bytes:
            log.warning(
                "stream=%s wid=%d decode prefix too large (%d > %d bytes); resetting until next IDR",
                self.stream_id,
                st.window_id,
                len(prefix.payload),
                max_bytes,
            )
            self._decode_prefix = DecodePrefix()
        else:
            self._decode_prefix = prefix
        return extra_ms

    def _scores_aligned(self, st: WindowState, kept_frames: Sequence[BgrFrame]) -> List[float]:
        """Line up packet scores (one per selected packet) with decoded frames.

        The number of decoded frames ≤ number of packets because multi-NAL
        packets can coalesce or some packets contribute only SPS/PPS. We use a
        linear interpolation by index — good enough for anchor-weighted α.
        """
        scores = [float(p.header.get("score") or 0.0) for p in sorted(st.packets, key=lambda p: p.seq)]
        if not scores or not kept_frames:
            return []
        n_s = len(scores)
        n_f = len(kept_frames)
        if n_s == n_f:
            return scores
        out: List[float] = []
        for f in kept_frames:
            j = min(n_s - 1, max(0, int(round(f.index * (n_s - 1) / max(1, n_f - 1)))))
            out.append(scores[j])
        return out

    @staticmethod
    def _jpeg_encode_batch(bgr_list: Sequence) -> List[Dict[str, str]]:
        """CPU-bound — designed to run inside asyncio.to_thread."""
        out: List[Dict[str, str]] = []
        for bgr in bgr_list:
            out.append({"data": encode_bgr_to_jpeg_data_uri(bgr)})
        return out

    @staticmethod
    def _is_conflict_error(exc: Exception) -> bool:
        resp = getattr(exc, "response", None)
        return int(getattr(resp, "status_code", 0) or 0) == 409

    @staticmethod
    def _is_lost_append_response_error(exc: Exception) -> bool:
        return exc.__class__.__name__ in {
            "ConnectError",
            "ConnectTimeout",
            "NetworkError",
            "PoolTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "TransportError",
            "WriteError",
            "WriteTimeout",
        }

    async def _recover_lost_append_response_locked(
        self,
        *,
        request_id: str,
        decision_id: int,
        expected_appended_frames: int,
        stream_end: bool,
        phase: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            poll = await self.vllm.poll(request_id)
        except Exception as poll_error:
            log.warning(
                "stream=%s engine=%d decision=%d request=%s lost append response recovery poll unavailable phase=%s poll_error=%s",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
                phase,
                poll_error,
            )
            return None

        raw = dict(poll.raw or {})
        try:
            appended_frames = int(raw.get("appended_frames") or 0)
        except (TypeError, ValueError):
            appended_frames = 0
        status = str(raw.get("status") or "").strip().lower()
        if stream_end:
            applied = bool(
                raw.get("stream_end_received")
                or raw.get("decode_started")
                or raw.get("finished")
                or raw.get("early_finalized")
                or status in {"waiting_for_decode", "decoding", "finished", "early_finalized"}
            )
            if appended_frames and expected_appended_frames:
                applied = applied and appended_frames >= expected_appended_frames
        else:
            applied = appended_frames >= expected_appended_frames

        if not applied:
            log.warning(
                "stream=%s engine=%d decision=%d request=%s lost append response recovery unconfirmed phase=%s status=%s appended_frames=%d expected_frames=%d stream_end_received=%s",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
                phase,
                status,
                appended_frames,
                expected_appended_frames,
                bool(raw.get("stream_end_received")),
            )
            return None

        log.warning(
            "stream=%s engine=%d decision=%d request=%s recovered lost append response phase=%s status=%s appended_frames=%d expected_frames=%d stream_end_received=%s",
            self.stream_id,
            self.engine_index,
            decision_id,
            request_id,
            phase,
            status,
            appended_frames,
            expected_appended_frames,
            bool(raw.get("stream_end_received")),
        )
        return raw

    def _hard_frame_cap(self) -> int:
        """Emergency/debug cap only.

        `BAVA_MAX_FRAMES_PER_WINDOW` describes the edge-side per-chunk budget
        for cost modeling and budget allocation. It must not cause intake to
        drop frames. If a local operator needs a last-resort intake safety cap,
        they can opt into `BAVA_INTAKE_HARD_FRAME_CAP`.
        """
        return max(0, _int_env("BAVA_INTAKE_HARD_FRAME_CAP", 0))

    async def _prepare_window_frames(
        self,
        st: WindowState,
    ) -> Optional[tuple[List[Dict[str, str]], float, float, float, float, int, int]]:
        if not st.packets:
            return None
        decoded = await self._decode_window_bgr(st)
        if decoded is None:
            return None
        frames, decode_ms, decode_mode = decoded
        if not frames:
            log.warning("stream=%s wid=%d: decoded 0 frames", self.stream_id, st.window_id)
            return None
        st.decoded_frames = len(frames)
        st.decode_ms = decode_ms
        self.last_decode_ms = decode_ms
        log.info(
            "stream=%s wid=%d decoded=%d frames in %.1fms mode=%s",
            self.stream_id,
            st.window_id,
            len(frames),
            decode_ms,
            decode_mode,
        )

        raw_tokens_est = sum(
            tokens_for_image_shape(f.bgr.shape[0], f.bgr.shape[1])
            for f in frames
        )
        st.raw_tokens_est = raw_tokens_est
        self.flow.observe_edge_window(
            raw_tokens=raw_tokens_est,
            window_seconds=self._window_seconds(),
        )

        hard_frame_cap = self._hard_frame_cap()
        if hard_frame_cap > 0 and len(frames) > hard_frame_cap:
            frames = subsample_uniform(frames, hard_frame_cap)
            log.warning(
                "stream=%s wid=%d hard-capped by BAVA_INTAKE_HARD_FRAME_CAP to %d",
                self.stream_id,
                st.window_id,
                len(frames),
            )

        if self.inference_mode == "completion" and frames:
            prompt_tokens = max(8, len(str(self.hello.get("prompt") or "")) // 3)
            try:
                prompt_tokens = int(self.hello.get("prompt_tokens") or prompt_tokens)
            except Exception:
                pass
            try:
                max_tokens = int(self.hello.get("max_tokens") or 24)
            except Exception:
                max_tokens = 24
            context_tokens = max(1, _int_env("BAVA_COMPLETION_CONTEXT_TOKENS", 8192))
            reserve = prompt_tokens + max_tokens + int(os.environ.get("BAVA_BUDGET_BLOCK_OVERHEAD_TOKENS", "64"))
            available = max(1, context_tokens - reserve)
            per_frame_tokens = max(
                1,
                max(tokens_for_image_shape(f.bgr.shape[0], f.bgr.shape[1]) for f in frames),
            )
            context_cap = max(1, available // per_frame_tokens)
            if len(frames) > context_cap:
                frames = subsample_uniform(frames, context_cap)
                log.info(
                    "stream=%s wid=%d subsampled to %d (completion context cap=%d tokens/frame≈%d)",
                    self.stream_id,
                    st.window_id,
                    len(frames),
                    context_cap,
                    per_frame_tokens,
                )

        # --- α executor: resize in pixel space or forward α to vLLM tokenmerger ---
        alpha = self._current_alpha()
        st.alpha_used = alpha
        bgr_list = [f.bgr for f in frames]
        alpha_mode = self._alpha_executor_mode()
        if alpha_mode in {"resize", "both"}:
            policy = AlphaPolicy(
                alpha=alpha,
                min_side=self.alpha_policy_template.min_side,
                max_side=self.alpha_policy_template.max_side,
                align=self.alpha_policy_template.align,
                skip_threshold=self.alpha_policy_template.skip_threshold,
                score_power=self.alpha_policy_template.score_power,
                per_frame_floor=self.alpha_policy_template.per_frame_floor,
            )
            weighted = os.environ.get("BAVA_ALPHA_WEIGHTED", "0") != "0"
            scores_aligned = self._scores_aligned(st, frames) if weighted else None
            t_alpha = time.perf_counter()
            if weighted and scores_aligned:
                prepared_bgr = await asyncio.to_thread(
                    apply_alpha_weighted, bgr_list, scores_aligned, policy
                )
            else:
                prepared_bgr = await asyncio.to_thread(apply_alpha_uniform, bgr_list, policy)
            alpha_ms = (time.perf_counter() - t_alpha) * 1000.0
        else:
            prepared_bgr = bgr_list
            alpha_ms = 0.0

        # JPEG-encode (runs in a thread; all the CPU work for N frames at once)
        t_jpeg = time.perf_counter()
        jpeg_frames: List[Dict[str, str]] = await asyncio.to_thread(
            StreamSession._jpeg_encode_batch, prepared_bgr
        )
        st.prepared_frames = list(jpeg_frames)
        dims: List[int] = []
        for bgr in prepared_bgr:
            h, w = bgr.shape[:2]
            dims.append(int((h * w) ** 0.5))
        if alpha_mode in {"tokenmerger"}:
            kv_tokens_est = max(len(prepared_bgr), math.ceil(raw_tokens_est * alpha))
        elif alpha_mode in {"off"}:
            kv_tokens_est = raw_tokens_est
        else:
            kv_tokens_est = sum(
                tokens_for_image_shape(bgr.shape[0], bgr.shape[1])
                for bgr in prepared_bgr
            )
        st.kv_tokens_est = kv_tokens_est
        jpeg_ms = (time.perf_counter() - t_jpeg) * 1000.0
        if dims:
            st.avg_dim = int(sum(dims) / len(dims))
        st.alpha_ms = alpha_ms
        st.jpeg_ms = jpeg_ms
        log.info(
            "stream=%s engine=%d wid=%d alpha=%.3f raw_tok≈%d kv_tok≈%d alpha_mode=%s resize=%.1fms jpeg=%.1fms avg_side≈%s",
            self.stream_id,
            self.engine_index,
            st.window_id,
            alpha,
            raw_tokens_est,
            kv_tokens_est,
            alpha_mode,
            alpha_ms,
            jpeg_ms,
            st.avg_dim,
        )
        return jpeg_frames, alpha, decode_ms, alpha_ms, jpeg_ms, raw_tokens_est, kv_tokens_est

    async def _ensure_active_request_locked(self, decision_id: Optional[int] = None) -> Optional[str]:
        if self._closing:
            log.info(
                "stream=%s active request create skipped; session closing",
                self.stream_id,
            )
            return None
        if decision_id is not None:
            self._active_decision_id = int(decision_id)
        if self._active_request_id is not None:
            return self._active_request_id
        self._last_create_admission_denied = False
        if self.admission_gate is not None:
            ok, reason = self.admission_gate.decide_create(
                delta_tokens=self._request_overhead_tokens()
            )
            if not ok:
                self._last_create_admission_denied = True
                log.warning(
                    "stream=%s admission deny phase=create reason=%s",
                    self.stream_id,
                    reason,
                )
                if self.on_result is not None:
                    try:
                        await self.on_result(
                            {
                                "kind": "admission_deny",
                                "stream_id": self.stream_id,
                                "phase": "create",
                                "reason": reason,
                            }
                        )
                    except Exception:
                        pass
                return None

        request_id = f"{self.stream_id}-d{self._active_decision_id}-{uuid.uuid4().hex[:6]}"
        self._active_opened_at = time.time()
        self._active_sent_frames = 0
        self._active_decode_ms = 0.0
        self._active_alpha_ms = 0.0
        self._active_jpeg_ms = 0.0
        self._active_append_ms = 0.0
        self._active_avg_dims = []
        self._active_chunk_window_ids = []
        visual_memory = self._visual_memory if self.visual_memory_enabled else None
        export_memory_id = self._visual_memory_id_for_decision(self._active_decision_id)
        try:
            await self.vllm.create_session(
                request_id=request_id,
                model=str(self.hello.get("model") or ""),
                prompt=str(self.hello.get("prompt") or ""),
                max_tokens=int(self.hello.get("max_tokens") or 24),
                visual_memory=visual_memory,
                export_visual_memory=bool(self.visual_memory_enabled),
                export_visual_memory_num_frames=self.visual_memory_num_frames,
                export_visual_memory_tokens_per_frame=self.visual_memory_tokens_per_frame,
                export_visual_memory_id=export_memory_id,
                export_visual_memory_text_prefix=self.visual_memory_text_prefix,
                warm_visual_memory_prefix_cache=bool(
                    self.visual_memory_enabled
                    and self.visual_memory_warm_prefix_cache
                ),
            )
        except Exception as e:
            resp = getattr(e, "response", None)
            status = int(getattr(resp, "status_code", 0) or 0)
            retry_without_memory = (
                self.visual_memory_enabled
                and not self.visual_memory_strict
                and status in {400, 422}
            )
            if not retry_without_memory:
                log.exception("create_session failed: %s", e)
                return None
            self._visual_memory = None
            visual_memory = None
            self._visual_memory_error = f"create_session rejected visual memory fields: {e}"
            log.warning(
                "stream=%s decision=%d create_session rejected visual memory fields "
                "(status=%s); retrying without memory/export",
                self.stream_id,
                self._active_decision_id,
                status,
            )
            try:
                await self.vllm.create_session(
                    request_id=request_id,
                    model=str(self.hello.get("model") or ""),
                    prompt=str(self.hello.get("prompt") or ""),
                    max_tokens=int(self.hello.get("max_tokens") or 24),
                )
            except Exception as retry_error:
                log.exception("create_session fallback failed: %s", retry_error)
                return None
        self._active_request_id = request_id
        if visual_memory is not None:
            log.info(
                "stream=%s decision=%d attached visual memory memory_id=%s",
                self.stream_id,
                self._active_decision_id,
                visual_memory.get("memory_id"),
            )
        if self.rid_sink is not None:
            try:
                self.rid_sink(request_id)
            except Exception:
                pass
        self.flow.observe_request_open(self._request_overhead_tokens())
        return request_id

    async def _append_window_locked(self, st: WindowState) -> Optional[Dict[str, Any]]:
        if self._closing:
            st.dropped_after_finalize = True
            log.info(
                "stream=%s wid=%d append skipped; session closing",
                self.stream_id,
                st.window_id,
            )
            return None
        if st.appended or st.dropped_after_finalize:
            return None
        if self._is_window_sealed(st.window_id):
            st.dropped_after_finalize = True
            log.info(
                "stream=%s wid=%d append skipped; finalized_until=%d active_decision=%d",
                self.stream_id,
                st.window_id,
                int(self._sealed_window_id or -1),
                self._active_decision_id,
            )
            return None
        prepared = await self._prepare_window_frames(st)
        if prepared is None:
            st.appended = True
            return None
        jpeg_frames, alpha, decode_ms, alpha_ms, jpeg_ms, raw_tokens_est, kv_tokens_est = prepared
        if self.inference_mode == "completion":
            st.sent_frames += len(jpeg_frames)
            st.appended = True
            log.info(
                "stream=%s engine=%d wid=%d prepared completion frames=%d decision=%d raw_tok≈%d kv_tok≈%d",
                self.stream_id,
                self.engine_index,
                st.window_id,
                len(jpeg_frames),
                self._active_decision_id,
                raw_tokens_est,
                kv_tokens_est,
            )
            return None
        request_id = await self._ensure_active_request_locked()
        if request_id is None:
            if self._last_create_admission_denied:
                st.appended = True
                self._last_create_admission_denied = False
            return None
        self._last_create_admission_denied = False
        st.request_id = request_id
        if self.admission_gate is not None:
            ok, reason = self.admission_gate.decide_append(delta_tokens=kv_tokens_est)
            if not ok:
                log.warning(
                    "stream=%s admission deny phase=append reason=%s wid=%d",
                    self.stream_id,
                    reason,
                    st.window_id,
                )
                await self._finish_active_request_locked(
                    self._active_decision_id,
                    st.window_id,
                    send_stream_end=False,
                    abort_instead_of_poll=True,
                )
                if self.on_result is not None:
                    try:
                        await self.on_result(
                            {
                                "kind": "admission_deny",
                                "stream_id": self.stream_id,
                                "phase": "append",
                                "reason": reason,
                                "window_id": st.window_id,
                            }
                        )
                    except Exception:
                        pass
                st.appended = True
                return None
        t_append = time.perf_counter()
        self.inflight_appends += 1
        tokenmerger_kwargs: Dict[str, Any] = {}
        if self._use_tokenmerger_alpha() and 0.0 < alpha < 1.0:
            tokenmerger_kwargs = {
                "visual_token_merger_alpha": alpha,
                "visual_token_merger_block_t": self._tokenmerger_block_t(),
                "visual_token_merger_block_hw": self._tokenmerger_block_hw(),
            }
            log.info(
                "stream=%s engine=%d wid=%d tokenmerger_alpha=%.3f block_t=%d block_hw=%d",
                self.stream_id,
                self.engine_index,
                st.window_id,
                alpha,
                tokenmerger_kwargs["visual_token_merger_block_t"],
                tokenmerger_kwargs["visual_token_merger_block_hw"],
            )
        expected_appended_frames = self._active_sent_frames + len(jpeg_frames)
        try:
            append_resp = await self.vllm.append_frames(
                request_id=request_id,
                frames=jpeg_frames,
                stream_end=False,
                **tokenmerger_kwargs,
            )
            self._update_alpha_init_from_response(append_resp, st.window_id)
            append_response_early_finalized = self._response_has_early_finalized(append_resp)
        except Exception as e:
            self.inflight_appends = max(0, self.inflight_appends - 1)
            if self._is_conflict_error(e):
                if self._exception_mentions_early_finalized(e):
                    self._mark_early_finalized(
                        "append_conflict",
                        active_windows=len(self._active_chunk_window_ids),
                    )
                log.warning(
                    "stream=%s wid=%d append conflict on %s; closing active decision early",
                    self.stream_id,
                    st.window_id,
                    request_id,
                )
                await self._finish_active_request_locked(
                    self._active_decision_id,
                    st.window_id,
                    send_stream_end=False,
                )
                return None
            else:
                recovered_resp = None
                if self._is_lost_append_response_error(e):
                    recovered_resp = await self._recover_lost_append_response_locked(
                        request_id=request_id,
                        decision_id=self._active_decision_id,
                        expected_appended_frames=expected_appended_frames,
                        stream_end=False,
                        phase="append_response_lost",
                    )
                if recovered_resp is None:
                    log.exception(
                        "append failed stream=%s engine=%d decision=%d wid=%d request=%s: %s",
                        self.stream_id,
                        self.engine_index,
                        self._active_decision_id,
                        st.window_id,
                        request_id,
                        e,
                    )
                    st.dropped_after_finalize = True
                    await self._abort_active_request_locked(
                        request_id=request_id,
                        decision_id=self._active_decision_id,
                        last_window_id=st.window_id,
                        phase="append_error",
                    )
                    return None
                append_resp = recovered_resp
                self._update_alpha_init_from_response(append_resp, st.window_id)
                append_response_early_finalized = self._response_has_early_finalized(
                    append_resp
                )
        self.inflight_appends = max(0, self.inflight_appends - 1)
        st.sent_frames += len(jpeg_frames)
        append_ms = (time.perf_counter() - t_append) * 1000.0
        self.last_append_ms = append_ms
        st.append_ms = append_ms
        st.appended = True
        self.flow.observe_kv_window(
            kv_tokens=kv_tokens_est,
            window_seconds=self._window_seconds(),
        )
        self._active_sent_frames += len(jpeg_frames)
        self._active_decode_ms += decode_ms
        self._active_alpha_ms += alpha_ms
        self._active_jpeg_ms += jpeg_ms
        self._active_append_ms += append_ms
        if st.avg_dim is not None:
            self._active_avg_dims.append(st.avg_dim)
        self._active_chunk_window_ids.append(st.window_id)
        log.info(
            "stream=%s engine=%d wid=%d appended frames=%d decision=%d raw_tok≈%d kv_tok≈%d append_ms=%.1f",
            self.stream_id,
            self.engine_index,
            st.window_id,
            len(jpeg_frames),
            self._active_decision_id,
            raw_tokens_est,
            kv_tokens_est,
            append_ms,
        )
        if append_response_early_finalized:
            self._mark_early_finalized(
                "append_response",
                active_windows=len(self._active_chunk_window_ids),
            )
            log.warning(
                "stream=%s wid=%d request=%s append response early_finalized; closing active decision",
                self.stream_id,
                st.window_id,
                request_id,
            )
            await self._finish_active_request_locked(
                self._active_decision_id,
                st.window_id,
                send_stream_end=False,
            )
        return None

    async def _finish_completion_request_locked(
        self,
        decision_id: int,
        last_window_id: Optional[int],
        stream_end_perf: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if stream_end_perf is None:
            stream_end_perf = time.perf_counter()
        windows = [
            self.windows[wid]
            for wid in sorted(self.windows)
            if self.windows[wid].closed
            and self.windows[wid].appended
            and (last_window_id is None or wid <= last_window_id)
        ]
        if not windows:
            log.info("stream=%s completion decision=%d with no closed windows", self.stream_id, decision_id)
            return None

        frames: List[Dict[str, str]] = []
        for st in windows:
            frames.extend(st.prepared_frames)
        if not frames:
            log.warning(
                "stream=%s completion decision=%d windows=%s had no decoded frames",
                self.stream_id,
                decision_id,
                [st.window_id for st in windows],
            )
            for st in windows:
                self.windows.pop(st.window_id, None)
            self._reset_active_request_locked(next_decision_id=decision_id + 1)
            return None

        request_id = f"{self.stream_id}-c{decision_id}-{uuid.uuid4().hex[:6]}"
        opened_at = min(st.opened_at for st in windows)
        t_request = time.perf_counter()
        self.inflight_appends += 1
        try:
            result = await self.vllm.chat_completion(
                model=str(self.hello.get("model") or ""),
                prompt=str(self.hello.get("prompt") or ""),
                frames=frames,
                max_tokens=int(self.hello.get("max_tokens") or 24),
                temperature=0.0,
            )
        except Exception as e:
            log.exception("completion request failed: %s", e)
            for st in windows:
                self.windows.pop(st.window_id, None)
            self._reset_active_request_locked(next_decision_id=decision_id + 1)
            return None
        finally:
            self.inflight_appends = max(0, self.inflight_appends - 1)

        request_ms = result.elapsed_ms or ((time.perf_counter() - t_request) * 1000.0)
        self.last_append_ms = request_ms
        stream_end_to_result_ms = (time.perf_counter() - stream_end_perf) * 1000.0
        e2e_ms = (time.time() - opened_at) * 1000.0
        pre_stream_end_ms = max(0.0, e2e_ms - stream_end_to_result_ms)
        chunk_window_ids = [st.window_id for st in windows]
        sent_frames = len(frames)
        decode_ms = sum(st.decode_ms for st in windows)
        alpha_ms = sum(st.alpha_ms for st in windows)
        jpeg_ms = sum(st.jpeg_ms for st in windows)
        avg_dims = [int(st.avg_dim) for st in windows if st.avg_dim is not None]
        avg_dim = int(sum(avg_dims) / len(avg_dims)) if avg_dims else None
        if self.latency_tracker is not None:
            self.latency_tracker.record(
                WindowLatencySample(
                    stream_id=self.stream_id,
                    window_id=decision_id,
                    at_wall=time.time(),
                    frames=sent_frames,
                    append_ms=request_ms,
                    e2e_ms=e2e_ms,
                    output_len=len(result.output_text or ""),
                    stream_end_to_result_ms=stream_end_to_result_ms,
                    pre_stream_end_ms=pre_stream_end_ms,
                )
            )

        alpha_init, alpha_init_version = self.alpha_init_state.get()
        result_msg = {
            "kind": "result",
            "mode": "completion",
            "stream_id": self.stream_id,
            "engine_index": self.engine_index,
            "window_id": decision_id,
            "last_window_id": last_window_id,
            "request_id": request_id,
            "text": result.output_text,
            "frames": sent_frames,
            "chunk_window_ids": chunk_window_ids,
            "alpha": 1.0,
            "alpha_init": alpha_init,
            "alpha_init_version": alpha_init_version,
            "avg_side": avg_dim,
            "decode_ms": decode_ms,
            "alpha_ms": alpha_ms,
            "jpeg_ms": jpeg_ms,
            "append_ms": request_ms,
            "request_ms": request_ms,
            "stream_end_to_result_ms": stream_end_to_result_ms,
            "pre_stream_end_ms": pre_stream_end_ms,
            "e2e_ms": e2e_ms,
            "raw": result.raw,
        }
        log.info(
            "stream=%s engine=%d completion decision=%d done frames=%d chunks=%s side≈%s request_ms=%.1f final_ms=%.1f e2e_ms=%.1f text=%r",
            self.stream_id,
            self.engine_index,
            decision_id,
            sent_frames,
            chunk_window_ids,
            avg_dim,
            request_ms,
            stream_end_to_result_ms,
            e2e_ms,
            (result.output_text or "")[:120],
        )
        if self.on_result is not None:
            try:
                await self.on_result(result_msg)
            except Exception as e:
                log.debug("on_result callback failed: %s", e)
        for st in windows:
            self.windows.pop(st.window_id, None)
        self._reset_active_request_locked(next_decision_id=decision_id + 1)
        return result_msg

    async def _abort_unfinished_poll_locked(
        self,
        request_id: str,
        decision_id: int,
        poll: PollResult,
        *,
        phase: str,
    ) -> PollResult:
        if poll.finished:
            return poll
        raw = dict(poll.raw)
        raw["timed_out"] = bool(getattr(poll, "timed_out", False) or raw.get("timed_out"))
        log.warning(
            "stream=%s engine=%d decision=%d request=%s poll unfinished phase=%s timed_out=%s; aborting stale vLLM session",
            self.stream_id,
            self.engine_index,
            decision_id,
            request_id,
            phase,
            raw["timed_out"],
        )
        aborted = await self.vllm.abort(request_id)
        raw["abort_after_timeout"] = bool(aborted)
        raw["aborted"] = bool(aborted)
        raw["abort_phase"] = phase
        if aborted:
            log.info(
                "stream=%s engine=%d decision=%d request=%s stale session aborted",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
            )
        else:
            log.warning(
                "stream=%s engine=%d decision=%d request=%s stale session abort failed",
                self.stream_id,
                self.engine_index,
                decision_id,
                request_id,
            )
        return PollResult(
            finished=False,
            output_text=poll.output_text,
            raw=raw,
            timed_out=bool(raw["timed_out"]),
            aborted=bool(aborted),
        )

    async def _finish_active_request_locked(
        self,
        decision_id: int,
        last_window_id: Optional[int],
        send_stream_end: bool = True,
        abort_instead_of_poll: bool = False,
        stream_end_perf: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if stream_end_perf is None:
            stream_end_perf = time.perf_counter()
        request_id = self._active_request_id
        if request_id is None:
            log.info("stream=%s stream_end decision=%d with no active request", self.stream_id, decision_id)
            return None
        hint_window_id = (
            last_window_id
            if last_window_id is not None
            else (self._active_chunk_window_ids[-1] if self._active_chunk_window_ids else decision_id)
        )

        if send_stream_end:
            t_append = time.perf_counter()
            self.inflight_appends += 1
            stream_end_append_ms = 0.0
            try:
                end_resp = await self.vllm.append_frames(request_id=request_id, frames=[], stream_end=True)
                self._update_alpha_init_from_response(end_resp, hint_window_id)
                if self._response_has_early_finalized(end_resp):
                    self._mark_early_finalized(
                        "stream_end_response",
                        decision_id,
                        active_windows=len(self._active_chunk_window_ids),
                    )
            except Exception as e:
                self.inflight_appends = max(0, self.inflight_appends - 1)
                if self._is_conflict_error(e):
                    if self._exception_mentions_early_finalized(e):
                        self._mark_early_finalized(
                            "stream_end_conflict",
                            decision_id,
                            active_windows=len(self._active_chunk_window_ids),
                        )
                    log.warning(
                        "stream=%s decision=%d stream_end conflict on %s; polling existing result",
                        self.stream_id,
                        decision_id,
                        request_id,
                    )
                else:
                    recovered_resp = None
                    if self._is_lost_append_response_error(e):
                        recovered_resp = await self._recover_lost_append_response_locked(
                            request_id=request_id,
                            decision_id=decision_id,
                            expected_appended_frames=self._active_sent_frames,
                            stream_end=True,
                            phase="stream_end_response_lost",
                        )
                    if recovered_resp is None:
                        log.exception(
                            "stream_end append failed stream=%s engine=%d decision=%d request=%s: %s",
                            self.stream_id,
                            self.engine_index,
                            decision_id,
                            request_id,
                            e,
                        )
                        await self._abort_active_request_locked(
                            request_id=request_id,
                            decision_id=decision_id,
                            last_window_id=last_window_id,
                            phase="stream_end_append_error",
                        )
                        return None
                    stream_end_append_ms = (time.perf_counter() - t_append) * 1000.0
                    self._update_alpha_init_from_response(recovered_resp, hint_window_id)
                    if self._response_has_early_finalized(recovered_resp):
                        self._mark_early_finalized(
                            "stream_end_response_lost",
                            decision_id,
                            active_windows=len(self._active_chunk_window_ids),
                        )
            else:
                self.inflight_appends = max(0, self.inflight_appends - 1)
                stream_end_append_ms = (time.perf_counter() - t_append) * 1000.0
            if stream_end_append_ms > 0:
                self._active_append_ms += stream_end_append_ms
                self.last_append_ms = self._active_append_ms
            poll = await self.vllm.wait_until_finished(
                request_id,
                poll_interval_s=self._result_poll_interval_s(),
                timeout_s=self._result_timeout_s(),
                wait_for_visual_memory=bool(self.visual_memory_enabled),
            )
            poll = await self._abort_unfinished_poll_locked(
                request_id,
                decision_id,
                poll,
                phase="stream_end_poll",
            )
            self._update_alpha_init_from_response(poll.raw, hint_window_id)
            if self._response_has_early_finalized(poll.raw):
                self._mark_early_finalized(
                    "poll_response",
                    decision_id,
                    active_windows=len(self._active_chunk_window_ids),
                )
        elif abort_instead_of_poll:
            aborted = await self.vllm.abort(request_id)
            if aborted:
                log.info(
                    "stream=%s decision=%d aborted active request %s",
                    self.stream_id,
                    decision_id,
                    request_id,
                )
            else:
                log.warning(
                    "stream=%s decision=%d abort failed for active request %s",
                    self.stream_id,
                    decision_id,
                    request_id,
                )
            poll = PollResult(
                finished=bool(aborted),
                output_text="",
                raw={"aborted": bool(aborted), "stream_end_sent": False},
                aborted=bool(aborted),
            )
        else:
            poll = await self.vllm.wait_until_finished(
                request_id,
                poll_interval_s=self._result_poll_interval_s(),
                timeout_s=self._result_timeout_s(),
                wait_for_visual_memory=bool(self.visual_memory_enabled),
            )
            poll = await self._abort_unfinished_poll_locked(
                request_id,
                decision_id,
                poll,
                phase="early_finalized_poll",
            )
            self._update_alpha_init_from_response(poll.raw, hint_window_id)
            if self._response_has_early_finalized(poll.raw):
                self._mark_early_finalized(
                    "poll_response",
                    decision_id,
                    active_windows=len(self._active_chunk_window_ids),
                )

        decision_duration_s = max(1e-3, time.time() - self._active_opened_at)
        session_released = bool(poll.finished or getattr(poll, "aborted", False))
        if poll.finished:
            self._update_visual_memory_from_response(
                poll.raw,
                decision_id=decision_id,
                request_id=request_id,
            )
        released_tokens_est = 0
        if session_released:
            released_tokens_est = self.flow.observe_release(
                decision_duration_s=decision_duration_s
            )
        if session_released and self.rid_drop is not None:
            # Finished or explicitly aborted — no need to DELETE at shutdown.
            try:
                self.rid_drop(request_id)
            except Exception:
                pass
        stream_end_to_result_ms = (time.perf_counter() - stream_end_perf) * 1000.0
        e2e_ms = (time.time() - self._active_opened_at) * 1000.0
        pre_stream_end_ms = max(0.0, e2e_ms - stream_end_to_result_ms)
        avg_dim = (
            int(sum(self._active_avg_dims) / len(self._active_avg_dims))
            if self._active_avg_dims
            else None
        )
        if self.latency_tracker is not None:
            self.latency_tracker.record(
                WindowLatencySample(
                    stream_id=self.stream_id,
                    window_id=decision_id,
                    at_wall=time.time(),
                    frames=self._active_sent_frames,
                    append_ms=self._active_append_ms,
                    e2e_ms=e2e_ms,
                    output_len=len(poll.output_text or ""),
                    stream_end_to_result_ms=stream_end_to_result_ms,
                    pre_stream_end_ms=pre_stream_end_ms,
                )
            )
        alpha_init, alpha_init_version = self.alpha_init_state.get()
        result_msg = {
            "kind": "result",
            "stream_id": self.stream_id,
            "engine_index": self.engine_index,
            "window_id": decision_id,
            "last_window_id": last_window_id,
            "request_id": request_id,
            "text": poll.output_text,
            "frames": self._active_sent_frames,
            "chunk_window_ids": list(self._active_chunk_window_ids),
            "alpha": self._current_alpha(),
            "alpha_init": alpha_init,
            "alpha_init_version": alpha_init_version,
            "avg_side": avg_dim,
            "decode_ms": self._active_decode_ms,
            "alpha_ms": self._active_alpha_ms,
            "jpeg_ms": self._active_jpeg_ms,
            "append_ms": self._active_append_ms,
            "stream_end_to_result_ms": stream_end_to_result_ms,
            "pre_stream_end_ms": pre_stream_end_ms,
            "e2e_ms": e2e_ms,
            "released_tokens_est": released_tokens_est,
            "visual_memory": self.visual_memory_snapshot(),
            "early_finalized": bool(self._active_early_finalized),
            "early_finalized_count": int(self.early_finalized_count),
            "finished": bool(poll.finished),
            "timed_out": bool(getattr(poll, "timed_out", False)),
            "abort_after_timeout": bool(
                getattr(poll, "timed_out", False)
                and getattr(poll, "aborted", False)
            ),
            "raw": poll.raw,
        }
        log.info(
            "stream=%s engine=%d decision=%d done frames=%d chunks=%s side≈%s append_ms=%.1f final_ms=%.1f text=%r",
            self.stream_id,
            self.engine_index,
            decision_id,
            self._active_sent_frames,
            self._active_chunk_window_ids,
            avg_dim,
            self._active_append_ms,
            stream_end_to_result_ms,
            (poll.output_text or "")[:120],
        )
        if self.on_result is not None:
            try:
                await self.on_result(result_msg)
            except Exception as e:
                log.debug("on_result callback failed: %s", e)
        for wid in list(self.windows.keys()):
            if last_window_id is not None and wid <= last_window_id:
                st = self.windows.get(wid)
                if st is not None and not st.appended:
                    st.dropped_after_finalize = True
                self.windows.pop(wid, None)
        self._seal_windows_locked(last_window_id, reason="active_request_finished")
        self._reset_active_request_locked(next_decision_id=decision_id + 1)
        return result_msg

    def _reset_active_request_locked(self, next_decision_id: int) -> None:
        self._active_request_id = None
        self._active_decision_id = int(next_decision_id)
        self._active_opened_at = time.time()
        self._active_sent_frames = 0
        self._active_decode_ms = 0.0
        self._active_alpha_ms = 0.0
        self._active_jpeg_ms = 0.0
        self._active_append_ms = 0.0
        self._active_avg_dims = []
        self._active_chunk_window_ids = []
        self._active_early_finalized = False
