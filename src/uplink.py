"""Edge→cloud WebSocket uplink with auto-reconnect and reverse-channel listener.

The packet pipeline is synchronous (PyAV blocking demux). To avoid turning the
whole edge into asyncio, we run a dedicated background thread that owns its
own asyncio event loop plus the WebSocket connection. The main thread talks to
it through `queue.Queue`-semantics `call_soon_threadsafe` into an asyncio.Queue.

Messages are opaque to this layer:
  * binary frames — already-packed `[hdr_len][hdr][payload]` bytes.
  * text frames   — already-serialised JSON strings.

Reverse channel: cloud text frames are parsed; `rho_update` writes to
`RhoState`, `result` is appended to a small ring log. Unknown kinds are
ignored.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

from .budget_state import BudgetState
from .config import EdgeConfig
from .rho_state import RhoState
from .wire import (
    MSG_BYE,
    MSG_BUDGET_UPDATE,
    MSG_EARLY_FINALIZE,
    MSG_HELLO,
    MSG_RESULT,
    MSG_RHO_UPDATE,
    pack_binary,
)

try:
    import websockets  # noqa: F401
    from websockets.asyncio.client import connect as _ws_connect
except Exception:  # pragma: no cover - helpful error at import time
    _ws_connect = None


class Uplink:
    """Background-thread WebSocket client with a blocking-style send API."""

    def __init__(
        self,
        cfg: EdgeConfig,
        rho_state: RhoState,
        budget_state: Optional[BudgetState] = None,
    ) -> None:
        if _ws_connect is None:
            raise RuntimeError(
                "websockets package not available; install with "
                "`pip install --user websockets>=12`"
            )
        self.cfg = cfg
        self.rho_state = rho_state
        self.budget_state = budget_state
        self._hello: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._out_q: Optional[asyncio.Queue] = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()

        self.sent_packets = 0
        self.sent_bytes = 0
        self.dropped_packets = 0
        self.last_send_wait_ms = 0.0
        self.ewma_send_wait_ms = 0.0
        self.results: deque = deque(maxlen=64)

    # -------- public API (main thread) --------

    def start(self, hello: Dict[str, Any]) -> None:
        self._hello = {"kind": MSG_HELLO, **hello}
        self._thread = threading.Thread(target=self._thread_main, name="edge-uplink", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2.0)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def send_packet(self, header: Dict[str, Any], payload: bytes) -> bool:
        return self._submit(("binary", pack_binary(header, payload)))

    def send_control(self, obj: Dict[str, Any]) -> bool:
        return self._submit(("text", json.dumps(obj, separators=(",", ":"))))

    @property
    def queue_size(self) -> int:
        if self._out_q is None:
            return 0
        return int(self._out_q.qsize())

    def wait_connected(self, timeout: float = 10.0) -> bool:
        """Block until the WS client has completed its initial handshake."""
        return self._connected.wait(timeout=timeout)

    def wait_drained(self, timeout: float = 30.0) -> bool:
        """Block until the outbound queue is empty. Caller chooses a timeout."""
        import time as _t
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            if self._out_q is None:
                return False
            if self._out_q.qsize() == 0:
                return True
            _t.sleep(0.05)
        return False

    def close(
        self,
        drain_timeout: float = 15.0,
        linger_s: float = 2.0,
        join_timeout: float = 3.0,
        linger_until_results: int = 0,
    ) -> None:
        self.wait_connected(timeout=2.0)
        self.wait_drained(timeout=drain_timeout)
        if linger_s > 0:
            import time as _t
            deadline = _t.time() + linger_s
            while _t.time() < deadline:
                if linger_until_results > 0 and len(self.results) >= linger_until_results:
                    break
                _t.sleep(min(0.25, max(0.0, deadline - _t.time())))
        self.send_control({"kind": MSG_BYE})
        self.wait_drained(timeout=1.0)
        self._stop.set()
        self._submit(("sentinel", None))
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    # -------- background thread internals --------

    def _submit(self, item) -> bool:
        if self._loop is None or self._out_q is None:
            self.dropped_packets += 1
            return False
        try:
            self._loop.call_soon_threadsafe(self._put_nowait, (*item, time.perf_counter()))
            return True
        except RuntimeError:
            self.dropped_packets += 1
            return False

    def _put_nowait(self, item) -> None:
        assert self._out_q is not None
        try:
            self._out_q.put_nowait(item)
        except asyncio.QueueFull:
            self.dropped_packets += 1

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._out_q = asyncio.Queue(maxsize=self.cfg.uplink_queue_max)
        self._ready.set()
        try:
            loop.run_until_complete(self._run_forever())
        finally:
            try:
                pending = asyncio.all_tasks(loop=loop)
                for t in pending:
                    t.cancel()
            except Exception:
                pass
            loop.close()

    async def _run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                await self._connect_and_serve()
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[edge-uplink] session error: {e!r}\n")
            self._connected.clear()
            if self._stop.is_set():
                break
            await asyncio.sleep(self.cfg.uplink_reconnect_s)

    async def _connect_and_serve(self) -> None:
        assert _ws_connect is not None
        async with _ws_connect(
            self.cfg.cloud_ws_url,
            max_size=2 ** 24,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=3.0,
        ) as ws:
            self._connected.set()
            await ws.send(json.dumps(self._hello, separators=(",", ":")))
            send_task = asyncio.create_task(self._sender(ws), name="uplink-send")
            recv_task = asyncio.create_task(self._receiver(ws), name="uplink-recv")
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    raise exc

    async def _sender(self, ws) -> None:
        assert self._out_q is not None
        while not self._stop.is_set():
            item = await self._out_q.get()
            if item is None:
                return
            if len(item) == 3:
                kind, data, enqueued_at = item
            else:
                kind, data = item
                enqueued_at = time.perf_counter()
            wait_ms = max(0.0, (time.perf_counter() - float(enqueued_at)) * 1000.0)
            self.last_send_wait_ms = wait_ms
            if self.ewma_send_wait_ms <= 0:
                self.ewma_send_wait_ms = wait_ms
            else:
                self.ewma_send_wait_ms = 0.8 * self.ewma_send_wait_ms + 0.2 * wait_ms
            if kind == "sentinel":
                return
            try:
                if kind == "text":
                    await ws.send(data)
                elif kind == "binary":
                    await ws.send(data)
                    self.sent_packets += 1
                    self.sent_bytes += len(data)
            except Exception:
                # let _receiver / connection close loop exit; main thread will retry
                raise

    async def _receiver(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            kind = msg.get("kind")
            if kind == MSG_RHO_UPDATE:
                try:
                    new_rho = float(msg["rho"])
                except (KeyError, TypeError, ValueError):
                    continue
                old = self.rho_state.current
                applied = self.rho_state.set(new_rho, reason=str(msg.get("reason", "cloud")))
                sys.stderr.write(
                    f"[edge-uplink] rho update {old:.3f} -> {applied:.3f} "
                    f"reason={msg.get('reason')!r}\n"
                )
            elif kind == MSG_BUDGET_UPDATE:
                if self.cfg.inference_mode == "completion":
                    continue
                if self.budget_state is None:
                    continue
                try:
                    version = int(msg["version"])
                    windows = int(msg["windows_per_decision"])
                except (KeyError, TypeError, ValueError):
                    continue
                reason = str(msg.get("reason") or "cloud")
                if self.budget_state.set(version, windows, reason):
                    sys.stderr.write(
                        f"[edge-uplink] budget update version={version} "
                        f"windows={windows} reason={reason}\n"
                    )
            elif kind == MSG_EARLY_FINALIZE:
                if self.cfg.inference_mode == "completion":
                    continue
                if self.budget_state is None:
                    continue
                reason = str(msg.get("reason") or "early_finalize")
                try:
                    if "version" in msg and "windows_per_decision" in msg:
                        version = int(msg["version"])
                        windows = int(msg["windows_per_decision"])
                        if self.budget_state.set(version, windows, reason):
                            sys.stderr.write(
                                f"[edge-uplink] budget update version={version} "
                                f"windows={windows} reason={reason}\n"
                            )
                except (TypeError, ValueError):
                    pass
                decision_id = None
                try:
                    if msg.get("decision_id") is not None:
                        decision_id = int(msg["decision_id"])
                except (TypeError, ValueError):
                    decision_id = None
                self.budget_state.request_force_close(
                    reason=reason,
                    decision_id=decision_id,
                    stream_id=str(msg.get("stream_id") or ""),
                )
                sys.stderr.write(
                    f"[edge-uplink] early_finalize decision={decision_id} "
                    f"reason={reason}; forcing current decision boundary\n"
                )
            elif kind == MSG_RESULT:
                self.results.append({"at": time.time(), **msg})
                sys.stderr.write(
                    f"[edge-uplink] result window={msg.get('window_id')} "
                    f"text={(msg.get('text') or '')[:80]!r}\n"
                )
