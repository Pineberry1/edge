from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = types.ModuleType("remote_intake")
PKG.__path__ = [str(ROOT)]
sys.modules.setdefault("remote_intake", PKG)

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INTER_AREA = 0
    cv2_stub.INTER_CUBIC = 1
    cv2_stub.resize = lambda img, size, interpolation=None: img
    cv2_stub.imencode = lambda ext, img, params=None: (True, b"")
    cv2_stub.IMWRITE_JPEG_QUALITY = 1
    sys.modules["cv2"] = cv2_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.bool_ = bool
    numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float, complex, bool))
    sys.modules["numpy"] = numpy_stub

if "av" not in sys.modules:
    av_stub = types.ModuleType("av")
    av_stub.AVError = Exception
    sys.modules["av"] = av_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _UnusedAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("test injects a fake vLLM client")

    httpx_stub.AsyncClient = _UnusedAsyncClient
    sys.modules["httpx"] = httpx_stub


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


window_assembler = _load_module(
    "remote_intake.window_assembler",
    "window_assembler.py",
)


class _ConflictResponse:
    status_code = 409

    @staticmethod
    def json() -> dict:
        return {"status": "early_finalized"}


class _ConflictError(Exception):
    response = _ConflictResponse()


class ReadError(Exception):
    pass


class _FakeVLLM:
    def __init__(
        self,
        *,
        append_error: Exception | None = None,
        stream_end_error: Exception | None = None,
        append_payload: dict | None = None,
        poll_payload: dict | None = None,
        append_started: asyncio.Event | None = None,
        append_blocker: asyncio.Event | None = None,
    ) -> None:
        self.append_error = append_error
        self.stream_end_error = stream_end_error
        self.append_payload = append_payload or {"ok": True}
        self.poll_payload = poll_payload or {}
        self.append_started = append_started
        self.append_blocker = append_blocker
        self.created: list[str] = []
        self.append_attempts: list[tuple[str, bool, int]] = []
        self.aborts: list[str] = []

    async def create_session(self, *, request_id: str, **_kwargs) -> dict:
        self.created.append(request_id)
        return {"request_id": request_id}

    async def append_frames(self, *, request_id: str, frames: list, stream_end: bool) -> dict:
        self.append_attempts.append((request_id, stream_end, len(frames)))
        if frames and self.append_started is not None:
            self.append_started.set()
        if frames and self.append_blocker is not None:
            await self.append_blocker.wait()
        if self.append_error is not None and frames:
            raise self.append_error
        if self.stream_end_error is not None and stream_end:
            raise self.stream_end_error
        return dict(self.append_payload)

    async def poll(self, request_id: str):
        raw = {"request_id": request_id}
        raw.update(self.poll_payload)
        return window_assembler.PollResult(
            finished=bool(raw.get("finished")),
            output_text=str(raw.get("output_text") or ""),
            raw=raw,
        )

    async def wait_until_finished(self, request_id: str, **_kwargs):
        return window_assembler.PollResult(
            finished=True,
            output_text="done",
            raw={"finished": True, "request_id": request_id},
        )

    async def abort(self, request_id: str) -> bool:
        self.aborts.append(request_id)
        return True


def _session(fake: _FakeVLLM, *, rid_drop):
    sess = window_assembler.StreamSession(
        stream_id="s0",
        hello={
            "model": "m",
            "prompt": "p",
            "max_tokens": 4,
            "window_seconds": 4,
            "decision_window_seconds": 40,
        },
        vllm=fake,
        engine_index=0,
        anchor_log_path=None,
        rid_drop=rid_drop,
    )

    async def _prepared(_st):
        return ([{"data": "jpeg"}], 1.0, 0.0, 0.0, 0.0, 10, 10)

    sess._prepare_window_frames = _prepared
    return sess


def _window(wid: int = 0):
    return window_assembler.WindowState(
        stream_id="s0",
        window_id=wid,
        rho=1.0,
        opened_at=1.0,
        closed=True,
    )


def test_append_conflict_seals_rejected_window_so_it_is_not_reappended() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM(append_error=_ConflictError("early finalized"))
        sess = _session(fake, rid_drop=dropped.append)
        st = _window(5)
        sess.windows[5] = st

        await sess._append_window_locked(st)

        assert len(fake.created) == 1
        assert len(fake.append_attempts) == 1
        assert dropped == fake.created
        assert st.dropped_after_finalize is True
        assert sess._active_request_id is None
        assert sess._sealed_window_id == 5

        await sess._append_window_locked(st)

        assert len(fake.created) == 1
        assert len(fake.append_attempts) == 1

    asyncio.run(_run())


def test_append_read_error_aborts_and_resets_active_request() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM(append_error=RuntimeError("read failed"))
        sess = _session(fake, rid_drop=dropped.append)
        st = _window(2)
        sess.windows[2] = st

        await sess._append_window_locked(st)

        assert len(fake.created) == 1
        assert fake.aborts == fake.created
        assert dropped == fake.created
        assert st.dropped_after_finalize is True
        assert sess._active_request_id is None
        assert sess._sealed_window_id == 2

    asyncio.run(_run())


def test_append_lost_response_recovers_when_server_applied_frames() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM(
            append_error=ReadError("lost response"),
            poll_payload={
                "status": "streaming",
                "appended_frames": 1,
                "stream_end_received": False,
            },
        )
        sess = _session(fake, rid_drop=dropped.append)
        st = _window(2)
        sess.windows[2] = st

        await sess._append_window_locked(st)

        assert len(fake.created) == 1
        assert fake.aborts == []
        assert dropped == []
        assert st.appended is True
        assert st.dropped_after_finalize is False
        assert sess._active_request_id == fake.created[0]
        assert sess._active_sent_frames == 1
        assert sess._active_chunk_window_ids == [2]

    asyncio.run(_run())


def test_append_response_early_finalized_closes_without_next_conflict() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM(append_payload={"early_finalized": True})
        sess = _session(fake, rid_drop=dropped.append)
        st = _window(3)
        sess.windows[3] = st

        await sess._append_window_locked(st)

        assert len(fake.created) == 1
        assert len(fake.append_attempts) == 1
        assert fake.aborts == []
        assert dropped == fake.created
        assert sess._active_request_id is None
        assert sess._sealed_window_id == 3
        assert sess.last_early_finalized_windows == 1

    asyncio.run(_run())


def test_stream_end_lost_response_recovers_when_server_received_end() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM(
            stream_end_error=ReadError("lost response"),
            poll_payload={
                "status": "waiting_for_decode",
                "appended_frames": 1,
                "stream_end_received": True,
                "decode_started": True,
            },
        )
        sess = _session(fake, rid_drop=dropped.append)
        st = _window(0)
        sess.windows[0] = st

        await sess._append_window_locked(st)
        result = await sess.on_stream_end({"decision_id": 0, "last_window_id": 0})

        assert result is not None
        assert result["text"] == "done"
        assert fake.aborts == []
        assert dropped == fake.created
        assert sess._active_request_id is None
        assert fake.append_attempts == [(fake.created[0], False, 1), (fake.created[0], True, 0)]

    asyncio.run(_run())


def test_stale_stream_end_uses_active_decision_after_early_finalize() -> None:
    async def _run() -> None:
        dropped: list[str] = []
        fake = _FakeVLLM()
        sess = _session(fake, rid_drop=dropped.append)
        sess._active_decision_id = 1
        sess._sealed_window_id = 3
        st = _window(4)
        sess.windows[4] = st

        result = await sess.on_stream_end({"decision_id": 0, "last_window_id": 4})

        assert result is not None
        assert result["window_id"] == 1
        assert result["last_window_id"] == 4
        assert sess._active_decision_id == 2
        assert len(fake.created) == 1
        assert fake.append_attempts == [(fake.created[0], False, 1), (fake.created[0], True, 0)]

    asyncio.run(_run())


def test_cleanup_cancels_ordered_ops_and_blocks_late_session_creation(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("BAVA_SESSION_CLEANUP_DRAIN_S", "0.01")
        monkeypatch.setenv("BAVA_SESSION_CLEANUP_CANCEL_S", "0.5")
        monkeypatch.setenv("BAVA_SESSION_CLEANUP_LOCK_TIMEOUT_S", "0.5")
        dropped: list[str] = []
        append_started = asyncio.Event()
        append_blocker = asyncio.Event()
        fake = _FakeVLLM(
            append_started=append_started,
            append_blocker=append_blocker,
        )
        sess = _session(fake, rid_drop=dropped.append)
        st0 = _window(0)
        st1 = _window(1)
        sess.windows[0] = st0
        sess.windows[1] = st1

        first = sess.schedule_ordered(lambda: sess.append_closed_window(st0))
        await asyncio.wait_for(append_started.wait(), timeout=1.0)
        second = sess.schedule_ordered(lambda: sess.append_closed_window(st1))

        await sess.cleanup()
        append_blocker.set()
        await asyncio.gather(first, second, return_exceptions=True)

        assert sess._closing is True
        assert len(fake.created) == 1
        assert fake.aborts == fake.created
        assert dropped == fake.created
        assert st1.appended is False

        third = sess.schedule_ordered(lambda: sess.append_closed_window(st1))
        await third

        assert len(fake.created) == 1

    asyncio.run(_run())
