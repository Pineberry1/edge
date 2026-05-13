from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _ReadError(Exception):
        pass

    class _RemoteProtocolError(Exception):
        pass

    class _ConnectError(Exception):
        pass

    class _ReadTimeout(Exception):
        pass

    class _UnusedAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("test should inject its own fake client")

    httpx_stub.AsyncClient = _UnusedAsyncClient
    httpx_stub.ReadError = _ReadError
    httpx_stub.RemoteProtocolError = _RemoteProtocolError
    httpx_stub.ConnectError = _ConnectError
    httpx_stub.ReadTimeout = _ReadTimeout
    sys.modules["httpx"] = httpx_stub

SPEC = importlib.util.spec_from_file_location(
    "remote_intake_vllm_client",
    ROOT / "vllm_client.py",
)
vllm_client = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = vllm_client
SPEC.loader.exec_module(vllm_client)


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200) -> None:
        self._payload = payload or {"ok": True}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.posts = []
        self.post_effects = []
        self.deletes = []
        self.get_payloads = []
        self.delete_status_code = 200

    async def post(self, path: str, json: dict):
        self.posts.append((path, json))
        if self.post_effects:
            effect = self.post_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        return _FakeResponse()

    async def get(self, path: str):
        payload = self.get_payloads.pop(0)
        return _FakeResponse(payload)

    async def delete(self, path: str, timeout: float):
        self.deletes.append((path, timeout))
        return _FakeResponse(status_code=self.delete_status_code)


def test_create_session_sends_visual_memory_export_fields() -> None:
    fake = _FakeAsyncClient()
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    asyncio.run(
        client.create_session(
            request_id="rid",
            model="model",
            prompt="prompt",
            max_tokens=8,
            visual_memory={"data": "abc", "memory_id": "mem0"},
            export_visual_memory=True,
            export_visual_memory_num_frames=4,
            export_visual_memory_tokens_per_frame=16,
            export_visual_memory_id="mem1",
            export_visual_memory_text_prefix="context\n",
            warm_visual_memory_prefix_cache=True,
        )
    )

    assert fake.posts[0][0] == "/v1/online_prefill/sessions"
    payload = fake.posts[0][1]
    assert payload["visual_memory"]["memory_id"] == "mem0"
    assert payload["export_visual_memory"] is True
    assert payload["export_visual_memory_num_frames"] == 4
    assert payload["export_visual_memory_tokens_per_frame"] == 16
    assert payload["export_visual_memory_id"] == "mem1"
    assert payload["export_visual_memory_text_prefix"] == "context\n"
    assert payload["warm_visual_memory_prefix_cache"] is True


def test_append_frames_can_send_visual_token_merger_params() -> None:
    fake = _FakeAsyncClient()
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    asyncio.run(
        client.append_frames(
            request_id="rid",
            frames=[{"data": "x", "mime_type": "image/jpeg"}],
            stream_end=False,
            visual_token_merger_alpha=0.75,
            visual_token_merger_block_t=1,
            visual_token_merger_block_hw=2,
        )
    )

    assert fake.posts[0][0] == "/v1/online_prefill/sessions/rid/append"
    payload = fake.posts[0][1]
    assert payload["frames"] == [{"data": "x", "mime_type": "image/jpeg"}]
    assert payload["stream_end"] is False
    assert payload["visual_token_merger_alpha"] == 0.75
    assert payload["visual_token_merger_block_t"] == 1
    assert payload["visual_token_merger_block_hw"] == 2


def test_wait_until_finished_can_wait_for_visual_memory_export() -> None:
    fake = _FakeAsyncClient()
    fake.get_payloads = [
        {
            "finished": True,
            "output_text": "done",
            "visual_memory_export_pending": True,
        },
        {
            "finished": True,
            "output_text": "done",
            "visual_memory_export_pending": False,
            "visual_memory": {"memory_id": "mem1"},
        },
    ]
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    result = asyncio.run(
        client.wait_until_finished(
            "rid",
            poll_interval_s=0.0,
            wait_for_visual_memory=True,
        )
    )

    assert result.raw["visual_memory"]["memory_id"] == "mem1"
    assert fake.get_payloads == []


def test_wait_until_finished_marks_unfinished_timeout() -> None:
    fake = _FakeAsyncClient()
    fake.get_payloads = [
        {
            "finished": False,
            "output_text": "",
            "status": "early_finalized",
        },
    ]
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    result = asyncio.run(
        client.wait_until_finished("rid", poll_interval_s=0.0, timeout_s=-1.0)
    )

    assert result.finished is False
    assert result.timed_out is True
    assert result.raw["timed_out"] is True


def test_abort_treats_missing_session_as_cleaned_up() -> None:
    fake = _FakeAsyncClient()
    fake.delete_status_code = 404
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    ok = asyncio.run(client.abort("rid"))

    assert ok is True
    assert fake.deletes[0][0] == "/v1/online_prefill/sessions/rid"


def test_chat_completion_retries_transport_read_error() -> None:
    fake = _FakeAsyncClient()
    fake.post_effects = [
        vllm_client.httpx.ReadError("lost response"),
        _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "No",
                        }
                    }
                ]
            }
        ),
    ]
    client = vllm_client.VLLMOnlinePrefillClient.__new__(
        vllm_client.VLLMOnlinePrefillClient
    )
    client._client = fake

    result = asyncio.run(
        client.chat_completion(
            model="model",
            prompt="prompt",
            frames=[{"data": "data:image/jpeg;base64,abc"}],
            max_tokens=8,
        )
    )

    assert result.output_text == "No"
    assert len(fake.posts) == 2
    assert fake.posts[0][1] == fake.posts[1][1]
