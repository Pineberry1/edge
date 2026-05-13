"""HTTP client wrapping the vLLM OpenAI-compatible endpoints.

Mirrors the shape used by `test/compare_online_prefill_latency.py`. One client
instance can drive many concurrent sessions; `httpx.AsyncClient` takes care of
connection pooling to the local vLLM server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger("intake.vllm")


def _completion_retry_count() -> int:
    raw = os.environ.get("BAVA_COMPLETION_RETRIES", "2")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _completion_retry_delay_s() -> float:
    raw = os.environ.get("BAVA_COMPLETION_RETRY_DELAY_MS", "250")
    try:
        return max(0.0, float(raw) / 1000.0)
    except ValueError:
        return 0.25


@dataclass
class PollResult:
    finished: bool
    output_text: str
    raw: Dict[str, Any]
    timed_out: bool = False
    aborted: bool = False


@dataclass
class CompletionResult:
    finished: bool
    output_text: str
    raw: Dict[str, Any]
    elapsed_ms: float


class VLLMOnlinePrefillClient:
    def __init__(self, api_base: str, timeout_s: float = 600.0) -> None:
        self.api_base = api_base.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            timeout=timeout_s,
            http2=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_session(
        self,
        request_id: str,
        model: str,
        prompt: str,
        max_tokens: int,
        visual_memory: Optional[Dict[str, Any]] = None,
        export_visual_memory: bool = False,
        export_visual_memory_num_frames: Optional[int] = None,
        export_visual_memory_tokens_per_frame: Optional[int] = None,
        export_visual_memory_id: Optional[str] = None,
        export_visual_memory_text_prefix: Optional[str] = None,
        warm_visual_memory_prefix_cache: bool = False,
        visual_token_merger_alpha: Optional[float] = None,
        visual_token_merger_block_t: Optional[int] = None,
        visual_token_merger_block_hw: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "request_id": request_id,
            "model": model,
            "prompt": prompt,
            "system_prompt": "",
            "max_tokens": max_tokens,
        }
        if visual_memory is not None:
            payload["visual_memory"] = visual_memory
        if export_visual_memory:
            payload["export_visual_memory"] = True
            if export_visual_memory_num_frames is not None:
                payload["export_visual_memory_num_frames"] = int(
                    export_visual_memory_num_frames
                )
            if export_visual_memory_tokens_per_frame is not None:
                payload["export_visual_memory_tokens_per_frame"] = int(
                    export_visual_memory_tokens_per_frame
                )
            if export_visual_memory_id:
                payload["export_visual_memory_id"] = str(export_visual_memory_id)
            if export_visual_memory_text_prefix is not None:
                payload["export_visual_memory_text_prefix"] = str(
                    export_visual_memory_text_prefix
                )
            if warm_visual_memory_prefix_cache:
                payload["warm_visual_memory_prefix_cache"] = True
        if visual_token_merger_alpha is not None:
            payload["visual_token_merger_alpha"] = float(visual_token_merger_alpha)
        if visual_token_merger_block_t is not None:
            payload["visual_token_merger_block_t"] = int(visual_token_merger_block_t)
        if visual_token_merger_block_hw is not None:
            payload["visual_token_merger_block_hw"] = int(visual_token_merger_block_hw)
        resp = await self._client.post("/v1/online_prefill/sessions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def append_frames(
        self,
        request_id: str,
        frames: List[Dict[str, str]],
        stream_end: bool,
        visual_token_merger_alpha: Optional[float] = None,
        visual_token_merger_block_t: Optional[int] = None,
        visual_token_merger_block_hw: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {"frames": frames, "stream_end": stream_end}
        if visual_token_merger_alpha is not None:
            payload["visual_token_merger_alpha"] = float(visual_token_merger_alpha)
        if visual_token_merger_block_t is not None:
            payload["visual_token_merger_block_t"] = int(visual_token_merger_block_t)
        if visual_token_merger_block_hw is not None:
            payload["visual_token_merger_block_hw"] = int(visual_token_merger_block_hw)
        resp = await self._client.post(
            f"/v1/online_prefill/sessions/{request_id}/append",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def poll(self, request_id: str) -> PollResult:
        resp = await self._client.get(f"/v1/online_prefill/sessions/{request_id}")
        resp.raise_for_status()
        data = resp.json()
        return PollResult(
            finished=bool(data.get("finished")),
            output_text=str(data.get("output_text") or ""),
            raw=data,
        )

    async def wait_until_finished(
        self,
        request_id: str,
        poll_interval_s: float = 0.25,
        timeout_s: float = 120.0,
        wait_for_visual_memory: bool = False,
    ) -> PollResult:
        deadline = time.time() + timeout_s
        while True:
            pr = await self.poll(request_id)
            if pr.finished:
                if not wait_for_visual_memory:
                    return pr
                if not bool(pr.raw.get("visual_memory_export_pending")):
                    return pr
            if time.time() > deadline:
                raw = dict(pr.raw)
                raw["timed_out"] = True
                raw["timeout_s"] = float(timeout_s)
                return PollResult(
                    finished=pr.finished,
                    output_text=pr.output_text,
                    raw=raw,
                    timed_out=True,
                )
            await asyncio.sleep(poll_interval_s)

    async def chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        frames: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Send all decoded frames as one native chat-completion request."""
        content: List[Dict[str, Any]] = []
        for frame in frames:
            url = str(frame.get("data") or frame.get("url") or "")
            if not url:
                continue
            content.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        started = time.perf_counter()
        retries = _completion_retry_count()
        retry_delay_s = _completion_retry_delay_s()
        retryable = tuple(
            cls
            for cls in (
                getattr(httpx, "ReadError", None),
                getattr(httpx, "RemoteProtocolError", None),
                getattr(httpx, "ConnectError", None),
                getattr(httpx, "ReadTimeout", None),
            )
            if cls is not None
        )
        for attempt in range(retries + 1):
            try:
                resp = await self._client.post("/v1/chat/completions", json=payload)
                break
            except retryable:
                if attempt >= retries:
                    raise
                log.warning(
                    "completion request transport failure; retrying attempt=%d/%d frames=%d",
                    attempt + 1,
                    retries,
                    len(frames),
                )
                if retry_delay_s > 0:
                    await asyncio.sleep(retry_delay_s * float(attempt + 1))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        msg = first.get("message") or {}
        text = str(msg.get("content") or "")
        return CompletionResult(
            finished=bool(choices),
            output_text=text,
            raw=data,
            elapsed_ms=elapsed_ms,
        )

    async def abort(self, request_id: str) -> bool:
        """DELETE a session on vLLM — used by intake shutdown cleanup to
        avoid leaving stale sessions that would show up as Q residue for the
        next intake run."""
        try:
            resp = await self._client.delete(
                f"/v1/online_prefill/sessions/{request_id}",
                timeout=5.0,
            )
            return resp.status_code < 400 or resp.status_code == 404
        except Exception as e:
            log.debug("abort %s failed: %s", request_id, e)
            return False
