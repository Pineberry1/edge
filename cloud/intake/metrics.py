"""Lightweight scraper for one or more vLLM Prometheus `/metrics` endpoints.

We only need a handful of scalars (`num_requests_waiting`,
`num_requests_running`, `kv_cache_usage_perc`, `num_preemptions_total`,
`prompt_tokens_total`, `generation_tokens_total`). To avoid a full
prometheus_client dependency, parse the text format ourselves with a
tolerant regex that ignores help/type lines and keeps the first matching
label set we encounter.

Returns a dataclass with both raw counters and a few derived rates computed
relative to the previous snapshot.
"""

from __future__ import annotations

import time
import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

_METRIC_LINE = re.compile(
    r"^(?P<name>vllm:[A-Za-z0-9_]+)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+]?[0-9.eE+-]+)\s*$"
)
_LABEL = re.compile(r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>[^"]*)"')


@dataclass
class VllmSnapshot:
    at_wall: float
    num_requests_waiting: float = 0.0
    num_requests_running: float = 0.0
    kv_cache_usage_perc: float = 0.0
    num_preemptions_total: float = 0.0
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    kv_total_tokens: Optional[int] = None
    raw: Dict[str, float] = field(default_factory=dict)

    # derived rates (tokens/s, requests/s), filled by Scraper.rate_vs
    prompt_token_rate: float = 0.0
    generation_token_rate: float = 0.0
    preemption_rate: float = 0.0


@dataclass
class VllmEngineSnapshot:
    index: int
    api_base: str
    ok: bool
    error: Optional[str] = None
    snapshot: Optional[VllmSnapshot] = None


_FIELDS_OF_INTEREST = (
    "vllm:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:gpu_cache_total_blocks",
    "vllm:num_gpu_blocks",
    "vllm:gpu_cache_total_tokens",
    "vllm:kv_cache_total_tokens",
    "vllm:gpu_cache_block_size",
    "vllm:block_size",
    "vllm:cache_config_info",
)


def _parse_labels(label_text: Optional[str]) -> Dict[str, str]:
    if not label_text:
        return {}
    return {m.group("key"): m.group("value") for m in _LABEL.finditer(label_text)}


def _kv_total_tokens_from_raw(raw: Dict[str, float]) -> Optional[int]:
    for key in ("vllm:gpu_cache_total_tokens", "vllm:kv_cache_total_tokens"):
        value = raw.get(key)
        if value is not None and value > 0:
            return int(value)
    blocks = raw.get("vllm:gpu_cache_total_blocks")
    if blocks is None:
        blocks = raw.get("vllm:num_gpu_blocks")
    block_size = raw.get("vllm:cache_config_block_size")
    if block_size is None:
        block_size = raw.get("vllm:gpu_cache_block_size")
    if block_size is None:
        block_size = raw.get("vllm:block_size")
    if blocks is None or block_size is None or blocks <= 0 or block_size <= 0:
        return None
    return int(blocks * block_size)


def _parse(text: str) -> Dict[str, float]:
    """Return {metric_name: first_value_seen} for the names we care about."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if m is None:
            continue
        name = m.group("name")
        if name not in _FIELDS_OF_INTEREST:
            continue
        if name == "vllm:cache_config_info":
            labels = _parse_labels(m.group("labels"))
            raw_block_size = labels.get("block_size") or labels.get("cache_block_size")
            if raw_block_size is not None:
                try:
                    out.setdefault("vllm:cache_config_block_size", float(raw_block_size))
                except ValueError:
                    pass
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        # Keep first sample — vLLM emits per-engine labels but a single engine
        # deployment only yields one line per name.
        out.setdefault(name, value)
    return out


class VllmMetricsScraper:
    def __init__(self, api_base: str, timeout_s: float = 2.0) -> None:
        self.api_base = api_base.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.api_base, timeout=timeout_s)
        self._prev: Optional[VllmSnapshot] = None

    async def aclose(self) -> None:
        await self._client.aclose()

    def last_snapshot(self) -> Optional[VllmSnapshot]:
        return self._prev

    def engine_states(self) -> List[VllmEngineSnapshot]:
        snap = self._prev
        return [
            VllmEngineSnapshot(
                index=0,
                api_base=self.api_base,
                ok=snap is not None,
                error=None if snap is not None else "not scraped yet",
                snapshot=snap,
            )
        ]

    async def scrape(self) -> VllmSnapshot:
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        raw = _parse(resp.text)
        snap = VllmSnapshot(
            at_wall=time.time(),
            num_requests_waiting=raw.get("vllm:num_requests_waiting", 0.0),
            num_requests_running=raw.get("vllm:num_requests_running", 0.0),
            kv_cache_usage_perc=raw.get("vllm:kv_cache_usage_perc", 0.0),
            num_preemptions_total=raw.get("vllm:num_preemptions_total", 0.0),
            prompt_tokens_total=raw.get("vllm:prompt_tokens_total", 0.0),
            generation_tokens_total=raw.get("vllm:generation_tokens_total", 0.0),
            kv_total_tokens=_kv_total_tokens_from_raw(raw),
            raw=raw,
        )
        if self._prev is not None:
            dt = max(1e-3, snap.at_wall - self._prev.at_wall)
            snap.prompt_token_rate = max(
                0.0, (snap.prompt_tokens_total - self._prev.prompt_tokens_total) / dt
            )
            snap.generation_token_rate = max(
                0.0,
                (snap.generation_tokens_total - self._prev.generation_tokens_total) / dt,
            )
            snap.preemption_rate = max(
                0.0,
                (snap.num_preemptions_total - self._prev.num_preemptions_total) / dt,
            )
        self._prev = snap
        return snap


class MultiVllmMetricsScraper:
    """Scrape multiple vLLM engines and aggregate them into one snapshot."""

    def __init__(self, api_bases: List[str], timeout_s: float = 2.0) -> None:
        bases = [b.rstrip("/") for b in api_bases if b.strip()]
        if not bases:
            raise ValueError("api_bases must not be empty")
        self.api_bases = bases
        self._scrapers = [VllmMetricsScraper(api_base=base, timeout_s=timeout_s) for base in bases]
        self._engine_states: List[VllmEngineSnapshot] = [
            VllmEngineSnapshot(
                index=idx,
                api_base=base,
                ok=False,
                error="not scraped yet",
                snapshot=None,
            )
            for idx, base in enumerate(bases)
        ]
        self._last: Optional[VllmSnapshot] = None

    async def aclose(self) -> None:
        await asyncio.gather(
            *[scraper.aclose() for scraper in self._scrapers],
            return_exceptions=True,
        )

    def last_snapshot(self) -> Optional[VllmSnapshot]:
        return self._last

    def engine_states(self) -> List[VllmEngineSnapshot]:
        return list(self._engine_states)

    async def scrape(self) -> VllmSnapshot:
        results = await asyncio.gather(
            *[scraper.scrape() for scraper in self._scrapers],
            return_exceptions=True,
        )
        healthy: List[VllmSnapshot] = []
        engine_states: List[VllmEngineSnapshot] = []
        for idx, (api_base, scraper, result) in enumerate(
            zip(self.api_bases, self._scrapers, results)
        ):
            if isinstance(result, Exception):
                engine_states.append(
                    VllmEngineSnapshot(
                        index=idx,
                        api_base=api_base,
                        ok=False,
                        error=str(result),
                        snapshot=scraper.last_snapshot(),
                    )
                )
                continue
            healthy.append(result)
            engine_states.append(
                VllmEngineSnapshot(
                    index=idx,
                    api_base=api_base,
                    ok=True,
                    error=None,
                    snapshot=result,
                )
            )
        self._engine_states = engine_states
        if not healthy:
            raise RuntimeError("no healthy vLLM metrics endpoints")
        snap = VllmSnapshot(
            at_wall=max(s.at_wall for s in healthy),
            num_requests_waiting=sum(s.num_requests_waiting for s in healthy),
            num_requests_running=sum(s.num_requests_running for s in healthy),
            kv_cache_usage_perc=sum(s.kv_cache_usage_perc for s in healthy) / len(healthy),
            num_preemptions_total=sum(s.num_preemptions_total for s in healthy),
            prompt_tokens_total=sum(s.prompt_tokens_total for s in healthy),
            generation_tokens_total=sum(s.generation_tokens_total for s in healthy),
            kv_total_tokens=(
                sum(int(s.kv_total_tokens) for s in healthy if s.kv_total_tokens is not None)
                if any(s.kv_total_tokens is not None for s in healthy)
                else None
            ),
            raw={
                "vllm:num_requests_waiting": sum(s.num_requests_waiting for s in healthy),
                "vllm:num_requests_running": sum(s.num_requests_running for s in healthy),
                "vllm:kv_cache_usage_perc": sum(s.kv_cache_usage_perc for s in healthy) / len(healthy),
                "vllm:num_preemptions_total": sum(s.num_preemptions_total for s in healthy),
                "vllm:prompt_tokens_total": sum(s.prompt_tokens_total for s in healthy),
                "vllm:generation_tokens_total": sum(s.generation_tokens_total for s in healthy),
            },
            prompt_token_rate=sum(s.prompt_token_rate for s in healthy),
            generation_token_rate=sum(s.generation_token_rate for s in healthy),
            preemption_rate=sum(s.preemption_rate for s in healthy),
        )
        self._last = snap
        return snap
