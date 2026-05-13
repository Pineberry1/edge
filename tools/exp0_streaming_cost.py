"""Exp0: streaming online-prefill cost scaling probe.

The tool drives one stream against an existing vLLM endpoint and records the
prompt/prefill token counter delta from ``/metrics``.  Baseline semantics are:

* b1/b2/b3: create one chat-completion request per window; request i includes
  all frames from windows [0, i], so visual prefill work scales as T^2 unless
  the server-side baseline removes it.
* b4: create one online-prefill session, append each window once, and decode at
  final stream_end; visual prefill work scales as T.

Server flags are intentionally external to the tool.  Run b1 on a server with
prefix cache disabled, b2 with prefix cache enabled, b3 with native chunked
prefill enabled, and b4 with online prefill enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import request as urlreq

import av

from edge.tools.completion_bench import (
    Frame,
    discover_model,
    extract_frames,
)


REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMPT = (
    "Does the streamed video segment contain any abnormal, criminal, or unsafe "
    "activity? Answer with only 'Yes' or 'No'."
)


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urlreq.Request(url, data=data, method=method, headers=headers)
    with urlreq.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
        return resp.status, json.loads(raw) if raw else {}


def _http_text(url: str, *, timeout: float = 5.0) -> str:
    req = urlreq.Request(url, method="GET")
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def _video_duration_s(path: Path) -> float:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        if container.duration:
            return float(container.duration / av.time_base)
    finally:
        container.close()
    raise RuntimeError(f"could not determine video duration for {path}")


def _extract_looped_window_frames(
    source: Path,
    *,
    window_idx: int,
    window_seconds: float,
    frames_per_window: int,
    jpeg_quality: int,
) -> list[Frame]:
    duration = _video_duration_s(source)
    if duration <= 0:
        raise RuntimeError(f"video duration must be positive: {source}")

    start = (window_idx * window_seconds) % duration
    end = start + window_seconds
    if end <= duration:
        frames = extract_frames(
            source, start, end, frames_per_window, jpeg_quality=jpeg_quality
        )
        if frames:
            return frames

    first_span = max(0.0, duration - start)
    first_n = int(round(frames_per_window * first_span / window_seconds))
    first_n = max(0, min(frames_per_window, first_n))
    second_n = frames_per_window - first_n
    frames: list[Frame] = []
    if first_n:
        frames.extend(
            extract_frames(source, start, duration, first_n, jpeg_quality=jpeg_quality)
        )
    if second_n:
        frames.extend(
            extract_frames(
                source, 0.0, max(0.01, end - duration), second_n,
                jpeg_quality=jpeg_quality,
            )
        )
    if not frames:
        frames = extract_frames(
            source, 0.0, min(duration, window_seconds), frames_per_window,
            jpeg_quality=jpeg_quality,
        )
    return frames


def _metric_snapshot(api_base: str) -> dict[str, float]:
    text = _http_text(f"{api_base.rstrip('/')}/metrics", timeout=5.0)
    metrics: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        lhs, rhs = line.rsplit(None, 1)
        name = lhs.split("{", 1)[0]
        try:
            value = float(rhs)
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def _delta_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(keys)
        if after.get(key, 0.0) - before.get(key, 0.0) != 0.0
    }


def _select_prompt_cost(delta: dict[str, float]) -> tuple[str | None, float | None]:
    candidates = [
        "vllm:request_prefill_kv_computed_tokens_sum",
        "vllm:prompt_tokens_total",
        "vllm:request_prompt_tokens_sum",
        "vllm:prompt_tokens_sum",
        "vllm:num_prompt_tokens_total",
        "vllm:input_tokens_total",
    ]
    for name in candidates:
        value = delta.get(name)
        if isinstance(value, (int, float)) and value > 0:
            return name, float(value)

    fuzzy = [
        (name, value)
        for name, value in delta.items()
        if value > 0 and "prompt" in name and "token" in name
    ]
    if fuzzy:
        name, value = sorted(fuzzy, key=lambda item: item[0])[0]
        return name, float(value)
    return None, None


def _kv_from_snapshot(metrics: dict[str, float]) -> float | None:
    vals = [
        value
        for name, value in metrics.items()
        if (
            name.endswith("kv_cache_usage_perc")
            or name.endswith("gpu_cache_usage_perc")
        ) and math.isfinite(value)
    ]
    return max(vals) if vals else None


class MetricsPoller:
    def __init__(self, api_base: str, interval_s: float, out_path: Path) -> None:
        self.api_base = api_base.rstrip("/")
        self.interval_s = interval_s
        self.out_path = out_path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples: list[dict[str, Any]] = []

    def __enter__(self) -> "MetricsPoller":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 4))

    def _run(self) -> None:
        with self.out_path.open("w") as fp:
            while not self._stop.is_set():
                rec: dict[str, Any] = {"t": time.time()}
                try:
                    metrics = _metric_snapshot(self.api_base)
                    rec.update({
                        "kv": _kv_from_snapshot(metrics),
                        "waiting": _sum_metric_suffix(metrics, "num_requests_waiting"),
                        "running": _sum_metric_suffix(metrics, "num_requests_running"),
                        "swapped": _sum_metric_suffix(metrics, "num_requests_swapped"),
                    })
                except Exception as exc:
                    rec["error"] = repr(exc)
                self.samples.append(rec)
                fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
                fp.flush()
                self._stop.wait(self.interval_s)

    @property
    def kv_max(self) -> float | None:
        vals = [
            float(item["kv"])
            for item in self.samples
            if isinstance(item.get("kv"), (int, float))
        ]
        return max(vals) if vals else None


def _sum_metric_suffix(metrics: dict[str, float], suffix: str) -> float | None:
    vals = [value for name, value in metrics.items() if name.endswith(suffix)]
    return sum(vals) if vals else None


def _percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = math.ceil(q * len(xs)) - 1
    return xs[max(0, min(idx, len(xs) - 1))]


def _describe(values: list[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if math.isfinite(v)]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "p50": _percentile(xs, 0.50),
        "p95": _percentile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def _image_content(frames: list[Frame], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"},
        })
    content.append({"type": "text", "text": prompt})
    return content


def _chat_completion(
    api_base: str,
    model: str,
    frames: list[Frame],
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": _image_content(frames, prompt)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    started = time.time()
    _, result = _http_json(
        "POST",
        f"{api_base.rstrip('/')}/v1/chat/completions",
        body,
        timeout=timeout,
    )
    elapsed_ms = (time.time() - started) * 1000.0
    text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text, {
        "elapsed_ms": elapsed_ms,
        "usage": result.get("usage") or {},
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "request_id": result.get("id"),
    }


def _run_full_prefill_baseline(
    args: argparse.Namespace,
    *,
    model: str,
    windows: list[list[Frame]],
    responses_path: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with responses_path.open("w") as fp:
        for idx in range(args.T):
            frames = [frame for window in windows[: idx + 1] for frame in window]
            text, meta = _chat_completion(
                args.vllm_api_base,
                model,
                frames,
                args.prompt,
                args.max_tokens,
                args.request_timeout,
            )
            usage = meta.get("usage") or {}
            rec = {
                "window_id": idx,
                "request_frames": len(frames),
                "elapsed_ms": meta.get("elapsed_ms"),
                "text": text,
                "usage": usage,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": meta.get("finish_reason"),
                "request_id": meta.get("request_id"),
            }
            records.append(rec)
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
    prompt_tokens = [
        float(r["prompt_tokens"])
        for r in records
        if isinstance(r.get("prompt_tokens"), (int, float))
    ]
    return {
        "requests": records,
        "request_elapsed_ms": _describe([
            float(r["elapsed_ms"])
            for r in records
            if isinstance(r.get("elapsed_ms"), (int, float))
        ]),
        "usage_prompt_tokens_sum": sum(prompt_tokens) if prompt_tokens else None,
        "usage_prompt_tokens": _describe(prompt_tokens),
    }


def _run_chunked_prefill_baseline(
    args: argparse.Namespace,
    *,
    model: str,
    windows: list[list[Frame]],
    responses_path: Path,
) -> dict[str, Any]:
    frames = [frame for window in windows for frame in window]
    text, meta = _chat_completion(
        args.vllm_api_base,
        model,
        frames,
        args.prompt,
        args.max_tokens,
        args.request_timeout,
    )
    usage = meta.get("usage") or {}
    rec = {
        "window_id": "all",
        "request_frames": len(frames),
        "elapsed_ms": meta.get("elapsed_ms"),
        "text": text,
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
        "request_id": meta.get("request_id"),
    }
    responses_path.write_text(json.dumps(rec, ensure_ascii=False) + "\n")
    prompt_tokens = [
        float(rec["prompt_tokens"])
        if isinstance(rec.get("prompt_tokens"), (int, float))
        else math.nan
    ]
    return {
        "requests": [rec],
        "request_elapsed_ms": _describe([
            float(rec["elapsed_ms"])
            if isinstance(rec.get("elapsed_ms"), (int, float))
            else math.nan
        ]),
        "usage_prompt_tokens_sum": (
            sum(v for v in prompt_tokens if math.isfinite(v))
            if any(math.isfinite(v) for v in prompt_tokens)
            else None
        ),
        "usage_prompt_tokens": _describe(prompt_tokens),
    }


def _run_online_prefill_baseline(
    args: argparse.Namespace,
    *,
    model: str,
    windows: list[list[Frame]],
    responses_path: Path,
) -> dict[str, Any]:
    request_id = args.request_id or f"exp0-{uuid.uuid4().hex[:12]}"
    create_body = {
        "request_id": request_id,
        "model": model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    _, create_result = _http_json(
        "POST",
        f"{args.vllm_api_base.rstrip('/')}/v1/online_prefill/sessions",
        create_body,
        timeout=args.request_timeout,
    )

    append_records: list[dict[str, Any]] = []
    with responses_path.open("w") as fp:
        fp.write(json.dumps({"event": "create", "response": create_result}) + "\n")
        for idx, frames in enumerate(windows):
            body = {
                "frames": [
                    {
                        "data": f"data:image/jpeg;base64,{frame.image_b64}",
                        "mime_type": "image/jpeg",
                    }
                    for frame in frames
                ],
                "stream_end": idx == len(windows) - 1,
            }
            started = time.time()
            _, append_result = _http_json(
                "POST",
                (
                    f"{args.vllm_api_base.rstrip('/')}/v1/online_prefill/sessions/"
                    f"{request_id}/append"
                ),
                body,
                timeout=args.request_timeout,
            )
            elapsed_ms = (time.time() - started) * 1000.0
            rec = {
                "event": "append",
                "window_id": idx,
                "request_frames": len(frames),
                "elapsed_ms": elapsed_ms,
                "response": append_result,
            }
            append_records.append(rec)
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()

        decode_start = time.time()
        deadline = decode_start + args.request_timeout
        result: dict[str, Any] = {}
        while time.time() < deadline:
            _, result = _http_json(
                "GET",
                (
                    f"{args.vllm_api_base.rstrip('/')}/v1/online_prefill/sessions/"
                    f"{request_id}/result"
                ),
                timeout=10.0,
            )
            if result.get("finished") or result.get("error"):
                break
            time.sleep(args.poll_interval)
        decode_elapsed_ms = (time.time() - decode_start) * 1000.0
        fp.write(json.dumps({
            "event": "result",
            "decode_wait_ms": decode_elapsed_ms,
            "response": result,
        }, ensure_ascii=False) + "\n")

    return {
        "request_id": request_id,
        "create_response": create_result,
        "append_records": append_records,
        "append_elapsed_ms": _describe([
            float(r["elapsed_ms"])
            for r in append_records
            if isinstance(r.get("elapsed_ms"), (int, float))
        ]),
        "decode_wait_ms": decode_elapsed_ms,
        "result": result,
        "output_text": result.get("output_text"),
        "finished": bool(result.get("finished")),
        "error": result.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=["b1", "b2", "b3", "b4"])
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--vllm-api-base", required=True)
    parser.add_argument("--frames-per-window", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--metrics-interval", type=float, default=0.2)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.T <= 0:
        raise SystemExit("--T must be positive")
    if args.frames_per_window <= 0:
        raise SystemExit("--frames-per-window must be positive")
    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cmd.txt").write_text(" ".join(sys.argv) + "\n")

    model = discover_model(args.vllm_api_base)
    windows = [
        _extract_looped_window_frames(
            args.source,
            window_idx=idx,
            window_seconds=args.window_seconds,
            frames_per_window=args.frames_per_window,
            jpeg_quality=args.jpeg_quality,
        )
        for idx in range(args.T)
    ]

    before_metrics = _metric_snapshot(args.vllm_api_base)
    started_at = time.time()
    started_perf = time.perf_counter()
    responses_path = args.out / "responses.jsonl"
    with MetricsPoller(
        args.vllm_api_base, args.metrics_interval, args.out / "metrics_timeseries.jsonl"
    ) as poller:
        if args.baseline == "b4":
            run = _run_online_prefill_baseline(
                args, model=model, windows=windows, responses_path=responses_path
            )
        elif args.baseline == "b3":
            run = _run_chunked_prefill_baseline(
                args, model=model, windows=windows, responses_path=responses_path
            )
        else:
            run = _run_full_prefill_baseline(
                args, model=model, windows=windows, responses_path=responses_path
            )
        kv_max = poller.kv_max
    ended_at = time.time()
    wall_s = time.perf_counter() - started_perf
    after_metrics = _metric_snapshot(args.vllm_api_base)
    metric_delta = _delta_metrics(before_metrics, after_metrics)
    cost_metric, accumulated_prefill_cost = _select_prompt_cost(metric_delta)

    summary = {
        "tool": "edge.tools.exp0_streaming_cost",
        "baseline": args.baseline,
        "T": args.T,
        "window_seconds": args.window_seconds,
        "frames_per_window": args.frames_per_window,
        "source": str(args.source),
        "model": model,
        "vllm_api_base": args.vllm_api_base,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": wall_s,
        "cost_metric": cost_metric,
        "accumulated_prefill_cost": accumulated_prefill_cost,
        "kv_max": kv_max,
        "metric_delta": metric_delta,
        "run": run,
    }
    if accumulated_prefill_cost is None and args.baseline != "b4":
        summary["accumulated_prefill_cost"] = run.get("usage_prompt_tokens_sum")
        summary["cost_metric"] = "usage.prompt_tokens_sum"

    for name in ("summary.json", "ab_summary.json"):
        (args.out / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"[exp0] baseline={args.baseline} T={args.T} "
        f"cost={summary.get('accumulated_prefill_cost')} "
        f"metric={summary.get('cost_metric')} kv_max={kv_max}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
