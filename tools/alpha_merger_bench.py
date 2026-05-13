"""Direct vLLM alpha/token-merger benchmark for video clips.

This tool intentionally bypasses the BAVA intake path.  It sends pre-cut
video clips to vLLM's OpenAI-compatible chat endpoint with `video_url` input,
then controls Qwen3-VL's visual token merger via request-level
`mm_processor_kwargs`.

Output layout is compatible with tools/summarize_anomaly_f1.py:

  <out>/<config>/manifest.json
  <out>/<config>/edge-<stream_id>.log
  <out>/<config>/summary.json

Example:

  python -m edge.tools.alpha_merger_bench \
      --manifest edge/data/eval_slices_videos.tsv \
      --vllm-api-base-list http://127.0.0.1:18011,http://127.0.0.1:18012 \
      --frames-per-window 80 --concurrency 2 \
      --configs static_full static_alpha050 bava_dynamic \
      --out edge/data/bench_runs/alpha_merger_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import mimetypes
import re
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlreq


REPO = Path(__file__).resolve().parent.parent.parent


DEFAULT_PROMPT = (
    "Does this video clip contain any abnormal, criminal, or unsafe activity? "
    "Answer with only 'Yes' or 'No'."
)


@dataclass(frozen=True)
class BenchConfig:
    name: str
    alpha: float | None
    rho: float
    dynamic: bool
    description: str


CONFIGS: dict[str, BenchConfig] = {
    "static_full": BenchConfig(
        name="static_full",
        alpha=1.0,
        rho=1.0,
        dynamic=False,
        description="Native video baseline: rho=1.0, alpha=1.0, no merger kwargs.",
    ),
    "static_alpha075": BenchConfig(
        name="static_alpha075",
        alpha=0.75,
        rho=1.0,
        dynamic=False,
        description="Static cloud token folding: rho=1.0, alpha=0.75.",
    ),
    "static_alpha050": BenchConfig(
        name="static_alpha050",
        alpha=0.50,
        rho=1.0,
        dynamic=False,
        description="Static cloud token folding: rho=1.0, alpha=0.50.",
    ),
    "static_alpha025": BenchConfig(
        name="static_alpha025",
        alpha=0.25,
        rho=1.0,
        dynamic=False,
        description="Static cloud token folding: rho=1.0, alpha=0.25.",
    ),
    "bava_dynamic": BenchConfig(
        name="bava_dynamic",
        alpha=None,
        rho=0.50,
        dynamic=True,
        description=(
            "Direct-tool BAVA controller: dynamically adjusts rho/num_frames "
            "and merger alpha from vLLM metrics."
        ),
    ),
}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            row: dict[str, Any] = dict(zip(header, line.split("\t")))
            if "duration_s" in row:
                row["duration_s"] = float(row["duration_s"])
            rows.append(row)
    return rows


def resolve_sources(
    rows: list[dict[str, Any]],
    local_root: Path,
    remote_prefix: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    remote_prefix = remote_prefix.rstrip("/")
    for row in rows:
        rec = dict(row)
        if rec.get("local_path"):
            source = str(rec["local_path"])
        else:
            cloud_path = str(rec.get("cloud_path") or rec.get("source") or "")
            if remote_prefix and cloud_path.startswith(remote_prefix + "/"):
                source = str(local_root / cloud_path[len(remote_prefix) + 1:])
            else:
                source = cloud_path
        rec["source"] = source
        rec["parent_video_id"] = rec.get("parent_video_id") or rec.get("video_id")
        out.append(rec)
    return out


def http_json(method: str, url: str, payload: dict[str, Any] | None = None,
              timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urlreq.Request(url, data=data, method=method, headers=headers)
    with urlreq.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
        return resp.status, json.loads(raw) if raw else {}


def http_text(url: str, timeout: float = 5.0) -> str:
    req = urlreq.Request(url, method="GET")
    with urlreq.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


class KeepAliveJsonClient:
    """Tiny stdlib HTTP/1.1 JSON client with one reusable connection."""

    def __init__(self, api_base: str, timeout: float) -> None:
        parsed = urlparse.urlparse(api_base.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"unsupported api base: {api_base}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._prefix = parsed.path.rstrip("/")
        self._timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Connection": "keep-alive"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        raw = self._request(method, path, body=body, headers=headers)
        return raw[0], json.loads(raw[1].decode()) if raw[1] else {}

    def text(self, path: str) -> str:
        _, body = self._request(
            "GET",
            path,
            body=None,
            headers={"Connection": "keep-alive"},
        )
        return body.decode(errors="replace")

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is not None:
            return self._conn
        if self._scheme == "https":
            self._conn = http.client.HTTPSConnection(
                self._host, self._port, timeout=self._timeout
            )
        else:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout
            )
        return self._conn

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
    ) -> tuple[int, bytes]:
        request_path = f"{self._prefix}{path}" if self._prefix else path
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                conn = self._connect()
                conn.request(method, request_path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status >= 400:
                    raise RuntimeError(
                        f"HTTP {resp.status} {resp.reason}: "
                        f"{data[:500].decode(errors='replace')}"
                    )
                return resp.status, data
            except (http.client.HTTPException, OSError, RuntimeError) as exc:
                last_exc = exc
                self.close()
                if attempt == 1:
                    break
        assert last_exc is not None
        raise last_exc


def discover_model(api_base: str) -> str:
    _, data = http_json("GET", f"{api_base.rstrip('/')}/v1/models", timeout=10.0)
    return data["data"][0]["id"]


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not xs:
        return None
    k = math.ceil(q * len(xs)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def describe(values: list[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def even_frame_count(base_frames: int, rho: float) -> int:
    n = max(2, int(round(base_frames * max(0.0, min(1.0, rho)))))
    if n % 2:
        n += 1
    return min(max(2, n), max(2, base_frames))


def data_uri_for_video(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def build_mm_kwargs(
    alpha: float,
    num_frames: int,
    block_t: int,
    block_hw: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_frames": num_frames}
    if alpha < 0.999:
        kwargs.update({
            "visual_token_merger_alpha": alpha,
            "visual_token_merger_block_t": block_t,
            "visual_token_merger_block_hw": block_hw,
        })
    return kwargs


def chat_completion_video(
    api_base: str,
    model: str,
    video_path: Path,
    prompt: str,
    max_tokens: int,
    mm_kwargs: dict[str, Any],
    timeout: float,
    client: KeepAliveJsonClient | None = None,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": data_uri_for_video(video_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "mm_processor_kwargs": mm_kwargs,
    }
    started = time.perf_counter()
    if client is not None:
        _, result = client.json("POST", "/v1/chat/completions", payload=body)
    else:
        _, result = http_json(
            "POST",
            f"{api_base.rstrip('/')}/v1/chat/completions",
            payload=body,
            timeout=timeout,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text, {
        "elapsed_ms": elapsed_ms,
        "usage": result.get("usage") or {},
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "request_id": result.get("id"),
    }


_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def scrape_vllm_metrics(api_base: str, timeout: float = 3.0) -> dict[str, float | None]:
    try:
        text = http_text(f"{api_base.rstrip('/')}/metrics", timeout=timeout)
    except Exception:
        return {"kv": None, "waiting": None, "running": None, "swapped": None}

    kv_values: list[float] = []
    waiting = 0.0
    running = 0.0
    swapped = 0.0
    saw_waiting = saw_running = saw_swapped = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        value = float(m.group("value"))
        if name.endswith("gpu_cache_usage_perc"):
            kv_values.append(value)
        elif name.endswith("num_requests_waiting"):
            waiting += value
            saw_waiting = True
        elif name.endswith("num_requests_running"):
            running += value
            saw_running = True
        elif name.endswith("num_requests_swapped"):
            swapped += value
            saw_swapped = True
    return {
        "kv": max(kv_values) if kv_values else None,
        "waiting": waiting if saw_waiting else None,
        "running": running if saw_running else None,
        "swapped": swapped if saw_swapped else None,
    }


class DynamicController:
    """Small direct-benchmark controller for rho and alpha.

    The production BAVA controller lives in intake.  This class is deliberately
    local to the benchmark so the experiment can exercise vLLM's merger path
    without modifying intake.
    """

    def __init__(
        self,
        *,
        log_path: Path,
        initial_rho: float,
        initial_alpha: float,
        rho_floor: float,
        alpha_floor: float,
        step: float,
        target_kv: float,
        target_waiting: float,
    ) -> None:
        self._lock = threading.Lock()
        self._rho = initial_rho
        self._alpha = initial_alpha
        self._rho_floor = rho_floor
        self._alpha_floor = alpha_floor
        self._step = step
        self._target_kv = target_kv
        self._target_waiting = target_waiting
        self._log_path = log_path
        self._log_fp = log_path.open("w")

    def close(self) -> None:
        self._log_fp.close()

    def current(self) -> tuple[float, float]:
        with self._lock:
            return self._rho, self._alpha

    def update(
        self,
        *,
        api_base: str,
        stream_id: str,
        elapsed_ms: float | None,
        prompt_tokens: int | None,
        ok: bool,
    ) -> dict[str, Any]:
        metrics = scrape_vllm_metrics(api_base)
        with self._lock:
            old_rho = self._rho
            old_alpha = self._alpha
            kv = metrics.get("kv")
            waiting = metrics.get("waiting")
            running = metrics.get("running")
            pressure = False
            reason = "hold"

            if ok is False:
                pressure = True
                reason = "request_error"
            if isinstance(kv, float) and kv > self._target_kv + 0.05:
                pressure = True
                reason = "kv_high"
            if isinstance(waiting, float) and waiting > self._target_waiting + 1.0:
                pressure = True
                reason = "queue_high"

            relaxed = (
                (kv is None or kv < self._target_kv - 0.10)
                and (waiting is None or waiting <= self._target_waiting)
                and ok
            )

            if pressure:
                self._rho = max(self._rho_floor, self._rho - self._step)
                self._alpha = max(self._alpha_floor, self._alpha - self._step)
            elif relaxed:
                self._rho = min(1.0, self._rho + self._step)
                self._alpha = min(1.0, self._alpha + self._step)
                reason = "climb_back" if (self._rho != old_rho or self._alpha != old_alpha) else "hold"

            rec = {
                "t": time.time(),
                "stream_id": stream_id,
                "api_base": api_base,
                "ok": ok,
                "reason": reason,
                "old_rho": old_rho,
                "old_alpha": old_alpha,
                "rho": self._rho,
                "alpha": self._alpha,
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": prompt_tokens,
                "kv": kv,
                "waiting": waiting,
                "running": running,
                "swapped": metrics.get("swapped"),
            }
            self._log_fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._log_fp.flush()
            return rec


def run_one_config(
    cfg: BenchConfig,
    rows: list[dict[str, Any]],
    model: str,
    api_bases: list[str],
    args: argparse.Namespace,
    root_out: Path,
) -> dict[str, Any]:
    cfg_out = root_out / cfg.name
    cfg_out.mkdir(parents=True, exist_ok=True)

    controller: DynamicController | None = None
    if cfg.dynamic:
        controller = DynamicController(
            log_path=cfg_out / "controller.jsonl",
            initial_rho=args.dynamic_initial_rho,
            initial_alpha=args.dynamic_initial_alpha,
            rho_floor=args.dynamic_rho_floor,
            alpha_floor=args.dynamic_alpha_floor,
            step=args.dynamic_step,
            target_kv=args.dynamic_target_kv,
            target_waiting=args.dynamic_target_waiting,
        )

    completed: list[dict[str, Any]] = []
    completed_lock = threading.Lock()
    work_lock = threading.Lock()
    work_idx = [0]

    def worker(slot: int) -> None:
        api_base = api_bases[slot % len(api_bases)]
        client = None if args.no_http_keepalive else KeepAliveJsonClient(
            api_base, timeout=args.request_timeout
        )
        try:
            while True:
                with work_lock:
                    if work_idx[0] >= len(rows):
                        return
                    idx = work_idx[0]
                    work_idx[0] += 1
                row = rows[idx]
                sid = f"v-{idx:03d}"
                video_path = Path(str(row["source"]))
                log_path = cfg_out / f"edge-{sid}.log"
                started = time.time()
                started_perf = time.perf_counter()
                per_window: list[dict[str, Any]] = []
                ok = False
                error: str | None = None

                if controller is not None:
                    rho, alpha = controller.current()
                else:
                    rho = cfg.rho
                    alpha = float(cfg.alpha if cfg.alpha is not None else 1.0)
                num_frames = even_frame_count(args.frames_per_window, rho)
                mm_kwargs = build_mm_kwargs(
                    alpha=alpha,
                    num_frames=num_frames,
                    block_t=args.visual_token_merger_block_t,
                    block_hw=args.visual_token_merger_block_hw,
                )

                with log_path.open("w") as log_fp:
                    log_fp.write(
                        f"[alpha-bench] config={cfg.name} slot={slot} api_base={api_base} "
                        f"rho={rho:.4f} alpha={alpha:.4f} num_frames={num_frames} "
                        f"keepalive={client is not None} source={video_path}\n"
                    )
                    log_fp.flush()
                    print(
                        f"[alpha][{cfg.name}][slot{slot}] {sid} {row.get('label')} "
                        f"{row.get('video_id')} rho={rho:.2f} alpha={alpha:.2f} "
                        f"frames={num_frames}",
                        flush=True,
                    )
                    try:
                        text, meta = chat_completion_video(
                            api_base=api_base,
                            model=model,
                            video_path=video_path,
                            prompt=args.prompt,
                            max_tokens=args.max_tokens,
                            mm_kwargs=mm_kwargs,
                            timeout=args.request_timeout,
                            client=client,
                        )
                        ok = True
                        usage = meta.get("usage") or {}
                        prompt_tokens = usage.get("prompt_tokens")
                        per_window.append({
                            "window_id": 0,
                            "text": text,
                            "elapsed_ms": meta.get("elapsed_ms"),
                            "usage": usage,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": usage.get("completion_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                            "finish_reason": meta.get("finish_reason"),
                            "request_id": meta.get("request_id"),
                            "rho": rho,
                            "alpha": alpha,
                            "num_frames": num_frames,
                            "mm_processor_kwargs": mm_kwargs,
                        })
                        log_fp.write(f"[edge-uplink] result window=0 text={text!r}\n")
                        log_fp.flush()
                    except Exception as exc:
                        error = repr(exc)
                        log_fp.write(f"[edge-uplink] vllm-fail window=0 err={error}\n")
                        log_fp.flush()
                        usage = {}
                        prompt_tokens = None

                ended = time.time()
                wall_s = time.perf_counter() - started_perf
                if controller is not None:
                    elapsed_ms = per_window[0].get("elapsed_ms") if per_window else None
                    controller.update(
                        api_base=api_base,
                        stream_id=sid,
                        elapsed_ms=float(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else None,
                        prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
                        ok=ok,
                    )

                rec = {
                    "stream_id": sid,
                    "video_id": row.get("video_id"),
                    "parent_video_id": row.get("parent_video_id") or row.get("video_id"),
                    "label": row.get("label"),
                    "source": str(video_path),
                    "duration_s": row.get("duration_s"),
                    "api_base": api_base,
                    "config": cfg.name,
                    "rho": rho,
                    "alpha": alpha,
                    "num_frames": num_frames,
                    "started_at": started,
                    "ended_at": ended,
                    "wall_s": wall_s,
                    "returncode": 0 if ok else 1,
                    "error": error,
                    "log": str(log_path),
                    "per_window": per_window,
                }
                with completed_lock:
                    completed.append(rec)
        finally:
            if client is not None:
                client.close()

    started_at = time.time()
    started_perf = time.perf_counter()
    threads = [
        threading.Thread(target=worker, args=(slot,), daemon=False)
        for slot in range(max(1, args.concurrency))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ended_at = time.time()
    wall_s = time.perf_counter() - started_perf
    if controller is not None:
        controller.close()

    completed_sorted = sorted(completed, key=lambda r: r["stream_id"])
    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": wall_s,
        "tool": "edge.tools.alpha_merger_bench",
        "config": cfg.name,
        "description": cfg.description,
        "n_videos": len(rows),
        "concurrency": args.concurrency,
        "frames_per_window": args.frames_per_window,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "model": model,
        "vllm_api_base_list": ",".join(api_bases),
        "http_keepalive": not args.no_http_keepalive,
        "visual_token_merger_block_t": args.visual_token_merger_block_t,
        "visual_token_merger_block_hw": args.visual_token_merger_block_hw,
        "static_alpha": cfg.alpha,
        "static_rho": cfg.rho,
        "dynamic": cfg.dynamic,
        "streams": completed_sorted,
    }
    (cfg_out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    windows = [w for s in completed_sorted for w in (s.get("per_window") or [])]
    elapsed_ms = [
        float(w["elapsed_ms"]) for w in windows
        if isinstance(w.get("elapsed_ms"), (int, float))
    ]
    prompt_tokens = [
        float(w["prompt_tokens"]) for w in windows
        if isinstance(w.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        float(w["completion_tokens"]) for w in windows
        if isinstance(w.get("completion_tokens"), (int, float))
    ]
    total_tokens = [
        float(w["total_tokens"]) for w in windows
        if isinstance(w.get("total_tokens"), (int, float))
    ]
    summary = {
        "config": cfg.name,
        "manifest": str(cfg_out / "manifest.json"),
        "n_videos": len(rows),
        "n_success": sum(1 for s in completed_sorted if s.get("returncode") == 0),
        "n_error": sum(1 for s in completed_sorted if s.get("returncode") != 0),
        "n_total_windows": len(windows),
        "wall_s": wall_s,
        "requests_per_min": (len(windows) / wall_s * 60.0)
        if wall_s > 0 else None,
        "request_elapsed_ms": describe(elapsed_ms),
        "prompt_tokens": describe(prompt_tokens),
        "completion_tokens": describe(completion_tokens),
        "total_tokens": describe(total_tokens),
    }
    (cfg_out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[alpha][{cfg.name}] DONE success={summary['n_success']}/{len(rows)} "
        f"wall={summary['wall_s']:.1f}s rpm={summary['requests_per_min']:.2f}",
        flush=True,
    )
    return summary


def parse_api_bases(args: argparse.Namespace) -> list[str]:
    bases: list[str] = []
    if args.vllm_api_base_list:
        bases.extend(s.strip().rstrip("/") for s in args.vllm_api_base_list.split(",") if s.strip())
    if args.vllm_api_base:
        bases.append(args.vllm_api_base.rstrip("/"))
    seen: set[str] = set()
    uniq: list[str] = []
    for base in bases:
        if base and base not in seen:
            uniq.append(base)
            seen.add(base)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--vllm-api-base", default="")
    parser.add_argument("--vllm-api-base-list", default="")
    parser.add_argument("--frames-per-window", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument(
        "--no-http-keepalive",
        action="store_true",
        help="disable per-worker reusable HTTP connections",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["static_full", "static_alpha075", "static_alpha050", "static_alpha025", "bava_dynamic"],
        choices=sorted(CONFIGS),
    )
    parser.add_argument("--visual-token-merger-block-t", type=int, default=1)
    parser.add_argument("--visual-token-merger-block-hw", type=int, default=2)
    parser.add_argument("--local-root", type=Path, default=REPO / "edge/data/ucf_eval")
    parser.add_argument("--remote-prefix", default="/home/mambauser/tangxuan/ucf_crime_hf")
    parser.add_argument("--dynamic-initial-rho", type=float, default=0.50)
    parser.add_argument("--dynamic-initial-alpha", type=float, default=1.00)
    parser.add_argument("--dynamic-rho-floor", type=float, default=0.10)
    parser.add_argument("--dynamic-alpha-floor", type=float, default=0.25)
    parser.add_argument("--dynamic-step", type=float, default=0.05)
    parser.add_argument("--dynamic-target-kv", type=float, default=0.60)
    parser.add_argument("--dynamic-target-waiting", type=float, default=1.0)
    args = parser.parse_args()

    api_bases = parse_api_bases(args)
    if not api_bases:
        print("[alpha] need --vllm-api-base or --vllm-api-base-list")
        return 2

    rows = resolve_sources(read_manifest(args.manifest), args.local_root, args.remote_prefix)
    if args.limit:
        rows = rows[: args.limit]
    missing = [row for row in rows if not Path(str(row["source"])).exists()]
    if missing:
        for row in missing[:5]:
            print(f"[alpha] MISSING: {row.get('source')}")
        print(f"[alpha] {len(missing)} missing sources")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    model = discover_model(api_bases[0])
    print(
        f"[alpha] rows={len(rows)} configs={','.join(args.configs)} "
        f"concurrency={args.concurrency} engines={len(api_bases)} model={model}",
        flush=True,
    )

    started_at = time.time()
    summaries: dict[str, Any] = {}
    for name in args.configs:
        summaries[name] = run_one_config(CONFIGS[name], rows, model, api_bases, args, args.out)
    ended_at = time.time()

    top = {
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": ended_at - started_at,
        "manifest": str(args.manifest),
        "model": model,
        "vllm_api_bases": api_bases,
        "configs": summaries,
    }
    (args.out / "alpha_merger_summary.json").write_text(
        json.dumps(top, indent=2, ensure_ascii=False)
    )
    print(f"[alpha] wrote {args.out / 'alpha_merger_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
