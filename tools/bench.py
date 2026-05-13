"""N-stream trace-replay benchmark for BAVA.

Launches N edge processes concurrently against an existing intake, each
reading a different UCF sample (round-robining when --n > #sources), all
with `--pace-realtime --loop-source` so window_close events arrive on a
predictable wall-clock cadence. In parallel this process:

  * Scrapes the intake `/healthz` every --probe-interval and stores a
    tick-level JSONL of (Q, KV, rho_i, alpha_i per stream).
  * At end, pulls `/stats/latency` for per-stream P50/P95 append_ms / e2e_ms.
  * Writes a run manifest JSON covering every stream's CLI args and PID.

Usage:
  python -m edge.tools.bench --n 4 --duration 60 \
      --cloud-ws-url ws://127.0.0.1:19100/stream \
      --intake-admin-base http://127.0.0.1:19100 \
      --sources edge/data/ucf/*.mp4 \
      --out /tmp/bava_bench_$(date +%s)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urlreq

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent  # edge/ dir's parent = /home/admin123/tangxuan
PYTHON = sys.executable


def _http_json(method: str, url: str, payload: Optional[dict] = None, timeout: float = 3.0) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode()
    req = urlreq.Request(url, data=body, method=method,
                         headers={"Content-Type": "application/json"} if body else {})
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:
        return 0, {"error": str(e)}


def launch_edge(
    stream_id: str,
    source: str,
    source_list: Optional[Path],
    cloud_ws_url: str,
    rho: float,
    window_s: float,
    decision_window_s: Optional[float],
    max_tokens: int,
    prompt: str,
    log_path: Path,
    frames_per_window: int,
    alpha: Optional[float] = None,
    inference_mode: str = "online_prefill",
    visual_memory_merge: bool = False,
    loop_source: bool = True,
    align_source_switch_to_decision: bool = False,
    linger_s: float = 15.0,
    linger_until_results: int = 0,
    max_run_seconds: Optional[float] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    # Edge owns temporal selection. This env is reported in hello as the
    # expected per-chunk frame budget for cloud-side budget accounting; intake
    # must not use it to re-sample decoded frames.
    env["BAVA_MAX_FRAMES_PER_WINDOW"] = str(max(1, int(frames_per_window)))
    cmd = [
        PYTHON, "-m", "edge.src.edge_main",
        "--cloud-ws-url", cloud_ws_url,
        "--stream-id", stream_id,
        "--rho", str(rho),
        "--window-seconds", str(window_s),
        "--max-tokens", str(max_tokens),
        "--linger-s", str(linger_s),
        "--pace-realtime",
        "--prompt", prompt,
        "--inference-mode", inference_mode,
    ]
    if max_run_seconds is not None and max_run_seconds > 0:
        cmd += ["--max-run-seconds", str(max_run_seconds)]
    if linger_until_results > 0:
        cmd += ["--linger-until-results", str(linger_until_results)]
    if source_list is not None:
        cmd += ["--source-list", str(source_list)]
    else:
        cmd += ["--source", source]
    if loop_source:
        cmd.append("--loop-source")
    if align_source_switch_to_decision:
        cmd.append("--align-source-switch-to-decision")
    if decision_window_s is not None and decision_window_s > 0:
        cmd += ["--decision-window-seconds", str(decision_window_s)]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if visual_memory_merge:
        cmd.append("--visual-memory-merge")
    log_fp = log_path.open("w")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env,
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def _source_duration_s(path: str) -> Optional[float]:
    try:
        import av  # type: ignore
    except Exception:
        return None
    try:
        with av.open(path) as container:
            if container.duration:
                return float(container.duration / av.time_base)
            streams = [s for s in container.streams if s.type == "video"]
            if streams and streams[0].duration and streams[0].time_base:
                return float(streams[0].duration * streams[0].time_base)
    except Exception:
        return None
    return None


def _write_partitioned_source_lists(
    out: Path,
    sources: List[str],
    n: int,
) -> List[Dict[str, Any]]:
    source_list_dir = out / "source_lists"
    source_list_dir.mkdir(parents=True, exist_ok=True)
    duration_cache: Dict[str, Optional[float]] = {}
    partitions: List[Dict[str, Any]] = []
    for i in range(n):
        assigned = sources[i::n]
        if not assigned:
            assigned = [sources[i % len(sources)]]
        path = source_list_dir / f"source-list-bench-{i:02d}.txt"
        path.write_text("\n".join(assigned) + "\n")
        total_duration = 0.0
        duration_known = True
        for src in assigned:
            if src not in duration_cache:
                duration_cache[src] = _source_duration_s(src)
            dur = duration_cache[src]
            if dur is None:
                duration_known = False
            else:
                total_duration += dur
        partitions.append(
            {
                "path": path,
                "sources": assigned,
                "source_count": len(assigned),
                "duration_s": total_duration if duration_known else None,
            }
        )
    return partitions


def _expand_sources(patterns: List[str]) -> List[str]:
    sources: List[str] = []
    for pat in patterns:
        if pat.startswith("@"):
            for raw in Path(pat[1:]).read_text().splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    sources.append(line)
            continue
        sources += sorted(glob.glob(pat))
    return list(dict.fromkeys(sources))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--duration", type=float, default=60.0, help="wall-clock seconds to run edges before stopping")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--sources", nargs="+", required=True,
                   help="glob(s) or @file lists of mp4 sources; round-robins when n > len(sources)")
    p.add_argument(
        "--playlist-mode",
        choices=["none", "partition"],
        default="none",
        help="partition source pool into per-camera source-list files",
    )
    p.add_argument(
        "--loop-source",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="loop each source or playlist after EOF; defaults on for normal sources and off for partition playlists",
    )
    p.add_argument(
        "--align-source-switch-to-decision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="for partition/source-list cameras, start each next source at a decision boundary",
    )
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=None, help="initial α (static baselines only); omit for dynamic controller")
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument(
        "--decision-window-seconds",
        type=float,
        default=0.0,
        help="coarser stream_end cadence; defaults to --window-seconds when <= 0",
    )
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--prompt", default="In one short sentence, describe what is happening in this clip.")
    p.add_argument(
        "--frames-per-window",
        type=int,
        default=int(os.environ.get("BAVA_MAX_FRAMES_PER_WINDOW", "8")),
        help="expected edge-selected frames per online-prefill chunk; passed in hello for budget accounting",
    )
    p.add_argument("--inference-mode", choices=["online_prefill", "completion"], default="online_prefill")
    p.add_argument(
        "--visual-memory-merge",
        action="store_true",
        help="set hello.visual_memory_merge=true for each stream",
    )
    p.add_argument("--probe-interval", type=float, default=0.5)
    p.add_argument(
        "--edge-linger-s",
        type=float,
        default=15.0,
        help="seconds each edge waits after graceful source/input end for cloud results",
    )
    p.add_argument(
        "--edge-max-run-seconds",
        type=float,
        default=0.0,
        help="pass --max-run-seconds to each edge so the input horizon ends without SIGTERM",
    )
    p.add_argument(
        "--edge-linger-until-results",
        type=int,
        default=0,
        help="pass --linger-until-results to each edge for result-aware graceful close",
    )
    p.add_argument(
        "--graceful-stop-timeout",
        type=float,
        default=0.0,
        help="after --duration, wait this long for edges to exit naturally before SIGTERM",
    )
    p.add_argument(
        "--stagger-seconds", type=float, default=0.0,
        help="Spread the launch of N edges evenly across this many seconds; "
             "0 (default) keeps the original tight 0.1s spacing.",
    )
    p.add_argument("--out", default=f"/tmp/bava_bench_{int(time.time())}")
    args = p.parse_args()

    sources = _expand_sources(args.sources)
    if not sources:
        print("no sources matched", file=sys.stderr)
        return 2
    print(f"sources: {len(sources)} ({sources[0]} ...)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probes_path = out / "probes.jsonl"
    manifest_path = out / "manifest.json"
    summary_path = out / "summary.json"
    loop_source = bool(args.loop_source) if args.loop_source is not None else args.playlist_mode == "none"
    align_source_switch = (
        bool(args.align_source_switch_to_decision)
        if args.align_source_switch_to_decision is not None
        else os.environ.get("BAVA_ALIGN_SOURCE_SWITCH_TO_DECISION", "").strip().lower()
        in {"1", "true", "yes", "on", "y"}
    )
    partitions = (
        _write_partitioned_source_lists(out, sources, args.n)
        if args.playlist_mode == "partition"
        else []
    )

    # Launch N edges
    procs: List[subprocess.Popen] = []
    manifest = {
        "started_at": time.time(),
        "n": args.n,
        "duration_s": args.duration,
        "rho_initial": args.rho,
        "window_seconds": args.window_seconds,
        "decision_window_seconds": args.decision_window_seconds,
        "max_tokens": args.max_tokens,
        "frames_per_window": args.frames_per_window,
        "prompt": args.prompt,
        "inference_mode": args.inference_mode,
        "visual_memory_merge": bool(args.visual_memory_merge),
        "edge_linger_s": args.edge_linger_s,
        "edge_linger_until_results": args.edge_linger_until_results,
        "edge_max_run_seconds": args.edge_max_run_seconds,
        "graceful_stop_timeout": args.graceful_stop_timeout,
        "playlist_mode": args.playlist_mode,
        "loop_source": loop_source,
        "align_source_switch_to_decision": align_source_switch,
        "intake_admin_base": args.intake_admin_base,
        "cloud_ws_url": args.cloud_ws_url,
        "sources_pool": sources,
        "streams": [],
    }

    for i in range(args.n):
        sid = f"bench-{i:02d}"
        part = partitions[i] if partitions else None
        src = (part["sources"][0] if part else sources[i % len(sources)])
        log = out / f"edge-{sid}.log"
        proc = launch_edge(
            stream_id=sid, source=src,
            source_list=part["path"] if part else None,
            cloud_ws_url=args.cloud_ws_url, rho=args.rho,
            window_s=args.window_seconds,
            decision_window_s=args.decision_window_seconds,
            max_tokens=args.max_tokens,
            prompt=args.prompt, log_path=log,
            frames_per_window=args.frames_per_window,
            alpha=args.alpha,
            inference_mode=args.inference_mode,
            visual_memory_merge=bool(args.visual_memory_merge),
            loop_source=loop_source,
            align_source_switch_to_decision=align_source_switch,
            linger_s=args.edge_linger_s,
            linger_until_results=max(0, int(args.edge_linger_until_results or 0)),
            max_run_seconds=(
                args.edge_max_run_seconds if args.edge_max_run_seconds > 0 else None
            ),
        )
        procs.append(proc)
        stream_row = {"stream_id": sid, "source": src, "pid": proc.pid, "log": str(log)}
        if part:
            stream_row.update(
                {
                    "source_list": str(part["path"]),
                    "source_count": part["source_count"],
                    "source_list_duration_s": part["duration_s"],
                    "source_list_first": part["sources"][:5],
                }
            )
        manifest["streams"].append(stream_row)
        if args.stagger_seconds > 0 and args.n > 1:
            time.sleep(args.stagger_seconds / (args.n - 1))
        else:
            time.sleep(0.1)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"launched {len(procs)} edges, probing every {args.probe_interval}s")

    # Probe loop
    t0 = time.time()
    n_ticks = 0
    stopped = False
    with probes_path.open("w") as pf:
        try:
            while time.time() - t0 < args.duration:
                tick_t = time.time()
                _, health = _http_json("GET", f"{args.intake_admin_base}/healthz")
                record = {
                    "t": tick_t,
                    "elapsed": tick_t - t0,
                    **{
                        k: health.get(k)
                        for k in (
                            "vllm_snapshot",
                            "vllm_engines",
                            "controller_streams",
                            "active_streams",
                            "send_window",
                            "token_flow",
                        )
                    },
                }
                pf.write(json.dumps(record, separators=(",", ":")) + "\n")
                n_ticks += 1
                # print a terse status line every ~2s
                if n_ticks % max(1, int(2 / args.probe_interval)) == 0:
                    snap = health.get("vllm_snapshot") or {}
                    streams = health.get("controller_streams") or {}
                    avg_rho = (sum(v["rho"] for v in streams.values()) / len(streams)) if streams else 0
                    avg_alpha = (sum(v["alpha"] for v in streams.values()) / len(streams)) if streams else 0
                    q = snap.get("Q") if snap.get("Q") is not None else -1.0
                    kv = snap.get("KV") if snap.get("KV") is not None else -1.0
                    run = snap.get("running") if snap.get("running") is not None else -1.0
                    print(f"  t+{tick_t-t0:5.1f}s  Q={q:.1f} KV={kv:.3f} "
                          f"running={run:.0f} ρ̄={avg_rho:.2f} ᾱ={avg_alpha:.2f} N={len(streams)}")
                time.sleep(max(0.0, args.probe_interval - (time.time() - tick_t)))
        except KeyboardInterrupt:
            stopped = True
            print("\n[bench] interrupted, stopping edges")

    graceful_wait_s = max(0.0, float(args.graceful_stop_timeout or 0.0))
    if graceful_wait_s > 0 and not stopped:
        print(f"[bench] waiting up to {graceful_wait_s:.1f}s for graceful edge exit")
        deadline = time.time() + graceful_wait_s
        last_alive = len(procs)
        while time.time() < deadline:
            alive = sum(1 for p in procs if p.poll() is None)
            if alive == 0:
                break
            if alive != last_alive:
                print(f"[bench] graceful wait alive_edges={alive}")
                last_alive = alive
            time.sleep(1.0)

    remaining_procs = [p for p in procs if p.poll() is None]
    if remaining_procs:
        # Stop edges: SIGTERM first (lets uplink flush), then SIGKILL after 8s.
        # We use SIGTERM because SIGINT inside pace-realtime's time.sleep wakes up
        # as KeyboardInterrupt, but the looped source + auto-reconnect can keep
        # the process alive longer than expected.
        print(f"[bench] sending SIGTERM to {len(remaining_procs)} remaining edges")
        for p in remaining_procs:
            try:
                p.terminate()
            except Exception:
                pass
        deadline = time.time() + 8
        for p in remaining_procs:
            remaining = max(0.1, deadline - time.time())
            try:
                p.wait(remaining)
            except subprocess.TimeoutExpired:
                print(f"[bench] edge pid={p.pid} didn't stop in time, SIGKILL")
                p.kill()
                try:
                    p.wait(2.0)
                except subprocess.TimeoutExpired:
                    pass
    else:
        print("[bench] all edges exited gracefully")
    for p in procs:
        try:
            p._log_fp.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Pull latency summary
    _, lat = _http_json("GET", f"{args.intake_admin_base}/stats/latency", timeout=3.0)

    summary = {
        "manifest": str(manifest_path),
        "probes_jsonl": str(probes_path),
        "n_probes": n_ticks,
        "latency_stats": lat,
        "stopped_early": stopped,
        "run_wall_seconds": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("\n=== BENCH SUMMARY ===")
    print(f"streams:         {args.n}")
    print(f"run wall:        {summary['run_wall_seconds']:.1f}s")
    print(f"probes written:  {n_ticks}")
    print(f"manifest:        {manifest_path}")
    print(f"probes:          {probes_path}")
    if isinstance(lat, dict) and "__all__" in lat:
        a = lat["__all__"]
        print(f"aggregate latency: n={int(a['n'])} "
              f"append p50={a['append_p50']:.0f}ms p95={a['append_p95']:.0f}ms "
              f"e2e p50={a['e2e_p50']:.0f}ms p95={a['e2e_p95']:.0f}ms")
        for sid, s in lat.items():
            if sid == "__all__":
                continue
            print(f"  {sid}: n={int(s['n'])} append p95={s['append_p95']:.0f}ms e2e p95={s['e2e_p95']:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
