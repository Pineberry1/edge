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
    cloud_ws_url: str,
    rho: float,
    window_s: float,
    max_tokens: int,
    prompt: str,
    log_path: Path,
    alpha: Optional[float] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    cmd = [
        PYTHON, "-m", "edge.src.edge_main",
        "--source", source,
        "--cloud-ws-url", cloud_ws_url,
        "--stream-id", stream_id,
        "--rho", str(rho),
        "--window-seconds", str(window_s),
        "--max-tokens", str(max_tokens),
        "--linger-s", "15",
        "--pace-realtime",
        "--loop-source",
        "--prompt", prompt,
    ]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    log_fp = log_path.open("w")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env,
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    proc._log_fp = log_fp  # type: ignore[attr-defined]
    return proc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--duration", type=float, default=60.0, help="wall-clock seconds to run edges before stopping")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--sources", nargs="+", required=True,
                   help="glob(s) of mp4 sources; round-robins when n > len(sources)")
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=None, help="initial α (static baselines only); omit for dynamic controller")
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--prompt", default="In one short sentence, describe what is happening in this clip.")
    p.add_argument("--probe-interval", type=float, default=0.5)
    p.add_argument("--out", default=f"/tmp/bava_bench_{int(time.time())}")
    args = p.parse_args()

    sources: List[str] = []
    for pat in args.sources:
        sources += sorted(glob.glob(pat))
    if not sources:
        print("no sources matched", file=sys.stderr)
        return 2
    print(f"sources: {len(sources)} ({sources[0]} ...)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probes_path = out / "probes.jsonl"
    manifest_path = out / "manifest.json"
    summary_path = out / "summary.json"

    # Launch N edges
    procs: List[subprocess.Popen] = []
    manifest = {
        "started_at": time.time(),
        "n": args.n,
        "duration_s": args.duration,
        "rho_initial": args.rho,
        "window_seconds": args.window_seconds,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "intake_admin_base": args.intake_admin_base,
        "cloud_ws_url": args.cloud_ws_url,
        "sources_pool": sources,
        "streams": [],
    }

    for i in range(args.n):
        sid = f"bench-{i:02d}"
        src = sources[i % len(sources)]
        log = out / f"edge-{sid}.log"
        proc = launch_edge(
            stream_id=sid, source=src,
            cloud_ws_url=args.cloud_ws_url, rho=args.rho,
            window_s=args.window_seconds, max_tokens=args.max_tokens,
            prompt=args.prompt, log_path=log,
            alpha=args.alpha,
        )
        procs.append(proc)
        manifest["streams"].append({"stream_id": sid, "source": src, "pid": proc.pid, "log": str(log)})
        time.sleep(0.1)  # stagger a little
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
                        for k in ("vllm_snapshot", "vllm_engines", "controller_streams", "active_streams")
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

    # Stop edges: SIGTERM first (lets uplink flush), then SIGKILL after 8s.
    # We use SIGTERM because SIGINT inside pace-realtime's time.sleep wakes up
    # as KeyboardInterrupt, but the looped source + auto-reconnect can keep
    # the process alive longer than expected.
    print("[bench] sending SIGTERM to edges")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.time() + 8
    for p in procs:
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
