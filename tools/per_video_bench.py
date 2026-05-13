"""Per-video anomaly-detection bench harness for BAVA.

Iterates a TSV manifest of (video_id, label, source_path, duration_s) and
launches one edge subprocess per video sequentially within K parallel
slots. Each edge process runs **without --loop-source**, plays its source
mp4 to natural EOF, lingers --linger-s seconds for cloud results, then
exits. The harness logs which video occupied which (slot, time-window) so
the aggregator can join (stream_id, window_id) -> (video_id, label).

This is the per-video sibling of edge/tools/bench.py — instead of running
all streams for a fixed wall-clock duration, we drive each video to
completion. Used by edge/tools/per_video_ab.py for 3-config A/B accuracy
evaluation. The result-text logging contract (uplink.py:228) is unchanged,
so the same `[edge-uplink] result window=N text='...'` log lines are
parsed by edge/tools/summarize_anomaly_f1.py.

Usage:
  python -m edge.tools.per_video_bench \
      --manifest edge/data/eval_videos.tsv \
      --rho 1.0 --alpha 1.0 \
      --window-seconds 40 --max-tokens 8 \
      --concurrency 4 --linger-s 30 \
      --prompt "Classify this video clip for surveillance anomaly detection. Output only Yes or No. Yes means the clip shows one of these abnormal categories: arrest, arson, assault, burglary, fighting, road accident, robbery, shooting, shoplifting, stealing, vandalism, explosion, abuse, or any clearly unsafe/criminal behavior. No means normal non-criminal activity without visible danger." \
      --cloud-ws-url ws://127.0.0.1:19100/stream \
      --intake-admin-base http://127.0.0.1:19100 \
      --out edge/data/bench_runs/.../static_full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib import request as urlreq

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
PYTHON = sys.executable


def _http_json(method: str, url: str, payload=None, timeout: float = 3.0):
    body = None if payload is None else json.dumps(payload).encode()
    req = urlreq.Request(url, data=body, method=method,
                         headers={"Content-Type": "application/json"} if body else {})
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:
        return 0, {"error": str(e)}


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            row["duration_s"] = float(row["duration_s"])
            rows.append(row)
    return rows


def launch_edge_once(
    stream_id: str,
    source: str,
    cloud_ws_url: str,
    rho: float,
    alpha: Optional[float],
    window_s: float,
    decision_window_s: float,
    max_tokens: int,
    prompt: str,
    log_path: Path,
    linger_s: float,
    pace_realtime: bool,
    inference_mode: str,
    visual_memory_merge: bool,
) -> subprocess.Popen:
    """Launch a single edge_main subprocess that plays `source` once to EOF.

    Critically does NOT pass --loop-source, so edge_main exits naturally
    after EOF + --linger-s seconds.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    cmd = [
        PYTHON, "-m", "edge.src.edge_main",
        "--source", source,
        "--cloud-ws-url", cloud_ws_url,
        "--stream-id", stream_id,
        "--rho", str(rho),
        "--window-seconds", str(window_s),
        "--decision-window-seconds", str(decision_window_s),
        "--max-tokens", str(max_tokens),
        "--linger-s", str(linger_s),
        "--prompt", prompt,
        "--inference-mode", inference_mode,
    ]
    if pace_realtime:
        cmd.append("--pace-realtime")
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


def log_has_result(log_path: Path) -> bool:
    try:
        return "[edge-uplink] result window=" in log_path.read_text(errors="replace")
    except FileNotFoundError:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path,
                   help="TSV with header video_id\\tlabel\\tcloud_path\\tduration_s; cloud_path is overridden by --source-prefix")
    p.add_argument("--source-prefix", default="",
                   help="if set, replace the directory portion of each cloud_path with this prefix")
    p.add_argument("--local-root", type=Path, default=None,
                   help="local root used to remap cloud_path when it starts with --remote-prefix")
    p.add_argument("--remote-prefix",
                   default="/home/mambauser/tangxuan/ucf_crime_hf",
                   help="remote dataset prefix to replace with --local-root")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--window-seconds", type=float, default=40.0)
    p.add_argument(
        "--decision-window-seconds",
        type=float,
        default=40.0,
        help="edge-side stream_end/detection cadence; window-seconds remains prefill chunk size",
    )
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument(
        "--prompt",
        default=(
            "Classify this video clip for surveillance anomaly detection. "
            "Output only Yes or No. Yes means the clip shows one of these "
            "abnormal categories: arrest, arson, assault, burglary, fighting, "
            "road accident, robbery, shooting, shoplifting, stealing, vandalism, "
            "explosion, abuse, or any clearly unsafe/criminal behavior. No means "
            "normal non-criminal activity without visible danger."
        ),
    )
    p.add_argument("--inference-mode", choices=["online_prefill", "completion"], default="online_prefill")
    p.add_argument("--completion-mode", action="store_true",
                   help="shortcut for --inference-mode completion")
    p.add_argument(
        "--visual-memory-merge",
        action="store_true",
        help="set hello.visual_memory_merge=true for each edge stream",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--linger-s", type=float, default=30.0)
    p.add_argument("--pace-realtime", action="store_true",
                   help="play each source at media time; omit for pure inference throughput")
    p.add_argument("--stop-after-result", action="store_true",
                   help="terminate each edge subprocess as soon as its result is logged")
    p.add_argument("--probe-interval", type=float, default=2.0)
    p.add_argument("--per-video-timeout", type=float, default=600.0,
                   help="hard SIGKILL ceiling per-video (duration + linger margin)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0,
                   help="limit number of videos (0 = all)")
    args = p.parse_args()
    if args.completion_mode:
        args.inference_mode = "completion"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    videos = read_manifest(args.manifest)
    if args.local_root is not None:
        remote_prefix = args.remote_prefix.rstrip("/")
        for v in videos:
            cloud_path = str(v["cloud_path"])
            if cloud_path.startswith(remote_prefix + "/"):
                rel = cloud_path[len(remote_prefix) + 1:]
                v["source"] = str(args.local_root / rel)
            else:
                v["source"] = cloud_path
    elif args.source_prefix:
        for v in videos:
            v["source"] = str(Path(args.source_prefix) / Path(v["cloud_path"]).name) \
                if Path(v["cloud_path"]).parent != Path(args.source_prefix) else \
                str(Path(args.source_prefix) / Path(v["cloud_path"]).name)
    else:
        for v in videos:
            v["source"] = v["cloud_path"]
    if args.limit:
        videos = videos[: args.limit]
    n = len(videos)
    print(f"[per-video] {n} videos, concurrency={args.concurrency}, "
          f"prefill_window={args.window_seconds}s, "
          f"decision_window={args.decision_window_seconds}s, linger={args.linger_s}s")

    # Validate sources exist locally
    missing = [v for v in videos if not Path(v["source"]).exists()]
    if missing:
        for v in missing[:5]:
            print(f"[per-video] MISSING source: {v['source']}", file=sys.stderr)
        print(f"[per-video] {len(missing)} sources missing on edge; abort", file=sys.stderr)
        return 2

    # Worker queue: parallel slots take videos one by one
    work_lock = threading.Lock()
    work_idx = [0]
    completed: list[dict] = []
    completed_lock = threading.Lock()
    probes: list[dict] = []
    probes_lock = threading.Lock()
    stop_probe = threading.Event()

    def worker(slot: int):
        while True:
            with work_lock:
                if work_idx[0] >= n:
                    return
                vidx = work_idx[0]
                work_idx[0] += 1
            v = videos[vidx]
            sid = f"v-{vidx:03d}"
            log_path = out / f"edge-{sid}.log"
            window_count = max(1, math.ceil(v["duration_s"] / args.decision_window_seconds))
            started = time.time()
            print(f"[per-video][slot{slot}] {sid} {v['label']} {v['video_id']} "
                  f"dur={v['duration_s']:.1f}s decision_windows={window_count} -> launching")
            proc = launch_edge_once(
                stream_id=sid,
                source=v["source"],
                cloud_ws_url=args.cloud_ws_url,
                rho=args.rho,
                alpha=args.alpha,
                window_s=args.window_seconds,
                decision_window_s=args.decision_window_seconds,
                max_tokens=args.max_tokens,
                prompt=args.prompt,
                log_path=log_path,
                linger_s=args.linger_s,
                pace_realtime=args.pace_realtime,
                inference_mode=args.inference_mode,
                visual_memory_merge=bool(args.visual_memory_merge),
            )
            timeout = v["duration_s"] + args.linger_s + 60.0
            timeout = min(timeout, args.per_video_timeout)
            if args.stop_after_result:
                deadline = time.time() + timeout
                rc = None
                while time.time() < deadline:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if log_has_result(log_path):
                        proc.terminate()
                        try:
                            proc.wait(5.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        rc = proc.returncode
                        break
                    time.sleep(0.2)
                if rc is None:
                    print(f"[per-video][slot{slot}] {sid} TIMEOUT after {timeout:.0f}s; SIGTERM", file=sys.stderr)
                    proc.terminate()
                    try:
                        proc.wait(8.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    rc = -1
            else:
                try:
                    proc.wait(timeout=timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    print(f"[per-video][slot{slot}] {sid} TIMEOUT after {timeout:.0f}s; SIGTERM", file=sys.stderr)
                    proc.terminate()
                    try:
                        proc.wait(8.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    rc = -1
            try:
                proc._log_fp.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            ended = time.time()
            with completed_lock:
                completed.append({
                    "stream_id": sid,
                    "video_id": v["video_id"],
                    "parent_video_id": v.get("parent_video_id", v["video_id"]),
                    "label": v["label"],
                    "source": v["source"],
                    "duration_s": v["duration_s"],
                    "window_seconds": args.window_seconds,
                    "decision_window_seconds": args.decision_window_seconds,
                    "window_count": window_count,
                    "started_at": started,
                    "ended_at": ended,
                    "wall_s": ended - started,
                    "returncode": rc,
                    "log": str(log_path),
                })
            print(f"[per-video][slot{slot}] {sid} done rc={rc} wall={ended-started:.1f}s")

    def probe_loop():
        t0 = time.time()
        while not stop_probe.is_set():
            t = time.time()
            _, h = _http_json("GET", f"{args.intake_admin_base}/healthz")
            with probes_lock:
                probes.append({
                    "t": t,
                    "elapsed": t - t0,
                    **{k: h.get(k) for k in
                       ("vllm_snapshot", "vllm_engines", "controller_streams", "active_streams")},
                })
            stop_probe.wait(args.probe_interval)

    started_at = time.time()
    probe_thread = threading.Thread(target=probe_loop, daemon=True)
    probe_thread.start()

    workers = [threading.Thread(target=worker, args=(i,), daemon=False)
               for i in range(args.concurrency)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    stop_probe.set()
    probe_thread.join(timeout=5)
    ended_at = time.time()

    # Pull final latency from intake (best-effort)
    _, lat = _http_json("GET", f"{args.intake_admin_base}/stats/latency", timeout=5.0)

    # Manifest and probes
    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": ended_at - started_at,
        "n_videos": n,
        "concurrency": args.concurrency,
        "window_seconds": args.window_seconds,
        "decision_window_seconds": args.decision_window_seconds,
        "max_tokens": args.max_tokens,
        "rho": args.rho,
        "alpha": args.alpha,
        "prompt": args.prompt,
        "inference_mode": args.inference_mode,
        "visual_memory_merge": bool(args.visual_memory_merge),
        "intake_admin_base": args.intake_admin_base,
        "cloud_ws_url": args.cloud_ws_url,
        "linger_s": args.linger_s,
        "pace_realtime": args.pace_realtime,
        "stop_after_result": args.stop_after_result,
        "streams": sorted(completed, key=lambda r: r["started_at"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "probes.jsonl").open("w") as pf:
        for rec in probes:
            pf.write(json.dumps(rec, separators=(",", ":")) + "\n")
    summary = {
        "manifest": str(out / "manifest.json"),
        "probes_jsonl": str(out / "probes.jsonl"),
        "n_videos": n,
        "n_probes": len(probes),
        "run_wall_seconds": ended_at - started_at,
        "latency_stats": lat,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[per-video] DONE: {n} videos in {ended_at-started_at:.1f}s "
          f"({n/(ended_at-started_at)*60:.1f} videos/min)")
    print(f"[per-video] manifest: {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
