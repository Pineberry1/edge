"""Run one VLM request per pre-cut evaluation window clip.

The manifest is window-level TSV:

    video_id label window_index start_s end_s cloud_path local_path

Each row is a 40s evaluation sample. We launch one edge process per row,
with a large internal edge window so the whole clip maps to one VLM session.
"""

from __future__ import annotations

import argparse
import json
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
            row = dict(zip(header, line.split("\t")))
            row["window_index"] = int(row["window_index"])
            row["start_s"] = float(row["start_s"])
            row["end_s"] = float(row["end_s"])
            rows.append(row)
    return rows


def launch_edge_once(
    stream_id: str,
    source: str,
    cloud_ws_url: str,
    rho: float,
    alpha: Optional[float],
    internal_window_s: float,
    max_tokens: int,
    prompt: str,
    log_path: Path,
    linger_s: float,
    pace_realtime: bool,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    cmd = [
        PYTHON, "-m", "edge.src.edge_main",
        "--source", source,
        "--cloud-ws-url", cloud_ws_url,
        "--stream-id", stream_id,
        "--rho", str(rho),
        "--window-seconds", str(internal_window_s),
        "--max-tokens", str(max_tokens),
        "--linger-s", str(linger_s),
        "--prompt", prompt,
    ]
    if pace_realtime:
        cmd.append("--pace-realtime")
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    parser.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--internal-window-seconds", type=float, default=9999.0,
                        help="edge session window; keep larger than clip duration")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--prompt", default="Does this 40-second video clip contain any "
                        "abnormal, criminal, or unsafe activity? Answer with only Yes or No.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--linger-s", type=float, default=12.0)
    parser.add_argument("--pace-realtime", action="store_true")
    parser.add_argument("--stop-after-result", action="store_true",
                        help="terminate the edge subprocess once its log contains a VLM result")
    parser.add_argument("--probe-interval", type=float, default=2.0)
    parser.add_argument("--per-window-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    missing = [row for row in rows if not Path(row["local_path"]).exists()]
    if missing:
        for row in missing[:5]:
            print(f"[window-bench] MISSING {row['local_path']}", file=sys.stderr)
        print(f"[window-bench] {len(missing)} missing sources", file=sys.stderr)
        return 2

    print(f"[window-bench] {len(rows)} windows, concurrency={args.concurrency}, "
          f"pace={args.pace_realtime}, linger={args.linger_s}s")

    work_lock = threading.Lock()
    work_idx = [0]
    completed: list[dict] = []
    completed_lock = threading.Lock()
    probes: list[dict] = []
    probes_lock = threading.Lock()
    stop_probe = threading.Event()

    def worker(slot: int) -> None:
        while True:
            with work_lock:
                if work_idx[0] >= len(rows):
                    return
                idx = work_idx[0]
                work_idx[0] += 1
            row = rows[idx]
            sid = f"w-{idx:04d}"
            log_path = out / f"edge-{sid}.log"
            started = time.time()
            print(f"[window-bench][slot{slot}] {sid} {row['label']} {row['video_id']} "
                  f"win={row['window_index']} {row['start_s']:.1f}-{row['end_s']:.1f}s")
            proc = launch_edge_once(
                stream_id=sid,
                source=row["local_path"],
                cloud_ws_url=args.cloud_ws_url,
                rho=args.rho,
                alpha=args.alpha,
                internal_window_s=args.internal_window_seconds,
                max_tokens=args.max_tokens,
                prompt=args.prompt,
                log_path=log_path,
                linger_s=args.linger_s,
                pace_realtime=args.pace_realtime,
            )
            deadline = time.time() + args.per_window_timeout
            rc = None
            if args.stop_after_result:
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
                    time.sleep(0.5)
                if rc is None:
                    print(f"[window-bench][slot{slot}] {sid} TIMEOUT; terminating", file=sys.stderr)
                    proc.terminate()
                    try:
                        proc.wait(5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    rc = -1
            else:
                try:
                    proc.wait(timeout=args.per_window_timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    print(f"[window-bench][slot{slot}] {sid} TIMEOUT; terminating", file=sys.stderr)
                    proc.terminate()
                    try:
                        proc.wait(5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    rc = -1
            try:
                proc._log_fp.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            ended = time.time()
            done = {
                **row,
                "stream_id": sid,
                "source": row["local_path"],
                "started_at": started,
                "ended_at": ended,
                "wall_s": ended - started,
                "returncode": rc,
                "log": str(log_path),
            }
            with completed_lock:
                completed.append(done)

    def probe_loop() -> None:
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
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()
    stop_probe.set()
    probe_thread.join(timeout=5)
    ended_at = time.time()

    _, lat = _http_json("GET", f"{args.intake_admin_base}/stats/latency", timeout=5.0)
    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": ended_at - started_at,
        "n_windows": len(rows),
        "concurrency": args.concurrency,
        "rho": args.rho,
        "alpha": args.alpha,
        "prompt": args.prompt,
        "internal_window_seconds": args.internal_window_seconds,
        "max_tokens": args.max_tokens,
        "streams": sorted(completed, key=lambda r: r["stream_id"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "probes.jsonl").open("w") as pf:
        for rec in probes:
            pf.write(json.dumps(rec, separators=(",", ":")) + "\n")
    summary = {
        "manifest": str(out / "manifest.json"),
        "probes_jsonl": str(out / "probes.jsonl"),
        "n_windows": len(rows),
        "n_probes": len(probes),
        "run_wall_seconds": ended_at - started_at,
        "latency_stats": lat,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[window-bench] DONE {len(rows)} windows in {ended_at-started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
