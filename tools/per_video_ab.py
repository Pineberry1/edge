"""3-config A/B harness for per-video UCF-Crime anomaly detection.

For each config (static_full, static_half, bava_dynamic):

  1. SIGTERM any running intake.server on cloud, wait until gone.
  2. Start a fresh intake with the right BAVA env (controller on/off,
     frames-per-window etc).
  3. Wait until intake /healthz shows all expected vLLM engines healthy.
  4. POST /admin/purge_sessions to clear any leftover vLLM sessions.
  5. Run `python -m edge.tools.per_video_bench …` against the manifest.
  6. Pull cloud-side controller.jsonl / anchors.jsonl / intake.log.

Output goes under `--out/<config_name>/`.

Reuses helpers from edge/tools/ab_bench.py (start_intake / stop_intake /
wait_intake_engines_ready / purge_sessions / _fetch_remote_text /
_parse_api_bases / _ssh / _shell_env). Reuses the 3 corresponding rows of
its CONFIGS list.

Usage:
  python -m edge.tools.per_video_ab \
      --manifest edge/data/eval_videos.tsv \
      --concurrency 4 --window-seconds 4 --decision-window-seconds 40 \
      --frames-per-window 80 --max-tokens 8 \
      --linger-s 30 \
      --vllm-api-base-list http://127.0.0.1:8011,http://127.0.0.1:8012,http://127.0.0.1:8013,http://127.0.0.1:8014 \
      --intake-admin-base http://127.0.0.1:19100 \
      --cloud-ws-url ws://127.0.0.1:19100/stream \
      --out edge/data/bench_runs/anomaly_f1_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from edge.tools.ab_bench import (
    CONFIGS as AB_CONFIGS,
    _fetch_remote_text,
    _parse_api_bases,
    _shell_env,
    _ssh,
    purge_sessions,
    start_intake,
    stop_intake,
    wait_intake_engines_ready,
    wait_intake_ready,
)

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent

CONFIG_NAMES = ["static_full", "static_half", "bava_dynamic"]


def get_configs() -> list[dict]:
    by_name = {c["name"]: c for c in AB_CONFIGS}
    return [by_name[n] for n in CONFIG_NAMES]


def run_per_video_bench(
    manifest_path: Path,
    cfg: dict,
    args: argparse.Namespace,
    out_dir: Path,
    visual_memory_merge: bool = False,
) -> dict:
    cmd = [
        sys.executable, "-m", "edge.tools.per_video_bench",
        "--manifest", str(manifest_path),
        "--cloud-ws-url", args.cloud_ws_url,
        "--intake-admin-base", args.intake_admin_base,
        "--rho", str(cfg["rho"]),
        "--window-seconds", str(args.window_seconds),
        "--decision-window-seconds", str(args.decision_window_seconds),
        "--max-tokens", str(args.max_tokens),
        "--prompt", args.prompt,
        "--concurrency", str(args.concurrency),
        "--linger-s", str(args.linger_s),
        "--per-video-timeout", str(args.per_video_timeout),
        "--out", str(out_dir),
    ]
    if not cfg["controller_enabled"]:
        cmd += ["--alpha", str(cfg["alpha"])]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.pace_realtime:
        cmd.append("--pace-realtime")
    if args.stop_after_result:
        cmd.append("--stop-after-result")
    if visual_memory_merge:
        cmd.append("--visual-memory-merge")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    print(f"[ab] -> {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    sumf = out_dir / "summary.json"
    if sumf.exists():
        return json.loads(sumf.read_text())
    return {"error": f"per_video_bench rc={proc.returncode}, no summary.json"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--cloud-host", default="210.45.123.163")
    p.add_argument("--cloud-ssh-port", type=int, default=2222)
    p.add_argument("--cloud-ssh-user", default="mambauser")
    p.add_argument("--cloud-ssh-key", default=os.path.expanduser("~/.ssh/jupyterhub.pem"))
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--vllm-api-base", default="http://127.0.0.1:8011")
    p.add_argument("--vllm-api-base-list", default="")
    p.add_argument("--window-seconds", type=float, default=4.0,
                   help="small online-prefill chunk size")
    p.add_argument("--decision-window-seconds", type=float, default=40.0,
                   help="edge-side stream_end/detection cadence")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--frames-per-window", type=int, default=80,
                   help="BAVA_MAX_FRAMES_PER_WINDOW (cloud subsamples to this)")
    p.add_argument("--stream-concurrency", type=int, default=4,
                   help="BAVA_STREAM_CONCURRENCY")
    p.add_argument("--max-queued-windows", type=int, default=8,
                   help="BAVA_MAX_QUEUED_WINDOWS")
    p.add_argument("--concurrency", type=int, default=4,
                   help="how many videos to process in parallel on the edge")
    p.add_argument("--linger-s", type=float, default=30.0)
    p.add_argument("--pace-realtime", action="store_true",
                   help="play each source at media time; omit for pure inference throughput")
    p.add_argument(
        "--visual-memory-merge",
        action="store_true",
        help="set hello.visual_memory_merge=true for each video stream",
    )
    p.add_argument(
        "--visual-memory-num-frames",
        type=int,
        default=int(os.environ.get("BAVA_VISUAL_MEMORY_NUM_FRAMES", "8")),
    )
    p.add_argument(
        "--visual-memory-tokens-per-frame",
        type=int,
        default=int(os.environ.get("BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME", "32")),
    )
    p.add_argument(
        "--visual-memory-text-prefix",
        default=os.environ.get("BAVA_VISUAL_MEMORY_TEXT_PREFIX", ""),
    )
    p.add_argument(
        "--visual-memory-warm-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="warm prefix cache after exporting visual memory (default: intake env)",
    )
    p.add_argument("--stop-after-result", action="store_true",
                   help="terminate edge subprocesses after the first VLM result")
    p.add_argument("--per-video-timeout", type=float, default=600.0)
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
    p.add_argument("--limit", type=int, default=0,
                   help="limit number of videos (0 = all)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--configs", nargs="+", default=None,
                   help="subset of configs by name (default all 3)")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    vllm_api_bases = _parse_api_bases(args.vllm_api_base, args.vllm_api_base_list)

    cfgs = get_configs()
    if args.configs:
        wanted = set(args.configs)
        cfgs = [c for c in cfgs if c["name"] in wanted]

    results: dict[str, Any] = {
        "started_at": time.time(),
        "manifest": str(args.manifest),
        "vllm_api_bases": vllm_api_bases,
        "window_seconds": args.window_seconds,
        "decision_window_seconds": args.decision_window_seconds,
        "frames_per_window": args.frames_per_window,
        "concurrency": args.concurrency,
        "linger_s": args.linger_s,
        "pace_realtime": args.pace_realtime,
        "visual_memory_merge": bool(args.visual_memory_merge),
        "visual_memory": {
            "num_frames": int(args.visual_memory_num_frames),
            "tokens_per_frame": int(args.visual_memory_tokens_per_frame),
            "text_prefix": args.visual_memory_text_prefix,
            "warm_prefix_cache": args.visual_memory_warm_prefix_cache,
        },
        "stop_after_result": args.stop_after_result,
        "prompt": args.prompt,
        "configs_run": [c["name"] for c in cfgs],
        "results": {},
    }

    for cfg in cfgs:
        name = cfg["name"]
        cfg_out = out / name
        cfg_out.mkdir(parents=True, exist_ok=True)
        remote_controller_log = f"/tmp/bava_controller_{name}.jsonl"
        remote_anchor_log = f"/tmp/bava_anchors_{name}.jsonl"
        remote_intake_log = f"/home/mambauser/tangxuan/online_vllm/intake/logs/intake_{name}.log"

        print(f"\n================ A/B config: {name} ================")
        print(f"  {cfg['description']}")

        print("[ab] stopping old intake…")
        stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)
        _ssh(
            args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user,
            "rm -f " + " ".join(
                shlex.quote(p) for p in (remote_controller_log, remote_anchor_log, remote_intake_log)
            ),
        )

        print(f"[ab] starting intake (controller={cfg['controller_enabled']}, "
              f"vllm={','.join(vllm_api_bases)}, frames_per_window={args.frames_per_window})…")
        start_intake(
            args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user,
            args.vllm_api_base,
            vllm_api_bases=vllm_api_bases,
            controller_enabled=cfg["controller_enabled"],
            extra_env={
                "BAVA_CONTROLLER_LOG": remote_controller_log,
                "BAVA_ANCHOR_LOG": remote_anchor_log,
                "BAVA_MAX_FRAMES_PER_WINDOW": str(args.frames_per_window),
                "BAVA_MAX_QUEUED_WINDOWS": str(args.max_queued_windows),
                "BAVA_STREAM_CONCURRENCY": str(args.stream_concurrency),
                "BAVA_VISUAL_MEMORY_NUM_FRAMES": str(args.visual_memory_num_frames),
                "BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME": str(
                    args.visual_memory_tokens_per_frame
                ),
                "BAVA_VISUAL_MEMORY_TEXT_PREFIX": args.visual_memory_text_prefix,
                **(
                    {
                        "BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE": (
                            "1" if args.visual_memory_warm_prefix_cache else "0"
                        )
                    }
                    if args.visual_memory_warm_prefix_cache is not None
                    else {}
                ),
            },
            log_suffix=f"_{name}",
        )

        if not wait_intake_ready(args.intake_admin_base, timeout_s=30.0):
            print(f"[ab][{name}] intake not ready, skip", file=sys.stderr)
            results["results"][name] = {"error": "intake not ready"}
            continue
        if not wait_intake_engines_ready(args.intake_admin_base, vllm_api_bases, timeout_s=60.0):
            print(f"[ab][{name}] vLLM engines not healthy, skip", file=sys.stderr)
            results["results"][name] = {"error": "engines not healthy"}
            continue

        aborted = purge_sessions(args.intake_admin_base)
        print(f"[ab] post-start purge: aborted={aborted}")

        summary = run_per_video_bench(
            args.manifest,
            cfg,
            args,
            cfg_out,
            visual_memory_merge=bool(args.visual_memory_merge),
        )

        for remote, local in (
            (remote_controller_log, cfg_out / "controller.jsonl"),
            (remote_anchor_log, cfg_out / "anchors.jsonl"),
            (remote_intake_log, cfg_out / "intake.log"),
        ):
            _fetch_remote_text(
                args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user,
                remote, local,
            )

        results["results"][name] = {
            "config": cfg,
            "summary_path": str(cfg_out / "summary.json"),
            "manifest_path": str(cfg_out / "manifest.json"),
            "controller_log_path": str(cfg_out / "controller.jsonl"),
            "anchor_log_path": str(cfg_out / "anchors.jsonl"),
            "intake_log_path": str(cfg_out / "intake.log"),
            "n_videos": summary.get("n_videos"),
            "run_wall_seconds": summary.get("run_wall_seconds"),
            "latency_stats": summary.get("latency_stats"),
        }

    print("\n[ab] stopping final intake…")
    stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)

    (out / "ab_summary.json").write_text(json.dumps(results, indent=2, default=str))

    # Brief comparison table
    print("\n================ A/B per-video summary ================")
    hdr = f"{'config':<15} {'n_videos':>10} {'wall_s':>10} {'append_p95':>12} {'e2e_p95':>12}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results["results"].items():
        if "error" in r:
            print(f"{name:<15} ERROR: {r['error']}")
            continue
        lat = (r.get("latency_stats") or {}).get("__all__") or {}
        ap = lat.get("append_p95")
        ep = lat.get("e2e_p95")
        print(f"{name:<15} "
              f"{int(r.get('n_videos') or 0):>10} "
              f"{(r.get('run_wall_seconds') or 0):>10.1f} "
              f"{(f'{ap:.0f}ms' if isinstance(ap,(int,float)) else '-'):>12} "
              f"{(f'{ep:.0f}ms' if isinstance(ep,(int,float)) else '-'):>12}")
    print(f"\nfull results: {out / 'ab_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
