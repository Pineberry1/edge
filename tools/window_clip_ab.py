"""A/B harness for paired 40s sliding-window anomaly evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from edge.tools.ab_bench import (
    CONFIGS,
    _fetch_remote_text,
    _parse_api_bases,
    _ssh,
    purge_sessions,
    start_intake,
    stop_intake,
    wait_intake_engines_ready,
    wait_intake_ready,
)


REPO = Path(__file__).resolve().parent.parent.parent


def run_window_bench(args: argparse.Namespace, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "edge.tools.window_clip_bench",
        "--manifest",
        str(args.manifest),
        "--cloud-ws-url",
        args.cloud_ws_url,
        "--intake-admin-base",
        args.intake_admin_base,
        "--rho",
        str(cfg["rho"]),
        "--internal-window-seconds",
        str(args.internal_window_seconds),
        "--max-tokens",
        str(args.max_tokens),
        "--prompt",
        args.prompt,
        "--concurrency",
        str(args.concurrency),
        "--linger-s",
        str(args.linger_s),
        "--per-window-timeout",
        str(args.per_window_timeout),
        "--out",
        str(out_dir),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.pace_realtime:
        cmd.append("--pace-realtime")
    if args.stop_after_result:
        cmd.append("--stop-after-result")
    if not cfg["controller_enabled"]:
        cmd += ["--alpha", str(cfg["alpha"])]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    print(f"[window-ab] -> {' '.join(shlex.quote(part) for part in cmd)}")
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True)
    summary_path = out_dir / "summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cloud-host", default="210.45.123.163")
    parser.add_argument("--cloud-ssh-port", type=int, default=2222)
    parser.add_argument("--cloud-ssh-user", default="mambauser")
    parser.add_argument("--cloud-ssh-key", default=os.path.expanduser("~/.ssh/jupyterhub.pem"))
    parser.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    parser.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    parser.add_argument("--vllm-api-base", default="http://127.0.0.1:8011")
    parser.add_argument("--vllm-api-base-list", default="")
    parser.add_argument("--configs", nargs="+", default=["static_full", "static_rho005", "bava_dynamic"])
    parser.add_argument("--frames-per-window", type=int, default=4)
    parser.add_argument("--max-queued-windows", type=int, default=4)
    parser.add_argument("--stream-concurrency", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--internal-window-seconds", type=float, default=9999.0)
    parser.add_argument("--linger-s", type=float, default=12.0)
    parser.add_argument("--per-window-timeout", type=float, default=180.0)
    parser.add_argument("--prompt", default="Does this 40-second video clip contain any abnormal, "
                        "criminal, or unsafe activity? Answer with only Yes or No.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pace-realtime", action="store_true")
    parser.add_argument("--stop-after-result", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    selected = set(args.configs)
    cfgs = [cfg for cfg in CONFIGS if cfg["name"] in selected]
    missing = selected.difference({cfg["name"] for cfg in cfgs})
    if missing:
        raise SystemExit(f"unknown configs: {', '.join(sorted(missing))}")
    args.out.mkdir(parents=True, exist_ok=True)
    vllm_bases = _parse_api_bases(args.vllm_api_base, args.vllm_api_base_list)

    combined: dict[str, Any] = {
        "started_at": time.time(),
        "manifest": str(args.manifest),
        "configs_run": [cfg["name"] for cfg in cfgs],
        "vllm_api_bases": vllm_bases,
        "frames_per_window": args.frames_per_window,
        "prompt": args.prompt,
        "results": {},
    }

    try:
        for cfg in cfgs:
            name = cfg["name"]
            cfg_out = args.out / name
            cfg_out.mkdir(parents=True, exist_ok=True)
            remote_controller_log = f"/tmp/bava_controller_window_{name}.jsonl"
            remote_anchor_log = f"/tmp/bava_anchors_window_{name}.jsonl"
            remote_intake_log = f"/home/mambauser/tangxuan/online_vllm/intake/logs/intake_window_{name}.log"
            print(f"\n================ window A/B config: {name} ================")
            stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)
            _ssh(
                args.cloud_host,
                args.cloud_ssh_port,
                args.cloud_ssh_key,
                args.cloud_ssh_user,
                "rm -f "
                + " ".join(
                    shlex.quote(path)
                    for path in (remote_controller_log, remote_anchor_log, remote_intake_log)
                ),
            )
            start_intake(
                args.cloud_host,
                args.cloud_ssh_port,
                args.cloud_ssh_key,
                args.cloud_ssh_user,
                args.vllm_api_base,
                controller_enabled=bool(cfg["controller_enabled"]),
                vllm_api_bases=vllm_bases,
                extra_env={
                    "BAVA_CONTROLLER_LOG": remote_controller_log,
                    "BAVA_ANCHOR_LOG": remote_anchor_log,
                    "BAVA_MAX_FRAMES_PER_WINDOW": str(args.frames_per_window),
                    "BAVA_MAX_QUEUED_WINDOWS": str(args.max_queued_windows),
                    "BAVA_STREAM_CONCURRENCY": str(args.stream_concurrency),
                },
                log_suffix=f"_window_{name}",
            )
            if not wait_intake_ready(args.intake_admin_base, timeout_s=20.0):
                combined["results"][name] = {"error": "intake not ready"}
                continue
            if not wait_intake_engines_ready(args.intake_admin_base, vllm_bases, timeout_s=45.0):
                combined["results"][name] = {"error": "engines not healthy"}
                continue
            print(f"[window-ab] post-start purge: aborted={purge_sessions(args.intake_admin_base)}")
            summary = run_window_bench(args, cfg, cfg_out)
            _fetch_remote_text(
                args.cloud_host,
                args.cloud_ssh_port,
                args.cloud_ssh_key,
                args.cloud_ssh_user,
                remote_controller_log,
                cfg_out / "controller.jsonl",
            )
            _fetch_remote_text(
                args.cloud_host,
                args.cloud_ssh_port,
                args.cloud_ssh_key,
                args.cloud_ssh_user,
                remote_anchor_log,
                cfg_out / "anchors.jsonl",
            )
            _fetch_remote_text(
                args.cloud_host,
                args.cloud_ssh_port,
                args.cloud_ssh_key,
                args.cloud_ssh_user,
                remote_intake_log,
                cfg_out / "intake.log",
            )
            combined["results"][name] = {
                "config": cfg,
                "summary_path": str(cfg_out / "summary.json"),
                "manifest_path": str(cfg_out / "manifest.json"),
                "latency_stats": summary.get("latency_stats"),
                "n_windows": summary.get("n_windows"),
                "run_wall_seconds": summary.get("run_wall_seconds"),
            }
    finally:
        print("\n[window-ab] stopping final intake...")
        stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)

    combined["ended_at"] = time.time()
    combined["wall_s"] = combined["ended_at"] - combined["started_at"]
    (args.out / "window_ab_summary.json").write_text(json.dumps(combined, indent=2, default=str))
    print(f"[window-ab] results: {args.out / 'window_ab_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
