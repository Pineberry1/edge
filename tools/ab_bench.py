"""Three-config A/B harness for BAVA vs static baselines.

For each configuration:

  1. Issue POST /admin/purge_sessions on the running intake so leftover vLLM
     sessions from the previous config don't inflate Q_waiting.
  2. SIGTERM the remote intake; on shutdown it runs the same purge path plus
     its bookkeeping cleanup — belt and suspenders.
  3. Start a fresh intake with env-vars that reflect the config (controller
     on/off, initial ρ/α bounds).
  4. Run `edge.tools.bench` for `--per-config-duration` seconds with the
     config's initial ρ and α.
  5. Collect summary.json into a single A/B result.

The goal is comparing **code improvement** (BAVA dynamic vs static full /
static half), not pushing N to some absolute maximum. Use N=4 or N=8 on the
single vLLM instance we have running. If the user wants per-engine sharding
later, that's a follow-up change to intake (round-robin VLLM_API_BASE).

Usage:
  python -m edge.tools.ab_bench \
      --n 4 --per-config-duration 60 \
      --cloud-host 210.45.123.163 --cloud-ssh-key ~/.ssh/jupyterhub.pem \
      --cloud-ssh-port 2222 --cloud-ssh-user mambauser \
      --intake-admin-base http://127.0.0.1:19100 \
      --cloud-ws-url ws://127.0.0.1:19100/stream \
      --sources 'edge/data/ucf/*.mp4' \
      --vllm-api-base http://127.0.0.1:8003 \
      --vllm-api-base-list 'http://127.0.0.1:8011,http://127.0.0.1:8012' \
      --out /tmp/ab_n4_$(date +%s)

The cloud-side intake is launched via SSH. Requires that vLLM is already
running and the key auth works.
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
from typing import Any, Dict, List, Optional
from urllib import request as urlreq

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent  # /home/admin123/tangxuan


CONFIGS = [
    {
        "name": "static_full",
        "description": "ρ=1.0 α=1.0 static (no compression baseline)",
        "controller_enabled": False,
        "rho": 1.0,
        "alpha": 1.0,
    },
    {
        "name": "static_half",
        "description": "ρ=0.5 α=0.5 static (double-axis static baseline)",
        "controller_enabled": False,
        "rho": 0.5,
        "alpha": 0.5,
    },
    {
        "name": "static_rho075",
        "description": "ρ=0.75 α=1.0 static (rho-only sweep)",
        "controller_enabled": False,
        "rho": 0.75,
        "alpha": 1.0,
    },
    {
        "name": "static_rho050",
        "description": "ρ=0.5 α=1.0 static (rho-only sweep)",
        "controller_enabled": False,
        "rho": 0.5,
        "alpha": 1.0,
    },
    {
        "name": "static_rho025",
        "description": "ρ=0.25 α=1.0 static (rho-only sweep)",
        "controller_enabled": False,
        "rho": 0.25,
        "alpha": 1.0,
    },
    {
        "name": "static_rho005",
        "description": "ρ=0.05 α=1.0 static (rho-only emergency-brake baseline)",
        "controller_enabled": False,
        "rho": 0.05,
        "alpha": 1.0,
    },
    {
        "name": "bava_dynamic",
        "description": "ρ=0.5 initial, controller ON, climb-back + per-stream weighting",
        "controller_enabled": True,
        "rho": 0.5,
        "alpha": 1.0,
    },
]


def _http_json(method: str, url: str, payload=None, timeout: float = 5.0):
    body = None if payload is None else json.dumps(payload).encode()
    req = urlreq.Request(url, data=body, method=method,
                         headers={"Content-Type": "application/json"} if body else {})
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:
        return 0, {"error": str(e)}


def _ssh(cloud_host: str, ssh_port: int, ssh_key: str, user: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    # -n: read stdin from /dev/null so the ssh client doesn't hold the remote
    # command's stdin open. Crucial when the remote command backgrounds itself
    # (nohup ... & disown) — without -n ssh hangs waiting for EOF on stdin.
    full = [
        "ssh", "-n", "-p", str(ssh_port), "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{user}@{cloud_host}", cmd,
    ]
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


def _fetch_remote_text(
    cloud_host: str,
    ssh_port: int,
    ssh_key: str,
    user: str,
    remote_path: str,
    local_path: Path,
    timeout: int = 60,
) -> bool:
    cmd = f"test -f {shlex.quote(remote_path)} && cat {shlex.quote(remote_path)}"
    r = _ssh(cloud_host, ssh_port, ssh_key, user, cmd, timeout=timeout)
    if r.returncode != 0:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(r.stdout)
    return True


def _shell_env(name: str, value: str) -> str:
    return f"{name}={shlex.quote(str(value))}"


def _parse_api_bases(api_base: str, api_base_list: Optional[str]) -> List[str]:
    if api_base_list:
        out = [item.strip().rstrip("/") for item in api_base_list.split(",") if item.strip()]
        if out:
            return out
    return [api_base.strip().rstrip("/")]


def stop_intake(cloud_host: str, ssh_port: int, ssh_key: str, user: str) -> None:
    """SIGTERM any running intake.server on the cloud, wait up to 10s."""
    find_pids = r"""ps -eo pid=,args= | awk '/python -m intake.server/ && $0 !~ /awk/ {print $1}'"""
    _ssh(
        cloud_host,
        ssh_port,
        ssh_key,
        user,
        f"for p in $({find_pids}); do kill -TERM $p; done",
    )
    for _ in range(20):
        r = _ssh(cloud_host, ssh_port, ssh_key, user, find_pids, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return
        time.sleep(0.5)
    # last resort
    _ssh(
        cloud_host,
        ssh_port,
        ssh_key,
        user,
        f"for p in $({find_pids}); do kill -9 $p; done",
    )


def start_intake(
    cloud_host: str,
    ssh_port: int,
    ssh_key: str,
    user: str,
    vllm_api_base: str,
    controller_enabled: bool,
    vllm_api_bases: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    log_suffix: str = "",
) -> None:
    resolved_bases = [base.rstrip("/") for base in (vllm_api_bases or [vllm_api_base]) if base.strip()]
    env_parts = [
        _shell_env("VLLM_API_BASE", resolved_bases[0]),
        _shell_env("BAVA_MAX_FRAMES_PER_WINDOW", "4"),
        _shell_env("BAVA_STREAM_CONCURRENCY", "2"),
        _shell_env("BAVA_MAX_QUEUED_WINDOWS", "4"),
        _shell_env("BAVA_CONTROLLER_ENABLED", "1" if controller_enabled else "0"),
    ]
    if len(resolved_bases) > 1:
        env_parts.append(_shell_env("VLLM_API_BASE_LIST", ",".join(resolved_bases)))
    if extra_env:
        env_parts += [_shell_env(k, v) for k, v in extra_env.items()]
    log_path = f"/home/mambauser/tangxuan/online_vllm/intake/logs/intake{log_suffix}.log"
    # Use a subshell + `setsid` to fully detach. `& disown` inside a
    # non-interactive `bash -c` is unreliable (job control disabled), and
    # even `nohup` on its own leaves an ssh stdin/stdout file descriptor
    # that keeps the ssh session alive.
    cmd = (
        "cd /home/mambauser/tangxuan/online_vllm && "
        "( setsid env " + " ".join(env_parts) + " "
        f"bash intake/start_intake.sh > {log_path} 2>&1 </dev/null & )"
    )
    _ssh(cloud_host, ssh_port, ssh_key, user, cmd)


def wait_intake_ready(admin_base: str, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, _ = _http_json("GET", f"{admin_base}/healthz", timeout=2)
        if status == 200:
            return True
        time.sleep(0.5)
    return False


def wait_intake_engines_ready(admin_base: str, expected_api_bases: List[str], timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    expected = [base.rstrip("/") for base in expected_api_bases]
    while time.time() < deadline:
        status, data = _http_json("GET", f"{admin_base}/healthz", timeout=3)
        if status == 200:
            engines = data.get("vllm_engines") or []
            by_base = {str(row.get("api_base") or "").rstrip("/"): row for row in engines}
            if len(by_base) == len(expected) and all(by_base.get(base, {}).get("ok") for base in expected):
                return True
        time.sleep(0.5)
    return False


def purge_sessions(admin_base: str) -> int:
    status, data = _http_json("POST", f"{admin_base}/admin/purge_sessions")
    return int(data.get("aborted") or 0) if status == 200 else -1


def run_bench(
    n: int,
    duration: float,
    cloud_ws_url: str,
    admin_base: str,
    sources_globs: List[str],
    rho: float,
    alpha: Optional[float],
    window_seconds: float,
    max_tokens: int,
    prompt: str,
    out_dir: Path,
) -> Dict[str, Any]:
    cmd = [
        sys.executable, "-m", "edge.tools.bench",
        "--n", str(n),
        "--duration", str(duration),
        "--cloud-ws-url", cloud_ws_url,
        "--intake-admin-base", admin_base,
        "--rho", str(rho),
        "--window-seconds", str(window_seconds),
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
        "--probe-interval", "0.5",
        "--out", str(out_dir),
        "--sources", *sources_globs,
    ]
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    print(f"[ab] → {' '.join(shlex.quote(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=duration + 120)
    print(r.stdout.decode(errors="replace")[-2000:])
    sumf = out_dir / "summary.json"
    if sumf.exists():
        return json.loads(sumf.read_text())
    return {"error": "no summary.json"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--per-config-duration", type=float, default=60.0)
    p.add_argument("--cloud-host", default="210.45.123.163")
    p.add_argument("--cloud-ssh-port", type=int, default=2222)
    p.add_argument("--cloud-ssh-user", default="mambauser")
    p.add_argument("--cloud-ssh-key", default=os.path.expanduser("~/.ssh/jupyterhub.pem"))
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--vllm-api-base", default="http://127.0.0.1:8003")
    p.add_argument("--vllm-api-base-list", default="")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument(
        "--prompt",
        default="In one short sentence, describe what is happening in this clip.",
        help="prompt sent to the VLM for every window",
    )
    p.add_argument("--frames-per-window", type=int, default=4,
                   help="BAVA_MAX_FRAMES_PER_WINDOW for the intake — raise to push KV pressure")
    p.add_argument("--max-queued-windows", type=int, default=4,
                   help="BAVA_MAX_QUEUED_WINDOWS for the intake")
    p.add_argument("--stream-concurrency", type=int, default=2,
                   help="BAVA_STREAM_CONCURRENCY for the intake")
    p.add_argument("--out", default=f"/tmp/ab_n4_{int(time.time())}")
    p.add_argument("--configs", nargs="+", default=None,
                   help="which configs by name to run; default = all 3")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vllm_api_bases = _parse_api_bases(args.vllm_api_base, args.vllm_api_base_list)

    cfgs = CONFIGS
    if args.configs:
        cfgs = [c for c in CONFIGS if c["name"] in set(args.configs)]

    results: Dict[str, Any] = {
        "started_at": time.time(),
        "n": args.n,
        "per_config_duration": args.per_config_duration,
        "vllm_api_base": args.vllm_api_base,
        "vllm_api_bases": vllm_api_bases,
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

        print(
            f"[ab] starting intake (controller={cfg['controller_enabled']}, "
            f"vllm={','.join(vllm_api_bases)})…"
        )
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
            },
            log_suffix=f"_{name}",
        )

        if not wait_intake_ready(args.intake_admin_base):
            print(f"[ab][{name}] intake never came ready; skipping", file=sys.stderr)
            results["results"][name] = {"error": "intake not ready"}
            continue
        if not wait_intake_engines_ready(args.intake_admin_base, vllm_api_bases):
            print(f"[ab][{name}] intake engines never became healthy; skipping", file=sys.stderr)
            results["results"][name] = {"error": "engines not healthy"}
            continue

        # Belt-and-suspenders: make sure no leftover vLLM sessions leak from
        # a previous run. The new intake didn't create any but a previous
        # intake might have left some in vLLM's dict.
        aborted = purge_sessions(args.intake_admin_base)
        print(f"[ab] post-start purge: aborted={aborted}")

        summary = run_bench(
            n=args.n,
            duration=args.per_config_duration,
            cloud_ws_url=args.cloud_ws_url,
            admin_base=args.intake_admin_base,
            sources_globs=args.sources,
            rho=float(cfg["rho"]),
            alpha=float(cfg["alpha"]) if not cfg["controller_enabled"] else None,
            window_seconds=args.window_seconds,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            out_dir=cfg_out,
        )
        controller_log_local = cfg_out / "controller.jsonl"
        anchor_log_local = cfg_out / "anchors.jsonl"
        intake_log_local = cfg_out / "intake.log"
        _fetch_remote_text(
            args.cloud_host,
            args.cloud_ssh_port,
            args.cloud_ssh_key,
            args.cloud_ssh_user,
            remote_controller_log,
            controller_log_local,
        )
        _fetch_remote_text(
            args.cloud_host,
            args.cloud_ssh_port,
            args.cloud_ssh_key,
            args.cloud_ssh_user,
            remote_anchor_log,
            anchor_log_local,
        )
        _fetch_remote_text(
            args.cloud_host,
            args.cloud_ssh_port,
            args.cloud_ssh_key,
            args.cloud_ssh_user,
            remote_intake_log,
            intake_log_local,
        )
        results["results"][name] = {
            "config": cfg,
            "summary_path": str(cfg_out / "summary.json"),
            "controller_log_path": str(controller_log_local),
            "anchor_log_path": str(anchor_log_local),
            "intake_log_path": str(intake_log_local),
            "latency_stats": summary.get("latency_stats"),
            "n_probes": summary.get("n_probes"),
            "run_wall_seconds": summary.get("run_wall_seconds"),
        }

    # Final combined report
    comparison: List[Dict[str, Any]] = []
    for name in [c["name"] for c in cfgs]:
        r = results["results"].get(name) or {}
        lat = (r.get("latency_stats") or {}).get("__all__") or {}
        comparison.append({
            "name": name,
            "description": next((c["description"] for c in CONFIGS if c["name"] == name), ""),
            "n_windows": int(lat.get("n") or 0),
            "append_p50_ms": lat.get("append_p50"),
            "append_p95_ms": lat.get("append_p95"),
            "e2e_p50_ms": lat.get("e2e_p50"),
            "e2e_p95_ms": lat.get("e2e_p95"),
        })
    results["comparison"] = comparison

    (out / "ab_summary.json").write_text(json.dumps(results, indent=2, default=str))

    # Stop intake at the end; leave cloud as we found it.
    print("\n[ab] stopping final intake…")
    stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)

    # Print comparison table
    print("\n================ A/B SUMMARY ================")
    hdr = f"{'config':<15} {'n_win':>6} {'append_p50':>12} {'append_p95':>12} {'e2e_p50':>12} {'e2e_p95':>12}"
    print(hdr)
    print("-" * len(hdr))
    for row in comparison:
        def _fmt(x):
            return f"{x:.0f}ms" if isinstance(x, (int, float)) else "-"
        print(f"{row['name']:<15} {row['n_windows']:>6} "
              f"{_fmt(row['append_p50_ms']):>12} {_fmt(row['append_p95_ms']):>12} "
              f"{_fmt(row['e2e_p50_ms']):>12} {_fmt(row['e2e_p95_ms']):>12}")
    print(f"\nfull results: {out}/ab_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
