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
        "name": "static_alpha050",
        "description": "ρ=1.0 α=0.5 static (intake pixel-resize alpha; not token-only alpha)",
        "controller_enabled": False,
        "rho": 1.0,
        "alpha": 0.5,
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
    {
        "name": "exp1_e1_nocontrol",
        "description": "Exp1 E1: online-prefill bare run, static ρ=1 α=1, no admission/budget",
        "controller_enabled": False,
        "rho": 1.0,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "0",
            "BAVA_ADMISSION_KV_WARN": "999",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "exp1_e2_alpha_only",
        "description": "Exp1 E2: dynamic α only, ρ fixed at 1, no admission/budget",
        "controller_enabled": True,
        "rho": 1.0,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "0",
            "BAVA_MU_RHO": "0.0",
            "BAVA_CLIMB_BACK": "0",
            "BAVA_RHO_LO": "1.0",
            "BAVA_RHO_HI": "1.0",
            "BAVA_ADMISSION_KV_WARN": "999",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "exp1_e3_rho_only",
        "description": "Exp1 E3: dynamic ρ only, α fixed at 1, no admission/budget",
        "controller_enabled": True,
        "rho": 1.0,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "0",
            "BAVA_ALPHA_LO": "1.0",
            "BAVA_ALPHA_HI": "1.0",
            "BAVA_ADMISSION_KV_WARN": "999",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "exp1_e4_bava_full",
        "description": "Exp1 E4: full BAVA controller with admission/budget enabled",
        "controller_enabled": True,
        "rho": 0.5,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "1",
        },
    },
    {
        "name": "ef100_bava_earlyfinalizer",
        "description": "BAVA dynamic rho/alpha + vLLM early-finalizer, admission/budget disabled",
        "controller_enabled": True,
        "rho": 0.5,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "0",
            "BAVA_ADMISSION_KV_WARN": "999",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "ef100_alpha_fixedrho",
        "description": (
            "vLLM early-finalizer with cloud-side alpha adaptation only; "
            "rho fixed at the edge semantic default so cloud Q cannot force frame dropping"
        ),
        "controller_enabled": True,
        "rho": 0.5,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_BUDGET_ENABLED": "0",
            "BAVA_MU_RHO": "0.0",
            "BAVA_RHO_LO": "0.5",
            "BAVA_RHO_HI": "0.5",
            "BAVA_ADMISSION_KV_WARN": "999",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "dynamic_rho_only",
        "description": "controller ON, dynamic ρ only; α fixed at 1.0, admission thresholds disabled",
        "controller_enabled": True,
        "rho": 0.5,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_ALPHA_LO": "1.0",
            "BAVA_ALPHA_HI": "1.0",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
    {
        "name": "dynamic_alpha_only",
        "description": "controller ON, dynamic α only; ρ fixed at 1.0, admission thresholds disabled",
        "controller_enabled": True,
        "rho": 1.0,
        "alpha": 1.0,
        "extra_env": {
            "BAVA_MU_RHO": "0.0",
            "BAVA_CLIMB_BACK": "0",
            "BAVA_RHO_LO": "1.0",
            "BAVA_RHO_HI": "1.0",
            "BAVA_ADMISSION_KV_HIGH": "999",
            "BAVA_ADMISSION_KV_PANIC": "999",
            "BAVA_ADMISSION_PREEMPT_PANIC": "999",
        },
    },
]

VARIANT_ALIASES = {
    "c1": "static_full",
    "c2": "static_rho050",
    "c3": "static_alpha050",
    "c4": "dynamic_rho_only",
    "c5": "dynamic_alpha_only",
    "c6": "bava_dynamic",
    "e1": "exp1_e1_nocontrol",
    "e2": "exp1_e2_alpha_only",
    "e3": "exp1_e3_rho_only",
    "e4": "exp1_e4_bava_full",
    "ef100": "ef100_bava_earlyfinalizer",
    "ef100_fixedrho": "ef100_alpha_fixedrho",
    "ef100_alpha": "ef100_alpha_fixedrho",
}

REMOTE_BAVA_ENV_KEYS = (
    "BAVA_ADMISSION_KV_WARN",
    "BAVA_ADMISSION_KV_HIGH",
    "BAVA_ADMISSION_KV_PANIC",
    "BAVA_ADMISSION_PREEMPT_PANIC",
    "BAVA_KV_CAP_TOKENS",
    "BAVA_KV_SAFETY",
    "BAVA_BUDGET_POLICY",
    "BAVA_BUDGET_TARGET_WINDOWS",
    "BAVA_BUDGET_EF_WINDOWS",
    "BAVA_BUDGET_EF_COOLDOWN_S",
    "BAVA_BUDGET_MIN_WINDOWS",
    "BAVA_BUDGET_MAX_WINDOWS",
    "BAVA_BUDGET_BLOCK_OVERHEAD_TOKENS",
    "BAVA_ENGINE_ASSIGNMENT",
    "BAVA_SEND_WINDOW_ENABLED",
    "BAVA_SEND_WINDOW_INIT",
    "BAVA_SEND_WINDOW_LO",
    "BAVA_SEND_WINDOW_HI",
    "BAVA_SEND_WINDOW_INCREASE_STEP",
    "BAVA_SEND_WINDOW_STABLE_RESULTS",
    "BAVA_SEND_WINDOW_PROBE_INTERVAL_S",
    "BAVA_SEND_WINDOW_KV_PROBE_MAX",
    "BAVA_SEND_WINDOW_EF_REDUCE",
    "BAVA_SEND_WINDOW_FAILURE_REDUCE",
    "BAVA_SEND_WINDOW_MARGIN",
    "BAVA_SEND_WINDOW_EF_COOLDOWN_S",
    "BAVA_SEND_WINDOW_MIN_UPDATE_S",
    "BAVA_SEND_WINDOW_KV_PANIC",
    "BAVA_SEND_WINDOW_ENGINE_MONITOR_S",
    "BAVA_FRAME_H",
    "BAVA_FRAME_W",
    "BAVA_ALPHA_LO",
    "BAVA_ALPHA_HI",
    "BAVA_MU_ALPHA",
    "BAVA_FLOW_EMA_ALPHA",
    "BAVA_RHO_CONTROL_MODE",
    "BAVA_NET_CAP_BYTES_S",
    "BAVA_NET_TARGET_UTIL",
    "BAVA_NET_DEADBAND",
    "BAVA_NET_BACKLOG_TARGET_S",
    "BAVA_NET_SEND_WAIT_TARGET_MS",
    "BAVA_EDGE_CAP_TOKENS_S",
    "BAVA_EDGE_TARGET_UTIL",
    "BAVA_EDGE_DEADBAND",
    "BAVA_PREFILL_CAP_TOKENS_S",
    "BAVA_KV_HORIZON_S",
    "BAVA_KV_MARGIN_RATIO",
    "BAVA_ETA_FLOOR",
    "BAVA_ADMISSION_HORIZON_S",
    "BAVA_ADMISSION_KV_MARGIN_RATIO",
    "BAVA_ADMISSION_KV_MARGIN_TOKENS",
    "BAVA_BUDGET_REBALANCE_S",
    "BAVA_MEMORY_TOKENS_PER_WINDOW",
    "BAVA_VISUAL_MEMORY_NUM_FRAMES",
    "BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME",
    "BAVA_VISUAL_MEMORY_TEXT_PREFIX",
    "BAVA_VISUAL_MEMORY_ID_PREFIX",
    "BAVA_VISUAL_MEMORY_STRICT",
    "BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE",
)


def _forwarded_remote_env() -> Dict[str, str]:
    return {
        key: os.environ[key]
        for key in REMOTE_BAVA_ENV_KEYS
        if os.environ.get(key) not in (None, "")
    }


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


def _ssh_popen(
    cloud_host: str,
    ssh_port: int,
    ssh_key: str,
    user: str,
    cmd: str,
    stdout_path: Path,
) -> subprocess.Popen:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    fp = stdout_path.open("a")
    full = [
        "ssh", "-n", "-p", str(ssh_port), "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{user}@{cloud_host}", cmd,
    ]
    proc = subprocess.Popen(full, stdout=fp, stderr=subprocess.STDOUT, text=True)
    proc._stdout_fp = fp  # type: ignore[attr-defined]
    return proc


def _start_gpu_monitor(
    *,
    cloud_host: str,
    ssh_port: int,
    ssh_key: str,
    user: str,
    gpu_ids: str,
    interval_s: float,
    out_csv: Path,
) -> subprocess.Popen:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(
        "wall_time,index,utilization.gpu [%],utilization.memory [%],"
        "memory.used [MiB],power.draw [W]\n"
    )
    interval = max(0.1, float(interval_s))
    ids = ",".join(part.strip() for part in str(gpu_ids).split(",") if part.strip())
    if not ids:
        ids = "0"
    query = "index,utilization.gpu,utilization.memory,memory.used,power.draw"
    loop = (
        "while true; do "
        "ts=$(date +%s.%N); "
        f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits -i {shlex.quote(ids)} "
        r"""| awk -v ts="$ts" 'BEGIN{FS=",";OFS=","} """
        r"""{for(i=1;i<=NF;i++){gsub(/^ +| +$/,"",$i)}; """
        r"""print ts,$1,$2,$3,$4,$5; fflush()}'; """
        f"sleep {interval:.3f}; "
        "done"
    )
    return _ssh_popen(
        cloud_host,
        ssh_port,
        ssh_key,
        user,
        f"bash -lc {shlex.quote(loop)}",
        out_csv,
    )


def _stop_gpu_monitor(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
    try:
        proc._stdout_fp.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _percentile(values: List[float], pct: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lo = int(rank)
    hi = min(len(vals) - 1, lo + 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _summarize_values(values: List[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": sum(vals) / len(vals),
        "p50": _percentile(vals, 50),
        "p95": _percentile(vals, 95),
        "max": max(vals),
    }


def _summarize_gpu_csv(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    if not path.exists():
        return {"path": str(path), "n_samples": 0, "overall": {}, "per_gpu": {}}
    for raw in path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "wall_time": float(parts[0]),
                    "index": float(parts[1]),
                    "gpu": float(parts[2]),
                    "mem_util": float(parts[3]),
                    "mem_used": float(parts[4]),
                    "power": float(parts[5]),
                }
            )
        except ValueError:
            continue

    def _bucket(subset: List[Dict[str, float]]) -> Dict[str, Any]:
        return {
            "samples": len(subset),
            "sm_util_pct": _summarize_values([r["gpu"] for r in subset]),
            "mem_util_pct": _summarize_values([r["mem_util"] for r in subset]),
            "mem_used_mib": _summarize_values([r["mem_used"] for r in subset]),
            "power_w": _summarize_values([r["power"] for r in subset]),
        }

    gpu_ids = sorted({int(r["index"]) for r in rows})
    return {
        "path": str(path),
        "n_samples": len(rows),
        "overall": _bucket(rows),
        "per_gpu": {
            str(gpu_id): _bucket([r for r in rows if int(r["index"]) == gpu_id])
            for gpu_id in gpu_ids
        },
    }


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


def _parse_n_ramp(value: Optional[str], fallback: int) -> List[int]:
    if not value:
        return [int(fallback)]
    out: List[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        n = int(raw)
        if n <= 0:
            raise ValueError(f"--N-ramp values must be positive, got {n}")
        out.append(n)
    if not out:
        raise ValueError("--N-ramp did not contain any positive N values")
    return out


def _parse_float_list(value: str, default: Optional[List[float]] = None) -> List[float]:
    if not value:
        return list(default or [])
    out: List[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        v = float(raw)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"rho/alpha values must be in (0, 1], got {v}")
        out.append(v)
    return out


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _param_tag(name: str, value: float) -> str:
    return f"{name}{int(round(value * 1000)):04d}"


def _grid_configs(rhos: List[float], alphas: List[float]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for rho in rhos:
        for alpha in alphas:
            configs.append({
                "name": f"{_param_tag('rho', rho)}_{_param_tag('alpha', alpha)}",
                "description": f"static grid baseline ρ={rho:g}, α={alpha:g}",
                "controller_enabled": False,
                "rho": float(rho),
                "alpha": float(alpha),
            })
    return configs


def _select_configs(config_names: Optional[List[str]], variants: Optional[List[str]]) -> List[Dict[str, Any]]:
    by_name = {cfg["name"]: cfg for cfg in CONFIGS}
    requested = list(config_names or [])
    if variants:
        for variant in variants:
            key = variant.strip().lower()
            requested.append(VARIANT_ALIASES.get(key, variant.strip()))
    if not requested:
        return list(CONFIGS)

    selected: List[Dict[str, Any]] = []
    unknown: List[str] = []
    for name in requested:
        cfg = by_name.get(name)
        if cfg is None:
            unknown.append(name)
        else:
            selected.append(cfg)
    if unknown:
        valid = sorted([*by_name.keys(), *VARIANT_ALIASES.keys()])
        raise ValueError(f"unknown config/variant {unknown}; valid values: {valid}")
    return selected


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
    decision_window_seconds: float,
    frames_per_window: int,
    max_tokens: int,
    prompt: str,
    out_dir: Path,
    visual_memory_merge: bool = False,
    stagger_seconds: float = 0.0,
    playlist_mode: str = "none",
    align_source_switch_to_decision: bool = False,
) -> Dict[str, Any]:
    cmd = [
        sys.executable, "-m", "edge.tools.bench",
        "--n", str(n),
        "--duration", str(duration),
        "--cloud-ws-url", cloud_ws_url,
        "--intake-admin-base", admin_base,
        "--rho", str(rho),
        "--window-seconds", str(window_seconds),
        "--decision-window-seconds", str(decision_window_seconds),
        "--frames-per-window", str(frames_per_window),
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
        "--probe-interval", "0.5",
        "--out", str(out_dir),
        "--sources", *sources_globs,
    ]
    if playlist_mode != "none":
        cmd += ["--playlist-mode", playlist_mode]
    if align_source_switch_to_decision:
        cmd.append("--align-source-switch-to-decision")
    if alpha is not None:
        cmd += ["--alpha", str(alpha)]
    if visual_memory_merge:
        cmd.append("--visual-memory-merge")
    if stagger_seconds > 0:
        cmd += ["--stagger-seconds", str(stagger_seconds)]
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


def _latency_all(summary: Dict[str, Any]) -> Dict[str, Any]:
    lat = summary.get("latency_stats") or {}
    return lat.get("__all__") or {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--per-config-duration", type=float, default=60.0)
    p.add_argument("--duration", type=float, dest="per_config_duration",
                   help="alias for --per-config-duration")
    p.add_argument("--N-ramp", dest="n_ramp", default="",
                   help="comma-separated stream counts, e.g. 4,8,16,32,48")
    p.add_argument("--hold-seconds", type=float, default=None,
                   help="duration per N in --N-ramp; defaults to --per-config-duration")
    p.add_argument("--cloud-host", default="210.45.123.163")
    p.add_argument("--cloud-ssh-port", type=int, default=2222)
    p.add_argument("--cloud-ssh-user", default="mambauser")
    p.add_argument("--cloud-ssh-key", default=os.path.expanduser("~/.ssh/jupyterhub.pem"))
    p.add_argument("--intake-admin-base", default="http://127.0.0.1:19100")
    p.add_argument("--cloud-ws-url", default="ws://127.0.0.1:19100/stream")
    p.add_argument("--vllm-api-base", default="http://127.0.0.1:8003")
    p.add_argument("--vllm-api-base-list", default="")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument(
        "--playlist-mode",
        choices=["none", "partition"],
        default="none",
        help="partition source pool into per-camera playlists before launching edge streams",
    )
    p.add_argument(
        "--align-source-switch-to-decision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="for partition/source-list cameras, start each next source at a decision boundary",
    )
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument(
        "--decision-window-seconds",
        type=float,
        default=0.0,
        help="coarser edge-side stream_end cadence; <=0 keeps edge default (=window-seconds)",
    )
    p.add_argument("--max-tokens", type=int, default=12)
    p.add_argument(
        "--stagger-seconds", type=float, default=0.0,
        help="Spread the N stream launches evenly across this many seconds. "
             "Set to 10–20 to get a mix of decode/stream-end and append phases "
             "for partition / early-finalizer testing. Default 0 = tight launch.",
    )
    p.add_argument(
        "--prompt",
        default="In one short sentence, describe what is happening in this clip.",
        help="prompt sent to the VLM for every window",
    )
    p.add_argument("--frames-per-window", type=int, default=4,
                   help="expected edge-selected frames per online-prefill chunk; used for budget accounting, not intake subsampling")
    p.add_argument("--max-queued-windows", type=int, default=4,
                   help="BAVA_MAX_QUEUED_WINDOWS for the intake")
    p.add_argument("--stream-concurrency", type=int, default=2,
                   help="BAVA_STREAM_CONCURRENCY for the intake")
    p.add_argument(
        "--gpu-monitor",
        action="store_true",
        help="sample remote nvidia-smi during every ramp step and write gpu_smi.csv/gpu_summary.json",
    )
    p.add_argument(
        "--gpu-monitor-gpus",
        default="0,1,2,3",
        help="comma-separated GPU ids for --gpu-monitor",
    )
    p.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=0.5,
        help="seconds between nvidia-smi samples for --gpu-monitor",
    )
    p.add_argument(
        "--visual-memory",
        action="store_true",
        help="set hello.visual_memory_merge=true for post-stream_end visual memory export/import",
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
        "--visual-memory-strict",
        action="store_true",
        help="fail session creation if vLLM rejects visual-memory fields",
    )
    p.add_argument(
        "--visual-memory-warm-prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="warm prefix cache after exporting visual memory (default: intake env)",
    )
    p.add_argument("--out", default=f"/tmp/ab_n4_{int(time.time())}")
    p.add_argument("--configs", nargs="+", default=None,
                   help="which configs by name to run; default = all 3")
    p.add_argument("--variant", nargs="+", default=None,
                   help="Exp3 aliases c1..c6; may also name configs directly")
    p.add_argument("--rho-values", default="",
                   help="comma-separated static ρ values for a rho/alpha grid, e.g. 1,0.75,0.5")
    p.add_argument("--alpha-values", default="",
                   help="comma-separated static α values for a rho/alpha grid, e.g. 1,0.75,0.5")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cmd.txt").write_text(" ".join(shlex.quote(x) for x in sys.argv) + "\n")
    vllm_api_bases = _parse_api_bases(args.vllm_api_base, args.vllm_api_base_list)
    n_ramp = _parse_n_ramp(args.n_ramp, args.n)
    duration_per_step = float(args.hold_seconds if args.hold_seconds is not None else args.per_config_duration)
    visual_memory_enabled = bool(args.visual_memory or _truthy_env("BAVA_VISUAL_MEMORY_ENABLED"))
    visual_memory_strict = bool(args.visual_memory_strict or _truthy_env("BAVA_VISUAL_MEMORY_STRICT"))
    align_source_switch = (
        bool(args.align_source_switch_to_decision)
        if args.align_source_switch_to_decision is not None
        else _truthy_env("BAVA_ALIGN_SOURCE_SWITCH_TO_DECISION")
    )
    visual_memory_warm_prefix_cache = (
        args.visual_memory_warm_prefix_cache
        if args.visual_memory_warm_prefix_cache is not None
        else True
    )

    try:
        cfgs = _select_configs(args.configs, args.variant)
        grid_rhos = _parse_float_list(args.rho_values, default=[1.0] if args.alpha_values else [])
        grid_alphas = _parse_float_list(args.alpha_values, default=[1.0] if args.rho_values else [])
    except ValueError as e:
        print(f"[ab] {e}", file=sys.stderr)
        return 2
    grid_cfgs = _grid_configs(grid_rhos, grid_alphas)
    if grid_cfgs:
        if args.configs or args.variant:
            seen = {cfg["name"] for cfg in cfgs}
            cfgs.extend(cfg for cfg in grid_cfgs if cfg["name"] not in seen)
        else:
            cfgs = grid_cfgs

    results: Dict[str, Any] = {
        "started_at": time.time(),
        "n": args.n,
        "n_ramp": n_ramp,
        "per_config_duration": args.per_config_duration,
        "hold_seconds": duration_per_step,
        "vllm_api_base": args.vllm_api_base,
        "vllm_api_bases": vllm_api_bases,
        "prompt": args.prompt,
        "window_seconds": args.window_seconds,
        "decision_window_seconds": args.decision_window_seconds,
        "playlist_mode": args.playlist_mode,
        "align_source_switch_to_decision": align_source_switch,
        "visual_memory": {
            "enabled": visual_memory_enabled,
            "merge_signal": visual_memory_enabled,
            "num_frames": int(args.visual_memory_num_frames),
            "tokens_per_frame": int(args.visual_memory_tokens_per_frame),
            "text_prefix": args.visual_memory_text_prefix,
            "strict": visual_memory_strict,
            "warm_prefix_cache": visual_memory_warm_prefix_cache,
        },
        "gpu_monitor": {
            "enabled": bool(args.gpu_monitor),
            "gpus": args.gpu_monitor_gpus,
            "interval_s": float(args.gpu_monitor_interval),
        },
        "rho_values": grid_rhos,
        "alpha_values": grid_alphas,
        "configs_run": [c["name"] for c in cfgs],
        "variant_aliases": VARIANT_ALIASES,
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
        remote_env = {
            **_forwarded_remote_env(),
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
            "BAVA_VISUAL_MEMORY_STRICT": "1" if visual_memory_strict else "0",
            **(
                {
                    "BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE": (
                        "1" if visual_memory_warm_prefix_cache else "0"
                    )
                }
                if args.visual_memory_warm_prefix_cache is not None
                else {}
            ),
            **{str(k): str(v) for k, v in (cfg.get("extra_env") or {}).items()},
        }

        start_intake(
            args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user,
            args.vllm_api_base,
            vllm_api_bases=vllm_api_bases,
            controller_enabled=cfg["controller_enabled"],
            extra_env=remote_env,
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

        ramp_results: Dict[str, Any] = {}
        last_summary: Dict[str, Any] = {}
        for n_value in n_ramp:
            step_out = cfg_out if len(n_ramp) == 1 else cfg_out / f"N{n_value}"
            step_out.mkdir(parents=True, exist_ok=True)
            # Belt-and-suspenders: make sure no leftover vLLM sessions leak
            # from a previous config or previous ramp step.
            aborted = purge_sessions(args.intake_admin_base)
            print(f"[ab] pre-step purge N={n_value}: aborted={aborted}")

            gpu_proc: Optional[subprocess.Popen] = None
            gpu_summary: Optional[Dict[str, Any]] = None
            gpu_csv = step_out / "gpu_smi.csv"
            try:
                if args.gpu_monitor:
                    gpu_proc = _start_gpu_monitor(
                        cloud_host=args.cloud_host,
                        ssh_port=args.cloud_ssh_port,
                        ssh_key=args.cloud_ssh_key,
                        user=args.cloud_ssh_user,
                        gpu_ids=args.gpu_monitor_gpus,
                        interval_s=args.gpu_monitor_interval,
                        out_csv=gpu_csv,
                    )
                summary = run_bench(
                    n=n_value,
                    duration=duration_per_step,
                    cloud_ws_url=args.cloud_ws_url,
                    admin_base=args.intake_admin_base,
                    sources_globs=args.sources,
                    rho=float(cfg["rho"]),
                    alpha=float(cfg["alpha"]) if not cfg["controller_enabled"] else None,
                    window_seconds=args.window_seconds,
                    decision_window_seconds=args.decision_window_seconds,
                    frames_per_window=args.frames_per_window,
                    max_tokens=args.max_tokens,
                    prompt=args.prompt,
                    out_dir=step_out,
                    visual_memory_merge=bool(visual_memory_enabled),
                    stagger_seconds=args.stagger_seconds,
                    playlist_mode=args.playlist_mode,
                    align_source_switch_to_decision=align_source_switch,
                )
            finally:
                _stop_gpu_monitor(gpu_proc)
            if args.gpu_monitor:
                gpu_summary = _summarize_gpu_csv(gpu_csv)
                (step_out / "gpu_summary.json").write_text(json.dumps(gpu_summary, indent=2))
            post_step_aborted = purge_sessions(args.intake_admin_base)
            print(f"[ab] post-step purge N={n_value}: aborted={post_step_aborted}")
            last_summary = summary
            ramp_results[str(n_value)] = {
                "summary_path": str(step_out / "summary.json"),
                "latency_stats": summary.get("latency_stats"),
                "n_probes": summary.get("n_probes"),
                "run_wall_seconds": summary.get("run_wall_seconds"),
                "latency_all": _latency_all(summary),
                "gpu_smi_path": str(gpu_csv) if args.gpu_monitor else None,
                "gpu_summary_path": str(step_out / "gpu_summary.json") if args.gpu_monitor else None,
                "gpu_summary": gpu_summary,
                "post_step_purge_aborted": post_step_aborted,
            }
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
            "summary_path": str(cfg_out / "summary.json") if len(n_ramp) == 1 else None,
            "ramp_results": ramp_results,
            "controller_log_path": str(controller_log_local),
            "anchor_log_path": str(anchor_log_local),
            "intake_log_path": str(intake_log_local),
            "latency_stats": last_summary.get("latency_stats"),
            "n_probes": last_summary.get("n_probes"),
            "run_wall_seconds": last_summary.get("run_wall_seconds"),
        }

    # Final combined report
    comparison: List[Dict[str, Any]] = []
    for name in [c["name"] for c in cfgs]:
        r = results["results"].get(name) or {}
        ramp_results = r.get("ramp_results") or {}
        if ramp_results:
            items = [(int(n), step) for n, step in ramp_results.items()]
        else:
            items = [(args.n, {"latency_all": (r.get("latency_stats") or {}).get("__all__") or {}})]
        for n_value, step in sorted(items, key=lambda x: x[0]):
            lat = step.get("latency_all") or {}
            comparison.append({
                "name": name,
                "N": n_value,
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
    final_aborted = purge_sessions(args.intake_admin_base)
    print(f"\n[ab] final purge before stopping intake: aborted={final_aborted}")
    print("\n[ab] stopping final intake…")
    stop_intake(args.cloud_host, args.cloud_ssh_port, args.cloud_ssh_key, args.cloud_ssh_user)

    # Print comparison table
    print("\n================ A/B SUMMARY ================")
    hdr = f"{'config':<19} {'N':>4} {'n_win':>6} {'append_p50':>12} {'append_p95':>12} {'e2e_p50':>12} {'e2e_p95':>12}"
    print(hdr)
    print("-" * len(hdr))
    for row in comparison:
        def _fmt(x):
            return f"{x:.0f}ms" if isinstance(x, (int, float)) else "-"
        print(f"{row['name']:<19} {row['N']:>4} {row['n_windows']:>6} "
              f"{_fmt(row['append_p50_ms']):>12} {_fmt(row['append_p95_ms']):>12} "
              f"{_fmt(row['e2e_p50_ms']):>12} {_fmt(row['e2e_p95_ms']):>12}")
    print(f"\nfull results: {out}/ab_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
