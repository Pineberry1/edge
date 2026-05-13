#!/usr/bin/env python3
"""JSON-driven Mirage cloud launcher.

The config describes one or more vLLM OpenAI-compatible endpoints plus the
intake service that load-balances across them. A "DP" setup can either be
represented as multiple engines in the JSON, or passed through to vLLM via
`extra_args` if the local vLLM fork supports data-parallel CLI flags.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request as urlreq


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    return expand(json.loads(path.read_text()))


def log_root(cfg: dict[str, Any]) -> Path:
    root = Path(str(cfg.get("log_root") or "/tmp/bava_mirage"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def engine_name(engine: dict[str, Any]) -> str:
    return str(engine.get("name") or f"port{engine['port']}")


def engine_base(engine: dict[str, Any], host_override: str | None = None) -> str:
    host = host_override or str(engine.get("host") or "127.0.0.1")
    return f"http://{host}:{int(engine['port'])}"


def vllm_bases(cfg: dict[str, Any]) -> list[str]:
    return [engine_base(e, "127.0.0.1") for e in (cfg.get("vllm") or {}).get("engines", [])]


def flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def append_cli_arg(cmd: list[str], key: str, value: Any) -> None:
    flag = flag_name(key)
    if value is None or value is False:
        return
    if value is True:
        cmd.append(flag)
        return
    if isinstance(value, list):
        for item in value:
            cmd += [flag, str(item)]
        return
    cmd += [flag, str(value)]


def vllm_cmd(cfg: dict[str, Any], engine: dict[str, Any]) -> list[str]:
    python = str(cfg.get("python") or sys.executable)
    model = str(engine.get("model") or cfg.get("model"))
    common = dict((cfg.get("vllm") or {}).get("common_args") or {})
    args = dict(common)
    args.update(engine.get("args") or {})
    args["model"] = model
    args["port"] = int(engine["port"])
    if "host" not in args:
        args["host"] = str(engine.get("host") or "127.0.0.1")

    cmd = [python, "-m", "vllm.entrypoints.openai.api_server"]
    for key, value in args.items():
        append_cli_arg(cmd, key, value)
    cmd += [str(x) for x in (engine.get("extra_args") or [])]
    return cmd


def env_for_process(cfg: dict[str, Any], extra: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    repo_dir = str(cfg.get("repo_dir") or "")
    work_root = str(cfg.get("work_root") or "")
    vllm_src = str(cfg.get("vllm_src") or "")
    parts = [p for p in [work_root, f"{repo_dir}/cloud" if repo_dir else "", vllm_src, env.get("PYTHONPATH", "")] if p]
    env["PYTHONPATH"] = ":".join(parts)
    for key, value in extra.items():
        env[str(key)] = str(value)
    return env


def start_vllm(cfg: dict[str, Any]) -> None:
    root = log_root(cfg) / "vllm_logs"
    root.mkdir(parents=True, exist_ok=True)
    vcfg = cfg.get("vllm") or {}
    common_env = dict(vcfg.get("common_env") or {})
    work_root = str(cfg.get("work_root") or os.getcwd())
    for engine in vcfg.get("engines") or []:
        name = engine_name(engine)
        log_path = root / f"vllm_{name}_{int(engine['port'])}.log"
        pid_path = root / f"vllm_{name}_{int(engine['port'])}.pid"
        env = dict(common_env)
        env.update(engine.get("env") or {})
        if engine.get("cuda_visible_devices") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(engine["cuda_visible_devices"])
        cmd = vllm_cmd(cfg, engine)
        log_path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
        with log_path.open("w") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=work_root,
                env=env_for_process(cfg, env),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        pid_path.write_text(str(proc.pid) + "\n")
        print(f"[vllm] {name} port={engine['port']} pid={proc.pid} log={log_path}")


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urlreq.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def http_text(url: str, timeout: float = 3.0) -> str:
    try:
        with urlreq.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return ""


def metric_snapshot(base: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {"running": None, "waiting": None, "kv": None}
    text = http_text(base + "/metrics")
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            out["running"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_waiting"):
            out["waiting"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:gpu_cache_usage_perc"):
            out["kv"] = float(line.rsplit(" ", 1)[1])
    return out


def wait_vllm(cfg: dict[str, Any], timeout_s: float) -> int:
    deadline = time.time() + timeout_s
    engines = (cfg.get("vllm") or {}).get("engines") or []
    while time.time() < deadline:
        states = {engine_name(e): http_ok(engine_base(e) + "/health") for e in engines}
        print(f"[vllm] health {states}", flush=True)
        if states and all(states.values()):
            return 0
        time.sleep(5)
    return 1


def start_intake(cfg: dict[str, Any]) -> None:
    intake = cfg.get("intake") or {}
    root = log_root(cfg)
    log_path = Path(str(intake.get("log_path") or root / "intake.log"))
    pid_path = Path(str(intake.get("pid_path") or root / "intake.pid"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(intake.get("env") or {})
    env["INTAKE_HOST"] = str(intake.get("host") or "0.0.0.0")
    env["INTAKE_PORT"] = str(intake.get("port") or 9100)
    bases = [str(x) for x in (intake.get("vllm_api_base_list") or [])] or vllm_bases(cfg)
    if not bases:
        raise RuntimeError("no vLLM API bases configured")
    env["VLLM_API_BASE"] = str(intake.get("vllm_api_base") or bases[0])
    env["VLLM_API_BASE_LIST"] = ",".join(bases)
    if intake.get("controller_log"):
        env["BAVA_CONTROLLER_LOG"] = str(intake["controller_log"])
    if intake.get("anchor_log"):
        env["BAVA_ANCHOR_LOG"] = str(intake["anchor_log"])
    for path_key in ["controller_log", "anchor_log"]:
        if intake.get(path_key):
            Path(str(intake[path_key])).unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    pid_path.unlink(missing_ok=True)
    python = str(cfg.get("python") or sys.executable)
    work_root = str(cfg.get("work_root") or os.getcwd())
    with log_path.open("w") as log_fp:
        proc = subprocess.Popen(
            [python, "-m", "intake.server"],
            cwd=work_root,
            env=env_for_process(cfg, env),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid) + "\n")
    print(f"[intake] pid={proc.pid} log={log_path} api_bases={env['VLLM_API_BASE_LIST']}")


def stop_pid_file(path: Path, timeout_s: float = 30.0) -> None:
    if not path.exists():
        return
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def stop_intake(cfg: dict[str, Any]) -> None:
    intake = cfg.get("intake") or {}
    pid_path = Path(str(intake.get("pid_path") or log_root(cfg) / "intake.pid"))
    stop_pid_file(pid_path)
    print(f"[intake] stopped pid_file={pid_path}")


def stop_vllm(cfg: dict[str, Any]) -> None:
    root = log_root(cfg) / "vllm_logs"
    for engine in (cfg.get("vllm") or {}).get("engines") or []:
        name = engine_name(engine)
        pid_path = root / f"vllm_{name}_{int(engine['port'])}.pid"
        stop_pid_file(pid_path, timeout_s=60)
        print(f"[vllm] stopped {name} pid_file={pid_path}")


def status(cfg: dict[str, Any]) -> int:
    ok = True
    for engine in (cfg.get("vllm") or {}).get("engines") or []:
        base = engine_base(engine)
        healthy = http_ok(base + "/health")
        ok = ok and healthy
        metrics = metric_snapshot(base)
        print(
            f"{engine_name(engine)} {base} health={healthy} "
            f"running={metrics['running']} waiting={metrics['waiting']} kv={metrics['kv']}"
        )
    intake = cfg.get("intake") or {}
    base = f"http://127.0.0.1:{int(intake.get('port') or 9100)}"
    healthy = http_ok(base + "/healthz")
    ok = ok and healthy
    print(f"intake {base} healthz={healthy}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("cloud/mirage_cloud_config.example.json"))
    parser.add_argument(
        "command",
        choices=["start-vllm", "wait-vllm", "start-intake", "stop-intake", "stop-vllm", "status"],
    )
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "start-vllm":
        start_vllm(cfg)
        return 0
    if args.command == "wait-vllm":
        return wait_vllm(cfg, args.timeout_s)
    if args.command == "start-intake":
        start_intake(cfg)
        return 0
    if args.command == "stop-intake":
        stop_intake(cfg)
        return 0
    if args.command == "stop-vllm":
        stop_vllm(cfg)
        return 0
    if args.command == "status":
        return status(cfg)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
