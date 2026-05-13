"""Summarize Exp1 KV-waterline and stall signals from an ab_bench run.

Input is the output directory produced by ``edge.tools.ab_bench``.  The tool
walks each config / N step, reads ``probes.jsonl`` and ``summary.json``, then
emits a compact table suitable for deciding whether the run actually reached
the KV-spiral regime described in ``experiment_trace.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


OOM_PAT = re.compile(
    r"CUDA out of memory|out of memory|device-side assert", re.IGNORECASE
)
ENGINE_DEAD_PAT = re.compile(r"EngineDeadError|engine is dead", re.IGNORECASE)
EMERGENCY_PAT = re.compile(
    r"emergency|early[_ -]?(?:finish|finaliz(?:e|ed|er|ation))|panic|admission",
    re.IGNORECASE,
)
EARLY_FINALIZED_TRUE_PAT = re.compile(
    r"early[_ -]?finaliz(?:ed|e)[=:]\s*true|online_prefill early_finalized",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    idx = math.ceil(q * len(xs)) - 1
    return xs[max(0, min(idx, len(xs) - 1))]


def _mean(values: list[float]) -> float | None:
    xs = [v for v in values if math.isfinite(v)]
    return sum(xs) / len(xs) if xs else None


def _probe_summary(probes_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(probes_path)
    q_vals: list[float] = []
    kv_vals: list[float] = []
    running_vals: list[float] = []
    rho_vals: list[float] = []
    alpha_vals: list[float] = []
    elapsed: list[float] = []
    inflight_windows: list[float] = []
    live_kv_tokens: list[float] = []

    for rec in rows:
        t = rec.get("elapsed")
        if isinstance(t, (int, float)):
            elapsed.append(float(t))
        snap = rec.get("vllm_snapshot") or {}
        for key, bucket in (("Q", q_vals), ("KV", kv_vals), ("running", running_vals)):
            value = snap.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                bucket.append(float(value))

        streams = rec.get("controller_streams") or {}
        for st in streams.values():
            if not isinstance(st, dict):
                continue
            rho = st.get("rho")
            alpha = st.get("alpha")
            if isinstance(rho, (int, float)):
                rho_vals.append(float(rho))
            if isinstance(alpha, (int, float)):
                alpha_vals.append(float(alpha))
            iw = st.get("inflight_windows")
            if isinstance(iw, (int, float)):
                inflight_windows.append(float(iw))
            flow = st.get("flow") or {}
            live = flow.get("live_kv_tokens")
            if isinstance(live, (int, float)):
                live_kv_tokens.append(float(live))

    kv_slope_per_s = None
    if len(rows) >= 2:
        pairs: list[tuple[float, float]] = []
        for rec in rows:
            t = rec.get("elapsed")
            kv = (rec.get("vllm_snapshot") or {}).get("KV")
            if isinstance(t, (int, float)) and isinstance(kv, (int, float)):
                pairs.append((float(t), float(kv)))
        if len(pairs) >= 2 and pairs[-1][0] > pairs[0][0]:
            kv_slope_per_s = (pairs[-1][1] - pairs[0][1]) / (
                pairs[-1][0] - pairs[0][0]
            )

    return {
        "n_probes": len(rows),
        "q_max": max(q_vals) if q_vals else None,
        "q_p95": _percentile(q_vals, 0.95),
        "kv_max": max(kv_vals) if kv_vals else None,
        "kv_p95": _percentile(kv_vals, 0.95),
        "kv_slope_per_s": kv_slope_per_s,
        "running_max": max(running_vals) if running_vals else None,
        "rho_mean": _mean(rho_vals),
        "rho_min": min(rho_vals) if rho_vals else None,
        "alpha_mean": _mean(alpha_vals),
        "alpha_min": min(alpha_vals) if alpha_vals else None,
        "inflight_windows_max": max(inflight_windows) if inflight_windows else None,
        "live_kv_tokens_max": max(live_kv_tokens) if live_kv_tokens else None,
    }


def _count_patterns(paths: list[Path]) -> dict[str, int]:
    oom = engine_dead = emergency = 0
    early_finalized = 0
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        oom += len(OOM_PAT.findall(text))
        engine_dead += len(ENGINE_DEAD_PAT.findall(text))
        emergency += len(EMERGENCY_PAT.findall(text))
        early_finalized += len(EARLY_FINALIZED_TRUE_PAT.findall(text))
    return {
        "oom_freq": oom,
        "engine_dead_count": engine_dead,
        "emergency_mentions": emergency,
        "early_finalized_count": early_finalized,
    }


def _expected_decisions(manifest: dict[str, Any]) -> int:
    duration = float(manifest.get("duration_s") or manifest.get("duration") or 0.0)
    decision = float(manifest.get("decision_window_seconds") or 0.0)
    window = float(manifest.get("window_seconds") or 0.0)
    cadence = decision if decision > 0 else window
    if duration <= 0 or cadence <= 0:
        return 1
    return max(1, int(math.floor(duration / cadence)))


def _latency_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    lat = summary.get("latency_stats") or {}
    all_lat = lat.get("__all__") or {}
    expected = _expected_decisions(manifest)
    stable = 0
    stalled = 0
    missing = 0
    stream_rows = [
        row for sid, row in lat.items() if sid != "__all__" and isinstance(row, dict)
    ]
    min_ok = max(1, math.ceil(0.8 * expected))
    for row in stream_rows:
        n_done = int(row.get("n") or 0)
        if n_done >= min_ok:
            stable += 1
        else:
            missing += 1
        e2e_p95 = row.get("e2e_p95")
        if isinstance(e2e_p95, (int, float)) and float(e2e_p95) > 60000:
            stalled += 1
    return {
        "windows_completed": int(all_lat.get("n") or 0),
        "append_p95_ms": all_lat.get("append_p95"),
        "e2e_p95_ms": all_lat.get("e2e_p95"),
        "expected_decisions_per_stream": expected,
        "stable_stream_count": stable,
        "missing_stream_count": missing,
        "stall_count": stalled,
    }


def _collect_step(run_root: Path, config: str, n_value: int, summary_path: Path) -> dict[str, Any]:
    step_dir = summary_path.parent
    summary = _read_json(summary_path) if summary_path.exists() else {}
    manifest_path = Path(summary.get("manifest") or step_dir / "manifest.json")
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    probes_path = Path(summary.get("probes_jsonl") or step_dir / "probes.jsonl")
    logs = list(step_dir.glob("*.log"))
    # If the step crashed before bench wrote summary.json, include copied
    # config-level logs so EngineDead/OOM still surface on the failed step
    # instead of being smeared across earlier completed N levels.
    if not summary_path.exists():
        logs += list(step_dir.parent.glob("*.log"))
        logs += list(run_root.glob("*.log"))
    row: dict[str, Any] = {
        "config": config,
        "N": n_value,
        "step_dir": str(step_dir),
    }
    row.update(_latency_summary(summary, manifest))
    row.update(_probe_summary(probes_path))
    row.update(_count_patterns(logs))
    return row


def summarize(run_root: Path) -> list[dict[str, Any]]:
    ab_path = run_root / "ab_summary.json"
    if not ab_path.exists():
        rows: list[dict[str, Any]] = []
        for step_dir in sorted(run_root.glob("*/N*")):
            if not step_dir.is_dir():
                continue
            match = re.fullmatch(r"N(\d+)", step_dir.name)
            if match is None:
                continue
            config = step_dir.parent.name
            summary_path = step_dir / "summary.json"
            if summary_path.exists() or (step_dir / "probes.jsonl").exists():
                rows.append(_collect_step(run_root, config, int(match.group(1)), summary_path))
        return rows

    ab_summary = _read_json(ab_path)
    out: list[dict[str, Any]] = []
    for config, result in (ab_summary.get("results") or {}).items():
        ramp = result.get("ramp_results") or {}
        if ramp:
            for n_str, step in sorted(ramp.items(), key=lambda item: int(item[0])):
                summary_path = Path(step["summary_path"])
                out.append(_collect_step(run_root, config, int(n_str), summary_path))
        elif result.get("summary_path"):
            out.append(_collect_step(run_root, config, int(ab_summary.get("n") or 0), Path(result["summary_path"])))
    return out


def _write_outputs(rows: list[dict[str, Any]], run_root: Path) -> None:
    json_path = run_root / "exp1_stall_summary.json"
    csv_path = run_root / "exp1_stall_summary.csv"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    fields = [
        "config",
        "N",
        "windows_completed",
        "stable_stream_count",
        "missing_stream_count",
        "stall_count",
        "oom_freq",
        "engine_dead_count",
        "early_finalized_count",
        "q_max",
        "kv_max",
        "kv_slope_per_s",
        "rho_min",
        "alpha_min",
        "append_p95_ms",
        "e2e_p95_ms",
        "step_dir",
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "-"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    rows = summarize(args.run_root)
    _write_outputs(rows, args.run_root)
    header = [
        "config",
        "N",
        "win",
        "stable",
        "stall",
        "oom",
        "dead",
        "early_fin",
        "Qmax",
        "KVmax",
        "rho_min",
        "alpha_min",
        "e2e_p95",
    ]
    print(" ".join(f"{h:>14}" for h in header))
    for row in rows:
        vals = [
            row.get("config"),
            row.get("N"),
            row.get("windows_completed"),
            row.get("stable_stream_count"),
            row.get("stall_count"),
            row.get("oom_freq"),
            row.get("engine_dead_count"),
            row.get("early_finalized_count"),
            row.get("q_max"),
            row.get("kv_max"),
            row.get("rho_min"),
            row.get("alpha_min"),
            row.get("e2e_p95_ms"),
        ]
        print(" ".join(f"{_fmt(v):>14}" for v in vals))
    print(f"\nsummary: {args.run_root / 'exp1_stall_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
