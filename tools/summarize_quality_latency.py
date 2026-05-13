"""Summarize quality, latency, and pressure metrics for anomaly A/B runs.

The input is a run directory containing config subdirectories such as
``static_completion/``, ``static_full/``, and ``bava_dynamic/``. Each config
directory must contain a ``manifest.json`` and edge-style result logs.

This intentionally accepts both native completion runs and online-prefill
per-video runs:

* completion_bench writes per-window ``elapsed_ms`` / ``e2e_ms`` in manifest.
* per_video_bench writes online latency stats in ``summary.json`` and probes.

Example:

  python -m edge.tools.summarize_quality_latency \
      edge/data/bench_runs/latency_acc_edge_streamend_20260430_115609 \
      --configs static_completion static_full bava_dynamic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from edge.tools.summarize_anomaly_f1 import summarize_config


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not xs:
        return None
    k = math.ceil(q * len(xs)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def describe(values: list[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
        "max": max(xs),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def wall_seconds(manifest: dict[str, Any], summary: dict[str, Any]) -> float | None:
    value = (
        summary.get("run_wall_seconds")
        or summary.get("wall_s")
        or manifest.get("wall_s")
    )
    if isinstance(value, (int, float)):
        return float(value)
    started = manifest.get("started_at")
    ended = manifest.get("ended_at")
    if isinstance(started, (int, float)) and isinstance(ended, (int, float)):
        return float(ended - started)
    return None


def iter_result_lines(cfg_dir: Path, streams: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for stream in streams:
        sid = stream.get("stream_id")
        log_path = cfg_dir / f"edge-{sid}.log"
        if not log_path.exists():
            raw = stream.get("log")
            if raw:
                log_path = Path(str(raw))
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            if "[edge-uplink] result window=" in line:
                lines.append(line)
    return lines


def probe_metrics(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {"qmax": None, "kvmax": None, "running_max": None, "rho_avg": None, "alpha_avg": None}
    q: list[float] = []
    kv: list[float] = []
    running: list[float] = []
    rho: list[float] = []
    alpha: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        snap = row.get("vllm_snapshot") or {}
        if isinstance(snap.get("Q"), (int, float)):
            q.append(float(snap["Q"]))
        if isinstance(snap.get("KV"), (int, float)):
            kv.append(float(snap["KV"]))
        if isinstance(snap.get("running"), (int, float)):
            running.append(float(snap["running"]))
        streams = row.get("controller_streams") or {}
        if streams:
            rvals = [float(v["rho"]) for v in streams.values() if isinstance(v.get("rho"), (int, float))]
            avals = [float(v["alpha"]) for v in streams.values() if isinstance(v.get("alpha"), (int, float))]
            if rvals:
                rho.append(sum(rvals) / len(rvals))
            if avals:
                alpha.append(sum(avals) / len(avals))
    return {
        "qmax": max(q) if q else None,
        "kvmax": max(kv) if kv else None,
        "running_max": max(running) if running else None,
        "rho_avg": sum(rho) / len(rho) if rho else None,
        "alpha_avg": sum(alpha) / len(alpha) if alpha else None,
    }


def intake_decode_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"decode_failed": None, "decode_events": None, "decoded_frames": describe([])}
    text = path.read_text(errors="replace")
    frames = [float(m.group(1)) for m in re.finditer(r"decoded=(\d+) frames", text)]
    return {
        "decode_failed": len(re.findall(r"decode failed", text)),
        "decode_events": len(frames),
        "decoded_frames": describe(frames),
    }


def summarize_dir(cfg_dir: Path, positive_rule: str) -> dict[str, Any]:
    manifest = load_json(cfg_dir / "manifest.json")
    summary = load_json(cfg_dir / "summary.json")
    streams = manifest.get("streams") or []
    quality = summarize_config(cfg_dir, positive_rule)

    request_elapsed: list[float] = []
    completion_e2e: list[float] = []
    stream_wall: list[float] = []
    for stream in streams:
        if isinstance(stream.get("wall_s"), (int, float)):
            stream_wall.append(float(stream["wall_s"]))
        for window in stream.get("per_window") or []:
            if isinstance(window.get("elapsed_ms"), (int, float)):
                request_elapsed.append(float(window["elapsed_ms"]))
            if isinstance(window.get("e2e_ms"), (int, float)):
                completion_e2e.append(float(window["e2e_ms"]))

    latency_stats = summary.get("latency_stats") or {}
    online_all = latency_stats.get("__all__") if isinstance(latency_stats, dict) else {}
    result_lines = iter_result_lines(cfg_dir, streams)
    empty_results = sum(1 for line in result_lines if "text=''" in line or 'text=""' in line)

    wall_s = wall_seconds(manifest, summary)
    n_windows = (
        int(summary.get("n_total_windows") or summary.get("n_windows") or 0)
        or len(request_elapsed)
        or len(result_lines)
    )
    return {
        "config": cfg_dir.name,
        "path": str(cfg_dir),
        "quality": quality,
        "wall_s": wall_s,
        "n_windows": n_windows,
        "windows_per_min": (n_windows / wall_s * 60.0) if wall_s else None,
        "result_lines": len(result_lines),
        "empty_results": empty_results,
        "stream_wall_s": describe(stream_wall),
        "request_elapsed_ms": describe(request_elapsed),
        "completion_e2e_ms": describe(completion_e2e),
        "online_latency_all": online_all,
        "pressure": probe_metrics(cfg_dir / "probes.jsonl"),
        "decode": intake_decode_metrics(cfg_dir / "intake.log"),
    }


def fmt_ms(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value / 1000.0:.2f}s"


def fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return f"{value * 100.0:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--positive-rule", default="any",
                        choices=["any", "consecutive2", "consecutive3", "all", "majority", "yes_chunks10"])
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.configs:
        cfg_dirs = [args.run_dir / name for name in args.configs]
    else:
        cfg_dirs = sorted(d for d in args.run_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists())

    rows = [summarize_dir(d, args.positive_rule) for d in cfg_dirs if (d / "manifest.json").exists()]
    report = {
        "run_dir": str(args.run_dir),
        "positive_rule": args.positive_rule,
        "configs": {row["config"]: row for row in rows},
    }

    out_json = args.out_json or (args.run_dir / "quality_latency_summary.json")
    out_csv = args.out_csv or (args.run_dir / "quality_latency_summary.csv")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    fieldnames = [
        "config", "n_videos", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "accuracy",
        "n_total_windows", "n_invalid_windows", "result_lines", "empty_results",
        "wall_s", "windows_per_min", "append_p50", "append_p95", "e2e_p50", "e2e_p95",
        "request_p50", "request_p95", "completion_e2e_p50", "completion_e2e_p95",
        "qmax", "kvmax", "running_max", "rho_avg", "alpha_avg",
        "decode_failed", "decode_events", "decoded_frames_p50",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            q = row["quality"]
            online = row["online_latency_all"] or {}
            req = row["request_elapsed_ms"]
            ce2e = row["completion_e2e_ms"]
            pressure = row["pressure"]
            decode = row["decode"]
            writer.writerow({
                "config": row["config"],
                "n_videos": q.get("n_videos"),
                "TP": q.get("TP"),
                "FP": q.get("FP"),
                "FN": q.get("FN"),
                "TN": q.get("TN"),
                "precision": q.get("precision"),
                "recall": q.get("recall"),
                "F1": q.get("F1"),
                "accuracy": q.get("accuracy"),
                "n_total_windows": q.get("n_total_windows"),
                "n_invalid_windows": q.get("n_invalid_windows"),
                "result_lines": row["result_lines"],
                "empty_results": row["empty_results"],
                "wall_s": row["wall_s"],
                "windows_per_min": row["windows_per_min"],
                "append_p50": online.get("append_p50"),
                "append_p95": online.get("append_p95"),
                "e2e_p50": online.get("e2e_p50"),
                "e2e_p95": online.get("e2e_p95"),
                "request_p50": req.get("p50"),
                "request_p95": req.get("p95"),
                "completion_e2e_p50": ce2e.get("p50"),
                "completion_e2e_p95": ce2e.get("p95"),
                "qmax": pressure.get("qmax"),
                "kvmax": pressure.get("kvmax"),
                "running_max": pressure.get("running_max"),
                "rho_avg": pressure.get("rho_avg"),
                "alpha_avg": pressure.get("alpha_avg"),
                "decode_failed": decode.get("decode_failed"),
                "decode_events": decode.get("decode_events"),
                "decoded_frames_p50": (decode.get("decoded_frames") or {}).get("p50"),
            })

    print(f"wrote {out_json}")
    print(f"wrote {out_csv}\n")
    hdr = (
        f"{'config':<20} {'F1':>7} {'acc':>7} {'invalid':>12} {'empty':>7} "
        f"{'append95':>9} {'e2e95':>9} {'req95':>9} {'Qmax':>7} {'KVmax':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        q = row["quality"]
        online = row["online_latency_all"] or {}
        req = row["request_elapsed_ms"]
        invalid = f"{q.get('n_invalid_windows', 0)}/{q.get('n_total_windows', 0)}"
        pressure = row["pressure"]
        kvmax = pressure.get("kvmax")
        print(
            f"{row['config']:<20} {fmt_pct(q.get('F1')):>7} {fmt_pct(q.get('accuracy')):>7} "
            f"{invalid:>12} {row['empty_results']:>7} "
            f"{fmt_ms(online.get('append_p95')):>9} {fmt_ms(online.get('e2e_p95')):>9} "
            f"{fmt_ms(req.get('p95')):>9} "
            f"{(pressure.get('qmax') if pressure.get('qmax') is not None else '-'):>7} "
            f"{(f'{kvmax:.3f}' if isinstance(kvmax, (int, float)) else '-'):>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
