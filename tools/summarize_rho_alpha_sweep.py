"""Select rho/alpha operating points from an ab_bench grid run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    q = base / path
    if q.exists():
        return q
    return p


def _probe_metrics(probes_path: Path) -> Dict[str, Optional[float]]:
    if not probes_path.exists():
        return {
            "kvmax": None,
            "qmax": None,
            "running_max": None,
            "rho_avg": None,
            "alpha_avg": None,
        }
    kv: List[float] = []
    q: List[float] = []
    running: List[float] = []
    rho: List[float] = []
    alpha: List[float] = []
    for line in probes_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        snap = row.get("vllm_snapshot") or {}
        if isinstance(snap.get("KV"), (int, float)):
            kv.append(float(snap["KV"]))
        if isinstance(snap.get("Q"), (int, float)):
            q.append(float(snap["Q"]))
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
        "kvmax": max(kv) if kv else None,
        "qmax": max(q) if q else None,
        "running_max": max(running) if running else None,
        "rho_avg": sum(rho) / len(rho) if rho else None,
        "alpha_avg": sum(alpha) / len(alpha) if alpha else None,
    }


def _finite_or_inf(value: Optional[float], default: float) -> float:
    if value is None or math.isnan(value):
        return default
    return float(value)


def _candidate_key(row: Dict[str, Any], objective: str):
    throughput = _finite_or_inf(row.get("windows_per_min"), 0.0)
    e2e = _finite_or_inf(row.get("e2e_p95"), float("inf"))
    append = _finite_or_inf(row.get("append_p95"), float("inf"))
    quality = float(row["rho"]) * float(row["alpha"])
    if objective == "throughput":
        return (throughput, quality, -e2e, -append)
    if objective == "latency":
        return (-e2e, -append, quality, throughput)
    return (quality, throughput, -e2e, -append)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ab-summary", required=True)
    p.add_argument("--out-json", default="")
    p.add_argument("--out-csv", default="")
    p.add_argument("--target-e2e-p95-ms", type=float, default=2000.0)
    p.add_argument("--target-append-p95-ms", type=float, default=250.0)
    p.add_argument("--target-kvmax", type=float, default=0.60)
    p.add_argument("--target-qmax", type=float, default=8.0)
    p.add_argument("--objective", choices=["quality", "throughput", "latency"], default="quality")
    args = p.parse_args()

    summary_path = Path(args.ab_summary)
    base = summary_path.parent
    summary = json.loads(summary_path.read_text())
    rows: List[Dict[str, Any]] = []
    for name, result in (summary.get("results") or {}).items():
        cfg = result.get("config") or {}
        if cfg.get("controller_enabled"):
            continue
        if not isinstance(cfg.get("rho"), (int, float)) or not isinstance(cfg.get("alpha"), (int, float)):
            continue
        for n_value, step in (result.get("ramp_results") or {}).items():
            lat = step.get("latency_all") or {}
            summary_file = _resolve(step.get("summary_path") or "", base)
            probes = summary_file.parent / "probes.jsonl"
            metrics = _probe_metrics(probes)
            wall = float(step.get("run_wall_seconds") or 0.0)
            n_windows = float(lat.get("n") or 0.0)
            row = {
                "config": name,
                "N": int(n_value),
                "rho": float(cfg["rho"]),
                "alpha": float(cfg["alpha"]),
                "quality_proxy": float(cfg["rho"]) * float(cfg["alpha"]),
                "windows": n_windows,
                "windows_per_min": (n_windows / wall * 60.0) if wall > 0 else 0.0,
                "append_p50": lat.get("append_p50"),
                "append_p95": lat.get("append_p95"),
                "e2e_p50": lat.get("e2e_p50"),
                "e2e_p95": lat.get("e2e_p95"),
                **metrics,
            }
            row["feasible"] = (
                _finite_or_inf(row.get("e2e_p95"), float("inf")) <= args.target_e2e_p95_ms
                and _finite_or_inf(row.get("append_p95"), float("inf")) <= args.target_append_p95_ms
                and _finite_or_inf(row.get("kvmax"), 0.0) <= args.target_kvmax
                and _finite_or_inf(row.get("qmax"), 0.0) <= args.target_qmax
            )
            rows.append(row)

    best: Dict[str, Dict[str, Any]] = {}
    for n_value in sorted({row["N"] for row in rows}):
        group = [row for row in rows if row["N"] == n_value]
        feasible = [row for row in group if row["feasible"]]
        pool = feasible or group
        if not pool:
            continue
        key_objective = args.objective if feasible else "latency"
        best[str(n_value)] = max(pool, key=lambda row: _candidate_key(row, key_objective))

    report = {
        "ab_summary": str(summary_path),
        "objective": args.objective,
        "constraints": {
            "target_e2e_p95_ms": args.target_e2e_p95_ms,
            "target_append_p95_ms": args.target_append_p95_ms,
            "target_kvmax": args.target_kvmax,
            "target_qmax": args.target_qmax,
        },
        "best_by_N": best,
        "rows": rows,
    }

    out_json = Path(args.out_json) if args.out_json else base / "rho_alpha_best.json"
    out_csv = Path(args.out_csv) if args.out_csv else base / "rho_alpha_sweep.csv"
    out_json.write_text(json.dumps(report, indent=2))
    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "N", "config", "rho", "alpha", "quality_proxy", "feasible",
            "windows", "windows_per_min", "append_p50", "append_p95",
            "e2e_p50", "e2e_p95", "kvmax", "qmax", "running_max",
            "rho_avg", "alpha_avg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print("best_by_N:")
    for n_value, row in best.items():
        status = "feasible" if row.get("feasible") else "fallback"
        print(
            f"  N={n_value}: rho={row['rho']:.3g} alpha={row['alpha']:.3g} "
            f"({status}, e2e_p95={_finite_or_inf(row.get('e2e_p95'), 0):.0f}ms, "
            f"kvmax={_finite_or_inf(row.get('kvmax'), 0):.3f})"
        )
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
