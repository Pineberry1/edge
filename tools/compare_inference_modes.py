"""Compare online-prefill static_full with native chat-completion results.

The expected layout is:

  <online-run>/static_full/{manifest.json,summary.json,...}
  <completion-run>/static_completion/{manifest.json,summary.json,...}

Both directories are also compatible with summarize_anomaly_f1.py. By default
this script uses the current anomaly-detection rule: any Yes window/slice makes
the parent video positive.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from edge.tools.summarize_anomaly_f1 import summarize_config


def percentile(values: list[float], q: float) -> float | None:
    xs = sorted(v for v in values if isinstance(v, (int, float)))
    if not xs:
        return None
    k = math.ceil(q * len(xs)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def describe_values(values: list[float]) -> dict[str, float | int | None]:
    xs = [v for v in values if isinstance(v, (int, float))]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "p50": percentile(xs, 0.50),
        "p95": percentile(xs, 0.95),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def count_result_lines(streams: list[dict[str, Any]]) -> int:
    n = 0
    for stream in streams:
        log_path = Path(str(stream.get("log") or ""))
        if log_path.exists():
            n += log_path.read_text(errors="replace").count("[edge-uplink] result window=")
    return n


def describe_run(cfg_dir: Path, rule: str) -> dict[str, Any]:
    manifest = load_json(cfg_dir / "manifest.json")
    summary = load_json(cfg_dir / "summary.json")
    streams = manifest.get("streams") or []
    wall_s = (
        summary.get("run_wall_seconds")
        or summary.get("wall_s")
        or manifest.get("wall_s")
        or ((manifest.get("ended_at") or 0) - (manifest.get("started_at") or 0))
    )

    completion_elapsed_ms: list[float] = []
    stream_wall_s: list[float] = []
    n_windows = int(summary.get("n_total_windows") or summary.get("n_windows") or 0)
    for stream in streams:
        if isinstance(stream.get("wall_s"), (int, float)):
            stream_wall_s.append(float(stream["wall_s"]))
        for window in stream.get("per_window") or []:
            if isinstance(window.get("elapsed_ms"), (int, float)):
                completion_elapsed_ms.append(float(window["elapsed_ms"]))
    if not n_windows:
        n_windows = len(completion_elapsed_ms) or count_result_lines(streams)

    latency_stats = summary.get("latency_stats") or {}
    online_all = latency_stats.get("__all__") if isinstance(latency_stats, dict) else None

    return {
        "path": str(cfg_dir),
        "f1": summarize_config(cfg_dir, rule),
        "wall_s": wall_s,
        "n_windows": n_windows,
        "windows_per_min": (n_windows / wall_s * 60.0) if wall_s else None,
        "stream_wall_s": describe_values(stream_wall_s),
        "completion_request_elapsed_ms": describe_values(completion_elapsed_ms),
        "online_latency_stats_all": online_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", required=True, type=Path,
                        help="run dir containing static_full/")
    parser.add_argument("--completion-dir", required=True, type=Path,
                        help="run dir containing static_completion/")
    parser.add_argument("--positive-rule", default="any",
                        choices=["any", "consecutive2", "consecutive3", "all", "majority"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "positive_rule": args.positive_rule,
        "configs": {
            "static_full": describe_run(args.online_dir / "static_full", args.positive_rule),
            "static_completion": describe_run(
                args.completion_dir / "static_completion", args.positive_rule
            ),
        },
    }
    out_path = args.out or (args.completion_dir / "staticfull_vs_completion.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"wrote {out_path}\n")
    hdr = (
        f"{'config':<18} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
        f"{'F1':>7} {'acc':>7} {'windows':>8} {'wall_s':>9} {'win/min':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, rec in report["configs"].items():
        f1 = rec["f1"]
        def fmt_pct(x: Any) -> str:
            return f"{x * 100:6.1f}%" if isinstance(x, (int, float)) else "    -  "
        print(
            f"{name:<18} {f1.get('n_videos', 0):>4} "
            f"{f1.get('TP', 0):>4} {f1.get('FP', 0):>4} "
            f"{f1.get('FN', 0):>4} {f1.get('TN', 0):>4} "
            f"{fmt_pct(f1.get('F1'))} {fmt_pct(f1.get('accuracy'))} "
            f"{rec.get('n_windows') or 0:>8} "
            f"{(rec.get('wall_s') or 0):>9.1f} "
            f"{(rec.get('windows_per_min') or 0):>9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
