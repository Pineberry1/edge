"""Paired binary accuracy over the common stream/window keys of configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .summarize_binary_accuracy import summarize_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ab_dir")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ab_dir = Path(args.ab_dir)
    config_names = args.configs or [
        p.name
        for p in sorted(ab_dir.iterdir())
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    summaries = {name: summarize_config(ab_dir / name) for name in config_names}
    keyed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for name, summary in summaries.items():
        rows = summary.get("rows") or []
        keyed[name] = {
            (str(row["stream_id"]), int(row["window_id"])): row
            for row in rows
        }

    common: set[tuple[str, int]] | None = None
    for rows in keyed.values():
        keys = set(rows)
        common = keys if common is None else common & keys
    common = common or set()

    configs: dict[str, Any] = {}
    for name, rows in keyed.items():
        selected = [rows[key] for key in sorted(common)]
        n = len(selected)
        correct = sum(1 for row in selected if row.get("correct"))
        by_truth: dict[str, dict[str, int]] = {}
        for row in selected:
            truth = str(row.get("truth"))
            pred = str(row.get("prediction"))
            by_truth.setdefault(truth, {})
            by_truth[truth][pred] = by_truth[truth].get(pred, 0) + 1
        configs[name] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "by_truth": by_truth,
        }

    report = {
        "ab_dir": str(ab_dir),
        "configs": config_names,
        "common_n": len(common),
        "common_keys": [{"stream_id": sid, "window_id": wid} for sid, wid in sorted(common)],
        "paired": configs,
        "unpaired": {
            name: {
                "n": summaries[name].get("n"),
                "correct": summaries[name].get("correct"),
                "accuracy": summaries[name].get("accuracy"),
            }
            for name in config_names
        },
    }
    out_path = Path(args.out) if args.out else ab_dir / "paired_binary_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"wrote {out_path}")
    print(f"common windows: {len(common)}")
    for name in config_names:
        row = configs[name]
        acc = row["accuracy"]
        print(f"{name:<15} {row['correct']:>4}/{row['n']:<4} {(acc * 100 if acc is not None else 0):>6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
