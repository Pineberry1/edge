"""Summarize binary anomaly/normal accuracy from BAVA bench logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


def truth_from_source(path: str) -> str:
    parts = {p.lower() for p in Path(path).parts}
    if "anomaly" in parts:
        return "anomaly"
    if "normal" in parts:
        return "normal"
    stem = Path(path).stem.lower()
    if stem.startswith("normal") or "_normal_" in stem:
        return "normal"
    if stem.startswith("anomaly") or "_anomaly_" in stem:
        return "anomaly"
    return "anomaly"


def parse_prediction(text: str) -> Optional[str]:
    low = re.sub(r"[^a-z]", " ", text.lower())
    words = set(low.split())
    if "anomaly" in words or "abnormal" in words or "crime" in words:
        return "anomaly"
    if "normal" in words:
        return "normal"
    if "anomalous" in words or "suspicious" in words:
        return "anomaly"
    return None


def iter_result_texts(log_path: Path) -> Iterable[tuple[int, str]]:
    pattern = re.compile(r"result window=(?P<window>-?\d+) text=(?P<text>.*)$")
    if not log_path.exists():
        return
    for line in log_path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        raw = m.group("text").strip()
        try:
            text = ast.literal_eval(raw)
        except Exception:
            text = raw.strip("'\"")
        yield int(m.group("window")), str(text)


def summarize_config(config_dir: Path) -> dict[str, Any]:
    manifest_path = config_dir / "manifest.json"
    if not manifest_path.exists():
        return {"error": f"missing {manifest_path}"}
    manifest = json.loads(manifest_path.read_text())
    stream_sources = {
        str(row["stream_id"]): str(row["source"])
        for row in manifest.get("streams", [])
    }

    rows = []
    counts = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    for sid, source in stream_sources.items():
        truth = truth_from_source(source)
        log_path = config_dir / f"edge-{sid}.log"
        for window_id, text in iter_result_texts(log_path):
            pred = parse_prediction(text)
            correct = pred == truth
            rows.append(
                {
                    "stream_id": sid,
                    "window_id": window_id,
                    "source": source,
                    "truth": truth,
                    "prediction": pred,
                    "correct": correct,
                    "text": text,
                }
            )
            counts["n"] += 1
            if pred is None:
                counts["invalid"] += 1
                confusion[truth]["<invalid>"] += 1
            else:
                confusion[truth][pred] += 1
                if correct:
                    counts["correct"] += 1

    n = counts["n"]
    valid = n - counts["invalid"]
    return {
        "n": n,
        "valid": valid,
        "invalid": counts["invalid"],
        "correct": counts["correct"],
        "accuracy": (counts["correct"] / n) if n else None,
        "valid_accuracy": (counts["correct"] / valid) if valid else None,
        "by_truth": {k: dict(v) for k, v in sorted(confusion.items())},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ab_dir", help="A/B output directory containing config subdirs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ab_dir = Path(args.ab_dir)
    config_names = [
        p.name
        for p in sorted(ab_dir.iterdir())
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    report = {
        "ab_dir": str(ab_dir),
        "labels": ["anomaly", "normal"],
        "configs": {name: summarize_config(ab_dir / name) for name in config_names},
    }
    out_path = Path(args.out) if args.out else ab_dir / "binary_accuracy_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"wrote {out_path}")
    print(f"{'config':<15} {'n':>5} {'valid':>5} {'acc':>8} {'valid_acc':>10} {'invalid':>8}")
    for name, cfg in report["configs"].items():
        acc = cfg.get("accuracy")
        vacc = cfg.get("valid_accuracy")
        print(
            f"{name:<15} {int(cfg.get('n') or 0):>5} {int(cfg.get('valid') or 0):>5} "
            f"{(acc * 100 if acc is not None else 0):>7.1f}% "
            f"{(vacc * 100 if vacc is not None else 0):>9.1f}% "
            f"{int(cfg.get('invalid') or 0):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
