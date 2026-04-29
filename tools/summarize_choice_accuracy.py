"""Summarize UCF-Crime choice-answer accuracy from BAVA bench logs.

The edge receiver logs one line per cloud result:

    [edge-uplink] result window=3 text='Arson'

This utility maps each stream back to its source video via manifest.json,
derives the class from the filename prefix, parses the one-word choice from
the result text, and writes an aggregate JSON report for an A/B run directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


LABELS = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


LABEL_BY_NORM = {_norm(label): label for label in LABELS}
LABEL_PATTERNS = {
    **LABEL_BY_NORM,
    "roadaccident": "RoadAccidents",
    "roadaccidents": "RoadAccidents",
    "accident": "RoadAccidents",
    "road": "RoadAccidents",
    "shoplifting": "Shoplifting",
    "shoplift": "Shoplifting",
}


def label_from_source(path: str) -> str:
    stem = Path(path).stem
    # UCF names look like Abuse028_x264.mp4 or RoadAccidents002_x264.mp4.
    m = re.match(r"([A-Za-z]+)", stem)
    if not m:
        return ""
    raw = m.group(1)
    return LABEL_BY_NORM.get(_norm(raw), raw)


def parse_prediction(text: str) -> Optional[str]:
    ntext = _norm(text)
    if not ntext:
        return None
    # Prefer exact/early labels; the prompt asks for exactly one class, but
    # models sometimes answer "The answer is Arson."
    best: tuple[int, int, str] | None = None
    for key, label in LABEL_PATTERNS.items():
        idx = ntext.find(key)
        if idx < 0:
            continue
        cand = (idx, -len(key), label)
        if best is None or cand < best:
            best = cand
    return best[2] if best is not None else None


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


def summarize_config(config_dir: Path) -> Dict[str, Any]:
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
        truth = label_from_source(source)
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
    p = argparse.ArgumentParser()
    p.add_argument("ab_dir", help="A/B output directory containing config subdirs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ab_dir = Path(args.ab_dir)
    config_names = [
        p.name
        for p in sorted(ab_dir.iterdir())
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    report = {
        "ab_dir": str(ab_dir),
        "labels": LABELS,
        "configs": {name: summarize_config(ab_dir / name) for name in config_names},
    }
    out_path = Path(args.out) if args.out else ab_dir / "accuracy_summary.json"
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
