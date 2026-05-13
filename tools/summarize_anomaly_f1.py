"""Aggregate per-video anomaly-detection F1 from BAVA bench logs.

Each per-video bench config dir contains:
  - manifest.json — streams[*].{stream_id, video_id, label, window_count, ...}
  - edge-<stream_id>.log — `[edge-uplink] result window=N text='...'` lines
  - intake.log — optional, used by the `yes_chunks10` rule to recover how
    many 4s chunks each EF-shortened result actually covered

Each stream corresponds to exactly one video (per_video_bench launches a
fresh edge subprocess per video, with stream_id = "v-<idx>"). Within a
stream, window_id is 0-based against the source mp4's pts.

Aggregation rule (default --positive-rule any): a video is predicted
positive iff any slice/window verdict is Yes. Anomaly+positive = TP,
normal+positive = FP, etc.

For online-prefill EF runs, `--positive-rule yes_chunks10` preserves the
old full-window behavior while avoiding EF-short-window overcounting: a
video is predicted positive iff Yes results cumulatively cover at least
10 chunks. If chunk metadata is unavailable for a Yes result, it is treated
as a full 10-chunk decision so non-EF/full-window runs are unaffected.

Usage:
  python -m edge.tools.summarize_anomaly_f1 \
      edge/data/bench_runs/anomaly_f1_<TS> --positive-rule any
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional


def parse_yesno(text: str) -> Optional[bool]:
    if not text:
        return None
    norm = re.sub(r"[^a-z]", "", text.lower())
    if not norm:
        return None
    if norm.startswith("yes"):
        return True
    if norm.startswith("no"):
        return False
    yi = norm.find("yes")
    ni = norm.find("no")
    if yi == -1 and ni == -1:
        return None
    if yi == -1:
        return False
    if ni == -1:
        return True
    return yi < ni


def iter_results(log_path: Path) -> Iterable[tuple[int, str]]:
    """Yield (window_id, raw_text) tuples from an edge stream log.

    Mirrors edge/tools/summarize_choice_accuracy.py:iter_result_texts.
    """
    if not log_path.exists():
        return
    pattern = re.compile(r"result window=(?P<window>-?\d+) text=(?P<text>.*)$")
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


def parse_chunk_list(raw: str) -> list[int]:
    """Parse the intake `chunks=[...]` field into integer chunk ids."""
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    chunks: list[int] = []
    for item in value:
        if isinstance(item, int):
            chunks.append(item)
    return chunks


def intake_chunk_counts(cfg_dir: Path) -> dict[tuple[str, int], int]:
    """Map (stream_id, decision/window_id) to the number of chunks used.

    Online-prefill intake logs include lines like:

      stream=v-047 engine=3 decision=0 done frames=17 chunks=[0, 1] ...

    Completion-over-intake logs include `completion decision=...`; accept
    both forms so the rule remains reusable.
    """
    path = cfg_dir / "intake.log"
    if not path.exists():
        return {}
    pattern = re.compile(
        r"stream=(?P<stream>\S+).*?(?:completion\s+)?decision=(?P<decision>\d+)\s+"
        r"done\s+frames=(?P<frames>\d+)\s+chunks=(?P<chunks>\[[^\]]*\])"
    )
    out: dict[tuple[str, int], int] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        chunks = parse_chunk_list(m.group("chunks"))
        out[(m.group("stream"), int(m.group("decision")))] = len(chunks)
    return out


def positive_by_rule(verdicts: list[bool], rule: str) -> bool:
    """Return True if the per-window verdicts trigger a video-level positive."""
    if not verdicts:
        return False
    if rule == "any":
        return any(verdicts)
    if rule == "all":
        return all(verdicts)
    if rule == "majority":
        return sum(verdicts) * 2 > len(verdicts)
    if rule == "consecutive2":
        for i in range(len(verdicts) - 1):
            if verdicts[i] and verdicts[i + 1]:
                return True
        return False
    if rule == "consecutive3":
        for i in range(len(verdicts) - 2):
            if verdicts[i] and verdicts[i + 1] and verdicts[i + 2]:
                return True
        return False
    if rule == "yes_chunks10":
        return any(verdicts)
    raise ValueError(f"unknown rule: {rule}")


def positive_by_records(records: list[dict], rule: str) -> bool:
    verdicts = [bool(r["verdict"]) for r in records if r.get("verdict") is not None]
    if rule != "yes_chunks10":
        return positive_by_rule(verdicts, rule)
    yes_chunks = sum(
        int(r.get("chunks") or 0)
        for r in records
        if r.get("verdict") is True
    )
    return yes_chunks >= 10


def summarize_config(cfg_dir: Path, rule: str) -> dict:
    manifest_path = cfg_dir / "manifest.json"
    if not manifest_path.exists():
        return {"error": f"missing {manifest_path}"}
    manifest = json.loads(manifest_path.read_text())
    streams = manifest.get("streams") or []
    chunk_counts = intake_chunk_counts(cfg_dir)

    # Group streams by parent_video_id (each stream may be a slice of a video).
    # Aggregate per-stream verdicts, then collapse slices of the same parent
    # into a single video-level decision.
    by_parent: dict[str, dict] = {}
    n_invalid_windows = 0
    n_total_windows = 0

    for s in streams:
        sid = s.get("stream_id")
        vid = s.get("video_id")
        pid = s.get("parent_video_id") or vid
        label = s.get("label")
        log_path = cfg_dir / f"edge-{sid}.log"
        results = sorted(iter_results(log_path), key=lambda r: r[0])
        slice_verdicts: list[bool] = []
        slice_timeline: list[dict] = []
        for window_id, text in results:
            v = parse_yesno(text)
            n_total_windows += 1
            if v is None:
                n_invalid_windows += 1
                slice_timeline.append({"window_id": window_id, "verdict": None, "text": text})
                continue
            slice_verdicts.append(bool(v))
            slice_timeline.append({"window_id": window_id, "verdict": bool(v), "text": text})

        bucket = by_parent.setdefault(pid, {
            "parent_video_id": pid,
            "label": label,
            "all_verdicts": [],
            "all_records": [],
            "slices": [],
        })
        bucket["all_verdicts"].extend(slice_verdicts)
        default_chunks = 10
        try:
            decision_window_s = float(s.get("decision_window_seconds") or manifest.get("decision_window_seconds") or 40.0)
            window_s = float(s.get("window_seconds") or manifest.get("window_seconds") or 4.0)
            if window_s > 0:
                default_chunks = max(1, round(decision_window_s / window_s))
        except Exception:
            default_chunks = 10
        for rec in slice_timeline:
            if rec.get("verdict") is None:
                rec["chunks"] = 0
                rec["chunk_source"] = "invalid"
            else:
                key = (str(sid), int(rec["window_id"]))
                if key in chunk_counts:
                    rec["chunks"] = chunk_counts[key]
                    rec["chunk_source"] = "intake"
                else:
                    rec["chunks"] = default_chunks
                    rec["chunk_source"] = "fallback_full_window"
            bucket["all_records"].append(rec)
        bucket["slices"].append({
            "stream_id": sid,
            "video_id": vid,
            "n_windows_seen": len(results),
            "n_windows_valid": len(slice_verdicts),
            "any_yes": any(slice_verdicts),
            "yes_chunks": sum(int(rec.get("chunks") or 0) for rec in slice_timeline if rec.get("verdict") is True),
            "timeline": slice_timeline,
        })

    per_video: list[dict] = []
    tp = fp = fn = tn = 0
    for pid, bucket in by_parent.items():
        label = bucket["label"]
        verdicts = bucket["all_verdicts"]
        records = bucket["all_records"]
        predicted_positive = positive_by_records(records, rule)
        true_positive_label = (label == "anomaly")

        if predicted_positive and true_positive_label:
            outcome = "TP"; tp += 1
        elif predicted_positive and not true_positive_label:
            outcome = "FP"; fp += 1
        elif not predicted_positive and true_positive_label:
            outcome = "FN"; fn += 1
        else:
            outcome = "TN"; tn += 1

        per_video.append({
            "parent_video_id": pid,
            "label": label,
            "n_slices": len(bucket["slices"]),
            "n_total_yes": sum(1 for v in verdicts if v),
            "n_total_no": sum(1 for v in verdicts if not v),
            "n_total_yes_chunks": sum(int(r.get("chunks") or 0) for r in records if r.get("verdict") is True),
            "positive_chunk_threshold": 10 if rule == "yes_chunks10" else None,
            "predicted_positive": predicted_positive,
            "outcome": outcome,
            "slices": bucket["slices"],
        })

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / n if n else None

    return {
        "n_videos": n,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "confusion_matrix": {
            "labels": ["anomaly", "normal"],
            "rows": {
                "anomaly": {"predicted_anomaly": tp, "predicted_normal": fn},
                "normal": {"predicted_anomaly": fp, "predicted_normal": tn},
            },
        },
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "F1": f1,
        "accuracy": accuracy,
        "n_total_windows": n_total_windows,
        "n_invalid_windows": n_invalid_windows,
        "rule": rule,
        "per_video": per_video,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ab_dir", type=Path,
                   help="A/B output directory (contains <config>/manifest.json subdirs)")
    p.add_argument("--positive-rule",
                   choices=["consecutive2", "consecutive3", "any", "all", "majority", "yes_chunks10"],
                   default="any",
                   help="aggregation across slices/windows of the same parent video")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    config_dirs = sorted(
        d for d in args.ab_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )

    report = {
        "ab_dir": str(args.ab_dir),
        "positive_rule": args.positive_rule,
        "configs": {},
    }
    for cfg_dir in config_dirs:
        report["configs"][cfg_dir.name] = summarize_config(cfg_dir, args.positive_rule)

    out_path = args.out or (args.ab_dir / "anomaly_f1_summary.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}\n")

    print(f"rule = {args.positive_rule}")
    hdr = f"{'config':<15} {'n':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'prec':>7} {'rec':>7} {'F1':>7} {'acc':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, c in report["configs"].items():
        if "error" in c:
            print(f"{name:<15} ERROR: {c['error']}")
            continue

        def _f(x):
            return f"{x*100:6.1f}%" if isinstance(x, (int, float)) else "    -  "

        print(
            f"{name:<15} {c['n_videos']:>4} "
            f"{c['TP']:>4} {c['FP']:>4} {c['FN']:>4} {c['TN']:>4} "
            f"{_f(c['precision'])} {_f(c['recall'])} {_f(c['F1'])} {_f(c['accuracy'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
