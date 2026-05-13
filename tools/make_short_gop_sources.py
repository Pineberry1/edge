"""Build IDR-dense UCF sources for pressure testing.

The edge relay flushes open GOPs at online window boundaries. If a source has
long GOPs, decoder-safe flushing can leave many windows empty until the next
IDR. This tool re-encodes source videos to H.264 with a fixed short GOP so
benchmarks can create real cloud-side pressure instead of launching many
mostly-empty streams.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import av

from edge.src.features import FeatureExtractor
from edge.src.rtsp_source import iter_packets


def _source_label(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "anomaly" in parts:
        return "anomaly"
    if "normal" in parts:
        return "normal"
    return "misc"


def _output_path(source: Path, out_dir: Path, gop_seconds: float) -> Path:
    digest = hashlib.blake2b(str(source).encode("utf-8"), digest_size=4).hexdigest()
    gop_ms = int(round(gop_seconds * 1000))
    name = f"{source.stem}__sgop{gop_ms}ms__{digest}.mp4"
    return out_dir / _source_label(source) / name


def _stream_rate(stream) -> Fraction:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        return Fraction(25, 1)
    try:
        return Fraction(rate)
    except Exception:
        return Fraction(25, 1)


def _validate(path: Path, window_seconds: float) -> Dict[str, Any]:
    extractor = FeatureExtractor()
    n_packets = 0
    n_idr = 0
    idr_pts: List[float] = []
    window_has_idr: Dict[int, bool] = {}
    for rec in iter_packets(str(path)):
        feats = extractor.process(rec)
        n_packets += 1
        pts = feats.pts_s
        if pts is not None and window_seconds > 0:
            wid = int(pts / window_seconds)
            window_has_idr.setdefault(wid, False)
            if feats.is_idr:
                window_has_idr[wid] = True
        if feats.is_idr:
            n_idr += 1
            if pts is not None:
                idr_pts.append(float(pts))
    gaps = [b - a for a, b in zip(idr_pts, idr_pts[1:])]
    windows = len(window_has_idr)
    windows_with_idr = sum(1 for ok in window_has_idr.values() if ok)
    return {
        "n_packets": n_packets,
        "n_idr": n_idr,
        "idr_gap_s_max": max(gaps) if gaps else None,
        "idr_gap_s_mean": sum(gaps) / len(gaps) if gaps else None,
        "windows": windows,
        "windows_with_idr": windows_with_idr,
        "window_idr_coverage": (windows_with_idr / windows) if windows else None,
    }


def _transcode_one(
    source_str: str,
    out_dir_str: str,
    gop_seconds: float,
    window_seconds: float,
    preset: str,
    crf: int,
    overwrite: bool,
    max_frames: int,
) -> Dict[str, Any]:
    source = Path(source_str)
    out_dir = Path(out_dir_str)
    dst = _output_path(source, out_dir, gop_seconds)
    dst.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if dst.exists() and not overwrite:
        validation = _validate(dst, window_seconds)
        return {
            "source": str(source),
            "output": str(dst),
            "status": "exists",
            "wall_s": time.time() - started,
            **validation,
        }

    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    if tmp.exists():
        tmp.unlink()

    in_container = av.open(str(source))
    video_streams = [s for s in in_container.streams if s.type == "video"]
    if not video_streams:
        in_container.close()
        return {"source": str(source), "output": str(dst), "status": "no_video"}
    in_stream = video_streams[0]
    rate = _stream_rate(in_stream)
    fps = float(rate)
    frame_time_base = Fraction(rate.denominator, rate.numerator)
    gop_frames = max(1, int(round(fps * gop_seconds)))
    width = int(in_stream.codec_context.width or 0)
    height = int(in_stream.codec_context.height or 0)
    if width <= 0 or height <= 0:
        in_container.close()
        return {"source": str(source), "output": str(dst), "status": "bad_size"}

    out_container = av.open(str(tmp), "w", format="mp4")
    out_stream = out_container.add_stream("libx264", rate=rate)
    out_stream.time_base = frame_time_base
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.codec_context.gop_size = gop_frames
    out_stream.codec_context.max_b_frames = 0
    out_stream.options = {
        "preset": preset,
        "tune": "zerolatency",
        "crf": str(crf),
        "x264-params": (
            f"keyint={gop_frames}:min-keyint={gop_frames}:"
            "scenecut=0:open-gop=0:repeat-headers=1:bframes=0"
        ),
    }

    n_frames = 0
    try:
        for frame in in_container.decode(in_stream):
            if max_frames > 0 and n_frames >= max_frames:
                break
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")
            frame.pts = n_frames
            frame.time_base = frame_time_base
            for packet in out_stream.encode(frame):
                out_container.mux(packet)
            n_frames += 1
        for packet in out_stream.encode(None):
            out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()

    os.replace(tmp, dst)
    validation = _validate(dst, window_seconds)
    return {
        "source": str(source),
        "output": str(dst),
        "status": "ok",
        "wall_s": time.time() - started,
        "fps": fps,
        "gop_seconds": gop_seconds,
        "gop_frames": gop_frames,
        "frames": n_frames,
        **validation,
    }


def _expand_sources(patterns: List[str], limit: int) -> List[str]:
    sources: List[str] = []
    for pattern in patterns:
        sources.extend(sorted(glob.glob(pattern)))
    deduped = list(dict.fromkeys(sources))
    if limit > 0:
        deduped = deduped[:limit]
    return deduped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gop-seconds", type=float, default=1.0)
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    sources = _expand_sources(args.sources, args.limit)
    if not sources:
        print("no sources matched", flush=True)
        return 2
    if args.gop_seconds <= 0:
        print("--gop-seconds must be positive", flush=True)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"
    print(f"[short-gop] sources={len(sources)} out={out_dir}", flush=True)

    worker_args = [
        (
            src,
            str(out_dir),
            float(args.gop_seconds),
            float(args.window_seconds),
            args.preset,
            int(args.crf),
            bool(args.overwrite),
            int(args.max_frames),
        )
        for src in sources
    ]

    results: List[Dict[str, Any]] = []
    with manifest_path.open("w") as mf:
        if args.jobs <= 1:
            for item in worker_args:
                result = _transcode_one(*item)
                results.append(result)
                mf.write(json.dumps(result, separators=(",", ":")) + "\n")
                mf.flush()
                print(f"[short-gop] {result['status']:>7s} {result['output']}", flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futures = [ex.submit(_transcode_one, *item) for item in worker_args]
                for fut in concurrent.futures.as_completed(futures):
                    result = fut.result()
                    results.append(result)
                    mf.write(json.dumps(result, separators=(",", ":")) + "\n")
                    mf.flush()
                    print(f"[short-gop] {result['status']:>7s} {result['output']}", flush=True)

    ok = [r for r in results if r.get("status") in {"ok", "exists"}]
    coverages = [
        float(r["window_idr_coverage"])
        for r in ok
        if isinstance(r.get("window_idr_coverage"), (int, float)) and not math.isnan(float(r["window_idr_coverage"]))
    ]
    summary = {
        "out_dir": str(out_dir),
        "n_sources": len(sources),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "gop_seconds": args.gop_seconds,
        "window_seconds": args.window_seconds,
        "coverage_min": min(coverages) if coverages else None,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        "manifest_jsonl": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
