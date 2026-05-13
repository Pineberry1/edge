"""No-online-prefill completion baseline for video-level anomaly detection.

Bypasses BAVA's edge-cloud streaming entirely. For each video, slices into
40s sliding windows, extracts 80 evenly-spaced frames per window,
encodes as JPEG/base64, and sends ONE chat-completion request per window
to a vanilla vLLM (no `--enable-online-prefill`). Parses Yes/No.

Output layout matches edge/tools/per_video_bench.py so the existing
edge/tools/summarize_anomaly_f1.py aggregator works unchanged:

  <out>/manifest.json    streams[*].{stream_id, video_id, label, ...}
  <out>/edge-<sid>.log   `[edge-uplink] result window=N text='...'` lines
  <out>/summary.json     run-level stats

Usage:
  python -m edge.tools.completion_bench \
      --manifest edge/data/eval_videos.tsv \
      --vllm-api-base http://127.0.0.1:8021 \
      --window-seconds 40 --stride-seconds 20 \
      --frames-per-window 80 --max-tokens 16 \
      --concurrency 1 \
      --out edge/data/bench_runs/.../static_completion
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import request as urlreq

import av  # noqa: F401  - rely on existing av installation

from edge.tools.ab_bench import (
    _start_gpu_monitor,
    _stop_gpu_monitor,
    _summarize_gpu_csv,
)


REPO = Path(__file__).resolve().parent.parent.parent
JPEG_ENCODE_SEM = threading.Semaphore(
    max(1, int(os.environ.get("BAVA_COMPLETION_JPEG_CONCURRENCY", "4")))
)


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            row = dict(zip(header, line.split("\t")))
            row["duration_s"] = float(row["duration_s"])
            rows.append(row)
    return rows


@dataclass
class Frame:
    pts_s: float
    image_b64: str  # JPEG base64 (no data: prefix)


def encode_video_frame_jpeg_b64(frame: av.VideoFrame, jpeg_quality: int = 75) -> str:
    """Encode a PyAV frame to JPEG base64."""
    with JPEG_ENCODE_SEM:
        try:
            image = frame.to_image()
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=jpeg_quality)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

        codec = av.CodecContext.create("mjpeg", "w")
        codec.width = frame.width
        codec.height = frame.height
        codec.pix_fmt = "yuvj420p"
        fr = frame.reformat(format="yuvj420p")
        packets = codec.encode(fr)
        for pkt in codec.encode(None):
            packets.append(pkt)
        data = b"".join(bytes(pkt) for pkt in packets)
        return base64.b64encode(data).decode()


def extract_frames(source: Path, start_s: float, end_s: float, n: int,
                   jpeg_quality: int = 75) -> list[Frame]:
    """Extract frames from [start_s, end_s) of `source`.

    When n > 0, return n evenly-spaced frames. When n <= 0, return every
    decoded frame in the interval without duplicating short clips.
    """
    container = av.open(str(source))
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration else end_s
    end_s = min(end_s, duration)
    if end_s <= start_s:
        container.close()
        return []
    targets = [start_s + (end_s - start_s) * (i + 0.5) / n for i in range(n)] if n > 0 else []

    # Seek to start of window (pyav seeks to keyframe).
    seek_pts = max(0, int((start_s) / stream.time_base))
    try:
        container.seek(seek_pts, stream=stream)
    except Exception:
        container.seek(0)

    out: list[Frame] = []
    last_t = -1.0
    for packet in container.demux(stream):
        for frame in packet.decode():
            t = float(frame.pts * stream.time_base) if frame.pts is not None else last_t + 0.04
            last_t = t
            if t < start_s:
                continue
            if t >= end_s:
                container.close()
                if n <= 0:
                    return _finalize_all(out, jpeg_quality)
                return _finalize(out, targets, jpeg_quality)
            # collect this frame for later target matching
            out.append((t, frame))
    container.close()
    if n <= 0:
        return _finalize_all(out, jpeg_quality)
    return _finalize(out, targets, jpeg_quality)


def _finalize_all(samples, jpeg_quality):
    return [
        Frame(pts_s=t, image_b64=encode_video_frame_jpeg_b64(frame, jpeg_quality))
        for t, frame in samples
    ]


def _finalize(samples, targets, jpeg_quality):
    if not samples:
        return []
    chosen: list[Frame] = []
    j = 0
    sorted_samples = samples  # already in decode order ≈ pts order
    for tgt in targets:
        # advance j while next sample is closer to target than current
        while j + 1 < len(sorted_samples) and \
                abs(sorted_samples[j + 1][0] - tgt) < abs(sorted_samples[j][0] - tgt):
            j += 1
        t, frame = sorted_samples[min(j, len(sorted_samples) - 1)]
        chosen.append(Frame(pts_s=t, image_b64=encode_video_frame_jpeg_b64(frame, jpeg_quality)))
    return chosen


def chat_completion(api_base: str, model: str, frames: list[Frame],
                    prompt: str, max_tokens: int, timeout: float = 90.0) -> tuple[str, dict]:
    content: list[dict] = []
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f.image_b64}"},
        })
    content.append({"type": "text", "text": prompt})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode()
    req = urlreq.Request(f"{api_base.rstrip('/')}/v1/chat/completions",
                         data=data, method="POST",
                         headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urlreq.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
    elapsed_ms = (time.time() - t0) * 1000.0
    text = (result.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text, {
        "elapsed_ms": elapsed_ms,
        "n_frames": len(frames),
        "usage": result.get("usage"),
    }


def discover_model(api_base: str) -> str:
    req = urlreq.Request(f"{api_base.rstrip('/')}/v1/models")
    with urlreq.urlopen(req, timeout=10.0) as resp:
        data = json.loads(resp.read().decode())
    return data["data"][0]["id"]


def process_video(slot: int, vidx: int, video: dict, model: str,
                  api_base: str, args, out_dir: Path) -> dict:  # api_base assigned per worker

    sid = f"v-{vidx:03d}"
    log_path = out_dir / f"edge-{sid}.log"
    log_fp = log_path.open("w")
    log_fp.write(f"[completion] slot={slot} api_base={api_base}\n")
    log_fp.flush()
    started = time.time()
    starts: list[float] = []
    t = 0.0
    while t < video["duration_s"]:
        starts.append(t)
        t += args.stride_seconds
    if not starts:
        starts = [0.0]
    src = Path(video["source"])
    print(f"[completion][slot{slot}] {sid} {video['label']} {video['video_id']} "
          f"dur={video['duration_s']:.1f}s windows={len(starts)} -> processing")

    per_window: list[dict] = []
    for w, start_s in enumerate(starts):
        end_s = min(start_s + args.window_seconds, video["duration_s"])
        if end_s - start_s < 1.0:
            continue
        if args.pace_realtime:
            due = started + end_s
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
        try:
            with args.frame_extract_sem:
                frames = extract_frames(src, start_s, end_s, args.frames_per_window)
        except Exception as e:
            log_fp.write(f"[edge-uplink] frames-fail window={w} err={e}\n")
            log_fp.flush()
            continue
        if not frames:
            log_fp.write(f"[edge-uplink] no-frames window={w}\n")
            continue
        try:
            text, meta = chat_completion(
                api_base, model, frames, args.prompt, args.max_tokens,
                timeout=args.request_timeout)
        except Exception as e:
            log_fp.write(f"[edge-uplink] vllm-fail window={w} err={e}\n")
            log_fp.flush()
            continue
        e2e_ms = (time.time() - (started + start_s)) * 1000.0
        # Match the edge result-line format expected by summarize_anomaly_f1
        log_fp.write(f"[edge-uplink] result window={w} text='{text}'\n")
        log_fp.flush()
        usage = meta.get("usage") or {}
        per_window.append({
            "window_id": w,
            "n_frames": meta["n_frames"],
            "elapsed_ms": meta["elapsed_ms"],
            "e2e_ms": e2e_ms,
            "text": text,
            "usage": usage,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })
    log_fp.close()
    ended = time.time()
    return {
        "stream_id": sid,
        "video_id": video["video_id"],
        "parent_video_id": video.get("parent_video_id", video["video_id"]),
        "label": video["label"],
        "source": str(src),
        "duration_s": video["duration_s"],
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "window_count": len(starts),
        "started_at": started,
        "ended_at": ended,
        "wall_s": ended - started,
        "returncode": 0,
        "log": str(log_path),
        "per_window": per_window,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--vllm-api-base", default="",
                   help="single vLLM endpoint (for completion baseline)")
    p.add_argument("--vllm-api-base-list", default="",
                   help="comma-separated list of vLLM endpoints (round-robin per worker)")
    p.add_argument("--window-seconds", type=float, default=40.0)
    p.add_argument("--stride-seconds", type=float, default=20.0)
    p.add_argument(
        "--frames-per-window",
        type=int,
        default=80,
        help="number of evenly sampled frames per window; <=0 sends all decoded frames in the window",
    )
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument(
        "--prompt",
        default=(
            "Classify this video clip for surveillance anomaly detection. "
            "Output only Yes or No. Yes means the clip shows one of these "
            "abnormal categories: arrest, arson, assault, burglary, fighting, "
            "road accident, robbery, shooting, shoplifting, stealing, vandalism, "
            "explosion, abuse, or any clearly unsafe/criminal behavior. No means "
            "normal non-criminal activity without visible danger."
        ),
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--pace-realtime", action="store_true",
                   help="wait until each window's end time before sending completion")
    p.add_argument("--request-timeout", type=float, default=180.0)
    p.add_argument("--frame-extract-concurrency", type=int, default=4,
                   help="maximum concurrent local PyAV decode/JPEG extraction jobs")
    p.add_argument("--local-root", type=Path,
                   default=REPO / "edge/data/ucf_eval",
                   help="local root used to remap cloud_path when local_path is absent")
    p.add_argument("--remote-prefix",
                   default="/home/mambauser/tangxuan/ucf_crime_hf",
                   help="remote dataset prefix to replace with --local-root")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cloud-host", default="210.45.123.163")
    p.add_argument("--cloud-ssh-port", type=int, default=2222)
    p.add_argument("--cloud-ssh-user", default="mambauser")
    p.add_argument("--cloud-ssh-key", default=os.path.expanduser("~/.ssh/jupyterhub.pem"))
    p.add_argument(
        "--gpu-monitor",
        action="store_true",
        help="sample remote nvidia-smi during the run and write gpu_smi.csv/gpu_summary.json",
    )
    p.add_argument("--gpu-monitor-gpus", default="0,1,2,3")
    p.add_argument("--gpu-monitor-interval", type=float, default=0.5)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.frame_extract_sem = threading.Semaphore(max(1, args.frame_extract_concurrency))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.stride_seconds <= 0:
        args.stride_seconds = args.window_seconds

    videos = read_manifest(args.manifest)
    for v in videos:
        if v.get("local_path"):
            v["source"] = v["local_path"]
        else:
            cloud_path = str(v["cloud_path"])
            if cloud_path.startswith(args.remote_prefix.rstrip("/") + "/"):
                rel = cloud_path[len(args.remote_prefix.rstrip("/")) + 1:]
                v["source"] = str(args.local_root / rel)
            else:
                v["source"] = cloud_path
    if args.limit:
        videos = videos[: args.limit]
    n = len(videos)

    missing = [v for v in videos if not Path(v["source"]).exists()]
    if missing:
        for v in missing[:5]:
            print(f"[completion] MISSING: {v['source']}")
        return 2

    api_bases = []
    if args.vllm_api_base_list:
        api_bases = [s.strip().rstrip("/") for s in args.vllm_api_base_list.split(",") if s.strip()]
    if not api_bases and args.vllm_api_base:
        api_bases = [args.vllm_api_base.rstrip("/")]
    if not api_bases:
        print("[completion] need --vllm-api-base or --vllm-api-base-list", flush=True)
        return 2

    model = discover_model(api_bases[0])
    print(f"[completion] {n} videos, concurrency={args.concurrency}, model={model}, "
          f"engines={len(api_bases)}, "
          f"frames_per_window={args.frames_per_window}, window_seconds={args.window_seconds}s, "
          f"frame_extract_concurrency={args.frame_extract_concurrency}")

    completed: list[dict] = []
    work_lock = threading.Lock()
    work_idx = [0]
    completed_lock = threading.Lock()

    def worker(slot: int):
        api_base = api_bases[slot % len(api_bases)]
        while True:
            with work_lock:
                if work_idx[0] >= n:
                    return
                vidx = work_idx[0]
                work_idx[0] += 1
            v = videos[vidx]
            rec = process_video(slot, vidx, v, model, api_base, args, out)
            rec["api_base"] = api_base
            with completed_lock:
                completed.append(rec)

    gpu_proc = None
    gpu_summary = None
    gpu_csv = out / "gpu_smi.csv"
    started_at = time.time()
    try:
        if args.gpu_monitor:
            gpu_proc = _start_gpu_monitor(
                cloud_host=args.cloud_host,
                ssh_port=args.cloud_ssh_port,
                ssh_key=args.cloud_ssh_key,
                user=args.cloud_ssh_user,
                gpu_ids=args.gpu_monitor_gpus,
                interval_s=args.gpu_monitor_interval,
                out_csv=gpu_csv,
            )
        workers = [threading.Thread(target=worker, args=(i,), daemon=False)
                   for i in range(args.concurrency)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        _stop_gpu_monitor(gpu_proc)
    ended_at = time.time()
    if args.gpu_monitor:
        gpu_summary = _summarize_gpu_csv(gpu_csv)
        (out / "gpu_summary.json").write_text(json.dumps(gpu_summary, indent=2))

    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_s": ended_at - started_at,
        "n_videos": n,
        "concurrency": args.concurrency,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "frames_per_window": args.frames_per_window,
        "frame_extract_concurrency": args.frame_extract_concurrency,
        "max_tokens": args.max_tokens,
        "pace_realtime": args.pace_realtime,
        "vllm_api_base": args.vllm_api_base,
        "vllm_api_base_list": args.vllm_api_base_list,
        "model": model,
        "prompt": args.prompt,
        "gpu_monitor": {
            "enabled": bool(args.gpu_monitor),
            "gpus": args.gpu_monitor_gpus,
            "interval_s": float(args.gpu_monitor_interval),
            "gpu_smi_path": str(gpu_csv) if args.gpu_monitor else None,
            "gpu_summary_path": str(out / "gpu_summary.json") if args.gpu_monitor else None,
        },
        "streams": sorted(completed, key=lambda r: r["started_at"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    summary = {
        "n_videos": n,
        "wall_s": ended_at - started_at,
        "n_total_windows": sum(len(s["per_window"]) for s in completed),
        "manifest": str(out / "manifest.json"),
        "gpu_summary": gpu_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[completion] DONE: {n} videos in {ended_at-started_at:.1f}s "
          f"({n/(ended_at-started_at)*60:.1f} videos/min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
