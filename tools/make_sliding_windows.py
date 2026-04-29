"""Create fixed-duration sliding MP4 clips and a paired evaluation manifest.

Input TSV:

    video_id label cloud_path duration_s

Output TSV:

    video_id label window_index start_s end_s cloud_path local_path

The script is intentionally path-agnostic: run it on the machine that has
ffmpeg and the input videos. The resulting TSV can later be rewritten to
point local_path at copied clips.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


def _read_videos(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _fmt_s(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _window_starts(duration_s: float, window_s: float, stride_s: float) -> list[float]:
    if duration_s + 1e-6 < window_s:
        return []
    n = int(math.floor((duration_s - window_s) / stride_s + 1e-6)) + 1
    return [round(i * stride_s, 6) for i in range(max(0, n))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=40.0)
    parser.add_argument("--stride-seconds", type=float, default=20.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-videos", type=int, default=0)
    args = parser.parse_args()

    rows = _read_videos(args.videos)
    if args.limit_videos:
        rows = rows[: args.limit_videos]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, str]] = []
    for row in rows:
        video_id = row["video_id"]
        label = row["label"]
        src = Path(row["cloud_path"])
        duration_s = float(row["duration_s"])
        starts = _window_starts(duration_s, args.window_seconds, args.stride_seconds)
        label_dir = args.out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx, start_s in enumerate(starts):
            end_s = start_s + args.window_seconds
            dst = label_dir / f"{video_id}__w{idx:03d}_s{int(round(start_s)):05d}.mp4"
            if args.overwrite or not dst.exists():
                cmd = [
                    args.ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    _fmt_s(start_s),
                    "-t",
                    _fmt_s(args.window_seconds),
                    "-i",
                    str(src),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "veryfast",
                    "-an",
                    str(dst),
                ]
                subprocess.run(cmd, check=True)
            written.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "window_index": str(idx),
                    "start_s": _fmt_s(start_s),
                    "end_s": _fmt_s(end_s),
                    "cloud_path": str(dst),
                    "local_path": str(dst),
                }
            )

    with args.manifest.open("w", newline="") as f:
        fieldnames = [
            "video_id",
            "label",
            "window_index",
            "start_s",
            "end_s",
            "cloud_path",
            "local_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(written)

    print(
        f"[make-windows] videos={len(rows)} windows={len(written)} "
        f"window={args.window_seconds:g}s stride={args.stride_seconds:g}s "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
