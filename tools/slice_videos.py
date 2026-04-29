"""Pre-slice mp4 videos into 40s stream-copy chunks (no re-encode).

Given an `eval_videos.tsv` of full videos, split each video into
non-overlapping 40s slices keyed by parent video_id. Outputs a
`slices/<label>/<video_id>__win<NN>.mp4` for each slice and a
`eval_slices.tsv` with rows:

    slice_id  parent_video_id  label  slice_path  window_index  start_s  end_s

Stream-copy means we keep the original H.264 packets (IDR / GOP intact),
so the edge-side ρ packet filter still has work to do at runtime.

Uses pyav (no system ffmpeg). Only writes whole GOPs that start with an IDR
keyframe within each 40s segment, similar to ffmpeg `-c copy -segment_time`.

Usage:
  python -m edge.tools.slice_videos \
      --in-tsv  edge/data/eval_videos.tsv \
      --out-dir edge/data/ucf_full_slices \
      --out-tsv edge/data/eval_slices.tsv \
      --segment-seconds 40
"""
from __future__ import annotations

import argparse
import math
from fractions import Fraction
from pathlib import Path

import av


def read_videos_tsv(path: Path) -> list[dict]:
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


def slice_video(
    src_path: Path,
    label: str,
    video_id: str,
    out_root: Path,
    segment_s: float,
) -> list[dict]:
    """Slice src_path into 40s chunks via stream-copy. Return slice metadata.

    Implementation: walk packets in decode order; whenever a video keyframe
    crosses a segment boundary, finalize the current output and start a new
    one. Within a segment, copy packets verbatim (re-stamping pts to 0).
    """
    in_container = av.open(str(src_path))
    in_video = in_container.streams.video[0]
    in_audio = in_container.streams.audio[0] if in_container.streams.audio else None
    time_base = in_video.time_base or Fraction(1, 1000)
    duration_s = float(in_video.duration * time_base) if in_video.duration else 0.0
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    slices: list[dict] = []
    cur_idx = -1
    out_container = None
    out_video = None
    base_pts = 0  # first packet pts in the current segment

    def open_segment(idx: int):
        nonlocal out_container, out_video
        slice_path = out_dir / f"{video_id}__win{idx:02d}.mp4"
        oc = av.open(str(slice_path), mode="w")
        ov = oc.add_stream_from_template(in_video)
        out_container = oc
        out_video = ov
        return slice_path

    def close_segment(idx: int, start_s: float, end_s: float, slice_path: Path):
        nonlocal out_container
        try:
            out_container.mux(av.Packet(b""))  # flush
        except Exception:
            pass
        out_container.close()
        out_container = None
        slices.append({
            "slice_id": f"{video_id}__win{idx:02d}",
            "parent_video_id": video_id,
            "label": label,
            "slice_path": str(slice_path),
            "window_index": idx,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
        })

    cur_path: Path | None = None
    cur_start_s = 0.0
    last_pts_s = 0.0

    # We only mux video packets (the eval doesn't need audio anyway).
    for packet in in_container.demux(in_video):
        if packet.dts is None:
            continue
        pts_s = float((packet.pts if packet.pts is not None else packet.dts) * time_base)
        last_pts_s = pts_s
        target_idx = int(pts_s // segment_s)
        is_keyframe = bool(packet.is_keyframe)

        # Boundary advance: only on keyframe (so segments are decodable from start).
        if target_idx != cur_idx and is_keyframe:
            if out_container is not None:
                close_segment(cur_idx, cur_start_s, pts_s, cur_path)
            cur_idx = target_idx
            cur_start_s = cur_idx * segment_s
            base_pts = packet.pts if packet.pts is not None else packet.dts
            cur_path = open_segment(cur_idx)
            # rewrite stream time_base to match input
            out_video.time_base = time_base

        if out_container is None:
            # Pre-roll before first IDR — skip (rare, since UCF has GOP=25 with IDR every sec)
            continue

        # rebase pts/dts so each segment starts at 0
        new_pts = (packet.pts - base_pts) if packet.pts is not None else None
        new_dts = (packet.dts - base_pts) if packet.dts is not None else None
        new_packet = av.Packet(bytes(packet))
        new_packet.dts = new_dts
        new_packet.pts = new_pts
        new_packet.time_base = time_base
        new_packet.stream = out_video
        out_container.mux(new_packet)

    if out_container is not None:
        close_segment(cur_idx, cur_start_s, last_pts_s, cur_path)
    in_container.close()
    return slices


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-tsv", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--out-tsv", required=True, type=Path)
    p.add_argument("--segment-seconds", type=float, default=40.0)
    args = p.parse_args()

    videos = read_videos_tsv(args.in_tsv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_slices: list[dict] = []
    for i, v in enumerate(videos):
        src = Path(v.get("local_path") or v["cloud_path"])
        if not src.exists():
            print(f"[slice] MISSING {src}; skip")
            continue
        try:
            slices = slice_video(
                src_path=src,
                label=str(v["label"]),
                video_id=str(v["video_id"]),
                out_root=args.out_dir,
                segment_s=args.segment_seconds,
            )
        except Exception as e:
            print(f"[slice] FAILED {v['video_id']}: {e}")
            continue
        print(f"[slice] [{i+1:>2}/{len(videos)}] {v['label']} {v['video_id']} "
              f"-> {len(slices)} slices")
        all_slices.extend(slices)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w") as f:
        f.write("slice_id\tparent_video_id\tlabel\tslice_path\twindow_index\tstart_s\tend_s\n")
        for s in all_slices:
            f.write(f"{s['slice_id']}\t{s['parent_video_id']}\t{s['label']}\t"
                    f"{s['slice_path']}\t{s['window_index']}\t{s['start_s']}\t{s['end_s']}\n")
    print(f"[slice] wrote {len(all_slices)} slices -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
