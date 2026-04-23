"""Generate a short H.264 test clip with a known GOP structure.

Used for offline testing of the edge pipeline without a real RTSP camera.
Creates 5s @ 25fps with a moving gradient so P/B frames are non-trivial.
"""

from __future__ import annotations

import argparse
import os

import av
import numpy as np


def make_clip(path: str, seconds: float = 5.0, fps: int = 25, width: int = 640, height: int = 360,
              gop: int = 25) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": str(gop), "x264-params": "scenecut=0:keyint_min=" + str(gop)}

    total = int(seconds * fps)
    for i in range(total):
        t = i / fps
        grad = np.zeros((height, width, 3), dtype=np.uint8)
        x_shift = int((i % fps) / fps * width)
        grad[:, :, 0] = (np.arange(width) + x_shift) % 256
        grad[:, :, 1] = (np.arange(height)[:, None] + i * 2) % 256
        grad[:, :, 2] = int(127 + 127 * np.sin(2 * np.pi * t))
        frame = av.VideoFrame.from_ndarray(grad, format="rgb24").reformat(format="yuv420p")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="edge/data/test.mp4")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--gop", type=int, default=25)
    args = p.parse_args(argv)
    make_clip(args.out, args.seconds, args.fps, gop=args.gop)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
