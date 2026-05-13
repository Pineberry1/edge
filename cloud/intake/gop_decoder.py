"""Decode H.264 Annex-B byte streams to BGR ndarrays, then (separately)
encode to the JPEG-base64 form vLLM's `/v1/online_prefill/*/append` expects.

Separation lets us insert the α executor between decode and encode without
re-decoding frames. `decode_to_bgr` stays pure (no JPEG work); `encode_bgr_to_jpeg_data_uri`
is a tiny helper used by the window assembler.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np


@dataclass
class BgrFrame:
    index: int
    pts_s: Optional[float]
    bgr: np.ndarray


@dataclass
class EncodedFrame:
    index: int
    pts_s: Optional[float]
    jpeg_b64: str
    width: int
    height: int


def decode_to_bgr(data: bytes, max_frames: Optional[int] = None) -> Tuple[List[BgrFrame], float]:
    """Decode a concatenated Annex-B bytestream into BGR frames."""
    if not data:
        return [], 0.0
    started = time.perf_counter()
    buf = io.BytesIO(data)
    frames: List[BgrFrame] = []
    try:
        container = av.open(buf, format="h264")
    except av.AVError as e:
        raise RuntimeError(f"pyav open failed: {e}") from e
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else 0.0
        idx = 0
        for frame in container.decode(stream):
            if max_frames is not None and idx >= max_frames:
                break
            bgr = frame.to_ndarray(format="bgr24")
            pts_s = (float(frame.pts) * tb) if (frame.pts is not None and tb) else None
            frames.append(BgrFrame(index=idx, pts_s=pts_s, bgr=bgr))
            idx += 1
    finally:
        container.close()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return frames, elapsed_ms


def encode_bgr_to_jpeg_data_uri(bgr: np.ndarray, quality: int = 95) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def subsample_uniform(frames: List[BgrFrame], keep: int) -> List[BgrFrame]:
    """Keep up to `keep` frames uniformly across the list."""
    n = len(frames)
    if keep >= n or keep <= 0:
        return list(frames)
    step = n / float(keep)
    picks: List[BgrFrame] = []
    used = set()
    for i in range(keep):
        j = min(n - 1, int(i * step))
        if j in used:
            j = min(n - 1, j + 1)
        used.add(j)
        picks.append(frames[j])
    return picks
