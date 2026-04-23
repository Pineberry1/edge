"""Per-packet feature extraction and a rolling window aggregator.

Features live in the compressed domain only (packet size, NAL types, frame type,
GOP position). No pixels. No decode.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from .h264_parser import FrameInfo, extract_frame_info
from .rtsp_source import PacketRecord


@dataclass
class PacketFeatures:
    seq: int
    pts_s: Optional[float]
    size: int
    is_keyframe: bool
    is_idr: bool
    slice_type: Optional[str]
    nal_types: List[int]
    gop_index: int
    gop_pos: int
    size_zscore: float
    size_delta_norm: float
    iat_ms: float


class RollingStats:
    __slots__ = ("cap", "buf", "_sum", "_sq")

    def __init__(self, cap: int = 256) -> None:
        self.cap = cap
        self.buf: Deque[float] = deque(maxlen=cap)
        self._sum = 0.0
        self._sq = 0.0

    def push(self, x: float) -> None:
        if len(self.buf) == self.buf.maxlen:
            old = self.buf[0]
            self._sum -= old
            self._sq -= old * old
        self.buf.append(x)
        self._sum += x
        self._sq += x * x

    def mean(self) -> float:
        n = len(self.buf)
        return self._sum / n if n else 0.0

    def std(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        m = self._sum / n
        var = max(0.0, self._sq / n - m * m)
        return math.sqrt(var)


class FeatureExtractor:
    """Stateful: tracks GOP boundaries, inter-arrival, rolling size stats."""

    def __init__(self, stats_window: int = 256) -> None:
        self._size_stats = RollingStats(cap=stats_window)
        self._gop_index = -1
        self._gop_pos = 0
        self._last_size: Optional[int] = None
        self._last_wall: Optional[float] = None

    def process(self, rec: PacketRecord) -> PacketFeatures:
        info: FrameInfo = extract_frame_info(rec.payload)
        is_idr = info.is_idr or rec.is_keyframe
        if is_idr:
            self._gop_index += 1
            self._gop_pos = 0
        else:
            self._gop_pos += 1

        mean = self._size_stats.mean()
        std = self._size_stats.std()
        size_z = (rec.size - mean) / std if std > 1e-6 else 0.0
        if self._last_size and self._last_size > 0:
            size_delta = abs(rec.size - self._last_size) / float(self._last_size)
        else:
            size_delta = 0.0

        if self._last_wall is not None:
            iat_ms = max(0.0, (rec.wall_time - self._last_wall) * 1000.0)
        else:
            iat_ms = 0.0

        pts_s = rec.pts * rec.time_base if (rec.pts is not None and rec.time_base) else None
        feats = PacketFeatures(
            seq=rec.seq,
            pts_s=pts_s,
            size=rec.size,
            is_keyframe=rec.is_keyframe,
            is_idr=is_idr,
            slice_type=info.slice_type,
            nal_types=info.nal_types,
            gop_index=max(0, self._gop_index),
            gop_pos=self._gop_pos,
            size_zscore=float(size_z),
            size_delta_norm=float(size_delta),
            iat_ms=float(iat_ms),
        )
        self._size_stats.push(float(rec.size))
        self._last_size = rec.size
        self._last_wall = rec.wall_time
        return feats


def anchor_embedding(feats: PacketFeatures, dim: int = 16) -> np.ndarray:
    """Tiny deterministic embedding for semantic alignment on the cloud side.

    This is a placeholder for a learned edge embedding; it packs compressed-domain
    signals into a fixed-size vector that the cloud can consume as a prior.
    """
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0 if feats.is_idr else 0.0
    base[1] = float(feats.slice_type == "P")
    base[2] = float(feats.slice_type == "B")
    base[3] = math.log1p(feats.size) / 12.0
    base[4] = math.tanh(feats.size_zscore)
    base[5] = min(1.0, feats.size_delta_norm)
    base[6] = min(1.0, feats.gop_pos / 60.0)
    base[7] = min(1.0, feats.iat_ms / 200.0)
    for k, nt in enumerate(feats.nal_types[: dim - 8]):
        base[8 + k] = nt / 31.0
    return base
