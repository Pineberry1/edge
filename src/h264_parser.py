"""Minimal H.264 Annex-B / AVCC NAL unit walker + slice_type extraction.

We intentionally do NOT decode pixels. We only parse NAL headers and the first
few exp-Golomb codes of the slice header to recover:
    - NAL unit type (IDR vs non-IDR slice vs SPS/PPS/...)
    - slice_type (I / P / B / SP / SI)

Following PacketGame (SIGCOMM '23): this is enough signal (together with packet
size + GOP position) for a lightweight importance scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

NAL_TYPE_NON_IDR_SLICE = 1
NAL_TYPE_IDR_SLICE = 5
NAL_TYPE_SEI = 6
NAL_TYPE_SPS = 7
NAL_TYPE_PPS = 8
NAL_TYPE_AUD = 9

_SLICE_TYPE_NAMES = {0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI"}


@dataclass
class NalUnit:
    nal_type: int
    nal_ref_idc: int
    rbsp: bytes
    offset: int
    length: int


@dataclass
class FrameInfo:
    is_idr: bool
    slice_type: Optional[str]
    nal_types: List[int]


def _find_start_codes(buf: bytes) -> List[Tuple[int, int]]:
    """Return list of (start_of_nal_payload, end_exclusive) for Annex-B NALs."""
    n = len(buf)
    positions: List[int] = []
    i = 0
    while i < n - 3:
        if buf[i] == 0 and buf[i + 1] == 0:
            if buf[i + 2] == 1:
                positions.append(i + 3)
                i += 3
                continue
            if i + 3 < n and buf[i + 2] == 0 and buf[i + 3] == 1:
                positions.append(i + 4)
                i += 4
                continue
        i += 1
    if not positions:
        return []
    spans: List[Tuple[int, int]] = []
    for k, start in enumerate(positions):
        end = positions[k + 1] - 3 if k + 1 < len(positions) else n
        if end > start:
            spans.append((start, end))
    return spans


def _strip_emulation_prevention(ebsp: bytes) -> bytes:
    """Remove 0x03 emulation-prevention bytes to recover RBSP."""
    out = bytearray()
    i = 0
    n = len(ebsp)
    while i < n:
        if i + 2 < n and ebsp[i] == 0 and ebsp[i + 1] == 0 and ebsp[i + 2] == 0x03:
            out.append(0)
            out.append(0)
            i += 3
        else:
            out.append(ebsp[i])
            i += 1
    return bytes(out)


class _BitReader:
    __slots__ = ("buf", "bit_pos", "bit_len")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.bit_pos = 0
        self.bit_len = len(buf) * 8

    def read_bit(self) -> int:
        if self.bit_pos >= self.bit_len:
            return 0
        byte = self.buf[self.bit_pos >> 3]
        bit = (byte >> (7 - (self.bit_pos & 7))) & 1
        self.bit_pos += 1
        return bit

    def read_ue(self) -> int:
        zeros = 0
        while self.bit_pos < self.bit_len and self.read_bit() == 0:
            zeros += 1
            if zeros > 31:
                return 0
        value = 0
        for _ in range(zeros):
            value = (value << 1) | self.read_bit()
        return (1 << zeros) - 1 + value


def iter_nal_units(payload: bytes) -> Iterator[NalUnit]:
    spans = _find_start_codes(payload)
    if not spans:
        return
    for start, end in spans:
        if start >= end:
            continue
        header = payload[start]
        nal_type = header & 0x1F
        nal_ref_idc = (header >> 5) & 0x03
        rbsp = _strip_emulation_prevention(payload[start + 1:end])
        yield NalUnit(
            nal_type=nal_type,
            nal_ref_idc=nal_ref_idc,
            rbsp=rbsp,
            offset=start,
            length=end - start,
        )


def parse_slice_type(rbsp: bytes) -> Optional[str]:
    """Parse first_mb_in_slice then slice_type from a slice NAL's RBSP."""
    if not rbsp:
        return None
    br = _BitReader(rbsp)
    try:
        _first_mb = br.read_ue()
        slice_type_raw = br.read_ue()
    except Exception:
        return None
    slice_type = slice_type_raw % 5
    return _SLICE_TYPE_NAMES.get(slice_type)


def extract_frame_info(payload: bytes) -> FrameInfo:
    """Summarize a compressed video access unit into a FrameInfo."""
    nal_types: List[int] = []
    is_idr = False
    slice_type: Optional[str] = None
    for nu in iter_nal_units(payload):
        nal_types.append(nu.nal_type)
        if nu.nal_type == NAL_TYPE_IDR_SLICE:
            is_idr = True
            if slice_type is None:
                slice_type = parse_slice_type(nu.rbsp) or "I"
        elif nu.nal_type == NAL_TYPE_NON_IDR_SLICE and slice_type is None:
            slice_type = parse_slice_type(nu.rbsp)
    if is_idr and slice_type is None:
        slice_type = "I"
    return FrameInfo(is_idr=is_idr, slice_type=slice_type, nal_types=nal_types)
