"""Packet-level source reader. Demuxes RTSP or a local file WITHOUT decoding.

Yields a PacketRecord per video packet. The raw bitstream bytes are preserved so
the cloud side (or a later GPU decoder on the edge) can decode only selected
packets.

All payloads emitted here are normalized to H.264/H.265 Annex-B (start-code
delimited). MP4/MOV containers use AVCC length-prefixed NALs; we transparently
convert once at demux so downstream code sees a single format.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import av

START_CODE = b"\x00\x00\x00\x01"


@dataclass
class PacketRecord:
    seq: int
    pts: Optional[int]
    dts: Optional[int]
    time_base: float
    size: int
    is_keyframe: bool
    payload: bytes
    wall_time: float
    stream_index: int
    codec_name: str


def _detect_avcc_length_size(extradata: Optional[bytes]) -> Optional[int]:
    """If extradata is AVCC/HVCC, return the NAL length field size (1/2/4)."""
    if not extradata or len(extradata) < 5:
        return None
    if extradata[0] != 0x01:
        return None
    return (extradata[4] & 0x03) + 1


def _avcc_to_annexb(payload: bytes, length_size: int) -> bytes:
    out = bytearray()
    i = 0
    n = len(payload)
    while i + length_size <= n:
        if length_size == 4:
            (nal_len,) = struct.unpack_from(">I", payload, i)
        elif length_size == 2:
            (nal_len,) = struct.unpack_from(">H", payload, i)
        elif length_size == 1:
            nal_len = payload[i]
        else:
            return payload
        i += length_size
        if nal_len <= 0 or i + nal_len > n:
            break
        out += START_CODE
        out += payload[i:i + nal_len]
        i += nal_len
    return bytes(out) if out else payload


def _prepend_extradata_as_annexb(extradata: Optional[bytes], length_size: Optional[int]) -> bytes:
    """Convert AVCC extradata (SPS/PPS arrays) to Annex-B prefix bytes.

    Real RTSP cameras deliver SPS/PPS inline; MP4 containers park them in
    extradata only. We emit them ahead of the first keyframe so the cloud
    decoder can initialize without the container's codec config.
    """
    if not extradata or length_size is None or len(extradata) < 7:
        return b""
    i = 5
    num_sps = extradata[i] & 0x1F
    i += 1
    out = bytearray()
    for _ in range(num_sps):
        if i + 2 > len(extradata):
            return bytes(out)
        sps_len = struct.unpack_from(">H", extradata, i)[0]
        i += 2
        out += START_CODE + extradata[i:i + sps_len]
        i += sps_len
    if i >= len(extradata):
        return bytes(out)
    num_pps = extradata[i]
    i += 1
    for _ in range(num_pps):
        if i + 2 > len(extradata):
            return bytes(out)
        pps_len = struct.unpack_from(">H", extradata, i)[0]
        i += 2
        out += START_CODE + extradata[i:i + pps_len]
        i += pps_len
    return bytes(out)


def open_source(source: str, rtsp_transport: str = "tcp", timeout_s: float = 5.0):
    """Open an RTSP URL or a local file. No decoding is performed."""
    options = {}
    if source.lower().startswith("rtsp://"):
        options["rtsp_transport"] = rtsp_transport
        options["stimeout"] = str(int(timeout_s * 1_000_000))
    container = av.open(source, options=options, timeout=timeout_s)
    video_streams = [s for s in container.streams if s.type == "video"]
    if not video_streams:
        container.close()
        raise RuntimeError(f"No video stream in source: {source}")
    stream = video_streams[0]
    stream.thread_type = "NONE"
    return container, stream


def iter_packets(
    source: str,
    rtsp_transport: str = "tcp",
    timeout_s: float = 5.0,
) -> Iterator[PacketRecord]:
    container, stream = open_source(source, rtsp_transport, timeout_s)
    tb = float(stream.time_base) if stream.time_base else 0.0
    codec_name = stream.codec_context.name if stream.codec_context else "unknown"
    extradata = bytes(stream.codec_context.extradata) if stream.codec_context.extradata else b""
    length_size = _detect_avcc_length_size(extradata)
    param_sets_annexb = _prepend_extradata_as_annexb(extradata, length_size)
    seq = 0
    try:
        for packet in container.demux(stream):
            if packet.size == 0 or packet.buffer_size == 0:
                continue
            raw = bytes(packet)
            if length_size is not None:
                raw = _avcc_to_annexb(raw, length_size)
            if packet.is_keyframe and param_sets_annexb:
                raw = param_sets_annexb + raw
            yield PacketRecord(
                seq=seq,
                pts=packet.pts,
                dts=packet.dts,
                time_base=tb,
                size=len(raw),
                is_keyframe=bool(packet.is_keyframe),
                payload=raw,
                wall_time=time.time(),
                stream_index=packet.stream.index,
                codec_name=codec_name,
            )
            seq += 1
    finally:
        container.close()
