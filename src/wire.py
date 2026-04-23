"""Wire protocol for the edge→cloud WebSocket uplink.

Two framings share one WS connection:

  TEXT frames  — JSON control messages (hello, window_open, window_close,
                 rho_update, result, bye). One object per frame.
  BINARY frames — per-packet data: `[4B BE header_len][JSON header][payload]`.
                 The WebSocket frame boundary is the message boundary, so no
                 outer length prefix is needed; the inner 4-byte header length
                 lets the receiver split JSON header from Annex-B payload.

Rationale: JSON control keeps the cloud trivially introspectable during dev;
binary payloads stay untouched H.264 Annex-B for the cloud decoder.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HEADER_LEN_STRUCT = struct.Struct(">I")

MSG_HELLO = "hello"
MSG_PACKET = "packet"
MSG_WINDOW_OPEN = "window_open"
MSG_WINDOW_CLOSE = "window_close"
MSG_RHO_UPDATE = "rho_update"
MSG_RESULT = "result"
MSG_BYE = "bye"


def pack_binary(header: Dict[str, Any], payload: bytes) -> bytes:
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return HEADER_LEN_STRUCT.pack(len(hdr_bytes)) + hdr_bytes + payload


def unpack_binary(frame: bytes) -> Tuple[Dict[str, Any], bytes]:
    if len(frame) < HEADER_LEN_STRUCT.size:
        raise ValueError("binary frame too short for header length prefix")
    (hdr_len,) = HEADER_LEN_STRUCT.unpack_from(frame, 0)
    off = HEADER_LEN_STRUCT.size
    if off + hdr_len > len(frame):
        raise ValueError("binary frame truncated (header)")
    header = json.loads(frame[off:off + hdr_len].decode("utf-8"))
    payload = frame[off + hdr_len:]
    return header, payload


def packet_header(
    seq: int,
    gop_index: int,
    gop_pos: int,
    window_id: int,
    pts_s: Optional[float],
    size: int,
    is_idr: bool,
    slice_type: Optional[str],
    nal_types: List[int],
    score: float,
    gop_mode: str,
    anchor: np.ndarray,
    rho: float,
) -> Dict[str, Any]:
    return {
        "kind": MSG_PACKET,
        "seq": seq,
        "gop_index": gop_index,
        "gop_pos": gop_pos,
        "window_id": window_id,
        "pts_s": pts_s,
        "size": size,
        "is_idr": is_idr,
        "slice_type": slice_type,
        "nal_types": nal_types,
        "score": float(score),
        "gop_mode": gop_mode,
        "anchor": [float(x) for x in anchor.tolist()],
        "rho": float(rho),
    }
