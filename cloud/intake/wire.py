"""Cloud-side counterpart of edge/src/wire.py.

Keep this in sync with the edge module - same field names, same binary
framing. We only duplicate because the two sides ship as separate packages.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

HEADER_LEN_STRUCT = struct.Struct(">I")

MSG_HELLO = "hello"
MSG_PACKET = "packet"
MSG_WINDOW_OPEN = "window_open"
MSG_WINDOW_CLOSE = "window_close"
MSG_EDGE_STATS = "edge_stats"
MSG_STREAM_END = "stream_end"
MSG_RHO_UPDATE = "rho_update"
MSG_BUDGET_UPDATE = "budget_update"
MSG_EARLY_FINALIZE = "early_finalize"
MSG_RESULT = "result"
MSG_BYE = "bye"


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


def pack_binary(header: Dict[str, Any], payload: bytes) -> bytes:
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return HEADER_LEN_STRUCT.pack(len(hdr_bytes)) + hdr_bytes + payload
