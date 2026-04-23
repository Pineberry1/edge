"""Minimal cloud-side receiver for validating the edge uplink end-to-end.

Accepts one edge connection, reassembles per-GOP mp4/h264 files, and prints
rolling stats. This stands in for the real cloud serving stack; the same wire
format is what a production vLLM-side visual ingest would consume.

Usage:
    python -m edge.tools.cloud_sink --port 9000 --out-dir /tmp/cloud_gops
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.wire import recv_message, MSG_HELLO, MSG_PACKET, MSG_BYE  # noqa: E402


START_CODE = b"\x00\x00\x00\x01"


def _needs_annexb(payload: bytes) -> bool:
    return not (
        payload.startswith(b"\x00\x00\x01") or payload.startswith(b"\x00\x00\x00\x01")
    )


def _to_annexb(payload: bytes) -> bytes:
    if _needs_annexb(payload):
        return START_CODE + payload
    return payload


def run(host: str, port: int, out_dir: Optional[str]) -> int:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    sys.stderr.write(f"[cloud] listening on {host}:{port}\n")

    conn, addr = srv.accept()
    sys.stderr.write(f"[cloud] accepted {addr}\n")
    srv.close()

    t0 = time.time()
    total_packets = 0
    total_bytes = 0
    per_gop_counts: Dict[int, int] = {}
    per_gop_bytes: Dict[int, int] = {}
    per_gop_file: Dict[int, object] = {}
    hello: Dict[str, object] = {}

    try:
        while True:
            header, payload = recv_message(conn)
            kind = header.get("kind")
            if kind == MSG_HELLO:
                hello = header
                sys.stderr.write(f"[cloud] hello: {header}\n")
                continue
            if kind == MSG_BYE:
                sys.stderr.write("[cloud] bye\n")
                break
            if kind != MSG_PACKET:
                sys.stderr.write(f"[cloud] unknown msg kind: {kind!r}\n")
                continue

            total_packets += 1
            total_bytes += len(payload)
            g = int(header.get("gop_index", 0))
            per_gop_counts[g] = per_gop_counts.get(g, 0) + 1
            per_gop_bytes[g] = per_gop_bytes.get(g, 0) + len(payload)

            if out_dir:
                if g not in per_gop_file:
                    fp = open(os.path.join(out_dir, f"gop_{g:06d}.h264"), "wb")
                    per_gop_file[g] = fp
                per_gop_file[g].write(_to_annexb(payload))

            if total_packets % 50 == 0:
                dt = time.time() - t0
                sys.stderr.write(
                    f"[cloud] rx packets={total_packets} bytes={total_bytes/1e6:.2f}MB "
                    f"gops={len(per_gop_counts)} rate={total_packets/max(dt,1e-3):.1f} pps\n"
                )
    except ConnectionError:
        sys.stderr.write("[cloud] connection closed by peer\n")
    finally:
        for fp in per_gop_file.values():
            try:
                fp.close()
            except OSError:
                pass
        conn.close()

    dt = time.time() - t0
    sys.stderr.write(
        f"[cloud] done packets={total_packets} bytes={total_bytes/1e6:.2f}MB gops={len(per_gop_counts)} "
        f"elapsed={dt:.1f}s\n"
    )
    if per_gop_counts:
        widths = sorted(per_gop_counts.items())[:5]
        sys.stderr.write(f"[cloud] first gops: {widths}\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Minimal cloud receiver for edge uplink")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--out-dir", default=None, help="if set, dump per-GOP Annex-B files")
    args = p.parse_args(argv)
    return run(args.host, args.port, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
