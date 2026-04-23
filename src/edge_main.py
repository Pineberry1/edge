"""Edge pipeline entry point.

  RTSP/file source  →  packet demuxer (no decode)
                    →  H.264 NAL parse + features
                    →  PacketGame-style scoring
                    →  GOP buffer + decoder-safe selection
                    →  WebSocket uplink  →  cloud intake

Windowing: one vLLM online-prefill session per time window. Window id is
`int(pts_s / window_seconds)`; the edge emits `window_open` / `window_close`
control messages around the packets that belong to each window so the cloud
intake can map windows to vLLM sessions 1:1.

No pixel decoding on the edge. Selected raw NAL bytes are relayed unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Optional

from .config import EdgeConfig
from .features import FeatureExtractor, anchor_embedding
from .gop_buffer import GopBatch, GopBuffer
from .rho_state import RhoState
from .rtsp_source import iter_packets
from .scorer import PacketScorer, ScorerWeights
from .uplink import Uplink
from .wire import (
    MSG_WINDOW_CLOSE,
    MSG_WINDOW_OPEN,
    packet_header,
)


def _build_hello(cfg: EdgeConfig, codec_name: str) -> Dict[str, Any]:
    return {
        "stream_id": cfg.stream_id,
        "source": cfg.source,
        "codec_hint": cfg.codec_hint,
        "codec": codec_name,
        "rho": cfg.rho,
        "rho_min": cfg.rho_min,
        "rho_max": cfg.rho_max,
        "window_seconds": cfg.window_seconds,
        "prompt": cfg.prompt,
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "anchor_embed_dim": cfg.anchor_embed_dim,
    }


def _window_id_of(feats, cfg: EdgeConfig, t0: float) -> int:
    if feats.pts_s is not None:
        return int(feats.pts_s / cfg.window_seconds)
    return int((time.time() - t0) / cfg.window_seconds)


def _emit_batch(
    batch: GopBatch,
    window_id: int,
    uplink: Uplink,
    cfg: EdgeConfig,
) -> tuple[int, int]:
    sent = 0
    total_bytes = 0
    for idx in batch.selected_local_indices:
        rec = batch.packets[idx]
        feats = batch.features[idx]
        score = batch.scores[idx]
        anchor = anchor_embedding(feats, dim=cfg.anchor_embed_dim)
        header = packet_header(
            seq=rec.seq,
            gop_index=batch.gop_index,
            gop_pos=feats.gop_pos,
            window_id=window_id,
            pts_s=feats.pts_s,
            size=feats.size,
            is_idr=feats.is_idr,
            slice_type=feats.slice_type,
            nal_types=feats.nal_types,
            score=score,
            gop_mode=batch.decision.mode,
            anchor=anchor,
            rho=batch.decision.rho_used,
        )
        if uplink.send_packet(header, rec.payload):
            sent += 1
            total_bytes += rec.size
    return sent, total_bytes


def _batch_window_id(batch: GopBatch, cfg: EdgeConfig, t0: float) -> int:
    for feats in batch.features:
        if feats.is_idr:
            return _window_id_of(feats, cfg, t0)
    return _window_id_of(batch.features[0], cfg, t0)


def run(cfg: EdgeConfig) -> int:
    rho_state = RhoState(cfg.rho, lo=cfg.rho_min, hi=cfg.rho_max)
    extractor = FeatureExtractor()
    scorer = PacketScorer(
        ScorerWeights(
            i_boost=cfg.score_i_boost,
            size_weight=cfg.score_size_weight,
            novelty_weight=cfg.score_novelty_weight,
            window_alpha=cfg.score_window_alpha,
        )
    )
    gop_buffer = GopBuffer(cfg, scorer_fn=scorer.score, rho_state=rho_state)
    uplink = Uplink(cfg, rho_state)

    codec_name = cfg.codec_hint
    uplink.start(_build_hello(cfg, codec_name))

    seen = 0
    selected = 0
    selected_bytes = 0
    t0 = time.time()
    cur_window: Optional[int] = None

    def open_window(wid: int) -> None:
        uplink.send_control(
            {
                "kind": MSG_WINDOW_OPEN,
                "stream_id": cfg.stream_id,
                "window_id": wid,
                "rho": rho_state.current,
                "opened_at": time.time(),
            }
        )

    def close_window(wid: int) -> None:
        uplink.send_control(
            {
                "kind": MSG_WINDOW_CLOSE,
                "stream_id": cfg.stream_id,
                "window_id": wid,
                "closed_at": time.time(),
            }
        )

    def transition_to(wid: int) -> None:
        nonlocal cur_window
        if cur_window is not None and wid != cur_window:
            close_window(cur_window)
        if cur_window != wid:
            open_window(wid)
            cur_window = wid

    try:
        for rec in iter_packets(cfg.source, cfg.rtsp_transport, cfg.source_timeout_s):
            if seen == 0:
                codec_name = rec.codec_name or codec_name

            feats = extractor.process(rec)
            batch = gop_buffer.push(rec, feats)
            if batch is not None:
                wid = _batch_window_id(batch, cfg, t0)
                transition_to(wid)
                s, b = _emit_batch(batch, wid, uplink, cfg)
                selected += s
                selected_bytes += b

            seen += 1
            if seen % cfg.log_every_packets == 0:
                dt = time.time() - t0
                sys.stderr.write(
                    f"[edge] seen={seen} sel={selected} sel_bytes={selected_bytes/1e6:.2f}MB "
                    f"rho={rho_state.current:.3f} win={cur_window} "
                    f"sent={uplink.sent_packets} dropped={uplink.dropped_packets} "
                    f"rate={seen/max(dt,1e-3):.1f} pps\n"
                )
    except KeyboardInterrupt:
        sys.stderr.write("[edge] interrupted\n")
    finally:
        tail = gop_buffer.flush()
        if tail is not None:
            wid = _batch_window_id(tail, cfg, t0)
            transition_to(wid)
            s, b = _emit_batch(tail, wid, uplink, cfg)
            selected += s
            selected_bytes += b
        if cur_window is not None:
            close_window(cur_window)
        uplink.close(drain_timeout=15.0, linger_s=cfg.linger_s, join_timeout=3.0)

    dt = time.time() - t0
    sys.stderr.write(
        f"[edge] done seen={seen} sel={selected} sel_bytes={selected_bytes/1e6:.2f}MB "
        f"elapsed={dt:.1f}s sent={uplink.sent_packets} dropped={uplink.dropped_packets}\n"
    )
    return 0


def parse_args(argv=None) -> EdgeConfig:
    p = argparse.ArgumentParser(description="PacketGame-style edge packet relay")
    p.add_argument("--source", help="RTSP URL or local file path")
    p.add_argument("--cloud-ws-url", default=None, help="ws://host:port/stream")
    p.add_argument("--rho", type=float, default=None, help="initial per-GOP non-IDR keep ratio")
    p.add_argument("--window-seconds", type=float, default=None)
    p.add_argument("--stream-id", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--rtsp-transport", default=None, choices=["tcp", "udp"])
    p.add_argument(
        "--linger-s",
        type=float,
        default=30.0,
        help="seconds to wait after source EOF so cloud can return inference results",
    )
    args = p.parse_args(argv)

    cfg = EdgeConfig()
    if args.source is not None:
        cfg.source = args.source
    if args.cloud_ws_url is not None:
        cfg.cloud_ws_url = args.cloud_ws_url
    if args.rho is not None:
        cfg.rho = args.rho
    if args.window_seconds is not None:
        cfg.window_seconds = args.window_seconds
    if args.stream_id is not None:
        cfg.stream_id = args.stream_id
    if args.prompt is not None:
        cfg.prompt = args.prompt
    if args.model is not None:
        cfg.model = args.model
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.rtsp_transport is not None:
        cfg.rtsp_transport = args.rtsp_transport
    cfg.linger_s = float(args.linger_s)
    return cfg


def main(argv=None) -> int:
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
