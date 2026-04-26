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
import signal
import sys
import threading
import time
from dataclasses import replace as dc_replace
from typing import Any, Dict, Iterator, Optional

from .config import EdgeConfig
from .features import FeatureExtractor, anchor_embedding
from .gop_buffer import GopBatch, GopBuffer
from .rho_state import RhoState
from .rtsp_source import PacketRecord, iter_packets
from .scorer import PacketScorer, ScorerWeights
from .uplink import Uplink
from .wire import (
    MSG_WINDOW_CLOSE,
    MSG_WINDOW_OPEN,
    packet_header,
)


STOP = threading.Event()


def _install_signal_handlers() -> None:
    """Flip the module-level STOP flag on SIGINT/SIGTERM so the main loop and
    the pacing/loop-source helpers can exit on their own terms. The bench
    harness uses SIGTERM; operator Ctrl-C uses SIGINT; both should behave
    identically — cleanly break out of whichever loop is current.

    Re-entrant: a second signal within 2s escalates to SystemExit(130) so a
    stuck `time.sleep` inside a thread can still be killed by pressing Ctrl-C
    twice.
    """
    last = [0.0]

    def _handler(signum, _frame):
        now = time.time()
        if STOP.is_set() and (now - last[0]) < 2.0:
            sys.stderr.write(f"[edge] signal {signum} again — hard exit\n")
            sys.exit(130)
        STOP.set()
        last[0] = now
        sys.stderr.write(f"[edge] signal {signum} received, stopping\n")

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _paced_iter(cfg: EdgeConfig) -> Iterator[PacketRecord]:
    """Wrap iter_packets with optional real-time pacing and loop support.

    - Pacing: sleep until wall-clock catches up to packet.pts_s relative to the
      stream start. Mimics a live camera so window_close events arrive on a
      predictable cadence.
    - Looping: when the source hits EOF, reopen and continue, keeping pts_s
      monotonically increasing by carrying forward an offset.
    """
    wall_start = time.time()
    pts_offset = 0.0
    last_pts = 0.0
    loop = 0
    while not STOP.is_set():
        loop_first_pts: Optional[float] = None
        for rec in iter_packets(cfg.source, cfg.rtsp_transport, cfg.source_timeout_s):
            if STOP.is_set():
                return
            raw_pts = rec.pts * rec.time_base if (rec.pts is not None and rec.time_base) else None
            if raw_pts is not None:
                if loop_first_pts is None:
                    loop_first_pts = raw_pts
                shifted = raw_pts - (loop_first_pts or 0.0) + pts_offset
                rec = dc_replace(
                    rec,
                    pts=int(shifted / rec.time_base) if rec.time_base > 0 else rec.pts,
                )
                last_pts = shifted
                if cfg.pace_realtime:
                    due = wall_start + shifted
                    slack = due - time.time()
                    if slack > 0:
                        # STOP.wait is interruptible and respects the signal
                        # handler immediately — unlike time.sleep, which
                        # swallows the second Ctrl-C on some platforms.
                        if STOP.wait(timeout=slack):
                            return
            yield rec
        if not cfg.loop_source or STOP.is_set():
            return
        pts_offset = last_pts + 1.0 / 25.0  # assume ~25fps spacing between loops
        loop += 1
        sys.stderr.write(f"[edge] source loop {loop}, pts_offset={pts_offset:.2f}\n")


def _build_hello(cfg: EdgeConfig, codec_name: str) -> Dict[str, Any]:
    return {
        "stream_id": cfg.stream_id,
        "source": cfg.source,
        "codec_hint": cfg.codec_hint,
        "codec": codec_name,
        "rho": cfg.rho,
        "rho_min": cfg.rho_min,
        "rho_max": cfg.rho_max,
        "alpha": cfg.alpha,
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
        for rec in _paced_iter(cfg):
            if STOP.is_set():
                break
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
        STOP.set()
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
        # If we stopped by signal, shorten both drain and linger so bench
        # harness shutdowns don't block waiting on results that aren't coming.
        if STOP.is_set():
            uplink.close(drain_timeout=1.5, linger_s=0.2, join_timeout=2.0)
        else:
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
    p.add_argument("--alpha", type=float, default=None, help="initial cloud α (token keep rate) — static baselines only")
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
    p.add_argument(
        "--pace-realtime",
        action="store_true",
        help="throttle packet emission to each packet's pts_s (simulates a live camera)",
    )
    p.add_argument(
        "--loop-source",
        action="store_true",
        help="loop the source forever — useful for long-running benchmarks on short clips",
    )
    args = p.parse_args(argv)

    cfg = EdgeConfig()
    if args.source is not None:
        cfg.source = args.source
    if args.cloud_ws_url is not None:
        cfg.cloud_ws_url = args.cloud_ws_url
    if args.rho is not None:
        cfg.rho = args.rho
    if args.alpha is not None:
        cfg.alpha = args.alpha
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
    cfg.pace_realtime = bool(args.pace_realtime)
    cfg.loop_source = bool(args.loop_source)
    return cfg


def main(argv=None) -> int:
    _install_signal_handlers()
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
