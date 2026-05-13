"""Edge pipeline entry point.

  RTSP/file source  →  packet demuxer (no decode)
                    →  H.264 NAL parse + features
                    →  PacketGame-style scoring
                    →  GOP buffer + decoder-safe selection
                    →  WebSocket uplink  →  cloud intake

Windowing: `window_seconds` is the small online-prefill chunk cadence. Window
id is `int(pts_s / window_seconds)`; the edge emits `window_open` /
`window_close` around packets that belong to each chunk. A coarser
`decision_window_seconds` cadence emits `stream_end`, telling the cloud to end
the current online-prefill session and run one detection.

No pixel decoding on the edge. Selected raw NAL bytes are relayed unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .budget_state import BudgetState
from .config import EdgeConfig
from .features import FeatureExtractor, anchor_embedding
from .gop_buffer import GopBatch, GopBuffer
from .rho_state import RhoState
from .rtsp_source import PacketRecord, iter_packets
from .scorer import PacketScorer, ScorerWeights
from .uplink import Uplink
from .wire import (
    MSG_STREAM_END,
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


def _read_source_list(path: str) -> List[str]:
    sources: List[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sources.append(line)
    if not sources:
        raise RuntimeError(f"source list is empty: {path}")
    return sources


def _source_paths(cfg: EdgeConfig) -> List[str]:
    if cfg.source_list:
        return _read_source_list(cfg.source_list)
    return [cfg.source]


def _paced_iter(cfg: EdgeConfig) -> Iterator[PacketRecord]:
    """Wrap iter_packets with optional real-time pacing and loop support.

    - Pacing: sleep until wall-clock catches up to packet.pts_s relative to the
      stream start. Mimics a live camera so window_close events arrive on a
      predictable cadence.
    - Playlist: when --source-list is set, play each file in order as one
      continuous camera feed.
    - Looping: when the source or playlist hits EOF, reopen and continue,
      keeping pts_s monotonically increasing by carrying forward an offset.
    """
    sources = _source_paths(cfg)
    wall_start = time.time()
    pts_offset = 0.0
    last_pts = 0.0
    loop = 0
    while not STOP.is_set():
        for source_idx, source in enumerate(sources):
            loop_first_pts: Optional[float] = None
            loop_first_dts: Optional[float] = None
            prev_raw_time: Optional[float] = None
            min_raw_delta: Optional[float] = None
            source_last_time: Optional[float] = None
            for rec in iter_packets(source, cfg.rtsp_transport, cfg.source_timeout_s):
                if STOP.is_set():
                    return
                raw_pts = rec.pts * rec.time_base if (rec.pts is not None and rec.time_base) else None
                raw_dts = rec.dts * rec.time_base if (rec.dts is not None and rec.time_base) else None
                raw_time = raw_dts if raw_dts is not None else raw_pts
                if raw_time is not None:
                    if prev_raw_time is not None:
                        delta = raw_time - prev_raw_time
                        if delta > 1e-6:
                            min_raw_delta = delta if min_raw_delta is None else min(min_raw_delta, delta)
                    prev_raw_time = raw_time

                shifted_pts: Optional[float] = None
                shifted_dts: Optional[float] = None
                if raw_pts is not None:
                    if loop_first_pts is None:
                        loop_first_pts = raw_pts
                    shifted_pts = raw_pts - (loop_first_pts or 0.0) + pts_offset
                if raw_dts is not None:
                    if loop_first_dts is None:
                        loop_first_dts = raw_dts
                    shifted_dts = raw_dts - (loop_first_dts or 0.0) + pts_offset
                if shifted_pts is not None or shifted_dts is not None:
                    rec = dc_replace(
                        rec,
                        pts=(
                            int(shifted_pts / rec.time_base)
                            if shifted_pts is not None and rec.time_base > 0
                            else rec.pts
                        ),
                        dts=(
                            int(shifted_dts / rec.time_base)
                            if shifted_dts is not None and rec.time_base > 0
                            else rec.dts
                        ),
                    )
                    shifted_times = [
                        value for value in (shifted_pts, shifted_dts) if value is not None
                    ]
                    source_last_time = max(shifted_times)
                    last_pts = max(last_pts, source_last_time)
                    if cfg.pace_realtime:
                        due_time = shifted_pts if shifted_pts is not None else shifted_dts or 0.0
                        due = wall_start + due_time
                        slack = due - time.time()
                        if slack > 0:
                            # STOP.wait is interruptible and respects the signal
                            # handler immediately — unlike time.sleep, which
                            # swallows the second Ctrl-C on some platforms.
                            if STOP.wait(timeout=slack):
                                return
                yield rec
            if source_last_time is not None:
                raw_next_offset = source_last_time + (min_raw_delta or 1.0 / 25.0)
                pts_offset = raw_next_offset
                if cfg.source_list and cfg.align_source_switch_to_decision:
                    align_s = max(1e-3, _decision_window_seconds(cfg))
                    pts_offset = math.ceil(max(0.0, raw_next_offset) / align_s) * align_s
                    if pts_offset <= source_last_time:
                        pts_offset += align_s
                if cfg.source_list:
                    align_note = (
                        f" aligned_from={raw_next_offset:.2f}"
                        if abs(pts_offset - raw_next_offset) > 1e-6
                        else ""
                    )
                    sys.stderr.write(
                        f"[edge] source-list advance {source_idx + 1}/{len(sources)} "
                        f"next_pts_offset={pts_offset:.2f}{align_note}\n"
                    )
        if not cfg.loop_source or STOP.is_set():
            return
        pts_offset = last_pts + 1.0 / 25.0  # assume ~25fps spacing between loops
        loop += 1
        sys.stderr.write(f"[edge] source loop {loop}, pts_offset={pts_offset:.2f}\n")


def _build_hello(cfg: EdgeConfig, codec_name: str) -> Dict[str, Any]:
    decision_window_seconds = _decision_window_seconds(cfg)
    frames_per_window = _env_int("BAVA_MAX_FRAMES_PER_WINDOW", 8)
    return {
        "stream_id": cfg.stream_id,
        "source": cfg.source_list or cfg.source,
        "codec_hint": cfg.codec_hint,
        "codec": codec_name,
        "rho": cfg.rho,
        "rho_min": cfg.rho_min,
        "rho_max": cfg.rho_max,
        "alpha": cfg.alpha,
        "window_seconds": cfg.window_seconds,
        "decision_window_seconds": decision_window_seconds,
        "prompt": cfg.prompt,
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "frame_height": cfg.frame_height,
        "frame_width": cfg.frame_width,
        "frames_per_window": frames_per_window,
        "prompt_tokens": max(8, len(cfg.prompt) // 3),
        "anchor_embed_dim": cfg.anchor_embed_dim,
        "inference_mode": cfg.inference_mode,
        "visual_memory_merge": bool(cfg.visual_memory_merge),
        "align_source_switch_to_decision": bool(cfg.align_source_switch_to_decision),
        "max_run_seconds": float(cfg.max_run_seconds),
    }


def _window_id_of(feats, cfg: EdgeConfig, t0: float) -> int:
    if feats.pts_s is not None:
        return int(feats.pts_s / cfg.window_seconds)
    return int((time.time() - t0) / cfg.window_seconds)


def _packet_time_s(rec: PacketRecord, feats, t0: float) -> float:
    # The edge sees packets in decode/arrival order. DTS is monotonic for
    # B-frame streams where PTS can briefly move backwards, so use it for
    # streaming window boundaries when available.
    if rec.dts is not None and rec.time_base:
        return float(rec.dts * rec.time_base)
    if feats.pts_s is not None:
        return float(feats.pts_s)
    return time.time() - t0


def _window_id_at(ts_s: float, cfg: EdgeConfig) -> int:
    return int(ts_s / cfg.window_seconds)


def _decision_window_seconds(cfg: EdgeConfig) -> float:
    return cfg.decision_window_seconds if cfg.decision_window_seconds > 0 else cfg.window_seconds


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _decision_id_at(ts_s: float, cfg: EdgeConfig) -> int:
    return int(ts_s / _decision_window_seconds(cfg))


def _emit_batch(
    batch: GopBatch,
    window_id: int,
    uplink: Uplink,
    cfg: EdgeConfig,
) -> tuple[int, int, int, int]:
    sent = 0
    selected_bytes = 0
    full_bytes = sum(max(0, int(getattr(rec, "size", len(rec.payload)))) for rec in batch.packets)
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
            selected_bytes += rec.size
    return sent, selected_bytes, full_bytes, len(batch.packets)


def _batch_window_id(batch: GopBatch, cfg: EdgeConfig, t0: float) -> int:
    for feats in batch.features:
        if feats.is_idr:
            return _window_id_of(feats, cfg, t0)
    return _window_id_of(batch.features[0], cfg, t0)


def run(cfg: EdgeConfig) -> int:
    rho_state = RhoState(cfg.rho, lo=cfg.rho_min, hi=cfg.rho_max)
    budget_state = BudgetState(
        default_windows=_env_int("BAVA_DEFAULT_WINDOWS_PER_DECISION", 10)
    )
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
    uplink = Uplink(cfg, rho_state, budget_state=budget_state)

    codec_name = cfg.codec_hint
    uplink.start(_build_hello(cfg, codec_name))

    seen = 0
    selected = 0
    selected_bytes = 0
    window_offer_bytes: Dict[int, int] = defaultdict(int)
    window_full_bytes: Dict[int, int] = defaultdict(int)
    window_offer_packets: Dict[int, int] = defaultdict(int)
    window_full_packets: Dict[int, int] = defaultdict(int)
    t0 = time.time()
    cur_window: Optional[int] = None
    cur_decision: Optional[int] = None
    decision_window_count = 0
    next_window_floor = 0
    next_decision_floor = 0
    window_id_offset = 0
    decision_id_offset = 0

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
                "edge_offer_bytes": int(window_offer_bytes.pop(wid, 0)),
                "edge_full_bytes": int(window_full_bytes.pop(wid, 0)),
                "edge_offer_packets": int(window_offer_packets.pop(wid, 0)),
                "edge_full_packets": int(window_full_packets.pop(wid, 0)),
                "edge_queue_size": uplink.queue_size,
                "edge_dropped_packets": uplink.dropped_packets,
                "edge_send_wait_ms": uplink.last_send_wait_ms,
                "edge_send_wait_ms_ewma": uplink.ewma_send_wait_ms,
            }
        )

    def close_decision(decision_id: int, last_window_id: Optional[int]) -> None:
        uplink.send_control(
            {
                "kind": MSG_STREAM_END,
                "stream_id": cfg.stream_id,
                "decision_id": decision_id,
                "last_window_id": last_window_id,
                "decision_window_seconds": _decision_window_seconds(cfg),
                "ended_at": time.time(),
            }
        )

    def emit_tail_for_window(wid: int) -> None:
        nonlocal selected, selected_bytes
        tail = gop_buffer.flush(discard_until_idr=True)
        if tail is None:
            return
        s, b, full_b, full_n = _emit_batch(tail, wid, uplink, cfg)
        selected += s
        selected_bytes += b
        window_offer_bytes[wid] += b
        window_full_bytes[wid] += full_b
        window_offer_packets[wid] += s
        window_full_packets[wid] += full_n

    def force_close_current_decision() -> None:
        nonlocal cur_window, cur_decision, decision_window_count
        nonlocal next_window_floor, next_decision_floor
        nonlocal window_id_offset, decision_id_offset
        if cur_window is None:
            decision_window_count = 0
            budget_state.reset_decision()
            return
        old_window = cur_window
        old_decision = cur_decision
        emit_tail_for_window(old_window)
        close_window(old_window)
        next_window_floor = max(next_window_floor, old_window + 1)
        if old_decision is not None:
            close_decision(old_decision, old_window)
            next_decision_floor = max(next_decision_floor, old_decision + 1)
            decision_id_offset += 1
        window_id_offset += 1
        cur_window = None
        cur_decision = None
        decision_window_count = 0
        budget_state.reset_decision()

    def transition_to(pkt_window: int, pkt_decision: int) -> None:
        nonlocal cur_window, cur_decision, decision_window_count
        nonlocal next_window_floor, next_decision_floor
        nonlocal decision_id_offset
        if cur_window is None:
            pkt_window = max(pkt_window, next_window_floor)
            pkt_decision = max(pkt_decision, next_decision_floor)
            open_window(pkt_window)
            cur_window = pkt_window
            cur_decision = pkt_decision
            return
        if pkt_window == cur_window and pkt_decision == cur_decision:
            return

        old_window = cur_window
        old_decision = cur_decision
        emit_tail_for_window(old_window)
        close_window(old_window)
        decision_window_count += 1
        next_window_floor = max(next_window_floor, old_window + 1)
        budget_reached = budget_state.consume_window(decision_window_count)
        decision_changed = old_decision is not None and pkt_decision != old_decision
        if old_decision is not None and (decision_changed or budget_reached):
            close_decision(old_decision, old_window)
            next_decision_floor = max(next_decision_floor, old_decision + 1)
            if budget_reached and not decision_changed:
                decision_id_offset += 1
            decision_window_count = 0
            budget_state.reset_decision()

        pkt_window = max(pkt_window, next_window_floor)
        pkt_decision = max(pkt_decision, next_decision_floor)
        open_window(pkt_window)
        cur_window = pkt_window
        cur_decision = pkt_decision

    try:
        for rec in _paced_iter(cfg):
            if STOP.is_set():
                break
            if cfg.max_run_seconds > 0 and (time.time() - t0) >= cfg.max_run_seconds:
                sys.stderr.write(
                    f"[edge] max_run_seconds={cfg.max_run_seconds:.1f} reached; "
                    "closing gracefully\n"
                )
                break
            force_event = budget_state.consume_force_close()
            if force_event is not None:
                sys.stderr.write(
                    "[edge] cloud early_finalize force boundary "
                    f"decision={force_event.get('decision_id')} "
                    f"reason={force_event.get('reason')}\n"
                )
                force_close_current_decision()
            if (
                budget_state.changed_since_last_check
                and cur_window is not None
                and decision_window_count >= budget_state.windows_per_decision
            ):
                force_close_current_decision()
            if seen == 0:
                codec_name = rec.codec_name or codec_name

            feats = extractor.process(rec)
            pkt_ts_s = _packet_time_s(rec, feats, t0)
            pkt_window = _window_id_at(pkt_ts_s, cfg) + window_id_offset
            pkt_decision = _decision_id_at(pkt_ts_s, cfg) + decision_id_offset
            pkt_window = max(pkt_window, next_window_floor)
            pkt_decision = max(pkt_decision, next_decision_floor)
            if cur_window is not None and pkt_window < cur_window:
                pkt_window = cur_window
            if cur_decision is not None and pkt_decision < cur_decision:
                pkt_decision = cur_decision
            transition_to(pkt_window, pkt_decision)

            batch = gop_buffer.push(rec, feats)
            if batch is not None:
                wid = cur_window if cur_window is not None else _batch_window_id(batch, cfg, t0)
                s, b, full_b, full_n = _emit_batch(batch, wid, uplink, cfg)
                selected += s
                selected_bytes += b
                window_offer_bytes[wid] += b
                window_full_bytes[wid] += full_b
                window_offer_packets[wid] += s
                window_full_packets[wid] += full_n

            seen += 1
            if seen % cfg.log_every_packets == 0:
                dt = time.time() - t0
                sys.stderr.write(
                    f"[edge] seen={seen} sel={selected} sel_bytes={selected_bytes/1e6:.2f}MB "
                    f"rho={rho_state.current:.3f} win={cur_window} "
                    f"budget={budget_state.windows_per_decision} "
                    f"sent={uplink.sent_packets} dropped={uplink.dropped_packets} "
                    f"rate={seen/max(dt,1e-3):.1f} pps\n"
                )
    except KeyboardInterrupt:
        STOP.set()
        sys.stderr.write("[edge] interrupted\n")
    finally:
        tail = gop_buffer.flush()
        if tail is not None:
            wid = cur_window if cur_window is not None else _batch_window_id(tail, cfg, t0)
            s, b, full_b, full_n = _emit_batch(tail, wid, uplink, cfg)
            selected += s
            selected_bytes += b
            window_offer_bytes[wid] += b
            window_full_bytes[wid] += full_b
            window_offer_packets[wid] += s
            window_full_packets[wid] += full_n
        if cur_window is not None:
            close_window(cur_window)
        if cur_decision is not None:
            close_decision(cur_decision, cur_window)
        # If we stopped by signal, shorten both drain and linger so bench
        # harness shutdowns don't block waiting on results that aren't coming.
        if STOP.is_set():
            uplink.close(drain_timeout=1.5, linger_s=0.2, join_timeout=2.0)
        else:
            uplink.close(
                drain_timeout=15.0,
                linger_s=cfg.linger_s,
                join_timeout=3.0,
                linger_until_results=cfg.linger_until_results,
            )

    dt = time.time() - t0
    sys.stderr.write(
        f"[edge] done seen={seen} sel={selected} sel_bytes={selected_bytes/1e6:.2f}MB "
        f"elapsed={dt:.1f}s sent={uplink.sent_packets} dropped={uplink.dropped_packets}\n"
    )
    return 0


def parse_args(argv=None) -> EdgeConfig:
    p = argparse.ArgumentParser(description="PacketGame-style edge packet relay")
    p.add_argument("--source", help="RTSP URL or local file path")
    p.add_argument("--source-list", default=None, help="text file of local file paths to play as one camera")
    p.add_argument("--cloud-ws-url", default=None, help="ws://host:port/stream")
    p.add_argument("--rho", type=float, default=None, help="initial per-GOP non-IDR keep ratio")
    p.add_argument("--alpha", type=float, default=None, help="initial cloud α (token keep rate) — static baselines only")
    p.add_argument("--window-seconds", type=float, default=None)
    p.add_argument("--frame-height", type=int, default=None)
    p.add_argument("--frame-width", type=int, default=None)
    p.add_argument(
        "--decision-window-seconds",
        type=float,
        default=None,
        help="coarser edge-side stream_end/detection cadence; defaults to --window-seconds",
    )
    p.add_argument("--stream-id", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument(
        "--inference-mode",
        choices=["online_prefill", "completion"],
        default=None,
        help="cloud intake mode; completion keeps H.264 transport but uses one chat-completion request at stream_end",
    )
    p.add_argument(
        "--visual-memory-merge",
        action="store_true",
        help="ask intake to export visual memory and warm prefix cache for this stream",
    )
    p.add_argument(
        "--completion-mode",
        action="store_true",
        help="shortcut for --inference-mode completion",
    )
    p.add_argument("--rtsp-transport", default=None, choices=["tcp", "udp"])
    p.add_argument(
        "--linger-s",
        type=float,
        default=30.0,
        help="seconds to wait after source EOF so cloud can return inference results",
    )
    p.add_argument(
        "--linger-until-results",
        type=int,
        default=None,
        help="during normal close, stop lingering early after receiving this many result messages",
    )
    p.add_argument(
        "--max-run-seconds",
        type=float,
        default=None,
        help=(
            "wall-clock input horizon for this edge process; when reached, "
            "the current window/decision is closed and normal linger/drain is used"
        ),
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
    p.add_argument(
        "--align-source-switch-to-decision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="when using --source-list, start each next source at the next decision boundary",
    )
    args = p.parse_args(argv)

    cfg = EdgeConfig()
    if args.source_list is not None:
        cfg.source_list = args.source_list
        try:
            cfg.source = _read_source_list(args.source_list)[0]
        except Exception:
            cfg.source = args.source_list
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
    if args.frame_height is not None:
        cfg.frame_height = args.frame_height
    if args.frame_width is not None:
        cfg.frame_width = args.frame_width
    if args.decision_window_seconds is not None:
        cfg.decision_window_seconds = args.decision_window_seconds
    if args.align_source_switch_to_decision is not None:
        cfg.align_source_switch_to_decision = bool(args.align_source_switch_to_decision)
    if args.stream_id is not None:
        cfg.stream_id = args.stream_id
    if args.prompt is not None:
        cfg.prompt = args.prompt
    if args.model is not None:
        cfg.model = args.model
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.inference_mode is not None:
        cfg.inference_mode = args.inference_mode
    if args.completion_mode:
        cfg.inference_mode = "completion"
    cfg.visual_memory_merge = bool(args.visual_memory_merge)
    if args.rtsp_transport is not None:
        cfg.rtsp_transport = args.rtsp_transport
    cfg.linger_s = float(args.linger_s)
    if args.linger_until_results is not None:
        cfg.linger_until_results = max(0, int(args.linger_until_results))
    if args.max_run_seconds is not None:
        cfg.max_run_seconds = float(args.max_run_seconds)
    cfg.pace_realtime = bool(args.pace_realtime)
    cfg.loop_source = bool(args.loop_source)
    return cfg


def main(argv=None) -> int:
    _install_signal_handlers()
    cfg = parse_args(argv)
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
