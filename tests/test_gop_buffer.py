from edge.src.config import EdgeConfig
from edge.src.features import PacketFeatures
from edge.src.gop_buffer import GopBuffer
from edge.src.rho_state import RhoState
from edge.src.rtsp_source import PacketRecord


def _packet(seq: int, is_idr: bool) -> PacketRecord:
    return PacketRecord(
        seq=seq,
        pts=seq,
        dts=seq,
        time_base=1.0,
        size=100,
        is_keyframe=is_idr,
        payload=b"idr" if is_idr else b"p",
        wall_time=float(seq),
        stream_index=0,
        codec_name="h264",
    )


def _features(seq: int, is_idr: bool) -> PacketFeatures:
    return PacketFeatures(
        seq=seq,
        pts_s=float(seq),
        size=100,
        is_keyframe=is_idr,
        is_idr=is_idr,
        slice_type="I" if is_idr else "P",
        nal_types=[5 if is_idr else 1],
        gop_index=0 if seq < 4 else 1,
        gop_pos=0 if is_idr else seq,
        size_zscore=0.0,
        size_delta_norm=0.0,
        iat_ms=0.0,
    )


def _push(buf: GopBuffer, seq: int, is_idr: bool):
    return buf.push(_packet(seq, is_idr), _features(seq, is_idr))


def _seqs(batch):
    return [p.seq for p in batch.packets]


def _selected_seqs(batch):
    return [batch.packets[i].seq for i in batch.selected_local_indices]


def test_open_gop_flush_discards_tail_until_next_idr():
    cfg = EdgeConfig(rho=1.0)
    buf = GopBuffer(cfg, scorer_fn=lambda _f: 1.0, rho_state=RhoState(1.0))

    assert _push(buf, 0, True) is None
    assert _push(buf, 1, False) is None

    batch = buf.flush(discard_until_idr=True)
    assert batch is not None
    assert _seqs(batch) == [0, 1]

    assert _push(buf, 2, False) is None
    assert _push(buf, 3, False) is None

    assert _push(buf, 4, True) is None
    assert _push(buf, 5, False) is None

    batch = buf.flush()
    assert batch is not None
    assert _seqs(batch) == [4, 5]


def test_unanchored_non_idr_packets_are_not_emitted():
    cfg = EdgeConfig(rho=1.0)
    buf = GopBuffer(cfg, scorer_fn=lambda _f: 1.0, rho_state=RhoState(1.0))

    assert _push(buf, 0, False) is None
    assert _push(buf, 1, False) is None
    assert buf.flush() is None

    assert _push(buf, 2, True) is None
    batch = buf.flush()
    assert batch is not None
    assert _seqs(batch) == [2]


def test_hard_cap_allows_idr_only_when_rho_is_tiny():
    cfg = EdgeConfig(rho=0.0, rho_min=0.0, min_keep_per_gop=0, rho_hard_cap=True)
    buf = GopBuffer(cfg, scorer_fn=lambda _f: 10.0, rho_state=RhoState(0.001, lo=0.0))

    assert _push(buf, 0, True) is None
    for seq in range(1, 11):
        assert _push(buf, seq, False) is None

    batch = buf.flush()
    assert batch is not None
    assert _selected_seqs(batch) == [0]
    assert batch.decision.mode == "idr_only"
    assert batch.decision.keep_non_idr == 0


def test_hard_cap_does_not_boost_above_rho_budget():
    cfg = EdgeConfig(rho=0.0, rho_min=0.0, min_keep_per_gop=0, rho_hard_cap=True)
    buf = GopBuffer(cfg, scorer_fn=lambda _f: 10.0, rho_state=RhoState(0.25, lo=0.0))

    assert _push(buf, 0, True) is None
    for seq in range(1, 11):
        assert _push(buf, seq, False) is None

    batch = buf.flush()
    assert batch is not None
    assert _selected_seqs(batch) == [0, 1, 2]
    assert batch.decision.mode == "prefix"
    assert batch.decision.keep_non_idr == 2
