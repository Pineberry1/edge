import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class EdgeConfig:
    source: str = "rtsp://127.0.0.1:8554/cam"
    source_list: str = ""
    rtsp_transport: str = "tcp"
    source_timeout_s: float = 5.0

    window_seconds: float = 4.0
    # If > 0, edge emits a separate stream_end control message at this
    # coarser cadence. window_seconds remains the online-prefill chunk
    # granularity.
    decision_window_seconds: float = 0.0

    frame_height: int = field(default_factory=lambda: _env_int("BAVA_FRAME_H", 320))
    frame_width: int = field(default_factory=lambda: _env_int("BAVA_FRAME_W", 240))

    rho: float = 0.3
    rho_min: float = field(default_factory=lambda: _env_float("BAVA_EDGE_RHO_MIN", 0.02))
    rho_max: float = field(default_factory=lambda: _env_float("BAVA_EDGE_RHO_MAX", 1.0))
    min_keep_per_gop: int = field(default_factory=lambda: _env_int("BAVA_MIN_KEEP_PER_GOP", 1))
    rho_hard_cap: bool = field(default_factory=lambda: _env_bool("BAVA_EDGE_RHO_HARD_CAP", False))
    keep_all_idr: bool = True

    score_i_boost: float = 1.5
    score_size_weight: float = 1.0
    score_novelty_weight: float = 0.8
    score_window_alpha: float = 0.2

    cloud_ws_url: str = "ws://127.0.0.1:9100/stream"
    uplink_reconnect_s: float = 1.0
    uplink_queue_max: int = 2048

    stream_id: str = "cam-01"
    prompt: str = "Describe the whole video in one short sentence."
    model: str = "/home/mambauser/tangxuan/models/Qwen3-VL-8B-Instruct-FP8"
    max_tokens: int = 24

    # Initial α that the cloud controller starts from. Mostly useful for
    # static A/B baselines where controller is OFF and α stays at this value.
    alpha: float = 1.0

    anchor_embed_dim: int = 16
    inference_mode: str = "online_prefill"
    visual_memory_merge: bool = False

    log_every_packets: int = 100
    codec_hint: str = "h264"

    linger_s: float = 30.0
    linger_until_results: int = 0
    max_run_seconds: float = 0.0

    pace_realtime: bool = False
    loop_source: bool = False
    align_source_switch_to_decision: bool = field(
        default_factory=lambda: _env_bool("BAVA_ALIGN_SOURCE_SWITCH_TO_DECISION", False)
    )
