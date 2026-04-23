from dataclasses import dataclass


@dataclass
class EdgeConfig:
    source: str = "rtsp://127.0.0.1:8554/cam"
    rtsp_transport: str = "tcp"
    source_timeout_s: float = 5.0

    window_seconds: float = 4.0

    rho: float = 0.3
    rho_min: float = 0.02
    rho_max: float = 1.0
    min_keep_per_gop: int = 1
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
    model: str = "/home/mambauser/tangxuan/models/Qwen3-VL-8B-Instruct"
    max_tokens: int = 24

    anchor_embed_dim: int = 16

    log_every_packets: int = 100
    codec_hint: str = "h264"

    linger_s: float = 30.0
