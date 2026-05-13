from edge.src.config import EdgeConfig
from edge.src.edge_main import _build_hello


def test_build_hello_marks_visual_memory_merge_flag() -> None:
    cfg = EdgeConfig()
    assert _build_hello(cfg, "h264")["visual_memory_merge"] is False

    cfg.visual_memory_merge = True
    hello = _build_hello(cfg, "h264")
    assert hello["visual_memory_merge"] is True
