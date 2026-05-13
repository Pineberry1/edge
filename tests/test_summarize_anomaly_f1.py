import json

from edge.tools.summarize_anomaly_f1 import summarize_config


def write_config(tmp_path, chunks0, chunks1=None, texts=None):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    streams = [
        {
            "stream_id": "v-000",
            "video_id": "video_a",
            "parent_video_id": "video_a",
            "label": "anomaly",
            "window_seconds": 4.0,
            "decision_window_seconds": 40.0,
        }
    ]
    (cfg / "manifest.json").write_text(json.dumps({"streams": streams}))
    texts = texts or {0: "Yes", 1: "Yes"}
    lines = [f"[edge-uplink] result window={wid} text='{text}'" for wid, text in texts.items()]
    (cfg / "edge-v-000.log").write_text("\n".join(lines) + "\n")
    intake_lines = [
        f"INFO intake.window stream=v-000 engine=0 decision=0 done frames=10 chunks={chunks0} side≈448 append_ms=1 final_ms=1 text='Yes'"
    ]
    if chunks1 is not None:
        intake_lines.append(
            f"INFO intake.window stream=v-000 engine=0 decision=1 done frames=10 chunks={chunks1} side≈448 append_ms=1 final_ms=1 text='Yes'"
        )
    (cfg / "intake.log").write_text("\n".join(intake_lines) + "\n")
    return cfg


def test_yes_chunks10_requires_ten_yes_chunks(tmp_path):
    cfg = write_config(tmp_path, "[0, 1]", "[2, 3, 4, 5, 6, 7, 8]")

    report = summarize_config(cfg, "yes_chunks10")
    row = report["per_video"][0]

    assert row["n_total_yes"] == 2
    assert row["n_total_yes_chunks"] == 9
    assert row["predicted_positive"] is False
    assert report["FN"] == 1


def test_yes_chunks10_accepts_accumulated_ten_yes_chunks(tmp_path):
    cfg = write_config(tmp_path, "[0, 1]", "[2, 3, 4, 5, 6, 7, 8, 9]")

    report = summarize_config(cfg, "yes_chunks10")
    row = report["per_video"][0]

    assert row["n_total_yes_chunks"] == 10
    assert row["predicted_positive"] is True
    assert report["TP"] == 1


def test_yes_chunks10_fallback_preserves_full_window_any_behavior(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    streams = [
        {
            "stream_id": "v-000",
            "video_id": "video_a",
            "parent_video_id": "video_a",
            "label": "anomaly",
            "window_seconds": 4.0,
            "decision_window_seconds": 40.0,
        }
    ]
    (cfg / "manifest.json").write_text(json.dumps({"streams": streams}))
    (cfg / "edge-v-000.log").write_text("[edge-uplink] result window=0 text='Yes'\n")

    report = summarize_config(cfg, "yes_chunks10")
    row = report["per_video"][0]

    assert row["n_total_yes_chunks"] == 10
    assert row["predicted_positive"] is True
