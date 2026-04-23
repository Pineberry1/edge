# Edge-side packet relay (PacketGame-style)

Edge implementation of the $\rho$ executor in the BAVA pipeline. Runs on a Jetson
AGX-class device at the camera side. **No pixels are decoded on the edge.**
Only the compressed bitstream is inspected (packet sizes, H.264 NAL types,
slice_type), a lightweight importance score is computed, and a decoder-safe
subset of GOP-aligned packets is forwarded to the cloud for real decode and
VLM inference.

The design follows PacketGame (SIGCOMM '23): analytics on the compressed
domain, not pixels. The decoder-safe packet-set invariant means the cloud
does not need any modified FFmpeg/NVDEC — it consumes the received bytes
unchanged.

## Pipeline

```
camera  ── RTSP (H.264) ──►  rtsp_source      (PyAV demux, no decode)
                                 │
                                 ▼
                          h264_parser         (Annex-B, NAL types, slice_type)
                                 │
                                 ▼
                          features            (packet size z-score, novelty,
                                               GOP pos, anchor embedding)
                                 │
                                 ▼
                          scorer              (PacketGame-style, CPU)
                                 │
                                 ▼
                          gop_buffer          (decoder-safe prefix selection)
                                 │
                                 ▼
                          uplink  ── TCP ──►  cloud_sink / vLLM ingest
```

All payloads on the wire are **H.264 Annex-B** (start-code delimited) with
SPS/PPS re-injected before every IDR so each GOP is independently decodable
even if earlier GOPs were dropped.

## Files

| path | role |
|---|---|
| `src/config.py` | `EdgeConfig` dataclass with $\rho$, scorer weights, uplink addr |
| `src/rtsp_source.py` | PyAV demux, AVCC→Annex-B, SPS/PPS injection |
| `src/h264_parser.py` | NAL walker, exp-Golomb, slice_type extraction |
| `src/features.py` | per-packet feature extractor + anchor embedding |
| `src/scorer.py` | PacketGame-style heuristic scorer |
| `src/gop_buffer.py` | GOP-aligned decoder-safe selection |
| `src/wire.py` | length-framed JSON-header + binary-payload protocol |
| `src/uplink.py` | TCP sender with reconnect |
| `src/edge_main.py` | CLI entry point |
| `tools/cloud_sink.py` | minimal cloud receiver for local testing |
| `tools/make_test_clip.py` | generate an H.264 clip with known GOP structure |

## Local smoke test

```bash
export PATH="$HOME/.local/bin:$PATH"

# 1) generate a 4s H.264 clip with GOP=25 (4 IDRs, mix of P and B)
python3 -m edge.tools.make_test_clip --out edge/data/test.mp4 --seconds 4 --fps 25 --gop 25

# 2) start the cloud sink (dumps per-GOP Annex-B files)
python3 -m edge.tools.cloud_sink --port 9000 --out-dir /tmp/cloud_gops &

# 3) run the edge relay at rho=0.3
python3 -m edge.src.edge_main --source edge/data/test.mp4 \
    --cloud-host 127.0.0.1 --cloud-port 9000 --rho 0.3

# 4) verify each GOP decodes standalone on the cloud
python3 -c "import av; [print(p, sum(1 for _ in av.open(p).decode(video=0))) \
    for p in sorted(__import__('glob').glob('/tmp/cloud_gops/*.h264'))]"
```

Representative output from the included test clip (100 packets, 4 GOPs):

| rho | packets sent | bytes uplinked |
|---:|---:|---:|
| 0.1 | 20 | 0.05 MB |
| 0.3 | 47 | 0.10 MB |
| 0.6 | 84 | 0.17 MB |
| 0.9 | 100 | 0.21 MB |

## Wire protocol

One TCP connection per edge→cloud pair. Messages are length-framed:

```
[4B BE header_len][JSON header][4B BE payload_len][payload bytes]
```

Header `kind` is `hello`, `packet`, or `bye`. A `packet` header carries all
edge-side metadata (seq, gop_index, gop_pos, pts_s, is_idr, slice_type,
nal_types, score, gop_mode, anchor) so the cloud can use the anchor embedding
as a token-allocation prior without re-parsing the bitstream.

## Jetson AGX Orin deployment notes

The dev-machine pipeline works unmodified on Jetson with CPU-only parsing
(which is cheap — a few µs per packet). To take advantage of on-device
hardware at the edge when needed, replace modules as follows:

- **Ingest**: prefer GStreamer `rtspsrc ! rtph264depay` over PyAV's FFmpeg
  demuxer. Wrap into a Python generator that yields the same `PacketRecord`.
- **Parser**: keep on CPU. NAL walking + exp-Golomb is ~5 µs/packet; GPU
  launch overhead dwarfs the compute.
- **Scoring (optional)**: if upgrading to a learned scorer (small CNN over
  a window of packet-size sequences as in PacketGame), export to **TensorRT**
  FP16 and run inference on the AGX iGPU; batch 32–64 windows.
- **Anchor embedding**: for a learned embedding (small MLP), also TensorRT.
- **Optional selective decode on edge**: some deployments want the edge to
  decode I-frames only for a local sanity check. Use **NVDEC** via
  `nvv4l2decoder` in GStreamer, gated by our `is_idr` flag. This stays
  compatible with the "edge does not decode P/B" invariant.

Tuning knobs worth exposing on deployment:

| env var | effect |
|---|---|
| `BAVA_RHO` | target per-GOP non-IDR keep ratio |
| `BAVA_WINDOW_S` | window length for BAVA controller (not yet wired) |
| `BAVA_CLOUD_HOST/PORT` | uplink target |

## What is intentionally out of scope here

- Motion-vector extraction. PacketGame explicitly avoids this; packet sizes
  plus frame type are enough signal. MV extraction requires CABAC/CAVLC
  entropy decoding which is not worth the latency on edge CPUs.
- B-frame-aware fine selection. The current GOP buffer only ever forwards an
  IDR-aligned prefix, which is decoder-safe in both closed and open GOPs.
  Finer selection (drop trailing B only) requires slice_header ref lists and
  is TODO.
- The BAVA closed-loop controller itself. Only the edge actuator for $\rho$
  lives here; the controller that drives $\rho$ from cloud telemetry lives
  in the cloud repo.
