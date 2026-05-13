# ctx_mirage

## Role

You are the edge-side agent for the Mirage reproduction server. Your job is to
unpack the edge prototype, connect it to the cloud intake endpoint, run smoke
tests, then run the N100 no-control accuracy experiment with the revised
`yes_chunks10` aggregation rule.

The edge side sends compressed H.264 packets/chunks over WebSocket. It does not
run vLLM and does not decode pixels for inference.

## Bundle Layout

After unpacking `edge.zip`, expect this shape:

```text
edge_bundle/
  edge/                 # GitHub repo checkout / Python namespace package
    src/
    tools/
    tests/
    cloud/intake/       # cloud-side intake source, included for reference
    data/ucf_crime_hf/  # small manifests only, no mp4 videos
    ctx_mirage.md
    ctx_cloudbuild.md
```

The zip intentionally does not include the UCF-Crime mp4 videos. They are too
large for GitHub/zip. You must either mount/copy the reconstructed mp4 dataset
onto the edge host, or rewrite the manifest paths to wherever the dataset
exists.

## Environment Setup

Run from the directory that contains the `edge/` directory:

```bash
cd /path/to/edge_bundle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r edge/requirements.txt pytest
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pytest edge/tests/test_summarize_anomaly_f1.py -q
```

Expected unit-test result:

```text
3 passed
```

## Data Preparation

The target manifest is:

```text
edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.tsv
```

It contains 200 videos, balanced 100 anomaly + 100 normal. The `cloud_path`
column currently points to the original lab path. Rewrite it to the local mp4
root before running.

Example when videos are under:

```text
/data/ucf_crime_hf/recon_mp4/...
```

rewrite with:

```bash
python - <<'PY'
from pathlib import Path

src = Path("edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.tsv")
dst = Path("edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.local.tsv")
old = "/home/admin123/tangxuan/edge/data/ucf_crime_hf/recon_mp4"
new = "/data/ucf_crime_hf/recon_mp4"
dst.write_text(src.read_text().replace(old, new))
print(dst)
PY
```

Check all paths exist:

```bash
python - <<'PY'
import csv
from pathlib import Path

manifest = Path("edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.local.tsv")
missing = []
with manifest.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if not Path(row["cloud_path"]).exists():
            missing.append(row["cloud_path"])
print(f"missing={len(missing)}")
for p in missing[:10]:
    print(p)
raise SystemExit(1 if missing else 0)
PY
```

## Cloud Endpoint Contract

Coordinate with the cloud-side agent using `ctx_cloudbuild.md`. The edge needs:

- WebSocket ingest URL, normally `ws://<cloud-host>:9100/stream`.
- Admin/health base URL, normally `http://<cloud-host>:9100`.
- The remote intake log path, normally `/tmp/bava_mirage/intake_no_control.log`.
- SSH/scp access if you will copy cloud logs back for aggregation.

Before launching videos:

```bash
export CLOUD_HOST=<cloud-host-or-tunnel-host>
curl -fsS "http://${CLOUD_HOST}:9100/healthz" | python -m json.tool
```

The health response should show four healthy vLLM engines.

## Smoke Run

Create a tiny manifest with two videos:

```bash
MANIFEST=edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.local.tsv
SMOKE=/tmp/mirage_smoke.tsv
{ head -n 1 "$MANIFEST"; sed -n '2,3p' "$MANIFEST"; } > "$SMOKE"
```

Run the smoke:

```bash
RUN_ROOT=runs/mirage_smoke_$(date +%Y%m%d_%H%M%S)
OUT="$RUN_ROOT/N2/no_control_rho1_ef_safety_yes_chunks10"
mkdir -p "$OUT"

python -m edge.tools.per_video_bench \
  --manifest "$SMOKE" \
  --cloud-ws-url "ws://${CLOUD_HOST}:9100/stream" \
  --intake-admin-base "http://${CLOUD_HOST}:9100" \
  --rho 1.0 \
  --alpha 1.0 \
  --window-seconds 4 \
  --decision-window-seconds 40 \
  --max-tokens 12 \
  --prompt "Classify this video clip for surveillance anomaly detection. Output only Yes or No. Yes means the clip shows one of these abnormal categories: arrest, arson, assault, burglary, fighting, road accident, robbery, shooting, shoplifting, stealing, vandalism, explosion, abuse, or any clearly unsafe/criminal behavior. No means normal non-criminal activity without visible danger." \
  --concurrency 2 \
  --linger-s 60 \
  --per-video-timeout 600 \
  --pace-realtime \
  --out "$OUT" | tee "$OUT/bench_stdout.log"
```

Copy the cloud intake log into the config directory before aggregation:

```bash
scp <cloud-user>@${CLOUD_HOST}:/tmp/bava_mirage/intake_no_control.log "$OUT/intake.log"

python -m edge.tools.summarize_anomaly_f1 \
  "$RUN_ROOT/N2" \
  --positive-rule yes_chunks10 \
  --out "$RUN_ROOT/N2/anomaly_f1_summary.json"
```

Important: `yes_chunks10` must read `intake.log`. If it falls back to full
windows, the result is not the revised experiment.

Check chunk provenance:

```bash
python - "$RUN_ROOT/N2/anomaly_f1_summary.json" <<'PY'
import collections, json, sys
summary = json.load(open(sys.argv[1]))
cfg = next(iter(summary["configs"].values()))
sources = collections.Counter()
for row in cfg["per_video"]:
    for sl in row["slices"]:
        for rec in sl["timeline"]:
            sources[rec.get("chunk_source")] += 1
print(sources)
raise SystemExit(1 if sources.get("fallback_full_window", 0) else 0)
PY
```

## Target N100 No-Control Run

Use this exact command shape for the accepted no-control accuracy experiment:

```bash
MANIFEST=edge/data/ucf_crime_hf/eval_videos_balanced_100a_100n.local.tsv
RUN_ROOT=runs/mirage_n100_yes_chunks10_$(date +%Y%m%d_%H%M%S)
OUT="$RUN_ROOT/N100/no_control_rho1_ef_safety_yes_chunks10"
mkdir -p "$OUT"

python -m edge.tools.per_video_bench \
  --manifest "$MANIFEST" \
  --cloud-ws-url "ws://${CLOUD_HOST}:9100/stream" \
  --intake-admin-base "http://${CLOUD_HOST}:9100" \
  --rho 1.0 \
  --alpha 1.0 \
  --window-seconds 4 \
  --decision-window-seconds 40 \
  --max-tokens 12 \
  --prompt "Classify this video clip for surveillance anomaly detection. Output only Yes or No. Yes means the clip shows one of these abnormal categories: arrest, arson, assault, burglary, fighting, road accident, robbery, shooting, shoplifting, stealing, vandalism, explosion, abuse, or any clearly unsafe/criminal behavior. No means normal non-criminal activity without visible danger." \
  --concurrency 100 \
  --linger-s 180 \
  --per-video-timeout 1200 \
  --pace-realtime \
  --out "$OUT" | tee "$OUT/bench_stdout.log"
```

After the bench completes:

```bash
scp <cloud-user>@${CLOUD_HOST}:/tmp/bava_mirage/intake_no_control.log "$OUT/intake.log"
scp <cloud-user>@${CLOUD_HOST}:/tmp/bava_mirage/controller_no_control.jsonl "$OUT/controller.jsonl" || true
scp <cloud-user>@${CLOUD_HOST}:/tmp/bava_mirage/anchors_no_control.jsonl "$OUT/anchors.jsonl" || true

python -m edge.tools.summarize_anomaly_f1 \
  "$RUN_ROOT/N100" \
  --positive-rule yes_chunks10 \
  --out "$RUN_ROOT/N100/anomaly_f1_summary.json"
```

Acceptance checks:

- All stream return codes in `$OUT/manifest.json` are `0`.
- `intake.log` has no `Traceback`, `ReadError`, `RemoteProtocolError`,
  `EngineDead`, `AssertionError`, `OOM`, `CUDA error`, `Killed`, or `SIGSEGV`.
- `anomaly_f1_summary.json` has chunk source `intake` for valid windows and
  no `fallback_full_window`.
- Cloud `/metrics` ends at `running=0 waiting=0` on all vLLM ports. If not,
  ask the cloud agent to restore services and record that restore was needed.

Reference result from the original accepted run:

```text
TP/FP/FN/TN = 49/0/51/100
precision = 100.00%
recall = 49.00%
F1 = 65.77%
accuracy = 74.50%
valid windows / invalid windows = 1174 / 69
```

Small nondeterministic drift is possible, but large differences usually mean
the manifest paths, prompt, EF setting, or `intake.log` chunk aggregation are
not aligned.
