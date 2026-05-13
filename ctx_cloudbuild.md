# ctx_cloudbuild

## Role

You are the cloud-side agent for the Mirage reproduction server. Your job is
to pull the prototype from GitHub, build the cloud inference topology from a
JSON config file, start the BAVA intake service, expose the intake endpoint to
the edge agent, and preserve logs needed for `yes_chunks10` aggregation.

The target experiment is N100 no-control accuracy with online-prefill
early-finalizer enabled and video-level positives defined as cumulative
`Yes` chunks >= 10.

Important: the cloud does not have to be exactly four independent
online-prefill vLLM processes. Use `cloud/mirage_cloud_config.example.json` as
the source of truth and adjust it for the server: TP, DP/multiple replicas, GPU
mapping, ports, model path, extra vLLM flags, and intake settings all live
there.

## Repository

Clone the edge prototype repo:

```bash
export WORK=/home/$USER/mirage_bava
mkdir -p "$WORK"
cd "$WORK"
git clone -b codex/online-prefill-intake-edge git@github.com:Pineberry1/edge.git edge
```

If SSH is unavailable, use HTTPS:

```bash
git clone -b codex/online-prefill-intake-edge https://github.com/Pineberry1/edge.git edge
```

The intake package lives at:

```text
$WORK/edge/cloud/intake
```

The edge Python package is a namespace directory named `edge`, so commands that
run `python -m edge.tools...` need the parent directory on `PYTHONPATH`.

## Python Environment

Use the same Python environment that can import the online-prefill vLLM fork.
Install intake dependencies if they are missing:

```bash
cd "$WORK"
python -m pip install -r edge/requirements.txt
python -m pip install -r edge/cloud/requirements.txt
```

Set paths:

```bash
export REPO="$WORK/edge"
export VLLM_SRC=/path/to/vllm_online_prefill_finalizer
export PYTHONPATH="$WORK:$REPO/cloud:$VLLM_SRC:${PYTHONPATH:-}"
```

If this server does not already have the patched vLLM fork, inspect
`$REPO/vllm_patches/overlay/`. It contains the local overlay files used in the
prototype. Apply those files on top of the matching online-prefill/tokenmerger
vLLM fork, then run that fork's own tests before benchmarking.

## Configure Cloud Inference JSON

Copy and edit the example config:

```bash
cd "$REPO"
cp cloud/mirage_cloud_config.example.json cloud/mirage_cloud_config.local.json
$EDITOR cloud/mirage_cloud_config.local.json
```

Fields to adapt:

| JSON field | meaning |
| --- | --- |
| `work_root` | parent directory that contains the `edge/` repo |
| `repo_dir` | full path to this cloned repo |
| `vllm_src` | path to the online-prefill/tokenmerger vLLM fork |
| `python` | Python executable from the environment that can import vLLM |
| `model` | Qwen3-VL FP8 model path |
| `log_root` | stable directory for intake/vLLM logs, usually `/tmp/bava_mirage` |
| `vllm.common_args` | default vLLM CLI flags for every engine |
| `vllm.engines[]` | one OpenAI-compatible endpoint per entry |
| `vllm.engines[].cuda_visible_devices` | GPU set for that endpoint, e.g. `"0"` or `"0,1"` |
| `vllm.engines[].args.tensor_parallel_size` | per-engine TP override |
| `vllm.engines[].extra_args` | pass-through flags for local vLLM, including DP flags if supported |
| `intake.env` | BAVA intake/controller/no-control settings |

Examples:

```json
{
  "name": "tp2_replica0",
  "port": 8011,
  "cuda_visible_devices": "0,1",
  "args": {
    "tensor_parallel_size": 2,
    "max_num_seqs": 16
  }
}
```

For DP, prefer explicit replicas when possible because intake load-balances
across OpenAI endpoints:

```json
[
  {"name": "replica0_tp2", "port": 8011, "cuda_visible_devices": "0,1", "args": {"tensor_parallel_size": 2}},
  {"name": "replica1_tp2", "port": 8012, "cuda_visible_devices": "2,3", "args": {"tensor_parallel_size": 2}}
]
```

If the local vLLM fork supports data-parallel CLI flags inside one process,
put those flags in `extra_args`; the launcher passes them through unchanged.

## Start Configured vLLM Engines

Use the JSON-driven launcher:

```bash
cd "$REPO"
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json start-vllm
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json wait-vllm --timeout-s 900
```

The launcher writes per-engine logs and pid files under:

```text
<log_root>/vllm_logs/
```

## Start Intake For No-Control

Use the same JSON. The intake API base list is derived from
`vllm.engines[]` unless `intake.vllm_api_base_list` is explicitly set.

```bash
cd "$REPO"
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json start-intake
```

The edge-side agent must copy the configured `intake.log` after the run because
`yes_chunks10` reads the `chunks=[...]` lines from it. In the example config:

```text
/tmp/bava_mirage/intake_no_control.log
```

Health check:

```bash
curl -fsS "http://127.0.0.1:9100/healthz" | python -m json.tool
```

The edge agent must be able to reach:

```text
ws://<cloud-host>:9100/stream
http://<cloud-host>:9100/healthz
```

If the server is behind SSH, create a tunnel from the edge host:

```bash
ssh -N -L 19100:127.0.0.1:9100 <cloud-user>@<cloud-host>
```

Then tell the edge agent to use:

```text
ws://127.0.0.1:19100/stream
http://127.0.0.1:19100
```

## During The Run

Watch logs:

```bash
tail -f /tmp/bava_mirage/intake_no_control.log
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json status
```

Fatal signatures that should remain zero in intake logs:

```text
Traceback
ReadError
RemoteProtocolError
EngineDead
AssertionError
OOM
CUDA error
Killed
SIGSEGV
append failed
stream_end append failed
```

## After The Run

Stop intake cleanly:

```bash
curl -fsS --max-time 8 -X POST http://127.0.0.1:9100/admin/purge_sessions || true
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json stop-intake
```

Verify vLLM idle:

```bash
python cloud/mirage_cloud.py --config cloud/mirage_cloud_config.local.json status
```

If any port remains with `waiting > 0` or `running > 0` after a few minutes,
restart the configured vLLM engines and record that restore was needed. The
reference accepted run had `fatal_total=0` but still needed a final restore
because one port had a residual `waiting=1`.

Keep these files available for the edge agent:

```text
/tmp/bava_mirage/intake_no_control.log
/tmp/bava_mirage/controller_no_control.jsonl
/tmp/bava_mirage/anchors_no_control.jsonl
/tmp/bava_mirage/vllm_logs/*.log
```

## Reference Result

The original accepted N100 no-control EF-safety run with real chunk
aggregation produced:

```text
TP/FP/FN/TN = 49/0/51/100
precision = 100.00%
recall = 49.00%
F1 = 65.77%
accuracy = 74.50%
result windows = 1243
valid/invalid windows = 1174/69
edge early_finalize total = 362
vLLM early_finalized total = 2031
source-active GPU SM mean/p50/p95/max = 52.345/77/100/100%
```

Use it as a sanity target, not as a bit-exact guarantee.
