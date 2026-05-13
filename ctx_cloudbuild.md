# ctx_cloudbuild

## Role

You are the cloud-side agent for the Mirage reproduction server. Your job is
to pull the prototype from GitHub, start four online-prefill vLLM engines,
start the BAVA intake service, expose the intake endpoint to the edge agent,
and preserve logs needed for `yes_chunks10` aggregation.

The target experiment is N100 no-control accuracy with online-prefill
early-finalizer enabled and video-level positives defined as cumulative
`Yes` chunks >= 10.

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

## Start Four vLLM Engines

The accepted reference used GPU0-3 and ports 8011-8014 with:

- model: Qwen3-VL-8B-Instruct-FP8
- online-prefill enabled
- online-prefill early-finalizer enabled
- `max_model_len=40960`
- `max_num_seqs=12`
- prefix caching disabled
- eager mode
- static visual tokenmerger env alpha `1.0`, input `image`, block_t `1`,
  block_hw `2`

Template launcher:

```bash
export MODEL=/path/to/Qwen3-VL-8B-Instruct-FP8
export LOG_DIR=/tmp/bava_mirage/vllm_logs
mkdir -p "$LOG_DIR"

for i in 0 1 2 3; do
  port=$((8011 + i))
  log="$LOG_DIR/vllm_${port}.log"
  pid="$LOG_DIR/vllm_${port}.pid"
  rm -f "$log" "$pid"
  (
    cd "$WORK"
    setsid env \
      CUDA_VISIBLE_DEVICES="$i" \
      PYTHONPATH="$VLLM_SRC:${PYTHONPATH:-}" \
      VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_ALPHA=1.0 \
      VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_T=1 \
      VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_BLOCK_HW=2 \
      VLLM_ONLINE_PREFILL_VISUAL_TOKEN_MERGER_INPUT=image \
      python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port "$port" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 40960 \
        --enable-online-prefill \
        --enable-online-prefill-early-finalizer \
        --online-prefill-early-finalize-kv-usage-threshold 0.90 \
        --online-prefill-early-finalize-min-tokens 0 \
        --max-num-seqs 12 \
        --no-enable-prefix-caching \
        --enforce-eager \
        --mm-processor-cache-gb 0 \
        > "$log" 2>&1 </dev/null &
    echo $! > "$pid"
  )
done
```

Wait for health:

```bash
for p in 8011 8012 8013 8014; do
  until curl -fsS --max-time 3 "http://127.0.0.1:${p}/health" >/dev/null; do
    echo "waiting for vLLM $p"
    sleep 5
  done
done
```

## Start Intake For No-Control

Use a stable log path. The edge-side agent must copy this exact log after the
run because `yes_chunks10` reads the `chunks=[...]` lines from it.

```bash
export MIRAGE=/tmp/bava_mirage
mkdir -p "$MIRAGE"

export INTAKE_HOST=0.0.0.0
export INTAKE_PORT=9100
export VLLM_API_BASE=http://127.0.0.1:8011
export VLLM_API_BASE_LIST=http://127.0.0.1:8011,http://127.0.0.1:8012,http://127.0.0.1:8013,http://127.0.0.1:8014

export BAVA_MAX_FRAMES_PER_WINDOW=100
export BAVA_STREAM_CONCURRENCY=2
export BAVA_MAX_QUEUED_WINDOWS=4

export BAVA_CONTROLLER_ENABLED=0
export BAVA_BUDGET_ENABLED=0
export BAVA_SEND_WINDOW_ENABLED=0
export BAVA_ALPHA_EXECUTOR_MODE=off

export BAVA_ADMISSION_KV_WARN=999
export BAVA_ADMISSION_KV_HIGH=999
export BAVA_ADMISSION_KV_PANIC=999
export BAVA_ADMISSION_PREEMPT_PANIC=999

export BAVA_CONTROLLER_LOG="$MIRAGE/controller_no_control.jsonl"
export BAVA_ANCHOR_LOG="$MIRAGE/anchors_no_control.jsonl"

rm -f "$MIRAGE/intake_no_control.log" "$BAVA_CONTROLLER_LOG" "$BAVA_ANCHOR_LOG"

cd "$WORK"
setsid env PYTHONPATH="$WORK:$REPO/cloud:$VLLM_SRC:${PYTHONPATH:-}" \
  python -m intake.server \
  > "$MIRAGE/intake_no_control.log" 2>&1 </dev/null &
echo $! > "$MIRAGE/intake_no_control.pid"
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
tail -f "$MIRAGE/intake_no_control.log"
for p in 8011 8012 8013 8014; do
  curl -fsS "http://127.0.0.1:${p}/metrics" | \
    egrep 'vllm:num_requests_running|vllm:num_requests_waiting|vllm:gpu_cache_usage_perc' || true
done
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
pkill -TERM -f 'python -m intake.server' || true
sleep 2
pkill -KILL -f 'python -m intake.server' || true
```

Verify vLLM idle:

```bash
for p in 8011 8012 8013 8014; do
  echo "port=$p"
  curl -fsS "http://127.0.0.1:${p}/health" >/dev/null && echo health=ok
  curl -fsS "http://127.0.0.1:${p}/metrics" | \
    egrep 'vllm:num_requests_running|vllm:num_requests_waiting|vllm:gpu_cache_usage_perc' || true
done
```

If any port remains with `waiting > 0` or `running > 0` after a few minutes,
restart the four vLLM engines and record that restore was needed. The reference
accepted run had `fatal_total=0` but still needed a final restore because one
port had a residual `waiting=1`.

Keep these files available for the edge agent:

```text
/tmp/bava_mirage/intake_no_control.log
/tmp/bava_mirage/controller_no_control.jsonl
/tmp/bava_mirage/anchors_no_control.jsonl
/tmp/bava_mirage/vllm_logs/vllm_8011.log
/tmp/bava_mirage/vllm_logs/vllm_8012.log
/tmp/bava_mirage/vllm_logs/vllm_8013.log
/tmp/bava_mirage/vllm_logs/vllm_8014.log
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
