#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/mambauser/tangxuan/online_vllm"
VENV_DIR="$ROOT_DIR/.venv"
MAMBA_ENV="/home/mambauser/micromamba/envs/onlinevllm"

export PATH="$MAMBA_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$MAMBA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8000}"
export VLLM_API_BASE_LIST="${VLLM_API_BASE_LIST:-}"
export INTAKE_HOST="${INTAKE_HOST:-0.0.0.0}"
export INTAKE_PORT="${INTAKE_PORT:-9100}"
export BAVA_LOG="${BAVA_LOG:-info}"

# Controller core
export BAVA_CONTROLLER_ENABLED="${BAVA_CONTROLLER_ENABLED:-1}"
export BAVA_TICK_S="${BAVA_TICK_S:-0.5}"
export BAVA_Q_TARGET="${BAVA_Q_TARGET:-2.0}"
export BAVA_KV_TARGET="${BAVA_KV_TARGET:-0.6}"
export BAVA_MU_RHO="${BAVA_MU_RHO:-0.05}"
export BAVA_MU_ALPHA="${BAVA_MU_ALPHA:-0.05}"
export BAVA_Q_DEADBAND="${BAVA_Q_DEADBAND:-1.0}"
export BAVA_KV_DEADBAND="${BAVA_KV_DEADBAND:-0.05}"
export BAVA_MIN_UPDATE_S="${BAVA_MIN_UPDATE_S:-1.0}"
export BAVA_RHO_LO="${BAVA_RHO_LO:-0.05}"
export BAVA_RHO_HI="${BAVA_RHO_HI:-1.0}"
export BAVA_ALPHA_LO="${BAVA_ALPHA_LO:-0.1}"
export BAVA_ALPHA_HI="${BAVA_ALPHA_HI:-1.0}"
export BAVA_PREEMPT_PANIC="${BAVA_PREEMPT_PANIC:-0.5}"
export BAVA_PREEMPT_STEP="${BAVA_PREEMPT_STEP:-0.1}"

# v10 dual-valve pressure model.
export BAVA_RHO_CONTROL_MODE="${BAVA_RHO_CONTROL_MODE:-network}"
export BAVA_NET_CAP_BYTES_S="${BAVA_NET_CAP_BYTES_S:-0}"
export BAVA_NET_TARGET_UTIL="${BAVA_NET_TARGET_UTIL:-0.90}"
export BAVA_NET_DEADBAND="${BAVA_NET_DEADBAND:-0.05}"
export BAVA_NET_BACKLOG_TARGET_S="${BAVA_NET_BACKLOG_TARGET_S:-2.0}"
export BAVA_NET_SEND_WAIT_TARGET_MS="${BAVA_NET_SEND_WAIT_TARGET_MS:-250}"
export BAVA_EDGE_CAP_TOKENS_S="${BAVA_EDGE_CAP_TOKENS_S:-0}"
export BAVA_EDGE_TARGET_UTIL="${BAVA_EDGE_TARGET_UTIL:-0.90}"
export BAVA_EDGE_DEADBAND="${BAVA_EDGE_DEADBAND:-0.05}"
export BAVA_PREFILL_CAP_TOKENS_S="${BAVA_PREFILL_CAP_TOKENS_S:-0}"
export BAVA_KV_HORIZON_S="${BAVA_KV_HORIZON_S:-10.0}"
export BAVA_KV_MARGIN_RATIO="${BAVA_KV_MARGIN_RATIO:-0.10}"
export BAVA_ETA_FLOOR="${BAVA_ETA_FLOOR:-0.5}"

# Asymmetric recovery (new 2026-04-24)
export BAVA_CLIMB_BACK="${BAVA_CLIMB_BACK:-1}"
export BAVA_CLIMB_BACK_TICKS="${BAVA_CLIMB_BACK_TICKS:-6}"
export BAVA_CLIMB_BACK_STEP_RHO="${BAVA_CLIMB_BACK_STEP_RHO:-0.02}"
export BAVA_CLIMB_BACK_STEP_ALPHA="${BAVA_CLIMB_BACK_STEP_ALPHA:-0.02}"
export BAVA_CLIMB_BACK_Q_SLACK="${BAVA_CLIMB_BACK_Q_SLACK:-0.5}"
export BAVA_CLIMB_BACK_KV_SLACK="${BAVA_CLIMB_BACK_KV_SLACK:-0.1}"

# Per-stream weighting (new 2026-04-24)
export BAVA_PER_STREAM_WEIGHTING="${BAVA_PER_STREAM_WEIGHTING:-1}"
export BAVA_STREAM_WEIGHT_MIN="${BAVA_STREAM_WEIGHT_MIN:-0.5}"
export BAVA_STREAM_WEIGHT_MAX="${BAVA_STREAM_WEIGHT_MAX:-2.0}"

# Window / α executor. BAVA_MAX_FRAMES_PER_WINDOW is edge-side per-chunk
# metadata for the budget model; intake does not use it to subsample online
# prefill inputs. BAVA_INTAKE_HARD_FRAME_CAP is an opt-in emergency cap.
export BAVA_MAX_FRAMES_PER_WINDOW="${BAVA_MAX_FRAMES_PER_WINDOW:-16}"
export BAVA_INTAKE_HARD_FRAME_CAP="${BAVA_INTAKE_HARD_FRAME_CAP:-0}"
export BAVA_ALPHA_EXECUTOR_MODE="${BAVA_ALPHA_EXECUTOR_MODE:-resize}"
export BAVA_ALPHA_WEIGHTED="${BAVA_ALPHA_WEIGHTED:-0}"
export BAVA_ALPHA_MIN_SIDE="${BAVA_ALPHA_MIN_SIDE:-112}"
export BAVA_ALPHA_MAX_SIDE="${BAVA_ALPHA_MAX_SIDE:-1568}"
export BAVA_TOKEN_MERGER_BLOCK_T="${BAVA_TOKEN_MERGER_BLOCK_T:-1}"
export BAVA_TOKEN_MERGER_BLOCK_HW="${BAVA_TOKEN_MERGER_BLOCK_HW:-2}"
export BAVA_STREAM_CONCURRENCY="${BAVA_STREAM_CONCURRENCY:-2}"
export BAVA_MAX_QUEUED_WINDOWS="${BAVA_MAX_QUEUED_WINDOWS:-8}"
export BAVA_RESULT_TIMEOUT_S="${BAVA_RESULT_TIMEOUT_S:-120}"
export BAVA_EF_RESULT_TIMEOUT_S="${BAVA_EF_RESULT_TIMEOUT_S:-30}"
export BAVA_RESULT_POLL_INTERVAL_S="${BAVA_RESULT_POLL_INTERVAL_S:-0.25}"

# Decision budget. Default keeps the nominal camera decision horizon fixed
# (10 chunks for the standard 4s/40s setup). It only shortens windows after
# vLLM reports online-prefill early_finalized, making the budget channel an EF
# safety guard rather than the normal pressure controller. The legacy
# `fair_share` policy can be enabled for ablations, but it shortens stream_end
# windows under high N and therefore changes the latency semantics.
export BAVA_BUDGET_POLICY="${BAVA_BUDGET_POLICY:-ef_guard}"
export BAVA_BUDGET_TARGET_WINDOWS="${BAVA_BUDGET_TARGET_WINDOWS:-10}"
export BAVA_BUDGET_EF_WINDOWS="${BAVA_BUDGET_EF_WINDOWS:-6}"
export BAVA_BUDGET_EF_DYNAMIC="${BAVA_BUDGET_EF_DYNAMIC:-1}"
export BAVA_BUDGET_EF_MARGIN_WINDOWS="${BAVA_BUDGET_EF_MARGIN_WINDOWS:-1}"
export BAVA_BUDGET_EF_COOLDOWN_S="${BAVA_BUDGET_EF_COOLDOWN_S:-60}"
export BAVA_BUDGET_MIN_WINDOWS="${BAVA_BUDGET_MIN_WINDOWS:-2}"
export BAVA_BUDGET_MAX_WINDOWS="${BAVA_BUDGET_MAX_WINDOWS:-32}"
export BAVA_EF_NOTIFY_EDGE="${BAVA_EF_NOTIFY_EDGE:-1}"
export BAVA_ENGINE_ASSIGNMENT="${BAVA_ENGINE_ASSIGNMENT:-least_sessions}"

# Stable edge visual send-window search. This learns a per-engine rho ceiling
# from EF feedback while keeping the semantic decision window fixed.
export BAVA_SEND_WINDOW_ENABLED="${BAVA_SEND_WINDOW_ENABLED:-${BAVA_CONTROLLER_ENABLED}}"
export BAVA_SEND_WINDOW_INIT="${BAVA_SEND_WINDOW_INIT:-0}"
export BAVA_SEND_WINDOW_LO="${BAVA_SEND_WINDOW_LO:-${BAVA_RHO_LO}}"
export BAVA_SEND_WINDOW_HI="${BAVA_SEND_WINDOW_HI:-${BAVA_RHO_HI}}"
export BAVA_SEND_WINDOW_INCREASE_STEP="${BAVA_SEND_WINDOW_INCREASE_STEP:-0.02}"
export BAVA_SEND_WINDOW_STABLE_RESULTS="${BAVA_SEND_WINDOW_STABLE_RESULTS:-20}"
export BAVA_SEND_WINDOW_PROBE_INTERVAL_S="${BAVA_SEND_WINDOW_PROBE_INTERVAL_S:-120}"
export BAVA_SEND_WINDOW_KV_PROBE_MAX="${BAVA_SEND_WINDOW_KV_PROBE_MAX:-0.85}"
export BAVA_SEND_WINDOW_EF_REDUCE="${BAVA_SEND_WINDOW_EF_REDUCE:-0.80}"
export BAVA_SEND_WINDOW_FAILURE_REDUCE="${BAVA_SEND_WINDOW_FAILURE_REDUCE:-0.50}"
export BAVA_SEND_WINDOW_MARGIN="${BAVA_SEND_WINDOW_MARGIN:-0.85}"
export BAVA_SEND_WINDOW_EF_COOLDOWN_S="${BAVA_SEND_WINDOW_EF_COOLDOWN_S:-90}"
export BAVA_SEND_WINDOW_MIN_UPDATE_S="${BAVA_SEND_WINDOW_MIN_UPDATE_S:-1.0}"
# A positive value enables a KV waterline emergency rollback. The normal
# send-window unsafe signal is vLLM early-finalizer feedback, so this defaults
# to disabled.
export BAVA_SEND_WINDOW_KV_PANIC="${BAVA_SEND_WINDOW_KV_PANIC:-0}"
export BAVA_SEND_WINDOW_ENGINE_MONITOR_S="${BAVA_SEND_WINDOW_ENGINE_MONITOR_S:-1.0}"

# Optional post-decision visual memory carry-over. The per-stream
# hello.visual_memory_merge flag decides whether intake asks vLLM to export
# memory; these env vars only tune the exported memory shape/behavior.
export BAVA_VISUAL_MEMORY_NUM_FRAMES="${BAVA_VISUAL_MEMORY_NUM_FRAMES:-8}"
export BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME="${BAVA_VISUAL_MEMORY_TOKENS_PER_FRAME:-32}"
export BAVA_VISUAL_MEMORY_TEXT_PREFIX="${BAVA_VISUAL_MEMORY_TEXT_PREFIX:-}"
export BAVA_VISUAL_MEMORY_ID_PREFIX="${BAVA_VISUAL_MEMORY_ID_PREFIX:-bava_mem}"
export BAVA_VISUAL_MEMORY_STRICT="${BAVA_VISUAL_MEMORY_STRICT:-0}"
export BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE="${BAVA_VISUAL_MEMORY_WARM_PREFIX_CACHE:-1}"

# Logs
export BAVA_CONTROLLER_LOG="${BAVA_CONTROLLER_LOG:-/tmp/bava_controller.jsonl}"
export BAVA_ANCHOR_LOG="${BAVA_ANCHOR_LOG:-/tmp/bava_anchors.jsonl}"

exec python -m intake.server "$@"
