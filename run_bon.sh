#!/usr/bin/env bash
set -e

# Path to your GGUF model
MODEL_PATH="$HOME/.lmstudio/models/Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark/Bonsai-27B-antidoom-1bit-Q1_0.gguf"

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# CUDA env
# UVM hid OOM by paging weights to system RAM (~29W GPU, ~10 t/s). Prefer hard OOM.
# export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export CUDA_MODULE_LOADING=LAZY

echo "==> Launching llama-server with hardware optimizations..."

# Was: -b 2048 -ub 2048 — cut batch only; keep 64k for Hermes.
# If OOM without UVM: try -c 8192 briefly to confirm GPU residency, then raise.
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -c 65536 \
    -b 512 \
    -ub 256 \
    -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --ctx-checkpoints 0 \
    --graph-reuse \
    --jinja \
    -t 6 \
    -tb 12 \
    --port 8080
