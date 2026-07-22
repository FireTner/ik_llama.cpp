#!/usr/bin/env bash
set -e

# Path to your GGUF model
MODEL_DIR="$HOME/.lmstudio/models/Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark"
MODEL_PATH="$MODEL_DIR/Bonsai-27B-antidoom-1bit-Q1_0.gguf"
DRAFT_PATH="$MODEL_DIR/Bonsai-27B-dspark-Q4_1.gguf"

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# CUDA env
# UVM hid OOM by paging weights to system RAM (~29W GPU, ~10 t/s). Prefer hard OOM.
# export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export CUDA_MODULE_LOADING=LAZY

# --- VRAM notes (RTX 4060 Laptop 8GB, ~1GB desktop headroom) ---
# Main Q1 GGUF ≈ 4.35 GiB. 64k q4_0 KV is the squeeze.
# DSpark Q4_1 drafter ≈ 1.8 GiB extra — often will not fit with full -ngl + 64k.
# Fallback order (never UVM):
#   1) keep -c 65536, drop DSpark / lower -ngld
#   2) --fit + --fit-margin, or slightly lower -ngl
#   3) only then shrink -c for diagnosis
#
# DSpark (opt-in): USE_DSPARK=1 ./run_bon.sh
# Requires: Bonsai-27B-dspark-Q4_1.gguf next to the main model.
# Download:
#   huggingface-cli download Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark \
#     Bonsai-27B-dspark-Q4_1.gguf --local-dir "$MODEL_DIR"
USE_DSPARK="${USE_DSPARK:-0}"

DSPARK_ARGS=()
if [ "$USE_DSPARK" = "1" ]; then
    if [ ! -f "$DRAFT_PATH" ]; then
        echo "USE_DSPARK=1 but drafter not found:"
        echo "  $DRAFT_PATH"
        echo "Download with:"
        echo "  huggingface-cli download Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark \\"
        echo "    Bonsai-27B-dspark-Q4_1.gguf --local-dir \"$MODEL_DIR\""
        echo "Continuing without DSpark (main model only, 64k)."
    else
        # ik flag: --spec-type draft-dspark:n_max=4 (not Prism's --spec-draft-n-max)
        # -np 1: DSpark disables cross-request prompt-cache reuse
        # Prefer lowering -ngld before -ngl / -c if OOM
        DSPARK_ARGS+=(
            -md "$DRAFT_PATH"
            -ngld 99
            --spec-type draft-dspark:n_max=4
            -np 1
        )
        echo "==> DSpark enabled: $DRAFT_PATH"
    fi
fi

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
    --port 8080 \
    "${DSPARK_ARGS[@]}"
