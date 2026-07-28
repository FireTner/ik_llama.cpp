#!/usr/bin/env bash
set -euo pipefail

# Path to your GGUF model
MODEL_DIR="$HOME/.lmstudio/models/prism-ml/Bonsai-27B-gguf"
MODEL_PATH="$MODEL_DIR/Bonsai-27B-Q1_0.gguf"
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

# --- Hermès needs -c 65536. Do not shrink CTX. ---
# Current default GGUF: prism-ml Bonsai-27B-Q1_0 (~3446 MiB CUDA weights measured 2026-07-28).
# Older antidoom pack was ~4270 MiB weights / ~5960 MiB full offload — do not mix those numbers
# into Prism VRAM math. See .local/findings/00-trust-and-stack.md (gitignored) and FINDINGS.md banner.
# NEVER -ngl ≤64: parks output on CPU, compute 252→1247 (VRAM wash + slower tg).
#
# VRAM for DSpark @64k (only clean levers left):
#   CPU_LAYERS=0-N  → -ot early blk.* to CPU, keep -ngl 99 (output stays GPU).
#                     Off by default: park taxes target graph (splits≈200) and collapses tg
#                     even when the drafter stays on GPU. Measured regression on this laptop.
#   quit Vesktop/Brave → desktop VRAM (often ~80–150+ MiB; was higher with more Electron)
#   -ngld low + -cd 4096 → draft KV tiny; IO-share aliases embd/lm_head with target
# Never UVM. Never trade away 64k for Hermès.
#
# Default: do NOT park layers. Target -ot park fractures the main graph (splits≈200)
# and collapses decode even when the drafter stays on GPU. IO-share already fits DSpark
# @64k without parking (~6.8–7.0 GiB class). USE_DSPARK=1 + autotune currently stays
# target-only (n_max=0) on this 8GB stack — drafting works but loses tg (see local findings).
# Use CPU_LAYERS=0-15 only if you need VRAM headroom and accept the PCIe/split tax.
#
#   USE_DSPARK=1 ./run_bon.sh
#   CPU_LAYERS=0-19 USE_DSPARK=1 ./run_bon.sh   # more headroom if OOM (slower)
#   CPU_LAYERS=0-15 USE_DSPARK=1 ./run_bon.sh   # explicit park
#
# Drafter lives next to the Prism target by default. Historical antidoom HF id (if needed):
#   huggingface-cli download Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark \
#     Bonsai-27B-dspark-Q4_1.gguf --local-dir "$MODEL_DIR"
USE_DSPARK="${USE_DSPARK:-0}"
CTX=65536

# Resolve DSpark first so a missing drafter does not still apply CPU layer parking.
DSPARK_ARGS=()
DSPARK_ACTIVE=0
if [ "$USE_DSPARK" = "1" ]; then
    if [ ! -f "$DRAFT_PATH" ]; then
        echo "USE_DSPARK=1 but drafter not found: $DRAFT_PATH"
        echo "  huggingface-cli download Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark \\"
        echo "    Bonsai-27B-dspark-Q4_1.gguf --local-dir \"$MODEL_DIR\""
        echo "Continuing without DSpark (no default CPU layer park)."
    else
        DSPARK_ACTIVE=1
        # 64k main KV already heavy — keep draft ctx small; start conservative on -ngld.
        NGLD="${NGLD:-20}"
        CD="${CD:-4096}"
        DSPARK_ARGS+=(
            -md "$DRAFT_PATH"
            -ngld "$NGLD"
            -cd "$CD"
            --spec-type draft-dspark:n_max=4
            --spec-autotune
            -np 1
        )
        echo "==> DSpark @64k: $DRAFT_PATH (-ngld $NGLD, -cd $CD). Raise NGLD if VRAM allows."
    fi
fi

# No default park. Explicit CPU_LAYERS=0-N enables it; CPU_LAYERS=0 is a no-op.
CPU_LAYERS="${CPU_LAYERS:-}"
if [ "$CPU_LAYERS" = "0" ]; then
    CPU_LAYERS=""
fi

OT_ARGS=()
if [ -n "$CPU_LAYERS" ]; then
    if [[ "$CPU_LAYERS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        lo="${BASH_REMATCH[1]}"; hi="${BASH_REMATCH[2]}"
        seq_re=$(seq -s '|' "$lo" "$hi")
        OT_ARGS+=(-ot "^blk\\.(${seq_re})\\..*=CPU")
        echo "==> Hermès 64k: parking blk.${lo}-${hi}.* on CPU (~$(( (hi - lo + 1) * 51 )) MiB), output stays GPU"
    else
        echo "CPU_LAYERS must look like 0-15 (got: $CPU_LAYERS)"; exit 1
    fi
fi

# Batch sizes. Defaults (-b 512 -ub 256) were chosen to avoid OOM on the full-GPU stack.
# DSpark IO-tensor sharing frees ~1.2-1.4 GiB by aliasing the drafter's token_embd/output to the
# target's, so a larger -ub (e.g. UBATCH=512) can speed up prompt processing when VRAM allows.
# Only raise UBATCH within available VRAM headroom (watch nvidia-smi / load-log compute buffer);
# too large will OOM at 64k. Keep BATCH >= UBATCH.
#   BATCH=512 UBATCH=512 USE_DSPARK=1 ./run_bon.sh
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-256}"

# Prompt cache (host RAM). ik default is 8192 MiB — on 14 GiB RAM that fills during Hermès
# multi-turn (full KV snapshots + embedded ctx-checkpoints), then kswapd OOM-kills llama-server
# even though load looked fine. Default OFF on this machine. Raise only with spare RAM:
#   CACHE_RAM=512 ./run_bon.sh
CACHE_RAM="${CACHE_RAM:-0}"

# Context checkpoints also live in host RAM (needed for Qwen3.5 / Hermès tool turns).
# 32 is fine if each is small (PARTIAL_ONLY); lower if host RSS still climbs.
#   CTX_CHECKPOINTS=8 ./run_bon.sh
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-16}"

LOG_FILE="${LOG_FILE:-/tmp/run_bon.log}"

echo "==> Launching llama-server"
echo "    -c $CTX  -ngl 99  -b $BATCH -ub $UBATCH"
echo "    --cache-ram $CACHE_RAM  --ctx-checkpoints $CTX_CHECKPOINTS"
echo "    log: $LOG_FILE"
echo "    VERIFY: startup must print: prompt cache is disabled   OR   size limit: ${CACHE_RAM} MiB"
if [ "$DSPARK_ACTIVE" = "1" ]; then
    echo "    VERIFY: load log should mention DSpark skipping drafter token_embd/output (IO share)"
fi

# shellcheck disable=SC2086
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    "${OT_ARGS[@]}" \
    -c "$CTX" \
    -b "$BATCH" \
    -ub "$UBATCH" \
    -fa on \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -khad \
    -vhad \
    --ctx-checkpoints "$CTX_CHECKPOINTS" \
    --cache-ram "$CACHE_RAM" \
    --graph-reuse \
    --jinja \
    -t 6 \
    -tb 12 \
    --port 8080 \
    "${DSPARK_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
