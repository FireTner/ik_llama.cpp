# Bonsai-27B on ik_llama.cpp — Findings

> ## Epistemic banner (2026-07-28) — read before using numbers
>
> This file documents the **`Bonsai-27B-antidoom-1bit-Q1_0`** journey (2026-07-22 era).
> It is **not** interchangeable with the current default in `run_bon.sh`:
>
> | | This file (historical) | Current Hermès default (`run_bon.sh`) |
> |---|---|---|
> | GGUF | `Bonsai-27B-antidoom-1bit-Q1_0.gguf` (~4.4 GB disk) | `prism-ml/.../Bonsai-27B-Q1_0.gguf` |
> | CUDA0 weights (typical) | **~4270 MiB** | **~3446 MiB** |
> | Short / @~20k tg (post Q1 CUDA) | **~35 / ~24 t/s** | **~40 / ~26.5 t/s** (re-measured 2026-07-28) |
> | Full offload smi @64k | **~5900–5960 MiB** | **~5.1–5.3 GiB** process (non-DSpark) |
>
> **Still trustworthy from this file (mechanism / cliffs):** UVM paging disaster; missing CUDA Q1 → huge splits; never `-ngl ≤ 64` (compute wash ~1247 MiB); prefer hard OOM over UVM; q4 KV + Hadamard as working KV class.
>
> **Do not treat as current baseline without re-measure:** absolute VRAM headroom, “~1 GB free for DSpark”, tg ceilings, checkpoint create **~200 ms/512 tok** (not re-timed on Prism).
>
> **Current stack + trust tiers (local, gitignored):** `.local/findings/00-trust-and-stack.md`  
> **Current Hermès / DSpark benches:** `.local/findings/bonsai-64k-bench.md`
>
> Keep this file as archaeology. Prefer appending corrections to `.local/findings/` over silently rewriting history here.

Notes from tuning `Bonsai-27B-antidoom-1bit-Q1_0` for Hermès on this machine (2026-07-22).
Launch script: `run_bon.sh` (defaults have since moved to the Prism GGUF — see banner). Build: `build_native.sh` (Zen4 + CUDA sm_89).

---

## Hardware & model

| Piece | Detail |
|---|---|
| CPU | AMD Ryzen 9 7845HX (Zen 4, AVX-512) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, CC 8.9 |
| RAM | ~14 GB (+ swap) — tight |
| Desktop | KDE Wayland + Vesktop/Brave/Electron ≈ **1.0–1.2 GB** VRAM |
| Model | `qwen35` hybrid, 64 layers, n_embd=5120, vocab=248320 |
| GGUF | `Bonsai-27B-antidoom-1bit-Q1_0.gguf` (~4.4 GB on disk) |
| Goal | Hermès harness, **`-c 65536`**, target ~35 t/s (esp. long ctx) |

### Weight breakdown (measured from GGUF)

| Tensor group | Size |
|---|---|
| `blk.*` (64 layers) | **3275.74 MiB** (~51.2 MiB/layer) |
| `output.weight` (q6_K) | **994.63 MiB** |
| `token_embd.weight` | **170.51 MiB** (stays on CPU) |
| **Total** | **~4441 MiB** |

---

## Performance journey

### 1) UVM spill (~10 t/s, terrible TTFT)

**Symptoms**
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `-c 65536` + `-b/-ub 2048`
- `nvidia-smi`: header showed ~6–7 GB used, but `llama-server` only **~160–200 MiB**
- GPU power ~**26–29 W** / 140 W
- Decode ~**10 t/s**; 20k prompt TTFT could hit **tens of minutes**

**Cause:** Weights/KV paged host↔GPU. Not a “slow kernel” problem.

**Fix:** Disable UVM. Prefer hard OOM over silent paging. Cut batch to `-b 512 -ub 256`.

### 2) Residency fixed, then Q1 CUDA missing (~2.4 t/s)

After UVM off, `llama-server` showed **multi-GB** VRAM (real GPU residency), but:

- Log: `unknown type q1_0_g128` / `model ftype = unknown`
- **`graph splits = 626`** (healthy single-GPU ≈ 1–2)
- Decode ~**2.4 t/s**, GPU still ~26 W

**Cause:** ik had **CPU** `Q1_0_G128` (IQK) but **no CUDA** kernels → every matmul bounced to CPU.

**Fix:** Port CUDA Q1_0 / Q1_0_G128 (commit `e76f0775` “Added CUDA Q1_0_G128 and DSpark”).

### 3) After Q1 CUDA port (current baseline)

| Metric | Value |
|---|---|
| `ftype` | `Q1_0_G128` |
| `graph splits` | ≈ **2** |
| Short-prompt tg | ~**35 t/s** |
| ~20k context tg | ~**24 t/s** |
| GPU under load | ~**106 W**, ~98% util |
| Full offload @64k | `llama-server` ≈ **5900–5960 MiB** |

One intermittent CUDA illegal-memory access was seen once; not attributed to Hadamard.

---

## Current good flags (`run_bon.sh`)

```text
-ngl 99
-c 65536                 # Hermès hard requirement
-b 512 -ub 256
-fa on
--cache-type-k/v q4_0
-khad -vhad              # Hadamard before q4 KV (quality @ same VRAM)
--ctx-checkpoints 32     # NOT 0
--graph-reuse
--jinja
-t 6 -tb 12
# UVM: off
```

### Why not TurboQuant for KV

Prefer **q4_0 + Hadamard** (`-khad/-vhad`) over chasing TurboQuant for this setup: same VRAM class, simpler, already working.

### Hermès / checkpoints

| Setting | Effect |
|---|---|
| `--ctx-checkpoints 0` | Full re-PP of ~20k every tool turn → wall-clock disaster |
| `--ctx-checkpoints 32` | Restores ~18 ms; incremental PP much faster |

Checkpoint **create** still costs ~**200 ms / 512 tokens** — meaningful Hermès wall-clock; optional future win is faster async checkpoint create (~15–25% session time), not Q1 MMVQ polish (~3–8% tg).

---

## VRAM accounting @64k (full GPU)

Measured load lines with `-ngl 99`:

| Buffer | Size |
|---|---|
| CUDA0 weights | **4270.39 MiB** (= blk 3276 + output 995) |
| CPU weights | **170.51 MiB** (embd) |
| CUDA0 KV | **1301.63 MiB** |
| CUDA0 compute | **252.50 MiB** |
| **Sum ≈** | **~5.8 GiB** → nvidia-smi process **~5960 MiB** |

Desktop + model ≈ **7.1–7.2 GB / 8.2 GB** → ~**1 GB** free for DSpark if lucky.

KV scales roughly with ctx (same quant):

| `-c` | KV ≈ |
|---|---|
| 65536 | 1301 MiB |
| 32768 | ~650 MiB |
| 16384 | ~325 MiB |

**Hermès keeps 64k** — shrinking ctx is not an option for the intended use.

---

## Critical finding: parking `output.weight` is a VRAM wash

Tried `-ngl 64` and/or `-ot` to move `output.weight` (~995 MiB q6_K) to CPU for DSpark headroom.

| Config | Weights (CUDA0) | Compute (CUDA0) | nvidia-smi ≈ |
|---|---|---|---|
| `-ngl 99` (output GPU) | 4270 MiB | **252 MiB** | ~5960 MiB |
| `-ngl 64` (output CPU) | 3276 MiB | **1247 MiB** | ~5960 MiB |

**Net VRAM unchanged.** CPU lm_head graph inflates the CUDA compute buffer by ~1 GB. tg also slows.

### ik offload rule (important)

- Model has **64** repeating layers; “output” is an extra offload slot → **65/65** with `-ngl 99`.
- Output goes to GPU only if **`n_gpu_layers > n_layer`** (i.e. need **`-ngl 65+` / `99`**).
- Any **`-ngl ≤ 64`** parks output on CPU → compute wash above.

### `-ot` regex footgun

`-ot 'output\.weight=CPU'` uses **substring** regex match and also hits every  
`blk.*.attn_output.weight` (~4.2 MiB × 16 full-attn layers).

**Use:** `-ot '^output\.weight$=CPU'` if you ever force output to host  
(not recommended here — wash).

---

## How to scrape VRAM *without* the wash (keep 64k)

Keep **`-ngl 99`** so lm_head stays on GPU, then:

| Lever | Frees | Notes |
|---|---|---|
| `CPU_LAYERS=0-N` via `-ot '^blk\.(0\|…\|N)\..*=CPU'` | ~**51 MiB/layer** | Early layers on CPU; some graph splits; output stays GPU |
| `0-11` | ~0.6 GiB | Mild |
| `0-15` | ~0.8 GiB | Default when `USE_DSPARK=1` in `run_bon.sh` |
| `0-19` | ~1.0 GiB | If OOM with DSpark |
| Quit Vesktop / Brave GPU | ~300–400 MiB | Desktop only |
| Lower draft `-ngld` / `-cd 4096` | draft-side | Main ctx stays 64k |

**Do not** use `-ngl 48` etc. hoping to free layers while keeping output on GPU — ik will still put output on CPU whenever `ngl ≤ 64`.

---

## DSpark (path to ~30–35 t/s at long ctx)

### Facts

- Drafter file: `Bonsai-27B-dspark-Q4_1.gguf` ≈ **1.79 GB** on disk (not yet downloaded here; `/home` had ~3.5 GB free).
- Prism: drafter is a **compact ~6-layer** block-parallel model; **unique weights ~0.5 GB** at serve if embd/lm_head shared with target.
- Speculative decode is **lossless** vs target distribution; speedup depends on accept length.
- Published big-GPU speedups ~**1.3–1.4×**; on 4060 laptop expect maybe **~28–34 t/s** from a ~24 t/s long-ctx base if draft fits and accepts well — not guaranteed “always 35”.
- Prism’s own quickstart often uses **`-c 16384`** with full `-ngl/-ngld`; we cannot follow that for Hermès.

### Recommended @64k (in `run_bon.sh`)

```bash
# Download once:
huggingface-cli download Danny-Dasilva/Bonsai-27B-antidoom-1bit-DSpark \
  Bonsai-27B-dspark-Q4_1.gguf --local-dir "$MODEL_DIR"

USE_DSPARK=1 ./run_bon.sh
# defaults: CPU_LAYERS=0-15, NGLD=20, CD=4096, CTX=65536

# If OOM:
CPU_LAYERS=0-19 NGLD=12 USE_DSPARK=1 ./run_bon.sh

# If layer-park hurts tg too much:
CPU_LAYERS=0-8 NGLD=16 USE_DSPARK=1 ./run_bon.sh
```

Never re-enable UVM to “make DSpark fit.”

---

## What will / won’t get you 35 t/s

| Approach | Expectation |
|---|---|
| Short prompt, current stack | **Already ~35 t/s** |
| Long ctx (~20k), no DSpark | **~24 t/s** — bandwidth / KV bound |
| DSpark @64k with layer park | Best shot at **~30–35** long |
| Q1 MMVQ micro-opts | Small (~3–8% tg) |
| Faster ctx-checkpoint create | Big **Hermès wall-clock** win, not peak tg |
| TurboQuant KV | Not preferred vs q4+Hadamard here |
| UVM | Makes everything worse |
| `-ngl 64` / output→CPU | No VRAM win; slower |

---

## Load-log checklist

Healthy full-GPU @64k:

```text
offloaded 65/65 layers to GPU
CUDA0 buffer size =  4270.xx MiB
CPU buffer size   =   170.xx MiB
CUDA0 KV buffer   =  1301.xx MiB
CUDA0 compute     =   252.xx MiB
graph splits ≈ 2
ftype = Q1_0_G128
```

With DSpark layer park (`0-15`), expect CUDA weights **~800 MiB lower**, still **`offloaded 65/65`** (output on GPU), compute still ~**252** (not ~1247).

Red flags:

```text
graph splits = hundreds     → CPU fallback / missing CUDA op
llama-server ~200 MiB       → UVM / host spill
CUDA compute ≈ 1247 MiB     → output (lm_head) on CPU
CUDA0 buffer still 4270
  after intending to free output → override didn’t apply / old process
```

---

## Commits / code notes

- Working Q1 CUDA: `e76f0775` (“Added CUDA Q1_0_G128 and DSpark”).
- Build target GPU arch: **sm_89** (4060 Laptop).
- ik LAYER split: `buft_output = GPU` iff `n_gpu_layers > n_layer`.

---

## Open items

1. Download DSpark drafter; measure tg @ ~20k with `USE_DSPARK=1` and tune `CPU_LAYERS` / `NGLD`.
2. Confirm IO-share VRAM delta in nvidia-smi / load log (`DSpark: skipping drafter token_embd/output`).
3. Optional later: sm_89-specific Q1 MMVQ polish (modest).
4. One unexplained CUDA illegal-memory crash — repro if it returns.

## Code status (2026-07-26)

- **DSpark IO-tensor sharing** is wired: draft load skips embd/output, then aliases the target's tensors after dim + shape checks (logs types).
- **Context checkpoints**: KV serialize stays synchronous (required for qwen35 hybrid + to avoid racing speculative accept). Host-side list commit/eviction runs on a worker; callers that need a durable checkpoint use `create_checkpoint_committed()` so `checkpoint_pos` / release paths only advance after a successful list commit.
- Missing DSpark GGUF no longer applies default `CPU_LAYERS=0-15` parking.
- Unsafe overlapping GPU→host checkpoint D2H was **not** shipped (race with speculative accept; no-op on hybrid anyway).
