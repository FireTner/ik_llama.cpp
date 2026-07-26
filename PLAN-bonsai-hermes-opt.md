<!-- /autoplan restore point: /home/cachyfire/.gstack/projects/ikawrakow-ik_llama.cpp/main-autoplan-restore-20260723-004118.md -->
Status: APPROVED

# Plan: Bonsai-27B Hermès — exhaust optimizations, ship Top 10

**Branch:** main  
**Machine:** Ryzen 9 7845HX + RTX 4060 Laptop 8GB + ~14GB RAM  
**Goal:** Hermès at `-c 65536`, maximize long-ctx tokens/s and multi-turn wall-clock  
**Baseline (FINDINGS.md):** short ~35 t/s; ~20k ctx ~24 t/s; full offload ~5960 MiB; DSpark path not fully measured yet  
**Shipped code:** DSpark IO-tensor sharing + sync KV serialize + async host-commit checkpoint worker (unsafe async D2H dropped)

## Problem statement

Hermès needs 64k context on an 8GB laptop GPU with desktop VRAM tax. We already fixed UVM paging and missing Q1 CUDA. Remaining wins are: fit DSpark without destroying residency, raise accept-length / effective tg at long ctx, cut Hermès checkpoint/PP wall-clock, and squeeze leftover kernel/config headroom — without trading away 64k or re-enabling UVM.

## Success metrics

| Metric | Target |
|---|---|
| Long-ctx (~20k) tg with DSpark | ≥30 t/s (stretch 35) |
| Short-ctx tg | Hold ≥35 t/s |
| Hermès tool-turn TTFT after checkpoint | stay ~ms-class; cut checkpoint-create cost |
| VRAM | Fit target+draft+64k without UVM; no output→CPU wash |
| Correctness | Speculative decode remains lossless vs target |

## Constraints (hard)

1. Keep `-c 65536` (Hermès).
2. Never re-enable `GGML_CUDA_ENABLE_UNIFIED_MEMORY`.
3. Keep `-ngl 99` (output on GPU); never `-ngl ≤64`.
4. Prefer q4_0 + Hadamard KV over TurboQuant chase for this setup.
5. Prefer measured wins over speculative micro-opts.

## Approach

1. Enumerate ~50 candidate optimizations across VRAM fit, speculative decode, Hermès multi-turn, CUDA/kernels, server/config, and measurement.
2. Rank by expected *rate of improvement* for THIS usecase (long-ctx tg × Hermès wall-clock), not generic paper speedups.
3. Autoplan selects Top 10 for implementation in this pass; rest → TODOS.md / deferred.
4. Implement Top 10 with before/after bench protocol in `run_bon.sh` / FINDINGS.

## Candidate catalog (draft — will be ranked in CEO+Eng phases)

### A. Speculative / DSpark (largest expected tg lever)
1. Finish/verify Feature 1: share draft `token_embd`/`output` with target (~1.2–1.4 GiB claim)
2. Download + measure DSpark @20k with default `CPU_LAYERS=0-15`, `NGLD=20`, `CD=4096`
3. Sweep `n_max` / `--spec-autotune` accept length vs long-ctx bandwidth
4. Tune `CPU_LAYERS` park window (0-8 / 0-15 / 0-19) vs graph splits / tg
5. Tune `-ngld` to max layers that still fit after IO share
6. Raise `-cd` if VRAM allows (draft KV quality / accept rate)
7. Ensure draft KV types match efficient path (q4/q8) without double buffers
8. Overlap draft decode with target verify where possible
9. Spec stage config: draft-dspark only vs multi-stage
10. Avoid per-slot draft context duplication (server TODO TAG_SERVER_SPEC_REWORK)

### B. Hermès multi-turn wall-clock
11. Async / cheaper ctx-checkpoint create (FINDINGS: ~200ms/512 tok)
12. Checkpoint interval / eviction tune for tool-heavy sessions
13. Avoid full re-PP on tool turns (`--ctx-checkpoints 32` already; verify harness)
14. Prompt-cache / slot reuse across Hermès turns
15. Shrink checkpoint payload (quantize / compress serialized state)
16. Overlap checkpoint I/O with decode (server already notes stream overlap — verify)

### C. VRAM / residency (enablers, not tg themselves)
17. Confirm IO-share frees claimed GiB in nvidia-smi + load log
18. After free VRAM: raise `UBATCH` 256→512 for faster PP
19. Raise `BATCH` with UBATCH carefully
20. Quit GPU desktop apps (ops, not code) — document in run_bon
21. Keep embd on CPU; never move output to CPU
22. Draft-only weight residency: park unused draft tensors
23. Measure true unique draft weight size (~0.5 vs 1.79 GB pack)

### D. Decode / CUDA / kernels (smaller % but free)
24. sm_89-specific Q1_0_G128 MMVQ polish (~3–8% tg per FINDINGS)
25. Flash-attn path audit at long ctx (already `-fa on`)
26. Graph reuse verify under spec (`--graph-reuse`)
27. Reduce host↔device syncs in speculative verify loop
28. Fuse / reduce copies around Hadamard KV write
29. CUDA graph capture stability (illegal mem access once — repro/fix)
30. Attention batch / `attn_max_batch` if applicable to qwen35 hybrid

### E. Threading / CPU side
31. Retune `-t` / `-tb` for Zen4 under GPU-bound decode
32. Ensure early CPU-parked layers use IQK fast path
33. Pin CPU affinity / avoid oversubscription with Hermès host
34. Lazy CUDA module loading already on — keep

### F. KV / memory bandwidth
35. Confirm q4_0+khad/vhad still best vs q5/q8 at 64k
36. Partial KV offload experiments (likely lose) — measure only if spare
37. Sliding / secondary cache for Hermès cold prefix (if supported)

### G. Server / API / DX
38. Document measured flag matrix in FINDINGS + run_bon help
39. Env presets: `PROFILE=long-dspark` / `PROFILE=short`
40. Fail-loud OOM messages with next lever to try
41. Autotune log line: accept rate, draft ms, verify ms per step
42. Health endpoint exposing VRAM + graph splits + ftype

### H. Measurement protocol (required for ranking truth)
43. Fixed prompt set: 0.5k / 4k / 20k / 40k ctx fill + 256 gen
44. Hermès tool-turn scripted replay (checkpoint hit/miss)
45. nvidia-smi + load-log scrape checklist automation
46. A/B harness for Top 10 with statistical repeats (n≥3)

### I. Explicit non-goals / traps
47. Do NOT shrink ctx below 64k
48. Do NOT re-enable UVM
49. Do NOT park `output.weight` on CPU
50. Do NOT chase TurboQuant unless q4+Hadamard regresses
51. Do NOT `-ngl ≤64` hoping for free VRAM

## Proposed Top 10 (pre-review guess — CEO/Eng will re-rank)

| Rank | ID | Opt | Est. improvement | Effort |
|---|---|---|---|---|
| 1 | A1 | Finish/verify DSpark IO-tensor sharing | Enables draft @64k; unlocks rest | Med |
| 2 | A2 | Measure + tune DSpark long-ctx | +20–40% tg if accept good | Low |
| 3 | A3/A5 | n_max / NGLD / autotune sweep | +5–15% effective tg | Low |
| 4 | A4 | CPU_LAYERS park sweep | Fit vs split trade | Low |
| 5 | B11 | Faster/async checkpoint create | 15–25% Hermès session time | Med-High |
| 6 | C18 | Raise UBATCH after VRAM free | Faster PP / tool turns | Low |
| 7 | A10 | Shared draft context across slots | VRAM + latency | Med |
| 8 | H43–46 | Bench harness | Makes ranking real | Low-Med |
| 9 | D24 | Q1 MMVQ sm_89 polish | +3–8% tg | Med |
| 10 | G41 | Spec timing/accept telemetry | Guides further opts | Low |

## Implementation phases (after approval)

1. Land measurement harness + baseline numbers
2. Complete IO-share; prove VRAM delta
3. DSpark parameter sweeps; lock defaults in `run_bon.sh`
4. Checkpoint create path improvements
5. Kernel polish + DX docs if time

## NOT in scope (yet)

- Upstream PR packaging / API redesign beyond local Hermès needs
- New draft model training
- Desktop compositor / Electron VRAM engineering
- Metal / non-CUDA backends

## What already exists

- `FINDINGS.md` — measured journey + constraints
- `run_bon.sh` — Hermès launch + DSpark env toggles
- CUDA Q1_0_G128 (commit e76f0775)
- WIP: `spec_share_io_tensors`, server-context DSpark wiring, ggml-cuda helpers



## Design doc (office-hours)
APPROVED: /home/cachyfire/.gstack/projects/ikawrakow-ik_llama.cpp/cachyfire-main-design-20260723-005342.md
Approach C: C0 DSpark IO-share → C1 hybrid checkpoint harness → C2 suffix→draft-dspark (validator allow; skip if C1 enough). Sequencing source of truth over prior Top-10 guess.

---

# /autoplan Review Outputs

## Premises (CONFIRMED D14)

1. Success = Hermès session feel vs non-DSpark ~24 t/s @20k
2. Keep 64k, no UVM, ngl 99 / output on GPU
3. Tonight wedge = DSpark @64k (IO-share + measure)
4. Hybrid checkpoint correctness co-equal with DSpark
5. Rank feel → tg; C0→C1→C2(skip-ok)

Mode: SELECTIVE EXPANSION (auto-decided)

## USER CHALLENGE (single-model — Codex unavailable)

**Challenge 1: Laptop-8GB substrate** (from CEO Phase)
You said: Hermès must run on this RTX 4060 8GB laptop at 64k.
CEO subagent recommends: re-evaluate buy used 3090 / cloud GPU before more engineering.
Why: highest $/hour lever; plan never scored hardware alternatives.
What we might be missing: privacy, portability, offline constraint you already live with.
If we're wrong, the cost is: you spend days on IO-share when $700 hardware would erase the problem.
⚠️ Feasibility preference, not a security vulnerability. Your original direction stands unless you explicitly change it.

## Taste decisions (surfaced at gate)

1. **Keep Approach C** (user already chose) vs demote to B — recommend keep C with skip gate.
2. **Async checkpoint worker** — Eng finds it may be no-op for hybrid qwen35. Recommend: measure first; do NOT count in Top 10 until timers prove D2H cost on this model.
3. **Quality smoke** on 1-bit Hermès tool transcripts — recommend Include (small).

## Auto-decided expansions (SELECTIVE)

| Item | Decision | Principle |
|---|---|---|
| Quality smoke (N Hermès turns graded) | Include | P1 completeness |
| Commit WIP to feature branch | Include | P6 action / risk |
| Kill criteria if C0 VRAM/accept fails | Include | P3 pragmatic |
| Hardware buy analysis paragraph | Defer TODOS | User substrate confirmed |
| PLD-only skip DSpark | Skip | Conflicts C0 wedge |
| Async-ckpt as Hermès win | Defer until timed | P5 explicit |

## NOT in scope

- Shrinking ctx below 64k
- Re-enabling UVM
- Parking output.weight on CPU
- TurboQuant KV chase
- Multi-slot DSpark (`-np>1`)
- Upstream PR packaging this pass
- Buying new GPU (deferred; see User Challenge)

## What already exists

| Sub-problem | Existing |
|---|---|
| Q1 CUDA kernels | commit e76f0775 |
| DSpark IO-share | `llama_model_share_dspark_io_tensors`, load skip path |
| Spec chain validator | `common_speculative_validate_chain` |
| Suffix self-spec | `suffix` + `suffix_corpus` |
| Ctx checkpoints | `--ctx-checkpoints 32`, server create/restore |
| Async ckpt worker | WIP in server-context + ggml-cuda (likely unused for hybrid) |
| Launch + VRAM notes | `run_bon.sh`, `FINDINGS.md` |

## Dream state

```
CURRENT          THIS PLAN (C0-C2)           12-MONTH IDEAL
~24 t/s @20k  →  DSpark+ckpt+suffix     →  ≥35 t/s @64k filled
tool re-PP     →  hit restores + p95↓    →  zero-pain multi-hour loops
hand VRAM      →  measured IO-share      →  auto-VRAM pilot
personal notes →  FINDINGS+warns in bin  →  published playbook
```

## CEO DUAL VOICES — CONSENSUS TABLE
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Premises valid?                   No*     N/A    FLAGGED
  2. Right problem to solve?           No*     N/A    USER CHALLENGE
  3. Scope calibration correct?        Unc.    N/A    Uncertain
  4. Alternatives sufficiently explored? No    N/A    FLAGGED
  5. Competitive/market risks covered? No      N/A    FLAGGED
  6. 6-month trajectory sound?         No      N/A    FLAGGED
═══════════════════════════════════════════════════════════════
*CEO: unverified numbers listed as premises; laptop substrate unexamined.
SOURCE: subagent-only (Codex unavailable)
Confirmed: 0/6 | Disagree/flagged: 6 → gate

### CLAUDE SUBAGENT (CEO — strategic independence)
See agent output: substrate challenge, circular premises, quality unmeasured, WIP uncommitted, obsolescence risk.

### CODEX SAYS (CEO)
[codex-unavailable]

## Error & Rescue Registry

| Codepath | What can go wrong | Rescued? | User sees | Fix in Top 10? |
|---|---|---|---|---|
| DSpark init fail | vocab/meta mismatch | Partial | LOG_ERR; ctx leak | Yes #4 |
| Draft GGUF missing | continue w/o DSpark but CPU park on | Silent | slower tg | Yes #2 |
| IO-share dim/type mismatch | wrong logits / fail | Dim only | low accept | Yes #9 |
| Two-stage suffix→dspark | validator fail | Y loud | error string | Yes #7 |
| Hybrid ckpt miss | full re-PP | Silent slow | long TTFT | Yes #3 |
| UVM enabled | paging | Silent | ~10 t/s | Yes #6 |
| ngl≤64 | compute wash | Silent | same VRAM slow | Yes #6 |
| OOM @64k+draft | abort | Hard | crash | C0 fallbacks |

## Failure Modes Registry

| Mode | Severity | Mitigated? |
|---|---|---|
| Async ckpt unused on hybrid | High waste | Measure; demote |
| Silent wrong logits via IO alias | High | type warn + accept telemetry |
| Unbounded DSpark meta OOM | Med | cap metadata |
| flush_checkpoints race | High (if async used) | assert/wrap KV mutators |
| 1-bit quality insufficient | Critical product | quality smoke Include |
| Upstream rebase bitrot | Med | branch + kill criteria |

## CEO Completion Summary

Solid measurement discipline (C0–C2 gates) on a personal Hermès stack. CEO voice challenges the *substrate*; Eng voice finds concrete bugs (run_bon footgun, draft leak, hybrid async no-op). DX voice finds silent UVM/ngl traps. Top 10 below prioritizes measured C0/C1 + cheap correctness/DX before C2/PLD.

**Phase 1 complete.** Codex: unavailable. Claude CEO: 6 flagged dims + 1 User Challenge. Passing to Phase 2.

## Phase 2: Design Review
**Skipped — no UI scope.**

## ENG DUAL VOICES — CONSENSUS TABLE
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Architecture sound?               Unc.    N/A    FLAGGED
  2. Test coverage sufficient?         No      N/A    FLAGGED
  3. Performance risks addressed?      Yes*    N/A    FLAGGED
  4. Security threats covered?         Unc.    N/A    FLAGGED
  5. Error paths handled?              No      N/A    FLAGGED
  6. Deployment risk manageable?       No      N/A    FLAGGED
═══════════════════════════════════════════════════════════════
*Performance risks exist and were identified (not yet fixed).
SOURCE: subagent-only
Confirmed: 0/6

### Architecture ASCII

```
Hermès client
    │ HTTP
    ▼
llama-server ──┬── target model (Q1_0_G128, hybrid qwen35)
               │     ├─ GPU layers (-ngl 99, optional -ot CPU park)
               │     ├─ KV q4+Hadamard @64k
               │     └─ ctx-checkpoints (± async worker?)
               └── draft DSpark ── IO-share ──▶ target embd/lm_head
                         │
                         └── speculative verify (lossless)
                               optional: suffix → draft-dspark (needs validator)
```

### Test diagram (codepaths → coverage)

| Codepath | Test today | Gap → Top 10 |
|---|---|---|
| IO-share load+alias | none | C0 measure script |
| validate_chain combos | none | #10 unit |
| Hermès restore≡rePP | none | #3 harness |
| run_bon missing draft | none | #2 fix + manual |
| flush_checkpoints races | none | defer until async proven |
| suffix→dspark | blocked | #7 after validator |

**Phase 3 complete.** Eng: critical hybrid-async finding + footguns. Passing to DX.

## DX DUAL VOICES — CONSENSUS TABLE
═══════════════════════════════════════════════════════════════
  Dimension                           Claude  Codex  Consensus
  ──────────────────────────────────── ─────── ─────── ─────────
  1. Getting started < 5 min?          No      N/A    FLAGGED
  2. API/CLI naming guessable?         Partial N/A    FLAGGED
  3. Error messages actionable?        Partial N/A    FLAGGED
  4. Docs findable & complete?         Partial N/A    FLAGGED
  5. Upgrade path safe?                Yes     N/A    CONFIRMED*
  6. Dev environment friction-free?    No      N/A    FLAGGED
═══════════════════════════════════════════════════════════════
*Legacy flag migration is strong.
DX overall ~5.4/10. TTHW: ~30–60 min → target <15 with MODEL_PATH env + warns.

### DX Scorecard
Getting started 3 | Docs 6 | API/CLI 7 | Errors 5 | Feedback 5 | Magical 6 | Escapes 7 | Community 4

**Phase 3.5 complete.**

## Cross-Phase Themes

**Theme: Silent failure modes** — CEO (unverified premises), Eng (CPU park without draft, async no-op), DX (UVM/ngl silent). High-confidence: add loud warnings + measurement before clever opts.

**Theme: Hybrid checkpoint truth > more decode polish** — CEO eureka, Eng A1, Design C1. Confirmed across phases.

**Theme: WIP uncommitted risk** — CEO 6-month regret + Eng deployment. Branch now.

## Ranked optimization catalog (50 → impact for THIS usecase)

Est. improvement = Hermès wall-clock / long-ctx tg / risk reduction. Ranks 1–10 = implement this pass.

| Rank | ID | Optimization | Est. rate | Effort |
|---:|---|---|---|---|
| 1 | C0 | Finish IO-share; run DSpark@64k; measure VRAM+tg | Enables all / +20–40% tg if accept | M |
| 2 | B1 | run_bon: no CPU_LAYERS unless draft file exists | Stops silent tg cliff | S |
| 3 | C1 | Hermès replay + ckpt hit/miss + greedy seq identity | Session feel primary | M |
| 4 | A3 | Free draft ctx on speculative init failure | VRAM leak fix | S |
| 5 | G41 | Accept-rate + draft/verify ms telemetry | Makes ranking real | S |
| 6 | DX | Warn UVM on + warn ngl≤n_layer wash | Prevents known cliffs | S |
| 7 | C2 | Allow `suffix→draft-dspark` + tool-schema corpus | Tool-turn PLD whoa | M |
| 8 | C18 | Raise UBATCH after free VRAM | Faster PP/tool | S |
| 9 | D1 | Cap DSpark meta + warn IO ggml_type mismatch | Correctness | S |
| 10 | T1 | validate_chain unit tests + scripted C1 assert | 2am safety | M |
| 11 | Kill | Timebox C0; abort to hardware reeval if fail | Process | S |
| 12 | Qual | Smoke grade 1-bit tool-call quality | Product truth | S |
| 13 | Branch | Commit WIP to feature branch | Risk | S |
| 14 | A2 | Assert flush before KV mutate | Race safety | M |
| 15 | Async | Time hybrid ckpt path; only then optimize | May be no-op | M |
| 16 | n_max | Sweep n_max/NGLD/CPU_LAYERS | +5–15% | S |
| 17 | CD | Raise draft -cd if VRAM allows | Accept | S |
| 18 | Share | Shared draft ctx across slots | VRAM | L |
| 19 | PLD | Skip C2 if C1 enough (gate) | Process | — |
| 20 | Markov | Wire confidence/markov heads | Accept | L |
| 21 | Q1 | sm_89 MMVQ polish | +3–8% tg | M |
| 22 | AsyncC | Async ckpt create for non-hybrid | Session | L |
| 23 | AutoV | Auto-VRAM pilot | DX | L |
| 24 | Docs | Promote FINDINGS into docs/ | DX | S |
| 25 | MODEL | MODEL_PATH env override | DX | S |
| 26 | Health | /health VRAM+splits+ftype | DX | S |
| 27 | Desktop | Quit Vesktop ops note | Ops | S |
| 28 | FA | FA path audit long ctx | Small | S |
| 29 | Graph | Graph-reuse under spec verify | Small | S |
| 30 | Sync | Reduce H↔D sync in verify | Small | M |
| 31 | Threads | Retune -t/-tb Zen4 | Small | S |
| 32 | KV | Re-check q4+had vs q5/q8 | Small | S |
| 33 | Slot | Slot save/restore Hermès | Med | M |
| 34 | Proxy | External KV disk proxy | Med | M |
| 35 | ngram | Other ngram-* vs suffix | Small | S |
| 36 | Presets | PROFILE=long-dspark | DX | S |
| 37 | OOM | Fail-loud next-lever messages | DX | S |
| 38 | Illegal | Repro CUDA illegal mem | Bug | M |
| 39 | Playbook | Publish FINDINGS | C3 | M |
| 40 | Dual | Dual-stage without dspark (suffix→draft) | Weak alt | M |
| 41 | Small | Try smaller Q4 model vs 1-bit 27B | Alt | M |
| 42 | Cloud | Cloud A10 for Hermès | UC | — |
| 43 | 3090 | Buy used 24GB | UC | — |
| 44 | Ctx32k | Shrink ctx (FORBIDDEN) | — | — |
| 45 | UVM | Re-enable UVM (FORBIDDEN) | — | — |
| 46 | ngl64 | Park output CPU (FORBIDDEN) | — | — |
| 47 | TQ | TurboQuant chase (deferred) | — | — |
| 48 | Train | Train better draft | Out | XL |
| 49 | Metal | Non-CUDA backends | Out | — |
| 50 | Multi | Multi-user server | Out | — |

## Top 10 to implement (this pass)

1. C0 DSpark IO-share + measure
2. run_bon draft-file gate for CPU_LAYERS
3. C1 Hermès checkpoint harness
4. Draft ctx free on init fail
5. Spec accept/timing telemetry
6. UVM + ngl≤layer runtime warnings
7. Validator allow suffix→draft-dspark + corpus (skip if C1 enough)
8. UBATCH raise when VRAM allows
9. DSpark meta caps + IO type warn
10. validate_chain tests + scripted C1 assert

## Decision Audit Trail

| # | Phase | Decision | Class | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | CEO | SELECTIVE EXPANSION | Mechanical | autoplan | Default mode | — |
| 2 | CEO | Keep laptop substrate | User Challenge | user default | User confirmed Hermès laptop | Buy GPU now |
| 3 | CEO | Include quality smoke | Mechanical | P1 | Completeness | Skip |
| 4 | CEO | Defer hardware analysis | Mechanical | P3 | User constraint | Expand |
| 5 | CEO | Skip PLD-only | Mechanical | P4/C0 | Keep DSpark wedge | Alt stack |
| 6 | Eng | Demote async-ckpt until timed | Taste | P5 | Hybrid no-op risk | Count as win |
| 7 | Eng | Include B1/A3 fixes | Mechanical | P2 | Blast radius | Defer |
| 8 | DX | Include UVM/ngl warns | Mechanical | P1/P5 | Silent cliffs | Docs only |
| 9 | Eng | C2 after C1 skip gate | Mechanical | Design | User chose C | Force C2 |
| 10 | All | Top 10 as listed | Mechanical | P1+P6 | Impact rank | Micro-opts first |

<!-- AUTONOMOUS DECISION LOG -->


## GSTACK REVIEW REPORT
Status: APPROVED
Via: autoplan
Timestamp: 2026-07-23T12:05:16Z
Commit: e7790591
User: A (as-is); subagent model preference: Auto
User Challenge: laptop substrate — kept (user default)
Taste: demote async-ckpt until timed; keep Approach C
