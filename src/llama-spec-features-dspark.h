#pragma once

#include "llama.h"

#include <cstdint>

struct llama_model;

// Public accessors for LLM_ARCH_DSPARK metadata, mirroring the llama_model_dflash_* helpers in
// llama-spec-features-dflash.h. These only expose static, loaded-at-startup hparams/tensor
// presence -- they do not depend on the (not yet implemented) build_dspark() forward graph or on
// any runtime speculative-decoding state.

struct llama_dspark_meta {
    int32_t block_size                  = 0;
    int32_t mask_token_id                = 0;
    int32_t n_target_layers              = 0;
    int32_t markov_rank                  = 0;
    bool    has_confidence_head          = false;
    bool    confidence_head_with_markov  = false;
    bool    log_snr_conditioning         = false;
    float   min_log_snr                  = 0.0f;
    float   max_log_snr                  = 0.0f;
};

// Fills `out_meta` from `model`'s hparams. Returns false (and leaves `out_meta` default) if
// `model` is null or is not an LLM_ARCH_DSPARK model.
bool llama_model_dspark_get_meta(const struct llama_model * model, llama_dspark_meta * out_meta);

int32_t llama_model_dspark_block_size(const struct llama_model * model);
int32_t llama_model_dspark_mask_token_id(const struct llama_model * model);
int32_t llama_model_dspark_n_target_layers(const struct llama_model * model);

// Copies up to `capacity` target-model layer indices into `layer_ids`. Returns the number of
// entries written (0 if `model`/`layer_ids` is null, `capacity` <= 0, or the model isn't dspark).
int32_t llama_model_dspark_target_layer_ids(const struct llama_model * model, int32_t * layer_ids, int32_t capacity);

int32_t llama_model_dspark_markov_rank(const struct llama_model * model);

// Feature 1: IO-tensor sharing, mirroring llama_model_share_dflash_io_tensors(). DSpark's
// token_embd/output are frozen copies of the target model's tensors, so aliasing the drafter's
// (skipped-at-load, hence null) pointers to the target's higher-precision resident tensors is
// numerically safe (can only improve accept rate) and avoids ~1.2-1.4 GiB of duplicated weights.

// Dimension guard: returns true iff `draft_model` is a DSpark model whose IO contract (drafter
// hidden size and vocab width recorded at load) matches the given target (n_embd, n_vocab).
bool llama_model_dspark_io_tensors_match(const struct llama_model * draft_model, int32_t n_embd, int32_t n_vocab);

// Aliases draft_model->tok_embd/output/output_mtp to target_model's resident tensors when they
// were skipped at load (spec_share_io_tensors). No-op returning true for non-DSpark drafters or
// when the drafter already loaded its own IO tensors (self-contained). Returns false only if
// sharing was intended (tensors null) but the target is missing the tensors to alias.
bool llama_model_share_dspark_io_tensors(struct llama_model * draft_model, const struct llama_model * target_model);

// True only if the model both declares confidence_head in its metadata AND the weight tensor
// actually loaded (some exports mark it TENSOR_NOT_REQUIRED).
bool llama_model_dspark_has_confidence_head(const struct llama_model * model);
bool llama_model_dspark_confidence_head_with_markov(const struct llama_model * model);

// True only if markov_rank > 0 AND both markov_head_a/b tensors actually loaded.
bool llama_model_dspark_has_markov_head(const struct llama_model * model);

bool llama_model_dspark_log_snr_conditioning(const struct llama_model * model);
float llama_model_dspark_min_log_snr(const struct llama_model * model);
float llama_model_dspark_max_log_snr(const struct llama_model * model);

// Runtime side channel for the non-causal target-tap context window that build_dspark() (see
// src/graphs/build_dspark.cpp) reads from every graph build. `ctx_features` is the raw,
// multi-layer-concatenated tap feature matrix -- the SAME row format the DFlash capture
// callback produces (llama_set_dflash_capture_layers() + llama_spec_get_dflash_feature_view*()),
// i.e. row i is [layer_0_hidden(n_embd) | layer_1_hidden(n_embd) | ...] for
// `n_target_layers` layers, width n_floats/n_rows == n_target_layers * target_n_embd.
// The data is copied into `ctx`, so the caller's buffers do not need to outlive this call.
bool llama_set_dspark_ctx(
        struct llama_context * ctx,
        const float * ctx_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * ctx_positions);

// Clears the staged context window (e.g. on begin()/context-shift-to-empty).
void llama_reset_dspark_ctx(struct llama_context * ctx);

// Copies the staged `ctx` window into build_dspark()'s graph input tensors right before compute.
// Called from the main decode loop (llama.cpp) for LLM_ARCH_DSPARK contexts, analogous to
// llama_prepare_dflash_graph_inputs() for LLM_ARCH_DFLASH_DRAFT.
bool llama_prepare_dspark_graph_inputs(struct llama_context & lctx, uint32_t n_tokens);
