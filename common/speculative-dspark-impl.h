#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifdef LLAMA_DSPARK_MARKOV_CUDA
#include "dspark-markov.h"
#include <cuda_runtime.h>
#endif

// DSpark speculative-decoding state: block-diffusion drafting against a bounded window of
// multi-layer target hidden-state taps. Modeled on common_speculative_state_dflash
// (speculative-dflash-impl.h) but much simpler at the runtime level: DSpark's forward pass
// (src/graphs/build_dspark.cpp) has no persistent per-layer K/V cache, so this state only needs
// to track a plain bounded window of captured target-tap rows and re-stage it (via
// llama_set_dspark_ctx()) before every draft() call.
//
// Target-side capture reuses DFlash's multi-layer hidden-state capture callback verbatim
// (llama_set_dflash_capture_layers() / llama_spec_get_dflash_feature_view_for_seq()): both
// drafters tap a fixed, ordered set of TARGET layers and want the SAME per-row format (each
// row is the per-layer hidden states concatenated in target-layer order), so there is no need
// for a separate DSpark-specific capture mechanism.
//
// Host-side sampling (matches Prism common/speculative.cpp + docs/dspark-scope.md):
//   1. Always forward a full block_size draft (anchor = id_last + masks). Bidirectional
//      block attention was trained at that width; shrinking the batch changes the distribution.
//   2. Sequential vanilla Markov resample over the first n_use = min(n_max, block_size)
//      positions (next-token alignment: position 0 / anchor predicts the first draft token):
//          correction[v] = sum_r w1[prev*R+r] * w2[v*R+r]
//          step_logit[v] = base_logits[k][v] + correction[v]
//          out[k] = argmax_v step_logit[v]   (lowest v wins ties)
//      where prev is id_last for k==0 and out[k-1] for k>0. Never batch markov over mask ids.
//   3. Optional confidence gating (when confidence_head is present and p_min > 0):
//          c_k = sigmoid(W · concat(h_k, w1[prev]) + b)   // with_markov (this GGUF)
//          a_j = ∏_{i<=j} c_i                             // prefix survival (paper §3.2.1)
//      Truncate the draft at the first j where a_j < p_min; always keep >= 1 token if any
//      were sampled. Empty drafts still fall back cleanly (caller treats as no draft).
struct common_speculative_state_dspark : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    llama_batch batch = {};

    int32_t block_size      = 0;
    int32_t mask_token_id   = -1;
    int32_t n_target_layers = 0;
    int32_t n_embd_cap      = 0; // n_target_layers * target hidden size == capture row width
    int32_t cross_ctx       = 0; // max number of target-tap rows kept in the window
    int64_t n_vocab         = 0;
    int64_t n_embd          = 0;
    bool    ready           = false;

    std::vector<int32_t> target_layer_ids;

    // Bounded window of the most-recent target-tap rows (concatenated multi-layer format,
    // width n_embd_cap) and their absolute target-sequence positions. Unlike DFlash's ring
    // buffer + incremental-KV-cache-transition machinery, this is just plain data that gets
    // fully re-staged into the graph's input tensors on every draft() call (see
    // src/graphs/build_dspark.cpp's file header for why that's fine here).
    std::vector<float>     ctx_window;
    std::vector<llama_pos> ctx_window_pos;

    llama_pos last_target_pos = -1;

    // Host-resident Markov / confidence heads (dense f32). Loaded once in the ctor.
    std::vector<float> markov_w1; // [n_vocab * rank], rank fastest
    std::vector<float> markov_w2;
    int64_t markov_rank = 0;
    bool    has_markov  = false;

    std::vector<float> conf_w;
    float   conf_bias = 0.0f;
    int64_t conf_in   = 0;
    bool    has_confidence = false;
    bool    conf_with_markov = false;

#ifdef LLAMA_DSPARK_MARKOV_CUDA
    bool                        markov_use_cuda = false;
    struct dspark_markov_cuda * markov_cuda     = nullptr;
#endif

    common_speculative_state_dspark(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            int32_t cross_ctx)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , cross_ctx(std::max(1, cross_ctx))
    {
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = llama_get_model(ctx_dft);

        // Prism DSpark GGUFs intentionally ship tokenizer.ggml.model=none and share the
        // target's token_embd/output (IO-share). DFlash's tokenizer-text check rejects that
        // as a vocab-type mismatch; for NONE drafts only n_vocab must agree.
        {
            const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
            const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
            const bool draft_has_no_vocab = llama_vocab_type(vocab_dft) == LLAMA_VOCAB_TYPE_NONE;
            bool compatible = false;
            if (draft_has_no_vocab) {
                const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
                const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
                compatible = (n_vocab_tgt == n_vocab_dft);
                if (!compatible) {
                    LOG_ERR("%s: DSpark draft has no tokenizer but vocab size mismatch (target=%d draft=%d)\n",
                            __func__, n_vocab_tgt, n_vocab_dft);
                }
            } else {
                compatible = common_speculative_are_dflash_compatible(model_tgt, model_dft);
                if (!compatible) {
                    LOG_ERR("%s: DSpark draft model vocab/tokenizer is incompatible with the target model\n", __func__);
                }
            }
            if (!compatible) {
                return;
            }
        }

        llama_dspark_meta meta;
        if (!llama_model_dspark_get_meta(model_dft, &meta)) {
            LOG_ERR("%s: draft model is not a DSpark model\n", __func__);
            return;
        }

        block_size      = meta.block_size;
        mask_token_id   = meta.mask_token_id;
        n_target_layers = meta.n_target_layers;
        markov_rank     = meta.markov_rank;
        conf_with_markov = meta.confidence_head_with_markov;
        n_embd          = llama_model_n_embd(model_dft);
        n_vocab         = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));

        if (block_size <= 1 || mask_token_id < 0 || n_target_layers <= 0) {
            LOG_ERR("%s: invalid DSpark metadata (block_size=%d, mask_token_id=%d, n_target_layers=%d)\n",
                    __func__, block_size, mask_token_id, n_target_layers);
            return;
        }

        target_layer_ids.resize((size_t) n_target_layers);
        if (llama_model_dspark_target_layer_ids(model_dft, target_layer_ids.data(), n_target_layers) != n_target_layers) {
            LOG_ERR("%s: failed to read DSpark target layer ids\n", __func__);
            target_layer_ids.clear();
            return;
        }

        const int32_t target_hidden_size = llama_model_n_embd(model_tgt);
        if (target_hidden_size <= 0) {
            LOG_ERR("%s: invalid target hidden size\n", __func__);
            return;
        }
        n_embd_cap = target_hidden_size * n_target_layers;

        const int32_t n_target_model_layers = llama_n_layer(model_tgt);
        for (int32_t layer_id : target_layer_ids) {
            if (layer_id < 0 || layer_id >= n_target_model_layers) {
                LOG_ERR("%s: invalid DSpark target layer id %d for target model with %d layers\n",
                        __func__, layer_id, n_target_model_layers);
                target_layer_ids.clear();
                return;
            }
        }

        // CRITICAL: enable multi-layer hidden-state capture on the TARGET context. Forgetting
        // this call is the #1 way a DSpark drafter would silently draft from stale/garbage
        // context features (this is called out explicitly in the port instructions -- the
        // upstream Prism CLI itself forgets to wire this up).
        if (!llama_set_dflash_capture_layers(ctx_tgt, target_layer_ids.data(), (int32_t) target_layer_ids.size())) {
            LOG_ERR("%s: failed to configure DSpark target capture callback\n", __func__);
            return;
        }

        has_markov = markov_rank > 0 && llama_model_dspark_get_markov(model_dft, markov_w1, markov_w2);
        if (markov_rank > 0 && !has_markov) {
            LOG_WRN("%s: dspark reports markov_rank=%lld but markov head weights could not be read "
                    "-- block logits will NOT be markov-corrected (greedy lm_head only)\n",
                    __func__, (long long) markov_rank);
        }

#ifdef LLAMA_DSPARK_MARKOV_CUDA
        // Device path needs ~2 * n_vocab * rank * 4 bytes (~485 MiB for Bonsai) resident.
        // On 8GB Hermès stacks the draft per-step buffer already ate most headroom — only
        // attempt CUDA when nvidia free memory looks sufficient; else OpenMP host path.
        bool want_cuda_markov = has_markov;
        if (const char * env = std::getenv("LLAMA_DSPARK_MARKOV_CUDA")) {
            const char c = env[0];
            want_cuda_markov = want_cuda_markov && !(c == '0' || c == 'n' || c == 'N' || c == 'f' || c == 'F');
        }
        if (want_cuda_markov) {
            size_t free_b = 0, total_b = 0;
            const size_t need_b = (size_t) n_vocab * (size_t) markov_rank * sizeof(float) * 2
                    + (size_t) block_size * (size_t) n_vocab * sizeof(float) // base logits scratch
                    + (32ull << 20); // ~32 MiB slack for partials / fragmentation
            if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b < need_b) {
                LOG_INF("%s: - device markov resample (CUDA) skipped (free=%.0f MiB need=%.0f MiB) -- host OpenMP path\n",
                        __func__, free_b / (1024.0 * 1024.0), need_b / (1024.0 * 1024.0));
                want_cuda_markov = false;
            }
        }
        if (want_cuda_markov) {
            markov_cuda     = dspark_markov_cuda_init(markov_w1.data(), markov_w2.data(), n_vocab, markov_rank);
            markov_use_cuda = markov_cuda != nullptr;
            LOG_INF("%s: - device markov resample (CUDA) %s\n", __func__,
                    markov_use_cuda ? "ENABLED (default; LLAMA_DSPARK_MARKOV_CUDA=0 to disable)"
                                    : "FAILED to init -- using host path");
        }
#endif

        has_confidence = llama_model_dspark_get_confidence(model_dft, conf_w, conf_bias, conf_in);
        if (meta.has_confidence_head && !has_confidence) {
            LOG_WRN("%s: dspark declares confidence_head but weights could not be read -- "
                    "confidence gating disabled\n", __func__);
        }
        if (has_confidence) {
            const int64_t expect = n_embd + (conf_with_markov ? markov_rank : 0);
            if (conf_in != expect) {
                LOG_WRN("%s: confidence_head width %lld != expected %lld (n_embd=%lld markov_rank=%lld "
                        "with_markov=%d) -- disabling confidence gating\n",
                        __func__, (long long) conf_in, (long long) expect,
                        (long long) n_embd, (long long) markov_rank, (int) conf_with_markov);
                has_confidence = false;
            }
        }

        batch = llama_batch_init(std::max(1, block_size), 0, 1);
        ctx_window.reserve((size_t) this->cross_ctx * (size_t) n_embd_cap);
        ctx_window_pos.reserve((size_t) this->cross_ctx);
        ready = true;

        LOG_INF("%s: DSpark context ready (n_ctx=%d, block_size=%d, cross_ctx=%d, n_target_layers=%d, "
                "n_embd_cap=%d, markov_rank=%lld, has_markov=%d, has_confidence=%d, conf_with_markov=%d)\n",
                __func__, llama_n_ctx(ctx_dft), block_size, this->cross_ctx, n_target_layers, n_embd_cap,
                (long long) markov_rank, (int) has_markov, (int) has_confidence, (int) conf_with_markov);
    }

    ~common_speculative_state_dspark() override {
        llama_clear_dflash_capture(ctx_tgt);
        llama_reset_dspark_ctx(ctx_dft);
#ifdef LLAMA_DSPARK_MARKOV_CUDA
        if (markov_cuda != nullptr) {
            dspark_markov_cuda_free(markov_cuda);
            markov_cuda = nullptr;
        }
#endif
        if (ctx_dft) {
            llama_free(ctx_dft);
        }
        if (batch.token != nullptr) {
            llama_batch_free(batch);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        // Match DFlash: reset only the draft-side state. Target-tap rows are populated by
        // prompt warmup (common_speculative_on_target_seq_batch) before begin() runs; clearing
        // them here leaves draft() with an empty window and #gen drafts = 0 forever.
        // Full target-feature reset belongs in clear_target_features() / clear_sequence_hidden().
        llama_kv_cache_clear(ctx_dft);
        llama_reset_dspark_ctx(ctx_dft);
    }

    // Appends (or, if the new rows alone fill the window, replaces) captured target-tap rows,
    // keeping only the most recent `cross_ctx` rows.
    bool append_target_features(const common_speculative_feature_view & features, llama_seq_id seq_id) {
        if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width != n_embd_cap || features.rows.empty()) {
            return false;
        }

        std::vector<float>     new_rows;
        std::vector<llama_pos> new_positions;
        new_rows.reserve(features.rows.size() * (size_t) n_embd_cap);
        new_positions.reserve(features.rows.size());

        for (const auto & row : features.rows) {
            if (row.seq_id != seq_id || row.data == nullptr) {
                continue;
            }
            new_positions.push_back(row.pos);
            new_rows.insert(new_rows.end(), row.data, row.data + n_embd_cap);
        }

        if (new_positions.empty()) {
            return false;
        }

        const int32_t n_new = (int32_t) new_positions.size();
        if (n_new >= cross_ctx) {
            const int32_t keep_from = n_new - cross_ctx;
            ctx_window_pos.assign(new_positions.begin() + keep_from, new_positions.end());
            ctx_window.assign(new_rows.begin() + (ptrdiff_t) keep_from * n_embd_cap, new_rows.end());
        } else {
            const int32_t keep_old_rows = std::min<int32_t>((int32_t) ctx_window_pos.size(), cross_ctx - n_new);
            if (keep_old_rows < (int32_t) ctx_window_pos.size()) {
                const int32_t drop = (int32_t) ctx_window_pos.size() - keep_old_rows;
                ctx_window_pos.erase(ctx_window_pos.begin(), ctx_window_pos.begin() + drop);
                ctx_window.erase(ctx_window.begin(), ctx_window.begin() + (ptrdiff_t) drop * n_embd_cap);
            }
            ctx_window_pos.insert(ctx_window_pos.end(), new_positions.begin(), new_positions.end());
            ctx_window.insert(ctx_window.end(), new_rows.begin(), new_rows.end());
        }

        last_target_pos = ctx_window_pos.back();
        return true;
    }

    void context_shift(llama_pos kv_keep, llama_pos kv_discard, llama_pos kv_past) {
        if (kv_discard <= 0 || ctx_window_pos.empty()) {
            return;
        }

        const llama_pos discard_begin = kv_keep;
        const llama_pos discard_end   = kv_keep + kv_discard;

        std::vector<float>     shifted_rows;
        std::vector<llama_pos> shifted_positions;
        shifted_rows.reserve(ctx_window.size());
        shifted_positions.reserve(ctx_window_pos.size());

        for (size_t row = 0; row < ctx_window_pos.size(); ++row) {
            llama_pos pos = ctx_window_pos[row];
            if (pos >= discard_begin && pos < discard_end) {
                continue;
            }
            if (pos >= discard_end && pos < kv_past) {
                pos -= kv_discard;
            }
            const float * src = ctx_window.data() + row * (size_t) n_embd_cap;
            shifted_rows.insert(shifted_rows.end(), src, src + n_embd_cap);
            shifted_positions.push_back(pos);
        }

        ctx_window = std::move(shifted_rows);
        ctx_window_pos = std::move(shifted_positions);
        last_target_pos = ctx_window_pos.empty() ? -1 : ctx_window_pos.back();
    }

    void clear_target_features() {
        ctx_window.clear();
        ctx_window_pos.clear();
        last_target_pos = -1;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(prompt_tgt);

        result.clear();
        if (!ready || ctx_window_pos.empty()) {
            return;
        }

        // Cap how many positions we *verify*; backbone always runs full block_size.
        const int32_t n_use = std::min<int32_t>(params.n_max, block_size);
        if (n_use <= 0) {
            return;
        }

        if (!llama_set_dspark_ctx(ctx_dft, ctx_window.data(), ctx_window.size(),
                    (int32_t) ctx_window_pos.size(), ctx_window_pos.data())) {
            LOG_ERR("%s: failed to stage DSpark target context features\n", __func__);
            return;
        }

        llama_kv_cache_clear(ctx_dft);
        batch.n_tokens = 0;
        const llama_pos draft_pos_base = last_target_pos >= 0 ? last_target_pos + 1 : (llama_pos) ctx_window_pos.size();
        const llama_pos seed_pos = last_target_pos >= 0 ? last_target_pos : draft_pos_base - 1;

        // Full block_size forward (Prism / mlx): row 0 = last-accepted anchor (logits ON —
        // next-token alignment: this row predicts the first draft token), rows 1..block_size-1
        // seeded from mask_token_id. Shrinking the block would change bidirectional attention.
        common_batch_add(batch, id_last, seed_pos, { 0 }, true);
        for (int32_t i = 1; i < block_size; ++i) {
            common_batch_add(batch, mask_token_id, draft_pos_base + (i - 1), { 0 }, true);
        }

        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed for DSpark draft batch\n", __func__);
            batch.n_tokens = 0;
            return;
        }

        const float * logits_base = llama_get_logits(ctx_dft);
        if (logits_base == nullptr) {
            LOG_ERR("%s: llama_get_logits(ctx_dft) returned null\n", __func__);
            batch.n_tokens = 0;
            return;
        }

        // Embeddings (= result_norm) are extracted alongside logits for LLM_ARCH_DSPARK so the
        // confidence head can run host-side. Missing embd only disables gating, not drafting.
        const float * embd_base = has_confidence ? llama_get_embeddings(ctx_dft) : nullptr;

        result.reserve((size_t) n_use);
        llama_token prev_token = id_last;

        bool did_cuda = false;
#ifdef LLAMA_DSPARK_MARKOV_CUDA
        if (markov_use_cuda && has_markov) {
            static_assert(sizeof(llama_token) == sizeof(int32_t),
                    "dspark cuda markov path assumes llama_token == int32_t");
            result.resize((size_t) n_use);
            if (dspark_markov_cuda_resample(markov_cuda, logits_base, (int32_t) id_last,
                                            n_use, (int32_t *) result.data())) {
                did_cuda = true;
            } else {
                LOG_WRN("%s: cuda markov resample failed -- falling back to the host path\n", __func__);
                markov_use_cuda = false;
                result.clear();
            }
        }
#endif

        if (!did_cuda) {
            for (int32_t k = 0; k < n_use; ++k) {
                if (k > 0 && prev_token == mask_token_id) {
                    static bool warned_host_mask = false;
                    if (!warned_host_mask) {
                        LOG_WRN("%s: dspark markov resample chained a mask_token_id prev at k=%d -- "
                                "drafter emitted the mask sentinel; the target verify will reject it\n",
                                __func__, k);
                        warned_host_mask = true;
                    }
                }

                const float * base_logits = logits_base + (size_t) k * (size_t) n_vocab;

                llama_token best_id = 0;
                float       best_v  = -std::numeric_limits<float>::infinity();

                if (has_markov) {
                    const float * emb = markov_w1.data() + (size_t) prev_token * (size_t) markov_rank;
#ifdef _OPENMP
                    // Parallel argmax over vocab. Critical for Bonsai (V≈248k, R=256): the
                    // serial loop is ~200ms/position and starves decode; OpenMP brings it
                    // in line with draft-forward cost when CUDA markov won't fit in VRAM.
                    #pragma omp parallel
                    {
                        llama_token thr_best_id = 0;
                        float       thr_best_v  = -std::numeric_limits<float>::infinity();
                        #pragma omp for schedule(static) nowait
                        for (int64_t v = 0; v < n_vocab; ++v) {
                            const float * w2row = markov_w2.data() + (size_t) v * (size_t) markov_rank;
                            float bias = 0.0f;
                            for (int64_t r = 0; r < markov_rank; ++r) {
                                bias += emb[r] * w2row[r];
                            }
                            const float logit = base_logits[v] + bias;
                            if (logit > thr_best_v) {
                                thr_best_v  = logit;
                                thr_best_id = (llama_token) v;
                            }
                        }
                        #pragma omp critical
                        {
                            if (thr_best_v > best_v || (thr_best_v == best_v && thr_best_id < best_id)) {
                                best_v  = thr_best_v;
                                best_id = thr_best_id;
                            }
                        }
                    }
#else
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        const float * w2row = markov_w2.data() + (size_t) v * (size_t) markov_rank;
                        float bias = 0.0f;
                        for (int64_t r = 0; r < markov_rank; ++r) {
                            bias += emb[r] * w2row[r];
                        }
                        const float logit = base_logits[v] + bias;
                        if (logit > best_v) {
                            best_v  = logit;
                            best_id = (llama_token) v;
                        }
                    }
#endif
                } else {
                    for (int64_t v = 0; v < n_vocab; ++v) {
                        if (base_logits[v] > best_v) {
                            best_v  = base_logits[v];
                            best_id = (llama_token) v;
                        }
                    }
                }

                result.push_back(best_id);
                prev_token = best_id; // chain the SAMPLED token, never mask_token_id
            }
        }

        // Confidence gating (mlx-dspark / paper §3.2.1): truncate when prefix survival
        // a_j = ∏ c_i drops below p_min. p_min <= 0 disables (explicit opt-out).
        if (has_confidence && embd_base != nullptr && params.p_min > 0.0f && !result.empty()) {
            float surv = 1.0f;
            int32_t keep = 0;
            prev_token = id_last;
            for (int32_t k = 0; k < (int32_t) result.size(); ++k) {
                const float * h = embd_base + (size_t) k * (size_t) n_embd;
                float logit = conf_bias;
                // W · h
                for (int64_t i = 0; i < n_embd; ++i) {
                    logit += conf_w[(size_t) i] * h[i];
                }
                if (conf_with_markov && has_markov) {
                    const float * memb = markov_w1.data() + (size_t) prev_token * (size_t) markov_rank;
                    for (int64_t r = 0; r < markov_rank; ++r) {
                        logit += conf_w[(size_t) (n_embd + r)] * memb[r];
                    }
                }
                // sigmoid
                const float c = 1.0f / (1.0f + expf(-logit));
                surv *= c;
                if (surv < params.p_min) {
                    break;
                }
                keep = k + 1;
                prev_token = result[(size_t) k];
            }
            if (keep <= 0) {
                keep = 1; // always propose >= 1 when the head fired at all
            }
            result.resize((size_t) keep);
        }

        batch.n_tokens = 0;
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};
