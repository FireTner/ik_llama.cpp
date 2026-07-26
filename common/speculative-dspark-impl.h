#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

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
struct common_speculative_state_dspark : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    llama_batch batch = {};

    int32_t block_size      = 0;
    int32_t mask_token_id   = -1;
    int32_t n_target_layers = 0;
    int32_t n_embd_cap      = 0; // n_target_layers * target hidden size == capture row width
    int32_t cross_ctx       = 0; // max number of target-tap rows kept in the window
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

        batch = llama_batch_init(std::max(1, block_size), 0, 1);
        ctx_window.reserve((size_t) this->cross_ctx * (size_t) n_embd_cap);
        ctx_window_pos.reserve((size_t) this->cross_ctx);
        ready = true;

        LOG_INF("%s: DSpark context ready (n_ctx=%d, block_size=%d, cross_ctx=%d, n_target_layers=%d, n_embd_cap=%d)\n",
                __func__, llama_n_ctx(ctx_dft), block_size, this->cross_ctx, n_target_layers, n_embd_cap);
    }

    ~common_speculative_state_dspark() override {
        llama_clear_dflash_capture(ctx_tgt);
        llama_reset_dspark_ctx(ctx_dft);
        if (ctx_dft) {
            llama_free(ctx_dft);
        }
        if (batch.token != nullptr) {
            llama_batch_free(batch);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        llama_kv_cache_clear(ctx_dft);
        ctx_window.clear();
        ctx_window_pos.clear();
        last_target_pos = -1;
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

        const int32_t n_keep = std::min<int32_t>(params.n_max, block_size - 1);
        if (n_keep <= 0) {
            return;
        }

        if (!llama_set_dspark_ctx(ctx_dft, ctx_window.data(), ctx_window.size(),
                    (int32_t) ctx_window_pos.size(), ctx_window_pos.data())) {
            LOG_ERR("%s: failed to stage DSpark target context features\n", __func__);
            return;
        }

        llama_kv_cache_clear(ctx_dft);
        batch.n_tokens = 0;
        const int32_t batch_len = n_keep + 1;
        const llama_pos draft_pos_base = last_target_pos >= 0 ? last_target_pos + 1 : (llama_pos) ctx_window_pos.size();
        const llama_pos seed_pos = last_target_pos >= 0 ? last_target_pos : draft_pos_base - 1;

        // Row 0 is the last-accepted real token (the anchor / least-noisy position in the
        // reference's block-diffusion formulation); rows 1..n_keep are seeded from
        // mask_token_id and are what we actually want lm_head logits for.
        common_batch_add(batch, id_last, seed_pos, { 0 }, false);
        for (int32_t i = 1; i < batch_len; ++i) {
            common_batch_add(batch, mask_token_id, draft_pos_base + (i - 1), { 0 }, true);
        }

        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_ERR("%s: llama_decode() failed for DSpark draft batch\n", __func__);
            batch.n_tokens = 0;
            return;
        }

        // Greedy sampling from the base trunk logits. This is the "sampling from lm_head
        // logits without markov" fallback called out as an acceptable starting point in the
        // port instructions -- the markov_head_a/b resample and confidence_head gating that
        // upstream Prism applies host-side on top of these logits are not implemented yet (the
        // weights are loaded -- see llama_model_dspark_has_markov_head() -- but unused).
        result.reserve((size_t) n_keep);
        for (int32_t i = 0; i < n_keep; ++i) {
            const llama_token id = common_sampler_sample_speculative(nullptr, ctx_dft, i + 1, nullptr);
            result.push_back(id);
        }

        batch.n_tokens = 0;
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};
