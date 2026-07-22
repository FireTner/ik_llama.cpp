#include "llama-spec-features-dspark.h"

// llama-sampling.h (pulled in via llama-context.h) uses std::mt19937 without including
// <random> itself; other translation units happen to pull it in transitively first, but
// nothing here does, so include it explicitly before llama-context.h.
#include <random>

#include "llama-arch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>

static bool llama_model_is_dspark(const struct llama_model * model) {
    return model != nullptr && model->arch == LLM_ARCH_DSPARK;
}

bool llama_model_dspark_get_meta(const struct llama_model * model, llama_dspark_meta * out_meta) {
    if (out_meta == nullptr) {
        return false;
    }

    *out_meta = llama_dspark_meta{};

    if (!llama_model_is_dspark(model)) {
        return false;
    }

    const auto & hparams = model->hparams;
    out_meta->block_size                 = (int32_t) hparams.dspark_block_size;
    out_meta->mask_token_id               = (int32_t) hparams.dspark_mask_token_id;
    out_meta->n_target_layers             = (int32_t) hparams.n_dspark_target_layers;
    out_meta->markov_rank                 = (int32_t) hparams.dspark_markov_rank;
    out_meta->has_confidence_head         = llama_model_dspark_has_confidence_head(model);
    out_meta->confidence_head_with_markov = hparams.dspark_confidence_head_with_markov;
    out_meta->log_snr_conditioning        = hparams.dspark_log_snr_conditioning;
    out_meta->min_log_snr                 = hparams.dspark_min_log_snr;
    out_meta->max_log_snr                 = hparams.dspark_max_log_snr;

    return true;
}

int32_t llama_model_dspark_block_size(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? (int32_t) model->hparams.dspark_block_size : 0;
}

int32_t llama_model_dspark_mask_token_id(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? (int32_t) model->hparams.dspark_mask_token_id : -1;
}

int32_t llama_model_dspark_n_target_layers(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? (int32_t) model->hparams.n_dspark_target_layers : 0;
}

int32_t llama_model_dspark_target_layer_ids(
        const struct llama_model * model,
        int32_t * layer_ids,
        int32_t capacity) {
    if (!llama_model_is_dspark(model) || layer_ids == nullptr || capacity <= 0) {
        return 0;
    }

    const int32_t n = std::min<int32_t>((int32_t) model->hparams.n_dspark_target_layers, capacity);
    for (int32_t i = 0; i < n; ++i) {
        layer_ids[i] = (int32_t) model->hparams.dspark_target_layers[i];
    }

    return n;
}

int32_t llama_model_dspark_markov_rank(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? (int32_t) model->hparams.dspark_markov_rank : 0;
}

bool llama_model_dspark_has_confidence_head(const struct llama_model * model) {
    return llama_model_is_dspark(model) &&
           model->hparams.dspark_confidence_head &&
           model->dspark_confidence_head != nullptr;
}

bool llama_model_dspark_confidence_head_with_markov(const struct llama_model * model) {
    return llama_model_is_dspark(model) && model->hparams.dspark_confidence_head_with_markov;
}

bool llama_model_dspark_has_markov_head(const struct llama_model * model) {
    return llama_model_is_dspark(model) &&
           model->hparams.dspark_markov_rank > 0 &&
           model->dspark_markov_head_a != nullptr &&
           model->dspark_markov_head_b != nullptr;
}

bool llama_model_dspark_log_snr_conditioning(const struct llama_model * model) {
    return llama_model_is_dspark(model) && model->hparams.dspark_log_snr_conditioning;
}

float llama_model_dspark_min_log_snr(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? model->hparams.dspark_min_log_snr : 0.0f;
}

float llama_model_dspark_max_log_snr(const struct llama_model * model) {
    return llama_model_is_dspark(model) ? model->hparams.dspark_max_log_snr : 0.0f;
}

bool llama_set_dspark_ctx(
        struct llama_context * ctx,
        const float * ctx_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * ctx_positions) {
    if (ctx == nullptr || n_rows <= 0 || ctx_features == nullptr || n_floats == 0 || ctx_positions == nullptr) {
        return false;
    }

    if (n_floats % (size_t) n_rows != 0) {
        return false;
    }

    ctx->dspark.ctx.feat_owned.assign(ctx_features, ctx_features + n_floats);
    ctx->dspark.ctx.pos_owned.assign(ctx_positions, ctx_positions + n_rows);
    ctx->dspark.ctx.n_rows = n_rows;

    return true;
}

void llama_reset_dspark_ctx(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dspark.ctx.feat_owned.clear();
    ctx->dspark.ctx.pos_owned.clear();
    ctx->dspark.ctx.n_rows = 0;
}

bool llama_prepare_dspark_graph_inputs(struct llama_context & lctx, uint32_t n_tokens) {
    GGML_UNUSED(n_tokens);

    ggml_tensor * ctx_feat_tensor = lctx.dspark.ctx_feat_tensor;
    ggml_tensor * ctx_pos_tensor  = lctx.dspark.ctx_pos_tensor;
    if (ctx_feat_tensor == nullptr || ctx_pos_tensor == nullptr) {
        LLAMA_LOG_ERROR("%s: DSpark graph inputs are not initialized\n", __func__);
        return false;
    }

    const int32_t n_rows = lctx.dspark.ctx.n_rows;
    const int64_t graph_n_rows = ctx_feat_tensor->ne[1];

    if (n_rows > 0 && (int64_t) n_rows == graph_n_rows) {
        if ((int64_t) lctx.dspark.ctx.feat_owned.size() != ggml_nelements(ctx_feat_tensor)) {
            LLAMA_LOG_ERROR("%s: DSpark staged feature size mismatch (staged=%zu expected=%lld)\n",
                    __func__, lctx.dspark.ctx.feat_owned.size(), (long long) ggml_nelements(ctx_feat_tensor));
            return false;
        }

        ggml_backend_tensor_set(ctx_feat_tensor, lctx.dspark.ctx.feat_owned.data(), 0, ggml_nbytes(ctx_feat_tensor));
        ggml_backend_tensor_set(ctx_pos_tensor, lctx.dspark.ctx.pos_owned.data(), 0, ggml_nbytes(ctx_pos_tensor));
    } else {
        // Buffer-reserve/warmup trial graph (see build_dspark()'s `have_staged_ctx` fallback):
        // no real context has been staged yet, or the staged window doesn't match this
        // particular graph build's placeholder shape. Zero-fill so the (unused) computation is
        // at least well-defined.
        std::vector<float> zero_feat((size_t) ggml_nelements(ctx_feat_tensor), 0.0f);
        std::vector<int32_t> zero_pos((size_t) ggml_nelements(ctx_pos_tensor), 0);
        ggml_backend_tensor_set(ctx_feat_tensor, zero_feat.data(), 0, ggml_nbytes(ctx_feat_tensor));
        ggml_backend_tensor_set(ctx_pos_tensor, zero_pos.data(), 0, ggml_nbytes(ctx_pos_tensor));
    }

    return true;
}
