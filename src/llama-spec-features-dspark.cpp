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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

bool llama_model_dspark_io_tensors_match(const struct llama_model * draft_model, int32_t n_embd, int32_t n_vocab) {
    if (!llama_model_is_dspark(draft_model) || n_embd <= 0 || n_vocab <= 0) {
        return false;
    }

    // The drafter operates in the target's embedding space, so its hidden size must equal the
    // target's for the shared token_embd/output to be valid.
    if ((int32_t) draft_model->hparams.n_embd != n_embd) {
        return false;
    }

    // Preferred: validate against the vocab width peeked from the drafter's token_embd metadata at
    // load time (recorded even when the tensor buffer itself was skipped for sharing).
    if (draft_model->dspark_io_n_vocab > 0) {
        return (int32_t) draft_model->dspark_io_n_vocab == n_vocab;
    }

    // Fallback: the drafter actually loaded its own IO tensors -- check their dims directly.
    if (draft_model->tok_embd != nullptr && draft_model->output != nullptr) {
        return (int32_t) draft_model->tok_embd->ne[0] == n_embd &&
               (int32_t) draft_model->tok_embd->ne[1] == n_vocab &&
               (int32_t) draft_model->output->ne[0]   == n_embd &&
               (int32_t) draft_model->output->ne[1]   == n_vocab;
    }

    return false;
}

bool llama_model_share_dspark_io_tensors(struct llama_model * draft_model, const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr) {
        return false;
    }

    // Only ever alias for DSpark drafters; leave every other arch untouched.
    if (draft_model->arch != LLM_ARCH_DSPARK) {
        return true;
    }

    // If the drafter loaded its own IO tensors (sharing was not requested / not skipped), keep the
    // self-contained tensors as-is -- nothing to alias.
    if (draft_model->tok_embd != nullptr && draft_model->output != nullptr) {
        return true;
    }

    const int32_t n_embd  = (int32_t) draft_model->hparams.n_embd;
    const int32_t n_vocab = draft_model->dspark_io_n_vocab > 0
        ? (int32_t) draft_model->dspark_io_n_vocab
        : 0;

    if (draft_model->tok_embd == nullptr) {
        if (target_model->tok_embd == nullptr) {
            return false;
        }
        if (n_embd > 0 && n_vocab > 0 &&
            ((int32_t) target_model->tok_embd->ne[0] != n_embd ||
             (int32_t) target_model->tok_embd->ne[1] != n_vocab)) {
            LLAMA_LOG_ERROR("%s: target token_embd shape [%lld, %lld] != draft contract [%d, %d]\n",
                    __func__,
                    (long long) target_model->tok_embd->ne[0], (long long) target_model->tok_embd->ne[1],
                    n_embd, n_vocab);
            return false;
        }
        draft_model->tok_embd = target_model->tok_embd;
    }

    if (draft_model->output == nullptr) {
        // Prefer the target's dedicated output (lm_head); fall back to its tok_embd if tied.
        struct ggml_tensor * tgt_out = target_model->output ? target_model->output : target_model->tok_embd;
        if (tgt_out == nullptr) {
            return false;
        }
        if (n_embd > 0 && n_vocab > 0 &&
            ((int32_t) tgt_out->ne[0] != n_embd || (int32_t) tgt_out->ne[1] != n_vocab)) {
            LLAMA_LOG_ERROR("%s: target output shape [%lld, %lld] != draft contract [%d, %d]\n",
                    __func__,
                    (long long) tgt_out->ne[0], (long long) tgt_out->ne[1],
                    n_embd, n_vocab);
            return false;
        }
        draft_model->output = tgt_out;
    }

    // build_dspark() uses model.output for the lm_head projection; keep output_mtp consistent with
    // the dense arches (points at output) so any shared code path stays well-defined.
    if (draft_model->output_mtp == nullptr) {
        draft_model->output_mtp = draft_model->output;
    }

    if (draft_model->tok_embd == nullptr || draft_model->output == nullptr) {
        return false;
    }

    // Higher-precision target IO is intentional (can only help accept rate). Log so operators can
    // confirm the alias and spot accidental type surprises during bring-up.
    LLAMA_LOG_INFO("%s: DSpark IO share ok: tok_embd type=%s output type=%s (n_embd=%d n_vocab=%d)\n",
            __func__,
            ggml_type_name(draft_model->tok_embd->type),
            ggml_type_name(draft_model->output->type),
            n_embd, n_vocab);

    return true;
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

// Dequantize / copy a model weight tensor into a dense f32 host buffer. Mirrors Prism's
// llama_model_dspark_get_markov helper (supports f32/f16/bf16 + quantized types with to_float).
static bool llama_dspark_tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        return false;
    }

    const int64_t n = ggml_nelements(t);
    out.resize((size_t) n);

    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(t, out.data(), 0, (size_t) n * sizeof(float));
            return true;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp((size_t) n);
            ggml_backend_tensor_get(t, tmp.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
            return true;
        }
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp((size_t) n);
            ggml_backend_tensor_get(t, tmp.data(), 0, (size_t) n * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), n);
            return true;
        }
        default: {
            if (!ggml_is_quantized(t->type)) {
                LLAMA_LOG_ERROR("%s: unsupported dspark aux-head tensor type %s\n",
                        __func__, ggml_type_name(t->type));
                return false;
            }
            const ggml_type_traits_t qtype = ggml_internal_get_type_traits(t->type);
            if (qtype.to_float == nullptr) {
                LLAMA_LOG_ERROR("%s: quantized type %s has no dequantizer\n",
                        __func__, ggml_type_name(t->type));
                return false;
            }
            std::vector<uint8_t> raw(ggml_nbytes(t));
            ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
            qtype.to_float(raw.data(), out.data(), n);
            return true;
        }
    }
}

bool llama_model_dspark_get_markov(
        const struct llama_model * model,
        std::vector<float> & w1,
        std::vector<float> & w2) {
    if (!llama_model_dspark_has_markov_head(model)) {
        return false;
    }

    const ggml_tensor * a = model->dspark_markov_head_a;
    const ggml_tensor * b = model->dspark_markov_head_b;
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) {
        LLAMA_LOG_ERROR("%s: markov_head_a/b shape mismatch ([%lld,%lld] vs [%lld,%lld])\n",
                __func__,
                (long long) a->ne[0], (long long) a->ne[1],
                (long long) b->ne[0], (long long) b->ne[1]);
        return false;
    }

    return llama_dspark_tensor_to_f32(a, w1) && llama_dspark_tensor_to_f32(b, w2);
}

bool llama_model_dspark_get_confidence(
        const struct llama_model * model,
        std::vector<float> & weight,
        float & bias,
        int64_t & conf_in) {
    bias = 0.0f;
    conf_in = 0;
    if (!llama_model_dspark_has_confidence_head(model)) {
        return false;
    }

    const ggml_tensor * w = model->dspark_confidence_head;
    conf_in = w->ne[0];
    if (w->ne[1] != 1 || conf_in <= 0) {
        LLAMA_LOG_ERROR("%s: unexpected confidence_head shape [%lld, %lld]\n",
                __func__, (long long) w->ne[0], (long long) w->ne[1]);
        return false;
    }

    if (!llama_dspark_tensor_to_f32(w, weight)) {
        return false;
    }

    if (model->dspark_confidence_head_b != nullptr) {
        std::vector<float> b;
        if (!llama_dspark_tensor_to_f32(model->dspark_confidence_head_b, b) || b.empty()) {
            return false;
        }
        bias = b[0];
    }

    return true;
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

    // GIDD LogSnrEmbed: host-side sinusoidal featurization matching Prism
    // src/models/dspark.cpp (anchor pos % block_size == 0 → max_log_snr, else min).
    ggml_tensor * log_snr_feat = lctx.dspark.log_snr_feat_tensor;
    if (log_snr_feat != nullptr) {
        const llama_hparams & hp = lctx.model.hparams;
        const int64_t n_freq = log_snr_feat->ne[0];
        const int64_t n_draft = log_snr_feat->ne[1];
        if (n_freq != 128) {
            LLAMA_LOG_ERROR("%s: unexpected LogSnrEmbed n_freq=%lld (expected 128)\n",
                    __func__, (long long) n_freq);
            return false;
        }
        if (n_tokens > 0 && (int64_t) n_tokens != n_draft) {
            LLAMA_LOG_ERROR("%s: LogSnrEmbed n_draft mismatch (n_tokens=%u tensor=%lld)\n",
                    __func__, n_tokens, (long long) n_draft);
            return false;
        }

        const int64_t half    = n_freq / 2;
        const float   min_snr = hp.dspark_min_log_snr;
        const float   max_snr = hp.dspark_max_log_snr;
        const int64_t bsz     = hp.dspark_block_size > 0 ? (int64_t) hp.dspark_block_size : n_draft;

        std::vector<float> feat((size_t) (n_freq * n_draft));
        for (int64_t pos = 0; pos < n_draft; ++pos) {
            const float log_snr = (pos % bsz == 0) ? max_snr : min_snr;
            const float t       = (log_snr - min_snr) / (max_snr - min_snr) * 1000.0f;
            for (int64_t i = 0; i < half; ++i) {
                const float freq  = expf(-logf(10000.0f) * (float) i / (float) half);
                const float angle = t * freq;
                feat[(size_t) (pos * n_freq + i)]        = sinf(angle);
                feat[(size_t) (pos * n_freq + half + i)] = cosf(angle);
            }
        }
        ggml_backend_tensor_set(log_snr_feat, feat.data(), 0, ggml_nbytes(log_snr_feat));
    }

    return true;
}
