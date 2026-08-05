#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>

// Nuh hybrid: KDA (recurrent) layers + MLA (NoPE GQA) every mla_every-th layer.

static ggml_tensor * nuh_as_f32(ggml_context * ctx, ggml_tensor * t) {
    if (!t || t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static ggml_tensor * nuh_build_kda(
        llm_build_context & llm,
        ggml_context * ctx0,
        ggml_cgraph * gf,
        ggml_tensor * inp,
        ggml_tensor * inp_s_seq,
        ggml_tensor * inp_out_ids,
        int il,
        const llm_build_cb & cb) {
    auto & lctx    = llm.lctx;
    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    auto & layer   = model.layers[il];

    const int64_t n_embd   = hparams.n_embd;
    const int64_t kda_rank = hparams.nuh_kda_rank;
    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t n_tok    = inp->ne[1];
    const int64_t conv_state_dim = (d_conv - 1) * n_embd;
    const int64_t state_dim = (int64_t) hparams.n_embd_v_s();

    GGML_ASSERT(kv_self.s_l[il] != nullptr);
    GGML_ASSERT((int64_t) kv_self.s_l[il]->ne[0] >= state_dim);
    GGML_ASSERT(layer.ssm_conv1d && layer.ssm_in && layer.ssm_dt && layer.ssm_dt_b && layer.ssm_out);
    GGML_ASSERT(layer.nuh_a_diag && layer.nuh_p && layer.nuh_q);

    ggml_tensor * residual = inp;

    ggml_tensor * cur = llm_build_context::llm_build_norm(
            ctx0, inp, hparams, nuh_as_f32(ctx0, layer.attn_norm), nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "kda_norm", il);

    // Single local recurrent slot (same convention as qwen3next).
    const bool reset_state = llm.batch.pos != nullptr && llm.batch.pos[0] == 0;
    ggml_tensor * state_dst = ggml_view_2d(ctx0, kv_self.s_l[il], state_dim, 1, kv_self.s_l[il]->nb[1], 0);
    ggml_tensor * state_f32 = reset_state ? ggml_scale(ctx0, state_dst, 0.0f) : state_dst;

    ggml_tensor * conv_flat = ggml_view_1d(ctx0, state_f32, conv_state_dim, 0);
    ggml_tensor * s_flat    = ggml_view_1d(ctx0, state_f32, kda_rank, conv_state_dim * sizeof(float));

    // Causal depthwise conv (OpenPangu / ssm_conv pattern).
    ggml_tensor * conv_w = ggml_reshape_2d(ctx0, ggml_cast(ctx0, layer.ssm_conv1d, GGML_TYPE_F32), d_conv, n_embd);
    ggml_tensor * conv_states = ggml_reshape_3d(ctx0, conv_flat, d_conv - 1, n_embd, 1);
    ggml_tensor * x_f32 = ggml_cast(ctx0, cur, GGML_TYPE_F32);

    ggml_tensor * conv_raw = ggml_ssm_conv(ctx0, conv_states, x_f32, conv_w, inp_s_seq, nullptr);
    cb(conv_raw, "kda_conv_raw", il);

    ggml_tensor * conv_out = ggml_view_2d(ctx0, conv_raw, n_embd, n_tok, n_embd * sizeof(float), 0);
    conv_out = ggml_cast(ctx0, conv_out, cur->type);
    cb(conv_out, "kda_conv", il);

    // New conv taps: skip x_out and the first (shifted-out) column of {d_conv, D}.
    ggml_tensor * new_conv = ggml_view_2d(ctx0, conv_raw, d_conv - 1, n_embd,
            d_conv * sizeof(float),
            (1 + n_embd * n_tok) * sizeof(float));
    new_conv = ggml_cont(ctx0, new_conv);
    new_conv = ggml_reshape_1d(ctx0, new_conv, conv_state_dim);

    // qkv / decay / write
    ggml_tensor * qkv = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_in, conv_out);
    cb(qkv, "kda_qkv", il);

    const size_t es = ggml_element_size(qkv);
    ggml_tensor * q0 = ggml_view_2d(ctx0, qkv, kda_rank, n_tok, qkv->nb[1], 0);
    ggml_tensor * k0 = ggml_view_2d(ctx0, qkv, kda_rank, n_tok, qkv->nb[1], kda_rank * es);
    ggml_tensor * v0 = ggml_view_2d(ctx0, qkv, kda_rank, n_tok, qkv->nb[1], 2 * kda_rank * es);

    ggml_tensor * q = ggml_silu(ctx0, ggml_cont(ctx0, q0));
    ggml_tensor * k = ggml_silu(ctx0, ggml_cont(ctx0, k0));
    ggml_tensor * write = ggml_mul(ctx0, k, ggml_cont(ctx0, v0));
    cb(write, "kda_write0", il);

    ggml_tensor * decay_logits = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_dt, conv_out);
    decay_logits = ggml_add(ctx0, decay_logits, nuh_as_f32(ctx0, layer.ssm_dt_b));
    // dplr = sum(P * Q, dim=-1); P,Q are {4, R} in ggml
    ggml_tensor * dplr = ggml_sum_rows(ctx0, ggml_mul(ctx0, nuh_as_f32(ctx0, layer.nuh_p), nuh_as_f32(ctx0, layer.nuh_q))); // {1, R}
    dplr = ggml_reshape_1d(ctx0, ggml_cont(ctx0, dplr), kda_rank);
    decay_logits = ggml_add(ctx0, decay_logits, nuh_as_f32(ctx0, layer.nuh_a_diag));
    decay_logits = ggml_add(ctx0, decay_logits, dplr);
    ggml_tensor * decay = ggml_sigmoid(ctx0, decay_logits);
    cb(decay, "kda_decay", il);

    if (hparams.nuh_surprise_gate && layer.nuh_surprise_proj != nullptr) {
        ggml_tensor * s_logits = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.nuh_surprise_proj, conv_out);
        s_logits = ggml_add(ctx0, s_logits, nuh_as_f32(ctx0, layer.nuh_surprise_proj_b));
        // energy = mean(write^2) over R → broadcast to [R, T]
        ggml_tensor * energy = ggml_mean(ctx0, ggml_sqr(ctx0, write)); // {1, T}
        ggml_tensor * e_sc = ggml_repeat(ctx0, nuh_as_f32(ctx0, layer.nuh_surprise_energy), energy);
        energy = ggml_mul(ctx0, energy, e_sc);
        energy = ggml_repeat(ctx0, energy, s_logits);
        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, s_logits, energy));
        write = ggml_mul(ctx0, write, gate);
        cb(write, "kda_write_gated", il);
    }

    decay = ggml_cont(ctx0, ggml_cast(ctx0, decay, GGML_TYPE_F32));
    write = ggml_cont(ctx0, ggml_cast(ctx0, write, GGML_TYPE_F32));
    ggml_tensor * s_in = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_cast(ctx0, s_flat, GGML_TYPE_F32)), kda_rank, 1);

    ggml_tensor * scan_raw = ggml_kda_scan(ctx0, s_in, decay, write, inp_s_seq);
    cb(scan_raw, "kda_scan_raw", il);

    ggml_tensor * states = ggml_view_2d(ctx0, scan_raw, kda_rank, n_tok, kda_rank * sizeof(float), 0);
    ggml_tensor * new_s  = ggml_view_1d(ctx0, scan_raw, kda_rank, kda_rank * n_tok * sizeof(float));

    ggml_tensor * new_state = ggml_concat(ctx0, new_conv, ggml_cont(ctx0, new_s), 0);
    GGML_ASSERT(new_state->ne[0] == state_dim);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, state_dst));

    states = ggml_cast(ctx0, states, q->type);
    ggml_tensor * mixed = ggml_mul(ctx0, q, states);
    ggml_tensor * out = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ssm_out, mixed);
    cb(out, "kda_out", il);

    if (inp_out_ids) {
        out = ggml_get_rows(ctx0, out, inp_out_ids);
        residual = ggml_get_rows(ctx0, residual, inp_out_ids);
    }

    out = ggml_add(ctx0, out, residual);
    cb(out, "kda_residual", il);
    return out;
}

static ggml_tensor * nuh_build_mla(
        llm_build_context & llm,
        ggml_context * ctx0,
        ggml_cgraph * gf,
        ggml_tensor * inp,
        ggml_tensor * KQ_mask,
        ggml_tensor * inp_out_ids,
        int il,
        const llm_build_cb & cb) {
    auto & lctx    = llm.lctx;
    auto & model   = lctx.model;
    auto & hparams = model.hparams;
    auto & kv_self = lctx.kv_self;
    auto & layer   = model.layers[il];

    const int64_t n_head    = hparams.n_head(il);
    const int64_t n_head_kv = hparams.n_head_kv(il);
    const int64_t head_dim  = hparams.n_embd_head_k(il);
    const int64_t n_tok     = inp->ne[1];
    const float kq_scale    = 1.0f / sqrtf(float(head_dim));

    ggml_tensor * residual = inp;

    ggml_tensor * cur = llm_build_context::llm_build_norm(
            ctx0, inp, hparams, nuh_as_f32(ctx0, layer.attn_norm), nullptr, LLM_NORM_RMS, cb, il);
    cb(cur, "mla_norm", il);

    ggml_tensor * q = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
    q = ggml_reshape_3d(ctx0, q, head_dim, n_head, n_tok);
    cb(q, "mla_q", il);

    ggml_tensor * latent = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wkv_a_mqa, cur);
    cb(latent, "mla_latent", il);

    ggml_tensor * k = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wk_b, latent);
    k = ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_tok);
    cb(k, "mla_k", il);

    ggml_tensor * v = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.wv_b, latent);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_tok);
    cb(v, "mla_v", il);

    cur = llm_build_context::llm_build_kv(ctx0, lctx, kv_self, gf,
            layer.wo, nullptr,
            k, v, q, KQ_mask,
            (int32_t) n_tok, llm.kv_head, llm.n_kv, kq_scale, cb, il);
    cb(cur, "mla_attn_out", il);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        residual = ggml_get_rows(ctx0, residual, inp_out_ids);
    }

    cur = ggml_add(ctx0, cur, residual);
    cb(cur, "mla_residual", il);
    return cur;
}

ggml_cgraph * llm_build_context::build_nuh() {
    ggml_cgraph * gf = new_graph_custom();

    GGML_ASSERT(hparams.n_embd_head_v(0) == hparams.n_embd_head_k(0));
    GGML_ASSERT(hparams.nuh_kda_rank > 0);

    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * ids = (il == n_layer - 1) ? inp_out_ids : nullptr;

        if (hparams.is_recurrent(il)) {
            cur = nuh_build_kda(*this, ctx0, gf, inpL, lctx.inp_s_seq_qnext, ids, il, cb);
        } else {
            cur = nuh_build_mla(*this, ctx0, gf, inpL, KQ_mask, ids, il, cb);
        }

        cur = llm_build_ffn(ctx0, lctx, nuh_as_f32(ctx0, model.layers[il].ffn_norm), cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, /*add_input=*/true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    // Final RMSNorm needs F32 weight for CUDA fused path
    ggml_tensor * out_norm = nuh_as_f32(ctx0, model.output_norm);
    cur = build_output(lctx, ctx0, inpL, model.output, out_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
