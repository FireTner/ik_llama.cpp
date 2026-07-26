#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <cmath>

// DSpark: EAGLE-style block-diffusion speculative-decoding drafter, ported from PrismML's
// `dspark` branch (src/models/dspark.cpp upstream) onto ik_llama.cpp's llm graph builder style,
// following the LLM_ARCH_DFLASH_DRAFT precedent in build_dflash.cpp.
//
// Shape of the forward pass (see the upstream dspark.cpp header comment for the full
// derivation, and src/llama-spec-features-dspark.h/.cpp for the runtime side channel):
//
//   1. Multi-layer target-model hidden-state tap: `n_dspark_target_layers` rows captured from
//      the TARGET model (reusing DFlash's llama_set_dflash_capture_layers()/feature-view
//      machinery -- see common/speculative-dspark-impl.h), concatenated to width
//      `n_dspark_target_layers * n_embd`, staged via llama_set_dspark_ctx(), projected down to
//      n_embd via dspark_fc, then RMSNorm'd via dspark_hidden_norm. This is computed ONCE per
//      call (outside the layer loop), matching upstream.
//   2. A small dense Qwen3-style trunk (attn_norm/wq/wk/wv/wo with attn_q_norm+attn_k_norm,
//      ffn_norm/ffn_gate/ffn_up/ffn_down, standard sandwich residual) over the draft block's own
//      residual stream (block_size positions: row 0 is the last-accepted real token, rows
//      1..block_size-1 seeded from mask_token_id).
//   3. At EVERY layer, K/V is the concatenation of (a) the fixed target-context feature from
//      step 1, re-projected fresh through *that* layer's own wk/wv (mathematically identical to
//      upstream's "concat-then-project", since a linear layer distributes over concatenation:
//      k_proj(concat(A,B)) == concat(k_proj(A), k_proj(B))), and (b) the draft block's own
//      current-layer K/V. Q is built ONLY for the draft rows -- the context rows never issue a
//      query in the reference forward, so there is nothing to slice away afterwards (unlike a
//      naive port that computes then discards context-row attention output). Attention is a
//      single, fully non-causal softmax (no mask at all: every draft row attends to the full
//      concatenated K/V, including every other draft row).
//   4. Optional GIDD LogSnrEmbed (log_snr_fc1/fc2): sinusoidal features of the fixed
//      round-1 anchor/mask log-SNR pattern are staged as a graph input, then
//      fc1 → SiLU → fc2 and added to the draft token embeddings BEFORE the layer loop
//      (matches Prism src/models/dspark.cpp). Without this, GGUFs trained with
//      log_snr_conditioning=true draft from the wrong embedding.
//   5. lm_head over the draft rows. Auxiliary heads (markov_head_a/b, confidence_head) are
//      loaded but not built into this graph -- markov resampling is a host-side sequential
//      block-diffusion loop (common/speculative-dspark-impl.h), same as upstream Prism.
//
// Unlike build_dflash(), there is no persistent per-layer K/V cache here: the target-tap
// projection is small and recomputed fresh on every call (exactly what the upstream reference
// forward does too -- see the file header comment there). An incremental cache analogous to
// DFlash's dflash_runtime::kv_runtime_state would be a pure performance optimization and is
// left as future work if profiling shows it's warranted.
ggml_cgraph * llm_build_context::build_dspark() {
    const int64_t n_embd_head_k = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = hparams.n_embd_head_v(0);
    const int64_t n_capture     = hparams.n_dspark_target_layers;
    const int64_t n_embd_cap    = n_capture * n_embd;

    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    GGML_ASSERT(n_capture > 0);
    GGML_ASSERT(hparams.dspark_block_size > 1);
    GGML_ASSERT(model.dspark_fc != nullptr);
    GGML_ASSERT(model.dspark_hidden_norm != nullptr);
    if (hparams.dspark_log_snr_conditioning) {
        GGML_ASSERT(model.dspark_log_snr_fc1 != nullptr && model.dspark_log_snr_fc1_b != nullptr);
        GGML_ASSERT(model.dspark_log_snr_fc2 != nullptr && model.dspark_log_snr_fc2_b != nullptr);
    }

    // Width of the staged target-tap context window (see llama_set_dspark_ctx() /
    // llama-spec-features-dspark.cpp). Zero during buffer-reserve/warmup trial graph builds that
    // happen before any real draft() call has staged context (llama_new_context_with_model's
    // sched_reserve() and similar) -- fall back to a single placeholder row so the graph still
    // exercises the concat/fc/hidden_norm/attention topology it will use for real at runtime.
    const int64_t n_ctx_rows = lctx.dspark.ctx.n_rows > 0 ? (int64_t) lctx.dspark.ctx.n_rows : 1;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0,
            model.max_nodes((int) std::max<int64_t>(n_tokens, n_ctx_rows)) + 32 * n_layer, false);

    // --- stage the raw target-tap context window (graph inputs; values are filled in by
    // llama_prepare_dspark_graph_inputs() right before compute) ---
    lctx.dspark.ctx_feat_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_cap, n_ctx_rows);
    ggml_set_input(lctx.dspark.ctx_feat_tensor);
    cb(lctx.dspark.ctx_feat_tensor, "dspark_ctx_feat", -1);

    lctx.dspark.ctx_pos_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_ctx_rows);
    ggml_set_input(lctx.dspark.ctx_pos_tensor);
    cb(lctx.dspark.ctx_pos_tensor, "dspark_ctx_pos", -1);

    lctx.dspark.log_snr_feat_tensor = nullptr;

    // fc + hidden_norm are applied ONCE for the whole call; the result is re-projected fresh
    // through each layer's own k_proj/v_proj below (it never itself passes through a layer's
    // attn_norm/FFN -- see file header).
    ggml_tensor * target_ctx = llm_build_lora_mm(lctx, ctx0, model.dspark_fc, lctx.dspark.ctx_feat_tensor);
    cb(target_ctx, "dspark_fc", -1);
    target_ctx = llm_build_norm(ctx0, target_ctx, hparams, model.dspark_hidden_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(target_ctx, "dspark_hidden_norm", -1);

    // --- draft-block token embeddings (the trunk residual stream) ---
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // --- GIDD log-SNR conditioning (LogSnrEmbed), Prism-compatible ---
    // Added to draft embeddings BEFORE the layer loop. Sinusoidal features are a pure function
    // of n_tokens / block_size / min/max_log_snr (anchor → max, mask → min); filled by
    // llama_prepare_dspark_graph_inputs().
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq = 128;
        lctx.dspark.log_snr_feat_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_freq, n_tokens);
        ggml_set_input(lctx.dspark.log_snr_feat_tensor);
        cb(lctx.dspark.log_snr_feat_tensor, "dspark_log_snr_feat", -1);

        ggml_tensor * snr_hidden = llm_build_lora_mm(lctx, ctx0, model.dspark_log_snr_fc1, lctx.dspark.log_snr_feat_tensor);
        snr_hidden = ggml_add(ctx0, snr_hidden, model.dspark_log_snr_fc1_b);
        snr_hidden = ggml_silu(ctx0, snr_hidden);
        cb(snr_hidden, "dspark_log_snr_fc1", -1);

        ggml_tensor * snr_embed = llm_build_lora_mm(lctx, ctx0, model.dspark_log_snr_fc2, snr_hidden);
        snr_embed = ggml_add(ctx0, snr_embed, model.dspark_log_snr_fc2_b);
        cb(snr_embed, "dspark_log_snr_fc2", -1);

        inpL = ggml_add(ctx0, inpL, snr_embed);
        cb(inpL, "dspark_draft_embd_snr", -1);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = (n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    const float kq_scale = 1.0f / std::sqrt((float) n_embd_head_k);

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // Q: draft rows only -- the context rows never issue a query in the reference forward
        // (see file header), so there is nothing to compute or slice away for them.
        ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k, n_head, n_tokens);
        Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Qcur, "Qcur_normed", il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_roped", il);

        // K/V for the draft rows, from the (normed) draft residual via this layer's own
        // k_proj/v_proj.
        ggml_tensor * Kcur_draft = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
        ggml_tensor * Vcur_draft = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
        cb(Kcur_draft, "Kcur_draft", il);
        cb(Vcur_draft, "Vcur_draft", il);

        Kcur_draft = ggml_reshape_3d(ctx0, Kcur_draft, n_embd_head_k, n_head_kv, n_tokens);
        Kcur_draft = llm_build_norm(ctx0, Kcur_draft, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Kcur_draft, "Kcur_draft_normed", il);
        Kcur_draft = ggml_rope_ext(ctx0, Kcur_draft, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_draft, "Kcur_draft_roped", il);
        Vcur_draft = ggml_reshape_3d(ctx0, Vcur_draft, n_embd_head_v, n_head_kv, n_tokens);

        // K/V for the context rows, from the SAME layer's k_proj/v_proj applied to the fixed
        // (dspark_fc + dspark_hidden_norm projected) target-tap feature -- recomputed fresh
        // every layer (see file header: mathematically identical to concat-then-project).
        ggml_tensor * Kcur_ctx = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, target_ctx);
        ggml_tensor * Vcur_ctx = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, target_ctx);
        cb(Kcur_ctx, "Kcur_ctx", il);
        cb(Vcur_ctx, "Vcur_ctx", il);

        Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head_k, n_head_kv, n_ctx_rows);
        Kcur_ctx = llm_build_norm(ctx0, Kcur_ctx, hparams, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(Kcur_ctx, "Kcur_ctx_normed", il);
        Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, lctx.dspark.ctx_pos_tensor, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur_ctx, "Kcur_ctx_roped", il);
        Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head_v, n_head_kv, n_ctx_rows);

        // Physical attention layout: [d, row, head]. No mask: attention here is fully
        // non-causal -- every draft query attends to the entire concatenated K/V (context AND
        // every other draft row), so there is nothing to mask out.
        ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        cb(q, "q", il);

        ggml_tensor * k_ctx   = ggml_cont(ctx0, ggml_permute(ctx0, Kcur_ctx, 0, 2, 1, 3));
        ggml_tensor * k_draft = ggml_cont(ctx0, ggml_permute(ctx0, Kcur_draft, 0, 2, 1, 3));
        ggml_tensor * k = ggml_concat(ctx0, k_ctx, k_draft, 1);
        cb(k, "dspark_k_full", il);

        // v_{ctx,draft} land as [d_v, row, head] after permute -- concat along the row dim
        // (dim 1) THEN transpose to [row, d_v, head] (the layout ggml_mul_mat(v, kq) needs),
        // rather than transposing each half first (which would put the varying row-count in
        // dim 0, the wrong axis to concat on).
        ggml_tensor * v_ctx   = ggml_cont(ctx0, ggml_permute(ctx0, Vcur_ctx, 0, 2, 1, 3));
        ggml_tensor * v_draft = ggml_cont(ctx0, ggml_permute(ctx0, Vcur_draft, 0, 2, 1, 3));
        ggml_tensor * v = ggml_concat(ctx0, v_ctx, v_draft, 1);
        v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
        cb(v, "dspark_v_full", il);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, 0.0f);
        cb(kq, "kq_soft_max", il);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);
        ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head_v * n_head, n_tokens);
        cb(cur, "dspark_attn_out", il);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
        cb(cur, "kqv_out", il);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        if (inp_out_ids != nullptr && il == n_layer - 1) {
            // skip computing FFN for unused rows (e.g. the seed row's output is never sampled)
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            cb(cur, "attn_residual_out_rows", -1);
        }

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_up, nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, true);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
