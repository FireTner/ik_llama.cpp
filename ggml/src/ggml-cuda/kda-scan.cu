#include "kda-scan.cuh"

static __global__ void kda_scan_f32_kernel(
        const float * __restrict__ s0,
        const float * __restrict__ decay,
        const float * __restrict__ write,
        const int32_t * __restrict__ sq,
        float * __restrict__ out_states,
        float * __restrict__ out_s,
        const int R,
        const int n_tokens,
        const int n_kv,
        const int s0_s1,
        const int decay_s1,
        const int write_s1,
        const int sq_s1) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) {
        return;
    }

    // Seed local state vector for all seqs (this rank only).
    // For typical decode n_kv==1; for multi-seq copy priors.
    for (int kv = 0; kv < n_kv; ++kv) {
        out_s[kv * R + r] = s0[kv * s0_s1 + r];
    }

    for (int t = 0; t < n_tokens; ++t) {
        const int32_t seq0 = sq[t * sq_s1];
        const float d = decay[t * decay_s1 + r];
        const float w = write[t * write_s1 + r];
        float * s_seq = out_s + seq0 * R;
        const float st = d * s_seq[r] + w;
        s_seq[r] = st;
        out_states[t * R + r] = st;

        for (int k = 1; k < n_kv; ++k) {
            const int32_t seq = sq[t * sq_s1 + k];
            if (0 <= seq && seq < n_kv && seq != seq0) {
                out_s[seq * R + r] = st;
            }
        }
    }
}

void ggml_cuda_op_kda_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // s
    const ggml_tensor * src1 = dst->src[1]; // decay
    const ggml_tensor * src2 = dst->src[2]; // write
    const ggml_tensor * src3 = dst->src[3]; // sq

    const int R        = src0->ne[0];
    const int n_kv     = src0->ne[1];
    const int n_tokens = src1->ne[1];

    GGML_ASSERT((int64_t) R * n_tokens + (int64_t) R * n_kv == ggml_nelements(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    float * dst_data = (float *) dst->data;
    float * out_states = dst_data;
    float * out_s = dst_data + (size_t) R * n_tokens;

    const int block = 256;
    const int grid  = (R + block - 1) / block;

    kda_scan_f32_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const float *) src0->data,
            (const float *) src1->data,
            (const float *) src2->data,
            (const int32_t *) src3->data,
            out_states,
            out_s,
            R, n_tokens, n_kv,
            (int) (src0->nb[1] / sizeof(float)),
            (int) (src1->nb[1] / sizeof(float)),
            (int) (src2->nb[1] / sizeof(float)),
            (int) (src3->nb[1] / sizeof(int32_t)));
}
