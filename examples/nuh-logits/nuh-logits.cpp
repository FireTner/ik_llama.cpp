// Dump per-token logits for a fixed token id sequence (Nuh parity harness).
// Build (from ik_llama.cpp root, after ./build_native.sh):
//   g++ -O2 -std=c++17 -Iinclude -Iggml/include \
//     examples/nuh-logits/nuh-logits.cpp \
//     -Lbuild/src -Lbuild/ggml/src -lllama -lggml -Wl,-rpath,$PWD/build/src:$PWD/build/ggml/src \
//     -o build/bin/nuh-logits
//
// Usage:
//   ./build/bin/nuh-logits -m model.gguf -t 3,100,250 -o /tmp/logits.bin [-ngl 99]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf -t id0,id1,... -o logits.bin [-ngl N] [-c CTX]\n"
            "  writes float32 logits [T, n_vocab] row-major\n",
            argv0);
}

static std::vector<llama_token> parse_tokens(const char * s) {
    std::vector<llama_token> out;
    const char * p = s;
    while (*p) {
        char * end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        out.push_back((llama_token) v);
        p = end;
        if (*p == ',') {
            ++p;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * tokens_arg = nullptr;
    const char * out_path = nullptr;
    int ngl = 99;
    int n_ctx = 256;

    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((!strcmp(argv[i], "-t") || !strcmp(argv[i], "--tokens")) && i + 1 < argc) {
            tokens_arg = argv[++i];
        } else if ((!strcmp(argv[i], "-o") || !strcmp(argv[i], "--out")) && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "-ngl") && i + 1 < argc) {
            ngl = atoi(argv[++i]);
        } else if ((!strcmp(argv[i], "-c") || !strcmp(argv[i], "--ctx")) && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !tokens_arg || !out_path) {
        usage(argv[0]);
        return 1;
    }

    auto tokens = parse_tokens(tokens_arg);
    if (tokens.empty()) {
        fprintf(stderr, "no tokens parsed from '%s'\n", tokens_arg);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = (uint32_t) tokens.size();
    cparams.n_ubatch = (uint32_t) tokens.size();
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_tok = (int) tokens.size();

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n_tok;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "failed to open %s\n", out_path);
        return 1;
    }

    // header: n_tok, n_vocab (int32)
    int32_t hdr[2] = { n_tok, n_vocab };
    fwrite(hdr, sizeof(int32_t), 2, f);

    for (int i = 0; i < n_tok; ++i) {
        float * logits = llama_get_logits_ith(ctx, i);
        if (!logits) {
            fprintf(stderr, "null logits at i=%d\n", i);
            return 1;
        }
        fwrite(logits, sizeof(float), n_vocab, f);
    }
    fclose(f);

    fprintf(stderr, "wrote %s  shape=[%d,%d]\n", out_path, n_tok, n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
