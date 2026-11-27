#include "bitrev/vk_bitrev.h"
#include "common/math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void lut_destroy(VkBitRevLutState* lut) {
    vk_buf_destroy(&lut->pairs_buf);
    vk_buf_destroy(&lut->idx_buf);
    for (int i = 0; i < 2; ++i) {
        vk_buf_destroy(&lut->rev_w[i]);
        vk_buf_destroy(&lut->b_pairs[i]);
        vk_buf_destroy(&lut->b_all_rev[i]);
        lut->nb_pairs[i] = 0;
    }
    lut->npairs = 0;
}

static bool lut_ensure(VkBitRevLutState* lut, VkBitRevLutKind kind) {
    if (kind == VK_BITREV_LUT_NONE || lut->logN <= 0 || !lut->ctx) return true;
    size_t N = (size_t)1u << lut->logN;
    VkCmdBatch up = {0};
    if (!vk_batch_init(&up, lut->ctx, 0)) return false;
    bool ok = true;

    if (kind == VK_BITREV_LUT_PAIRS) {
        if (lut->pairs_buf.buffer != VK_NULL_HANDLE) { vk_batch_destroy(&up); return true; }
        if (lut->logN <= 16) {
            uint32_t* packed = (uint32_t*)malloc((N / 2 + 1) * sizeof(uint32_t));
            if (!packed) { vk_batch_destroy(&up); return false; }
            size_t cnt = 0;
            for (uint32_t i = 0; i < (uint32_t)N; ++i) {
                uint32_t j = vk_bit_reverse(i, lut->logN);
                if (i < j) packed[cnt++] = (i & 0xFFFFu) | ((j & 0xFFFFu) << 16);
            }
            lut->npairs = cnt;
            lut->pairs_packed_u16 = true;
            ok = vk_buf_create(lut->ctx, cnt * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->pairs_buf) &&
                 vk_batch_upload_sync(&up, packed, vk_buf_slice(&lut->pairs_buf, 0, cnt * sizeof(uint32_t)), cnt * sizeof(uint32_t));
            free(packed);
        } else {
            uint32_t* host = (uint32_t*)malloc(N * sizeof(uint32_t));
            if (!host) { vk_batch_destroy(&up); return false; }
            size_t cnt = 0;
            for (uint32_t i = 0; i < (uint32_t)N; ++i) {
                uint32_t j = vk_bit_reverse(i, lut->logN);
                if (i < j) { host[2 * cnt] = i; host[2 * cnt + 1] = j; ++cnt; }
            }
            lut->npairs = cnt;
            lut->pairs_packed_u16 = false;
            ok = vk_buf_create(lut->ctx, 2 * cnt * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->pairs_buf) &&
                 vk_batch_upload_sync(&up, host, vk_buf_slice(&lut->pairs_buf, 0, 2 * cnt * sizeof(uint32_t)), 2 * cnt * sizeof(uint32_t));
            free(host);
        }
        vk_batch_destroy(&up);
        return ok;
    }

    if (kind == VK_BITREV_LUT_IDX) {
        if (lut->idx_buf.buffer != VK_NULL_HANDLE) { vk_batch_destroy(&up); return true; }
        if (lut->logN <= 16) {
            size_t nwords = (N + 1) / 2;
            uint32_t* packed = (uint32_t*)calloc(nwords, sizeof(uint32_t));
            if (!packed) { vk_batch_destroy(&up); return false; }
            for (uint32_t i = 0; i < (uint32_t)N; ++i) {
                uint32_t r = vk_bit_reverse(i, lut->logN) & 0xFFFFu;
                if ((i & 1u) == 0u) packed[i >> 1] |= r;
                else packed[i >> 1] |= (r << 16);
            }
            lut->idx_packed_u16 = true;
            ok = vk_buf_create(lut->ctx, nwords * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->idx_buf) &&
                 vk_batch_upload_sync(&up, packed, vk_buf_slice(&lut->idx_buf, 0, nwords * sizeof(uint32_t)), nwords * sizeof(uint32_t));
            free(packed);
        } else {
            uint32_t* host = (uint32_t*)malloc(N * sizeof(uint32_t));
            if (!host) { vk_batch_destroy(&up); return false; }
            for (uint32_t i = 0; i < (uint32_t)N; ++i) host[i] = vk_bit_reverse(i, lut->logN);
            lut->idx_packed_u16 = false;
            ok = vk_buf_create(lut->ctx, N * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->idx_buf) &&
                 vk_batch_upload_sync(&up, host, vk_buf_slice(&lut->idx_buf, 0, N * sizeof(uint32_t)), N * sizeof(uint32_t));
            free(host);
        }
        vk_batch_destroy(&up);
        return ok;
    }

    if (kind == VK_BITREV_LUT_COBRA_W4 || kind == VK_BITREV_LUT_COBRA_W5) {
        int w = (kind == VK_BITREV_LUT_COBRA_W4) ? 4 : 5;
        int wi = w - 4;
        if (lut->rev_w[wi].buffer != VK_NULL_HANDLE) { vk_batch_destroy(&up); return true; }
        int m = lut->logN - 2 * w;
        if (m < 0 || m > 16) { vk_batch_destroy(&up); return false; }
        uint32_t W = 1u << w;
        uint32_t num_b = 1u << m;
        size_t rw_words = (W + 1) / 2;
        size_t ba_words = (num_b + 1) / 2;
        uint32_t* h_rw = (uint32_t*)calloc(rw_words, sizeof(uint32_t));
        uint32_t* h_ba = (uint32_t*)calloc(ba_words, sizeof(uint32_t));
        uint32_t* h_bp = (uint32_t*)malloc((num_b + 64) * sizeof(uint32_t));
        if (!h_rw || !h_ba || !h_bp) {
            free(h_rw); free(h_ba); free(h_bp);
            vk_batch_destroy(&up);
            return false;
        }
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t rx = vk_bit_reverse(x, w) & 0xFFFFu;
            if ((x & 1u) == 0u) h_rw[x >> 1] |= rx;
            else h_rw[x >> 1] |= (rx << 16);
        }
        size_t nbp = 0;
        for (uint32_t b = 0; b < num_b; ++b) {
            uint32_t brev = (m == 0) ? 0u : vk_bit_reverse(b, m);
            if ((b & 1u) == 0u) h_ba[b >> 1] |= (brev & 0xFFFFu);
            else h_ba[b >> 1] |= ((brev & 0xFFFFu) << 16);
            if (b <= brev) h_bp[nbp++] = (b & 0xFFFFu) | ((brev & 0xFFFFu) << 16);
        }
        lut->nb_pairs[wi] = nbp;
        ok = vk_buf_create(lut->ctx, rw_words * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->rev_w[wi]) &&
             vk_buf_create(lut->ctx, nbp * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->b_pairs[wi]) &&
             vk_buf_create(lut->ctx, ba_words * sizeof(uint32_t), VK_MEM_DEVICE_LOCAL, &lut->b_all_rev[wi]) &&
             vk_batch_upload_sync(&up, h_rw, vk_buf_slice(&lut->rev_w[wi], 0, rw_words * sizeof(uint32_t)), rw_words * sizeof(uint32_t)) &&
             vk_batch_upload_sync(&up, h_bp, vk_buf_slice(&lut->b_pairs[wi], 0, nbp * sizeof(uint32_t)), nbp * sizeof(uint32_t)) &&
             vk_batch_upload_sync(&up, h_ba, vk_buf_slice(&lut->b_all_rev[wi], 0, ba_words * sizeof(uint32_t)), ba_words * sizeof(uint32_t));
        free(h_rw);
        free(h_ba);
        free(h_bp);
    }
    vk_batch_destroy(&up);
    return ok;
}

static VkShaderId pick_bitrev_shader(int strategy, size_t elem_bytes) {
    int sz_idx = (elem_bytes <= 4u) ? 0 : (elem_bytes <= 8u ? 1 : 2);
    static const VkShaderId kTable[7][3] = {
        {VK_SHADER_bitrev_naive_4b,        VK_SHADER_bitrev_naive_8b,        VK_SHADER_bitrev_naive_16b},
        {VK_SHADER_bitrev_lut_pairs_4b,    VK_SHADER_bitrev_lut_pairs_8b,    VK_SHADER_bitrev_lut_pairs_16b},
        {VK_SHADER_bitrev_lut_idx_4b,      VK_SHADER_bitrev_lut_idx_8b,      VK_SHADER_bitrev_lut_idx_16b},
        {VK_SHADER_bitrev_whole_smem_4b,   VK_SHADER_bitrev_whole_smem_8b,   VK_SHADER_bitrev_whole_smem_16b},
        {VK_SHADER_bitrev_cobra_w4_4b,     VK_SHADER_bitrev_cobra_w4_8b,     VK_SHADER_bitrev_cobra_w4_16b},
        {VK_SHADER_bitrev_cobra_w5_4b,     VK_SHADER_bitrev_cobra_w5_8b,     VK_SHADER_bitrev_cobra_w5_16b},
        {VK_SHADER_bitrev_scalar_cobra_4b, VK_SHADER_bitrev_scalar_cobra_8b, VK_SHADER_bitrev_scalar_cobra_16b},
    };
    return kTable[strategy][sz_idx];
}

static size_t build_and_filter_cands(
    const VkBitRevEngine* eng,
    bool in_place,
    VkBitRevCandidate* out) {
    const VkDeviceFftConfig* cfg = vk_ctx_config(eng->ctx);
    bool allow_smem = cfg->has_native_local_mem || eng->launch_cfg.allow_emulated_local_mem;
    VkBitRevCandidate raw[VK_BITREV_MAX_CANDS];
    size_t nr = 0;

    int max_whole = (eng->elem_bytes <= 4u) ? 14 : (eng->elem_bytes <= 8u ? 13 : 12);
    if (eng->logN >= 10 && eng->logN <= max_whole) {
        size_t sb = ((size_t)1u << eng->logN) * eng->elem_bytes;
        raw[nr++] = (VkBitRevCandidate){"whole_smem_xor_w256", 256, sb, true, false, VK_BITREV_LUT_NONE, 3, false};
        raw[nr++] = (VkBitRevCandidate){"whole_smem_xor_w128", 128, sb, true, false, VK_BITREV_LUT_NONE, 3, false};
    }
    if (eng->logN >= 10 && eng->logN <= 26) {
        size_t sw5 = ((size_t)1u << 10) * (in_place ? 2u : 1u) * eng->elem_bytes;
        raw[nr++] = (VkBitRevCandidate){"cobra_w5_lut_wg256", 256, sw5, true, false, VK_BITREV_LUT_COBRA_W5, 5, true};
        raw[nr++] = (VkBitRevCandidate){"cobra_w5_wg256",     256, sw5, true, false, VK_BITREV_LUT_NONE,     5, false};
        raw[nr++] = (VkBitRevCandidate){"cobra_w5_wg128",     128, sw5, true, false, VK_BITREV_LUT_NONE,     5, false};
    }
    if (eng->logN >= 8 && eng->logN <= 24) {
        size_t sw4 = ((size_t)1u << 8) * (in_place ? 2u : 1u) * eng->elem_bytes;
        raw[nr++] = (VkBitRevCandidate){"cobra_w4_lut_wg256", 256, sw4, true, false, VK_BITREV_LUT_COBRA_W4, 4, true};
        raw[nr++] = (VkBitRevCandidate){"cobra_w4_wg256",     256, sw4, true, false, VK_BITREV_LUT_NONE,     4, false};
        raw[nr++] = (VkBitRevCandidate){"cobra_w4_wg128",     128, sw4, true, false, VK_BITREV_LUT_NONE,     4, false};
        raw[nr++] = (VkBitRevCandidate){"scalar_cobra_w4",     32,   0, false, true, VK_BITREV_LUT_COBRA_W4, 6, true};
    }
    if (in_place) {
        raw[nr++] = (VkBitRevCandidate){"lut_pairs_w256", 256, 0, false, false, VK_BITREV_LUT_PAIRS, 1, true};
    } else {
        raw[nr++] = (VkBitRevCandidate){"lut_idx_w256",   256, 0, false, false, VK_BITREV_LUT_IDX,   2, true};
    }
    raw[nr++] = (VkBitRevCandidate){"naive_w256", 256, 0, false, false, VK_BITREV_LUT_NONE, 0, false};
    raw[nr++] = (VkBitRevCandidate){"naive_w128", 128, 0, false, false, VK_BITREV_LUT_NONE, 0, false};

    size_t n_out = 0;
    /* Pass 1: CPU preferred first on CPU devices */
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < nr; ++i) {
            bool want_cpu_pref = (!cfg->is_gpu && pass == 0);
            if (!cfg->is_gpu) {
                if (raw[i].cpu_preferred != want_cpu_pref) continue;
            } else if (pass == 1) {
                continue;
            }
            if (raw[i].uses_smem && (!allow_smem || raw[i].smem_bytes > cfg->local_mem_size)) continue;
            if (raw[i].wg_size > cfg->max_work_group_size) continue;
            out[n_out++] = raw[i];
        }
    }
    return n_out;
}

bool vk_bitrev_init(
    VkBitRevEngine* eng,
    VkDeviceContext* ctx,
    int logN,
    size_t elem_bytes,
    const char* prec_tag_str,
    const VkFftLaunchConfig* launch_cfg) {
    memset(eng, 0, sizeof(*eng));
    eng->ctx = ctx;
    eng->logN = logN;
    eng->elem_bytes = elem_bytes;
    snprintf(eng->prec_tag, sizeof(eng->prec_tag), "%s", prec_tag_str ? prec_tag_str : "f");
    eng->launch_cfg = launch_cfg ? *launch_cfg : vk_fft_launch_config_default();
    vk_autotune_resolve_cache_path(eng->launch_cfg.cache_path, eng->cache_path, sizeof(eng->cache_path));
    eng->xor_k = vk_whole_smem_xor_k(
        vk_bank_model_from_kind(vk_ctx_config(ctx)->bank_model), (int)elem_bytes);
    eng->lut.ctx = ctx;
    eng->lut.logN = logN;
    eng->num_ip = build_and_filter_cands(eng, true, eng->valid_ip);
    eng->num_oop = build_and_filter_cands(eng, false, eng->valid_oop);
    return eng->num_ip > 0 && eng->num_oop > 0;
}

void vk_bitrev_destroy(VkBitRevEngine* eng) {
    if (!eng) return;
    lut_destroy(&eng->lut);
    memset(eng, 0, sizeof(*eng));
}

static size_t effective_wg(const VkBitRevEngine* eng, const VkBitRevCandidate* cand) {
    if (!cand->uses_smem && eng->launch_cfg.bitrev_wg_size > 0) {
        size_t max_wg = vk_ctx_config(eng->ctx)->max_work_group_size;
        return eng->launch_cfg.bitrev_wg_size < max_wg ? eng->launch_cfg.bitrev_wg_size : max_wg;
    }
    return cand->wg_size;
}

static bool record_cand_impl(
    VkBitRevEngine* eng,
    VkCmdBatch* batch,
    const VkBitRevCandidate* cand,
    bool in_place,
    VkBufferSlice src,
    VkBufferSlice dst,
    size_t bs,
    uint32_t batch_offset) {
    if (bs == 0 || eng->logN <= 0) return true;
    if (cand->lut_kind != VK_BITREV_LUT_NONE && !lut_ensure(&eng->lut, cand->lut_kind)) return false;

    uint32_t wg = (uint32_t)effective_wg(eng, cand);
    VkShaderId sid = pick_bitrev_shader(cand->strategy, eng->elem_bytes);

    if (cand->strategy == 0) {
        struct { uint32_t total, logN, batch_offset; } pc = {
            (uint32_t)(bs << eng->logN), (uint32_t)eng->logN, batch_offset
        };
        VkSpecConstEntry sp[2] = {{0u, wg}, {1u, in_place ? 1u : 0u}};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 2, sizeof(pc), sp, 2, 0);
        VkBufferSlice b[2] = {in_place ? dst : src, dst};
        return vk_batch_dispatch(batch, p, b, 2, &pc, sizeof(pc), (pc.total + wg - 1u) / wg, 1, 1);
    }

    if (cand->strategy == 1) {
        struct { uint32_t npairs, total_pairs, logN, batch_offset; } pc = {
            (uint32_t)eng->lut.npairs, (uint32_t)(bs * eng->lut.npairs), (uint32_t)eng->logN, batch_offset
        };
        if (pc.total_pairs == 0) return true;
        VkSpecConstEntry sp[2] = {{0u, wg}, {1u, eng->lut.pairs_packed_u16 ? 1u : 0u}};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 2, sizeof(pc), sp, 2, 0);
        VkBufferSlice b[2] = {dst, vk_buf_slice(&eng->lut.pairs_buf, 0, VK_WHOLE_SIZE)};
        return vk_batch_dispatch(batch, p, b, 2, &pc, sizeof(pc), (pc.total_pairs + wg - 1u) / wg, 1, 1);
    }

    if (cand->strategy == 2) {
        struct { uint32_t total, logN, batch_offset; } pc = {
            (uint32_t)(bs << eng->logN), (uint32_t)eng->logN, batch_offset
        };
        VkSpecConstEntry sp[2] = {{0u, wg}, {1u, eng->lut.idx_packed_u16 ? 1u : 0u}};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 3, sizeof(pc), sp, 2, 0);
        VkBufferSlice b[3] = {src, dst, vk_buf_slice(&eng->lut.idx_buf, 0, VK_WHOLE_SIZE)};
        return vk_batch_dispatch(batch, p, b, 3, &pc, sizeof(pc), (pc.total + wg - 1u) / wg, 1, 1);
    }

    if (cand->strategy == 3) {
        struct { uint32_t logN, xor_k, batch_offset; } pc = {
            (uint32_t)eng->logN, (uint32_t)eng->xor_k, batch_offset
        };
        VkSpecConstEntry sp[3] = {{0u, wg}, {1u, 1u << eng->logN}, {2u, in_place ? 1u : 0u}};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 2, sizeof(pc), sp, 3, 0);
        VkBufferSlice b[2] = {in_place ? dst : src, dst};
        return vk_batch_dispatch(batch, p, b, 2, &pc, sizeof(pc), (uint32_t)bs, 1, 1);
    }

    if (cand->strategy == 4 || cand->strategy == 5) {
        int wbits = (cand->strategy == 4) ? 4 : 5;
        int wi = wbits - 4;
        int m = eng->logN - 2 * wbits;
        uint32_t smem_elems = (1u << (2 * wbits)) * (in_place ? 2u : 1u);
        uint32_t work_per_batch = 1u << m;
        VkBufferSlice rev_s = vk_ctx_dummy_buffer(eng->ctx);
        VkBufferSlice b_s = vk_ctx_dummy_buffer(eng->ctx);
        if (cand->use_lut) {
            rev_s = vk_buf_slice(&eng->lut.rev_w[wi], 0, VK_WHOLE_SIZE);
            if (in_place) {
                b_s = vk_buf_slice(&eng->lut.b_pairs[wi], 0, VK_WHOLE_SIZE);
                work_per_batch = (uint32_t)eng->lut.nb_pairs[wi];
            } else {
                b_s = vk_buf_slice(&eng->lut.b_all_rev[wi], 0, VK_WHOLE_SIZE);
            }
        }
        struct { uint32_t work_per_batch, logN, batch_offset; } pc = {
            work_per_batch, (uint32_t)eng->logN, batch_offset
        };
        VkSpecConstEntry sp[4] = {
            {0u, wg}, {1u, in_place ? 1u : 0u}, {2u, cand->use_lut ? 1u : 0u}, {3u, smem_elems}
        };
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 4, sizeof(pc), sp, 4, 0);
        VkBufferSlice b[4] = {in_place ? dst : src, dst, rev_s, b_s};
        return vk_batch_dispatch(batch, p, b, 4, &pc, sizeof(pc), (uint32_t)(bs * work_per_batch), 1, 1);
    }

    int m = eng->logN - 8;
    uint32_t wpb = in_place ? (uint32_t)eng->lut.nb_pairs[0] : (1u << m);
    uint32_t total_work = (uint32_t)(bs * wpb);
    if (total_work == 0) return true;
    struct { uint32_t work_per_batch, total_work, logN, batch_offset; } pc = {
        wpb, total_work, (uint32_t)eng->logN, batch_offset
    };
    VkSpecConstEntry sp[2] = {{0u, wg}, {1u, in_place ? 1u : 0u}};
    const VkCompiledPipeline* p = vk_ctx_get_pipeline(eng->ctx, sid, 4, sizeof(pc), sp, 2, 0);
    VkBufferSlice b[4] = {
        in_place ? dst : src, dst,
        vk_buf_slice(&eng->lut.rev_w[0], 0, VK_WHOLE_SIZE),
        vk_buf_slice(in_place ? &eng->lut.b_pairs[0] : &eng->lut.b_all_rev[0], 0, VK_WHOLE_SIZE)
    };
    return vk_batch_dispatch(batch, p, b, 4, &pc, sizeof(pc), (total_work + wg - 1u) / wg, 1, 1);
}

typedef struct BitRevTuneCtx {
    VkBitRevEngine*          eng;
    const VkBitRevCandidate* cands;
    bool                     in_place;
    VkCmdBatch*              batch;
    VkBufferSlice            buf_a;
    VkBufferSlice            buf_b;
    size_t                   tune_bs;
} BitRevTuneCtx;

static bool bitrev_bench_run(size_t ci, int reps, void* ud) {
    BitRevTuneCtx* tc = (BitRevTuneCtx*)ud;
    if (!vk_batch_begin(tc->batch)) return false;
    for (int r = 0; r < reps; ++r) {
        if (!record_cand_impl(tc->eng, tc->batch, &tc->cands[ci], tc->in_place,
                              tc->buf_a, tc->in_place ? tc->buf_a : tc->buf_b, tc->tune_bs, 0)) {
            return false;
        }
        if (r + 1 < reps) vk_batch_barrier_c2c(tc->batch);
    }
    return vk_batch_submit_and_wait(tc->batch);
}

static bool bitrev_bench_conserv(size_t a, size_t b, void* ud) {
    BitRevTuneCtx* tc = (BitRevTuneCtx*)ud;
    if (tc->cands[a].smem_bytes != tc->cands[b].smem_bytes) {
        return tc->cands[a].smem_bytes < tc->cands[b].smem_bytes;
    }
    return a < b;
}

static int tune_candidates(
    VkBitRevEngine* eng,
    bool in_place,
    const VkBitRevCandidate* cands,
    size_t ncands,
    bool write_cache,
    const char* cache_key) {
    size_t N = (size_t)1u << eng->logN;
    size_t target = vk_ctx_config(eng->ctx)->is_gpu ? ((size_t)8 << 20) : ((size_t)1 << 20);
    size_t tune_bs = target / (N * eng->elem_bytes > 0 ? N * eng->elem_bytes : 1u);
    if (tune_bs < 4u) tune_bs = 4u;
    if (tune_bs > 256u) tune_bs = 256u;
    size_t bytes = tune_bs * N * eng->elem_bytes;

    VkBufferAlloc ba = {0}, bb = {0};
    VkCmdBatch batch = {0};
    if (!vk_buf_create(eng->ctx, bytes, VK_MEM_DEVICE_LOCAL, &ba)) return 0;
    if (!in_place && !vk_buf_create(eng->ctx, bytes, VK_MEM_DEVICE_LOCAL, &bb)) {
        vk_buf_destroy(&ba);
        return 0;
    }
    if (!vk_batch_init(&batch, eng->ctx, 0)) {
        vk_buf_destroy(&ba);
        vk_buf_destroy(&bb);
        return 0;
    }
    vk_batch_begin(&batch);
    vk_batch_fill(&batch, vk_buf_slice(&ba, 0, bytes), bytes, 0x3f800000u);
    if (!in_place) vk_batch_fill(&batch, vk_buf_slice(&bb, 0, bytes), bytes, 0u);
    vk_batch_submit_and_wait(&batch);

    BitRevTuneCtx tc = {
        eng, cands, in_place, &batch,
        vk_buf_slice(&ba, 0, bytes),
        in_place ? vk_buf_slice(&ba, 0, bytes) : vk_buf_slice(&bb, 0, bytes),
        tune_bs
    };
    VkInterleavedBenchResult res = vk_run_interleaved_bench(
        ncands, bitrev_bench_run, bitrev_bench_conserv, &tc, 5, 2.0, 16, 0.03, 0.08);
    int winner = res.ok ? (int)res.chosen_idx : 0;
    if (write_cache && res.ok) {
        VkAutotuneEntry e = {0};
        e.mode = VK_PLAN_SINGLE_PASS;
        snprintf(e.variant_id, sizeof(e.variant_id), "%s", cands[winner].variant_id);
        e.margin_pct = res.margin_pct;
        if (res.has_runner_up) {
            snprintf(e.runner_up_id, sizeof(e.runner_up_id), "%s", cands[res.runner_up_idx].variant_id);
        }
        vk_autotune_cache_put(eng->cache_path, cache_key, &e);
    }

    vk_batch_destroy(&batch);
    vk_buf_destroy(&ba);
    vk_buf_destroy(&bb);
    return winner;
}

bool vk_bitrev_resolve(VkBitRevEngine* eng, bool in_place) {
    if (in_place && eng->resolved_ip) return true;
    if (!in_place && eng->resolved_oop) return true;
    const VkBitRevCandidate* cands = in_place ? eng->valid_ip : eng->valid_oop;
    size_t ncands = in_place ? eng->num_ip : eng->num_oop;
    if (ncands == 0) return false;

    const char* slot_tag = in_place ? "bitrev_ip" : "bitrev_oop";
    int chosen = 0;
    bool hit = false;

    if (eng->launch_cfg.bitrev_variant_override[0] != '\0') {
        bool found = false;
        for (size_t i = 0; i < ncands; ++i) {
            if (strcmp(eng->launch_cfg.bitrev_variant_override, cands[i].variant_id) == 0) {
                chosen = (int)i;
                found = true;
                break;
            }
        }
        if (!found) return false;
    } else {
        bool allow_cache = (eng->launch_cfg.autotune == VK_AUTOTUNE_AUTO);
        bool force_tune = (eng->launch_cfg.autotune == VK_AUTOTUNE_FORCE);
        char key[384];
        vk_autotune_make_key(vk_ctx_config(eng->ctx), eng->prec_tag, eng->logN, slot_tag, 0, key, sizeof(key));
        if (allow_cache && !vk_fft_launch_has_bitrev_overrides(&eng->launch_cfg)) {
            VkAutotuneEntry cached;
            if (vk_autotune_cache_get(eng->cache_path, key, &cached)) {
                for (size_t i = 0; i < ncands; ++i) {
                    if (strcmp(cached.variant_id, cands[i].variant_id) == 0) {
                        chosen = (int)i;
                        hit = true;
                        break;
                    }
                }
            }
        }
        if (!hit && eng->launch_cfg.autotune != VK_AUTOTUNE_OFF && ncands > 1) {
            chosen = tune_candidates(eng, in_place, cands, ncands, allow_cache || force_tune, key);
        }
    }

    if (in_place) {
        eng->chosen_ip = chosen;
        eng->cache_hit_ip = hit;
        eng->resolved_ip = true;
    } else {
        eng->chosen_oop = chosen;
        eng->cache_hit_oop = hit;
        eng->resolved_oop = true;
    }
    return true;
}

bool vk_bitrev_record_in_place(VkBitRevEngine* eng, VkCmdBatch* batch, VkBufferSlice buf, size_t bs, uint32_t batch_offset) {
    if (bs == 0 || eng->logN <= 0) return true;
    if (!vk_bitrev_resolve(eng, true)) return false;
    return record_cand_impl(eng, batch, &eng->valid_ip[eng->chosen_ip], true, buf, buf, bs, batch_offset);
}

bool vk_bitrev_record_out_of_place(VkBitRevEngine* eng, VkCmdBatch* batch, VkBufferSlice src, VkBufferSlice dst, size_t bs, uint32_t batch_offset) {
    if (bs == 0 || eng->logN <= 0) return true;
    if (src.buffer == dst.buffer && src.offset == dst.offset) {
        return vk_bitrev_record_in_place(eng, batch, dst, bs, batch_offset);
    }
    if (!vk_bitrev_resolve(eng, false)) return false;
    return record_cand_impl(eng, batch, &eng->valid_oop[eng->chosen_oop], false, src, dst, bs, batch_offset);
}

bool vk_bitrev_record_candidate(
    VkBitRevEngine* eng, VkCmdBatch* batch, bool in_place, const char* variant_id,
    VkBufferSlice src, VkBufferSlice dst, size_t bs, uint32_t batch_offset) {
    const VkBitRevCandidate* cands = in_place ? eng->valid_ip : eng->valid_oop;
    size_t ncands = in_place ? eng->num_ip : eng->num_oop;
    for (size_t i = 0; i < ncands; ++i) {
        if (strcmp(variant_id, cands[i].variant_id) == 0) {
            return record_cand_impl(eng, batch, &cands[i], in_place, src, dst, bs, batch_offset);
        }
    }
    return false;
}

const char* vk_bitrev_variant_name(VkBitRevEngine* eng, bool in_place) {
    if (!vk_bitrev_resolve(eng, in_place)) return "none";
    return in_place ? eng->valid_ip[eng->chosen_ip].variant_id
                    : eng->valid_oop[eng->chosen_oop].variant_id;
}

bool vk_bitrev_cache_hit(VkBitRevEngine* eng, bool in_place) {
    if (!vk_bitrev_resolve(eng, in_place)) return false;
    return in_place ? eng->cache_hit_ip : eng->cache_hit_oop;
}
