#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/math.h"
#include "fft/vk_dif.h"

typedef struct SpDesc {
    const char* variant_id;
    int logN, logE, logChunks, B;
    VkStagingPolicy staging;
    VkT1LayoutPolicy t1_layout;
    VkPhase3Policy phase3;
    bool stash_chunk1;
} SpDesc;

static const SpDesc kNarrowDescs[] = {
    {"split_10_e3_b2_aff", 10, 3, 0, 2, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"split_10_e3_b1_aff", 10, 3, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"split_10_e4_b1_aff", 10, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"seq_10_e4_b1_2tstr", 10, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_STRIDE, false},
    {"split_11_e3_b1_aff", 11, 3, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"split_11_e4_b1_aff", 11, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"split_11_e4_b1_2tstr", 11, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_2T_STRIDE, false},
    {"split_12_e4_b1_aff", 12, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"seq_12_e4_b1_sg", 12, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"seq_12_e4_b1_2txor", 12, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_XOR, false},
    {"seq_13_e4_b1_sg", 13, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"seq_13_e5_c2_sg", 13, 5, 1, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"seq_13_e5_c2_2txor", 13, 5, 1, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_XOR, false},
    {"seq_14_e5_c2_sg", 14, 5, 1, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
};

static const SpDesc kWideDescs[] = {
    {"seq_10_e4_b1_2tstr", 10, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_STRIDE, false},
    {"split_10_e4_b1_xor", 10, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_XOR, VK_PHASE3_SHUFFLE, false},
    {"wide_11_e4_b1_2tstr", 11, 4, 0, 1, VK_STAGING_SPLIT, VK_T1_AFFINE, VK_PHASE3_2T_STRIDE, false},
    {"seq_11_e4_b1_sg", 11, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"wide_12_e4_b1_2txor", 12, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_XOR, false},
    {"seq_12_e4_b1_sg", 12, 4, 0, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_SHUFFLE, false},
    {"wide_13_e5_c2_xor_stash", 13, 5, 1, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_XOR, true},
    {"wide_13_e5_c2_xor_nostash", 13, 5, 1, 1, VK_STAGING_SEQ, VK_T1_AFFINE, VK_PHASE3_2T_XOR, false},
};

static VkShaderId lookup_sp_shader(VkPrecKind prec, const char* vid) {
    char name[96];
    snprintf(name, sizeof(name), "dif_%s_%s", vk_prec_tag(prec), vid);
    const VkEmbeddedSpirv* spv = vk_find_embedded_spirv(name);
    if (!spv) return VK_SHADER_COUNT;
    const VkEmbeddedSpirv* base = vk_get_embedded_spirv((VkShaderId)0);
    return (VkShaderId)(spv - base);
}

static VkShaderId lookup_pass0_shader(VkPrecKind prec, int log_n1) {
    switch (prec) {
        case VK_PREC_HALF:
            if (log_n1 == 3) return VK_SHADER_dif_h_pass0_r3;
            if (log_n1 == 4) return VK_SHADER_dif_h_pass0_r4;
            if (log_n1 == 5) return VK_SHADER_dif_h_pass0_r5;
            break;
        case VK_PREC_FLOAT:
            if (log_n1 == 3) return VK_SHADER_dif_f_pass0_r3;
            if (log_n1 == 4) return VK_SHADER_dif_f_pass0_r4;
            if (log_n1 == 5) return VK_SHADER_dif_f_pass0_r5;
            break;
        case VK_PREC_DF:
            if (log_n1 == 3) return VK_SHADER_dif_df_pass0_r3;
            if (log_n1 == 4) return VK_SHADER_dif_df_pass0_r4;
            if (log_n1 == 5) return VK_SHADER_dif_df_pass0_r5;
            break;
        case VK_PREC_DOUBLE:
            if (log_n1 == 3) return VK_SHADER_dif_d_pass0_r3;
            if (log_n1 == 4) return VK_SHADER_dif_d_pass0_r4;
            if (log_n1 == 5) return VK_SHADER_dif_d_pass0_r5;
            break;
    }
    return VK_SHADER_COUNT;
}

static size_t collect_sp_candidates(const VkDifFftPlan* plan, int logN, bool apply_override, VkDifCandidate* out, size_t cap) {
    const VkDeviceFftConfig* cfg = vk_ctx_config(plan->ctx);
    if (!cfg->has_native_local_mem && !plan->launch_cfg.allow_emulated_local_mem) return 0;

    bool is_wide = (plan->prec == VK_PREC_DF || plan->prec == VK_PREC_DOUBLE);
    const SpDesc* descs = is_wide ? kWideDescs : kNarrowDescs;
    size_t ndescs = is_wide ? (sizeof(kWideDescs) / sizeof(kWideDescs[0])) : (sizeof(kNarrowDescs) / sizeof(kNarrowDescs[0]));
    VkBankModel bank = vk_bank_model_from_kind(cfg->bank_model);
    int elem_b = (int)plan->smem_elem_bytes;

    size_t cnt = 0;
    for (size_t i = 0; i < ndescs && cnt < cap; ++i) {
        const SpDesc* d = &descs[i];
        if (d->logN != logN) continue;
        if (apply_override && plan->launch_cfg.variant_override[0] != '\0' && strcmp(plan->launch_cfg.variant_override, d->variant_id) != 0) {
            continue;
        }
        if (!plan->launch_cfg.allow_wide_kernels && d->phase3 != VK_PHASE3_SHUFFLE) continue;
        if (plan->launch_cfg.stash_8192_override == 0 && d->stash_chunk1) continue;
        if (plan->launch_cfg.stash_8192_override == 1 && logN == 13 && is_wide && !d->stash_chunk1) continue;

        int logT = d->logN - d->logE;
        int T = 1 << logT;
        int logChunk = d->logE - d->logChunks;
        int Chunk = 1 << logChunk;
        int logW = logT - logChunk;
        int W = 1 << logW;
        size_t wg = (size_t)(d->B * T);
        size_t req_sg = (d->phase3 == VK_PHASE3_SHUFFLE) ? (size_t)W : 1u;

        if (wg > cfg->max_work_group_size) continue;
        if (req_sg > 1u && plan->effective_sg_size < req_sg) continue;

        int stride2 = (d->phase3 == VK_PHASE3_2T_STRIDE) ? vk_find_min_conflict_stride2(bank, elem_b, W) : 0;
        int min_row = (d->phase3 == VK_PHASE3_2T_STRIDE && Chunk * stride2 > T) ? (Chunk * stride2) : T;
        int row_stride = (d->t1_layout == VK_T1_XOR) ? T
                                                     : ((d->staging == VK_STAGING_SEQ && d->phase3 != VK_PHASE3_2T_STRIDE)
                                                            ? T
                                                            : vk_find_min_conflict_row_stride(bank, elem_b, T, W, min_row, stride2, Chunk));
        size_t smem_elems = (size_t)d->B * (size_t)Chunk * (size_t)row_stride;
        size_t smem_bufs = (d->staging == VK_STAGING_SPLIT) ? 2u : 1u;
        size_t smem_bytes = smem_bufs * smem_elems * plan->smem_elem_bytes;
        if (smem_bytes > cfg->local_mem_size) continue;

        VkShaderId sid = lookup_sp_shader(plan->prec, d->variant_id);
        if (sid == VK_SHADER_COUNT) continue;

        VkDifCandidate c = {0};
        c.variant_id = d->variant_id;
        c.shader_id = sid;
        c.logN = d->logN;
        c.logE = d->logE;
        c.logChunks = d->logChunks;
        c.B = d->B;
        c.staging = d->staging;
        c.t1_layout = d->t1_layout;
        c.phase3 = d->phase3;
        c.stash_chunk1 = d->stash_chunk1;
        c.wg_size = wg;
        c.row_stride = (size_t)row_stride;
        c.stride2 = (size_t)(stride2 > 0 ? stride2 : W);
        c.smem_elems_per_buf = smem_elems;
        c.smem_bytes = smem_bytes;
        c.req_sg = req_sg;
        out[cnt++] = c;
    }

    /* For FP64 default heuristic, put non-stash ahead of stash at logN=13 */
    if (plan->prec == VK_PREC_DOUBLE && cnt == 2 && out[0].stash_chunk1 && !out[1].stash_chunk1) {
        VkDifCandidate tmp = out[0];
        out[0] = out[1];
        out[1] = tmp;
    }
    return cnt;
}

static bool record_sp_cand(const VkDifFftPlan* plan, VkCmdBatch* batch, const VkDifCandidate* cand, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs,
                           uint32_t in_offset, uint32_t out_offset, const VkFftCompactParams* ep) {
    if (bs == 0) return true;
    size_t pb = plan->packed_batch;
    size_t num_pairs = (bs + pb - 1u) / pb;
    size_t num_wg = (num_pairs + (size_t)cand->B - 1u) / (size_t)cand->B;

    VkSpecConstEntry specs[7] = {
        {0u, (uint32_t)cand->row_stride}, {1u, (uint32_t)cand->stride2},    {2u, (uint32_t)cand->smem_elems_per_buf}, {3u, (plan->sign > 0) ? 1u : 0u},
        {4u, (uint32_t)ep->compact_mode}, {5u, ep->apply_deconv ? 1u : 0u}, {15u, plan->df_use_veltkamp ? 1u : 0u},
    };
    size_t nspecs = (plan->prec == VK_PREC_DF) ? 7u : 6u;

    struct {
        uint32_t bs, log_n1, in_offset, out_offset;
    } pc = {(uint32_t)bs, (uint32_t)ep->log_n1, in_offset, out_offset};
    const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(plan->ctx, cand->shader_id, 3, sizeof(pc), specs, nspecs, 0);
    if (!pipe) return false;

    VkBufferSlice dec = (ep->apply_deconv && vk_slice_valid(ep->d_slot_deconv)) ? ep->d_slot_deconv : vk_ctx_dummy_buffer(plan->ctx);
    VkBufferSlice binds[3] = {in_buf, out_buf, dec};
    return vk_batch_dispatch(batch, pipe, binds, 3, &pc, sizeof(pc), (uint32_t)num_wg, 1, 1);
}

static bool record_pass0_stage(const VkDifFftPlan* plan, VkCmdBatch* batch, int log_n1, int log_n2, VkBufferSlice in_buf, VkBufferSlice out_buf,
                               size_t chunk_bs, uint32_t in_offset, uint32_t out_offset) {
    if (chunk_bs == 0) return true;
    VkShaderId sid = lookup_pass0_shader(plan->prec, log_n1);
    if (sid == VK_SHADER_COUNT) return false;

    bool has_tw = (log_n2 > 0);
    size_t n2 = has_tw ? ((size_t)1u << log_n2) : 1u;
    size_t pb = plan->packed_batch;
    size_t total_threads = ((chunk_bs + pb - 1u) / pb) * n2;
    uint32_t wg = (uint32_t)plan->pass0_wg;
    if (wg == 0) wg = 128u;
    uint32_t groups = (uint32_t)((total_threads + wg - 1u) / wg);

    VkSpecConstEntry specs[4] = {
        {0u, wg},
        {1u, has_tw ? 1u : 0u},
        {2u, (plan->sign > 0) ? 1u : 0u},
        {15u, plan->df_use_veltkamp ? 1u : 0u},
    };
    size_t nspecs = (plan->prec == VK_PREC_DF) ? 4u : 3u;

    VkBufferSlice binds[2] = {in_buf, out_buf};
    if (plan->prec == VK_PREC_DOUBLE) {
        struct {
            uint32_t chunk_bs, log_n2, in_off, out_off;
            double inv_m;
        } pc = {(uint32_t)chunk_bs, (uint32_t)(has_tw ? log_n2 : 0), in_offset, out_offset, has_tw ? (1.0 / (double)((size_t)1u << (log_n1 + log_n2))) : 0.0};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 2, sizeof(pc), specs, nspecs, 0);
        return vk_batch_dispatch(batch, p, binds, 2, &pc, sizeof(pc), groups, 1, 1);
    } else {
        struct {
            uint32_t chunk_bs, log_n2, in_off, out_off;
            float inv_m;
        } pc = {(uint32_t)chunk_bs, (uint32_t)(has_tw ? log_n2 : 0), in_offset, out_offset, has_tw ? (1.0f / (float)((size_t)1u << (log_n1 + log_n2))) : 0.0f};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 2, sizeof(pc), specs, nspecs, 0);
        return vk_batch_dispatch(batch, p, binds, 2, &pc, sizeof(pc), groups, 1, 1);
    }
}

static int multistage_factors(int logN, int stages[4]) {
    switch (logN) {
        case 10:
            stages[0] = 4;
            stages[1] = 3;
            stages[2] = 3;
            return 3;
        case 11:
            stages[0] = 4;
            stages[1] = 4;
            stages[2] = 3;
            return 3;
        case 12:
            stages[0] = 4;
            stages[1] = 4;
            stages[2] = 4;
            return 3;
        case 13:
            stages[0] = 5;
            stages[1] = 4;
            stages[2] = 4;
            return 3;
        case 14:
            stages[0] = 5;
            stages[1] = 5;
            stages[2] = 4;
            return 3;
        case 15:
            stages[0] = 5;
            stages[1] = 5;
            stages[2] = 5;
            return 3;
        case 16:
            stages[0] = 4;
            stages[1] = 4;
            stages[2] = 4;
            stages[3] = 4;
            return 4;
        case 17:
            stages[0] = 5;
            stages[1] = 4;
            stages[2] = 4;
            stages[3] = 4;
            return 4;
        case 18:
            stages[0] = 5;
            stages[1] = 5;
            stages[2] = 4;
            stages[3] = 4;
            return 4;
        default:
            return 0;
    }
}

typedef struct SpTuneCtx {
    const VkDifFftPlan* plan;
    const VkDifCandidate* cands;
    VkCmdBatch* batch;
    VkBufferSlice in_s;
    VkBufferSlice out_s;
    size_t bs;
} SpTuneCtx;

static bool sp_bench_run(size_t ci, int reps, void* ud) {
    SpTuneCtx* tc = (SpTuneCtx*)ud;
    VkFftCompactParams ep = {0};
    if (!vk_batch_begin(tc->batch)) return false;
    for (int r = 0; r < reps; ++r) {
        if (!record_sp_cand(tc->plan, tc->batch, &tc->cands[ci], tc->in_s, tc->out_s, tc->bs, 0, 0, &ep)) {
            return false;
        }
        if (r + 1 < reps) vk_batch_barrier_c2c(tc->batch);
    }
    return vk_batch_submit_and_wait(tc->batch);
}

static bool sp_bench_conserv(size_t a, size_t b, void* ud) {
    SpTuneCtx* tc = (SpTuneCtx*)ud;
    if (tc->cands[a].smem_bytes != tc->cands[b].smem_bytes) {
        return tc->cands[a].smem_bytes < tc->cands[b].smem_bytes;
    }
    return a < b;
}

static bool resolve_sp_for(VkDifFftPlan* plan, int logN, bool apply_override, bool allow_cache, bool force_tune, VkDifCandidate* out_cand, bool* out_hit) {
    VkDifCandidate cands[16];
    size_t ncands = collect_sp_candidates(plan, logN, apply_override, cands, 16);
    if (ncands == 0) return false;

    *out_hit = false;
    if (ncands == 1 || plan->launch_cfg.autotune == VK_AUTOTUNE_OFF) {
        *out_cand = cands[0];
        return true;
    }

    char key[384];
    vk_autotune_make_key(vk_ctx_config(plan->ctx), vk_prec_tag(plan->prec), logN, "fft_sp", plan->sign, key, sizeof(key));
    if (allow_cache && !vk_fft_launch_has_plan_overrides(&plan->launch_cfg)) {
        VkAutotuneEntry cached;
        if (vk_autotune_cache_get(plan->cache_path, key, &cached) && cached.mode == VK_PLAN_SINGLE_PASS) {
            for (size_t i = 0; i < ncands; ++i) {
                if (strcmp(cached.variant_id, cands[i].variant_id) == 0) {
                    *out_cand = cands[i];
                    *out_hit = true;
                    return true;
                }
            }
        }
    }

    size_t N = (size_t)1u << logN;
    size_t target = vk_ctx_config(plan->ctx)->is_gpu ? ((size_t)8 << 20) : ((size_t)1 << 20);
    size_t bs = target / (N * plan->complex_bytes);
    if (bs < 4u) bs = 4u;
    if (bs > 128u) bs = 128u;
    if (plan->packed_batch == 2u && (bs & 1u)) ++bs;
    size_t bytes = bs * N * plan->complex_bytes;

    VkBufferAlloc ba = {0}, bb = {0};
    VkCmdBatch batch = {0};
    int winner = 0;
    if (vk_buf_create(plan->ctx, bytes, VK_MEM_DEVICE_LOCAL, &ba) && vk_buf_create(plan->ctx, bytes, VK_MEM_DEVICE_LOCAL, &bb) &&
        vk_batch_init(&batch, plan->ctx, 0)) {
        vk_batch_begin(&batch);
        vk_batch_fill(&batch, vk_buf_slice(&ba, 0, bytes), bytes, 0x3f000000u);
        vk_batch_fill(&batch, vk_buf_slice(&bb, 0, bytes), bytes, 0u);
        vk_batch_submit_and_wait(&batch);

        SpTuneCtx tc = {plan, cands, &batch, vk_buf_slice(&ba, 0, bytes), vk_buf_slice(&bb, 0, bytes), bs};
        VkInterleavedBenchResult res = vk_run_interleaved_bench(ncands, sp_bench_run, sp_bench_conserv, &tc, 5, 2.0, 12, 0.03, 0.08);
        if (res.ok) {
            winner = (int)res.chosen_idx;
            if (allow_cache || force_tune) {
                VkAutotuneEntry e = {0};
                e.mode = VK_PLAN_SINGLE_PASS;
                snprintf(e.variant_id, sizeof(e.variant_id), "%s", cands[winner].variant_id);
                e.margin_pct = res.margin_pct;
                if (res.has_runner_up) {
                    snprintf(e.runner_up_id, sizeof(e.runner_up_id), "%s", cands[res.runner_up_idx].variant_id);
                }
                vk_autotune_cache_put(plan->cache_path, key, &e);
            }
        }
    }
    vk_batch_destroy(&batch);
    vk_buf_destroy(&ba);
    vk_buf_destroy(&bb);

    *out_cand = cands[winner];
    return true;
}

bool vk_dif_plan_init(VkDifFftPlan* plan, VkDeviceContext* ctx, VkPrecKind prec, int sign, int logN, size_t max_bs, const VkFftLaunchConfig* cfg_in) {
    memset(plan, 0, sizeof(*plan));
    if (!ctx || logN < 10 || logN > 18 || (sign != 1 && sign != -1)) return false;

    plan->ctx = ctx;
    plan->prec = prec;
    plan->sign = sign;
    plan->logN = logN;
    plan->N = (size_t)1u << logN;
    plan->launch_cfg = cfg_in ? *cfg_in : vk_fft_launch_config_default();
    vk_autotune_resolve_cache_path(plan->launch_cfg.cache_path, plan->cache_path, sizeof(plan->cache_path));

    switch (prec) {
        case VK_PREC_HALF:
            plan->complex_bytes = 4;
            plan->smem_elem_bytes = 4;
            plan->packed_batch = 2;
            break;
        case VK_PREC_FLOAT:
            plan->complex_bytes = 8;
            plan->smem_elem_bytes = 4;
            plan->packed_batch = 1;
            break;
        case VK_PREC_DF:
            plan->complex_bytes = 16;
            plan->smem_elem_bytes = 8;
            plan->packed_batch = 1;
            break;
        case VK_PREC_DOUBLE:
            plan->complex_bytes = 16;
            plan->smem_elem_bytes = 8;
            plan->packed_batch = 1;
            break;
    }

    const VkDeviceFftConfig* dcfg = vk_ctx_config(ctx);
    VkDeviceProbeResult sg = vk_ctx_probe_subgroup(ctx);
    plan->effective_sg_size = sg.effective_sg_size;

    if (prec == VK_PREC_DF && !plan->launch_cfg.skip_df_kat) {
        VkDeviceProbeResult dfk = vk_ctx_probe_df_kat(ctx);
        if (!dfk.df_kat_passed) return false;
        plan->df_use_veltkamp = dfk.df_use_veltkamp;
    }
    if (prec == VK_PREC_HALF && !plan->launch_cfg.skip_fp16_kat) {
        VkDeviceProbeResult h16 = vk_ctx_probe_fp16_kat(ctx);
        if (!h16.fp16_kat_passed) return false;
    }
    if (prec == VK_PREC_DOUBLE) {
        if (plan->launch_cfg.fp64_compute_heavy_override >= 0) {
            plan->is_compute_heavy = (plan->launch_cfg.fp64_compute_heavy_override != 0);
        } else {
            VkDeviceProbeResult r64 = vk_ctx_probe_fp64_ratio(ctx);
            plan->fp64_to_fp32_ratio = r64.fp64_to_fp32_ratio;
            plan->is_compute_heavy = r64.fp64_compute_heavy;
        }
    }

    plan->pass0_wg = (plan->launch_cfg.pass0_wg_size > 0) ? plan->launch_cfg.pass0_wg_size : vk_device_default_padded_wg(dcfg, 256);

    bool allow_cache = (plan->launch_cfg.autotune == VK_AUTOTUNE_AUTO);
    bool force_tune = (plan->launch_cfg.autotune == VK_AUTOTUNE_FORCE);

    bool can_sp = !plan->launch_cfg.force_multistage && (plan->launch_cfg.two_pass_min_logN <= 0 || logN < plan->launch_cfg.two_pass_min_logN);
    if (can_sp && resolve_sp_for(plan, logN, true, allow_cache, force_tune, &plan->sp_cand, &plan->cache_hit)) {
        plan->exec_mode = VK_PLAN_SINGLE_PASS;
        snprintf(plan->variant_name, sizeof(plan->variant_name), "%s", plan->sp_cand.variant_id);
        plan->chunk_bs = (max_bs > 0) ? max_bs : 1u;
        return true;
    }

    /* Try TwoPass: factor logN = log_n1 + log_n2 with log_n1 in {3,4,5} and SinglePass log_n2 */
    if (!plan->launch_cfg.force_multistage) {
        int best_n1 = 0;
        VkDifCandidate best_c = {0};
        bool best_hit = false;
        int n1_order[3] = {4, 5, 3};
        if (plan->launch_cfg.two_pass_log_n1 >= 3 && plan->launch_cfg.two_pass_log_n1 <= 5) {
            n1_order[0] = plan->launch_cfg.two_pass_log_n1;
            n1_order[1] = 0;
            n1_order[2] = 0;
        }
        for (int i = 0; i < 3; ++i) {
            int ln1 = n1_order[i];
            if (ln1 <= 0) continue;
            int ln2 = logN - ln1;
            if (ln2 < 10 || ln2 > 14) continue;
            VkDifCandidate c;
            bool h = false;
            if (resolve_sp_for(plan, ln2, false, allow_cache, force_tune, &c, &h)) {
                best_n1 = ln1;
                best_c = c;
                best_hit = h;
                break;
            }
        }
        if (best_n1 > 0) {
            plan->exec_mode = VK_PLAN_TWO_PASS;
            plan->log_n1 = best_n1;
            plan->log_n2 = logN - best_n1;
            plan->sp_cand = best_c;
            plan->cache_hit = best_hit;
            snprintf(plan->variant_name, sizeof(plan->variant_name), "2p_r%d_%s", best_n1, best_c.variant_id);

            size_t fft_bytes = plan->N * plan->complex_bytes;
            size_t l2_chunk =
                (plan->launch_cfg.l2_chunk_bytes > 0)
                    ? plan->launch_cfg.l2_chunk_bytes
                    : vk_device_compute_l2_chunk_bytes(dcfg, fft_bytes, (size_t)1u << best_n1, plan->is_compute_heavy, plan->launch_cfg.l2_budget_num,
                                                       plan->launch_cfg.l2_budget_den, plan->launch_cfg.min_waves_per_cu);
            size_t cbs = l2_chunk / fft_bytes;
            if (cbs < 1u) cbs = 1u;
            if (max_bs > 0 && cbs > max_bs) cbs = max_bs;
            if (plan->packed_batch == 2u && cbs > 1u && (cbs & 1u)) --cbs;
            if (plan->packed_batch == 2u && cbs == 1u && max_bs >= 2u) cbs = 2u;
            plan->chunk_bs = cbs;
            if (!vk_buf_create(ctx, cbs * fft_bytes, VK_MEM_DEVICE_LOCAL, &plan->scratch)) {
                return false;
            }
            return true;
        }
    }

    /* Fallback: MultiStageGlobal (zero-SMEM, CPU/llvmpipe) */
    if (!plan->launch_cfg.allow_multistage_fallback) return false;
    plan->exec_mode = VK_PLAN_MULTISTAGE_GLOBAL;
    snprintf(plan->variant_name, sizeof(plan->variant_name), "multistage_logN%d", logN);
    size_t fft_bytes = plan->N * plan->complex_bytes;
    size_t l2_chunk = vk_device_compute_l2_chunk_bytes(dcfg, fft_bytes, 16, plan->is_compute_heavy, 1, 4, 1);
    size_t cbs = l2_chunk / fft_bytes;
    if (cbs < 1u) cbs = 1u;
    if (max_bs > 0 && cbs > max_bs) cbs = max_bs;
    plan->chunk_bs = cbs;
    return true;
}

void vk_dif_plan_destroy(VkDifFftPlan* plan) {
    if (!plan) return;
    vk_buf_destroy(&plan->scratch);
    if (plan->has_bitrev) vk_bitrev_destroy(&plan->bitrev);
    memset(plan, 0, sizeof(*plan));
}

bool vk_dif_record_execute_raw(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                               uint32_t out_offset_elems, size_t chunk_bs_override) {
    if (bs == 0) return true;
    VkFftCompactParams ep = {0};

    if (plan->exec_mode == VK_PLAN_SINGLE_PASS) {
        return record_sp_cand(plan, batch, &plan->sp_cand, in_buf, out_buf, bs, in_offset_elems, out_offset_elems, &ep);
    }

    size_t eff_chunk = (chunk_bs_override > 0 && chunk_bs_override < plan->chunk_bs) ? chunk_bs_override : plan->chunk_bs;
    if (eff_chunk == 0) eff_chunk = 1;

    if (plan->exec_mode == VK_PLAN_MULTISTAGE_GLOBAL) {
        int stages[4] = {0};
        int nstages = multistage_factors(plan->logN, stages);
        if (nstages == 0) return false;
        size_t done = 0;
        while (done < bs) {
            size_t cur = (bs - done < eff_chunk) ? (bs - done) : eff_chunk;
            uint32_t c_in_off = in_offset_elems + (uint32_t)(done * plan->N);
            uint32_t c_out_off = out_offset_elems + (uint32_t)(done * plan->N);
            size_t sub_bs = cur;
            int rem = plan->logN;
            for (int s = 0; s < nstages; ++s) {
                int r = stages[s];
                rem -= r;
                VkBufferSlice src = (s == 0) ? in_buf : out_buf;
                uint32_t soff = (s == 0) ? c_in_off : c_out_off;
                if (!record_pass0_stage(plan, batch, r, rem, src, out_buf, sub_bs, soff, c_out_off)) {
                    return false;
                }
                if (s + 1 < nstages) vk_batch_barrier_c2c(batch);
                sub_bs <<= r;
            }
            done += cur;
            if (done < bs) vk_batch_barrier_c2c(batch);
        }
        return true;
    }

    /* TwoPass */
    size_t n1 = (size_t)1u << plan->log_n1;
    VkBufferSlice scr = vk_buf_slice(&plan->scratch, 0, VK_WHOLE_SIZE);
    size_t done = 0;
    while (done < bs) {
        size_t cur = (bs - done < eff_chunk) ? (bs - done) : eff_chunk;
        uint32_t c_in_off = in_offset_elems + (uint32_t)(done * plan->N);
        uint32_t c_out_off = out_offset_elems + (uint32_t)(done * plan->N);
        if (!record_pass0_stage(plan, batch, plan->log_n1, plan->log_n2, in_buf, scr, cur, c_in_off, 0u)) {
            return false;
        }
        vk_batch_barrier_c2c(batch);
        if (!record_sp_cand(plan, batch, &plan->sp_cand, scr, out_buf, cur * n1, 0u, c_out_off, &ep)) {
            return false;
        }
        done += cur;
        if (done < bs) vk_batch_barrier_c2c(batch);
    }
    return true;
}

bool vk_dif_record_execute_raw_compact(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                                       uint32_t out_offset_elems, int compact_mode, VkBufferSlice d_slot_deconv, bool apply_deconv, size_t chunk_bs_override) {
    if (bs == 0) return true;
    if ((compact_mode != 1 && compact_mode != 2) || plan->exec_mode == VK_PLAN_MULTISTAGE_GLOBAL) {
        return false;
    }
    if (plan->exec_mode == VK_PLAN_SINGLE_PASS) {
        VkFftCompactParams ep = {compact_mode, 0, d_slot_deconv, apply_deconv};
        return record_sp_cand(plan, batch, &plan->sp_cand, in_buf, out_buf, bs, in_offset_elems, out_offset_elems, &ep);
    }

    size_t eff_chunk = (chunk_bs_override > 0 && chunk_bs_override < plan->chunk_bs) ? chunk_bs_override : plan->chunk_bs;
    if (eff_chunk == 0) eff_chunk = 1;
    size_t n1 = (size_t)1u << plan->log_n1;
    size_t full_nout = (compact_mode == 1) ? (plan->N >> 1) : ((plan->N >> 2) * 3u);
    VkFftCompactParams ep = {compact_mode, plan->log_n1, d_slot_deconv, apply_deconv};
    VkBufferSlice scr = vk_buf_slice(&plan->scratch, 0, VK_WHOLE_SIZE);

    size_t done = 0;
    while (done < bs) {
        size_t cur = (bs - done < eff_chunk) ? (bs - done) : eff_chunk;
        uint32_t c_in_off = in_offset_elems + (uint32_t)(done * plan->N);
        uint32_t c_out_off = out_offset_elems + (uint32_t)(done * full_nout);
        if (!record_pass0_stage(plan, batch, plan->log_n1, plan->log_n2, in_buf, scr, cur, c_in_off, 0u)) {
            return false;
        }
        vk_batch_barrier_c2c(batch);
        if (!record_sp_cand(plan, batch, &plan->sp_cand, scr, out_buf, cur * n1, 0u, c_out_off, &ep)) {
            return false;
        }
        done += cur;
        if (done < bs) vk_batch_barrier_c2c(batch);
    }
    return true;
}

bool vk_dif_record_bit_reverse(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice src, VkBufferSlice dst, size_t bs, uint32_t batch_offset) {
    if (!plan->has_bitrev) {
        if (!vk_bitrev_init(&plan->bitrev, plan->ctx, plan->logN, plan->complex_bytes, vk_prec_tag(plan->prec), &plan->launch_cfg)) {
            return false;
        }
        plan->has_bitrev = true;
    }
    return vk_bitrev_record_out_of_place(&plan->bitrev, batch, src, dst, bs, batch_offset);
}

bool vk_dif_record_execute(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                           uint32_t out_offset_elems, size_t chunk_bs_override) {
    if (!vk_dif_record_execute_raw(plan, batch, in_buf, out_buf, bs, in_offset_elems, out_offset_elems, chunk_bs_override)) {
        return false;
    }
    vk_batch_barrier_c2c(batch);
    return vk_dif_record_bit_reverse(plan, batch, out_buf, out_buf, bs, out_offset_elems);
}
