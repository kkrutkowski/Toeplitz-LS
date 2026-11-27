/* Plain C23 Quadruple-Precision Vulkan Type-1 NuFFT Engine. */
#include "nufft/vk_nufft1.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Push Constant Structs (matching gen_nufft.py std430 layouts byte-for-byte)
 * ============================================================================ */

typedef struct VkPrecompPtsDfPush {
    float    df_hi;
    float    df_lo;
    int32_t  M;
    int32_t  nfft;
    int32_t  w;
    int32_t  phase_shift;
    uint32_t flags;
} VkPrecompPtsDfPush;

typedef struct VkPrecompPtsDPush {
    double   df_scaled;
    int32_t  M;
    int32_t  nfft;
    int32_t  w;
    int32_t  phase_shift;
    uint32_t flags;
    uint32_t _pad;
} VkPrecompPtsDPush;

typedef struct VkBucketWinPush {
    int32_t M;
    int32_t w;
    int32_t log2_win;
    int32_t win_mask;
    int32_t nw_mask;
} VkBucketWinPush;

typedef struct VkBucketScanPush {
    int32_t num_wins;
} VkBucketScanPush;

typedef struct VkPrecompWPtsPush {
    int32_t M;
} VkPrecompWPtsPush;

typedef struct VkPackGeomPush {
    int32_t w;
    int32_t num_wins;
    int32_t has_w;
} VkPackGeomPush;

typedef struct VkSpreadPush {
    int32_t  M;
    int32_t  nfft;
    int32_t  win_elems;
    int32_t  num_wins;
    int32_t  log2_nw;
    uint32_t total_M_stride;
    uint32_t chunk_bs;
    uint32_t y_offset_elems;
} VkSpreadPush;

typedef struct VkSpreadCpuPush {
    int32_t  M;
    int32_t  nfft;
    uint32_t total_M_stride;
    uint32_t chunk_bs;
    uint32_t y_offset_elems;
} VkSpreadCpuPush;

typedef struct VkDeconvTblF32Push {
    float   output_scale;
    int32_t nfft;
    int32_t log_nfft;
    int32_t nout;
    int32_t output_shift;
    int32_t mode43;
    int32_t table_kind;
} VkDeconvTblF32Push;

typedef struct VkDeconvTblDfPush {
    float   scale_hi;
    float   scale_lo;
    int32_t nfft;
    int32_t log_nfft;
    int32_t nout;
    int32_t output_shift;
    int32_t mode43;
    int32_t table_kind;
} VkDeconvTblDfPush;

typedef struct VkDeconvTblF64Push {
    double  output_scale;
    int32_t nfft;
    int32_t log_nfft;
    int32_t nout;
    int32_t output_shift;
    int32_t mode43;
    int32_t table_kind;
} VkDeconvTblF64Push;

typedef struct VkDeconvEmitF32Push {
    float    output_scale;
    int32_t  nfft;
    int32_t  log_nfft;
    int32_t  nout;
    int32_t  output_shift;
    int32_t  mode43;
    int32_t  order_mode;
    int32_t  batch_tile;
    uint32_t chunk_bs;
    uint32_t total_nout_stride;
    uint32_t out_offset_elems;
} VkDeconvEmitF32Push;

typedef struct VkDeconvEmitDfPush {
    float    scale_hi;
    float    scale_lo;
    int32_t  nfft;
    int32_t  log_nfft;
    int32_t  nout;
    int32_t  output_shift;
    int32_t  mode43;
    int32_t  order_mode;
    int32_t  batch_tile;
    uint32_t chunk_bs;
    uint32_t total_nout_stride;
    uint32_t out_offset_elems;
} VkDeconvEmitDfPush;

typedef struct VkDeconvEmitF64Push {
    double   output_scale;
    int32_t  nfft;
    int32_t  log_nfft;
    int32_t  nout;
    int32_t  output_shift;
    int32_t  mode43;
    int32_t  order_mode;
    int32_t  batch_tile;
    uint32_t chunk_bs;
    uint32_t total_nout_stride;
    uint32_t out_offset_elems;
} VkDeconvEmitF64Push;

static_assert(sizeof(VkPrecompPtsDfPush) == 28);
static_assert(sizeof(VkPrecompPtsDPush) == 32);
static_assert(sizeof(VkSpreadPush) == 32);
static_assert(sizeof(VkSpreadCpuPush) == 20);
static_assert(sizeof(VkDeconvTblF32Push) == 28);
static_assert(sizeof(VkDeconvTblDfPush) == 32);
static_assert(sizeof(VkDeconvTblF64Push) == 32);
static_assert(sizeof(VkDeconvEmitF32Push) == 44);
static_assert(sizeof(VkDeconvEmitDfPush) == 48);
static_assert(sizeof(VkDeconvEmitF64Push) == 48);

/* ============================================================================
 * Small Integer & Shader ID Helpers
 * ============================================================================ */

static int next_pow2_i(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static int pow2_floor_i(int v) {
    int p = 1;
    while ((p << 1) <= v) p <<= 1;
    return p;
}

static int pow2_clamp_i(int v, int lo, int hi) {
    if (v <= lo) return lo;
    if (v >= hi) return hi;
    return pow2_floor_i(v);
}

static int log2_pow2_i(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

static bool valid_width_for_prec(VkPrecKind prec, int w) {
    switch (prec) {
        case VK_PREC_HALF:   return w == 4 || w == 5;
        case VK_PREC_FLOAT:  return w == 8 || w == 9;
        case VK_PREC_DF:     return w == 14 || w == 15 || w == 16;
        case VK_PREC_DOUBLE: return w == 16 || w == 17 || w == 18;
    }
    return false;
}

static bool reg_q_supported(VkPrecKind prec, int q_val) {
    switch (prec) {
        case VK_PREC_HALF:   return q_val == 8 || q_val == 16 || q_val == 32;
        case VK_PREC_FLOAT:  return q_val == 4 || q_val == 8 || q_val == 16 || q_val == 32;
        case VK_PREC_DF:     return q_val == 4 || q_val == 8 || q_val == 16;
        case VK_PREC_DOUBLE: return q_val == 2 || q_val == 4 || q_val == 8 || q_val == 16;
    }
    return false;
}

static VkShaderId shader_precompute_points(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return VK_SHADER_nufft_precompute_points_h;
        case VK_PREC_FLOAT:  return VK_SHADER_nufft_precompute_points_f;
        case VK_PREC_DF:     return VK_SHADER_nufft_precompute_points_df;
        case VK_PREC_DOUBLE: return VK_SHADER_nufft_precompute_points_d;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_precompute_weights(VkPrecKind prec, int w) {
    switch (prec) {
        case VK_PREC_HALF:
            return (w == 4) ? VK_SHADER_nufft_precompute_weights_h_w4
                            : VK_SHADER_nufft_precompute_weights_h_w5;
        case VK_PREC_FLOAT:
            return (w == 8) ? VK_SHADER_nufft_precompute_weights_f_w8
                            : VK_SHADER_nufft_precompute_weights_f_w9;
        case VK_PREC_DF:
            if (w == 14) return VK_SHADER_nufft_precompute_weights_df_w14;
            if (w == 15) return VK_SHADER_nufft_precompute_weights_df_w15;
            return VK_SHADER_nufft_precompute_weights_df_w16;
        case VK_PREC_DOUBLE:
            if (w == 16) return VK_SHADER_nufft_precompute_weights_d_w16;
            if (w == 17) return VK_SHADER_nufft_precompute_weights_d_w17;
            return VK_SHADER_nufft_precompute_weights_d_w18;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_pack_bucket_geom(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return VK_SHADER_nufft_pack_bucket_geom_h;
        case VK_PREC_FLOAT:  return VK_SHADER_nufft_pack_bucket_geom_f;
        case VK_PREC_DF:     return VK_SHADER_nufft_pack_bucket_geom_df;
        case VK_PREC_DOUBLE: return VK_SHADER_nufft_pack_bucket_geom_d;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_spread_smem(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return VK_SHADER_nufft_spread_smem_h;
        case VK_PREC_FLOAT:  return VK_SHADER_nufft_spread_smem_f;
        case VK_PREC_DF:     return VK_SHADER_nufft_spread_smem_df;
        case VK_PREC_DOUBLE: return VK_SHADER_nufft_spread_smem_d;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_spread_reg(VkPrecKind prec, int q_val) {
    switch (prec) {
        case VK_PREC_HALF:
            if (q_val == 8)  return VK_SHADER_nufft_spread_reg_h_q8;
            if (q_val == 16) return VK_SHADER_nufft_spread_reg_h_q16;
            if (q_val == 32) return VK_SHADER_nufft_spread_reg_h_q32;
            break;
        case VK_PREC_FLOAT:
            if (q_val == 4)  return VK_SHADER_nufft_spread_reg_f_q4;
            if (q_val == 8)  return VK_SHADER_nufft_spread_reg_f_q8;
            if (q_val == 16) return VK_SHADER_nufft_spread_reg_f_q16;
            if (q_val == 32) return VK_SHADER_nufft_spread_reg_f_q32;
            break;
        case VK_PREC_DF:
            if (q_val == 4)  return VK_SHADER_nufft_spread_reg_df_q4;
            if (q_val == 8)  return VK_SHADER_nufft_spread_reg_df_q8;
            if (q_val == 16) return VK_SHADER_nufft_spread_reg_df_q16;
            break;
        case VK_PREC_DOUBLE:
            if (q_val == 2)  return VK_SHADER_nufft_spread_reg_d_q2;
            if (q_val == 4)  return VK_SHADER_nufft_spread_reg_d_q4;
            if (q_val == 8)  return VK_SHADER_nufft_spread_reg_d_q8;
            if (q_val == 16) return VK_SHADER_nufft_spread_reg_d_q16;
            break;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_spread_cpu_direct(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return VK_SHADER_nufft_spread_cpu_direct_h;
        case VK_PREC_FLOAT:  return VK_SHADER_nufft_spread_cpu_direct_f;
        case VK_PREC_DF:     return VK_SHADER_nufft_spread_cpu_direct_df;
        case VK_PREC_DOUBLE: return VK_SHADER_nufft_spread_cpu_direct_d;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_deconv_table(VkPrecKind prec, int w, bool h_f32) {
    if (h_f32 && prec == VK_PREC_HALF) {
        return (w == 4) ? VK_SHADER_nufft_deconv_table_h_f32_w4
                        : VK_SHADER_nufft_deconv_table_h_f32_w5;
    }
    switch (prec) {
        case VK_PREC_HALF:
            return (w == 4) ? VK_SHADER_nufft_deconv_table_h_w4
                            : VK_SHADER_nufft_deconv_table_h_w5;
        case VK_PREC_FLOAT:
            return (w == 8) ? VK_SHADER_nufft_deconv_table_f_w8
                            : VK_SHADER_nufft_deconv_table_f_w9;
        case VK_PREC_DF:
            if (w == 14) return VK_SHADER_nufft_deconv_table_df_w14;
            if (w == 15) return VK_SHADER_nufft_deconv_table_df_w15;
            return VK_SHADER_nufft_deconv_table_df_w16;
        case VK_PREC_DOUBLE:
            if (w == 16) return VK_SHADER_nufft_deconv_table_d_w16;
            if (w == 17) return VK_SHADER_nufft_deconv_table_d_w17;
            return VK_SHADER_nufft_deconv_table_d_w18;
    }
    return VK_SHADER_COUNT;
}

static VkShaderId shader_deconv_emit(VkPrecKind prec, int w) {
    switch (prec) {
        case VK_PREC_HALF:
            return (w == 4) ? VK_SHADER_nufft_deconv_emit_h_w4
                            : VK_SHADER_nufft_deconv_emit_h_w5;
        case VK_PREC_FLOAT:
            return (w == 8) ? VK_SHADER_nufft_deconv_emit_f_w8
                            : VK_SHADER_nufft_deconv_emit_f_w9;
        case VK_PREC_DF:
            if (w == 14) return VK_SHADER_nufft_deconv_emit_df_w14;
            if (w == 15) return VK_SHADER_nufft_deconv_emit_df_w15;
            return VK_SHADER_nufft_deconv_emit_df_w16;
        case VK_PREC_DOUBLE:
            if (w == 16) return VK_SHADER_nufft_deconv_emit_d_w16;
            if (w == 17) return VK_SHADER_nufft_deconv_emit_d_w17;
            return VK_SHADER_nufft_deconv_emit_d_w18;
    }
    return VK_SHADER_COUNT;
}

/* ============================================================================
 * Kernel Dispatch Recorders
 * ============================================================================ */

static bool record_precompute_points(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    VkBufferSlice d_x,
    int M,
    double df_scaled_d) {
    const uint32_t wg = (uint32_t)(plan->precompute_wg > 0 ? plan->precompute_wg : 64u);
    const uint32_t gx = ((uint32_t)M + wg - 1u) / wg;
    const bool mode43 = (plan->mode == VK_PSWF43);
    const int phase_shift = mode43 ? (plan->output_shift / 3) : plan->output_shift;
    const uint32_t flags = (mode43 ? 1u : 0u) | ((plan->sign > 0) ? 2u : 0u);

    VkSpecConstEntry specs[2] = {
        {0u, wg},
        {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (plan->prec == VK_PREC_DOUBLE) ? 1u : 2u;

    VkBufferSlice binds[4] = {
        d_x,
        vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_u, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_shift, 0, VK_WHOLE_SIZE),
    };

    if (plan->prec == VK_PREC_DOUBLE) {
        VkPrecompPtsDPush pc = {df_scaled_d, M, plan->nfft, plan->w, phase_shift, flags, 0u};
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(
            plan->ctx, shader_precompute_points(plan->prec), 4u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 4u, &pc, sizeof(pc), gx, 1u, 1u);
    } else {
        VkDf dfs = vk_df_from_double(df_scaled_d);
        VkPrecompPtsDfPush pc = {dfs.hi, dfs.lo, M, plan->nfft, plan->w, phase_shift, flags};
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(
            plan->ctx, shader_precompute_points(plan->prec), 4u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 4u, &pc, sizeof(pc), gx, 1u, 1u);
    }
}

static bool record_precompute_weights(VkNufft1Plan* plan, VkCmdBatch* batch, int M) {
    const uint32_t wg = (uint32_t)(plan->precompute_wg > 0 ? plan->precompute_wg : 64u);
    const uint32_t gx = ((uint32_t)M + wg - 1u) / wg;
    VkSpecConstEntry specs[2] = {
        {0u, wg},
        {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (plan->prec == VK_PREC_DF) ? 2u : 1u;
    VkPrecompWPtsPush pc = {M};
    const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(
        plan->ctx, shader_precompute_weights(plan->prec, plan->w), 3u, sizeof(pc), specs, nspec, 0u);
    if (!pipe) return false;
    VkBufferSlice binds[3] = {
        vk_buf_slice(&plan->d_pswf_tbl, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_u, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_weights, 0, VK_WHOLE_SIZE),
    };
    return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, 1u, 1u);
}

static bool record_build_window_buckets(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    int M,
    int win_elems,
    int num_wins,
    VkBufferSlice d_cnt,
    VkBufferSlice d_off,
    VkBufferSlice d_bm) {
    if (num_wins <= 1 || M <= 0) return true;
    const uint32_t wg = (uint32_t)(plan->precompute_wg > 0 ? plan->precompute_wg : 64u);
    const uint32_t gx = ((uint32_t)M + wg - 1u) / wg;

    VkSpecConstEntry spec_wg = {0u, wg};
    VkBucketWinPush pc = {M, plan->w, log2_pow2_i(win_elems), win_elems - 1, num_wins - 1};

    vk_batch_fill(batch, d_cnt, (VkDeviceSize)num_wins * sizeof(int32_t), 0u);
    vk_batch_barrier_t2c(batch);

    /* Pass 1: count */
    {
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, VK_SHADER_nufft_bucket_count, 2u, sizeof(pc), &spec_wg, 1u, 0u);
        if (!p) return false;
        VkBufferSlice b[2] = {vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE), d_cnt};
        if (!vk_batch_dispatch(batch, p, b, 2u, &pc, sizeof(pc), gx, 1u, 1u)) return false;
    }
    vk_batch_barrier_c2c(batch);

    /* Pass 2: exclusive scan */
    {
        VkBucketScanPush spc = {num_wins};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, VK_SHADER_nufft_bucket_scan, 2u, sizeof(spc), nullptr, 0u, 0u);
        if (!p) return false;
        VkBufferSlice b[2] = {d_cnt, d_off};
        if (!vk_batch_dispatch(batch, p, b, 2u, &spc, sizeof(spc), 1u, 1u, 1u)) return false;
    }
    vk_batch_barrier_c2c(batch);

    /* Pass 3: scatter */
    {
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, VK_SHADER_nufft_bucket_scatter, 3u, sizeof(pc), &spec_wg, 1u, 0u);
        if (!p) return false;
        VkBufferSlice b[3] = {vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE), d_cnt, d_bm};
        if (!vk_batch_dispatch(batch, p, b, 3u, &pc, sizeof(pc), gx, 1u, 1u)) return false;
    }
    vk_batch_barrier_c2c(batch);

    /* Pass 4: per-window bitonic sort (cnt <= 512) */
    {
        uint32_t sort_wg = wg;
        if (sort_wg < 64u) sort_wg = 64u;
        if (sort_wg > 256u) sort_wg = 256u;
        if (sort_wg > (uint32_t)vk_ctx_config(plan->ctx)->max_work_group_size) {
            sort_wg = (uint32_t)pow2_floor_i((int)vk_ctx_config(plan->ctx)->max_work_group_size);
        }
        VkSpecConstEntry spec_sort = {0u, sort_wg};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, VK_SHADER_nufft_bucket_sort, 2u, 0u, &spec_sort, 1u, 0u);
        if (!p) return false;
        VkBufferSlice b[2] = {d_off, d_bm};
        if (!vk_batch_dispatch(batch, p, b, 2u, nullptr, 0u, (uint32_t)num_wins, 1u, 1u)) return false;
    }
    return true;
}

static bool record_pack_bucket_geom(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    int num_wins,
    int max_entries,
    VkBufferSlice d_off,
    VkBufferSlice d_bm,
    VkBufferSlice d_bf,
    VkBufferSlice d_bs,
    VkBufferSlice d_bw) {
    if (num_wins <= 1 || max_entries <= 0) return true;
    const uint32_t wg = (uint32_t)(plan->precompute_wg > 0 ? plan->precompute_wg : 64u);
    const uint32_t gx = ((uint32_t)max_entries + wg - 1u) / wg;
    VkSpecConstEntry spec_wg = {0u, wg};
    VkPackGeomPush pc = {plan->w, num_wins, 1};
    const VkCompiledPipeline* p = vk_ctx_get_pipeline(
        plan->ctx, shader_pack_bucket_geom(plan->prec), 8u, sizeof(pc), &spec_wg, 1u, 0u);
    if (!p) return false;
    VkBufferSlice b[8] = {
        d_off,
        d_bm,
        vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_shift, 0, VK_WHOLE_SIZE),
        vk_buf_slice(&plan->d_weights, 0, VK_WHOLE_SIZE),
        d_bf,
        d_bs,
        d_bw,
    };
    return vk_batch_dispatch(batch, p, b, 8u, &pc, sizeof(pc), gx, 1u, 1u);
}

static bool record_spread_cfg(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    const VkSpreadConfig* sc,
    VkBufferSlice d_weights,
    VkBufferSlice d_first,
    VkBufferSlice d_shift,
    VkBufferSlice d_off,
    VkBufferSlice d_bm,
    VkBufferSlice d_y,
    uint32_t y_offset_elems,
    VkBufferSlice d_fft_buf,
    int M,
    size_t cur_bs) {
    const size_t bs_packed = (cur_bs + plan->lanes - 1u) / plan->lanes;

    if (sc->use_cpu_direct) {
        const uint32_t wg = 1u;
        VkSpecConstEntry specs[3] = {
            {0u, wg},
            {1u, (uint32_t)plan->w},
            {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
        };
        size_t nspec = (plan->prec == VK_PREC_DF) ? 3u : 2u;
        VkSpreadCpuPush pc = {M, plan->nfft, (uint32_t)M, (uint32_t)cur_bs, y_offset_elems};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, shader_spread_cpu_direct(plan->prec), 5u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        VkBufferSlice b[5] = {d_weights, d_first, d_shift, d_y, d_fft_buf};
        return vk_batch_dispatch(batch, p, b, 5u, &pc, sizeof(pc), (uint32_t)bs_packed, 1u, 1u);
    }

    const uint32_t num_b_groups = (uint32_t)((bs_packed + (size_t)sc->b_per_wg - 1u) / (size_t)sc->b_per_wg);
    const uint32_t total_wgs = num_b_groups * (uint32_t)sc->num_wins;
    const uint32_t sy_stride = (uint32_t)(sc->b_per_wg + 1);
    const uint32_t multipass = (sc->t_j < plan->w) ? 1u : 0u;

    VkSpreadPush pc = {
        M,
        plan->nfft,
        sc->win_elems,
        sc->num_wins,
        log2_pow2_i(sc->num_wins),
        (uint32_t)M,
        (uint32_t)cur_bs,
        y_offset_elems,
    };

    VkBufferSlice dummy = vk_ctx_dummy_buffer(plan->ctx);
    VkBufferSlice b[7] = {
        d_weights,
        d_first,
        d_shift,
        (sc->num_wins > 1 && vk_slice_valid(d_off)) ? d_off : dummy,
        (sc->num_wins > 1 && vk_slice_valid(d_bm))  ? d_bm  : dummy,
        d_y,
        d_fft_buf,
    };

    if (sc->use_reg_acc) {
        const int q_val = sc->win_elems / sc->t_j;
        VkShaderId sid = shader_spread_reg(plan->prec, q_val);
        if (sid == VK_SHADER_COUNT) return false;
        VkSpecConstEntry specs[11] = {
            {0u,  (uint32_t)sc->wg_size},
            {1u,  (uint32_t)plan->w},
            {2u,  (uint32_t)sc->t_j},
            {3u,  (uint32_t)log2_pow2_i(sc->t_j)},
            {4u,  (uint32_t)sc->b_per_wg},
            {5u,  (uint32_t)sc->tile_m},
            {6u,  (uint32_t)log2_pow2_i(sc->tile_m)},
            {9u,  (uint32_t)sc->tile_m * sy_stride},
            {10u, (uint32_t)sc->tile_m * (uint32_t)plan->w},
            {11u, multipass},
            {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
        };
        size_t nspec = (plan->prec == VK_PREC_DF) ? 11u : 10u;
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(
            plan->ctx, sid, 7u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 7u, &pc, sizeof(pc), total_wgs, 1u, 1u);
    }

    VkSpecConstEntry specs[13] = {
        {0u,  (uint32_t)sc->wg_size},
        {1u,  (uint32_t)plan->w},
        {2u,  (uint32_t)sc->t_j},
        {3u,  (uint32_t)log2_pow2_i(sc->t_j)},
        {4u,  (uint32_t)sc->b_per_wg},
        {5u,  (uint32_t)sc->tile_m},
        {6u,  (uint32_t)log2_pow2_i(sc->tile_m)},
        {7u,  (uint32_t)sc->grid_stride},
        {8u,  (uint32_t)(sc->b_per_wg * sc->grid_stride)},
        {9u,  (uint32_t)sc->tile_m * sy_stride},
        {10u, (uint32_t)sc->tile_m * (uint32_t)plan->w},
        {11u, multipass},
        {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (plan->prec == VK_PREC_DF) ? 13u : 12u;
    const VkCompiledPipeline* p = vk_ctx_get_pipeline(
        plan->ctx, shader_spread_smem(plan->prec), 7u, sizeof(pc), specs, nspec, 0u);
    if (!p) return false;
    return vk_batch_dispatch(batch, p, b, 7u, &pc, sizeof(pc), total_wgs, 1u, 1u);
}

static bool record_build_deconv_table(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    int table_kind,
    bool h_f32,
    VkBufferSlice d_out,
    double output_scale) {
    const uint32_t gx = ((uint32_t)plan->nout + 63u) / 64u;
    const int mode43 = (plan->mode == VK_PSWF43) ? 1 : 0;
    VkBufferSlice b[2] = {vk_buf_slice(&plan->d_pswf_tbl, 0, VK_WHOLE_SIZE), d_out};
    VkShaderId sid = shader_deconv_table(plan->prec, plan->w, h_f32);

    if (plan->prec == VK_PREC_DOUBLE) {
        VkDeconvTblF64Push pc = {output_scale, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift, mode43, table_kind};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 2u, sizeof(pc), nullptr, 0u, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 2u, &pc, sizeof(pc), gx, 1u, 1u);
    } else if (plan->prec == VK_PREC_DF) {
        VkDf sc = vk_df_from_double(output_scale);
        VkDeconvTblDfPush pc = {sc.hi, sc.lo, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift, mode43, table_kind};
        VkSpecConstEntry spec = {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 2u, sizeof(pc), &spec, 1u, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 2u, &pc, sizeof(pc), gx, 1u, 1u);
    } else {
        VkDeconvTblF32Push pc = {(float)output_scale, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift, mode43, table_kind};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 2u, sizeof(pc), nullptr, 0u, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 2u, &pc, sizeof(pc), gx, 1u, 1u);
    }
}

static bool record_deconv_emit(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    const VkDeconvConfig* dc,
    VkBufferSlice d_fft_buf,
    VkBufferSlice d_out,
    uint32_t out_offset_elems,
    size_t cur_bs,
    VkNufftOutOrder order) {
    const uint32_t wg_i = (uint32_t)(dc->wg_i > 0 ? dc->wg_i : 64);
    const uint32_t bt   = (uint32_t)(dc->batch_tile > 0 ? dc->batch_tile : 32);
    const size_t bs_packed = (cur_bs + plan->lanes - 1u) / plan->lanes;
    const uint32_t gx = ((uint32_t)plan->nout + wg_i - 1u) / wg_i;
    const uint32_t gy = (uint32_t)((bs_packed + bt - 1u) / bt);
    const int mode43 = (plan->mode == VK_PSWF43) ? 1 : 0;

    VkSpecConstEntry specs[2] = {
        {0u, wg_i},
        {15u, plan->fft_plan.df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (plan->prec == VK_PREC_DF) ? 2u : 1u;
    VkBufferSlice b[3] = {
        vk_buf_slice(&plan->d_pswf_tbl, 0, VK_WHOLE_SIZE),
        d_fft_buf,
        d_out,
    };
    VkShaderId sid = shader_deconv_emit(plan->prec, plan->w);

    if (plan->prec == VK_PREC_DOUBLE) {
        VkDeconvEmitF64Push pc = {
            plan->output_scale, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift,
            mode43, (int32_t)order, (int32_t)bt, (uint32_t)cur_bs, (uint32_t)plan->nout, out_offset_elems,
        };
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 3u, &pc, sizeof(pc), gx, gy, 1u);
    } else if (plan->prec == VK_PREC_DF) {
        VkDf sc = vk_df_from_double(plan->output_scale);
        VkDeconvEmitDfPush pc = {
            sc.hi, sc.lo, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift,
            mode43, (int32_t)order, (int32_t)bt, (uint32_t)cur_bs, (uint32_t)plan->nout, out_offset_elems,
        };
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 3u, &pc, sizeof(pc), gx, gy, 1u);
    } else {
        VkDeconvEmitF32Push pc = {
            (float)plan->output_scale, plan->nfft, plan->log_nfft, plan->nout, plan->output_shift,
            mode43, (int32_t)order, (int32_t)bt, (uint32_t)cur_bs, (uint32_t)plan->nout, out_offset_elems,
        };
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(plan->ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, b, 3u, &pc, sizeof(pc), gx, gy, 1u);
    }
}

/* ============================================================================
 * Hardware-Derived Candidate Generation (Zero Hardcoded GPU Constants)
 * ============================================================================ */

typedef struct VkSpreadCandidate {
    char           variant_id[64];
    VkSpreadConfig cfg;
    size_t         smem_bytes;
    int            score;
} VkSpreadCandidate;

typedef struct VkDeconvCandidate {
    char           variant_id[64];
    VkDeconvConfig cfg;
} VkDeconvCandidate;

static size_t calc_spread_smem_bytes(
    size_t real_bytes, size_t scalar_bytes, int w,
    int b_per_wg, int win_elems, int tile_m, int grid_pad, bool use_reg_acc) {
    int grid_stride = win_elems + grid_pad;
    int sy_stride = b_per_wg + 1;
    size_t grid_b = use_reg_acc ? 0u : (2u * (size_t)b_per_wg * (size_t)grid_stride * real_bytes);
    return grid_b +
           (size_t)tile_m * sizeof(int32_t) +
           2u * (size_t)tile_m * (size_t)sy_stride * real_bytes +
           (size_t)tile_m * (size_t)w * scalar_bytes;
}

static size_t build_spread_candidates(
    const VkDeviceFftConfig* dev_cfg,
    VkPrecKind prec,
    size_t real_bytes,
    size_t scalar_bytes,
    size_t eff_sg,
    int nfft,
    int w,
    bool allow_emulated_smem,
    VkSpreadCandidate* out,
    size_t cap) {
    size_t n = 0;
    VkBankModel bm = vk_bank_model_from_kind(dev_cfg->bank_model);
    size_t sg_unit = (eff_sg > 1u) ? eff_sg : (dev_cfg->sub_group_size > 1u ? dev_cfg->sub_group_size : 1u);
    int t_j_hi = next_pow2_i(w);
    bool has_mp = (w == 5 || w == 9 || w == 17 || w == 18);
    int tj_opts[2];
    int n_tj = 0;
    if (has_mp) tj_opts[n_tj++] = pow2_floor_i(w);
    tj_opts[n_tj++] = t_j_hi;

    bool smem_allowed = (dev_cfg->has_native_local_mem || allow_emulated_smem) && dev_cfg->local_mem_size >= 4096u;
    if (smem_allowed) {
        size_t smem_limit = dev_cfg->local_mem_size - (dev_cfg->local_mem_size / 16u);
        int min_win = next_pow2_i(w * 2 > 64 ? w * 2 : 64);
        if (min_win > nfft) min_win = nfft;
        int wg_max = (int)dev_cfg->max_work_group_size;
        if (wg_max > 256) wg_max = 256;

        for (int ti = 0; ti < n_tj; ++ti) {
            int t_j = tj_opts[ti];
            int grid_pad = 0;
            if (bm.banks > 1) {
                int elem_bytes = (real_bytes == 4u) ? 8 : (int)real_bytes;
                int bpe = (elem_bytes + bm.bank_width_bytes - 1) / bm.bank_width_bytes;
                int elem_banks = bm.banks / (bpe > 0 ? bpe : 1);
                if (elem_banks < 1) elem_banks = 1;
                if (t_j < elem_banks) grid_pad = t_j;
            }
            int wg_min = (t_j > (int)sg_unit) ? t_j : (int)sg_unit;

            int wg_list[4];
            int n_wg = 0;
            const int mults[4] = {2, 4, 8, 1};
            for (int mi = 0; mi < 4; ++mi) {
                int wg = pow2_floor_i((int)sg_unit * mults[mi]);
                if (wg >= wg_min && wg <= wg_max && (wg % t_j) == 0) {
                    bool dup = false;
                    for (int k = 0; k < n_wg; ++k) if (wg_list[k] == wg) dup = true;
                    if (!dup) wg_list[n_wg++] = wg;
                }
            }
            if (n_wg == 0 && wg_min <= (int)dev_cfg->max_work_group_size) {
                wg_list[n_wg++] = wg_min;
            }

            for (int wi = 0; wi < n_wg; ++wi) {
                int wg_size = wg_list[wi];
                int b_per_wg = wg_size / t_j;
                if (b_per_wg <= 0) continue;

                /* Register-resident candidates */
                const int q_list[5] = {32, 16, 8, 4, 2};
                for (int qi = 0; qi < 5; ++qi) {
                    int q_val = q_list[qi];
                    if (!reg_q_supported(prec, q_val)) continue;
                    int win_elems = q_val * t_j;
                    if (win_elems < min_win || win_elems > nfft || win_elems < 64) continue;

                    int tile_m = (wg_size < (int)(sg_unit > 32u ? sg_unit : 32u)) ? wg_size : (int)(sg_unit > 32u ? sg_unit : 32u);
                    tile_m = pow2_clamp_i(tile_m, 16, 64);
                    while (tile_m > 16 &&
                           calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m, grid_pad, true) > smem_limit) {
                        tile_m >>= 1;
                    }
                    size_t smem_b = calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m, grid_pad, true);
                    if (smem_b > smem_limit) continue;

                    char id[64];
                    if (t_j < w) snprintf(id, sizeof(id), "reg_wg%d_win%d_tj%d", wg_size, win_elems, t_j);
                    else         snprintf(id, sizeof(id), "reg_wg%d_win%d", wg_size, win_elems);
                    bool dup = false;
                    for (size_t k = 0; k < n; ++k) if (strcmp(out[k].variant_id, id) == 0) dup = true;
                    if (dup || n >= cap) continue;

                    VkSpreadConfig sc = {true, true, false, wg_size, t_j, b_per_wg, win_elems, nfft / win_elems, win_elems + grid_pad, tile_m};
                    int res_wgs = (int)(dev_cfg->local_mem_size / (smem_b > 0 ? smem_b : 1u));
                    if (res_wgs > 4) res_wgs = 4;
                    int act_w = res_wgs * (wg_size / (int)sg_unit > 0 ? wg_size / (int)sg_unit : 1);
                    int score = (act_w < 16 ? act_w : 16) * 20 + (b_per_wg < 32 ? b_per_wg : 32) * 10;
                    score += (q_val <= 8) ? 40 : ((q_val == 16) ? 15 : -20);
                    if (t_j < w) score += 25;
                    if (sc.num_wins == 1) score += 30;
                    else if (sc.num_wins <= 4) score += 18;
                    else score -= log2_pow2_i(sc.num_wins) * 8;
                    if (win_elems < 128) score -= 25;

                    VkSpreadCandidate cand = {0};
                    snprintf(cand.variant_id, sizeof(cand.variant_id), "%s", id);
                    cand.cfg = sc;
                    cand.smem_bytes = smem_b;
                    cand.score = score;
                    out[n++] = cand;
                }

                /* SMEM-grid candidates */
                int max_win = nfft;
                while (max_win >= min_win &&
                       calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, max_win, 16, grid_pad, false) > smem_limit) {
                    max_win >>= 1;
                }
                int win_tries[3] = {max_win, max_win / 2, max_win / 4};
                int n_wt = (max_win >= min_win) ? 1 : 0;
                if (n_wt == 1 && max_win / 2 >= min_win) n_wt = 2;
                if (n_wt == 2 && max_win / 4 >= min_win &&
                    calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, max_win / 2, 16, grid_pad, false) * 2u > dev_cfg->local_mem_size) {
                    n_wt = 3;
                }
                for (int wti = 0; wti < n_wt; ++wti) {
                    int win_elems = win_tries[wti];
                    int tile_m = (wg_size < (int)(sg_unit > 32u ? sg_unit : 32u)) ? wg_size : (int)(sg_unit > 32u ? sg_unit : 32u);
                    tile_m = pow2_clamp_i(tile_m, 16, 64);
                    while (tile_m > 16 &&
                           calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m, grid_pad, false) > smem_limit) {
                        tile_m >>= 1;
                    }
                    while (tile_m > 16) {
                        size_t cur_b  = calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m, grid_pad, false);
                        size_t next_b = calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m >> 1, grid_pad, false);
                        int cur_r = (int)(dev_cfg->local_mem_size / (cur_b > 0 ? cur_b : 1u));
                        int nxt_r = (int)(dev_cfg->local_mem_size / (next_b > 0 ? next_b : 1u));
                        if (cur_r > 4) cur_r = 4;
                        if (nxt_r > 4) nxt_r = 4;
                        if (nxt_r > cur_r) tile_m >>= 1;
                        else break;
                    }
                    size_t smem_b = calc_spread_smem_bytes(real_bytes, scalar_bytes, w, b_per_wg, win_elems, tile_m, grid_pad, false);
                    if (smem_b > smem_limit) continue;

                    char id[64];
                    if (t_j < w) snprintf(id, sizeof(id), "smem_wg%d_win%d_tj%d", wg_size, win_elems, t_j);
                    else         snprintf(id, sizeof(id), "smem_wg%d_win%d", wg_size, win_elems);
                    bool dup = false;
                    for (size_t k = 0; k < n; ++k) if (strcmp(out[k].variant_id, id) == 0) dup = true;
                    if (dup || n >= cap) continue;

                    VkSpreadConfig sc = {true, false, false, wg_size, t_j, b_per_wg, win_elems, nfft / win_elems, win_elems + grid_pad, tile_m};
                    int res_wgs = (int)(dev_cfg->local_mem_size / (smem_b > 0 ? smem_b : 1u));
                    if (res_wgs > 4) res_wgs = 4;
                    int act_w = res_wgs * (wg_size / (int)sg_unit > 0 ? wg_size / (int)sg_unit : 1);
                    int score = (act_w < 16 ? act_w : 16) * 20 + (b_per_wg < 32 ? b_per_wg : 32) * 10;
                    if (t_j < w) score += 25;
                    if (sc.num_wins == 1) score += 30;
                    else if (sc.num_wins <= 4) score += 18;
                    else score -= log2_pow2_i(sc.num_wins) * 8;
                    if (win_elems < 128) score -= 25;

                    VkSpreadCandidate cand = {0};
                    snprintf(cand.variant_id, sizeof(cand.variant_id), "%s", id);
                    cand.cfg = sc;
                    cand.smem_bytes = smem_b;
                    cand.score = score;
                    out[n++] = cand;
                }
            }
        }

        /* Stable insertion sort descending by score */
        for (size_t i = 1; i < n; ++i) {
            VkSpreadCandidate key = out[i];
            size_t j = i;
            while (j > 0 && out[j - 1].score < key.score) {
                out[j] = out[j - 1];
                --j;
            }
            out[j] = key;
        }
    }

    if ((!dev_cfg->has_native_local_mem || n == 0) && n < cap) {
        VkSpreadCandidate dc = {0};
        snprintf(dc.variant_id, sizeof(dc.variant_id), "cpu_direct");
        dc.cfg.use_smem = false;
        dc.cfg.use_reg_acc = false;
        dc.cfg.use_cpu_direct = true;
        dc.cfg.wg_size = 1;
        dc.cfg.t_j = t_j_hi;
        dc.cfg.b_per_wg = 1;
        dc.cfg.win_elems = nfft;
        dc.cfg.num_wins = 1;
        dc.cfg.grid_stride = nfft;
        dc.cfg.tile_m = 1;
        dc.score = -50;
        out[n++] = dc;
    }

    return n;
}

static size_t build_deconv_candidates(
    const VkDeviceFftConfig* dev_cfg,
    size_t eff_sg,
    size_t lanes,
    int nout,
    size_t chunk_bs,
    VkDeconvCandidate* out,
    size_t cap) {
    size_t n = 0;
    size_t sg_unit = (eff_sg > 1u) ? eff_sg : (dev_cfg->sub_group_size > 1u ? dev_cfg->sub_group_size : 1u);
    size_t bs_packed = ((chunk_bs > 0 ? chunk_bs : 1u) + lanes - 1u) / lanes;

    size_t wg_lo_u = sg_unit > 64u ? sg_unit : 64u;
    if (wg_lo_u > dev_cfg->max_work_group_size) wg_lo_u = dev_cfg->max_work_group_size;
    size_t wg_hi_u = sg_unit * 4u < 128u ? sg_unit * 4u : 128u;
    if (wg_hi_u < wg_lo_u) wg_hi_u = wg_lo_u;
    if (wg_hi_u > dev_cfg->max_work_group_size) wg_hi_u = dev_cfg->max_work_group_size;

    int wg_opts[2];
    int n_wg = 0;
    wg_opts[n_wg++] = pow2_floor_i((int)wg_lo_u);
    if (pow2_floor_i((int)wg_hi_u) != wg_opts[0]) {
        wg_opts[n_wg++] = pow2_floor_i((int)wg_hi_u);
    }

    const size_t waves_arr[3] = {4u, 2u, 8u};
    for (int wi = 0; wi < n_wg; ++wi) {
        int wg_i = wg_opts[wi];
        size_t groups_i = ((size_t)nout + (size_t)wg_i - 1u) / (size_t)wg_i;
        for (int wvi = 0; wvi < 3; ++wvi) {
            size_t target_wgs = dev_cfg->max_compute_units * waves_arr[wvi];
            if (target_wgs < 1u) target_wgs = 1u;
            size_t groups_b = (target_wgs + groups_i - 1u) / (groups_i > 0 ? groups_i : 1u);
            if (groups_b < 1u) groups_b = 1u;
            int raw_bt = (int)((bs_packed + groups_b - 1u) / groups_b);
            int bt = pow2_clamp_i(raw_bt, 8, 256);
            char id[64];
            snprintf(id, sizeof(id), "deconv_wg%d_bt%d", wg_i, bt);
            bool dup = false;
            for (size_t k = 0; k < n; ++k) if (strcmp(out[k].variant_id, id) == 0) dup = true;
            if (!dup && n < cap) {
                snprintf(out[n].variant_id, sizeof(out[n].variant_id), "%s", id);
                out[n].cfg.wg_i = wg_i;
                out[n].cfg.batch_tile = bt;
                ++n;
            }
        }
    }
    if (n == 0 && cap > 0) {
        snprintf(out[0].variant_id, sizeof(out[0].variant_id), "deconv_wg64_bt32");
        out[0].cfg.wg_i = pow2_floor_i((int)wg_lo_u);
        out[0].cfg.batch_tile = 32;
        n = 1;
    }
    return n;
}

/* ============================================================================
 * Spreading Autotuner Helpers
 * ============================================================================ */

typedef struct VkBucketCacheEntry {
    int           win_elems;
    VkBufferAlloc cnt, off, m, bf, bs, bw;
} VkBucketCacheEntry;

typedef struct VkSpreadTuneCtx {
    VkNufft1Plan*            plan;
    VkCmdBatch*              batch;
    const VkSpreadCandidate* cands;
    const size_t*            cand_indices;
    VkBufferAlloc*           d_tune_y;
    VkBucketCacheEntry*      bcache;
    size_t                   bcache_len;
    size_t                   tune_bs;
    int                      target_cons_wg;
    bool                     phase_b;
} VkSpreadTuneCtx;

static const VkBucketCacheEntry* find_bcache(const VkSpreadTuneCtx* tc, int win_elems) {
    for (size_t i = 0; i < tc->bcache_len; ++i) {
        if (tc->bcache[i].win_elems == win_elems) return &tc->bcache[i];
    }
    return nullptr;
}

static bool spread_bench_run(size_t pos, int reps, void* user) {
    VkSpreadTuneCtx* tc = (VkSpreadTuneCtx*)user;
    VkNufft1Plan* plan = tc->plan;
    const VkSpreadConfig* sc = &tc->cands[tc->cand_indices[pos]].cfg;

    VkBufferSlice first_s = vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE);
    VkBufferSlice shift_s = vk_buf_slice(&plan->d_shift, 0, VK_WHOLE_SIZE);
    VkBufferSlice w_s     = vk_buf_slice(&plan->d_weights, 0, VK_WHOLE_SIZE);
    VkBufferSlice off_s   = {0};
    VkBufferSlice bm_s    = {0};
    if (sc->use_smem && sc->num_wins > 1) {
        const VkBucketCacheEntry* bc = find_bcache(tc, sc->win_elems);
        if (!bc) return false;
        off_s   = vk_buf_slice(&bc->off, 0, VK_WHOLE_SIZE);
        bm_s    = vk_buf_slice(&bc->m, 0, VK_WHOLE_SIZE);
        first_s = vk_buf_slice(&bc->bf, 0, VK_WHOLE_SIZE);
        shift_s = vk_buf_slice(&bc->bs, 0, VK_WHOLE_SIZE);
        w_s     = vk_buf_slice(&bc->bw, 0, VK_WHOLE_SIZE);
    }

    if (!vk_batch_begin(tc->batch)) return false;
    for (int r = 0; r < reps; ++r) {
        if (r > 0) vk_batch_barrier_c2c(tc->batch);
        if (!record_spread_cfg(
                plan, tc->batch, sc, w_s, first_s, shift_s, off_s, bm_s,
                vk_buf_slice(tc->d_tune_y, 0, VK_WHOLE_SIZE), 0u,
                vk_buf_slice(&plan->d_fft_buf, 0, VK_WHOLE_SIZE),
                plan->M, tc->tune_bs)) {
            return false;
        }
    }
    return vk_batch_submit_and_wait(tc->batch);
}

static bool spread_bench_conserv(size_t a_pos, size_t b_pos, void* user) {
    const VkSpreadTuneCtx* tc = (const VkSpreadTuneCtx*)user;
    const VkSpreadCandidate* ca = &tc->cands[tc->cand_indices[a_pos]];
    const VkSpreadCandidate* cb = &tc->cands[tc->cand_indices[b_pos]];
    if (tc->phase_b) {
        int da = abs(ca->cfg.wg_size - tc->target_cons_wg);
        int db = abs(cb->cfg.wg_size - tc->target_cons_wg);
        if (da != db) return da < db;
        if (ca->smem_bytes != cb->smem_bytes) return ca->smem_bytes < cb->smem_bytes;
        return ca->score > cb->score;
    }
    if (ca->cfg.use_reg_acc != cb->cfg.use_reg_acc) return !ca->cfg.use_reg_acc;
    if (ca->cfg.use_reg_acc) {
        int qa = ca->cfg.win_elems / (ca->cfg.t_j > 0 ? ca->cfg.t_j : 1);
        int qb = cb->cfg.win_elems / (cb->cfg.t_j > 0 ? cb->cfg.t_j : 1);
        if (qa != qb) return qa < qb;
    }
    if (ca->smem_bytes != cb->smem_bytes) return ca->smem_bytes < cb->smem_bytes;
    return ca->score > cb->score;
}

static void fill_synth_complex(void* dst, VkPrecKind prec, size_t n, uint64_t seed_off) {
    for (size_t k = 0; k < n; ++k) {
        uint64_t s = k + seed_off;
        double r = -0.875 + 1.75 * (double)((s * 6364136223846793005ULL + 1ULL) >> 40) / 16777216.0;
        double im = -0.875 + 1.75 * (double)(((s ^ 0x9e3779b97f4a7c15ULL) * 2862933555777941757ULL + 1ULL) >> 40) / 16777216.0;
        if (prec == VK_PREC_HALF) {
            ((VkComplexh*)dst)[k] = vk_complexh_from_doubles(r, im);
        } else if (prec == VK_PREC_FLOAT) {
            ((VkComplexf*)dst)[k] = (VkComplexf){(float)r, (float)im};
        } else if (prec == VK_PREC_DF) {
            ((VkComplexdf*)dst)[k] = (VkComplexdf){vk_df_from_double(r), vk_df_from_double(im)};
        } else {
            ((VkComplex*)dst)[k] = (VkComplex){r, im};
        }
    }
}

static void fill_synth_scalar(void* dst, VkPrecKind prec, size_t n, uint64_t seed_off) {
    for (size_t k = 0; k < n; ++k) {
        uint64_t s = k + seed_off;
        double v = -0.9 + 1.8 * (double)((s * 2654435761ULL) & 0xFFFFu) / 65535.0;
        if (prec == VK_PREC_HALF)       ((uint16_t*)dst)[k] = vk_f32_to_f16_bits((float)v);
        else if (prec == VK_PREC_FLOAT) ((float*)dst)[k] = (float)v;
        else if (prec == VK_PREC_DF)    ((VkDf*)dst)[k] = vk_df_from_double(v);
        else                            ((double*)dst)[k] = v;
    }
}

/* ============================================================================
 * Public Lifecycle & Plan Methods
 * ============================================================================ */

static bool allocate_window_buckets(VkNufft1Plan* plan) {
    if (plan->d_bucket_counts.buffer != VK_NULL_HANDLE) return true;
    size_t max_ent = 2u * (size_t)plan->M_cap;
    if (!vk_buf_create(plan->ctx, (size_t)plan->spread_cfg.num_wins * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &plan->d_bucket_counts) ||
        !vk_buf_create(plan->ctx, (size_t)(plan->spread_cfg.num_wins + 1) * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &plan->d_bucket_off) ||
        !vk_buf_create(plan->ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &plan->d_bucket_m) ||
        !vk_buf_create(plan->ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &plan->d_b_first) ||
        !vk_buf_create(plan->ctx, max_ent * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_b_shift) ||
        !vk_buf_create(plan->ctx, max_ent * (size_t)plan->w * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_b_weights)) {
        return false;
    }
    return true;
}

bool vk_nufft1_plan_init(
    VkNufft1Plan* plan,
    VkDeviceContext* ctx,
    VkPrecKind prec,
    int sign,
    int M,
    int N,
    int w,
    VkPswfMode mode,
    size_t max_bs,
    const VkNufft1LaunchConfig* cfg) {
    memset(plan, 0, sizeof(*plan));
    if (!ctx || M <= 0 || N < 512 || N > (1 << 17) || (N & (N - 1)) != 0) return false;
    if (mode == VK_PSWF43 && (N % 4) != 0) return false;
    if (!valid_width_for_prec(prec, w)) return false;

    plan->ctx = ctx;
    plan->prec = prec;
    plan->sign = (sign >= 0) ? +1 : -1;
    plan->M = M;
    plan->M_cap = M;
    plan->N = N;
    plan->w = w;
    plan->mode = mode;
    plan->log_nfft = log2_pow2_i(2 * N);
    plan->nfft = 1 << plan->log_nfft;
    plan->nout = vk_nufft1_nout(N, mode);
    plan->output_shift = vk_nufft1_output_shift(N, mode);
    plan->max_bs = max_bs > 0 ? max_bs : 1u;
    plan->output_scale = 1.0;
    plan->launch_cfg = cfg ? *cfg : vk_nufft1_launch_config_default();

    switch (prec) {
        case VK_PREC_HALF:
            plan->scalar_bytes = 2u; plan->complex_bytes = 4u; plan->real_bytes = 4u; plan->lanes = 2u; break;
        case VK_PREC_FLOAT:
            plan->scalar_bytes = 4u; plan->complex_bytes = 8u; plan->real_bytes = 4u; plan->lanes = 1u; break;
        case VK_PREC_DF:
            plan->scalar_bytes = 8u; plan->complex_bytes = 16u; plan->real_bytes = 8u; plan->lanes = 1u; break;
        case VK_PREC_DOUBLE:
            plan->scalar_bytes = 8u; plan->complex_bytes = 16u; plan->real_bytes = 8u; plan->lanes = 1u; break;
    }

    if (!vk_dif_plan_init(&plan->fft_plan, ctx, prec, plan->sign, plan->log_nfft,
                          plan->max_bs, &plan->launch_cfg.fft_cfg)) {
        return false;
    }

    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(ctx);
    vk_autotune_resolve_cache_path(plan->launch_cfg.fft_cfg.cache_path, plan->cache_path, sizeof(plan->cache_path));
    plan->precompute_wg = vk_device_default_padded_wg(dev_cfg, 256u);

    /* 1. Build spreading candidates */
    VkSpreadCandidate cands[64];
    size_t ncands = build_spread_candidates(
        dev_cfg, prec, plan->real_bytes, plan->scalar_bytes,
        plan->fft_plan.effective_sg_size, plan->nfft, w,
        plan->launch_cfg.fft_cfg.allow_emulated_local_mem,
        cands, 64u);
    if (ncands == 0) {
        vk_nufft1_plan_destroy(plan);
        return false;
    }

    size_t chosen_spread = 0;
    bool need_spread_tune = false;
    bool has_spread_ov = plan->launch_cfg.spread_variant_override[0] != '\0' ||
                         plan->launch_cfg.fft_cfg.allow_emulated_local_mem;
    bool allow_rt_tune = (plan->max_bs >= 32u) || (plan->launch_cfg.fft_cfg.autotune == VK_AUTOTUNE_FORCE);
    int m_bin = (M <= 128) ? 0 : ((M <= 1024) ? 1 : ((M <= 8192) ? 2 : 3));
    char spr_slot[64], spr_key[512];
    snprintf(spr_slot, sizeof(spr_slot), "nufft1_spr2_w%d_m%d", w, m_bin);
    vk_autotune_make_key(dev_cfg, vk_prec_tag(prec), plan->log_nfft, spr_slot, plan->sign, spr_key, sizeof(spr_key));

    if (plan->launch_cfg.spread_variant_override[0] != '\0') {
        bool found = false;
        for (size_t i = 0; i < ncands; ++i) {
            if (strcmp(cands[i].variant_id, plan->launch_cfg.spread_variant_override) == 0) {
                chosen_spread = i;
                found = true;
                break;
            }
        }
        if (!found) {
            vk_nufft1_plan_destroy(plan);
            return false;
        }
    } else if (plan->launch_cfg.fft_cfg.autotune == VK_AUTOTUNE_AUTO && !has_spread_ov) {
        VkAutotuneEntry cached = {0};
        if (vk_autotune_cache_get(plan->cache_path, spr_key, &cached)) {
            for (size_t i = 0; i < ncands; ++i) {
                if (strcmp(cands[i].variant_id, cached.variant_id) == 0) {
                    chosen_spread = i;
                    plan->spread_cache_hit = true;
                    break;
                }
            }
        }
        if (!plan->spread_cache_hit && allow_rt_tune && ncands > 1u) {
            need_spread_tune = true;
        }
    } else if (plan->launch_cfg.fft_cfg.autotune == VK_AUTOTUNE_FORCE && ncands > 1u) {
        need_spread_tune = true;
    }

    plan->spread_cfg = cands[chosen_spread].cfg;
    snprintf(plan->spread_variant_name, sizeof(plan->spread_variant_name), "%.63s", cands[chosen_spread].variant_id);

    /* 2. Unified L2-aware chunk_bs coordinated with fft_plan */
    size_t fft_bytes = (size_t)plan->nfft * plan->complex_bytes;
    size_t l2_bytes = plan->launch_cfg.chunk_bytes_override > 0
        ? plan->launch_cfg.chunk_bytes_override
        : (plan->launch_cfg.fft_cfg.l2_chunk_bytes > 0
               ? plan->launch_cfg.fft_cfg.l2_chunk_bytes
               : vk_device_compute_l2_chunk_bytes(
                     dev_cfg, fft_bytes,
                     (size_t)(plan->spread_cfg.num_wins > 1 ? plan->spread_cfg.num_wins : 1),
                     plan->fft_plan.is_compute_heavy,
                     plan->launch_cfg.fft_cfg.l2_budget_num,
                     plan->launch_cfg.fft_cfg.l2_budget_den,
                     plan->launch_cfg.fft_cfg.min_waves_per_cu));
    size_t cbs = l2_bytes / (fft_bytes > 0 ? fft_bytes : 1u);
    if (cbs < 1u) cbs = 1u;
    if (cbs > plan->max_bs) cbs = plan->max_bs;
    if (plan->fft_plan.chunk_bs > 0 && cbs > plan->fft_plan.chunk_bs) cbs = plan->fft_plan.chunk_bs;
    if (plan->lanes == 2u && cbs > 1u && (cbs & 1u)) cbs &= ~(size_t)1u;
    plan->chunk_bs = cbs;

    /* 3. Upload PSWF table and allocate core device buffers */
    size_t tbl_bytes = 0;
    const void* tbl_host = vk_pswf_get_flat_table(prec, w, mode, plan->launch_cfg.use_geo_scale_fp16, &tbl_bytes);
    if (!tbl_host || tbl_bytes == 0) {
        vk_nufft1_plan_destroy(plan);
        return false;
    }
    if (!vk_buf_create(ctx, tbl_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_pswf_tbl) ||
        !vk_buf_create(ctx, (size_t)M * (size_t)w * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_weights) ||
        !vk_buf_create(ctx, (size_t)M * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &plan->d_first) ||
        !vk_buf_create(ctx, (size_t)M * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_u) ||
        !vk_buf_create(ctx, (size_t)M * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_shift) ||
        !vk_buf_create(ctx, plan->chunk_bs * (size_t)plan->nfft * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_fft_buf)) {
        vk_nufft1_plan_destroy(plan);
        return false;
    }

    VkCmdBatch init_batch = {0};
    if (!vk_batch_init(&init_batch, ctx, 0u)) {
        vk_nufft1_plan_destroy(plan);
        return false;
    }
    if (!vk_batch_upload_sync(&init_batch, tbl_host, vk_buf_slice(&plan->d_pswf_tbl, 0, VK_WHOLE_SIZE), tbl_bytes)) {
        vk_batch_destroy(&init_batch);
        vk_nufft1_plan_destroy(plan);
        return false;
    }

    /* 4. Two-phase interleaved spreading autotuner */
    if (need_spread_tune) {
        size_t phase_a[16];
        size_t n_pa = 0;
        size_t n_reg = 0, n_smem = 0;

        for (size_t i = 0; i < ncands && n_reg < 4u && n_pa < 16u; ++i) {
            if (!cands[i].cfg.use_smem || !cands[i].cfg.use_reg_acc) continue;
            bool seen = false;
            for (size_t k = 0; k < n_pa; ++k) {
                const VkSpreadConfig* p = &cands[phase_a[k]].cfg;
                if (p->use_smem == cands[i].cfg.use_smem && p->use_reg_acc == cands[i].cfg.use_reg_acc &&
                    p->use_cpu_direct == cands[i].cfg.use_cpu_direct &&
                    p->win_elems == cands[i].cfg.win_elems && p->t_j == cands[i].cfg.t_j) seen = true;
            }
            if (!seen) { phase_a[n_pa++] = i; ++n_reg; }
        }
        for (size_t i = 0; i < ncands && n_smem < 3u && n_pa < 16u; ++i) {
            if (!cands[i].cfg.use_smem || cands[i].cfg.use_reg_acc) continue;
            bool seen = false;
            for (size_t k = 0; k < n_pa; ++k) {
                const VkSpreadConfig* p = &cands[phase_a[k]].cfg;
                if (p->use_smem == cands[i].cfg.use_smem && p->use_reg_acc == cands[i].cfg.use_reg_acc &&
                    p->use_cpu_direct == cands[i].cfg.use_cpu_direct &&
                    p->win_elems == cands[i].cfg.win_elems && p->t_j == cands[i].cfg.t_j) seen = true;
            }
            if (!seen) { phase_a[n_pa++] = i; ++n_smem; }
        }

        if (n_pa > 1u) {
            size_t tune_bs = plan->chunk_bs;
            VkBufferAlloc d_tune_y = {0};
            if (vk_buf_create(ctx, tune_bs * (size_t)M * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &d_tune_y)) {
                size_t total_y = tune_bs * (size_t)M;
                void* h_y = malloc(total_y * plan->complex_bytes);
                int32_t* h_first = (int32_t*)malloc((size_t)M * sizeof(int32_t));
                void* h_u = malloc((size_t)M * plan->scalar_bytes);
                void* h_shift = malloc((size_t)M * plan->complex_bytes);
                if (h_y && h_first && h_u && h_shift) {
                    fill_synth_complex(h_y, prec, total_y, 17ULL);
                    for (int m = 0; m < M; ++m) {
                        h_first[m] = (int32_t)(((uint64_t)m * 2654435761ULL) & (uint64_t)(plan->nfft - 1));
                    }
                    fill_synth_scalar(h_u, prec, (size_t)M, 3ULL);
                    fill_synth_complex(h_shift, prec, (size_t)M, 101ULL);

                    bool up_ok =
                        vk_batch_upload_sync(&init_batch, h_y, vk_buf_slice(&d_tune_y, 0, VK_WHOLE_SIZE), total_y * plan->complex_bytes) &&
                        vk_batch_upload_sync(&init_batch, h_first, vk_buf_slice(&plan->d_first, 0, VK_WHOLE_SIZE), (size_t)M * sizeof(int32_t)) &&
                        vk_batch_upload_sync(&init_batch, h_u, vk_buf_slice(&plan->d_u, 0, VK_WHOLE_SIZE), (size_t)M * plan->scalar_bytes) &&
                        vk_batch_upload_sync(&init_batch, h_shift, vk_buf_slice(&plan->d_shift, 0, VK_WHOLE_SIZE), (size_t)M * plan->complex_bytes);
                    if (up_ok && vk_batch_begin(&init_batch) &&
                        record_precompute_weights(plan, &init_batch, M) &&
                        vk_batch_submit_and_wait(&init_batch)) {

                        VkBucketCacheEntry bcache[16];
                        memset(bcache, 0, sizeof(bcache));
                        size_t bcache_len = 0;

                        for (size_t k = 0; k < n_pa; ++k) {
                            const VkSpreadConfig* sc = &cands[phase_a[k]].cfg;
                            if (!sc->use_smem || sc->num_wins <= 1) continue;
                            bool exists = false;
                            for (size_t bi = 0; bi < bcache_len; ++bi) {
                                if (bcache[bi].win_elems == sc->win_elems) exists = true;
                            }
                            if (exists || bcache_len >= 16u) continue;
                            VkBucketCacheEntry* bc = &bcache[bcache_len];
                            bc->win_elems = sc->win_elems;
                            size_t max_ent = 2u * (size_t)M;
                            if (vk_buf_create(ctx, (size_t)sc->num_wins * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &bc->cnt) &&
                                vk_buf_create(ctx, (size_t)(sc->num_wins + 1) * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &bc->off) &&
                                vk_buf_create(ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &bc->m) &&
                                vk_buf_create(ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &bc->bf) &&
                                vk_buf_create(ctx, max_ent * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &bc->bs) &&
                                vk_buf_create(ctx, max_ent * (size_t)w * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &bc->bw) &&
                                vk_batch_begin(&init_batch) &&
                                record_build_window_buckets(plan, &init_batch, M, sc->win_elems, sc->num_wins,
                                    vk_buf_slice(&bc->cnt, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->off, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->m, 0, VK_WHOLE_SIZE)) &&
                                (vk_batch_barrier_c2c(&init_batch), true) &&
                                record_pack_bucket_geom(plan, &init_batch, sc->num_wins, (int)max_ent,
                                    vk_buf_slice(&bc->off, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->m, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->bf, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->bs, 0, VK_WHOLE_SIZE),
                                    vk_buf_slice(&bc->bw, 0, VK_WHOLE_SIZE)) &&
                                vk_batch_submit_and_wait(&init_batch)) {
                                ++bcache_len;
                            } else {
                                vk_buf_destroy(&bc->cnt); vk_buf_destroy(&bc->off); vk_buf_destroy(&bc->m);
                                vk_buf_destroy(&bc->bf); vk_buf_destroy(&bc->bs); vk_buf_destroy(&bc->bw);
                            }
                        }

                        VkSpreadTuneCtx tc = {
                            plan, &init_batch, cands, phase_a, &d_tune_y,
                            bcache, bcache_len, tune_bs,
                            (int)vk_device_clamp_pow2_wg(dev_cfg, 256u, 0u, 128u), false,
                        };
                        VkInterleavedBenchResult res_a = vk_run_interleaved_bench(
                            n_pa, spread_bench_run, spread_bench_conserv, &tc, 5, 2.0, 16, 0.015, 0.06);
                        if (res_a.ok) {
                            size_t best_idx = phase_a[res_a.chosen_idx];
                            double margin = res_a.margin_pct;
                            char runner_up[64] = {0};
                            if (res_a.has_runner_up) {
                                snprintf(runner_up, sizeof(runner_up), "%s", cands[phase_a[res_a.runner_up_idx]].variant_id);
                            }

                            /* Phase B: wg_size refinement on winning family */
                            if (cands[best_idx].cfg.use_smem) {
                                size_t phase_b[8];
                                size_t n_pb = 0;
                                phase_b[n_pb++] = best_idx;
                                const VkSpreadConfig* wf = &cands[best_idx].cfg;
                                for (size_t i = 0; i < ncands && n_pb < 8u; ++i) {
                                    if (i == best_idx) continue;
                                    const VkSpreadConfig* p = &cands[i].cfg;
                                    if (p->use_smem == wf->use_smem && p->use_reg_acc == wf->use_reg_acc &&
                                        p->use_cpu_direct == wf->use_cpu_direct &&
                                        p->win_elems == wf->win_elems && p->t_j == wf->t_j) {
                                        phase_b[n_pb++] = i;
                                    }
                                }
                                if (n_pb > 1u) {
                                    tc.cand_indices = phase_b;
                                    tc.phase_b = true;
                                    VkInterleavedBenchResult res_b = vk_run_interleaved_bench(
                                        n_pb, spread_bench_run, spread_bench_conserv, &tc, 5, 2.0, 16, 0.015, 0.06);
                                    if (res_b.ok) {
                                        best_idx = phase_b[res_b.chosen_idx];
                                        if (res_b.has_runner_up && (!res_a.has_runner_up || fabs(res_b.margin_pct) < fabs(margin))) {
                                            margin = res_b.margin_pct;
                                            snprintf(runner_up, sizeof(runner_up), "%s", cands[phase_b[res_b.runner_up_idx]].variant_id);
                                        }
                                    }
                                }
                            }

                            chosen_spread = best_idx;
                            plan->spread_cfg = cands[chosen_spread].cfg;
                            snprintf(plan->spread_variant_name, sizeof(plan->spread_variant_name), "%s", cands[chosen_spread].variant_id);

                            if (!has_spread_ov) {
                                VkAutotuneEntry ent = {0};
                                ent.mode = VK_PLAN_SINGLE_PASS;
                                snprintf(ent.variant_id, sizeof(ent.variant_id), "%s", plan->spread_variant_name);
                                ent.margin_pct = margin;
                                snprintf(ent.runner_up_id, sizeof(ent.runner_up_id), "%s", runner_up);
                                vk_autotune_cache_put(plan->cache_path, spr_key, &ent);
                            }
                        }

                        for (size_t bi = 0; bi < bcache_len; ++bi) {
                            vk_buf_destroy(&bcache[bi].cnt);
                            vk_buf_destroy(&bcache[bi].off);
                            vk_buf_destroy(&bcache[bi].m);
                            vk_buf_destroy(&bcache[bi].bf);
                            vk_buf_destroy(&bcache[bi].bs);
                            vk_buf_destroy(&bcache[bi].bw);
                        }
                    }
                }
                free(h_y);
                free(h_first);
                free(h_u);
                free(h_shift);
                vk_buf_destroy(&d_tune_y);
            }
        }
    }

    vk_batch_destroy(&init_batch);

    if (plan->spread_cfg.use_smem && plan->spread_cfg.num_wins > 1) {
        if (!allocate_window_buckets(plan)) {
            vk_nufft1_plan_destroy(plan);
            return false;
        }
    }

    /* 5. Deconv config */
    VkDeconvCandidate dcands[16];
    size_t ndc = build_deconv_candidates(
        dev_cfg, plan->fft_plan.effective_sg_size, plan->lanes, plan->nout, plan->chunk_bs, dcands, 16u);
    size_t chosen_dc = 0;
    if (plan->launch_cfg.deconv_variant_override[0] != '\0') {
        bool found = false;
        for (size_t i = 0; i < ndc; ++i) {
            if (strcmp(dcands[i].variant_id, plan->launch_cfg.deconv_variant_override) == 0) {
                chosen_dc = i;
                found = true;
                break;
            }
        }
        if (!found) {
            vk_nufft1_plan_destroy(plan);
            return false;
        }
    }
    plan->deconv_cfg = dcands[chosen_dc].cfg;
    bool fused_primary = (plan->fft_plan.exec_mode != VK_PLAN_MULTISTAGE_GLOBAL) &&
                         (plan->launch_cfg.deconv_variant_override[0] == '\0');
    snprintf(plan->deconv_variant_name, sizeof(plan->deconv_variant_name), "%s",
             fused_primary ? "fused_fft_epilogue" : dcands[chosen_dc].variant_id);

    return true;
}

void vk_nufft1_plan_destroy(VkNufft1Plan* plan) {
    if (!plan) return;
    vk_buf_destroy(&plan->d_pswf_tbl);
    vk_buf_destroy(&plan->d_weights);
    vk_buf_destroy(&plan->d_slot_phi_hat);
    vk_buf_destroy(&plan->d_slot_deconv);
    vk_buf_destroy(&plan->d_nat_deconv);
    vk_buf_destroy(&plan->d_slot_deconv_f32);
    vk_buf_destroy(&plan->d_first);
    vk_buf_destroy(&plan->d_u);
    vk_buf_destroy(&plan->d_shift);
    vk_buf_destroy(&plan->d_bucket_counts);
    vk_buf_destroy(&plan->d_bucket_off);
    vk_buf_destroy(&plan->d_bucket_m);
    vk_buf_destroy(&plan->d_b_first);
    vk_buf_destroy(&plan->d_b_shift);
    vk_buf_destroy(&plan->d_b_weights);
    vk_buf_destroy(&plan->d_fft_buf);
    vk_dif_plan_destroy(&plan->fft_plan);
    memset(plan, 0, sizeof(*plan));
}

bool vk_nufft1_ensure_point_capacity(VkNufft1Plan* plan, int M) {
    if (!plan || M <= 0) return false;
    if (M <= plan->M_cap) {
        plan->M = M;
        return true;
    }
    int new_cap = plan->M_cap + (plan->M_cap >> 1);
    if (new_cap < M) new_cap = M;

    VkBufferAlloc nf = {0}, nu = {0}, ns = {0}, nw = {0};
    if (!vk_buf_create(plan->ctx, (size_t)new_cap * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &nf) ||
        !vk_buf_create(plan->ctx, (size_t)new_cap * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &nu) ||
        !vk_buf_create(plan->ctx, (size_t)new_cap * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &ns) ||
        !vk_buf_create(plan->ctx, (size_t)new_cap * (size_t)plan->w * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &nw)) {
        vk_buf_destroy(&nf); vk_buf_destroy(&nu); vk_buf_destroy(&ns); vk_buf_destroy(&nw);
        return false;
    }

    if (plan->spread_cfg.use_smem && plan->spread_cfg.num_wins > 1) {
        size_t max_ent = 2u * (size_t)new_cap;
        VkBufferAlloc nbm = {0}, nbf = {0}, nbs = {0}, nbw = {0};
        if (!vk_buf_create(plan->ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &nbm) ||
            !vk_buf_create(plan->ctx, max_ent * sizeof(int32_t), VK_MEM_DEVICE_LOCAL, &nbf) ||
            !vk_buf_create(plan->ctx, max_ent * plan->complex_bytes, VK_MEM_DEVICE_LOCAL, &nbs) ||
            !vk_buf_create(plan->ctx, max_ent * (size_t)plan->w * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &nbw)) {
            vk_buf_destroy(&nf); vk_buf_destroy(&nu); vk_buf_destroy(&ns); vk_buf_destroy(&nw);
            vk_buf_destroy(&nbm); vk_buf_destroy(&nbf); vk_buf_destroy(&nbs); vk_buf_destroy(&nbw);
            return false;
        }
        vk_buf_destroy(&plan->d_bucket_m);
        vk_buf_destroy(&plan->d_b_first);
        vk_buf_destroy(&plan->d_b_shift);
        vk_buf_destroy(&plan->d_b_weights);
        plan->d_bucket_m = nbm;
        plan->d_b_first = nbf;
        plan->d_b_shift = nbs;
        plan->d_b_weights = nbw;
    }

    vk_buf_destroy(&plan->d_first);
    vk_buf_destroy(&plan->d_u);
    vk_buf_destroy(&plan->d_shift);
    vk_buf_destroy(&plan->d_weights);
    plan->d_first = nf;
    plan->d_u = nu;
    plan->d_shift = ns;
    plan->d_weights = nw;
    plan->M_cap = new_cap;
    plan->M = M;
    return true;
}

bool vk_nufft1_record_set_points(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    VkBufferSlice d_x,
    int M,
    double df,
    int freq_factor) {
    if (!plan || !batch || M <= 0 || M > plan->M_cap || freq_factor <= 0) return false;
    plan->M = M;
    double df_scaled = df * (double)freq_factor;

    if (!record_precompute_points(plan, batch, d_x, M, df_scaled)) return false;
    vk_batch_barrier_c2c(batch);

    if (!record_precompute_weights(plan, batch, M)) return false;

    if (plan->spread_cfg.use_smem && plan->spread_cfg.num_wins > 1) {
        if (!record_build_window_buckets(
                plan, batch, M, plan->spread_cfg.win_elems, plan->spread_cfg.num_wins,
                vk_buf_slice(&plan->d_bucket_counts, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_bucket_off, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_bucket_m, 0, VK_WHOLE_SIZE))) {
            return false;
        }
        vk_batch_barrier_c2c(batch);
        if (!record_pack_bucket_geom(
                plan, batch, plan->spread_cfg.num_wins, 2 * M,
                vk_buf_slice(&plan->d_bucket_off, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_bucket_m, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_b_first, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_b_shift, 0, VK_WHOLE_SIZE),
                vk_buf_slice(&plan->d_b_weights, 0, VK_WHOLE_SIZE))) {
            return false;
        }
    }
    vk_batch_barrier_c2c(batch);
    return true;
}

bool vk_nufft1_record_set_output_scale(VkNufft1Plan* plan, VkCmdBatch* batch, double scale) {
    if (!plan || !isfinite(scale) || scale == 0.0) return false;
    if (scale == plan->output_scale) return true;
    plan->output_scale = scale;
    bool rebuilt = false;
    if (plan->d_slot_deconv.buffer != VK_NULL_HANDLE) {
        if (!record_build_deconv_table(plan, batch, 1, false,
                vk_buf_slice(&plan->d_slot_deconv, 0, VK_WHOLE_SIZE), scale)) return false;
        rebuilt = true;
    }
    if (plan->d_nat_deconv.buffer != VK_NULL_HANDLE) {
        if (!record_build_deconv_table(plan, batch, 2, false,
                vk_buf_slice(&plan->d_nat_deconv, 0, VK_WHOLE_SIZE), scale)) return false;
        rebuilt = true;
    }
    if (plan->d_slot_deconv_f32.buffer != VK_NULL_HANDLE && plan->prec == VK_PREC_HALF) {
        if (!record_build_deconv_table(plan, batch, 1, true,
                vk_buf_slice(&plan->d_slot_deconv_f32, 0, VK_WHOLE_SIZE), scale)) return false;
        rebuilt = true;
    }
    if (rebuilt) vk_batch_barrier_c2c(batch);
    return true;
}

VkBufferSlice vk_nufft1_get_slot_phi_hat(VkNufft1Plan* plan, VkCmdBatch* batch) {
    VkBufferSlice empty = {0};
    if (!plan || !batch) return empty;
    if (plan->d_slot_phi_hat.buffer == VK_NULL_HANDLE) {
        if (!vk_buf_create(plan->ctx, (size_t)plan->nout * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_slot_phi_hat)) {
            return empty;
        }
        if (!record_build_deconv_table(plan, batch, 0, false,
                vk_buf_slice(&plan->d_slot_phi_hat, 0, VK_WHOLE_SIZE), 1.0)) {
            vk_buf_destroy(&plan->d_slot_phi_hat);
            return empty;
        }
        vk_batch_barrier_c2c(batch);
    }
    return vk_buf_slice(&plan->d_slot_phi_hat, 0, VK_WHOLE_SIZE);
}

VkBufferSlice vk_nufft1_get_slot_deconv(VkNufft1Plan* plan, VkCmdBatch* batch) {
    VkBufferSlice empty = {0};
    if (!plan || !batch) return empty;
    if (plan->d_slot_deconv.buffer == VK_NULL_HANDLE) {
        if (!vk_buf_create(plan->ctx, (size_t)plan->nout * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_slot_deconv)) {
            return empty;
        }
        if (!record_build_deconv_table(plan, batch, 1, false,
                vk_buf_slice(&plan->d_slot_deconv, 0, VK_WHOLE_SIZE), plan->output_scale)) {
            vk_buf_destroy(&plan->d_slot_deconv);
            return empty;
        }
        vk_batch_barrier_c2c(batch);
    }
    return vk_buf_slice(&plan->d_slot_deconv, 0, VK_WHOLE_SIZE);
}

VkBufferSlice vk_nufft1_get_nat_deconv(VkNufft1Plan* plan, VkCmdBatch* batch) {
    VkBufferSlice empty = {0};
    if (!plan || !batch) return empty;
    if (plan->d_nat_deconv.buffer == VK_NULL_HANDLE) {
        if (!vk_buf_create(plan->ctx, (size_t)plan->nout * plan->scalar_bytes, VK_MEM_DEVICE_LOCAL, &plan->d_nat_deconv)) {
            return empty;
        }
        if (!record_build_deconv_table(plan, batch, 2, false,
                vk_buf_slice(&plan->d_nat_deconv, 0, VK_WHOLE_SIZE), plan->output_scale)) {
            vk_buf_destroy(&plan->d_nat_deconv);
            return empty;
        }
        vk_batch_barrier_c2c(batch);
    }
    return vk_buf_slice(&plan->d_nat_deconv, 0, VK_WHOLE_SIZE);
}

VkBufferSlice vk_nufft1_get_slot_deconv_f32(VkNufft1Plan* plan, VkCmdBatch* batch) {
    VkBufferSlice empty = {0};
    if (!plan || !batch) return empty;
    if (plan->prec == VK_PREC_FLOAT) {
        return vk_nufft1_get_slot_deconv(plan, batch);
    }
    if (plan->prec != VK_PREC_HALF) return empty;
    if (plan->d_slot_deconv_f32.buffer == VK_NULL_HANDLE) {
        if (!vk_buf_create(plan->ctx, (size_t)plan->nout * sizeof(float), VK_MEM_DEVICE_LOCAL, &plan->d_slot_deconv_f32)) {
            return empty;
        }
        if (!record_build_deconv_table(plan, batch, 1, true,
                vk_buf_slice(&plan->d_slot_deconv_f32, 0, VK_WHOLE_SIZE), plan->output_scale)) {
            vk_buf_destroy(&plan->d_slot_deconv_f32);
            return empty;
        }
        vk_batch_barrier_c2c(batch);
    }
    return vk_buf_slice(&plan->d_slot_deconv_f32, 0, VK_WHOLE_SIZE);
}

bool vk_nufft1_record_execute(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    VkBufferSlice d_y,
    uint32_t y_offset_elems,
    VkBufferSlice d_out,
    uint32_t out_offset_elems,
    size_t bs,
    VkNufftOutOrder order) {
    if (!plan || !batch) return false;
    if (bs == 0) return true;

    const bool use_fused =
        (plan->fft_plan.exec_mode != VK_PLAN_MULTISTAGE_GLOBAL) &&
        (plan->launch_cfg.deconv_variant_override[0] == '\0') &&
        (order == VK_NUFFT_ORDER_SLOT_BIT_REVERSED || order == VK_NUFFT_ORDER_SLOT_BIT_REVERSED_RAW);
    const int compact_mode = (plan->mode == VK_PSWF43) ? 2 : 1;
    VkBufferSlice fused_deconv = {0};
    bool apply_deconv = false;
    if (use_fused && order == VK_NUFFT_ORDER_SLOT_BIT_REVERSED) {
        fused_deconv = vk_nufft1_get_slot_deconv(plan, batch);
        if (!vk_slice_valid(fused_deconv)) return false;
        apply_deconv = true;
    }

    const bool use_packed = plan->spread_cfg.use_smem && plan->spread_cfg.num_wins > 1;
    VkBufferSlice first_s = vk_buf_slice(use_packed ? &plan->d_b_first : &plan->d_first, 0, VK_WHOLE_SIZE);
    VkBufferSlice shift_s = vk_buf_slice(use_packed ? &plan->d_b_shift : &plan->d_shift, 0, VK_WHOLE_SIZE);
    VkBufferSlice w_s     = vk_buf_slice(use_packed ? &plan->d_b_weights : &plan->d_weights, 0, VK_WHOLE_SIZE);
    VkBufferSlice off_s   = use_packed ? vk_buf_slice(&plan->d_bucket_off, 0, VK_WHOLE_SIZE) : (VkBufferSlice){0};
    VkBufferSlice bm_s    = use_packed ? vk_buf_slice(&plan->d_bucket_m, 0, VK_WHOLE_SIZE) : (VkBufferSlice){0};
    VkBufferSlice fft_s   = vk_buf_slice(&plan->d_fft_buf, 0, VK_WHOLE_SIZE);

    for (size_t b_off = 0; b_off < bs; b_off += plan->chunk_bs) {
        if (b_off > 0) vk_batch_barrier_c2c(batch);
        size_t cur_bs = (bs - b_off < plan->chunk_bs) ? (bs - b_off) : plan->chunk_bs;
        uint32_t cur_y_off   = y_offset_elems + (uint32_t)(b_off * (size_t)plan->M);
        uint32_t cur_out_off = out_offset_elems + (uint32_t)(b_off * (size_t)plan->nout);

        if (!record_spread_cfg(
                plan, batch, &plan->spread_cfg, w_s, first_s, shift_s, off_s, bm_s,
                d_y, cur_y_off, fft_s, plan->M, cur_bs)) {
            return false;
        }
        vk_batch_barrier_c2c(batch);

        if (use_fused) {
            if (!vk_dif_record_execute_raw_compact(
                    &plan->fft_plan, batch, fft_s, d_out, cur_bs, 0u, cur_out_off,
                    compact_mode, fused_deconv, apply_deconv, plan->chunk_bs)) {
                return false;
            }
        } else {
            if (!vk_dif_record_execute_raw(
                    &plan->fft_plan, batch, fft_s, fft_s, cur_bs, 0u, 0u, plan->chunk_bs)) {
                return false;
            }
            vk_batch_barrier_c2c(batch);
            if (!record_deconv_emit(
                    plan, batch, &plan->deconv_cfg, fft_s, d_out, cur_out_off, cur_bs, order)) {
                return false;
            }
        }
    }
    return true;
}

void vk_nufft1_unpermute_slots_to_natural_host(
    const void* h_slots,
    void* h_nat,
    size_t elem_bytes,
    int N,
    VkPswfMode mode,
    size_t bs) {
    int nfft = next_pow2_i(2 * N);
    int log_nfft = log2_pow2_i(nfft);
    int nout = vk_nufft1_nout(N, mode);
    int output_shift = vk_nufft1_output_shift(N, mode);
    const uint8_t* src_all = (const uint8_t*)h_slots;
    uint8_t* dst_all = (uint8_t*)h_nat;

    for (size_t b = 0; b < bs; ++b) {
        const uint8_t* src = src_all + b * (size_t)nout * elem_bytes;
        uint8_t* dst = dst_all + b * (size_t)nout * elem_bytes;
        for (int i = 0; i < nout; ++i) {
            int slot_p = vk_nufft1_compact_idx_to_slot(i, mode);
            int k = vk_nufft1_slot_to_natural_idx(slot_p, log_nfft, nfft, output_shift);
            if (k >= 0 && k < nout) {
                memcpy(dst + (size_t)k * elem_bytes, src + (size_t)i * elem_bytes, elem_bytes);
            }
        }
    }
}
