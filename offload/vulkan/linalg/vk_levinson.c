/* Plain C23 Vulkan Fused GLS (d=1), Levinson-Durbin (d>=2), and IHS Solver Recorders. */
#include "linalg/vk_levinson.h"
#include "common/math.h"

typedef struct VkFastChi2F32Push {
    float    ws_fft;
    float    inv_ws_fft;
    float    yws_fft;
    float    chi2_ref;
    float    b_param;
    float    cyb_sum_N_eff;
    float    cyb_inv_2d;
    float    cyb_k_eff;
    float    cyb_theta_eff;
    float    cyb_lgamma_k;
    int32_t  degree, M, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count;
    uint32_t flags;
} VkFastChi2F32Push;

typedef struct VkFastChi2DfPush {
    VkDf     ws_fft;
    VkDf     inv_ws_fft;
    VkDf     yws_fft;
    VkDf     chi2_ref;
    VkDf     b_param;
    VkDf     cyb_sum_N_eff;
    VkDf     cyb_inv_2d;
    VkDf     cyb_k_eff;
    VkDf     cyb_theta_eff;
    VkDf     cyb_lgamma_k;
    int32_t  degree, M, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count;
    uint32_t flags;
} VkFastChi2DfPush;

typedef struct VkFastChi2DPush {
    double   ws_fft;
    double   inv_ws_fft;
    double   yws_fft;
    double   chi2_ref;
    double   b_param;
    double   cyb_sum_N_eff;
    double   cyb_inv_2d;
    double   cyb_k_eff;
    double   cyb_theta_eff;
    double   cyb_lgamma_k;
    int32_t  degree, M, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count;
    uint32_t flags;
} VkFastChi2DPush;

typedef struct VkIhsF32Push {
    float   base_scale;
    float   n_eff;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
} VkIhsF32Push;

typedef struct VkIhsDfPush {
    VkDf    base_scale;
    VkDf    n_eff;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
} VkIhsDfPush;

typedef struct VkIhsDPush {
    double  base_scale;
    double  n_eff;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
    int32_t _pad;
} VkIhsDPush;

static_assert(sizeof(VkFastChi2F32Push) == 80);
static_assert(sizeof(VkFastChi2DfPush) == 120);
static_assert(sizeof(VkFastChi2DPush) == 120);
static_assert(sizeof(VkIhsF32Push) == 52);
static_assert(sizeof(VkIhsDfPush) == 60);
static_assert(sizeof(VkIhsDPush) == 64);

static VkShaderId pick_fc2_shader(VkPrecKind prec, int degree, bool* needs_scratch) {
    *needs_scratch = false;
    if (degree == 1) {
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_gls_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_gls_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_gls_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_gls_d;
        }
    } else if (degree == 2) {
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_d2_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_d2_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_d2_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_d2_d;
        }
    } else if (degree == 3) {
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_d3_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_d3_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_d3_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_d3_d;
        }
    } else if (degree == 4) {
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_d4_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_d4_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_d4_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_d4_d;
        }
    } else if (degree == 5) {
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_d5_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_d5_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_d5_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_d5_d;
        }
    } else {
        *needs_scratch = true;
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_large_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_large_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_large_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_large_d;
        }
    }
    return VK_SHADER_COUNT;
}

bool vk_record_fastchi2_solve(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkBufferSlice d_trig,
    VkBufferSlice d_slot_phi_hat,
    VkBufferSlice d_slot_deconv,
    VkBufferSlice d_a_scratch,
    VkBufferSlice d_power,
    VkBufferSlice d_cond,
    bool has_cond,
    bool use_raw_deconv,
    const VkFastChi2SolveParams* p) {
    bool needs_scratch = false;
    VkShaderId sid = pick_fc2_shader(prec, p->degree, &needs_scratch);
    if (sid == VK_SHADER_COUNT) return false;

    int ipb = (prec == VK_PREC_HALF) ? (p->nout >> 1) : p->nout;
    uint32_t total_items = (uint32_t)p->b_count * (uint32_t)ipb;
    uint32_t wg = (uint32_t)(p->wg_size > 0 ? p->wg_size : 64u);
    if (wg > 256u) wg = 256u;
    uint32_t gx = (total_items + wg - 1u) / wg;

    uint32_t flags = ((p->mode == VK_PSWF43) ? 1u : 0u) |
                     (((uint32_t)p->norm & 7u) << 1) |
                     (has_cond ? (1u << 4) : 0u) |
                     (use_raw_deconv ? (1u << 5) : 0u);

    VkSpecConstEntry specs[2] = {
        {0u, wg},
        {15u, df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (prec == VK_PREC_DF) ? 2u : 1u;

    VkBufferSlice dummy = vk_ctx_dummy_buffer(ctx);
    VkBufferSlice binds[6] = {
        d_trig,
        (use_raw_deconv && vk_slice_valid(d_slot_phi_hat)) ? d_slot_phi_hat : dummy,
        (use_raw_deconv && vk_slice_valid(d_slot_deconv))  ? d_slot_deconv  : dummy,
        d_power,
        (has_cond && vk_slice_valid(d_cond)) ? d_cond : dummy,
        (needs_scratch && vk_slice_valid(d_a_scratch)) ? d_a_scratch : dummy,
    };
    uint32_t nbinds = needs_scratch ? 6u : 5u;

    const double inv_ws_fft = 1.0 / p->ws_fft;
    if (prec == VK_PREC_DOUBLE) {
        VkFastChi2DPush pc = {
            p->ws_fft, inv_ws_fft, p->yws_fft, p->chi2_ref, p->b_param,
            p->cyb.sum_N_eff, p->cyb.inv_2d, p->cyb.k_eff, p->cyb.theta_eff, p->cyb.lgamma_k,
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, 1u, 1u);
    } else if (prec == VK_PREC_DF) {
        VkFastChi2DfPush pc = {
            vk_df_from_double(p->ws_fft), vk_df_from_double(inv_ws_fft),
            vk_df_from_double(p->yws_fft),
            vk_df_from_double(p->chi2_ref), vk_df_from_double(p->b_param),
            vk_df_from_double(p->cyb.sum_N_eff), vk_df_from_double(p->cyb.inv_2d),
            vk_df_from_double(p->cyb.k_eff), vk_df_from_double(p->cyb.theta_eff),
            vk_df_from_double(p->cyb.lgamma_k),
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, 1u, 1u);
    } else {
        VkFastChi2F32Push pc = {
            (float)p->ws_fft, (float)inv_ws_fft, (float)p->yws_fft, (float)p->chi2_ref, (float)p->b_param,
            (float)p->cyb.sum_N_eff, (float)p->cyb.inv_2d, (float)p->cyb.k_eff,
            (float)p->cyb.theta_eff, (float)p->cyb.lgamma_k,
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, 1u, 1u);
    }
}

static bool record_ihs_common(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkShaderId sid,
    VkBufferSlice b0,
    VkBufferSlice b1,
    VkBufferSlice b2,
    double base_scale_d,
    bool first_harmonic,
    const VkIhsSolveParams* p) {
    uint32_t total_bins = (uint32_t)p->b_count * (uint32_t)p->nout;
    uint32_t wg = (uint32_t)(p->wg_size > 0 ? p->wg_size : 64u);
    if (wg > 256u) wg = 256u;
    uint32_t gx = (total_bins + wg - 1u) / wg;

    VkSpecConstEntry specs[2] = {
        {0u, wg},
        {15u, df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (prec == VK_PREC_DF) ? 2u : 1u;

    VkBufferSlice dummy = vk_ctx_dummy_buffer(ctx);
    VkBufferSlice binds[3] = {
        vk_slice_valid(b0) ? b0 : dummy,
        vk_slice_valid(b1) ? b1 : dummy,
        vk_slice_valid(b2) ? b2 : dummy,
    };
    int mode43 = (p->mode == VK_PSWF43) ? 1 : 0;

    if (prec == VK_PREC_DOUBLE) {
        VkIhsDPush pc = {
            base_scale_d, p->n_eff,
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, mode43, (int32_t)p->norm, first_harmonic ? 1 : 0, 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, 1u, 1u);
    } else if (prec == VK_PREC_DF) {
        VkIhsDfPush pc = {
            vk_df_from_double(base_scale_d), vk_df_from_double(p->n_eff),
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, mode43, (int32_t)p->norm, first_harmonic ? 1 : 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, 1u, 1u);
    } else {
        VkIhsF32Push pc = {
            (float)base_scale_d, (float)p->n_eff,
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, mode43, (int32_t)p->norm, first_harmonic ? 1 : 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, 1u, 1u);
    }
}

bool vk_record_ihs_single(
    VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, bool df_use_veltkamp,
    VkBufferSlice d_z, VkBufferSlice d_slot_deconv, VkBufferSlice d_power,
    const VkIhsSolveParams* p) {
    VkShaderId sid = VK_SHADER_COUNT;
    switch (prec) {
        case VK_PREC_HALF:   sid = VK_SHADER_ihs_single_h;  break;
        case VK_PREC_FLOAT:  sid = VK_SHADER_ihs_single_f;  break;
        case VK_PREC_DF:     sid = VK_SHADER_ihs_single_df; break;
        case VK_PREC_DOUBLE: sid = VK_SHADER_ihs_single_d;  break;
    }
    double scale = (p->norm == VK_NORM_STANDARD) ? p->standard_scale : p->nll_scale;
    return record_ihs_common(ctx, batch, prec, df_use_veltkamp, sid, d_z, d_slot_deconv, d_power, scale, true, p);
}

bool vk_record_ihs_accum(
    VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, bool df_use_veltkamp,
    VkBufferSlice d_z, VkBufferSlice d_slot_deconv, VkBufferSlice d_acc,
    bool first_harmonic, const VkIhsSolveParams* p) {
    VkShaderId sid = VK_SHADER_COUNT;
    switch (prec) {
        case VK_PREC_HALF:   sid = VK_SHADER_ihs_accum_h;  break;
        case VK_PREC_FLOAT:  sid = VK_SHADER_ihs_accum_f;  break;
        case VK_PREC_DF:     sid = VK_SHADER_ihs_accum_df; break;
        case VK_PREC_DOUBLE: sid = VK_SHADER_ihs_accum_d;  break;
    }
    return record_ihs_common(ctx, batch, prec, df_use_veltkamp, sid, d_z, d_slot_deconv, d_acc, p->nll_scale, first_harmonic, p);
}

bool vk_record_ihs_finalize(
    VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, bool df_use_veltkamp,
    VkBufferSlice d_acc, VkBufferSlice d_slot_deconv, VkBufferSlice d_power,
    const VkIhsSolveParams* p) {
    VkShaderId sid = VK_SHADER_COUNT;
    switch (prec) {
        case VK_PREC_HALF:   sid = VK_SHADER_ihs_finalize_h;  break;
        case VK_PREC_FLOAT:  sid = VK_SHADER_ihs_finalize_f;  break;
        case VK_PREC_DF:     sid = VK_SHADER_ihs_finalize_df; break;
        case VK_PREC_DOUBLE: sid = VK_SHADER_ihs_finalize_d;  break;
    }
    double scale = (p->norm == VK_NORM_STANDARD) ? p->standard_scale : p->nll_scale;
    return record_ihs_common(ctx, batch, prec, df_use_veltkamp, sid, d_acc, d_slot_deconv, d_power, scale, false, p);
}
