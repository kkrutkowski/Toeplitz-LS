/* Plain C23 Vulkan Fused GLS (d=1), Levinson-Durbin (d>=2), and IHS Solver Recorders. */
#include "linalg/vk_levinson.h"
#include "common/math.h"

typedef struct VkFastChi2F32Push {
    float    ws_fft;
    float    inv_ws_fft;
    float    yws_fft;
    float    inv_chi2_ref;
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
    VkDf     inv_chi2_ref;
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
    double   inv_chi2_ref;
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
    float   inv_n_eff;
    float   inv_n2_288;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
} VkIhsF32Push;

typedef struct VkIhsDfPush {
    VkDf    base_scale;
    VkDf    inv_n_eff;
    VkDf    inv_n2_288;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
} VkIhsDfPush;

typedef struct VkIhsDPush {
    double  base_scale;
    double  inv_n_eff;
    double  inv_n2_288;
    int32_t degree, Nf, nfft, log_nfft, nout, output_shift, b_start, b_count, mode43, norm, first_harmonic;
    int32_t _pad;
} VkIhsDPush;

static_assert(sizeof(VkFastChi2F32Push) == 80);
static_assert(sizeof(VkFastChi2DfPush) == 120);
static_assert(sizeof(VkFastChi2DPush) == 120);
static_assert(sizeof(VkIhsF32Push) == 56);
static_assert(sizeof(VkIhsDfPush) == 68);
static_assert(sizeof(VkIhsDPush) == 72);

static size_t solve_real_bytes(VkPrecKind prec) {
    return (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT) ? 4u : 8u;
}

static bool has_unrolled_fc2(VkPrecKind prec, int degree) {
    if (degree <= 5) return true;
    return (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT) && (degree <= 8);
}

bool vk_fastchi2_uses_smem(const VkDeviceFftConfig* dev_cfg, VkPrecKind prec, int degree, uint32_t* out_wg) {
    if (!dev_cfg || has_unrolled_fc2(prec, degree)) return false;
    if (!dev_cfg->has_native_local_mem || dev_cfg->local_mem_size < 16384u) return false;
    uint32_t wg = 64u;
    if (wg > (uint32_t)dev_cfg->max_work_group_size) wg = (uint32_t)dev_cfg->max_work_group_size;
    if (wg < 32u) return false;
    size_t slots = (size_t)(2 * degree - 1);
    size_t smem_bytes = 2u * slots * (size_t)wg * solve_real_bytes(prec);
    /* Require >= 4 resident workgroups per CU so SMEM never collapses occupancy */
    if (smem_bytes <= (dev_cfg->local_mem_size / 4u)) {
        if (out_wg) *out_wg = wg;
        return true;
    }
    return false;
}

bool vk_fastchi2_needs_scratch(const VkDeviceFftConfig* dev_cfg, VkPrecKind prec, int degree) {
    if (has_unrolled_fc2(prec, degree)) return false;
    return !vk_fastchi2_uses_smem(dev_cfg, prec, degree, nullptr);
}

static VkShaderId pick_fc2_shader(
    const VkDeviceFftConfig* dev_cfg,
    VkPrecKind prec,
    int degree,
    bool* uses_smem,
    bool* needs_scratch,
    uint32_t* smem_wg) {
    *uses_smem = false;
    *needs_scratch = false;
    if (smem_wg) *smem_wg = 0u;

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
    } else if (degree == 6 && (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT)) {
        return (prec == VK_PREC_HALF) ? VK_SHADER_fastchi2_lev_d6_h : VK_SHADER_fastchi2_lev_d6_f;
    } else if (degree == 7 && (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT)) {
        return (prec == VK_PREC_HALF) ? VK_SHADER_fastchi2_lev_d7_h : VK_SHADER_fastchi2_lev_d7_f;
    } else if (degree == 8 && (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT)) {
        return (prec == VK_PREC_HALF) ? VK_SHADER_fastchi2_lev_d8_h : VK_SHADER_fastchi2_lev_d8_f;
    } else if (vk_fastchi2_uses_smem(dev_cfg, prec, degree, smem_wg)) {
        *uses_smem = true;
        switch (prec) {
            case VK_PREC_HALF:   return VK_SHADER_fastchi2_lev_smem_h;
            case VK_PREC_FLOAT:  return VK_SHADER_fastchi2_lev_smem_f;
            case VK_PREC_DF:     return VK_SHADER_fastchi2_lev_smem_df;
            case VK_PREC_DOUBLE: return VK_SHADER_fastchi2_lev_smem_d;
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
    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(ctx);
    bool uses_smem = false;
    bool needs_scratch = false;
    uint32_t smem_wg = 0u;
    VkShaderId sid = pick_fc2_shader(dev_cfg, prec, p->degree, &uses_smem, &needs_scratch, &smem_wg);
    if (sid == VK_SHADER_COUNT) return false;

    int ipb = (prec == VK_PREC_HALF) ? (p->nout >> 1) : p->nout;
    uint32_t wg = (uint32_t)(p->wg_size > 0 ? p->wg_size : 64u);
    if (uses_smem && smem_wg > 0u) {
        wg = smem_wg;
    } else if (needs_scratch && (prec == VK_PREC_DF || prec == VK_PREC_DOUBLE) && wg > 64u) {
        /* Halves per-WG L1D scratch footprint while maintaining full warp occupancy */
        wg = 64u;
    }
    if (wg > 256u) wg = 256u;

    const bool use_2d = ((uint32_t)p->b_count <= dev_cfg->max_wg_count_xyz[1]);
    uint32_t gx = use_2d
        ? (((uint32_t)ipb + wg - 1u) / wg)
        : (((uint32_t)p->b_count * (uint32_t)ipb + wg - 1u) / wg);
    uint32_t gy = use_2d ? (uint32_t)p->b_count : 1u;

    uint32_t mode43 = (p->mode == VK_PSWF43) ? 1u : 0u;
    uint32_t flags = mode43 |
                     (((uint32_t)p->norm & 7u) << 1) |
                     (has_cond ? (1u << 4) : 0u) |
                     (use_raw_deconv ? (1u << 5) : 0u);

    uint32_t req_sg = 0u;
    VkSpecConstEntry specs[10];
    size_t nspec = 0;
    specs[nspec++] = (VkSpecConstEntry){0u, wg};
    specs[nspec++] = (VkSpecConstEntry){1u, use_2d ? 0u : 1u};
    specs[nspec++] = (VkSpecConstEntry){2u, (uint32_t)p->norm & 7u};
    specs[nspec++] = (VkSpecConstEntry){3u, has_cond ? 1u : 0u};
    specs[nspec++] = (VkSpecConstEntry){4u, use_raw_deconv ? 1u : 0u};
    specs[nspec++] = (VkSpecConstEntry){5u, mode43};
    if (uses_smem) {
        uint32_t smem_elems = wg * (uint32_t)(2 * p->degree - 1);
        specs[nspec++] = (VkSpecConstEntry){6u, smem_elems};
    } else if (needs_scratch) {
        uint32_t tile_lanes = 32u;
        uint32_t log2_tile = 5u;
        if (dev_cfg->has_subgroup_size_control && vk_device_supports_sg_size(dev_cfg, 32u)) {
            req_sg = 32u;
            tile_lanes = 32u;
            log2_tile = 5u;
        } else if (dev_cfg->sub_group_size >= 64u) {
            tile_lanes = 64u;
            log2_tile = 6u;
        }
        specs[nspec++] = (VkSpecConstEntry){6u, tile_lanes};
        specs[nspec++] = (VkSpecConstEntry){7u, log2_tile};
    }
    if (prec == VK_PREC_DF) {
        specs[nspec++] = (VkSpecConstEntry){15u, df_use_veltkamp ? 1u : 0u};
    }

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
    const double inv_chi2_ref = 1.0 / p->chi2_ref;
    if (prec == VK_PREC_DOUBLE) {
        VkFastChi2DPush pc = {
            p->ws_fft, inv_ws_fft, p->yws_fft, inv_chi2_ref, p->b_param,
            p->cyb.sum_N_eff, p->cyb.inv_2d, p->cyb.k_eff, p->cyb.theta_eff, p->cyb.lgamma_k,
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, req_sg);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, gy, 1u);
    } else if (prec == VK_PREC_DF) {
        VkFastChi2DfPush pc = {
            vk_df_from_double(p->ws_fft), vk_df_from_double(inv_ws_fft),
            vk_df_from_double(p->yws_fft),
            vk_df_from_double(inv_chi2_ref), vk_df_from_double(p->b_param),
            vk_df_from_double(p->cyb.sum_N_eff), vk_df_from_double(p->cyb.inv_2d),
            vk_df_from_double(p->cyb.k_eff), vk_df_from_double(p->cyb.theta_eff),
            vk_df_from_double(p->cyb.lgamma_k),
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, req_sg);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, gy, 1u);
    } else {
        VkFastChi2F32Push pc = {
            (float)p->ws_fft, (float)inv_ws_fft, (float)p->yws_fft, (float)inv_chi2_ref, (float)p->b_param,
            (float)p->cyb.sum_N_eff, (float)p->cyb.inv_2d, (float)p->cyb.k_eff,
            (float)p->cyb.theta_eff, (float)p->cyb.lgamma_k,
            p->degree, p->M, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, flags,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, nbinds, sizeof(pc), specs, nspec, req_sg);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, nbinds, &pc, sizeof(pc), gx, gy, 1u);
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
    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(ctx);
    uint32_t wg = (uint32_t)(p->wg_size > 0 ? p->wg_size : 64u);
    if (wg > 256u) wg = 256u;

    const bool use_2d = ((uint32_t)p->b_count <= dev_cfg->max_wg_count_xyz[1]);
    uint32_t gx = use_2d
        ? (((uint32_t)p->nout + wg - 1u) / wg)
        : (((uint32_t)p->b_count * (uint32_t)p->nout + wg - 1u) / wg);
    uint32_t gy = use_2d ? (uint32_t)p->b_count : 1u;

    uint32_t mode43 = (p->mode == VK_PSWF43) ? 1u : 0u;
    VkSpecConstEntry specs[6];
    size_t nspec = 0;
    specs[nspec++] = (VkSpecConstEntry){0u, wg};
    specs[nspec++] = (VkSpecConstEntry){1u, use_2d ? 0u : 1u};
    specs[nspec++] = (VkSpecConstEntry){2u, (uint32_t)p->norm};
    specs[nspec++] = (VkSpecConstEntry){3u, mode43};
    specs[nspec++] = (VkSpecConstEntry){4u, (p->degree > 1) ? 1u : 0u};
    if (prec == VK_PREC_DF) {
        specs[nspec++] = (VkSpecConstEntry){15u, df_use_veltkamp ? 1u : 0u};
    }

    VkBufferSlice dummy = vk_ctx_dummy_buffer(ctx);
    VkBufferSlice binds[3] = {
        vk_slice_valid(b0) ? b0 : dummy,
        vk_slice_valid(b1) ? b1 : dummy,
        vk_slice_valid(b2) ? b2 : dummy,
    };

    const double inv_n_eff = (p->n_eff > 0.0 && isfinite(p->n_eff)) ? (1.0 / p->n_eff) : 0.0;
    const double inv_n2_288 = (inv_n_eff * inv_n_eff) / 288.0;

    if (prec == VK_PREC_DOUBLE) {
        VkIhsDPush pc = {
            base_scale_d, inv_n_eff, inv_n2_288,
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, (int32_t)mode43, (int32_t)p->norm, first_harmonic ? 1 : 0, 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, gy, 1u);
    } else if (prec == VK_PREC_DF) {
        VkIhsDfPush pc = {
            vk_df_from_double(base_scale_d),
            vk_df_from_double(inv_n_eff),
            vk_df_from_double(inv_n2_288),
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, (int32_t)mode43, (int32_t)p->norm, first_harmonic ? 1 : 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, gy, 1u);
    } else {
        VkIhsF32Push pc = {
            (float)base_scale_d, (float)inv_n_eff, (float)inv_n2_288,
            p->degree, p->Nf, p->nfft, p->log_nfft, p->nout, p->output_shift,
            p->b_start, p->b_count, (int32_t)mode43, (int32_t)p->norm, first_harmonic ? 1 : 0,
        };
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, sid, 3u, sizeof(pc), specs, nspec, 0u);
        if (!pipe) return false;
        return vk_batch_dispatch(batch, pipe, binds, 3u, &pc, sizeof(pc), gx, gy, 1u);
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
