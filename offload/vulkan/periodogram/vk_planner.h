#ifndef TLS_VK_PLANNER_H
#define TLS_VK_PLANNER_H

/* Plain C23 Hardware-Generalized GPU/CPU Block-Size & Super-Chunk Planner.
 * Zero GPU-model-specific constants; strictly enforces per-stream memory budget
 * of 75% of max device memory / num_streams and safe_max_alloc_bytes(). */

#include "common/vk_device.h"
#include "nufft/vk_nufft1.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct VkGpuPlanChoice {
    int        N_block;
    int        nfft;
    int        nout;
    int        w;
    int        B_total;
    int        B_chunk;
    size_t     plan_max_bs;
    size_t     slot_budget_bytes;
    size_t     required_min_bytes;
    bool       fits_budget;
    VkPswfMode mode;
} VkGpuPlanChoice;

static inline int vk_kernel_width_for(VkPrecKind prec, VkPswfMode mode) {
    bool m43 = (mode == VK_PSWF43);
    switch (prec) {
        case VK_PREC_HALF:   return m43 ? 5  : 4;
        case VK_PREC_FLOAT:  return m43 ? 9  : 8;
        case VK_PREC_DF:     return m43 ? 16 : 14;
        case VK_PREC_DOUBLE: return m43 ? 18 : 16;
    }
    return 9;
}

static inline size_t vk_complex_bytes_for(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return 4u;
        case VK_PREC_FLOAT:  return 8u;
        case VK_PREC_DF:     return 16u;
        case VK_PREC_DOUBLE: return 16u;
    }
    return 8u;
}

static inline size_t vk_scalar_bytes_for(VkPrecKind prec) {
    switch (prec) {
        case VK_PREC_HALF:   return 2u;
        case VK_PREC_FLOAT:  return 4u;
        case VK_PREC_DF:     return 8u;
        case VK_PREC_DOUBLE: return 8u;
    }
    return 4u;
}

static inline size_t vk_dev_out_bytes_for(VkPrecKind prec) {
    return (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT) ? 4u : 8u;
}

static inline size_t vk_solve_real_bytes_for(VkPrecKind prec) {
    return (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT) ? 4u : 8u;
}

static inline size_t vk_pow2_floor_sz(size_t v) {
    if (v == 0u) return 1u;
    size_t p = 1u;
    while ((p << 1) <= v && (p << 1) > p) p <<= 1;
    return p;
}

static inline int vk_env_int_or(const char* name, int fallback) {
    const char* v = getenv(name);
    if (v && v[0] != '\0') {
        int parsed = atoi(v);
        if (parsed >= 0) return parsed;
    }
    return fallback;
}

typedef struct VkChunkMemoryFit {
    bool   fits;
    int    B_chunk;
    size_t plan_max_bs;
    size_t fixed_plus_plan;
    size_t bytes_per_blk;
} VkChunkMemoryFit;

static inline VkChunkMemoryFit vk_evaluate_chunk_memory(
    int Nf, int M, int degree, bool is_ihs, bool has_cond,
    VkPswfMode mode, VkPrecKind prec, int N_block,
    const VkDeviceFftConfig* dev_cfg, size_t slot_budget) {
    VkChunkMemoryFit fit = {0};
    int nfft = 2 * N_block;
    int nout = vk_nufft1_nout(N_block, mode);
    int w = vk_kernel_width_for(prec, mode);
    int B_total = (Nf + nout - 1) / nout;
    if (B_total < 1) B_total = 1;

    size_t cbytes = vk_complex_bytes_for(prec);
    size_t sbytes = vk_scalar_bytes_for(prec);
    size_t xbytes = 8u;
    size_t obytes = vk_dev_out_bytes_for(prec);
    size_t max_alloc = vk_device_safe_max_alloc_bytes(dev_cfg);

    size_t x_al = ((size_t)M * xbytes + 63u) & ~(size_t)63u;
    size_t w_al = ((size_t)M * sbytes + 63u) & ~(size_t)63u;
    size_t in_bytes = is_ihs ? (x_al + (size_t)M * sbytes) : (x_al + w_al + (size_t)M * sbytes);
    size_t out_bytes = (size_t)Nf * obytes;
    size_t cond_bytes = (!is_ihs && has_cond) ? out_bytes : 0u;

    size_t per_pt = 4u + sbytes + cbytes + (size_t)w * sbytes;
    size_t plan_pt_bytes = (size_t)M * per_pt * 3u + (size_t)nout * (sbytes * 2u + 4u);

    size_t fft_elem_bytes = (size_t)nfft * cbytes;
    size_t eff_l2 = vk_device_effective_l2_bytes(dev_cfg);
    size_t a_b = (2u * fft_elem_bytes > eff_l2) ? (2u * fft_elem_bytes) : eff_l2;
    size_t s_b = (2u * fft_elem_bytes > slot_budget / 6u) ? (2u * fft_elem_bytes) : (slot_budget / 6u);
    if (s_b > max_alloc / 2u) s_b = max_alloc / 2u;
    size_t plan_fft_budget = (a_b < s_b) ? a_b : s_b;
    size_t plan_bs = plan_fft_budget / (2u * fft_elem_bytes > 0u ? 2u * fft_elem_bytes : 1u);
    if (plan_bs < 2u) plan_bs = 2u;
    if (plan_bs > 512u) plan_bs = 512u;
    if (prec == VK_PREC_HALF && (plan_bs & 1u)) ++plan_bs;
    size_t plan_fft_bytes = 2u * plan_bs * fft_elem_bytes;

    size_t fixed_total = in_bytes + out_bytes + cond_bytes + plan_pt_bytes + plan_fft_bytes;
    fit.plan_max_bs = plan_bs;
    fit.fixed_plus_plan = fixed_total;

    size_t src_per_blk = (is_ihs ? 1u : 2u) * (size_t)M * cbytes;
    size_t trig_per_blk = (is_ihs ? 1u : (size_t)(3 * degree)) * (size_t)nout * cbytes;
    size_t acc_per_blk = (is_ihs && degree > 1) ? ((size_t)nout * obytes) : 0u;
    size_t scr_per_blk = 0u;
    if (!is_ihs && degree > 5) {
        size_t ipb = (prec == VK_PREC_HALF) ? (size_t)(nout >> 1) : (size_t)nout;
        size_t ipb_pad = (ipb + 31u) & ~(size_t)31u;
        scr_per_blk = 2u * (size_t)(2 * degree - 1) * ipb_pad * vk_solve_real_bytes_for(prec);
    }

    size_t per_blk_total = src_per_blk + trig_per_blk + acc_per_blk + scr_per_blk;
    size_t max_single = src_per_blk;
    if (trig_per_blk > max_single) max_single = trig_per_blk;
    if (acc_per_blk > max_single) max_single = acc_per_blk;
    if (scr_per_blk > max_single) max_single = scr_per_blk;
    fit.bytes_per_blk = per_blk_total;

    if (in_bytes > max_alloc || out_bytes > max_alloc ||
        fixed_total >= slot_budget || max_single > max_alloc) {
        fit.fits = false;
        return fit;
    }

    size_t avail = slot_budget - fixed_total;
    if (per_blk_total > avail) {
        fit.fits = false;
        return fit;
    }

    size_t b_chunk = (size_t)B_total;
    size_t b_by_bud = avail / (per_blk_total > 0u ? per_blk_total : 1u);
    size_t b_by_all = max_alloc / (max_single > 0u ? max_single : 1u);
    if (b_by_bud < b_chunk) b_chunk = b_by_bud;
    if (b_by_all < b_chunk) b_chunk = b_by_all;
    if (b_chunk == 0u) {
        fit.fits = false;
        return fit;
    }
    if (prec == VK_PREC_HALF && b_chunk < (size_t)B_total && b_chunk > 1u && (b_chunk & 1u)) {
        --b_chunk;
    }
    fit.B_chunk = (int)(b_chunk > 0u ? b_chunk : 1u);
    fit.fits = true;
    return fit;
}

static inline double vk_eval_gpu_block_cost(
    int Nf, int M, int degree, bool is_ihs,
    VkPswfMode mode, VkPrecKind prec, int N_block,
    const VkDeviceFftConfig* dev_cfg, int num_slots, double fp64_to_fp32_ratio) {
    int nout = vk_nufft1_nout(N_block, mode);
    int nfft = 2 * N_block;
    int log_nfft = 0;
    while ((1 << log_nfft) < nfft) ++log_nfft;
    int w = vk_kernel_width_for(prec, mode);
    double B = ceil((double)Nf / (double)nout);
    double n_exec = is_ihs ? (double)degree : (double)(3 * degree);
    double n_factors = is_ihs ? (double)degree : (double)(2 * degree);

    size_t cbytes = vk_complex_bytes_for(prec);
    size_t smem_elem_bytes = (prec == VK_PREC_HALF) ? 8u : cbytes;
    double prec_bytes = (double)cbytes;

    size_t smem_usable = dev_cfg->local_mem_size - (dev_cfg->local_mem_size / 16u);
    size_t sp_nfft = 2048u;
    if (dev_cfg->has_native_local_mem && smem_usable >= 4096u) {
        sp_nfft = vk_pow2_floor_sz(smem_usable / smem_elem_bytes) * 2u;
        if (sp_nfft < 1024u) sp_nfft = 1024u;
        if (sp_nfft > 8192u) sp_nfft = 8192u;
    }

    double fp64_alu_mult = 1.0;
    if (prec == VK_PREC_DOUBLE && fp64_to_fp32_ratio > 0.0 && fp64_to_fp32_ratio < 0.45) {
        fp64_alu_mult = sqrt(0.5 / fp64_to_fp32_ratio);
        if (fp64_alu_mult < 1.0) fp64_alu_mult = 1.0;
        if (fp64_alu_mult > 3.0) fp64_alu_mult = 3.0;
    }

    double fft_pass_mult = 1.0;
    if ((size_t)nfft > sp_nfft) fft_pass_mult = 2.25;
    else if ((size_t)nfft == sp_nfft && sp_nfft >= 4096u) fft_pass_mult = 1.25;
    double fft_cost = n_exec * B * (double)nfft * (0.35 + 0.065 * (double)log_nfft) * fft_pass_mult * (prec_bytes / 8.0);

    size_t win_cap = (size_t)nfft;
    if (dev_cfg->has_native_local_mem && smem_usable >= 4096u) {
        win_cap = vk_pow2_floor_sz(smem_usable / (4u * smem_elem_bytes));
        if (win_cap < 128u) win_cap = 128u;
        if (win_cap > 4096u) win_cap = 4096u;
    }
    double num_wins = (double)nfft / (double)win_cap;
    if (num_wins < 1.0) num_wins = 1.0;
    double win_pen = (num_wins <= 1.0) ? 1.0 : (1.0 + 0.12 * log2(num_wins));
    double spread_cost = n_exec * B * (double)M * (double)w * (0.55 * fp64_alu_mult) * win_pen;

    double phase_beta = (prec == VK_PREC_DF) ? 2.2 : ((prec == VK_PREC_DOUBLE) ? (1.2 * fp64_alu_mult) : 0.75);
    double phase_cost = n_factors * B * (double)M * phase_beta;
    double pre_cost = n_factors * (double)M * (2.0 + 0.5 * num_wins);

    double cus_per_slot = (double)dev_cfg->max_compute_units / (double)(num_slots > 0 ? num_slots : 1);
    if (cus_per_slot < 2.0) cus_per_slot = 2.0;
    double eff_bs = B * (is_ihs ? 1.0 : 1.5);
    double occ_pen = 1.0;
    if (dev_cfg->is_gpu && eff_bs < cus_per_slot) {
        occ_pen = 1.0 + 0.30 * ((cus_per_slot - eff_bs) / cus_per_slot);
    }

    double launch_overhead = n_exec * 400.0;
    return (fft_cost + spread_cost + phase_cost + launch_overhead) * occ_pen + pre_cost;
}

static inline VkGpuPlanChoice vk_choose_gpu_periodogram_plan(
    int Nf, int M, int degree, bool is_ihs, bool has_cond,
    VkPswfMode mode, VkPrecKind prec,
    const VkDeviceFftConfig* dev_cfg, int num_slots, double fp64_to_fp32_ratio) {
    VkGpuPlanChoice out = {0};
    out.mode = mode;
    out.w = vk_kernel_width_for(prec, mode);
    out.slot_budget_bytes = vk_device_per_stream_mem_budget(dev_cfg, (size_t)(num_slots > 0 ? num_slots : 1));

    int forced = vk_env_int_or("TLS_VK_BLOCK_SIZE", 0);
    int best_block = 512;
    VkChunkMemoryFit best_fit = {0};
    bool found_any = false;

    if (forced >= 512 && forced <= 131072 && (forced & (forced - 1)) == 0) {
        best_block = forced;
        best_fit = vk_evaluate_chunk_memory(Nf, M, degree, is_ihs, has_cond, mode, prec, best_block, dev_cfg, out.slot_budget_bytes);
        found_any = best_fit.fits;
    } else {
        int max_cand = 512;
        while (max_cand < 131072 && vk_nufft1_nout(max_cand, mode) < Nf) max_cand <<= 1;
        double best_cost = 0.0;
        for (int cand = 512; cand <= max_cand; cand <<= 1) {
            VkChunkMemoryFit fit = vk_evaluate_chunk_memory(Nf, M, degree, is_ihs, has_cond, mode, prec, cand, dev_cfg, out.slot_budget_bytes);
            if (!fit.fits) {
                if (!found_any) { best_block = cand; best_fit = fit; }
                continue;
            }
            double c = vk_eval_gpu_block_cost(Nf, M, degree, is_ihs, mode, prec, cand, dev_cfg, num_slots, fp64_to_fp32_ratio);
            if (!found_any || c < best_cost) {
                found_any = true;
                best_cost = c;
                best_block = cand;
                best_fit = fit;
            }
        }
    }

    out.N_block = best_block;
    out.nfft = 2 * best_block;
    out.nout = vk_nufft1_nout(best_block, mode);
    out.B_total = (Nf + out.nout - 1) / out.nout;
    if (out.B_total < 1) out.B_total = 1;
    out.B_chunk = best_fit.B_chunk > 0 ? best_fit.B_chunk : 1;
    out.plan_max_bs = best_fit.plan_max_bs > 0 ? best_fit.plan_max_bs : 2u;
    out.required_min_bytes = best_fit.fixed_plus_plan + best_fit.bytes_per_blk;
    out.fits_budget = found_any;
    return out;
}

#endif /* TLS_VK_PLANNER_H */
