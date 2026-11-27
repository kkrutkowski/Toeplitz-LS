#pragma once

// Hardware-Generalized GPU/CPU Block-Size & Super-Chunk Memory/Time Planner.
// Splits a frequency grid Nf into B = ceil(Nf / N_out) NuFFT-1 blocks of power-of-2
// base size N_block in [512, 131072], deriving ALL architectural ceilings and memory
// limits directly from SYCL device properties and runtime hardware probes (zero
// GPU-model-specific constants):
//   1. SinglePass vs TwoPass FFT ceiling derived from dev_cfg.local_mem_size
//   2. Spreading window capacity derived from dev_cfg.local_mem_size & kernel width w
//   3. Native FP64 compute weight derived from runtime probe_fp64_ratio()
//   4. Per-stream compute-unit occupancy derived from dev_cfg.max_compute_units / num_slots
//   5. Strict per-lightcurve memory budget: 75% of max device memory / num_streams,
//      also respecting dev_cfg.safe_max_alloc_bytes() on every individual buffer.
//
// Optional debug override:
//   TLS_SYCL_BLOCK_SIZE=<512|1024|2048|...|131072>  pin N_block directly

#include "common/device.hpp"
#include "nufft/nufft1.hpp"
#include "periodogram/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace sycl_periodogram {

struct GpuPlanChoice {
    int                  N_block            = 512;
    int                  nfft               = 1024;
    int                  nout               = 768;
    int                  w                  = 9;
    int                  B_total            = 1;
    int                  B_chunk            = 1;
    size_t               plan_max_bs        = 64;
    size_t               slot_budget_bytes  = 0;
    size_t               required_min_bytes = 0;
    bool                 fits_budget        = true;
    sycl_nufft::PswfMode mode               = sycl_nufft::PswfMode::PSWF43;
};

inline int kernel_width_for(PrecKind prec, sycl_nufft::PswfMode mode) {
    const bool m43 = (mode == sycl_nufft::PswfMode::PSWF43);
    switch (prec) {
        case TLS_SYCL_PREC_HALF:   return m43 ? 5  : 4;
        case TLS_SYCL_PREC_FLOAT:  return m43 ? 9  : 8;
        case TLS_SYCL_PREC_DF:     return m43 ? 16 : 14;
        case TLS_SYCL_PREC_DOUBLE: return m43 ? 18 : 16;
    }
    return 9;
}

inline size_t complex_bytes_for(PrecKind prec) {
    switch (prec) {
        case TLS_SYCL_PREC_HALF:   return 4;
        case TLS_SYCL_PREC_FLOAT:  return 8;
        case TLS_SYCL_PREC_DF:     return 16;
        case TLS_SYCL_PREC_DOUBLE: return 16;
    }
    return 8;
}

inline size_t xcoord_bytes_for(PrecKind prec) {
    return (prec == TLS_SYCL_PREC_DOUBLE) ? size_t{8} : size_t{8}; // df_t and double are both 8B
}

inline size_t scalar_bytes_for(PrecKind prec) {
    switch (prec) {
        case TLS_SYCL_PREC_HALF:   return 2;
        case TLS_SYCL_PREC_FLOAT:  return 4;
        case TLS_SYCL_PREC_DF:     return 8;
        case TLS_SYCL_PREC_DOUBLE: return 8;
    }
    return 4;
}

inline size_t dev_out_bytes_for(PrecKind prec) {
    switch (prec) {
        case TLS_SYCL_PREC_HALF:   return 4; // float
        case TLS_SYCL_PREC_FLOAT:  return 4; // float
        case TLS_SYCL_PREC_DF:     return 8; // df_t
        case TLS_SYCL_PREC_DOUBLE: return 8; // double
    }
    return 4;
}

inline size_t solve_real_bytes_for(PrecKind prec) {
    switch (prec) {
        case TLS_SYCL_PREC_HALF:   return 4; // f16x2_t (2 lanes packed)
        case TLS_SYCL_PREC_FLOAT:  return 4;
        case TLS_SYCL_PREC_DF:     return 8;
        case TLS_SYCL_PREC_DOUBLE: return 8;
    }
    return 4;
}

inline size_t pow2_floor_sz(size_t v) {
    if (v == 0) return 1;
    size_t p = 1;
    while ((p << 1) <= v && (p << 1) > p) p <<= 1;
    return p;
}

inline int env_int_or(const char* name, int fallback) {
    if (const char* v = std::getenv(name); v && v[0] != '\0') {
        const int parsed = std::atoi(v);
        if (parsed >= 0) return parsed;
    }
    return fallback;
}

// Compute exact per-slot memory footprint & feasible B_chunk for a candidate N_block.
struct ChunkMemoryFit {
    bool   fits             = false;
    int    B_chunk          = 1;
    size_t plan_max_bs      = 1;
    size_t fixed_plus_plan  = 0;
    size_t bytes_per_blk    = 0;
};

inline ChunkMemoryFit evaluate_chunk_memory(
    int Nf,
    int M,
    int degree,
    bool is_ihs,
    bool has_cond,
    sycl_nufft::PswfMode mode,
    PrecKind prec,
    int N_block,
    const sycl_fft::DeviceFftConfig& dev_cfg,
    size_t slot_budget) {
    ChunkMemoryFit fit{};
    const int nfft = 2 * N_block;
    const int nout = sycl_nufft::nufft1_nout(N_block, mode);
    const int w = kernel_width_for(prec, mode);
    const int B_total = std::max(1, (Nf + nout - 1) / nout);

    const size_t cbytes   = complex_bytes_for(prec);
    const size_t sbytes   = scalar_bytes_for(prec);
    const size_t xbytes   = xcoord_bytes_for(prec);
    const size_t obytes   = dev_out_bytes_for(prec);
    const size_t max_alloc = dev_cfg.safe_max_alloc_bytes();

    // 1. Fixed per-lightcurve device allocations: packed (x, w, [yw]) + d_power + optional d_cond
    auto align64 = [](size_t n) -> size_t { return (n + 63u) & ~size_t{63u}; };
    const size_t in_bytes = is_ihs
        ? (align64(static_cast<size_t>(M) * xbytes) + static_cast<size_t>(M) * sbytes)
        : (align64(static_cast<size_t>(M) * xbytes) + align64(static_cast<size_t>(M) * sbytes) + static_cast<size_t>(M) * sbytes);
    const size_t out_bytes = static_cast<size_t>(Nf) * obytes;
    const size_t cond_bytes = (!is_ihs && has_cond) ? out_bytes : size_t{0};

    // 2. Nufft1Plan internal working buffers (bounded so plan cache fits comfortably in slot_budget)
    //    Point geometry: d_first(4B) + d_u(sbytes) + d_shift(cbytes) + d_weights(w*sbytes) + buckets(~3x)
    const size_t per_pt_plan_bytes = size_t{4} + sbytes + cbytes + static_cast<size_t>(w) * sbytes;
    const size_t plan_pt_bytes = static_cast<size_t>(M) * per_pt_plan_bytes * size_t{3} +
                                 static_cast<size_t>(nout) * (sbytes * size_t{2} + size_t{4});

    //    FFT buffer + optional TwoPass scratch: target effective_l2_bytes(), capped to slot_budget / 6
    const size_t fft_elem_bytes = static_cast<size_t>(nfft) * cbytes;
    const size_t plan_fft_budget = std::min<size_t>(
        std::max<size_t>(size_t{2} * fft_elem_bytes, dev_cfg.effective_l2_bytes()),
        std::min<size_t>(max_alloc / 2, std::max<size_t>(size_t{2} * fft_elem_bytes, slot_budget / 6)));
    // Account for both d_fft_buf_ and potential TwoPass scratch_ (2x fft_elem_bytes per batch item)
    size_t plan_bs = std::max<size_t>(1, plan_fft_budget / std::max<size_t>(1, size_t{2} * fft_elem_bytes));
    if (prec == TLS_SYCL_PREC_HALF && plan_bs > 1 && (plan_bs & 1u)) --plan_bs;
    // Also no reason to exceed 2 * B_total (or 512 for plan reuse)
    plan_bs = std::clamp<size_t>(plan_bs, size_t{2}, size_t{512});
    if (prec == TLS_SYCL_PREC_HALF && (plan_bs & 1u)) ++plan_bs;
    const size_t plan_fft_bytes = size_t{2} * plan_bs * fft_elem_bytes;

    const size_t fixed_total = in_bytes + out_bytes + cond_bytes + plan_pt_bytes + plan_fft_bytes;
    fit.plan_max_bs = plan_bs;
    fit.fixed_plus_plan = fixed_total;

    // 3. Per-block chunk allocations: d_src + d_trig/d_z + optional d_ihs_acc + optional d_a_scratch
    const size_t src_mult = is_ihs ? size_t{1} : size_t{2};
    const size_t src_per_blk = src_mult * static_cast<size_t>(M) * cbytes;
    const size_t trig_rows = is_ihs ? size_t{1} : static_cast<size_t>(3 * degree);
    const size_t trig_per_blk = trig_rows * static_cast<size_t>(nout) * cbytes;
    const size_t ihs_acc_per_blk = (is_ihs && degree > 1) ? (static_cast<size_t>(nout) * obytes) : size_t{0};
    size_t scr_per_blk = 0;
    if (!is_ihs && degree > 12) {
        const size_t items_per_blk = (prec == TLS_SYCL_PREC_HALF)
            ? static_cast<size_t>(nout >> 1)
            : static_cast<size_t>(nout);
        scr_per_blk = size_t{2} * static_cast<size_t>(2 * degree + 1) * items_per_blk * solve_real_bytes_for(prec);
    }

    const size_t per_blk_total = src_per_blk + trig_per_blk + ihs_acc_per_blk + scr_per_blk;
    const size_t max_single_per_blk = std::max({src_per_blk, trig_per_blk, ihs_acc_per_blk, scr_per_blk});
    fit.bytes_per_blk = per_blk_total;

    if (in_bytes > max_alloc || out_bytes > max_alloc ||
        fixed_total >= slot_budget ||
        max_single_per_blk > max_alloc) {
        fit.fits = false;
        return fit;
    }

    const size_t avail_chunk = slot_budget - fixed_total;
    if (per_blk_total > avail_chunk) {
        fit.fits = false;
        return fit;
    }

    size_t b_by_budget = avail_chunk / std::max<size_t>(1, per_blk_total);
    size_t b_by_alloc  = max_alloc / std::max<size_t>(1, max_single_per_blk);
    size_t b_chunk = std::min<size_t>({static_cast<size_t>(B_total), b_by_budget, b_by_alloc});
    if (b_chunk == 0) {
        fit.fits = false;
        return fit;
    }
    if (prec == TLS_SYCL_PREC_HALF && b_chunk < static_cast<size_t>(B_total) && b_chunk > 1 && (b_chunk & 1u)) {
        --b_chunk;
    }
    fit.B_chunk = static_cast<int>(std::max<size_t>(1, b_chunk));
    fit.fits = true;
    return fit;
}

// Analytical hardware-parameterized cost model (zero GPU-model-specific constants).
inline double eval_gpu_block_cost(
    int Nf,
    int M,
    int degree,
    bool is_ihs,
    sycl_nufft::PswfMode mode,
    PrecKind prec,
    int N_block,
    const sycl_fft::DeviceFftConfig& dev_cfg,
    int num_slots,
    double fp64_to_fp32_ratio) {
    const int nout = sycl_nufft::nufft1_nout(N_block, mode);
    const int nfft = 2 * N_block;
    const int log_nfft = sycl_nufft::int_log2_pow2(nfft);
    const int w = kernel_width_for(prec, mode);
    const double B = std::ceil(static_cast<double>(Nf) / static_cast<double>(nout));
    const double n_exec = is_ihs ? static_cast<double>(degree) : static_cast<double>(3 * degree);
    const double n_factors = is_ihs ? static_cast<double>(degree) : static_cast<double>(2 * degree);

    const size_t cbytes = complex_bytes_for(prec);
    // FP16 processes 2 batch lanes packed in f16x2 (8 bytes per complex pair in SMEM)
    const size_t smem_elem_bytes = (prec == TLS_SYCL_PREC_HALF) ? size_t{8} : cbytes;
    const double prec_bytes = static_cast<double>(cbytes);

    // 1. SinglePass SMEM ceiling derived directly from dev_cfg.local_mem_size
    const size_t smem_usable = dev_cfg.local_mem_size - (dev_cfg.local_mem_size / 16);
    const int sp_max_nfft = (!dev_cfg.has_native_local_mem || smem_usable < 4096)
        ? 2048
        : static_cast<int>(std::clamp<size_t>(
              pow2_floor_sz(smem_usable / std::max<size_t>(1, smem_elem_bytes)) * 2,
              size_t{1024}, size_t{8192}));

    // 2. Native FP64 compute weight derived from runtime probe_fp64_ratio()
    //    (full-rate datacenter FP64 >= 0.45 -> 1.0; throttled consumer FP64 -> smoothly scaled)
    double fp64_alu_mult = 1.0;
    if (prec == TLS_SYCL_PREC_DOUBLE && fp64_to_fp32_ratio > 0.0 && fp64_to_fp32_ratio < 0.45) {
        fp64_alu_mult = std::clamp(std::sqrt(0.5 / fp64_to_fp32_ratio), 1.0, 3.0);
    }

    // 3. FFT traffic + compute cost:
    //    SinglePass = 1 global pass; TwoPass = 2 global passes + transpose overhead.
    double fft_pass_mult = 1.0;
    if (nfft > sp_max_nfft) {
        fft_pass_mult = 2.25;
    } else if (nfft == sp_max_nfft && sp_max_nfft >= 4096) {
        // Near-ceiling SinglePass reduces resident work-groups per CU
        fft_pass_mult = 1.25;
    }
    const double fft_per_blk =
        static_cast<double>(nfft) * (0.35 + 0.065 * static_cast<double>(log_nfft)) *
        fft_pass_mult * (prec_bytes / 8.0);
    const double fft_cost = n_exec * B * fft_per_blk;

    // 4. Spreading + window-bucketing cost:
    //    Window size derived from dev_cfg.local_mem_size & smem_elem_bytes (assuming ~4 concurrent batch lanes/WG).
    const size_t win_cap_sz = (!dev_cfg.has_native_local_mem || smem_usable < 4096)
        ? static_cast<size_t>(nfft)
        : std::clamp<size_t>(pow2_floor_sz(smem_usable / std::max<size_t>(1, size_t{4} * smem_elem_bytes)),
                             size_t{128}, size_t{4096});
    const double num_wins = std::max(1.0, static_cast<double>(nfft) / static_cast<double>(win_cap_sz));
    const double win_pen = (num_wins <= 1.0) ? 1.0 : (1.0 + 0.12 * std::log2(num_wins));
    const double spread_beta = 0.55 * fp64_alu_mult;
    const double spread_cost = n_exec * B * static_cast<double>(M) * static_cast<double>(w) * spread_beta * win_pen;

    // 5. Subgrid phase-modulation + O(M) set_points precompute:
    //    With strip-recurrence in subgrid.cpp, trig cost per block is amortized;
    //    DF pays ~3x FP32 op count algorithmically, FP64 scales with fp64_alu_mult.
    const double phase_beta = (prec == TLS_SYCL_PREC_DF)
        ? 2.2
        : ((prec == TLS_SYCL_PREC_DOUBLE) ? (1.2 * fp64_alu_mult) : 0.75);
    const double phase_cost = n_factors * B * static_cast<double>(M) * phase_beta;
    const double pre_cost = n_factors * static_cast<double>(M) * (2.0 + 0.5 * num_wins);

    // 6. Per-stream compute-unit occupancy penalty:
    //    Avoid overly large N_block that leaves B too small to occupy this stream's CU share.
    const double cus_per_slot = std::max(
        2.0,
        static_cast<double>(dev_cfg.max_compute_units) / static_cast<double>(std::max(1, num_slots)));
    const double eff_bs = B * (is_ihs ? 1.0 : 1.5);
    double occ_pen = 1.0;
    if (dev_cfg.is_gpu && eff_bs < cus_per_slot) {
        occ_pen = 1.0 + 0.30 * ((cus_per_slot - eff_bs) / cus_per_slot);
    }

    // 7. Small fixed dispatch overhead per harmonic
    const double launch_overhead = n_exec * 400.0;

    return (fft_cost + spread_cost + phase_cost + launch_overhead) * occ_pen + pre_cost;
}

inline GpuPlanChoice choose_gpu_periodogram_plan(
    int Nf,
    int M,
    int degree,
    bool is_ihs,
    bool has_cond,
    sycl_nufft::PswfMode mode,
    PrecKind prec,
    const sycl_fft::DeviceFftConfig& dev_cfg,
    int num_slots,
    double fp64_to_fp32_ratio = 0.0) {
    GpuPlanChoice out{};
    out.mode = mode;
    out.w = kernel_width_for(prec, mode);
    out.slot_budget_bytes = dev_cfg.per_stream_mem_budget(static_cast<size_t>(std::max(1, num_slots)));

    const int forced_block = env_int_or("TLS_SYCL_BLOCK_SIZE", 0);

    int best_block = 512;
    ChunkMemoryFit best_fit{};
    bool found_any = false;

    if (forced_block >= 512 && forced_block <= 131072 && (forced_block & (forced_block - 1)) == 0) {
        best_block = forced_block;
        best_fit = evaluate_chunk_memory(
            Nf, M, degree, is_ihs, has_cond, mode, prec, best_block, dev_cfg, out.slot_budget_bytes);
        found_any = best_fit.fits;
    } else {
        // Walk all power-of-2 N_block in [512, 131072] up to the smallest block covering Nf
        // (and fitting within the per-stream 75% device memory budget).
        int max_cand = 512;
        while (max_cand < 131072 && sycl_nufft::nufft1_nout(max_cand, mode) < Nf) {
            max_cand <<= 1;
        }

        double best_cost = 0.0;
        for (int cand = 512; cand <= max_cand; cand <<= 1) {
            const ChunkMemoryFit fit = evaluate_chunk_memory(
                Nf, M, degree, is_ihs, has_cond, mode, prec, cand, dev_cfg, out.slot_budget_bytes);
            if (!fit.fits) {
                if (!found_any) {
                    best_block = cand;
                    best_fit = fit;
                }
                continue;
            }
            const double c = eval_gpu_block_cost(
                Nf, M, degree, is_ihs, mode, prec, cand, dev_cfg, num_slots, fp64_to_fp32_ratio);
            if (!found_any || c < best_cost) {
                found_any = true;
                best_cost = c;
                best_block = cand;
                best_fit = fit;
            }
        }
    }

    out.N_block            = best_block;
    out.nfft               = 2 * best_block;
    out.nout               = sycl_nufft::nufft1_nout(best_block, mode);
    out.B_total            = std::max(1, (Nf + out.nout - 1) / out.nout);
    out.B_chunk            = best_fit.B_chunk;
    out.plan_max_bs        = best_fit.plan_max_bs;
    out.required_min_bytes = best_fit.fixed_plus_plan + best_fit.bytes_per_blk;
    out.fits_budget        = found_any;
    return out;
}

} // namespace sycl_periodogram
