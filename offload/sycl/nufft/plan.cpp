// Pure-Host NuFFT-1 Plan Orchestration Unit (ZERO SYCL device kernels / 0 KiB HCF).
// Derives all spreading, transposition, deconvolution, and L2-chunking parameters dynamically
// from DeviceFftConfig + BankModel + persistent AutotuneCache (zero hardcoded GPU constants).
#include "nufft/nufft1.hpp"

#include <chrono>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sycl_nufft {

namespace {

// Precision-tag dispatchers forwarding to decoupled device-kernel TUs
template <typename Traits>
void dispatch_precompute(
    sycl::queue& q,
    const typename Traits::XCoordT* d_x,
    typename Traits::XCoordT df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43, bool pos_sign,
    int* d_first,
    typename Traits::ScalarT* d_u,
    typename Traits::ComplexT* d_shift,
    size_t wg_size) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_precompute_pointsh(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, pos_sign, d_first, d_u, d_shift, wg_size);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_precompute_points(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, pos_sign, d_first, d_u, d_shift, wg_size);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_precompute_pointsdf(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, pos_sign, d_first, d_u, d_shift, wg_size);
    } else {
        detail::launch_precompute_pointsf(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, pos_sign, d_first, d_u, d_shift, wg_size);
    }
}

template <typename Traits>
sycl_fft::UsmPtr<uint8_t> dispatch_upload_pswf_table(
    sycl::queue& q, int w, PswfMode mode, bool geo_scale_fp16) {
    uint8_t* ptr = nullptr;
    if constexpr (Traits::kIsHalf) {
        ptr = detail::upload_pswf_tableh(q, w, mode, geo_scale_fp16);
    } else if constexpr (Traits::kIsFp64) {
        (void)geo_scale_fp16;
        ptr = detail::upload_pswf_table(q, w, mode);
    } else if constexpr (Traits::kIsDf) {
        (void)geo_scale_fp16;
        ptr = detail::upload_pswf_tabledf(q, w, mode);
    } else {
        (void)geo_scale_fp16;
        ptr = detail::upload_pswf_tablef(q, w, mode);
    }
    return sycl_fft::UsmPtr<uint8_t>(ptr, &q);
}

template <typename Traits>
void dispatch_precompute_weights(
    sycl::queue& q, int w, const void* d_tbl,
    const typename Traits::ScalarT* d_u, typename Traits::ScalarT* d_weights,
    int M, size_t wg_size) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_precompute_weightsh(q, w, d_tbl, d_u, d_weights, M, wg_size);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_precompute_weights(q, w, d_tbl, d_u, d_weights, M, wg_size);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_precompute_weightsdf(q, w, d_tbl, d_u, d_weights, M, wg_size);
    } else {
        detail::launch_precompute_weightsf(q, w, d_tbl, d_u, d_weights, M, wg_size);
    }
}

template <typename Traits>
void dispatch_pack_bucket_geom(
    sycl::queue& q, int w, int num_wins, int max_entries,
    const int* d_bucket_off, const int* d_bucket_m,
    const int* d_first, const typename Traits::ComplexT* d_shift, const typename Traits::ScalarT* d_weights,
    int* d_b_first, typename Traits::ComplexT* d_b_shift, typename Traits::ScalarT* d_b_weights,
    size_t wg_size) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_pack_bucket_geomh(q, w, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_pack_bucket_geom(q, w, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_pack_bucket_geomdf(q, w, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    } else {
        detail::launch_pack_bucket_geomf(q, w, num_wins, max_entries, d_bucket_off, d_bucket_m, d_first, d_shift, d_weights, d_b_first, d_b_shift, d_b_weights, wg_size);
    }
}

template <typename Traits>
sycl_fft::KernelProbeInfo dispatch_probe_spread_reg(
    sycl::queue& q, const sycl::device& dev, int w, const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        return detail::probe_spread_reg_kernelh(q, dev, w, cfg);
    } else if constexpr (Traits::kIsFp64) {
        return detail::probe_spread_reg_kernel(q, dev, w, cfg);
    } else if constexpr (Traits::kIsDf) {
        return detail::probe_spread_reg_kerneldf(q, dev, w, cfg);
    } else {
        return detail::probe_spread_reg_kernelf(q, dev, w, cfg);
    }
}

template <typename Traits>
void dispatch_spread_smem(
    sycl::queue& q, int w, const void* d_tbl, const typename Traits::ScalarT* d_weights,
    const int* d_first, const typename Traits::ScalarT* d_u, const typename Traits::ComplexT* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const typename Traits::ComplexT* d_y, typename Traits::ComplexT* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_spread_smem_winh(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_spread_smem_win(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_spread_smem_windf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        detail::launch_spread_smem_winf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
}

template <typename Traits>
void dispatch_spread_collab(
    sycl::queue& q, int w, const void* d_tbl, const typename Traits::ScalarT* d_weights,
    const int* d_first, const typename Traits::ScalarT* d_u, const typename Traits::ComplexT* d_shift,
    const typename Traits::ComplexT* d_y,
    typename Traits::RealT* d_grid_T_re, typename Traits::RealT* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_spread_collabh(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_spread_collab(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_spread_collabdf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        detail::launch_spread_collabf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    }
}

template <typename Traits>
void dispatch_spread_cpu_direct(
    sycl::queue& q, int w, const void* d_tbl, const typename Traits::ScalarT* d_weights,
    const int* d_first, const typename Traits::ScalarT* d_u, const typename Traits::ComplexT* d_shift,
    const typename Traits::ComplexT* d_y, typename Traits::ComplexT* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_spread_cpu_directh(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_spread_cpu_direct(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_spread_cpu_directdf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    } else {
        detail::launch_spread_cpu_directf(q, w, d_tbl, d_weights, d_first, d_u, d_shift, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs);
    }
}

template <typename Traits>
void dispatch_transpose_and_zero(
    sycl::queue& q,
    typename Traits::RealT* d_grid_T_re, typename Traits::RealT* d_grid_T_im,
    typename Traits::ComplexT* d_fft_buf,
    int nfft, size_t chunk_bs, size_t packed_stride,
    const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_transpose_and_zero_f16pair(
            q, d_grid_T_re, d_grid_T_im, d_fft_buf,
            nfft, chunk_bs, packed_stride, cfg.transpose_tile, cfg.transpose_stride);
    } else if constexpr (sizeof(typename Traits::RealT) == 4) {
        detail::launch_transpose_and_zero_4b(
            q, reinterpret_cast<uint32_t*>(d_grid_T_re), reinterpret_cast<uint32_t*>(d_grid_T_im),
            reinterpret_cast<uint64_t*>(d_fft_buf),
            nfft, chunk_bs, packed_stride, cfg.transpose_tile, cfg.transpose_stride);
    } else {
        detail::launch_transpose_and_zero_8b(
            q, reinterpret_cast<uint64_t*>(d_grid_T_re), reinterpret_cast<uint64_t*>(d_grid_T_im),
            reinterpret_cast<sycl::ulong2*>(d_fft_buf),
            nfft, chunk_bs, packed_stride, cfg.transpose_tile, cfg.transpose_stride);
    }
}

template <typename Traits>
void dispatch_build_deconv_table(
    sycl::queue& q, int w, const void* d_tbl,
    int nfft, int log_nfft, int nout, int output_shift,
    PswfMode mode, int table_kind, typename Traits::ScalarT* d_out,
    double output_scale) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_build_deconv_tableh(q, w, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_build_deconv_table(q, w, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_build_deconv_tabledf(q, w, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    } else {
        detail::launch_build_deconv_tablef(q, w, d_tbl, nfft, log_nfft, nout, output_shift, mode, table_kind, d_out, output_scale);
    }
}

template <typename Traits>
void dispatch_deconv(
    sycl::queue& q, int w, const void* d_tbl,
    const typename Traits::ComplexT* d_fft_buf, typename Traits::ComplexT* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg,
    double output_scale) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_deconv_emith(q, w, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_deconv_emit(q, w, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_deconv_emitdf(q, w, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    } else {
        detail::launch_deconv_emitf(q, w, d_tbl, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg, output_scale);
    }
}

inline int pow2_floor_int(int v) {
    int p = 1;
    while ((p << 1) <= v) p <<= 1;
    return p;
}

inline int pow2_clamp_int(int v, int lo, int hi) {
    if (v <= lo) return lo;
    if (v >= hi) return hi;
    return pow2_floor_int(v);
}

struct SpreadCandidate {
    std::string  variant_id;
    SpreadConfig cfg;
    size_t       smem_bytes = 0;
    int          score      = 0;
};

struct DeconvCandidate {
    std::string  variant_id;
    DeconvConfig cfg;
};

template <typename Traits>
std::vector<SpreadCandidate> build_spread_candidates(
    const sycl_fft::DeviceFftConfig& dev_cfg,
    size_t eff_sg,
    int nfft,
    int w,
    bool use_precomp_weights,
    bool force_collab,
    bool allow_emulated_smem) {
    using RealT    = typename Traits::RealT;
    using ScalarT  = typename Traits::ScalarT;
    using ComplexT = typename Traits::ComplexT;

    std::vector<SpreadCandidate> cands;
    const auto bm = sycl_fft::bank_model_from_kind(dev_cfg.bank_model);

    // Derived bank-conflict-free transpose parameters (for collab fallback)
    const int tr_tile = (dev_cfg.max_work_group_size >= 256 &&
                         dev_cfg.has_native_local_mem &&
                         2 * 16 * 17 * sizeof(RealT) <= dev_cfg.local_mem_size)
                            ? 16
                            : ((dev_cfg.max_work_group_size >= 64) ? 8 : 4);
    const int tr_stride = sycl_fft::find_min_conflict_stride2(
        bm, static_cast<int>(sizeof(RealT)), tr_tile);

    const size_t sg_unit = std::max<size_t>(
        1, (eff_sg > 1) ? eff_sg : std::max<size_t>(1, dev_cfg.sub_group_size));
    const int t_j_hi = next_pow2_int(w);
    const bool has_multipass_tj = (w == 5 || w == 9 || w == 17 || w == 18);
    std::vector<int> tj_options;
    if (has_multipass_tj) {
        tj_options.push_back(pow2_floor_int(w)); // prev_pow2(W): doubles b_per_wg & fixes 37.5%-47% lane idling
    }
    tj_options.push_back(t_j_hi);

    auto grid_pad_for_tj = [&](int t_j) -> int {
        if (bm.banks > 1) {
            // 4-byte RealT packs {re, im} into an 8-byte PairT (LDS.64/STS.64); >=8-byte RealT uses split arrays
            const int elem_bytes = (sizeof(RealT) == 4)
                ? static_cast<int>(2 * sizeof(RealT))
                : static_cast<int>(sizeof(RealT));
            const int banks_per_elem = (elem_bytes + bm.bank_width_bytes - 1) / bm.bank_width_bytes;
            const int elem_banks = std::max(1, bm.banks / std::max(1, banks_per_elem));
            if (t_j < elem_banks) {
                return t_j;
            }
        }
        return 0;
    };

    auto calc_smem_bytes = [&](int b_per_wg, int win_elems, int tile_m, int grid_pad, bool use_reg_acc) -> size_t {
        const int grid_stride = win_elems + grid_pad;
        const int sy_stride   = b_per_wg + 1;
        const size_t grid_b   = use_reg_acc
            ? size_t{0}
            : (2 * static_cast<size_t>(b_per_wg) * static_cast<size_t>(grid_stride) * sizeof(RealT));
        const size_t base =
            grid_b +
            static_cast<size_t>(tile_m) * sizeof(int) +
            2 * static_cast<size_t>(tile_m) * static_cast<size_t>(sy_stride) * sizeof(RealT);
        if (use_precomp_weights) {
            return base + static_cast<size_t>(tile_m) * static_cast<size_t>(w) * sizeof(ScalarT);
        }
        return base + static_cast<size_t>(w) * static_cast<size_t>(tile_m + 1) * sizeof(ScalarT);
    };

    const bool smem_allowed = !force_collab &&
                              (dev_cfg.has_native_local_mem || allow_emulated_smem) &&
                              dev_cfg.local_mem_size >= 4096;

    if (smem_allowed) {
        // Leave 1/16 headroom for driver/compiler local-memory metadata (overflow-safe for UINT64_MAX)
        const size_t smem_limit = dev_cfg.local_mem_size - (dev_cfg.local_mem_size / 16);
        const int min_win = std::min(nfft, next_pow2_int(std::max(w * 2, 64)));
        const int wg_max  = std::min<int>(static_cast<int>(dev_cfg.max_work_group_size), 256);

        for (int t_j : tj_options) {
            const int grid_pad = grid_pad_for_tj(t_j);
            const int wg_min   = std::max(t_j, static_cast<int>(sg_unit));

            std::vector<int> wg_list;
            for (int mult : {2, 4, 8, 1}) {
                const int wg = pow2_floor_int(static_cast<int>(sg_unit) * mult);
                if (wg >= wg_min && wg <= wg_max && (wg % t_j) == 0) {
                    if (std::find(wg_list.begin(), wg_list.end(), wg) == wg_list.end()) {
                        wg_list.push_back(wg);
                    }
                }
            }
            if (wg_list.empty() && wg_min <= static_cast<int>(dev_cfg.max_work_group_size)) {
                wg_list.push_back(wg_min);
            }

            auto add_smem_cand = [&](int wg_size, int win_elems, bool use_reg_acc) {
                const int b_per_wg = wg_size / t_j;
                if (b_per_wg <= 0 || win_elems < min_win || win_elems > nfft) return;
                if (use_reg_acc) {
                    const int q_val = win_elems / t_j;
                    const bool q_ok = (q_val == 4 || q_val == 8 || q_val == 16 || q_val == 32) ||
                                      (w > 16 && q_val == 2);
                    if (!use_precomp_weights || win_elems < 64 || !q_ok) return;
                }
                int tile_m = std::min(wg_size, std::max(static_cast<int>(sg_unit), 32));
                tile_m = pow2_clamp_int(tile_m, 16, 64);
                while (tile_m > 16 && calc_smem_bytes(b_per_wg, win_elems, tile_m, grid_pad, use_reg_acc) > smem_limit) {
                    tile_m >>= 1;
                }
                // Also step tile_m down to 16 if doing so strictly increases resident WGs/CU
                while (tile_m > 16) {
                    const size_t cur_b  = calc_smem_bytes(b_per_wg, win_elems, tile_m, grid_pad, use_reg_acc);
                    const size_t next_b = calc_smem_bytes(b_per_wg, win_elems, tile_m >> 1, grid_pad, use_reg_acc);
                    const int cur_res  = std::min(4, static_cast<int>(dev_cfg.local_mem_size / std::max<size_t>(1, cur_b)));
                    const int next_res = std::min(4, static_cast<int>(dev_cfg.local_mem_size / std::max<size_t>(1, next_b)));
                    if (next_res > cur_res) {
                        tile_m >>= 1;
                    } else {
                        break;
                    }
                }
                const size_t smem_b = calc_smem_bytes(b_per_wg, win_elems, tile_m, grid_pad, use_reg_acc);
                if (smem_b > smem_limit) return;

                std::string id =
                    (use_reg_acc ? "reg_wg" : "smem_wg") + std::to_string(wg_size) +
                    "_win" + std::to_string(win_elems);
                if (t_j < w) {
                    id += "_tj" + std::to_string(t_j);
                }
                for (const auto& existing : cands) {
                    if (existing.variant_id == id) return;
                }

                SpreadConfig sc{};
                sc.use_smem            = true;
                sc.use_reg_acc         = use_reg_acc;
                sc.use_cpu_direct      = false;
                sc.use_precomp_weights = use_precomp_weights;
                sc.wg_size             = wg_size;
                sc.t_j                 = t_j;
                sc.b_per_wg            = b_per_wg;
                sc.win_elems           = win_elems;
                sc.num_wins            = nfft / win_elems;
                sc.grid_stride         = win_elems + grid_pad;
                sc.tile_m              = tile_m;
                sc.transpose_tile      = tr_tile;
                sc.transpose_stride    = tr_stride;

                const int resident_wgs = std::min(4, static_cast<int>(dev_cfg.local_mem_size / std::max<size_t>(1, smem_b)));
                const int active_warps_per_cu = resident_wgs * std::max(1, wg_size / static_cast<int>(sg_unit));
                int score = std::min(active_warps_per_cu, 16) * 20;
                score += std::min(b_per_wg, 32) * 10;
                if (use_reg_acc) {
                    const int q_val = win_elems / t_j;
                    // Keep heuristic default conservative (avoid risky huge-Q when untuned);
                    // Phase A still benchmarks all widened Q families directly.
                    score += (q_val <= 8) ? 40 : ((q_val == 16) ? 15 : -20);
                }
                if (t_j < w) score += 25;     // Reward high lane utilization (prev_pow2(W))
                if (sc.num_wins == 1) score += 30;
                else if (sc.num_wins <= 4) score += 18;
                else score -= int_log2_pow2(sc.num_wins) * 8;
                if (win_elems < 128) score -= 25;

                cands.push_back(SpreadCandidate{id, sc, smem_b, score});
            };

            for (int wg_size : wg_list) {
                const int b_per_wg = wg_size / t_j;
                // Widened register-resident candidates: all Q in {2, 4, 8, 16, 32} (no hardcoded byte cap;
                // runtime private_mem_size spill probe + Phase A interleaved measurement decide).
                if (use_precomp_weights) {
                    for (int q_val : {32, 16, 8, 4, 2}) {
                        if (q_val == 2 && w <= 16) continue;
                        const int rwin = q_val * t_j;
                        if (rwin >= min_win && rwin <= nfft && rwin >= 64) {
                            add_smem_cand(wg_size, rwin, true);
                        }
                    }
                }
                int max_win = nfft;
                while (max_win >= min_win && calc_smem_bytes(b_per_wg, max_win, 16, grid_pad, false) > smem_limit) {
                    max_win >>= 1;
                }
                if (max_win >= min_win) {
                    add_smem_cand(wg_size, max_win, false);
                    if (max_win / 2 >= min_win) {
                        add_smem_cand(wg_size, max_win / 2, false);
                    }
                    if (max_win / 4 >= min_win && calc_smem_bytes(b_per_wg, max_win / 2, 16, grid_pad, false) * 2 > dev_cfg.local_mem_size) {
                        add_smem_cand(wg_size, max_win / 4, false);
                    }
                }
            }
        }

        std::stable_sort(cands.begin(), cands.end(), [](const SpreadCandidate& a, const SpreadCandidate& b) {
            return a.score > b.score;
        });
    }

    // Barrier-free direct CPU spreading (primary when !has_native_local_mem and not forced to collab)
    if (!dev_cfg.has_native_local_mem && !force_collab) {
        SpreadConfig dc{};
        dc.use_smem            = false;
        dc.use_cpu_direct      = true;
        dc.use_precomp_weights = use_precomp_weights;
        dc.wg_size             = 1;
        dc.t_j                 = t_j_hi;
        dc.b_per_wg            = 1;
        dc.win_elems           = nfft;
        dc.num_wins            = 1;
        dc.grid_stride         = nfft;
        dc.tile_m              = 1;
        dc.transpose_tile      = tr_tile;
        dc.transpose_stride    = tr_stride;
        cands.push_back(SpreadCandidate{"cpu_direct", dc, 0, -50});
    }

    // Collaborative batch-inner candidate (fallback / --force-collab-spread)
    {
        const size_t reg_cap = (sizeof(RealT) >= 8) ? 128 : 256;
        const int collab_wg = static_cast<int>(
            dev_cfg.default_wg_size(std::max<size_t>(64, sg_unit * 4), reg_cap));
        int tile_m = pow2_clamp_int(std::min(collab_wg, static_cast<int>(std::max<size_t>(32, sg_unit))), 16, 64);
        const size_t per_pt_bytes = sizeof(int) + sizeof(ComplexT) + static_cast<size_t>(w) * sizeof(ScalarT);
        while (tile_m > 16 && static_cast<size_t>(tile_m) * per_pt_bytes > dev_cfg.local_mem_size / 4) {
            tile_m >>= 1;
        }
        SpreadConfig cc{};
        cc.use_smem            = false;
        cc.use_cpu_direct      = false;
        cc.use_precomp_weights = use_precomp_weights;
        cc.wg_size             = std::max(1, collab_wg);
        cc.t_j                 = t_j_hi;
        cc.b_per_wg            = cc.wg_size;
        cc.win_elems           = nfft;
        cc.num_wins            = 1;
        cc.grid_stride         = nfft;
        cc.tile_m              = tile_m;
        cc.transpose_tile      = tr_tile;
        cc.transpose_stride    = tr_stride;

        const size_t collab_smem = static_cast<size_t>(tile_m) * per_pt_bytes;
        const std::string cid = "collab_wg" + std::to_string(cc.wg_size);
        cands.push_back(SpreadCandidate{cid, cc, collab_smem, -100});
    }

    return cands;
}

template <typename Traits>
std::vector<DeconvCandidate> build_deconv_candidates(
    const sycl_fft::DeviceFftConfig& dev_cfg,
    size_t eff_sg,
    int nout,
    size_t chunk_bs) {
    std::vector<DeconvCandidate> cands;
    const size_t sg_unit = std::max<size_t>(
        1, (eff_sg > 1) ? eff_sg : std::max<size_t>(1, dev_cfg.sub_group_size));
    const size_t bs_packed = (std::max<size_t>(1, chunk_bs) + Traits::kLanes - 1) / Traits::kLanes;

    const int wg_lo = static_cast<int>(std::min<size_t>(
        dev_cfg.max_work_group_size,
        std::max<size_t>(sg_unit, std::min<size_t>(64, dev_cfg.max_work_group_size))));
    const int wg_hi = static_cast<int>(std::min<size_t>(
        dev_cfg.max_work_group_size,
        std::max<size_t>(static_cast<size_t>(wg_lo), std::min<size_t>(128, sg_unit * 4))));

    std::vector<int> wg_opts;
    wg_opts.push_back(pow2_floor_int(wg_lo));
    if (pow2_floor_int(wg_hi) != wg_opts.front()) {
        wg_opts.push_back(pow2_floor_int(wg_hi));
    }

    auto add_cand = [&](int wg_i, int bt) {
        const std::string id = "deconv_wg" + std::to_string(wg_i) + "_bt" + std::to_string(bt);
        for (const auto& c : cands) {
            if (c.variant_id == id) return;
        }
        cands.push_back(DeconvCandidate{id, DeconvConfig{wg_i, bt}});
    };

    // Derive batch_tile analytically from max_compute_units * target_waves vs groups_i
    // so each thread evaluates eval_pswf_deconv ONCE and amortizes across many batch items.
    for (int wg_i : wg_opts) {
        const size_t groups_i = (static_cast<size_t>(nout) + static_cast<size_t>(wg_i) - 1) / static_cast<size_t>(wg_i);
        for (size_t waves : {4ULL, 2ULL, 8ULL}) {
            const size_t target_wgs = std::max<size_t>(1, dev_cfg.max_compute_units * waves);
            const size_t groups_b   = std::max<size_t>(1, (target_wgs + groups_i - 1) / groups_i);
            const int raw_bt = static_cast<int>((bs_packed + groups_b - 1) / groups_b);
            const int bt = pow2_clamp_int(raw_bt, 8, 256);
            add_cand(wg_i, bt);
        }
    }

    if (cands.empty()) {
        add_cand(std::max(1, pow2_floor_int(wg_lo)), 32);
    }
    return cands;
}

} // namespace

template <typename Traits>
Nufft1PlanBase<Traits>::Nufft1PlanBase(
    sycl::queue& q,
    int M,
    int N,
    int w,
    PswfMode mode,
    size_t max_bs,
    const Nufft1LaunchConfig& nufft_cfg)
    : q_(q),
      M_(M),
      M_cap_(M),
      N_(N),
      w_(w),
      mode_(mode),
      log_nfft_(validate_and_log_nfft(q, M, N, w, mode)),
      nfft_(1 << log_nfft_),
      nout_(nufft1_nout(N, mode)),
      output_shift_(nufft1_output_shift(N, mode)),
      max_bs_(std::max<size_t>(1, max_bs)),
      chunk_bs_(1),
      packed_stride_(1),
      launch_cfg_(nufft_cfg),
      fft_plan_(q, log_nfft_, std::max<size_t>(1, max_bs), nufft_cfg.fft_cfg) {
    const auto& dev_cfg  = fft_plan_.device_config();
    const size_t eff_sg  = fft_plan_.effective_sg_size();
    const bool is_heavy  = fft_plan_.is_compute_heavy();
    const std::string cache_path = sycl_fft::AutotuneCache::resolve_cache_path(launch_cfg_.fft_cfg.cache_path);
    const auto autotune_mode = launch_cfg_.fft_cfg.autotune;

    precompute_wg_ = dev_cfg.default_padded_wg_size(256);

    // Precompute L2-resident d_weights[M * W] in set_points() when weights fit comfortably in L2
    const size_t weight_bytes = static_cast<size_t>(M_) * static_cast<size_t>(w_) * sizeof(ScalarT);
    const size_t weight_cap = std::min<size_t>(size_t{4} * 1024 * 1024, std::max<size_t>(size_t{256} * 1024, dev_cfg.effective_l2_bytes() / 4));
    use_precomp_weights_ = (weight_bytes <= weight_cap);

    // 1. Build hardware-derived spreading candidates & filter via SYCL 2020 private_mem_size probe (when exposed)
    auto spread_cands = build_spread_candidates<Traits>(
        dev_cfg, eff_sg, nfft_, w_, use_precomp_weights_,
        launch_cfg_.force_collab_spread,
        launch_cfg_.fft_cfg.allow_emulated_local_mem);
    {
        const auto dev = q_.get_device();
        std::vector<SpreadCandidate> filtered;
        filtered.reserve(spread_cands.size());
        for (const auto& c : spread_cands) {
            if (c.cfg.use_smem && c.cfg.use_reg_acc) {
                const auto kinfo = dispatch_probe_spread_reg<Traits>(q_, dev, w_, c.cfg);
                if (kinfo.available) {
                    if (kinfo.max_wg_size > 0 && static_cast<size_t>(c.cfg.wg_size) > kinfo.max_wg_size) {
                        continue;
                    }
                    if (kinfo.private_mem_bytes > 0) {
                        continue; // Reject register-accumulator variant that spills to private memory
                    }
                }
            }
            filtered.push_back(c);
        }
        if (!filtered.empty()) {
            spread_cands = std::move(filtered);
        }
    }
    if (spread_cands.empty()) {
        throw std::runtime_error("Nufft1Plan: no valid spreading candidate for device");
    }

    // Resolve spreading candidate (override -> cache -> heuristic/autotune)
    int chosen_spread = 0;
    bool need_spread_tune = false;
    const bool allow_runtime_tune = (max_bs_ >= 32) || (autotune_mode == sycl_fft::AutotuneMode::Force);
    const int m_bin = (M_ <= 128) ? 0 : ((M_ <= 1024) ? 1 : ((M_ <= 8192) ? 2 : 3));
    const std::string spr_slot = "nufft1_spr2_w" + std::to_string(w_) + "_m" + std::to_string(m_bin);
    const std::string spr_key  = sycl_fft::AutotuneCache::make_key(
        dev_cfg, Traits::kPrecTag, log_nfft_, spr_slot.c_str(), Traits::kSign);

    if (!launch_cfg_.spread_variant_override.empty()) {
        bool found = false;
        for (size_t i = 0; i < spread_cands.size(); ++i) {
            if (spread_cands[i].variant_id == launch_cfg_.spread_variant_override) {
                chosen_spread = static_cast<int>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "Nufft1Plan: requested --nufft-spread '" + launch_cfg_.spread_variant_override +
                "' is not valid for this device/nfft/width");
        }
    } else if (autotune_mode == sycl_fft::AutotuneMode::Auto && !launch_cfg_.has_spread_overrides()) {
        sycl_fft::AutotuneEntry cached{};
        if (sycl_fft::AutotuneCache::get(cache_path, spr_key, cached)) {
            for (size_t i = 0; i < spread_cands.size(); ++i) {
                if (spread_cands[i].variant_id == cached.variant_id) {
                    chosen_spread = static_cast<int>(i);
                    spread_cache_hit_ = true;
                    break;
                }
            }
        }
        if (!spread_cache_hit_ && allow_runtime_tune && spread_cands.size() > 1) {
            need_spread_tune = true;
        }
    } else if (autotune_mode == sycl_fft::AutotuneMode::Force && spread_cands.size() > 1) {
        need_spread_tune = true;
    }

    spread_cfg_ = spread_cands[static_cast<size_t>(chosen_spread)].cfg;
    spread_variant_name_ = spread_cands[static_cast<size_t>(chosen_spread)].variant_id;

    // 2. Compute unified L2-aware chunk_bs coordinated with fft_plan_ (zero hardcoded 16 MiB)
    // Note: fft_plan_.chunk_bs() already accounts for Two-Pass scratch_ L2 budgeting,
    // so only collab spreading (extra transposed grid) multiplies the buffer count here.
    const size_t fft_bytes = static_cast<size_t>(nfft_) * sizeof(ComplexT);
    const size_t buf_mult  = (!spread_cfg_.use_smem && !spread_cfg_.use_cpu_direct) ? 2 : 1;
    const size_t l2_bytes  = (launch_cfg_.chunk_bytes_override > 0)
        ? launch_cfg_.chunk_bytes_override
        : ((launch_cfg_.fft_cfg.l2_chunk_bytes > 0)
               ? launch_cfg_.fft_cfg.l2_chunk_bytes
               : dev_cfg.compute_l2_chunk_bytes(
                     buf_mult * fft_bytes,
                     static_cast<size_t>(std::max(1, spread_cfg_.num_wins)),
                     is_heavy,
                     launch_cfg_.fft_cfg.l2_budget_num,
                     launch_cfg_.fft_cfg.l2_budget_den,
                     launch_cfg_.fft_cfg.min_waves_per_cu));

    chunk_bs_ = std::max<size_t>(1, l2_bytes / std::max<size_t>(1, buf_mult * fft_bytes));
    chunk_bs_ = std::min(max_bs_, chunk_bs_);
    if (fft_plan_.chunk_bs() > 0) {
        chunk_bs_ = std::min(chunk_bs_, fft_plan_.chunk_bs());
    }
    if constexpr (Traits::kLanes == 2) {
        if (chunk_bs_ > 1 && (chunk_bs_ & 1u) != 0u) {
            chunk_bs_ &= ~size_t{1};
        }
    }

    const size_t max_packed = (chunk_bs_ + Traits::kLanes - 1) / Traits::kLanes;
    const size_t align_elems = std::max<size_t>(
        std::max<size_t>(1, dev_cfg.sub_group_size),
        std::max<size_t>(1, dev_cfg.cache_line_size / sizeof(RealT)));
    packed_stride_ = ((max_packed + align_elems - 1) / align_elems) * align_elems;

    // 3. Allocate core device buffers (d_grid_T and d_bucket_* allocated ONLY if actually needed!)
    d_pswf_tbl_ = dispatch_upload_pswf_table<Traits>(q_, w_, mode_, launch_cfg_.use_geo_scale_fp16);
    if (use_precomp_weights_) {
        d_weights_ = sycl_fft::UsmPtr<ScalarT>::alloc_device(
            static_cast<size_t>(M_) * static_cast<size_t>(w_), q_);
        if (!d_weights_) {
            use_precomp_weights_ = false;
        }
    }
    d_first_    = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(M_), q_);
    d_u_        = sycl_fft::UsmPtr<ScalarT>::alloc_device(static_cast<size_t>(M_), q_);
    d_shift_    = sycl_fft::UsmPtr<ComplexT>::alloc_device(static_cast<size_t>(M_), q_);
    d_fft_buf_  = sycl_fft::UsmPtr<ComplexT>::alloc_device(chunk_bs_ * static_cast<size_t>(nfft_), q_);

    if (!d_pswf_tbl_ || !d_first_ || !d_u_ || !d_shift_ || !d_fft_buf_) {
        throw std::runtime_error("Nufft1Plan: core device memory allocation failed");
    }

    // Helper for deterministic non-zero complex/scalar synthetic data (avoids GPU zero-page compression)
    auto make_synth_complex = [](uint64_t seed) -> ComplexT {
        const double r = -0.875 + 1.75 * static_cast<double>((seed * 6364136223846793005ULL + 1ULL) >> 40) / 16777216.0;
        const double i = -0.875 + 1.75 * static_cast<double>(((seed ^ 0x9e3779b97f4a7c15ULL) * 2862933555777941757ULL + 1ULL) >> 40) / 16777216.0;
        if constexpr (Traits::kIsHalf) {
            return sycl_fft::complexh_from_doubles(r, i);
        } else if constexpr (Traits::kIsDf) {
            return sycl_fft::Complexdf{sycl_fft::df_from_double(r), sycl_fft::df_from_double(i)};
        } else if constexpr (Traits::kIsFp64) {
            return sycl_fft::Complex(r, i);
        } else {
            return sycl_fft::Complexf(static_cast<float>(r), static_cast<float>(i));
        }
    };
    auto make_synth_scalar = [](uint64_t seed) -> ScalarT {
        const double v = -0.9 + 1.8 * static_cast<double>((seed * 2654435761ULL) & 0xFFFFu) / 65535.0;
        if constexpr (Traits::kIsDf) {
            return sycl_fft::df_from_double(v);
        } else {
            return static_cast<ScalarT>(v);
        }
    };

    // 4. Factored two-phase benchmark-based spreading autotuner (mirrors DifFftPlan):
    //    Phase A: benchmark 1 representative per distinct (use_reg_acc, win_elems, t_j) family
    //             across hardware-feasible reg & smem candidates (collab only on CPU/forced).
    //    Phase B: on the winning family only, refine wg_size across alternate feasible WGs
    //             (0 new kernel JITs, reuses cached window buckets for that win_elems).
    if (need_spread_tune) {
        struct FamilyKey {
            bool use_smem;
            bool use_reg_acc;
            bool use_cpu_direct;
            int  win_elems;
            int  t_j;
            bool operator==(const FamilyKey& o) const {
                return use_smem == o.use_smem && use_reg_acc == o.use_reg_acc &&
                       use_cpu_direct == o.use_cpu_direct && win_elems == o.win_elems && t_j == o.t_j;
            }
        };
        auto family_of = [](const SpreadConfig& sc) -> FamilyKey {
            return FamilyKey{sc.use_smem, sc.use_reg_acc, sc.use_cpu_direct, sc.win_elems, sc.t_j};
        };

        std::vector<size_t> phase_a;
        std::vector<FamilyKey> seen_families;
        auto try_add_family = [&](size_t idx) -> bool {
            const FamilyKey fk = family_of(spread_cands[idx].cfg);
            if (std::find(seen_families.begin(), seen_families.end(), fk) != seen_families.end()) {
                return false;
            }
            seen_families.push_back(fk);
            phase_a.push_back(idx);
            return true;
        };

        // Pick top-scored representative from each distinct hardware-feasible family:
        // up to 4 register-resident families (covers widened Q in {2,4,8,16,32}) + up to 3 SMEM-grid families.
        size_t n_reg = 0, n_smem = 0;
        for (size_t i = 0; i < spread_cands.size() && n_reg < 4; ++i) {
            if (spread_cands[i].cfg.use_smem && spread_cands[i].cfg.use_reg_acc) {
                if (try_add_family(i)) ++n_reg;
            }
        }
        // Guarantee at least one wide-window SMEM family (win >= 128) if available
        for (size_t i = 0; i < spread_cands.size() && n_smem < 3; ++i) {
            if (spread_cands[i].cfg.use_smem && !spread_cands[i].cfg.use_reg_acc &&
                spread_cands[i].cfg.win_elems >= 128) {
                if (try_add_family(i)) { ++n_smem; break; }
            }
        }
        for (size_t i = 0; i < spread_cands.size() && n_smem < 3; ++i) {
            if (spread_cands[i].cfg.use_smem && !spread_cands[i].cfg.use_reg_acc) {
                if (try_add_family(i)) ++n_smem;
            }
        }
        if (!dev_cfg.has_native_local_mem || phase_a.empty() || launch_cfg_.force_collab_spread) {
            for (size_t i = 0; i < spread_cands.size(); ++i) {
                if (!spread_cands[i].cfg.use_smem) {
                    if (try_add_family(i)) break;
                }
            }
        }

        if (!phase_a.empty() && spread_cands.size() > 1) {
            const size_t tune_bs = std::min(chunk_bs_, std::max<size_t>(64, chunk_bs_));
            auto d_tune_y = sycl_fft::UsmPtr<ComplexT>::alloc_device(tune_bs * static_cast<size_t>(M_), q_);
            if (d_tune_y) {
                // Populate deterministic non-zero synthetic inputs for realistic timing
                const size_t pat_elems = std::min<size_t>(tune_bs * static_cast<size_t>(M_), size_t{4096});
                std::vector<ComplexT> h_pat(pat_elems);
                for (size_t k = 0; k < pat_elems; ++k) {
                    h_pat[k] = make_synth_complex(k + 17ULL);
                }
                const size_t total_tune_y = tune_bs * static_cast<size_t>(M_);
                for (size_t off = 0; off < total_tune_y; off += pat_elems) {
                    const size_t cnt = std::min(pat_elems, total_tune_y - off);
                    q_.memcpy(d_tune_y.get() + off, h_pat.data(), cnt * sizeof(ComplexT));
                }
                std::vector<int> h_first(static_cast<size_t>(M_));
                std::vector<ScalarT> h_u(static_cast<size_t>(M_));
                std::vector<ComplexT> h_shift(static_cast<size_t>(M_));
                for (int m = 0; m < M_; ++m) {
                    h_first[static_cast<size_t>(m)] = static_cast<int>((static_cast<uint64_t>(m) * 2654435761ULL) & static_cast<uint64_t>(nfft_ - 1));
                    h_u[static_cast<size_t>(m)]     = make_synth_scalar(static_cast<uint64_t>(m) + 3ULL);
                    h_shift[static_cast<size_t>(m)] = make_synth_complex(static_cast<uint64_t>(m) + 101ULL);
                }
                q_.memcpy(d_first_.get(), h_first.data(), static_cast<size_t>(M_) * sizeof(int));
                q_.memcpy(d_u_.get(), h_u.data(), static_cast<size_t>(M_) * sizeof(ScalarT));
                q_.memcpy(d_shift_.get(), h_shift.data(), static_cast<size_t>(M_) * sizeof(ComplexT));
                if (use_precomp_weights_ && d_weights_) {
                    dispatch_precompute_weights<Traits>(
                        q_, w_, d_pswf_tbl_.get(), d_u_.get(), d_weights_.get(), M_, precompute_wg_);
                }
                q_.wait();

                struct BucketTmp {
                    sycl_fft::UsmPtr<int>      cnt, off, m, bf;
                    sycl_fft::UsmPtr<ComplexT> bs;
                    sycl_fft::UsmPtr<ScalarT>  bw;
                };
                std::unordered_map<int, BucketTmp> bucket_cache;
                const ScalarT* tune_w_ptr = use_precomp_weights_ ? d_weights_.get() : nullptr;

                auto get_buckets = [&](int win_elems, int num_wins) -> const BucketTmp* {
                    auto it = bucket_cache.find(win_elems);
                    if (it != bucket_cache.end()) return &it->second;
                    BucketTmp bt{};
                    const size_t max_ent = size_t{2} * static_cast<size_t>(M_);
                    bt.cnt = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(num_wins), q_);
                    bt.off = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(num_wins + 1), q_);
                    bt.m   = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
                    if (!bt.cnt || !bt.off || !bt.m) return nullptr;
                    detail::launch_build_window_buckets(
                        q_, d_first_.get(), M_, nfft_, w_, win_elems, num_wins,
                        bt.cnt.get(), bt.off.get(), bt.m.get(), precompute_wg_);
                    if (use_precomp_weights_ && tune_w_ptr) {
                        bt.bf = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
                        bt.bs = sycl_fft::UsmPtr<ComplexT>::alloc_device(max_ent, q_);
                        bt.bw = sycl_fft::UsmPtr<ScalarT>::alloc_device(max_ent * static_cast<size_t>(w_), q_);
                        if (!bt.bf || !bt.bs || !bt.bw) return nullptr;
                        dispatch_pack_bucket_geom<Traits>(
                            q_, w_, num_wins, static_cast<int>(max_ent),
                            bt.off.get(), bt.m.get(),
                            d_first_.get(), d_shift_.get(), tune_w_ptr,
                            bt.bf.get(), bt.bs.get(), bt.bw.get(), precompute_wg_);
                    }
                    q_.wait();
                    auto [ins_it, _] = bucket_cache.emplace(win_elems, std::move(bt));
                    return &ins_it->second;
                };

                // Pre-build window buckets & optional collab scratch BEFORE interleaved timing
                sycl_fft::UsmPtr<RealT> collab_re, collab_im;
                for (size_t idx : phase_a) {
                    const auto& sc = spread_cands[idx].cfg;
                    if (sc.use_smem && sc.num_wins > 1) {
                        (void)get_buckets(sc.win_elems, sc.num_wins);
                    } else if (!sc.use_smem && !sc.use_cpu_direct && !collab_re) {
                        collab_re = sycl_fft::UsmPtr<RealT>::alloc_device(static_cast<size_t>(nfft_) * packed_stride_, q_);
                        collab_im = sycl_fft::UsmPtr<RealT>::alloc_device(static_cast<size_t>(nfft_) * packed_stride_, q_);
                        if (collab_re && collab_im) {
                            q_.memset(collab_re.get(), 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
                            q_.memset(collab_im.get(), 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
                            q_.wait();
                        }
                    }
                }

                auto launch_spread_cand = [&](size_t idx) -> bool {
                    const auto& cand = spread_cands[idx];
                    if (cand.cfg.use_smem) {
                        const int* off_arg = nullptr;
                        const int* m_arg = nullptr;
                        const int* first_arg = d_first_.get();
                        const ComplexT* shift_arg = d_shift_.get();
                        const ScalarT* weight_arg = tune_w_ptr;
                        if (cand.cfg.num_wins > 1) {
                            auto it = bucket_cache.find(cand.cfg.win_elems);
                            if (it == bucket_cache.end()) return false;
                            const BucketTmp* bt = &it->second;
                            off_arg = bt->off.get();
                            m_arg   = bt->m.get();
                            if (bt->bf && bt->bs && bt->bw) {
                                first_arg  = bt->bf.get();
                                shift_arg  = bt->bs.get();
                                weight_arg = bt->bw.get();
                            }
                        }
                        dispatch_spread_smem<Traits>(
                            q_, w_, d_pswf_tbl_.get(), weight_arg, first_arg, d_u_.get(), shift_arg,
                            off_arg, m_arg,
                            d_tune_y.get(), d_fft_buf_.get(), M_, nfft_, static_cast<size_t>(M_), tune_bs,
                            mode_, cand.cfg);
                        return true;
                    } else if (cand.cfg.use_cpu_direct) {
                        dispatch_spread_cpu_direct<Traits>(
                            q_, w_, d_pswf_tbl_.get(), tune_w_ptr, d_first_.get(), d_u_.get(), d_shift_.get(),
                            d_tune_y.get(), d_fft_buf_.get(), M_, nfft_, static_cast<size_t>(M_), tune_bs);
                        return true;
                    } else {
                        if (!collab_re || !collab_im) return false;
                        dispatch_spread_collab<Traits>(
                            q_, w_, d_pswf_tbl_.get(), tune_w_ptr, d_first_.get(), d_u_.get(), d_shift_.get(), d_tune_y.get(),
                            collab_re.get(), collab_im.get(),
                            M_, nfft_, static_cast<size_t>(M_), tune_bs, packed_stride_, mode_, cand.cfg);
                        dispatch_transpose_and_zero<Traits>(
                            q_, collab_re.get(), collab_im.get(), d_fft_buf_.get(),
                            nfft_, tune_bs, packed_stride_, cand.cfg);
                        return true;
                    }
                };
                auto sync_q = [&]() { q_.wait_and_throw(); };

                // Phase A: interleaved (ABBA) family benchmark with conservative noise-band preference
                // Conservatism order: SMEM-grid (!use_reg_acc) > smaller Q > smaller SMEM footprint.
                auto more_conservative_a = [&](size_t a_pos, size_t b_pos) -> bool {
                    const auto& ca = spread_cands[phase_a[a_pos]];
                    const auto& cb = spread_cands[phase_a[b_pos]];
                    if (ca.cfg.use_reg_acc != cb.cfg.use_reg_acc) {
                        return !ca.cfg.use_reg_acc;
                    }
                    if (ca.cfg.use_reg_acc) {
                        const int qa = ca.cfg.win_elems / std::max(1, ca.cfg.t_j);
                        const int qb = cb.cfg.win_elems / std::max(1, cb.cfg.t_j);
                        if (qa != qb) return qa < qb;
                    }
                    if (ca.smem_bytes != cb.smem_bytes) return ca.smem_bytes < cb.smem_bytes;
                    return ca.score > cb.score;
                };
                const auto res_a = sycl_fft::run_interleaved_bench(
                    phase_a.size(),
                    [&](size_t pos) { return launch_spread_cand(phase_a[pos]); },
                    sync_q,
                    more_conservative_a,
                    /*num_rounds=*/5,
                    /*target_slice_ms=*/2.0,
                    /*max_batch_reps=*/16);

                int best_idx = static_cast<int>(phase_a[0]);
                double recorded_margin_pct = 0.0;
                std::string recorded_runner_up;
                bool tune_ok = res_a.ok;

                if (res_a.ok) {
                    best_idx = static_cast<int>(phase_a[res_a.chosen_idx]);
                    recorded_margin_pct = res_a.margin_pct;
                    if (res_a.has_runner_up) {
                        recorded_runner_up = spread_cands[phase_a[res_a.runner_up_idx]].variant_id;
                    }

                    // Phase B: interleaved (ABBA) wg_size refinement within the winning family (0 new JITs)
                    if (spread_cands[static_cast<size_t>(best_idx)].cfg.use_smem) {
                        const FamilyKey win_fam = family_of(spread_cands[static_cast<size_t>(best_idx)].cfg);
                        std::vector<size_t> phase_b;
                        phase_b.push_back(static_cast<size_t>(best_idx));
                        for (size_t i = 0; i < spread_cands.size(); ++i) {
                            if (static_cast<int>(i) == best_idx) continue;
                            if (family_of(spread_cands[i].cfg) == win_fam) {
                                phase_b.push_back(i);
                            }
                        }
                        if (phase_b.size() > 1) {
                            const int target_cons_wg = static_cast<int>(
                                dev_cfg.default_wg_size(256, 128));
                            auto more_conservative_b = [&](size_t a_pos, size_t b_pos) -> bool {
                                const auto& ca = spread_cands[phase_b[a_pos]];
                                const auto& cb = spread_cands[phase_b[b_pos]];
                                const int da = std::abs(ca.cfg.wg_size - target_cons_wg);
                                const int db = std::abs(cb.cfg.wg_size - target_cons_wg);
                                if (da != db) return da < db;
                                if (ca.smem_bytes != cb.smem_bytes) return ca.smem_bytes < cb.smem_bytes;
                                return ca.score > cb.score;
                            };
                            const auto res_b = sycl_fft::run_interleaved_bench(
                                phase_b.size(),
                                [&](size_t pos) { return launch_spread_cand(phase_b[pos]); },
                                sync_q,
                                more_conservative_b,
                                /*num_rounds=*/5,
                                /*target_slice_ms=*/2.0,
                                /*max_batch_reps=*/16);
                            if (res_b.ok) {
                                best_idx = static_cast<int>(phase_b[res_b.chosen_idx]);
                                // Record the tighter margin between Phase A (family) and Phase B (wg)
                                if (res_b.has_runner_up &&
                                    (!res_a.has_runner_up || std::abs(res_b.margin_pct) < std::abs(recorded_margin_pct))) {
                                    recorded_margin_pct = res_b.margin_pct;
                                    recorded_runner_up = spread_cands[phase_b[res_b.runner_up_idx]].variant_id;
                                }
                            }
                        }
                    }
                }

                chosen_spread = best_idx;
                spread_cfg_ = spread_cands[static_cast<size_t>(chosen_spread)].cfg;
                spread_variant_name_ = spread_cands[static_cast<size_t>(chosen_spread)].variant_id;

                if (!launch_cfg_.has_spread_overrides() && tune_ok) {
                    sycl_fft::AutotuneEntry ent{};
                    ent.mode = sycl_fft::PlanExecMode::SinglePass;
                    ent.variant_id = spread_variant_name_;
                    ent.margin_pct = recorded_margin_pct;
                    ent.runner_up_id = recorded_runner_up;
                    sycl_fft::AutotuneCache::put(cache_path, spr_key, ent);
                }
            }
        }
    }

    // Allocate ONLY the auxiliary buffers required by the winning spreading strategy!
    if (spread_cfg_.use_smem) {
        if (spread_cfg_.num_wins > 1) {
            allocate_window_buckets();
        }
    } else if (!spread_cfg_.use_cpu_direct) {
        allocate_collab_grid();
    }

    // 5. Build and resolve hardware-derived deconvolution candidates.
    // On SinglePass / TwoPass GPU plans, SlotBitReversed & SlotBitReversedRaw fuse directly
    // into the FFT final store, so skip standalone deconv autotuning unless MultiStageGlobal
    // or explicitly overridden (saves 3 cold JIT compilations per precision).
    const bool deconv_fused_primary = !fft_plan_.is_multistage() && !launch_cfg_.has_deconv_overrides();
    const size_t rep_bs = std::max<size_t>(
        chunk_bs_,
        std::clamp<size_t>(dev_cfg.effective_l2_bytes() / std::max<size_t>(1, 4 * fft_bytes), 64, 2048));
    const auto deconv_cands = build_deconv_candidates<Traits>(dev_cfg, eff_sg, nout_, rep_bs);
    int chosen_deconv = 0;
    bool need_deconv_tune = false;
    const std::string dec_slot =
        "nufft1_dec_w" + std::to_string(w_) + ((mode_ == PswfMode::PSWF43) ? "_43" : "_21");
    const std::string dec_key = sycl_fft::AutotuneCache::make_key(
        dev_cfg, Traits::kPrecTag, log_nfft_, dec_slot.c_str(), Traits::kSign);

    if (!launch_cfg_.deconv_variant_override.empty()) {
        bool found = false;
        for (size_t i = 0; i < deconv_cands.size(); ++i) {
            if (deconv_cands[i].variant_id == launch_cfg_.deconv_variant_override) {
                chosen_deconv = static_cast<int>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "Nufft1Plan: requested --nufft-deconv '" + launch_cfg_.deconv_variant_override +
                "' is not valid for this device/nout");
        }
    } else if (!deconv_fused_primary && autotune_mode == sycl_fft::AutotuneMode::Auto && !launch_cfg_.has_deconv_overrides()) {
        sycl_fft::AutotuneEntry cached{};
        if (sycl_fft::AutotuneCache::get(cache_path, dec_key, cached)) {
            for (size_t i = 0; i < deconv_cands.size(); ++i) {
                if (deconv_cands[i].variant_id == cached.variant_id) {
                    chosen_deconv = static_cast<int>(i);
                    deconv_cache_hit_ = true;
                    break;
                }
            }
        }
        if (!deconv_cache_hit_ && allow_runtime_tune && deconv_cands.size() > 1) {
            need_deconv_tune = true;
        }
    } else if (!deconv_fused_primary && autotune_mode == sycl_fft::AutotuneMode::Force && deconv_cands.size() > 1) {
        need_deconv_tune = true;
    }

    if (need_deconv_tune) {
        const size_t lim = std::min<size_t>(deconv_cands.size(), 3);
        auto d_tune_out = sycl_fft::UsmPtr<ComplexT>::alloc_device(chunk_bs_ * static_cast<size_t>(nout_), q_);
        if (d_tune_out) {
            const size_t total_fft = chunk_bs_ * static_cast<size_t>(nfft_);
            const size_t pat_elems = std::min<size_t>(total_fft, size_t{4096});
            std::vector<ComplexT> h_pat(pat_elems);
            for (size_t k = 0; k < pat_elems; ++k) {
                h_pat[k] = make_synth_complex(k + 53ULL);
            }
            for (size_t off = 0; off < total_fft; off += pat_elems) {
                const size_t cnt = std::min(pat_elems, total_fft - off);
                q_.memcpy(d_fft_buf_.get() + off, h_pat.data(), cnt * sizeof(ComplexT));
            }
            q_.wait();
            const auto dec_res = sycl_fft::run_interleaved_bench(
                lim,
                [&](size_t i) {
                    dispatch_deconv<Traits>(
                        q_, w_, d_pswf_tbl_.get(), d_fft_buf_.get(), d_tune_out.get(), nfft_, log_nfft_, nout_, output_shift_,
                        static_cast<size_t>(nout_), chunk_bs_, mode_, NufftOutOrder::SlotBitReversed,
                        deconv_cands[i].cfg, output_scale_);
                    return true;
                },
                [&]() { q_.wait_and_throw(); },
                [](size_t a, size_t b) { return a < b; },
                /*num_rounds=*/5,
                /*target_slice_ms=*/2.0,
                /*max_batch_reps=*/16);
            if (dec_res.ok) {
                chosen_deconv = static_cast<int>(dec_res.chosen_idx);
                if (!launch_cfg_.has_deconv_overrides()) {
                    sycl_fft::AutotuneEntry ent{};
                    ent.mode = sycl_fft::PlanExecMode::SinglePass;
                    ent.variant_id = deconv_cands[static_cast<size_t>(chosen_deconv)].variant_id;
                    ent.margin_pct = dec_res.margin_pct;
                    if (dec_res.has_runner_up) {
                        ent.runner_up_id = deconv_cands[dec_res.runner_up_idx].variant_id;
                    }
                    sycl_fft::AutotuneCache::put(cache_path, dec_key, ent);
                }
            }
        }
    }

    deconv_cfg_ = deconv_cands[static_cast<size_t>(chosen_deconv)].cfg;
    deconv_variant_name_ = deconv_fused_primary
        ? std::string("fused_fft_epilogue")
        : deconv_cands[static_cast<size_t>(chosen_deconv)].variant_id;
}

template <typename Traits>
void Nufft1PlanBase<Traits>::allocate_collab_grid() {
    if (d_grid_T_re_ && d_grid_T_im_) return;
    auto re = sycl_fft::UsmPtr<RealT>::alloc_device(static_cast<size_t>(nfft_) * packed_stride_, q_);
    auto im = sycl_fft::UsmPtr<RealT>::alloc_device(static_cast<size_t>(nfft_) * packed_stride_, q_);
    if (!re || !im) {
        throw std::runtime_error("Nufft1Plan: collaborative grid allocation failed");
    }
    q_.memset(re.get(), 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.memset(im.get(), 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.wait();
    d_grid_T_re_ = std::move(re);
    d_grid_T_im_ = std::move(im);
}

template <typename Traits>
void Nufft1PlanBase<Traits>::allocate_window_buckets() {
    if (d_bucket_counts_ && d_bucket_off_ && d_bucket_m_) return;
    const size_t max_ent = size_t{2} * static_cast<size_t>(M_cap_);
    auto cnt = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(spread_cfg_.num_wins), q_);
    auto off = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(spread_cfg_.num_wins + 1), q_);
    auto bm  = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
    if (!cnt || !off || !bm) {
        throw std::runtime_error("Nufft1Plan: window bucket allocation failed");
    }
    if (use_precomp_weights_ && d_weights_) {
        auto bf = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
        auto bs = sycl_fft::UsmPtr<ComplexT>::alloc_device(max_ent, q_);
        auto bw = sycl_fft::UsmPtr<ScalarT>::alloc_device(max_ent * static_cast<size_t>(w_), q_);
        if (!bf || !bs || !bw) {
            throw std::runtime_error("Nufft1Plan: bucket geometry allocation failed");
        }
        d_b_first_   = std::move(bf);
        d_b_shift_   = std::move(bs);
        d_b_weights_ = std::move(bw);
    }
    d_bucket_counts_ = std::move(cnt);
    d_bucket_off_    = std::move(off);
    d_bucket_m_      = std::move(bm);
}

template <typename Traits>
void Nufft1PlanBase<Traits>::ensure_point_capacity(int M) {
    if (M <= 0) {
        throw std::invalid_argument("Nufft1Plan::set_points: M must be positive");
    }
    if (M <= M_cap_) {
        M_ = M;
        precompute_wg_ = fft_plan_.device_config().default_padded_wg_size(256);
        return;
    }
    const int new_cap = std::max(M, M_cap_ + (M_cap_ >> 1));
    q_.wait();
    auto new_first = sycl_fft::UsmPtr<int>::alloc_device(static_cast<size_t>(new_cap), q_);
    auto new_u     = sycl_fft::UsmPtr<ScalarT>::alloc_device(static_cast<size_t>(new_cap), q_);
    auto new_shift = sycl_fft::UsmPtr<ComplexT>::alloc_device(static_cast<size_t>(new_cap), q_);
    if (!new_first || !new_u || !new_shift) {
        throw std::runtime_error("Nufft1Plan: point buffer reallocation failed");
    }
    sycl_fft::UsmPtr<ScalarT> new_weights;
    if (use_precomp_weights_) {
        new_weights = sycl_fft::UsmPtr<ScalarT>::alloc_device(
            static_cast<size_t>(new_cap) * static_cast<size_t>(w_), q_);
        if (!new_weights) {
            throw std::runtime_error("Nufft1Plan: weights buffer reallocation failed");
        }
    }
    if (spread_cfg_.use_smem && spread_cfg_.num_wins > 1) {
        const size_t max_ent = size_t{2} * static_cast<size_t>(new_cap);
        auto new_bm = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
        if (!new_bm) {
            throw std::runtime_error("Nufft1Plan: bucket_m reallocation failed");
        }
        sycl_fft::UsmPtr<int> new_bf;
        sycl_fft::UsmPtr<ComplexT> new_bs;
        sycl_fft::UsmPtr<ScalarT> new_bw;
        if (use_precomp_weights_) {
            new_bf = sycl_fft::UsmPtr<int>::alloc_device(max_ent, q_);
            new_bs = sycl_fft::UsmPtr<ComplexT>::alloc_device(max_ent, q_);
            new_bw = sycl_fft::UsmPtr<ScalarT>::alloc_device(max_ent * static_cast<size_t>(w_), q_);
            if (!new_bf || !new_bs || !new_bw) {
                throw std::runtime_error("Nufft1Plan: bucket geometry reallocation failed");
            }
        }
        d_bucket_m_  = std::move(new_bm);
        d_b_first_   = std::move(new_bf);
        d_b_shift_   = std::move(new_bs);
        d_b_weights_ = std::move(new_bw);
    }
    d_first_   = std::move(new_first);
    d_u_       = std::move(new_u);
    d_shift_   = std::move(new_shift);
    if (use_precomp_weights_) {
        d_weights_ = std::move(new_weights);
    }
    M_cap_ = new_cap;
    M_ = M;
    precompute_wg_ = fft_plan_.device_config().default_padded_wg_size(256);
}

template <typename Traits>
Nufft1PlanBase<Traits>::~Nufft1PlanBase() {
    try {
        q_.wait();
    } catch (...) {
    }
}

template <typename Traits>
void Nufft1PlanBase<Traits>::set_points(
    const XCoordT* d_x, int M, XCoordT df, int freq_factor) {
    ensure_point_capacity(M);
    set_points(d_x, df, freq_factor);
}

template <typename Traits>
void Nufft1PlanBase<Traits>::set_points(
    const XCoordT* d_x, XCoordT df, int freq_factor) {
    if (freq_factor <= 0) {
        throw std::invalid_argument("Nufft1Plan::set_points: freq_factor must be >= 1");
    }
    XCoordT df_scaled{};
    if constexpr (Traits::kIsFp64) {
        df_scaled = df * static_cast<double>(freq_factor);
    } else {
        df_scaled = sycl_fft::df_mul_f(df, static_cast<float>(freq_factor));
    }

    dispatch_precompute<Traits>(
        q_, d_x, df_scaled, M_, nfft_, w_, output_shift_,
        mode_ == PswfMode::PSWF43, Traits::kPosSign,
        d_first_.get(), d_u_.get(), d_shift_.get(), precompute_wg_);

    if (use_precomp_weights_ && d_weights_) {
        dispatch_precompute_weights<Traits>(
            q_, w_, d_pswf_tbl_.get(), d_u_.get(), d_weights_.get(), M_, precompute_wg_);
    }

    if (spread_cfg_.use_smem && spread_cfg_.num_wins > 1) {
        detail::launch_build_window_buckets(
            q_, d_first_.get(), M_, nfft_, w_,
            spread_cfg_.win_elems, spread_cfg_.num_wins,
            d_bucket_counts_.get(), d_bucket_off_.get(), d_bucket_m_.get(),
            precompute_wg_);
        if (use_precomp_weights_ && d_weights_ && d_b_first_ && d_b_shift_ && d_b_weights_) {
            dispatch_pack_bucket_geom<Traits>(
                q_, w_, spread_cfg_.num_wins, 2 * M_,
                d_bucket_off_.get(), d_bucket_m_.get(),
                d_first_.get(), d_shift_.get(), d_weights_.get(),
                d_b_first_.get(), d_b_shift_.get(), d_b_weights_.get(),
                precompute_wg_);
        }
    }
}

template <typename Traits>
void Nufft1PlanBase<Traits>::set_output_scale(double scale) {
    if (!std::isfinite(scale) || scale == 0.0) {
        throw std::invalid_argument("Nufft1Plan::set_output_scale: scale must be finite and non-zero");
    }
    if (scale == output_scale_) return;
    output_scale_ = scale;
    // Rebuild already-allocated LUTs in place on the in-order queue (no blocking q_.wait()!)
    if (d_slot_deconv_) {
        dispatch_build_deconv_table<Traits>(
            q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
            mode_, 1, d_slot_deconv_.get(), output_scale_);
    }
    if (d_nat_deconv_) {
        dispatch_build_deconv_table<Traits>(
            q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
            mode_, 2, d_nat_deconv_.get(), output_scale_);
    }
    if (d_slot_deconv_f32_) {
        if constexpr (Traits::kIsHalf) {
            detail::launch_build_deconv_tableh_f32(
                q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
                mode_, 1, d_slot_deconv_f32_.get(), output_scale_);
        }
    }
}

template <typename Traits>
const typename Traits::ScalarT* Nufft1PlanBase<Traits>::d_slot_phi_hat() {
    if (!d_slot_phi_hat_) {
        auto buf = sycl_fft::UsmPtr<ScalarT>::alloc_device(static_cast<size_t>(nout_), q_);
        if (!buf) {
            throw std::runtime_error("Nufft1Plan: d_slot_phi_hat allocation failed");
        }
        dispatch_build_deconv_table<Traits>(
            q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
            mode_, 0, buf.get(), 1.0);
        d_slot_phi_hat_ = std::move(buf);
    }
    return d_slot_phi_hat_.get();
}

template <typename Traits>
const typename Traits::ScalarT* Nufft1PlanBase<Traits>::d_slot_deconv() {
    if (!d_slot_deconv_) {
        auto buf = sycl_fft::UsmPtr<ScalarT>::alloc_device(static_cast<size_t>(nout_), q_);
        if (!buf) {
            throw std::runtime_error("Nufft1Plan: d_slot_deconv allocation failed");
        }
        dispatch_build_deconv_table<Traits>(
            q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
            mode_, 1, buf.get(), output_scale_);
        d_slot_deconv_ = std::move(buf);
    }
    return d_slot_deconv_.get();
}

template <typename Traits>
const typename Traits::ScalarT* Nufft1PlanBase<Traits>::d_deconv() {
    if (!d_nat_deconv_) {
        auto buf = sycl_fft::UsmPtr<ScalarT>::alloc_device(static_cast<size_t>(nout_), q_);
        if (!buf) {
            throw std::runtime_error("Nufft1Plan: d_deconv allocation failed");
        }
        dispatch_build_deconv_table<Traits>(
            q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
            mode_, 2, buf.get(), output_scale_);
        d_nat_deconv_ = std::move(buf);
    }
    return d_nat_deconv_.get();
}

template <typename Traits>
const float* Nufft1PlanBase<Traits>::d_slot_deconv_f32() {
    if constexpr (Traits::kIsHalf) {
        if (!d_slot_deconv_f32_) {
            auto buf = sycl_fft::UsmPtr<float>::alloc_device(static_cast<size_t>(nout_), q_);
            if (!buf) {
                throw std::runtime_error("Nufft1Plan: d_slot_deconv_f32 allocation failed");
            }
            detail::launch_build_deconv_tableh_f32(
                q_, w_, d_pswf_tbl_.get(), nfft_, log_nfft_, nout_, output_shift_,
                mode_, 1, buf.get(), output_scale_);
            d_slot_deconv_f32_ = std::move(buf);
        }
        return d_slot_deconv_f32_.get();
    } else if constexpr (std::is_same_v<ScalarT, float>) {
        return d_slot_deconv();
    } else {
        return nullptr;
    }
}

template <typename Traits>
void Nufft1PlanBase<Traits>::execute(
    const ComplexT* d_y,
    ComplexT* d_out,
    size_t bs,
    NufftOutOrder order) {
    if (bs == 0) return;

    const size_t total_M_stride    = static_cast<size_t>(M_);
    const size_t total_nout_stride = static_cast<size_t>(nout_);
    const ScalarT* w_ptr           = use_precomp_weights_ ? d_weights_.get() : nullptr;

    const bool use_fused_epilogue =
        !fft_plan_.is_multistage() &&
        !launch_cfg_.has_deconv_overrides() &&
        (order == NufftOutOrder::SlotBitReversed || order == NufftOutOrder::SlotBitReversedRaw);
    const int compact_mode = (mode_ == PswfMode::PSWF43) ? 2 : 1;
    const ScalarT* fused_deconv_ptr =
        (use_fused_epilogue && order == NufftOutOrder::SlotBitReversed) ? d_slot_deconv() : nullptr;

    for (size_t b_off = 0; b_off < bs; b_off += chunk_bs_) {
        const size_t cur_bs = std::min(chunk_bs_, bs - b_off);
        const ComplexT* cur_y = d_y + b_off * total_M_stride;
        ComplexT* cur_out     = d_out + b_off * total_nout_stride;

        // 1. Hardware-generalized piecewise minimax PSWF spreading
        if (spread_cfg_.use_smem) {
            const bool use_packed_geom =
                (spread_cfg_.num_wins > 1) && use_precomp_weights_ &&
                d_b_first_ && d_b_shift_ && d_b_weights_;
            const int* first_arg        = use_packed_geom ? d_b_first_.get()   : d_first_.get();
            const ComplexT* shift_arg   = use_packed_geom ? d_b_shift_.get()   : d_shift_.get();
            const ScalarT* weight_arg   = use_packed_geom ? d_b_weights_.get() : w_ptr;
            dispatch_spread_smem<Traits>(
                q_, w_, d_pswf_tbl_.get(), weight_arg, first_arg, d_u_.get(), shift_arg,
                d_bucket_off_.get(), d_bucket_m_.get(),
                cur_y, d_fft_buf_.get(), M_, nfft_, total_M_stride, cur_bs, mode_, spread_cfg_);
        } else if (spread_cfg_.use_cpu_direct) {
            dispatch_spread_cpu_direct<Traits>(
                q_, w_, d_pswf_tbl_.get(), w_ptr, d_first_.get(), d_u_.get(), d_shift_.get(),
                cur_y, d_fft_buf_.get(), M_, nfft_, total_M_stride, cur_bs);
        } else {
            dispatch_spread_collab<Traits>(
                q_, w_, d_pswf_tbl_.get(), w_ptr, d_first_.get(), d_u_.get(), d_shift_.get(), cur_y,
                d_grid_T_re_.get(), d_grid_T_im_.get(),
                M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_, spread_cfg_);
            dispatch_transpose_and_zero<Traits>(
                q_, d_grid_T_re_.get(), d_grid_T_im_.get(), d_fft_buf_.get(),
                nfft_, cur_bs, packed_stride_, spread_cfg_);
        }

        // 2 & 3. Raw DIF Radix-2 FFT + fused compact-slot & deconvolution epilogue
        if (use_fused_epilogue) {
            fft_plan_.execute_raw_compact(
                d_fft_buf_.get(), cur_out, cur_bs, compact_mode, fused_deconv_ptr);
        } else {
            fft_plan_.execute_raw(d_fft_buf_.get(), d_fft_buf_.get(), cur_bs);
            dispatch_deconv<Traits>(
                q_, w_, d_pswf_tbl_.get(), d_fft_buf_.get(), cur_out, nfft_, log_nfft_, nout_, output_shift_,
                total_nout_stride, cur_bs, mode_, order, deconv_cfg_, output_scale_);
        }
    }
}

// Explicit instantiations for enabled signs across all 4 precisions (pure host code, 0 KiB HCF)
#if SYCL_FFT_ENABLE_POS_SIGN
template class Nufft1PlanBase<detail::HalfPlanTraits<+1>>;
template class Nufft1PlanBase<detail::FloatPlanTraits<+1>>;
template class Nufft1PlanBase<detail::DoublePlanTraits<+1>>;
template class Nufft1PlanBase<detail::DfPlanTraits<+1>>;
#endif
#if SYCL_FFT_ENABLE_NEG_SIGN
template class Nufft1PlanBase<detail::HalfPlanTraits<-1>>;
template class Nufft1PlanBase<detail::FloatPlanTraits<-1>>;
template class Nufft1PlanBase<detail::DoublePlanTraits<-1>>;
template class Nufft1PlanBase<detail::DfPlanTraits<-1>>;
#endif

} // namespace sycl_nufft
