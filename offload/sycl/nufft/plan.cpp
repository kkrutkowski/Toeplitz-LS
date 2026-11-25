// Pure-Host NuFFT-1 Plan Orchestration Unit (ZERO SYCL device kernels / 0 KiB HCF).
// Derives all spreading, transposition, deconvolution, and L2-chunking parameters dynamically
// from DeviceFftConfig + BankModel + persistent AutotuneCache (zero hardcoded GPU constants).
#define SYCL_FFT_EXTERN_DEVICE 1
#define SYCL_FFT_USE_EXTERN_TEMPLATES 1
#define SYCL_NUFFT_PLAN_IMPLEMENTATION_TU 1
#include "nufft/nufft1.hpp"

#include <chrono>
#include <limits>
#include <vector>

namespace sycl_nufft {

namespace {

// Precision-tag dispatchers forwarding to decoupled device-kernel TUs
template <typename Traits>
void dispatch_precompute(
    sycl::queue& q,
    const typename Traits::XCoordT* d_x,
    typename Traits::XCoordT df_scaled,
    int M, int nfft, int w, int output_shift, bool mode43,
    int* d_first,
    typename Traits::ScalarT* d_u,
    typename Traits::ComplexT* d_shift,
    size_t wg_size) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_precompute_pointsh(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_precompute_points(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_precompute_pointsdf(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
    } else {
        detail::launch_precompute_pointsf(q, d_x, df_scaled, M, nfft, w, output_shift, mode43, d_first, d_u, d_shift, wg_size);
    }
}

template <typename Traits>
void dispatch_spread_smem(
    sycl::queue& q, int w,
    const int* d_first, const typename Traits::ScalarT* d_u, const typename Traits::ComplexT* d_shift,
    const int* d_bucket_off, const int* d_bucket_m,
    const typename Traits::ComplexT* d_y, typename Traits::ComplexT* d_fft_buf,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs,
    PswfMode mode, const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_spread_smem_winh(q, w, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_spread_smem_win(q, w, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_spread_smem_windf(q, w, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    } else {
        detail::launch_spread_smem_winf(q, w, d_first, d_u, d_shift, d_bucket_off, d_bucket_m, d_y, d_fft_buf, M, nfft, total_M_stride, chunk_bs, mode, cfg);
    }
}

template <typename Traits>
void dispatch_spread_collab(
    sycl::queue& q, int w,
    const int* d_first, const typename Traits::ScalarT* d_u, const typename Traits::ComplexT* d_shift,
    const typename Traits::ComplexT* d_y,
    typename Traits::RealT* d_grid_T_re, typename Traits::RealT* d_grid_T_im,
    int M, int nfft, size_t total_M_stride, size_t chunk_bs, size_t packed_stride,
    PswfMode mode, const SpreadConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_spread_collabh(q, w, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_spread_collab(q, w, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_spread_collabdf(q, w, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
    } else {
        detail::launch_spread_collabf(q, w, d_first, d_u, d_shift, d_y, d_grid_T_re, d_grid_T_im, M, nfft, total_M_stride, chunk_bs, packed_stride, mode, cfg);
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
void dispatch_deconv(
    sycl::queue& q, int w,
    const typename Traits::ComplexT* d_fft_buf, typename Traits::ComplexT* d_out,
    int nfft, int log_nfft, int nout, int output_shift,
    size_t total_nout_stride, size_t chunk_bs,
    PswfMode mode, NufftOutOrder order, const DeconvConfig& cfg) {
    if constexpr (Traits::kIsHalf) {
        detail::launch_deconv_emith(q, w, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg);
    } else if constexpr (Traits::kIsFp64) {
        detail::launch_deconv_emit(q, w, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg);
    } else if constexpr (Traits::kIsDf) {
        detail::launch_deconv_emitdf(q, w, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg);
    } else {
        detail::launch_deconv_emitf(q, w, d_fft_buf, d_out, nfft, log_nfft, nout, output_shift, total_nout_stride, chunk_bs, mode, order, cfg);
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
    const int t_j = next_pow2_int(w); // Always >= w -> zero second-pass width cliff!

    // Bank-conflict-free inter-transform SMEM stride padding from BankModel
    int grid_pad = 0;
    if (bm.banks > 1) {
        const int banks_per_elem = (static_cast<int>(sizeof(RealT)) + bm.bank_width_bytes - 1) / bm.bank_width_bytes;
        const int elem_banks = std::max(1, bm.banks / std::max(1, banks_per_elem));
        if (t_j < elem_banks) {
            grid_pad = t_j;
        }
    }

    auto calc_smem_bytes = [&](int b_per_wg, int win_elems, int tile_m) -> size_t {
        const int grid_stride = win_elems + grid_pad;
        return 2 * static_cast<size_t>(b_per_wg) * static_cast<size_t>(grid_stride) * sizeof(RealT) +
               static_cast<size_t>(tile_m) * sizeof(int) +
               2 * static_cast<size_t>(b_per_wg) * static_cast<size_t>(tile_m) * sizeof(RealT) +
               static_cast<size_t>(tile_m) * static_cast<size_t>(w) * sizeof(ScalarT);
    };

    const bool smem_allowed = !force_collab &&
                              (dev_cfg.has_native_local_mem || allow_emulated_smem) &&
                              dev_cfg.local_mem_size >= 4096;

    if (smem_allowed) {
        // Leave 1/16 headroom for driver/compiler local-memory metadata
        const size_t smem_limit = (dev_cfg.local_mem_size * 15) / 16;
        const int min_win = std::min(nfft, next_pow2_int(std::max(w * 2, 64)));
        const int wg_min  = std::max(t_j, static_cast<int>(sg_unit));
        const int wg_max  = std::min<int>(static_cast<int>(dev_cfg.max_work_group_size), 256);

        // Hardware-derived WG sizes: multiples {1, 2, 4, 8} of sub-group size within [wg_min, wg_max]
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

        auto add_smem_cand = [&](int wg_size, int win_elems) {
            const int b_per_wg = wg_size / t_j;
            if (b_per_wg <= 0 || win_elems < min_win || win_elems > nfft) return;
            // Derive tile_m from wg_size and sg_unit, shrinking if needed to stay within smem_limit
            int tile_m = std::min(wg_size, std::max(static_cast<int>(sg_unit), 32));
            tile_m = pow2_clamp_int(tile_m, 16, 64);
            while (tile_m > 16 && calc_smem_bytes(b_per_wg, win_elems, tile_m) > smem_limit) {
                tile_m >>= 1;
            }
            const size_t smem_b = calc_smem_bytes(b_per_wg, win_elems, tile_m);
            if (smem_b > smem_limit) return;

            const std::string id =
                "smem_wg" + std::to_string(wg_size) + "_win" + std::to_string(win_elems);
            for (const auto& existing : cands) {
                if (existing.variant_id == id) return;
            }

            SpreadConfig sc{};
            sc.use_smem         = true;
            sc.wg_size          = wg_size;
            sc.t_j              = t_j;
            sc.b_per_wg         = b_per_wg;
            sc.win_elems        = win_elems;
            sc.num_wins         = nfft / win_elems;
            sc.grid_stride      = win_elems + grid_pad;
            sc.tile_m           = tile_m;
            sc.transpose_tile   = tr_tile;
            sc.transpose_stride = tr_stride;

            // Analytical heuristic score (higher is better):
            //   - Rewards multi-warp concurrency (wg_size >= 2 * sg_unit)
            //   - Rewards >= 2 resident WGs per CU in local memory (smem_b * 2 <= local_mem_size)
            //   - Rewards higher b_per_wg (polynomial evaluation amortization)
            //   - Mildly penalizes excessive window fragmentation (small win_elems / high num_wins)
            const int resident_wgs = std::min(4, static_cast<int>(dev_cfg.local_mem_size / std::max<size_t>(1, smem_b)));
            const int active_warps_per_cu = resident_wgs * std::max(1, wg_size / static_cast<int>(sg_unit));
            int score = std::min(active_warps_per_cu, 16) * 20;
            score += std::min(b_per_wg, 16) * 12;
            if (sc.num_wins == 1) score += 25;
            else if (sc.num_wins <= 4) score += 15;
            else score -= int_log2_pow2(sc.num_wins) * 6;
            if (win_elems < 128) score -= 20;

            cands.push_back(SpreadCandidate{id, sc, smem_b, score});
        };

        for (int wg_size : wg_list) {
            const int b_per_wg = wg_size / t_j;
            int max_win = nfft;
            while (max_win >= min_win && calc_smem_bytes(b_per_wg, max_win, 16) > smem_limit) {
                max_win >>= 1;
            }
            if (max_win >= min_win) {
                add_smem_cand(wg_size, max_win);
                if (max_win / 2 >= min_win) {
                    add_smem_cand(wg_size, max_win / 2);
                }
                if (max_win / 4 >= min_win && calc_smem_bytes(b_per_wg, max_win / 2, 16) * 2 > dev_cfg.local_mem_size) {
                    add_smem_cand(wg_size, max_win / 4);
                }
            }
        }

        std::stable_sort(cands.begin(), cands.end(), [](const SpreadCandidate& a, const SpreadCandidate& b) {
            return a.score > b.score;
        });
    }

    // Collaborative batch-inner candidate (primary on CPU/non-native-smem; fallback/contender on GPU)
    {
        const size_t reg_cap = (Traits::kIsFp64 || Traits::kIsDf) ? 128 : 256;
        const int collab_wg = static_cast<int>(
            dev_cfg.default_wg_size(std::max<size_t>(64, sg_unit * 4), reg_cap));
        int tile_m = pow2_clamp_int(std::min(collab_wg, static_cast<int>(std::max<size_t>(32, sg_unit))), 16, 64);
        const size_t per_pt_bytes = sizeof(int) + sizeof(ComplexT) + static_cast<size_t>(w) * sizeof(ScalarT);
        while (tile_m > 16 && static_cast<size_t>(tile_m) * per_pt_bytes > dev_cfg.local_mem_size / 4) {
            tile_m >>= 1;
        }
        SpreadConfig cc{};
        cc.use_smem         = false;
        cc.wg_size          = std::max(1, collab_wg);
        cc.t_j              = t_j;
        cc.b_per_wg         = cc.wg_size;
        cc.win_elems        = nfft;
        cc.num_wins         = 1;
        cc.grid_stride      = nfft;
        cc.tile_m           = tile_m;
        cc.transpose_tile   = tr_tile;
        cc.transpose_stride = tr_stride;

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
      N_(N),
      w_(w),
      mode_(mode),
      nfft_(next_pow2_int(2 * N)),
      log_nfft_(int_log2_pow2(next_pow2_int(2 * N))),
      nout_(nufft1_nout(N, mode)),
      output_shift_(nufft1_output_shift(N, mode)),
      max_bs_(std::max<size_t>(1, max_bs)),
      chunk_bs_(1),
      packed_stride_(1),
      launch_cfg_(nufft_cfg),
      fft_plan_(q, int_log2_pow2(next_pow2_int(2 * N)), std::max<size_t>(1, max_bs), nufft_cfg.fft_cfg) {
    if (M_ <= 0 || N_ <= 0) {
        throw std::invalid_argument("Nufft1Plan: M and N must be positive");
    }
    if (mode_ == PswfMode::PSWF43 && (N_ % 4) != 0) {
        throw std::invalid_argument("Nufft1Plan: PSWF43 requires N divisible by 4");
    }
    if (!Traits::is_valid_width(w_)) {
        throw std::invalid_argument("Nufft1Plan: unsupported kernel width w for this precision");
    }

    const auto& dev_cfg  = fft_plan_.device_config();
    const size_t eff_sg  = fft_plan_.effective_sg_size();
    const bool is_heavy  = fft_plan_.is_compute_heavy();
    const std::string cache_path = sycl_fft::AutotuneCache::resolve_cache_path(launch_cfg_.fft_cfg.cache_path);
    const auto autotune_mode = launch_cfg_.fft_cfg.autotune;

    precompute_wg_ = dev_cfg.default_wg_size(static_cast<size_t>(M_), 256);

    // 1. Build hardware-derived spreading candidates (zero hardcoded Blackwell constants)
    const auto spread_cands = build_spread_candidates<Traits>(
        dev_cfg, eff_sg, nfft_, w_,
        launch_cfg_.force_collab_spread,
        launch_cfg_.fft_cfg.allow_emulated_local_mem);
    if (spread_cands.empty()) {
        throw std::runtime_error("Nufft1Plan: no valid spreading candidate for device");
    }

    // Resolve spreading candidate (override -> cache -> heuristic/autotune)
    int chosen_spread = 0;
    bool need_spread_tune = false;
    const bool allow_runtime_tune = (max_bs_ >= 32) || (autotune_mode == sycl_fft::AutotuneMode::Force);
    const std::string spr_slot = "nufft1_spr_w" + std::to_string(w_);
    const std::string spr_key  = sycl_fft::AutotuneCache::make_key(
        dev_cfg, Traits::kPrecTag, log_nfft_, spr_slot.c_str());

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
    const size_t fft_bytes = static_cast<size_t>(nfft_) * sizeof(ComplexT);
    const size_t buf_mult  = (!spread_cfg_.use_smem || fft_plan_.is_two_pass()) ? 2 : 1;
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
        if (chunk_bs_ > 1 && chunk_bs_ < max_bs_ && (chunk_bs_ & 1u) != 0u) {
            ++chunk_bs_;
        }
    }

    const size_t max_packed = (chunk_bs_ + Traits::kLanes - 1) / Traits::kLanes;
    const size_t align_elems = std::max<size_t>(
        std::max<size_t>(1, dev_cfg.sub_group_size),
        std::max<size_t>(1, dev_cfg.cache_line_size / sizeof(RealT)));
    packed_stride_ = ((max_packed + align_elems - 1) / align_elems) * align_elems;

    // 3. Allocate core device buffers (d_grid_T and d_bucket_* allocated ONLY if actually needed!)
    d_first_   = sycl::malloc_device<int>(static_cast<size_t>(M_), q_);
    d_u_       = sycl::malloc_device<ScalarT>(static_cast<size_t>(M_), q_);
    d_shift_   = sycl::malloc_device<ComplexT>(static_cast<size_t>(M_), q_);
    d_fft_buf_ = sycl::malloc_device<ComplexT>(chunk_bs_ * static_cast<size_t>(nfft_), q_);

    if (!d_first_ || !d_u_ || !d_shift_ || !d_fft_buf_) {
        if (d_first_) sycl::free(d_first_, q_);
        if (d_u_) sycl::free(d_u_, q_);
        if (d_shift_) sycl::free(d_shift_, q_);
        if (d_fft_buf_) sycl::free(d_fft_buf_, q_);
        d_first_ = nullptr; d_u_ = nullptr; d_shift_ = nullptr; d_fft_buf_ = nullptr;
        throw std::runtime_error("Nufft1Plan: core device memory allocation failed");
    }

    // 4. Shortlist plan-time autotuning for spreading (at most top 3 SMEM + optional collab)
    if (need_spread_tune) {
        std::vector<size_t> shortlist;
        for (size_t i = 0; i < spread_cands.size() && shortlist.size() < 3; ++i) {
            if (spread_cands[i].cfg.use_smem) shortlist.push_back(i);
        }
        if (nfft_ >= 4096 || shortlist.empty()) {
            for (size_t i = 0; i < spread_cands.size(); ++i) {
                if (!spread_cands[i].cfg.use_smem) {
                    shortlist.push_back(i);
                    break;
                }
            }
        }

        if (shortlist.size() > 1) {
            const size_t tune_bs = std::min(chunk_bs_, std::max<size_t>(64, chunk_bs_));
            ComplexT* d_tune_y = sycl::malloc_device<ComplexT>(tune_bs * static_cast<size_t>(M_), q_);
            if (d_tune_y) {
                q_.memset(d_tune_y, 0, tune_bs * static_cast<size_t>(M_) * sizeof(ComplexT));
                // Populate deterministic synthetic first_idx spread across [0, nfft) for realistic timing
                std::vector<int> h_first(static_cast<size_t>(M_));
                for (int m = 0; m < M_; ++m) {
                    h_first[static_cast<size_t>(m)] = static_cast<int>((static_cast<uint64_t>(m) * 2654435761ULL) & static_cast<uint64_t>(nfft_ - 1));
                }
                q_.memcpy(d_first_, h_first.data(), static_cast<size_t>(M_) * sizeof(int));
                q_.memset(d_u_, 0, static_cast<size_t>(M_) * sizeof(ScalarT));
                q_.memset(d_shift_, 0, static_cast<size_t>(M_) * sizeof(ComplexT));
                q_.wait();

                int best_idx = static_cast<int>(shortlist[0]);
                double best_ms = std::numeric_limits<double>::max();

                for (size_t idx : shortlist) {
                    const auto& cand = spread_cands[idx];
                    try {
                        if (cand.cfg.use_smem) {
                            int* tmp_cnt = nullptr;
                            int* tmp_off = nullptr;
                            int* tmp_m   = nullptr;
                            if (cand.cfg.num_wins > 1) {
                                tmp_cnt = sycl::malloc_device<int>(static_cast<size_t>(cand.cfg.num_wins), q_);
                                tmp_off = sycl::malloc_device<int>(static_cast<size_t>(cand.cfg.num_wins + 1), q_);
                                tmp_m   = sycl::malloc_device<int>(static_cast<size_t>(2 * M_), q_);
                                detail::launch_build_window_buckets(
                                    q_, d_first_, M_, nfft_, w_, cand.cfg.win_elems, cand.cfg.num_wins,
                                    tmp_cnt, tmp_off, tmp_m, precompute_wg_);
                            }
                            dispatch_spread_smem<Traits>(
                                q_, w_, d_first_, d_u_, d_shift_, tmp_off, tmp_m,
                                d_tune_y, d_fft_buf_, M_, nfft_, static_cast<size_t>(M_), tune_bs,
                                mode_, cand.cfg);
                            q_.wait_and_throw();

                            double samples[3] = {};
                            for (int r = 0; r < 3; ++r) {
                                const auto t0 = std::chrono::steady_clock::now();
                                dispatch_spread_smem<Traits>(
                                    q_, w_, d_first_, d_u_, d_shift_, tmp_off, tmp_m,
                                    d_tune_y, d_fft_buf_, M_, nfft_, static_cast<size_t>(M_), tune_bs,
                                    mode_, cand.cfg);
                                q_.wait_and_throw();
                                samples[r] = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count();
                            }
                            if (tmp_cnt) sycl::free(tmp_cnt, q_);
                            if (tmp_off) sycl::free(tmp_off, q_);
                            if (tmp_m)   sycl::free(tmp_m, q_);

                            std::sort(samples, samples + 3);
                            if (samples[1] < best_ms) {
                                best_ms = samples[1];
                                best_idx = static_cast<int>(idx);
                            }
                        } else {
                            RealT* tmp_re = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
                            RealT* tmp_im = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
                            if (tmp_re && tmp_im) {
                                q_.memset(tmp_re, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
                                q_.memset(tmp_im, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
                                dispatch_spread_collab<Traits>(
                                    q_, w_, d_first_, d_u_, d_shift_, d_tune_y, tmp_re, tmp_im,
                                    M_, nfft_, static_cast<size_t>(M_), tune_bs, packed_stride_, mode_, cand.cfg);
                                dispatch_transpose_and_zero<Traits>(
                                    q_, tmp_re, tmp_im, d_fft_buf_, nfft_, tune_bs, packed_stride_, cand.cfg);
                                q_.wait_and_throw();

                                double samples[3] = {};
                                for (int r = 0; r < 3; ++r) {
                                    const auto t0 = std::chrono::steady_clock::now();
                                    dispatch_spread_collab<Traits>(
                                        q_, w_, d_first_, d_u_, d_shift_, d_tune_y, tmp_re, tmp_im,
                                        M_, nfft_, static_cast<size_t>(M_), tune_bs, packed_stride_, mode_, cand.cfg);
                                    dispatch_transpose_and_zero<Traits>(
                                        q_, tmp_re, tmp_im, d_fft_buf_, nfft_, tune_bs, packed_stride_, cand.cfg);
                                    q_.wait_and_throw();
                                    samples[r] = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t0).count();
                                }
                                std::sort(samples, samples + 3);
                                if (samples[1] < best_ms) {
                                    best_ms = samples[1];
                                    best_idx = static_cast<int>(idx);
                                }
                            }
                            if (tmp_re) sycl::free(tmp_re, q_);
                            if (tmp_im) sycl::free(tmp_im, q_);
                        }
                    } catch (...) {
                    }
                }
                sycl::free(d_tune_y, q_);

                chosen_spread = best_idx;
                spread_cfg_ = spread_cands[static_cast<size_t>(chosen_spread)].cfg;
                spread_variant_name_ = spread_cands[static_cast<size_t>(chosen_spread)].variant_id;

                if (!launch_cfg_.has_spread_overrides() && best_ms < std::numeric_limits<double>::max()) {
                    sycl_fft::AutotuneEntry ent{};
                    ent.mode = sycl_fft::PlanExecMode::SinglePass;
                    ent.variant_id = spread_variant_name_;
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
    } else {
        allocate_collab_grid();
    }

    // 5. Build and resolve hardware-derived deconvolution candidates
    // Size candidate batch tiles using at least a representative L2-saturated batch so small
    // verification calls and full-batch production calls share a consistent candidate space.
    const size_t rep_bs = std::max<size_t>(
        chunk_bs_,
        std::clamp<size_t>(dev_cfg.effective_l2_bytes() / std::max<size_t>(1, 4 * fft_bytes), 64, 2048));
    const auto deconv_cands = build_deconv_candidates<Traits>(dev_cfg, eff_sg, nout_, rep_bs);
    int chosen_deconv = 0;
    bool need_deconv_tune = false;
    const std::string dec_slot =
        "nufft1_dec_w" + std::to_string(w_) + ((mode_ == PswfMode::PSWF43) ? "_43" : "_21");
    const std::string dec_key = sycl_fft::AutotuneCache::make_key(
        dev_cfg, Traits::kPrecTag, log_nfft_, dec_slot.c_str());

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
    } else if (autotune_mode == sycl_fft::AutotuneMode::Auto && !launch_cfg_.has_deconv_overrides()) {
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
    } else if (autotune_mode == sycl_fft::AutotuneMode::Force && deconv_cands.size() > 1) {
        need_deconv_tune = true;
    }

    if (need_deconv_tune) {
        const size_t lim = std::min<size_t>(deconv_cands.size(), 3);
        ComplexT* d_tune_out = sycl::malloc_device<ComplexT>(chunk_bs_ * static_cast<size_t>(nout_), q_);
        if (d_tune_out) {
            q_.memset(d_fft_buf_, 0, chunk_bs_ * static_cast<size_t>(nfft_) * sizeof(ComplexT)).wait();
            int best_idx = 0;
            double best_ms = std::numeric_limits<double>::max();
            for (size_t i = 0; i < lim; ++i) {
                try {
                    dispatch_deconv<Traits>(
                        q_, w_, d_fft_buf_, d_tune_out, nfft_, log_nfft_, nout_, output_shift_,
                        static_cast<size_t>(nout_), chunk_bs_, mode_, NufftOutOrder::SlotBitReversed,
                        deconv_cands[i].cfg);
                    q_.wait_and_throw();

                    double samples[3] = {};
                    for (int r = 0; r < 3; ++r) {
                        const auto t0 = std::chrono::steady_clock::now();
                        dispatch_deconv<Traits>(
                            q_, w_, d_fft_buf_, d_tune_out, nfft_, log_nfft_, nout_, output_shift_,
                            static_cast<size_t>(nout_), chunk_bs_, mode_, NufftOutOrder::SlotBitReversed,
                            deconv_cands[i].cfg);
                        q_.wait_and_throw();
                        samples[r] = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
                    }
                    std::sort(samples, samples + 3);
                    if (samples[1] < best_ms) {
                        best_ms = samples[1];
                        best_idx = static_cast<int>(i);
                    }
                } catch (...) {
                }
            }
            sycl::free(d_tune_out, q_);
            chosen_deconv = best_idx;
            if (!launch_cfg_.has_deconv_overrides() && best_ms < std::numeric_limits<double>::max()) {
                sycl_fft::AutotuneEntry ent{};
                ent.mode = sycl_fft::PlanExecMode::SinglePass;
                ent.variant_id = deconv_cands[static_cast<size_t>(chosen_deconv)].variant_id;
                sycl_fft::AutotuneCache::put(cache_path, dec_key, ent);
            }
        }
    }

    deconv_cfg_ = deconv_cands[static_cast<size_t>(chosen_deconv)].cfg;
    deconv_variant_name_ = deconv_cands[static_cast<size_t>(chosen_deconv)].variant_id;
}

template <typename Traits>
void Nufft1PlanBase<Traits>::allocate_collab_grid() {
    if (d_grid_T_re_ && d_grid_T_im_) return;
    d_grid_T_re_ = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
    d_grid_T_im_ = sycl::malloc_device<RealT>(static_cast<size_t>(nfft_) * packed_stride_, q_);
    if (!d_grid_T_re_ || !d_grid_T_im_) {
        if (d_grid_T_re_) sycl::free(d_grid_T_re_, q_);
        if (d_grid_T_im_) sycl::free(d_grid_T_im_, q_);
        d_grid_T_re_ = nullptr; d_grid_T_im_ = nullptr;
        throw std::runtime_error("Nufft1Plan: collaborative grid allocation failed");
    }
    q_.memset(d_grid_T_re_, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.memset(d_grid_T_im_, 0, static_cast<size_t>(nfft_) * packed_stride_ * sizeof(RealT));
    q_.wait();
}

template <typename Traits>
void Nufft1PlanBase<Traits>::allocate_window_buckets() {
    if (d_bucket_counts_ && d_bucket_off_ && d_bucket_m_) return;
    d_bucket_counts_ = sycl::malloc_device<int>(static_cast<size_t>(spread_cfg_.num_wins), q_);
    d_bucket_off_    = sycl::malloc_device<int>(static_cast<size_t>(spread_cfg_.num_wins + 1), q_);
    d_bucket_m_      = sycl::malloc_device<int>(static_cast<size_t>(2 * M_), q_);
    if (!d_bucket_counts_ || !d_bucket_off_ || !d_bucket_m_) {
        if (d_bucket_counts_) sycl::free(d_bucket_counts_, q_);
        if (d_bucket_off_) sycl::free(d_bucket_off_, q_);
        if (d_bucket_m_) sycl::free(d_bucket_m_, q_);
        d_bucket_counts_ = nullptr; d_bucket_off_ = nullptr; d_bucket_m_ = nullptr;
        throw std::runtime_error("Nufft1Plan: window bucket allocation failed");
    }
}

template <typename Traits>
Nufft1PlanBase<Traits>::~Nufft1PlanBase() {
    q_.wait();
    if (d_first_) sycl::free(d_first_, q_);
    if (d_u_) sycl::free(d_u_, q_);
    if (d_shift_) sycl::free(d_shift_, q_);
    if (d_bucket_counts_) sycl::free(d_bucket_counts_, q_);
    if (d_bucket_off_) sycl::free(d_bucket_off_, q_);
    if (d_bucket_m_) sycl::free(d_bucket_m_, q_);
    if (d_grid_T_re_) sycl::free(d_grid_T_re_, q_);
    if (d_grid_T_im_) sycl::free(d_grid_T_im_, q_);
    if (d_fft_buf_) sycl::free(d_fft_buf_, q_);
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
        mode_ == PswfMode::PSWF43, d_first_, d_u_, d_shift_, precompute_wg_);

    if (spread_cfg_.use_smem && spread_cfg_.num_wins > 1) {
        detail::launch_build_window_buckets(
            q_, d_first_, M_, nfft_, w_,
            spread_cfg_.win_elems, spread_cfg_.num_wins,
            d_bucket_counts_, d_bucket_off_, d_bucket_m_,
            precompute_wg_);
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

    for (size_t b_off = 0; b_off < bs; b_off += chunk_bs_) {
        const size_t cur_bs = std::min(chunk_bs_, bs - b_off);
        const ComplexT* cur_y = d_y + b_off * total_M_stride;
        ComplexT* cur_out     = d_out + b_off * total_nout_stride;

        // 1. Hardware-generalized piecewise minimax PSWF spreading
        if (spread_cfg_.use_smem) {
            dispatch_spread_smem<Traits>(
                q_, w_, d_first_, d_u_, d_shift_, d_bucket_off_, d_bucket_m_,
                cur_y, d_fft_buf_, M_, nfft_, total_M_stride, cur_bs, mode_, spread_cfg_);
        } else {
            dispatch_spread_collab<Traits>(
                q_, w_, d_first_, d_u_, d_shift_, cur_y, d_grid_T_re_, d_grid_T_im_,
                M_, nfft_, total_M_stride, cur_bs, packed_stride_, mode_, spread_cfg_);
            dispatch_transpose_and_zero<Traits>(
                q_, d_grid_T_re_, d_grid_T_im_, d_fft_buf_, nfft_, cur_bs, packed_stride_, spread_cfg_);
        }

        // 2. Raw DIF Radix-2 FFT (Sign = +1, NO bit-reversal pass!)
        fft_plan_.execute_raw(d_fft_buf_, d_fft_buf_, cur_bs);

        // 3. FINUFFT-adjacent Gauss-Legendre-free self-FT deconvolution pass
        dispatch_deconv<Traits>(
            q_, w_, d_fft_buf_, cur_out, nfft_, log_nfft_, nout_, output_shift_,
            total_nout_stride, cur_bs, mode_, order, deconv_cfg_);
    }
}

// Explicit instantiations for all 4 precisions (pure host code, 0 KiB HCF)
template class Nufft1PlanBase<detail::HalfPlanTraits>;
template class Nufft1PlanBase<detail::FloatPlanTraits>;
template class Nufft1PlanBase<detail::DoublePlanTraits>;
template class Nufft1PlanBase<detail::DfPlanTraits>;

} // namespace sycl_nufft
