#pragma once

#include "sycl_fft_layout.hpp"
#include "sycl_fft_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// ============================================================================
// SYCL 2020 Kernel Bundle & Required Sub-Group Size Capability Detection
//   AdaptiveCpp 25.10 stubs sycl::kernel and lacks kernel_bundle / get_kernel_id
//   / [[sycl::reqd_sub_group_size]]. DPC++ / oneAPI implements them natively.
// ============================================================================
#if !defined(SYCL_IMPLEMENTATION_ACPP) && !defined(SYCL_IMPLEMENTATION_HIPSYCL) && \
    (defined(__SYCL_COMPILER_VERSION) || (defined(SYCL_LANGUAGE_VERSION) && SYCL_LANGUAGE_VERSION >= 202001))
#define SYCL_FFT_HAS_KERNEL_BUNDLE 1
#define SYCL_FFT_HAS_REQD_SG_SIZE  1
#else
#define SYCL_FFT_HAS_KERNEL_BUNDLE 0
#define SYCL_FFT_HAS_REQD_SG_SIZE  0
#endif

namespace sycl_fft {

enum class AutotuneMode : uint8_t {
    Auto  = 0, // Use persistent disk cache; tune once on cache miss
    Off   = 1, // Heuristic selection only (no timing runs)
    Force = 2  // Always re-run micro-autotuner and overwrite cache entry
};

enum class PlanExecMode : uint8_t {
    SinglePass       = 0,
    TwoPass          = 1,
    MultiStageGlobal = 2
};

inline std::string sanitize_cache_token(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char c : raw) {
        if (c == '\t' || c == '\n' || c == '\r') {
            out.push_back(' ');
        } else if (c >= 0x20 && c < 0x7f) {
            out.push_back(static_cast<char>(c));
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    size_t lead = 0;
    while (lead < out.size() && out[lead] == ' ') ++lead;
    if (lead > 0) out.erase(0, lead);
    return out.empty() ? std::string("unknown") : out;
}

inline std::string compiler_id_tag() {
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
#if defined(ACPP_VERSION_MAJOR) && defined(ACPP_VERSION_MINOR)
    return "acpp-" + std::to_string(ACPP_VERSION_MAJOR) + "." + std::to_string(ACPP_VERSION_MINOR);
#else
    return "acpp";
#endif
#elif defined(__INTEL_LLVM_COMPILER)
    return "dpcpp-" + std::to_string(__INTEL_LLVM_COMPILER);
#elif defined(__SYCL_COMPILER_VERSION)
    return "sycl-" + std::to_string(__SYCL_COMPILER_VERSION);
#else
    return "sycl-generic";
#endif
}

inline std::string backend_to_tag(const sycl::device& dev) {
    const auto b = dev.get_backend();
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
    switch (b) {
        case sycl::backend::cuda:       return "cuda";
        case sycl::backend::hip:        return "hip";
        case sycl::backend::level_zero: return "level_zero";
        case sycl::backend::ocl:        return "opencl";
        case sycl::backend::omp:        return "omp";
        case sycl::backend::metal:      return "metal";
        case sycl::backend::vk:         return "vulkan";
        default:                        return "unknown";
    }
#else
    switch (b) {
        case sycl::backend::opencl:               return "opencl";
        case sycl::backend::ext_oneapi_level_zero: return "level_zero";
        case sycl::backend::ext_oneapi_cuda:      return "cuda";
        case sycl::backend::ext_oneapi_hip:       return "hip";
        default:                                  return "unknown";
    }
#endif
}

// ============================================================================
// Adaptive SYCL Device Query & Launch Configuration
// ============================================================================
struct DeviceFftConfig {
    size_t max_compute_units = 1;
    size_t max_work_group_size = 256;
    size_t sub_group_size = 32;
    size_t min_sub_group_size = 32;
    size_t local_mem_size = 32768;
    size_t l2_cache_size = 0;
    size_t cache_line_size = 128;
    size_t global_mem_size = 0;
    size_t max_mem_alloc_size = 0;
    bool is_gpu = false;
    bool has_native_local_mem = false;
    bool has_fp64 = true;
    bool has_fp16_aspect = false;

    // Extended hardware identity & layout properties
    std::string device_name;
    std::string driver_version;
    std::string backend_tag;
    uint32_t vendor_id = 0;
    std::vector<size_t> sub_group_sizes;
    BankModelKind bank_model = BankModelKind::B32x4;

    std::string identity_key() const {
        char vid[16];
        std::snprintf(vid, sizeof(vid), "0x%x", vendor_id);
        return sanitize_cache_token(device_name) + "|" +
               std::string(vid) + "|" +
               sanitize_cache_token(driver_version) + "|" +
               backend_tag;
    }

    static DeviceFftConfig query(const sycl::device& dev) {
        DeviceFftConfig cfg{};
        cfg.is_gpu = dev.is_gpu();
        cfg.has_fp64 = dev.has(sycl::aspect::fp64);
        cfg.has_fp16_aspect = dev.has(sycl::aspect::fp16);
        cfg.device_name = sanitize_cache_token(dev.get_info<sycl::info::device::name>());
        cfg.driver_version = sanitize_cache_token(dev.get_info<sycl::info::device::driver_version>());
        cfg.backend_tag = backend_to_tag(dev);
        cfg.vendor_id = static_cast<uint32_t>(dev.get_info<sycl::info::device::vendor_id>());

        cfg.max_compute_units = std::max<size_t>(
            1, static_cast<size_t>(dev.get_info<sycl::info::device::max_compute_units>()));
        cfg.max_work_group_size = std::max<size_t>(
            1, static_cast<size_t>(dev.get_info<sycl::info::device::max_work_group_size>()));

        auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        cfg.sub_group_sizes.clear();
        for (size_t s : sg_sizes) {
            if (s > 0) cfg.sub_group_sizes.push_back(s);
        }
        std::sort(cfg.sub_group_sizes.begin(), cfg.sub_group_sizes.end());
        cfg.sub_group_sizes.erase(
            std::unique(cfg.sub_group_sizes.begin(), cfg.sub_group_sizes.end()),
            cfg.sub_group_sizes.end());

        if (!cfg.sub_group_sizes.empty()) {
            cfg.min_sub_group_size = cfg.sub_group_sizes.front();
            cfg.sub_group_size = cfg.sub_group_sizes.back();
        } else {
            cfg.min_sub_group_size = cfg.is_gpu ? 32 : 1;
            cfg.sub_group_size = cfg.min_sub_group_size;
            cfg.sub_group_sizes.push_back(cfg.sub_group_size);
        }

        cfg.has_native_local_mem =
            (dev.get_info<sycl::info::device::local_mem_type>() == sycl::info::local_mem_type::local);
        cfg.local_mem_size = static_cast<size_t>(dev.get_info<sycl::info::device::local_mem_size>());

        if (dev.get_info<sycl::info::device::global_mem_cache_type>() !=
            sycl::info::global_mem_cache_type::none) {
            const size_t reported =
                static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_cache_size>());
            if (reported >= size_t{64} * 1024) {
                cfg.l2_cache_size = reported;
            }
        }
        cfg.cache_line_size = std::max<size_t>(
            32, static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_cache_line_size>()));
        cfg.global_mem_size = static_cast<size_t>(dev.get_info<sycl::info::device::global_mem_size>());
        cfg.max_mem_alloc_size = static_cast<size_t>(dev.get_info<sycl::info::device::max_mem_alloc_size>());

        if (!cfg.has_native_local_mem || !cfg.is_gpu) {
            cfg.bank_model = BankModelKind::None;
        } else if (cfg.vendor_id == 0x8086u && cfg.max_work_group_size <= 256) {
            cfg.bank_model = BankModelKind::B16x4;
        } else {
            cfg.bank_model = BankModelKind::B32x4;
        }
        return cfg;
    }

    bool supports_sub_group_size(size_t s) const {
        return std::binary_search(sub_group_sizes.begin(), sub_group_sizes.end(), s);
    }

    size_t pick_sub_group_size(size_t min_req) const {
        for (size_t s : sub_group_sizes) {
            if (s >= min_req && (s & (s - 1)) == 0) {
                return s;
            }
        }
        return 0;
    }

    size_t clamp_pow2_wg(size_t total_threads, size_t requested_wg, size_t hard_cap = 0) const {
        size_t cap = std::min(requested_wg, max_work_group_size);
        if (hard_cap > 0) cap = std::min(cap, hard_cap);
        if (total_threads == 0) return std::max<size_t>(1, cap);
        cap = std::min(cap, total_threads);
        size_t wg = 1;
        while ((wg << 1) <= cap) wg <<= 1;
        while (wg > 1 && (total_threads % wg) != 0) wg >>= 1;
        return wg;
    }

    size_t default_wg_size(size_t total_threads, size_t reg_wg_cap = 256) const {
        const size_t sg_hint = is_gpu ? std::min<size_t>(sub_group_size, 64) : size_t{8};
        const size_t preferred = std::max<size_t>(64, sg_hint * 8);
        return clamp_pow2_wg(total_threads, preferred, reg_wg_cap);
    }

    size_t effective_l2_bytes() const {
        if (l2_cache_size > 0) return l2_cache_size;
        size_t fallback = is_gpu ? (size_t{16} * 1024 * 1024) : (size_t{4} * 1024 * 1024);
        if (global_mem_size > 0) {
            fallback = std::clamp<size_t>(
                global_mem_size / 256, size_t{2} * 1024 * 1024, size_t{32} * 1024 * 1024);
        }
        return fallback;
    }

    size_t compute_l2_chunk_bytes(
        size_t fft_bytes,
        size_t n1,
        bool is_compute_heavy,
        size_t l2_num = 1,
        size_t l2_den = 4,
        size_t min_waves_per_cu = 0) const {
        const size_t effective_l2 = effective_l2_bytes();
        const size_t den = (l2_den == 0) ? 4 : l2_den;
        const size_t num = (l2_num == 0) ? 1 : l2_num;
        const size_t base_budget = std::max<size_t>(size_t{256} * 1024, (effective_l2 * num) / den);
        const size_t max_l2_budget = std::max<size_t>(base_budget, effective_l2 / 2);

        const size_t target_waves = (min_waves_per_cu > 0)
            ? min_waves_per_cu
            : (!is_gpu ? size_t{1} : (is_compute_heavy ? size_t{8} : size_t{2}));
        const size_t target_wgs = max_compute_units * target_waves;
        const size_t occ_chunk_bs = (n1 > 0) ? ((target_wgs + n1 - 1) / n1) : 1;
        const size_t occ_budget = occ_chunk_bs * fft_bytes;

        size_t budget = std::max(base_budget, std::min(occ_budget, max_l2_budget));
        const size_t align = std::max(fft_bytes, cache_line_size);
        if (align > 0) {
            budget = ((budget + align - 1) / align) * align;
        }
        if (max_mem_alloc_size > 0) {
            budget = std::min(budget, std::max(fft_bytes, max_mem_alloc_size / 4));
        }
        return std::max(fft_bytes, budget);
    }
};

struct FftLaunchConfig {
    size_t l2_chunk_bytes = 0;        // 0 = auto from device L2 cache & compute units (or autotuner)
    size_t pass0_wg_size = 0;         // 0 = auto from device sub-group & max work-group size
    size_t bitrev_wg_size = 0;        // 0 = auto from device sub-group & max work-group size
    int    two_pass_min_logN = 0;     // 0 = auto from device candidates / autotuner
    int    two_pass_log_n1 = 0;       // 0 = auto
    size_t l2_budget_num = 1;         // base L2 fraction numerator (default 1/4)
    size_t l2_budget_den = 4;         // base L2 fraction denominator (default 1/4)
    size_t min_waves_per_cu = 0;      // 0 = auto (1 CPU, 2 FP32/DF/fast-FP64, 8 slow-FP64)
    bool   allow_wide_kernels = true; // allow 2-transpose kernels when hardware supports them

    // Generalization & autotuning controls
    AutotuneMode autotune = AutotuneMode::Auto; // default on with persistent disk cache
    std::string  cache_path;                    // empty = default env/XDG/$HOME cache path
    bool   allow_emulated_local_mem = false;    // refuse smem transpose kernels when !has_native_local_mem
    bool   allow_multistage_fallback = true;    // allow zero-smem multi-stage global fallback
    bool   force_multistage = false;            // force zero-smem multi-stage global path
    int    fp64_compute_heavy_override = -1;    // -1 = auto microbench, 0 = false, 1 = true
    int    stash_8192_override = -1;            // -1 = auto probe/autotune, 0 = false, 1 = true
    std::string variant_override;               // non-empty = pin specific single-pass variant_id
    bool   skip_df_kat = false;                 // false = enforce plan-time DF Known-Answer Test
    std::string bitrev_variant_override;        // non-empty = pin specific bit-reversal strategy
    bool   skip_fp16_kat = false;               // false = enforce plan-time FP16 KAT

    bool has_plan_overrides() const {
        return l2_chunk_bytes > 0 || pass0_wg_size > 0 || two_pass_min_logN > 0 ||
               two_pass_log_n1 > 0 || !allow_wide_kernels || force_multistage ||
               allow_emulated_local_mem || fp64_compute_heavy_override >= 0 ||
               stash_8192_override >= 0 || !variant_override.empty();
    }

    bool has_bitrev_overrides() const {
        return bitrev_wg_size > 0 || !bitrev_variant_override.empty() || allow_emulated_local_mem;
    }
};

// ============================================================================
// SYCL 2020 kernel_bundle / kernel_device_specific Probe
// ============================================================================
struct KernelProbeInfo {
    bool     available = false;
    size_t   max_wg_size = 0;
    size_t   private_mem_bytes = 0;
    size_t   preferred_wg_multiple = 0;
    uint32_t max_sg_size = 0;
    uint32_t compile_sg_size = 0;
};

template <typename KernelTag>
inline KernelProbeInfo probe_kernel_bundle(sycl::queue& q, const sycl::device& dev) {
    KernelProbeInfo info{};
#if SYCL_FFT_HAS_KERNEL_BUNDLE
    try {
        const auto kid = sycl::get_kernel_id<KernelTag>();
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            q.get_context(), {dev}, {kid});
        auto k = bundle.get_kernel(kid);
        info.available = true;
        info.max_wg_size = k.template get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
        info.private_mem_bytes = k.template get_info<sycl::info::kernel_device_specific::private_mem_size>(dev);
        info.preferred_wg_multiple =
            k.template get_info<sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(dev);
        info.max_sg_size = k.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(dev);
        info.compile_sg_size =
            k.template get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(dev);
    } catch (...) {
        info.available = false;
    }
#else
    (void)q;
    (void)dev;
#endif
    return info;
}

// ============================================================================
// Always-On Process-Cached Hardware Probes (Sub-Group Topology, DF KAT, FP64 Ratio)
// ============================================================================
#if defined(SYCL_FFT_DEVICE_IMPLEMENTATION_TU)
#define SYCL_FFT_DEVICE_LINKAGE
#elif defined(SYCL_FFT_EXTERN_DEVICE) || defined(SYCL_FFT_USE_EXTERN_TEMPLATES)
#define SYCL_FFT_DEVICE_DECL_ONLY 1
#else
#define SYCL_FFT_DEVICE_LINKAGE inline
#endif

struct DeviceProbeResult {
    bool     sg_probed = false;
    size_t   raw_sg_size = 1;
    size_t   effective_sg_size = 1;
    bool     sg_linear_ok = false;
    bool     sg_xor_ok = false;

    bool     df_kat_probed = false;
    bool     df_kat_passed = false;
    std::string df_kat_error;

    bool     fp16_kat_probed = false;
    bool     fp16_kat_passed = false;
    std::string fp16_kat_error;

    bool     fp64_probed = false;
    double   fp64_to_fp32_ratio = 0.0;
    bool     fp64_compute_heavy = true;
};

#if defined(SYCL_FFT_DEVICE_DECL_ONLY)

DeviceProbeResult probe_subgroup_topology(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_df_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_fp16_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_fp64_ratio(sycl::queue& q, const DeviceFftConfig& dev_cfg);

#else

namespace detail {

struct SubgroupTopologyProbeKernel {
    uint32_t* out;
    int probe_wg;

    void operator()(sycl::nd_item<1> item) const {
        const int t = static_cast<int>(item.get_local_linear_id());
        auto sg = item.get_sub_group();
        const uint32_t sg_sz = static_cast<uint32_t>(sg.get_max_local_range()[0]);
        if (t == 0) {
            out[0] = sg_sz;
        }
        const bool pow2 = (sg_sz > 0) && ((sg_sz & (sg_sz - 1u)) == 0u);
        if (!pow2 || sg.get_local_linear_id() != (static_cast<uint32_t>(t) & (sg_sz - 1u))) {
            out[1] = 0u;
        }
        const uint32_t max_h = (sg_sz < static_cast<uint32_t>(probe_wg))
            ? sg_sz : static_cast<uint32_t>(probe_wg);
        const uint32_t self = static_cast<uint32_t>(t);
        if (max_h >= 2u) {
            if (sycl::permute_group_by_xor(sg, self, 1u) != (self ^ 1u)) out[2] = 0u;
        }
        if (max_h >= 4u) {
            if (sycl::permute_group_by_xor(sg, self, 2u) != (self ^ 2u)) out[2] = 0u;
        }
        if (max_h >= 8u) {
            if (sycl::permute_group_by_xor(sg, self, 4u) != (self ^ 4u)) out[2] = 0u;
        }
        if (max_h >= 16u) {
            if (sycl::permute_group_by_xor(sg, self, 8u) != (self ^ 8u)) out[2] = 0u;
        }
        if (max_h >= 32u) {
            if (sycl::permute_group_by_xor(sg, self, 16u) != (self ^ 16u)) out[2] = 0u;
        }
    }
};

struct DfKatProbeKernel {
    const float* in;
    float* out;

    void operator()(sycl::id<1>) const {
        float r, e;
        // 0: TwoSum(1.0f, 3*2^-24)
        df_two_sum(in[0], in[1], r, e);
        out[0] = r; out[1] = e;
        // 1: TwoDiff(1.5f, 3*2^-24)
        df_two_difference(in[2], in[3], r, e);
        out[2] = r; out[3] = e;
        // 2: TwoProd(1 + 2^-12, 1 + 2^-13)
        df_two_product(in[4], in[5], r, e);
        out[4] = r; out[5] = e;
        // 3: df_add, df_mul, df_cmul_fused on non-trivial df_t inputs
        const df_t a{in[6], in[7]};
        const df_t b{in[8], in[9]};
        const df_t c{in[10], in[11]};
        const df_t d{in[12], in[13]};
        const df_t s = df_add(a, b);
        const df_t p = df_mul(a, c);
        df_t cm_r, cm_i;
        df_cmul_fused(a, b, c, d, cm_r, cm_i);
        out[6]  = s.hi;    out[7]  = s.lo;
        out[8]  = p.hi;    out[9]  = p.lo;
        out[10] = cm_r.hi; out[11] = cm_r.lo;
        out[12] = cm_i.hi; out[13] = cm_i.lo;
    }
};

struct Fp16KatProbeKernel {
    const Complexh* in0;
    const Complexh* in1;
    Complexh* out0;
    Complexh* out1;

    void operator()(sycl::id<1>) const {
        f16x2_t ar, ai, br, bi, wr, wi;
        f16x2_load_pair(in0, in1, 0, true, true, ar, ai);
        f16x2_load_pair(in0, in1, 1, true, true, br, bi);
        f16x2_load_pair(in0, in1, 2, true, true, wr, wi);

        // 0: average butterfly sum: 0.5 * (a + b)
        const f16x2_t sr = f16x2_scale_half(f16x2_add(ar, br));
        const f16x2_t si = f16x2_scale_half(f16x2_add(ai, bi));
        f16x2_store_pair(out0, out1, 0, true, true, sr, si);

        // 1: average butterfly diff * twiddle: (0.5 * (a - b)) * w
        const f16x2_t dr = f16x2_scale_half(f16x2_sub(ar, br));
        const f16x2_t di = f16x2_scale_half(f16x2_sub(ai, bi));
        f16x2_t tr, ti;
        f16x2_cmul(dr, di, wr, wi, tr, ti);
        f16x2_store_pair(out0, out1, 1, true, true, tr, ti);

        // 2: conditional sign flip + sqrt(1/2) rotation
        const f16x2_t nr = f16x2_cond_neg(ar, true);
        const f16x2_t ni = f16x2_cond_neg(ai, false);
        f16x2_t qr, qi;
        f16x2_cmul_sqrt1_2_rt<false>(nr, ni, /*pos_sign=*/false, qr, qi);
        f16x2_store_pair(out0, out1, 2, true, true, qr, qi);
    }
};

template <typename T>
struct FmaThroughputProbeKernel {
    T* out;

    void operator()(sycl::nd_item<1> item) const {
        const size_t gid = item.get_global_linear_id();
        const T seed = static_cast<T>(1.0) + static_cast<T>(gid & 15u) * static_cast<T>(1e-4);
        const T a = static_cast<T>(1.000001);
        const T b = static_cast<T>(0.0000005);
        T x0 = seed, x1 = seed + static_cast<T>(0.1), x2 = seed + static_cast<T>(0.2), x3 = seed + static_cast<T>(0.3);
        for (int it = 0; it < 256; ++it) {
            x0 = sycl::fma(x0, a, b);
            x1 = sycl::fma(x1, a, b);
            x2 = sycl::fma(x2, a, b);
            x3 = sycl::fma(x3, a, b);
            x0 = sycl::fma(x0, a, -b);
            x1 = sycl::fma(x1, a, -b);
            x2 = sycl::fma(x2, a, -b);
            x3 = sycl::fma(x3, a, -b);
        }
        out[gid] = x0 + x1 + x2 + x3;
    }
};

inline std::mutex& probe_cache_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<std::string, DeviceProbeResult>& probe_cache_map() {
    static std::unordered_map<std::string, DeviceProbeResult> cache;
    return cache;
}

} // namespace detail

SYCL_FFT_DEVICE_LINKAGE DeviceProbeResult probe_subgroup_topology(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
    const std::string key = dev_cfg.identity_key();
    {
        std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
        auto it = detail::probe_cache_map().find(key);
        if (it != detail::probe_cache_map().end() && it->second.sg_probed) {
            return it->second;
        }
    }

    DeviceProbeResult res{};
    res.sg_probed = true;
    const size_t probe_wg = dev_cfg.clamp_pow2_wg(64, 64);

    uint32_t* d_out = nullptr;
    try {
        d_out = sycl::malloc_device<uint32_t>(4, q);
        if (d_out) {
            const uint32_t h_init[4] = {0u, 1u, 1u, 0u};
            q.memcpy(d_out, h_init, sizeof(h_init)).wait();
            q.submit([&](sycl::handler& h) {
                h.parallel_for(
                    sycl::nd_range<1>(probe_wg, probe_wg),
                    detail::SubgroupTopologyProbeKernel{d_out, static_cast<int>(probe_wg)});
            }).wait_and_throw();
            uint32_t h_out[4] = {0u, 0u, 0u, 0u};
            q.memcpy(h_out, d_out, sizeof(h_out)).wait();
            sycl::free(d_out, q);
            d_out = nullptr;

            res.raw_sg_size = static_cast<size_t>(h_out[0]);
            res.sg_linear_ok = (h_out[1] != 0u);
            res.sg_xor_ok = (h_out[2] != 0u);
            const bool pow2 = (res.raw_sg_size > 0) && ((res.raw_sg_size & (res.raw_sg_size - 1)) == 0);
            res.effective_sg_size = (pow2 && res.sg_linear_ok && res.sg_xor_ok) ? res.raw_sg_size : 1;
        }
    } catch (...) {
        if (d_out) sycl::free(d_out, q);
        res.raw_sg_size = 1;
        res.effective_sg_size = 1;
        res.sg_linear_ok = false;
        res.sg_xor_ok = false;
    }

    std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
    auto& entry = detail::probe_cache_map()[key];
    entry.sg_probed = true;
    entry.raw_sg_size = res.raw_sg_size;
    entry.effective_sg_size = res.effective_sg_size;
    entry.sg_linear_ok = res.sg_linear_ok;
    entry.sg_xor_ok = res.sg_xor_ok;
    return entry;
}

SYCL_FFT_DEVICE_LINKAGE DeviceProbeResult probe_df_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
    const std::string key = dev_cfg.identity_key();
    {
        std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
        auto it = detail::probe_cache_map().find(key);
        if (it != detail::probe_cache_map().end() && it->second.df_kat_probed) {
            return it->second;
        }
    }

    bool passed = false;
    std::string err_msg;

    constexpr double kA = 0.7071067811865475244;
    constexpr double kB = 0.3826834323650897717;
    constexpr double kC = -0.9238795325112867561;
    constexpr double kD = 0.1950903220161282678;
    const df_t da = df_from_double(kA);
    const df_t db = df_from_double(kB);
    const df_t dc = df_from_double(kC);
    const df_t dd = df_from_double(kD);

    const float h_in[14] = {
        1.0f, 3.0f * std::ldexp(1.0f, -24),                        // TwoSum
        1.5f, 3.0f * std::ldexp(1.0f, -24),                        // TwoDiff
        1.0f + std::ldexp(1.0f, -12), 1.0f + std::ldexp(1.0f, -13), // TwoProd
        da.hi, da.lo, db.hi, db.lo, dc.hi, dc.lo, dd.hi, dd.lo
    };

    float* d_in = nullptr;
    float* d_out = nullptr;
    try {
        d_in = sycl::malloc_device<float>(14, q);
        d_out = sycl::malloc_device<float>(14, q);
        if (!d_in || !d_out) {
            err_msg = "allocation failure during DF KAT";
        } else {
            q.memcpy(d_in, h_in, sizeof(h_in)).wait();
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(1), detail::DfKatProbeKernel{d_in, d_out});
            }).wait_and_throw();
            float h_out[14] = {};
            q.memcpy(h_out, d_out, sizeof(h_out)).wait();

            const float exp_ts_r = 1.0f + std::ldexp(1.0f, -22);
            const float exp_ts_e = -std::ldexp(1.0f, -24);
            const float exp_td_r = 1.5f - std::ldexp(1.0f, -22);
            const float exp_td_e = std::ldexp(1.0f, -24);
            const float exp_tp_r = 1.0f + std::ldexp(1.0f, -12) + std::ldexp(1.0f, -13);
            const float exp_tp_e = std::ldexp(1.0f, -25);

            auto fmt_err = [](const char* op, double got_r, double got_e, double exp_r, double exp_e) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "%s mismatch (got r=%.9e e=%.9e, expected r=%.9e e=%.9e)",
                              op, got_r, got_e, exp_r, exp_e);
                return std::string(buf);
            };

            if (h_out[0] != exp_ts_r || h_out[1] != exp_ts_e) {
                err_msg = fmt_err("TwoSum", h_out[0], h_out[1], exp_ts_r, exp_ts_e);
            } else if (h_out[2] != exp_td_r || h_out[3] != exp_td_e) {
                err_msg = fmt_err("TwoDiff", h_out[2], h_out[3], exp_td_r, exp_td_e);
            } else if (h_out[4] != exp_tp_r || h_out[5] != exp_tp_e) {
                err_msg = fmt_err("TwoProd (FMA contraction suspected)", h_out[4], h_out[5], exp_tp_r, exp_tp_e);
            } else {
                const double a_d = df_to_double(da), b_d = df_to_double(db);
                const double c_d = df_to_double(dc), d_d = df_to_double(dd);
                const double got_add = static_cast<double>(h_out[6]) + static_cast<double>(h_out[7]);
                const double got_mul = static_cast<double>(h_out[8]) + static_cast<double>(h_out[9]);
                const double got_cmr = static_cast<double>(h_out[10]) + static_cast<double>(h_out[11]);
                const double got_cmi = static_cast<double>(h_out[12]) + static_cast<double>(h_out[13]);
                const double ref_add = a_d + b_d;
                const double ref_mul = a_d * c_d;
                const double ref_cmr = a_d * c_d - b_d * d_d;
                const double ref_cmi = a_d * d_d + b_d * c_d;
                const double tol = 1e-13;

                if (std::fabs(got_add - ref_add) > tol) {
                    err_msg = fmt_err("df_add", got_add, 0.0, ref_add, 0.0);
                } else if (std::fabs(got_mul - ref_mul) > tol) {
                    err_msg = fmt_err("df_mul", got_mul, 0.0, ref_mul, 0.0);
                } else if (std::fabs(got_cmr - ref_cmr) > tol || std::fabs(got_cmi - ref_cmi) > tol) {
                    err_msg = fmt_err("df_cmul_fused", got_cmr, got_cmi, ref_cmr, ref_cmi);
                } else {
                    passed = true;
                }
            }
        }
    } catch (const std::exception& ex) {
        err_msg = std::string("kernel exception: ") + ex.what();
    }
    if (d_in) sycl::free(d_in, q);
    if (d_out) sycl::free(d_out, q);

    std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
    auto& entry = detail::probe_cache_map()[key];
    entry.df_kat_probed = true;
    entry.df_kat_passed = passed;
    entry.df_kat_error = err_msg;
    return entry;
}

SYCL_FFT_DEVICE_LINKAGE DeviceProbeResult probe_fp16_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
    const std::string key = dev_cfg.identity_key();
    {
        std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
        auto it = detail::probe_cache_map().find(key);
        if (it != detail::probe_cache_map().end() && it->second.fp16_kat_probed) {
            return it->second;
        }
    }

    bool passed = false;
    std::string err_msg;

#if !defined(SYCL_IMPLEMENTATION_ACPP) && !defined(SYCL_IMPLEMENTATION_HIPSYCL)
    if (!dev_cfg.has_fp16_aspect) {
        err_msg = "device lacks sycl::aspect::fp16";
        std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
        auto& entry = detail::probe_cache_map()[key];
        entry.fp16_kat_probed = true;
        entry.fp16_kat_passed = false;
        entry.fp16_kat_error = err_msg;
        return entry;
    }
#endif

    const Complexh h_in0[3] = {
        complexh_from_doubles(1.25, -0.75),
        complexh_from_doubles(0.375, 0.625),
        complexh_from_doubles(0.70703125, -0.70703125)
    };
    const Complexh h_in1[3] = {
        complexh_from_doubles(-0.5, 0.875),
        complexh_from_doubles(1.125, -0.25),
        complexh_from_doubles(0.70703125, -0.70703125)
    };

    Complexh* d_in0 = nullptr;
    Complexh* d_in1 = nullptr;
    Complexh* d_out0 = nullptr;
    Complexh* d_out1 = nullptr;
    try {
        d_in0 = sycl::malloc_device<Complexh>(3, q);
        d_in1 = sycl::malloc_device<Complexh>(3, q);
        d_out0 = sycl::malloc_device<Complexh>(3, q);
        d_out1 = sycl::malloc_device<Complexh>(3, q);
        if (!d_in0 || !d_in1 || !d_out0 || !d_out1) {
            err_msg = "allocation failure during FP16 KAT";
        } else {
            q.memcpy(d_in0, h_in0, sizeof(h_in0)).wait();
            q.memcpy(d_in1, h_in1, sizeof(h_in1)).wait();
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(1), detail::Fp16KatProbeKernel{d_in0, d_in1, d_out0, d_out1});
            }).wait_and_throw();
            Complexh h_out0[3] = {}, h_out1[3] = {};
            q.memcpy(h_out0, d_out0, sizeof(h_out0)).wait();
            q.memcpy(h_out1, d_out1, sizeof(h_out1)).wait();

            auto check_lane = [&](const char* lane_tag, const Complexh in[3], const Complexh out[3]) -> bool {
                long double ar, ai, br, bi, wr, wi;
                complexh_to_long_doubles(in[0], ar, ai);
                complexh_to_long_doubles(in[1], br, bi);
                complexh_to_long_doubles(in[2], wr, wi);

                const double exp_r[3] = {
                    static_cast<double>(0.5L * (ar + br)),
                    static_cast<double>(0.5L * (ar - br) * wr - 0.5L * (ai - bi) * wi),
                    static_cast<double>(((-ar) + ai) * 0.7071067811865475244L)
                };
                const double exp_i[3] = {
                    static_cast<double>(0.5L * (ai + bi)),
                    static_cast<double>(0.5L * (ar - br) * wi + 0.5L * (ai - bi) * wr),
                    static_cast<double>((ai - (-ar)) * 0.7071067811865475244L)
                };
                constexpr double kTol = 3e-3;
                for (int i = 0; i < 3; ++i) {
                    long double gr, gi;
                    complexh_to_long_doubles(out[i], gr, gi);
                    if (!std::isfinite(static_cast<double>(gr)) || !std::isfinite(static_cast<double>(gi)) ||
                        std::fabs(static_cast<double>(gr) - exp_r[i]) > kTol ||
                        std::fabs(static_cast<double>(gi) - exp_i[i]) > kTol) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                                      "FP16 KAT %s op %d mismatch (got %.5f, %.5f; exp %.5f, %.5f)",
                                      lane_tag, i,
                                      static_cast<double>(gr), static_cast<double>(gi),
                                      exp_r[i], exp_i[i]);
                        err_msg = buf;
                        return false;
                    }
                }
                return true;
            };

            if (check_lane("lane0", h_in0, h_out0) && check_lane("lane1", h_in1, h_out1)) {
                passed = true;
            }
        }
    } catch (const std::exception& ex) {
        err_msg = std::string("kernel exception: ") + ex.what();
    }
    if (d_in0) sycl::free(d_in0, q);
    if (d_in1) sycl::free(d_in1, q);
    if (d_out0) sycl::free(d_out0, q);
    if (d_out1) sycl::free(d_out1, q);

    std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
    auto& entry = detail::probe_cache_map()[key];
    entry.fp16_kat_probed = true;
    entry.fp16_kat_passed = passed;
    entry.fp16_kat_error = err_msg;
    return entry;
}

SYCL_FFT_DEVICE_LINKAGE DeviceProbeResult probe_fp64_ratio(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
    const std::string key = dev_cfg.identity_key();
    {
        std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
        auto it = detail::probe_cache_map().find(key);
        if (it != detail::probe_cache_map().end() && it->second.fp64_probed) {
            return it->second;
        }
    }

    double ratio = 0.0;
    bool compute_heavy = true;

    if (dev_cfg.has_fp64) {
        const size_t num_items = std::clamp<size_t>(
            dev_cfg.max_compute_units * (dev_cfg.is_gpu ? size_t{512} : size_t{64}),
            size_t{256},
            size_t{65536});
        const size_t wg = dev_cfg.clamp_pow2_wg(num_items, 64);

        float* d_f32 = nullptr;
        double* d_f64 = nullptr;
        try {
            d_f32 = sycl::malloc_device<float>(num_items, q);
            d_f64 = sycl::malloc_device<double>(num_items, q);
            if (d_f32 && d_f64) {
                auto time_f32 = [&]() {
                    q.submit([&](sycl::handler& h) {
                        h.parallel_for(sycl::nd_range<1>(num_items, wg),
                                       detail::FmaThroughputProbeKernel<float>{d_f32});
                    }).wait_and_throw();
                };
                auto time_f64 = [&]() {
                    q.submit([&](sycl::handler& h) {
                        h.parallel_for(sycl::nd_range<1>(num_items, wg),
                                       detail::FmaThroughputProbeKernel<double>{d_f64});
                    }).wait_and_throw();
                };
                // Warmup JIT + clocks
                time_f32();
                time_f64();

                const auto t0 = std::chrono::steady_clock::now();
                time_f32();
                time_f32();
                const auto t1 = std::chrono::steady_clock::now();
                time_f64();
                time_f64();
                const auto t2 = std::chrono::steady_clock::now();

                const double ns32 = std::chrono::duration<double, std::nano>(t1 - t0).count();
                const double ns64 = std::chrono::duration<double, std::nano>(t2 - t1).count();
                ratio = ns32 / std::max(1.0, ns64);
                compute_heavy = (ratio < 0.15);
            }
        } catch (...) {
            ratio = 0.0;
            compute_heavy = dev_cfg.is_gpu;
        }
        if (d_f32) sycl::free(d_f32, q);
        if (d_f64) sycl::free(d_f64, q);
    }

    std::lock_guard<std::mutex> lk(detail::probe_cache_mutex());
    auto& entry = detail::probe_cache_map()[key];
    entry.fp64_probed = true;
    entry.fp64_to_fp32_ratio = ratio;
    entry.fp64_compute_heavy = compute_heavy;
    return entry;
}

#endif // SYCL_FFT_DEVICE_DECL_ONLY

// ============================================================================
// Persistent Disk + In-Memory Autotune Cache
//   Keyed on (schema, compiler, device, vendor, driver, backend, prec, sign, logN).
//   Atomic temp-file rename; degrades cleanly to in-memory cache on read-only FS.
// ============================================================================
struct AutotuneEntry {
    PlanExecMode mode = PlanExecMode::SinglePass;
    std::string  variant_id;
    int          log_n1 = 0;
    size_t       pass0_wg = 0;
    size_t       l2_chunk_bytes = 0;
};

class AutotuneCache {
public:
    static std::string resolve_cache_path(const std::string& explicit_path);
    static std::string make_key(
        const DeviceFftConfig& dev_cfg, const char* prec_tag, int logN, const char* slot_tag = "any");
    static bool get(const std::string& path, const std::string& key, AutotuneEntry& out);
    static void put(const std::string& path, const std::string& key, const AutotuneEntry& val);
    static void clear_memory_cache();

private:
    using EntryMap = std::unordered_map<std::string, AutotuneEntry>;

    static std::mutex& cache_mutex();
    static std::unordered_map<std::string, EntryMap>& stores();
    static EntryMap& get_store_locked(const std::string& path);
    static const char* mode_to_str(PlanExecMode m);
    static void flush_locked(const std::string& path, const EntryMap& map);
};

#if !defined(SYCL_FFT_DEVICE_DECL_ONLY)

SYCL_FFT_DEVICE_LINKAGE std::string AutotuneCache::resolve_cache_path(const std::string& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    if (const char* env_p = std::getenv("SYCL_FFT_CACHE_PATH"); env_p && env_p[0] != '\0') {
        return std::string(env_p);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
        return std::string(xdg) + "/toeplitz-ls/sycl_fft_cache_v1.txt";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::string(home) + "/.cache/toeplitz-ls/sycl_fft_cache_v1.txt";
    }
    return {};
}

SYCL_FFT_DEVICE_LINKAGE std::string AutotuneCache::make_key(
    const DeviceFftConfig& dev_cfg, const char* prec_tag, int logN, const char* slot_tag) {
    char vid[16];
    std::snprintf(vid, sizeof(vid), "0x%x", dev_cfg.vendor_id);
    std::ostringstream oss;
    oss << "v1\t" << compiler_id_tag()
        << "\t" << sanitize_cache_token(dev_cfg.device_name)
        << "\t" << vid
        << "\t" << sanitize_cache_token(dev_cfg.driver_version)
        << "\t" << dev_cfg.backend_tag
        << "\t" << prec_tag
        << "\t" << (slot_tag ? slot_tag : "any")
        << "\t" << logN;
    return oss.str();
}

SYCL_FFT_DEVICE_LINKAGE bool AutotuneCache::get(
    const std::string& path, const std::string& key, AutotuneEntry& out) {
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto& store = get_store_locked(path);
    auto it = store.find(key);
    if (it == store.end()) return false;
    out = it->second;
    return true;
}

SYCL_FFT_DEVICE_LINKAGE void AutotuneCache::put(
    const std::string& path, const std::string& key, const AutotuneEntry& val) {
    std::lock_guard<std::mutex> lk(cache_mutex());
    auto& store = get_store_locked(path);
    store[key] = val;
    if (!path.empty()) {
        flush_locked(path, store);
    }
}

SYCL_FFT_DEVICE_LINKAGE void AutotuneCache::clear_memory_cache() {
    std::lock_guard<std::mutex> lk(cache_mutex());
    stores().clear();
}

SYCL_FFT_DEVICE_LINKAGE std::mutex& AutotuneCache::cache_mutex() {
    static std::mutex m;
    return m;
}

SYCL_FFT_DEVICE_LINKAGE std::unordered_map<std::string, AutotuneCache::EntryMap>& AutotuneCache::stores() {
    static std::unordered_map<std::string, EntryMap> s;
    return s;
}

SYCL_FFT_DEVICE_LINKAGE AutotuneCache::EntryMap& AutotuneCache::get_store_locked(const std::string& path) {
    auto& all = stores();
    auto it = all.find(path);
    if (it != all.end()) return it->second;

    EntryMap& map = all[path];
    if (path.empty()) return map;

    std::ifstream ifs(path);
    if (!ifs.is_open()) return map;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        size_t pos = 0;
        while (pos <= line.size()) {
            const size_t tab = line.find('\t', pos);
            if (tab == std::string::npos) {
                cols.push_back(line.substr(pos));
                break;
            }
            cols.push_back(line.substr(pos, tab - pos));
            pos = tab + 1;
        }
        if (cols.size() != 14 || cols[0] != "v1") continue;

        std::string key = cols[0];
        for (size_t i = 1; i < 9; ++i) {
            key.push_back('\t');
            key += cols[i];
        }
        AutotuneEntry e{};
        if (cols[9] == "1pass") e.mode = PlanExecMode::SinglePass;
        else if (cols[9] == "2pass") e.mode = PlanExecMode::TwoPass;
        else if (cols[9] == "multistage") e.mode = PlanExecMode::MultiStageGlobal;
        else continue;

        e.variant_id = (cols[10] == "-") ? std::string{} : cols[10];
        try {
            e.log_n1 = std::stoi(cols[11]);
            e.pass0_wg = static_cast<size_t>(std::stoull(cols[12]));
            e.l2_chunk_bytes = static_cast<size_t>(std::stoull(cols[13]));
        } catch (...) {
            continue;
        }
        map[key] = std::move(e);
    }
    return map;
}

SYCL_FFT_DEVICE_LINKAGE const char* AutotuneCache::mode_to_str(PlanExecMode m) {
    switch (m) {
        case PlanExecMode::SinglePass:       return "1pass";
        case PlanExecMode::TwoPass:          return "2pass";
        case PlanExecMode::MultiStageGlobal: return "multistage";
    }
    return "1pass";
}

SYCL_FFT_DEVICE_LINKAGE void AutotuneCache::flush_locked(const std::string& path, const EntryMap& map) {
    try {
        const std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
#if !defined(_WIN32)
        const long pid = static_cast<long>(::getpid());
#else
        const long pid = 0;
#endif
        const std::string tmp = path + ".tmp." + std::to_string(pid);
        {
            std::ofstream ofs(tmp, std::ios::out | std::ios::trunc);
            if (!ofs.is_open()) return;
            ofs << "# sycl_fft autotune cache v1\n";
            for (const auto& kv : map) {
                const auto& e = kv.second;
                ofs << kv.first
                    << '\t' << mode_to_str(e.mode)
                    << '\t' << (e.variant_id.empty() ? "-" : e.variant_id)
                    << '\t' << e.log_n1
                    << '\t' << e.pass0_wg
                    << '\t' << e.l2_chunk_bytes
                    << '\n';
            }
            ofs.flush();
            if (!ofs.good()) return;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
        }
    } catch (...) {
        // Silently degrade to in-memory cache on read-only filesystems.
    }
}

#endif // !SYCL_FFT_DEVICE_DECL_ONLY

#undef SYCL_FFT_DEVICE_LINKAGE
#undef SYCL_FFT_DEVICE_DECL_ONLY

} // namespace sycl_fft
