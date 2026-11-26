// Dedicated compilation unit for SYCL device configuration, hardware probes,
// and persistent AutotuneCache.
// Keeps all hardware probe kernels in a single tiny HCF module instead of
// duplicating them across caller translation units.
#include "common/device.hpp"
#include "common/math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace sycl_fft {

std::string sanitize_cache_token(const std::string& raw) {
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

std::string compiler_id_tag() {
    std::string tag;
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
#if defined(ACPP_VERSION_MAJOR) && defined(ACPP_VERSION_MINOR)
    tag = "acpp-" + std::to_string(ACPP_VERSION_MAJOR) + "." + std::to_string(ACPP_VERSION_MINOR);
#if defined(ACPP_VERSION_PATCH)
    tag += "." + std::to_string(ACPP_VERSION_PATCH);
#endif
#if defined(ACPP_VERSION_SUFFIX)
    tag += ACPP_VERSION_SUFFIX;
#endif
#else
    tag = "acpp";
#endif
#elif defined(__INTEL_LLVM_COMPILER)
    tag = "dpcpp-" + std::to_string(__INTEL_LLVM_COMPILER);
#elif defined(__SYCL_COMPILER_VERSION)
    tag = "sycl-" + std::to_string(__SYCL_COMPILER_VERSION);
#else
    tag = "sycl-generic";
#endif
#if defined(__clang_major__) && defined(__clang_minor__)
    tag += "/clang-" + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#endif
    return sanitize_cache_token(tag);
}

std::string kernel_code_hash() {
#if defined(SYCL_FFT_KERNEL_HASH)
    return sanitize_cache_token(SYCL_FFT_KERNEL_HASH);
#else
    // Fallback deterministic compile-unit fingerprint if built outside Makefile
    uint64_t h = 14695981039346656037ULL;
    auto mix = [&](const char* s) {
        for (; s && *s; ++s) {
            h ^= static_cast<unsigned char>(*s);
            h *= 1099511628211ULL;
        }
    };
    mix(__DATE__);
    mix(__TIME__);
#if defined(__VERSION__)
    mix(__VERSION__);
#endif
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
#endif
}

std::string backend_to_tag(const sycl::device& dev) {
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
// DeviceFftConfig & FftLaunchConfig Methods
// ============================================================================
std::string DeviceFftConfig::identity_key() const {
    char vid[48];
    std::snprintf(vid, sizeof(vid), "0x%x|cu%zu_smem%zu", vendor_id, max_compute_units, local_mem_size);
    return sanitize_cache_token(device_name) + "|" +
           std::string(vid) + "|" +
           sanitize_cache_token(driver_version) + "|" +
           backend_tag;
}

DeviceFftConfig DeviceFftConfig::query(const sycl::device& dev) {
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
    try {
        cfg.has_host_unified_memory = dev.get_info<sycl::info::device::host_unified_memory>();
    } catch (...) {
        cfg.has_host_unified_memory = !cfg.is_gpu;
    }

    // Query backend-reported kernel & memcpy concurrency when available
    cfg.max_kernel_concurrency = cfg.is_gpu ? size_t{2} : size_t{1};
    cfg.max_memcpy_concurrency = cfg.is_gpu ? size_t{2} : size_t{1};
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
    try {
        const auto rt_id = dev.AdaptiveCpp_device_id();
        auto* rt_ptr = dev.AdaptiveCpp_runtime();
        if (rt_ptr) {
            auto* be = rt_ptr->backends().get(rt_id.get_backend());
            if (be && be->get_hardware_manager()) {
                auto* hw = be->get_hardware_manager()->get_device(rt_id.get_id());
                if (hw) {
                    cfg.max_kernel_concurrency = std::max<size_t>(1, hw->get_max_kernel_concurrency());
                    cfg.max_memcpy_concurrency = std::max<size_t>(1, hw->get_max_memcpy_concurrency());
                }
            }
        }
    } catch (...) {
    }
#endif

    if (!cfg.has_native_local_mem || !cfg.is_gpu) {
        cfg.bank_model = BankModelKind::None;
    } else if (cfg.vendor_id == 0x8086u && cfg.max_work_group_size <= 256) {
        cfg.bank_model = BankModelKind::B16x4;
    } else {
        cfg.bank_model = BankModelKind::B32x4;
    }
    return cfg;
}

bool DeviceFftConfig::supports_sub_group_size(size_t s) const {
    return std::binary_search(sub_group_sizes.begin(), sub_group_sizes.end(), s);
}

size_t DeviceFftConfig::pick_sub_group_size(size_t min_req) const {
    for (size_t s : sub_group_sizes) {
        if (s >= min_req && (s & (s - 1)) == 0) {
            return s;
        }
    }
    return 0;
}

size_t DeviceFftConfig::clamp_pow2_wg(
    size_t total_threads, size_t requested_wg, size_t hard_cap) const {
    size_t cap = std::min(requested_wg, max_work_group_size);
    if (hard_cap > 0) cap = std::min(cap, hard_cap);
    if (total_threads == 0) return std::max<size_t>(1, cap);
    cap = std::min(cap, total_threads);
    size_t wg = 1;
    while ((wg << 1) <= cap) wg <<= 1;
    while (wg > 1 && (total_threads % wg) != 0) wg >>= 1;
    return wg;
}

size_t DeviceFftConfig::default_wg_size(size_t total_threads, size_t reg_wg_cap) const {
    const size_t sg_hint = is_gpu ? std::min<size_t>(sub_group_size, 64) : size_t{8};
    const size_t preferred = std::max<size_t>(64, sg_hint * 8);
    return clamp_pow2_wg(total_threads, preferred, reg_wg_cap);
}

size_t DeviceFftConfig::default_padded_wg_size(size_t reg_wg_cap) const {
    const size_t sg_hint = is_gpu ? std::min<size_t>(sub_group_size, 64) : size_t{8};
    const size_t preferred = std::max<size_t>(64, sg_hint * 8);
    size_t cap = std::min(preferred, max_work_group_size);
    if (reg_wg_cap > 0) cap = std::min(cap, reg_wg_cap);
    size_t wg = 1;
    while ((wg << 1) <= cap) wg <<= 1;
    return wg;
}

size_t DeviceFftConfig::safe_max_alloc_bytes() const {
    const size_t gmem = global_mem_size;
    const size_t amax = max_mem_alloc_size;
    if (gmem > 0 && amax > 0) return std::min(gmem, amax);
    if (gmem > 0) return gmem;
    if (amax > 0 && amax < (std::numeric_limits<size_t>::max() >> 2)) return amax;
    return size_t{256} * 1024 * 1024;
}

size_t DeviceFftConfig::recommended_stream_count() const {
    // Non-GPU devices (OpenMP / OpenCL CPU): each SYCL kernel already fans out across
    // all max_compute_units CPU threads unless the backend explicitly reports >1 concurrent kernels.
    if (!is_gpu) {
        return std::clamp<size_t>(max_kernel_concurrency, size_t{1}, std::max<size_t>(1, max_compute_units));
    }

    // If hardware reports strictly serial kernel & copy execution, a single stream avoids overhead.
    if (max_kernel_concurrency <= 1 && max_memcpy_concurrency <= 1) {
        return 1;
    }

    // Scale GPU streams from hardware concurrency engines and compute-unit count:
    //   - overlap_floor ensures enough queues to overlap concurrent kernels + DMA engines
    //   - cu_streams keeps at least ~4 CUs per active stream on wider GPUs
    //   - host_cap avoids allocating idle queues beyond host CPU thread count
    const size_t overlap_floor = std::max<size_t>(2, max_kernel_concurrency + max_memcpy_concurrency);
    const size_t cu_streams    = std::max<size_t>(1, max_compute_units / 4);
    const size_t host_threads  = std::max<size_t>(4, static_cast<size_t>(std::thread::hardware_concurrency()));
    size_t streams = std::min<size_t>({std::max(overlap_floor, cu_streams), host_threads, size_t{16}});

    // Ensure each stream gets a meaningful working slice of the 75% device memory pool
    if (global_mem_size > 0) {
        const size_t usable_mem = (global_mem_size / 4) * 3;
        const size_t min_slice = std::max<size_t>(size_t{64} * 1024 * 1024, effective_l2_bytes() * 4);
        const size_t mem_cap = std::max<size_t>(1, usable_mem / std::max<size_t>(1, min_slice));
        streams = std::min(streams, mem_cap);
    }
    return std::max<size_t>(1, streams);
}

size_t DeviceFftConfig::per_stream_mem_budget(size_t num_streams) const {
    const size_t k = std::max<size_t>(1, num_streams);
    if (global_mem_size > 0) {
        // Overflow-safe 75% of maximum device memory divided by number of streams
        const size_t total_75 = (global_mem_size / 4) * 3 + ((global_mem_size % 4) * 3) / 4;
        return std::max<size_t>(size_t{1} * 1024 * 1024, total_75 / k);
    }
    return size_t{256} * 1024 * 1024;
}

size_t DeviceFftConfig::effective_l2_bytes() const {
    if (l2_cache_size > 0) return l2_cache_size;
    size_t fallback = is_gpu ? (size_t{16} * 1024 * 1024) : (size_t{4} * 1024 * 1024);
    if (global_mem_size > 0) {
        fallback = std::clamp<size_t>(
            global_mem_size / 256, size_t{2} * 1024 * 1024, size_t{32} * 1024 * 1024);
    }
    return fallback;
}

size_t DeviceFftConfig::compute_l2_chunk_bytes(
    size_t fft_bytes,
    size_t n1,
    bool is_compute_heavy,
    size_t l2_num,
    size_t l2_den,
    size_t min_waves_per_cu) const {
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

bool FftLaunchConfig::has_plan_overrides() const {
    return l2_chunk_bytes > 0 || pass0_wg_size > 0 || two_pass_min_logN > 0 ||
           two_pass_log_n1 > 0 || !allow_wide_kernels || force_multistage ||
           allow_emulated_local_mem || fp64_compute_heavy_override >= 0 ||
           stash_8192_override >= 0 || !variant_override.empty();
}

bool FftLaunchConfig::has_bitrev_overrides() const {
    return bitrev_wg_size > 0 || !bitrev_variant_override.empty() || allow_emulated_local_mem;
}

// ============================================================================
// Hardware Probe Kernels & Process-Cached Probe Dispatchers
// ============================================================================
namespace detail {

struct SubgroupTopologyProbeKernel {
    uint32_t* out;
    int probe_wg;

    void operator()(sycl::nd_item<1> item) const {
        const int t = static_cast<int>(item.get_local_linear_id());
        auto sg = item.get_sub_group();
        const uint32_t sg_sz = static_cast<uint32_t>(sg.get_max_local_range()[0]);
        auto mark_fail = [&](int idx) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(out[idx]).store(0u);
        };
        if (t == 0) {
            out[0] = sg_sz;
        }
        const bool pow2 = (sg_sz > 0) && ((sg_sz & (sg_sz - 1u)) == 0u);
        if (!pow2 || sg.get_local_linear_id() != (static_cast<uint32_t>(t) & (sg_sz - 1u))) {
            mark_fail(1);
        }
        const uint32_t max_h = (sg_sz < static_cast<uint32_t>(probe_wg))
            ? sg_sz : static_cast<uint32_t>(probe_wg);
        const uint32_t self = static_cast<uint32_t>(t);
        if (max_h >= 2u) {
            if (sycl::permute_group_by_xor(sg, self, 1u) != (self ^ 1u)) mark_fail(2);
        }
        if (max_h >= 4u) {
            if (sycl::permute_group_by_xor(sg, self, 2u) != (self ^ 2u)) mark_fail(2);
        }
        if (max_h >= 8u) {
            if (sycl::permute_group_by_xor(sg, self, 4u) != (self ^ 4u)) mark_fail(2);
        }
        if (max_h >= 16u) {
            if (sycl::permute_group_by_xor(sg, self, 8u) != (self ^ 8u)) mark_fail(2);
        }
        if (max_h >= 32u) {
            if (sycl::permute_group_by_xor(sg, self, 16u) != (self ^ 16u)) mark_fail(2);
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
        for (int it = 0; it < 2048; ++it) {
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

std::mutex& probe_cache_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, DeviceProbeResult>& probe_cache_map() {
    static std::unordered_map<std::string, DeviceProbeResult> cache;
    return cache;
}

} // namespace detail

DeviceProbeResult probe_subgroup_topology(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
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
#if !SYCL_FFT_HAS_REQD_SG_SIZE
            // On variable-SIMD GPUs (e.g. Intel Xe/Arc/PVC supporting {8, 16, 32}),
            // without [[sycl::reqd_sub_group_size]] the compiler may lower register-heavy
            // FFT kernels to a narrower SIMD width than the tiny probe kernel.
            // Clamp effective_sg_size to the device minimum so SubgroupShuffle is only
            // selected when guaranteed safe across all register pressures.
            if (dev_cfg.sub_group_sizes.size() > 1 && res.effective_sg_size > dev_cfg.min_sub_group_size) {
                res.effective_sg_size = dev_cfg.min_sub_group_size;
            }
#endif
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

DeviceProbeResult probe_df_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
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

DeviceProbeResult probe_fp16_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
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

DeviceProbeResult probe_fp64_ratio(sycl::queue& q, const DeviceFftConfig& dev_cfg) {
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
                auto run_f32_batch = [&](int reps) {
                    sycl::event ev{};
                    for (int r = 0; r < reps; ++r) {
                        ev = q.submit([&](sycl::handler& h) {
                            if (r > 0) h.depends_on(ev);
                            h.parallel_for(sycl::nd_range<1>(num_items, wg),
                                           detail::FmaThroughputProbeKernel<float>{d_f32});
                        });
                    }
                    ev.wait_and_throw();
                };
                auto run_f64_batch = [&](int reps) {
                    sycl::event ev{};
                    for (int r = 0; r < reps; ++r) {
                        ev = q.submit([&](sycl::handler& h) {
                            if (r > 0) h.depends_on(ev);
                            h.parallel_for(sycl::nd_range<1>(num_items, wg),
                                           detail::FmaThroughputProbeKernel<double>{d_f64});
                        });
                    }
                    ev.wait_and_throw();
                };
                // Warmup JIT + clocks
                run_f32_batch(1);
                run_f64_batch(1);

                constexpr int kBatchReps = 8;
                const auto t0 = std::chrono::steady_clock::now();
                run_f32_batch(kBatchReps);
                const auto t1 = std::chrono::steady_clock::now();
                run_f64_batch(kBatchReps);
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

// ============================================================================
// Persistent Disk + In-Memory Autotune Cache
// ============================================================================
namespace {

using EntryMap = std::unordered_map<std::string, AutotuneEntry>;

std::mutex& autotune_cache_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, EntryMap>& autotune_cache_stores() {
    static std::unordered_map<std::string, EntryMap> s;
    return s;
}

const char* plan_mode_to_str(PlanExecMode m) {
    switch (m) {
        case PlanExecMode::SinglePass:       return "1pass";
        case PlanExecMode::TwoPass:          return "2pass";
        case PlanExecMode::MultiStageGlobal: return "multistage";
    }
    return "1pass";
}

void parse_autotune_file(const std::string& path, EntryMap& map) {
    if (path.empty()) return;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;

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
        // v2 schema:
        //   cols[0..11] = key (v2, compiler_jit, code_hash, dev_name, vendor_id, cu_smem,
        //                      driver, backend, prec, sign, slot, logN)
        //   cols[12..18] = val (mode, variant_id, log_n1, pass0_wg, l2_chunk_bytes,
        //                       margin_pct, runner_up_id)
        if (cols.size() != 19 || cols[0] != "v2") continue;

        std::string key = cols[0];
        for (size_t i = 1; i < 12; ++i) {
            key.push_back('\t');
            key += cols[i];
        }
        AutotuneEntry e{};
        if (cols[12] == "1pass") e.mode = PlanExecMode::SinglePass;
        else if (cols[12] == "2pass") e.mode = PlanExecMode::TwoPass;
        else if (cols[12] == "multistage") e.mode = PlanExecMode::MultiStageGlobal;
        else continue;

        e.variant_id = (cols[13] == "-") ? std::string{} : cols[13];
        try {
            e.log_n1 = std::stoi(cols[14]);
            e.pass0_wg = static_cast<size_t>(std::stoull(cols[15]));
            e.l2_chunk_bytes = static_cast<size_t>(std::stoull(cols[16]));
            e.margin_pct = std::stod(cols[17]);
        } catch (...) {
            continue;
        }
        e.runner_up_id = (cols[18] == "-") ? std::string{} : cols[18];
        map[key] = std::move(e);
    }
}

EntryMap& get_autotune_store_locked(const std::string& path) {
    auto& all = autotune_cache_stores();
    auto it = all.find(path);
    if (it != all.end()) return it->second;

    EntryMap& map = all[path];
    parse_autotune_file(path, map);
    return map;
}

void flush_autotune_store_locked(const std::string& path, EntryMap& map) {
    try {
        const std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
#if !defined(_WIN32)
        const std::string lock_path = path + ".lock";
        const int lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR, 0644);
        if (lock_fd >= 0) {
            (void)::flock(lock_fd, LOCK_EX);
        }
        struct LockGuard {
            int fd;
            ~LockGuard() {
                if (fd >= 0) {
                    (void)::flock(fd, LOCK_UN);
                    ::close(fd);
                }
            }
        } lg{lock_fd};

        // Re-read on-disk entries written by concurrent processes, then overlay our in-memory updates.
        EntryMap merged;
        parse_autotune_file(path, merged);
        for (const auto& kv : map) {
            merged[kv.first] = kv.second;
        }
        map = merged;
        const long pid = static_cast<long>(::getpid());
#else
        const EntryMap& merged = map;
        const long pid = 0;
#endif
        const std::string tmp = path + ".tmp." + std::to_string(pid);
        {
            std::ofstream ofs(tmp, std::ios::out | std::ios::trunc);
            if (!ofs.is_open()) return;
            ofs << "# sycl_fft autotune cache v2\n";
            ofs << "# schema\tcompiler_jit\tcode_hash\tdevice\tvendor\tcu_smem\tdriver\tbackend\tprec\tsign\tslot\tlogN\tmode\tvariant\tlog_n1\tpass0_wg\tl2_chunk\tmargin_pct\trunner_up\n";
            for (const auto& kv : merged) {
                const auto& e = kv.second;
                char mbuf[32];
                std::snprintf(mbuf, sizeof(mbuf), "%.2f", e.margin_pct);
                ofs << kv.first
                    << '\t' << plan_mode_to_str(e.mode)
                    << '\t' << (e.variant_id.empty() ? "-" : e.variant_id)
                    << '\t' << e.log_n1
                    << '\t' << e.pass0_wg
                    << '\t' << e.l2_chunk_bytes
                    << '\t' << mbuf
                    << '\t' << (e.runner_up_id.empty() ? "-" : e.runner_up_id)
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

} // namespace

std::string AutotuneCache::resolve_cache_path(const std::string& explicit_path) {
    if (!explicit_path.empty()) return explicit_path;
    if (const char* env_p = std::getenv("SYCL_FFT_CACHE_PATH"); env_p && env_p[0] != '\0') {
        return std::string(env_p);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
        return std::string(xdg) + "/toeplitz-ls/sycl_fft_cache_v2.txt";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::string(home) + "/.cache/toeplitz-ls/sycl_fft_cache_v2.txt";
    }
    return {};
}

std::string AutotuneCache::make_key(
    const DeviceFftConfig& dev_cfg,
    const char* prec_tag,
    int logN,
    const char* slot_tag,
    int sign) {
    char vid[16];
    std::snprintf(vid, sizeof(vid), "0x%x", dev_cfg.vendor_id);
    char cu_smem[48];
    std::snprintf(cu_smem, sizeof(cu_smem), "cu%zu_smem%zu",
                  dev_cfg.max_compute_units, dev_cfg.local_mem_size);
    const char* sign_tag = (sign > 0) ? "pos" : ((sign < 0) ? "neg" : "any");
    std::ostringstream oss;
    oss << "v2\t" << compiler_id_tag()
        << "\t" << kernel_code_hash()
        << "\t" << sanitize_cache_token(dev_cfg.device_name)
        << "\t" << vid
        << "\t" << cu_smem
        << "\t" << sanitize_cache_token(dev_cfg.driver_version)
        << "\t" << dev_cfg.backend_tag
        << "\t" << (prec_tag ? prec_tag : "any")
        << "\t" << sign_tag
        << "\t" << (slot_tag ? slot_tag : "any")
        << "\t" << logN;
    return oss.str();
}

bool AutotuneCache::get(const std::string& path, const std::string& key, AutotuneEntry& out) {
    std::lock_guard<std::mutex> lk(autotune_cache_mutex());
    auto& store = get_autotune_store_locked(path);
    auto it = store.find(key);
    if (it == store.end()) return false;
    out = it->second;
    return true;
}

void AutotuneCache::put(const std::string& path, const std::string& key, const AutotuneEntry& val) {
    std::lock_guard<std::mutex> lk(autotune_cache_mutex());
    auto& store = get_autotune_store_locked(path);
    store[key] = val;
    if (!path.empty()) {
        flush_autotune_store_locked(path, store);
    }
}

void AutotuneCache::clear_memory_cache() {
    std::lock_guard<std::mutex> lk(autotune_cache_mutex());
    autotune_cache_stores().clear();
}

const std::vector<SyclDeviceDescriptor>& enumerate_sycl_devices() {
    static const std::vector<SyclDeviceDescriptor> devices = []() {
        std::vector<SyclDeviceDescriptor> out;
        try {
            auto raw = sycl::device::get_devices();
            out.reserve(raw.size());
            for (const auto& dev : raw) {
                SyclDeviceDescriptor d{};
                d.device = dev;
                d.name = sanitize_cache_token(dev.get_info<sycl::info::device::name>());
                d.backend = backend_to_tag(dev);
                d.is_gpu = dev.is_gpu();
                if (dev.is_gpu()) {
                    d.device_type = "gpu";
                } else if (dev.is_cpu()) {
                    d.device_type = "cpu";
                } else if (dev.is_accelerator()) {
                    d.device_type = "accelerator";
                } else {
                    d.device_type = "other";
                }
                d.max_compute_units = static_cast<size_t>(
                    dev.get_info<sycl::info::device::max_compute_units>());
                d.global_mem_size = static_cast<size_t>(
                    dev.get_info<sycl::info::device::global_mem_size>());
                d.has_fp64 = dev.has(sycl::aspect::fp64);
                d.has_fp16 = dev.has(sycl::aspect::fp16);
#if defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL)
                // AdaptiveCpp does not report sycl::aspect::fp16 on CUDA/HIP/OMP backends
                // even though sycl::half / f16x2 arithmetic is natively supported.
                if (!d.has_fp16 &&
                    (d.backend == "cuda" || d.backend == "hip" ||
                     d.backend == "level_zero" || d.backend == "metal" ||
                     d.backend == "omp")) {
                    d.has_fp16 = true;
                }
#endif
                out.push_back(std::move(d));
            }
        } catch (...) {
        }

        auto backend_rank = [](const std::string& b) -> int {
            if (b == "cuda" || b == "hip" || b == "level_zero") return 0;
            if (b == "metal" || b == "vulkan") return 1;
            if (b == "opencl") return 2;
            if (b == "omp") return 3;
            return 4;
        };

        std::stable_sort(out.begin(), out.end(), [&](const SyclDeviceDescriptor& a, const SyclDeviceDescriptor& b) {
            if (a.is_gpu != b.is_gpu) return a.is_gpu > b.is_gpu;
            const int ra = backend_rank(a.backend);
            const int rb = backend_rank(b.backend);
            if (ra != rb) return ra < rb;
            if (a.max_compute_units != b.max_compute_units) return a.max_compute_units > b.max_compute_units;
            if (a.global_mem_size != b.global_mem_size) return a.global_mem_size > b.global_mem_size;
            return a.name < b.name;
        });

        for (size_t i = 0; i < out.size(); ++i) {
            out[i].id = static_cast<int>(i);
        }
        return out;
    }();
    return devices;
}

} // namespace sycl_fft
