#pragma once

#include "common/layout.hpp"

#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <vector>

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

std::string sanitize_cache_token(const std::string& raw);
std::string compiler_id_tag();
std::string backend_to_tag(const sycl::device& dev);

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

    std::string identity_key() const;
    static DeviceFftConfig query(const sycl::device& dev);
    bool supports_sub_group_size(size_t s) const;
    size_t pick_sub_group_size(size_t min_req) const;
    size_t clamp_pow2_wg(size_t total_threads, size_t requested_wg, size_t hard_cap = 0) const;
    size_t default_wg_size(size_t total_threads, size_t reg_wg_cap = 256) const;
    size_t effective_l2_bytes() const;
    size_t compute_l2_chunk_bytes(
        size_t fft_bytes,
        size_t n1,
        bool is_compute_heavy,
        size_t l2_num = 1,
        size_t l2_den = 4,
        size_t min_waves_per_cu = 0) const;
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

    bool has_plan_overrides() const;
    bool has_bitrev_overrides() const;
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
// Always-On Process-Cached Hardware Probes (Sub-Group Topology, DF/FP16 KAT, FP64 Ratio)
//   Implemented in common/device.cpp (single isolated HCF module).
// ============================================================================
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

DeviceProbeResult probe_subgroup_topology(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_df_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_fp16_kat(sycl::queue& q, const DeviceFftConfig& dev_cfg);
DeviceProbeResult probe_fp64_ratio(sycl::queue& q, const DeviceFftConfig& dev_cfg);

// ============================================================================
// Persistent Disk + In-Memory Autotune Cache
//   Keyed on (schema, compiler, device, vendor, driver, backend, prec, slot, logN).
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
};

// Move-only RAII wrapper for SYCL USM device allocations.
template <typename T>
class UsmPtr {
public:
    UsmPtr() = default;
    UsmPtr(T* ptr, sycl::queue* q) noexcept : ptr_(ptr), q_(q) {}
    ~UsmPtr() { reset(); }

    UsmPtr(const UsmPtr&) = delete;
    UsmPtr& operator=(const UsmPtr&) = delete;

    UsmPtr(UsmPtr&& other) noexcept : ptr_(other.ptr_), q_(other.q_) {
        other.ptr_ = nullptr;
        other.q_ = nullptr;
    }

    UsmPtr& operator=(UsmPtr&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            q_ = other.q_;
            other.ptr_ = nullptr;
            other.q_ = nullptr;
        }
        return *this;
    }

    static UsmPtr alloc_device(size_t count, sycl::queue& q) {
        if (count == 0) return UsmPtr{};
        return UsmPtr(sycl::malloc_device<T>(count, q), &q);
    }

    void reset(T* ptr = nullptr, sycl::queue* q = nullptr) noexcept {
        if (ptr_ && q_) {
            try {
                sycl::free(ptr_, *q_);
            } catch (...) {
            }
        }
        ptr_ = ptr;
        q_ = q;
    }

    T* release() noexcept {
        T* p = ptr_;
        ptr_ = nullptr;
        q_ = nullptr;
        return p;
    }

    T* get() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    sycl::queue* q_ = nullptr;
};

} // namespace sycl_fft
