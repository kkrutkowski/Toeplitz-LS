// Pure-Host SYCL Multi-Stream Periodogram Engine (0 SYCL device kernels / 0 KiB HCF).
//
// Thread-safety & throughput design:
//   - Calling threads (e.g., Python ThreadPoolExecutor workers with GIL released) perform
//     all input validation and CPU preprocessing concurrently without holding any lock.
//   - Each SYCL device owns a lazy DeviceRuntime containing K reusable StreamSlots.
//   - Each StreamSlot owns:
//       * a dedicated in-order sycl::queue
//       * monotonically-grown pinned host staging buffers (sycl::malloc_host)
//       * monotonically-grown device USM workspace slabs (zero malloc_device in steady state)
//       * a cache of M-resizable Nufft1PlanBase instances keyed by (prec, N_block, mode, geo)
//   - DeviceManager is intentionally immortal (heap-allocated singleton) to avoid
//     destroying CUDA/SYCL USM handles during process teardown after driver unload.

#include "periodogram/engine.hpp"
#include "periodogram/planner.hpp"
#include "periodogram/preprocess.hpp"
#include "periodogram/subgrid.hpp"
#include "linalg/levinson.hpp"
#include "nufft/nufft1.hpp"
#include "common/device.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sycl_periodogram {

namespace {

thread_local std::string tls_last_err;

void set_last_error(const std::string& msg) noexcept {
    try {
        tls_last_err = msg;
    } catch (...) {
    }
}

void clear_last_error() noexcept {
    tls_last_err.clear();
}

// Monotonically growing raw byte buffer for USM device or pinned host memory
class GrowableDeviceBuf {
public:
    GrowableDeviceBuf() = default;
    ~GrowableDeviceBuf() { reset(); }
    GrowableDeviceBuf(const GrowableDeviceBuf&) = delete;
    GrowableDeviceBuf& operator=(const GrowableDeviceBuf&) = delete;

    size_t capacity() const noexcept { return cap_; }

    uint8_t* ensure(sycl::queue& q, size_t bytes, size_t max_cap = 0) {
        if (bytes == 0) return ptr_;
        if (bytes <= cap_) return ptr_;
        size_t new_cap = std::max(bytes, cap_ + (cap_ >> 1));
        if (max_cap >= bytes && new_cap > max_cap) {
            new_cap = max_cap;
        }
        if (ptr_) {
            q.wait();
            sycl::free(ptr_, q);
            ptr_ = nullptr;
            cap_ = 0;
        }
        ptr_ = sycl::malloc_device<uint8_t>(new_cap, q);
        if (!ptr_) {
            throw std::runtime_error("SYCL device workspace allocation failed");
        }
        q_ = &q;
        cap_ = new_cap;
        return ptr_;
    }

    void reset() noexcept {
        if (ptr_ && q_) {
            try { sycl::free(ptr_, *q_); } catch (...) {}
        }
        ptr_ = nullptr;
        q_ = nullptr;
        cap_ = 0;
    }

private:
    uint8_t*     ptr_ = nullptr;
    sycl::queue* q_   = nullptr;
    size_t       cap_ = 0;
};

class GrowableHostBuf {
public:
    GrowableHostBuf() = default;
    ~GrowableHostBuf() { reset(); }
    GrowableHostBuf(const GrowableHostBuf&) = delete;
    GrowableHostBuf& operator=(const GrowableHostBuf&) = delete;

    uint8_t* ensure(sycl::queue& q, size_t bytes) {
        if (bytes == 0) return ptr_;
        if (bytes <= cap_) return ptr_;
        const size_t new_cap = std::max(bytes, cap_ + (cap_ >> 1));
        if (ptr_) {
            q.wait();
            sycl::free(ptr_, q);
            ptr_ = nullptr;
            cap_ = 0;
        }
        ptr_ = sycl::malloc_host<uint8_t>(new_cap, q);
        if (!ptr_) {
            throw std::runtime_error("SYCL pinned host staging allocation failed");
        }
        q_ = &q;
        cap_ = new_cap;
        return ptr_;
    }

    void reset() noexcept {
        if (ptr_ && q_) {
            try { sycl::free(ptr_, *q_); } catch (...) {}
        }
        ptr_ = nullptr;
        q_ = nullptr;
        cap_ = 0;
    }

private:
    uint8_t*     ptr_ = nullptr;
    sycl::queue* q_   = nullptr;
    size_t       cap_ = 0;
};

struct PlanKey {
    int  N_block;
    int  mode;
    bool geo_fp16;
    bool operator==(const PlanKey& o) const noexcept {
        return N_block == o.N_block && mode == o.mode && geo_fp16 == o.geo_fp16;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const noexcept {
        return (static_cast<size_t>(k.N_block) << 2) ^
               (static_cast<size_t>(k.mode) << 1) ^
               static_cast<size_t>(k.geo_fp16 ? 1 : 0);
    }
};

struct StreamSlot {
    sycl::queue q;

    // Pinned host staging buffers
    GrowableHostBuf h_x;
    GrowableHostBuf h_power;
    GrowableHostBuf h_cond;

    // Reusable device USM workspace slabs
    GrowableDeviceBuf d_x;
    GrowableDeviceBuf d_src;
    GrowableDeviceBuf d_trig;
    GrowableDeviceBuf d_ihs_acc;
    GrowableDeviceBuf d_a_scratch;
    GrowableDeviceBuf d_power;
    GrowableDeviceBuf d_cond;

    // Cached M-resizable NuFFT-1 plans per precision
    std::unordered_map<PlanKey, std::unique_ptr<sycl_nufft::Nufft1Planh>,  PlanKeyHash> plans_h;
    std::unordered_map<PlanKey, std::unique_ptr<sycl_nufft::Nufft1Planf>,  PlanKeyHash> plans_f;
    std::unordered_map<PlanKey, std::unique_ptr<sycl_nufft::Nufft1Plandf>, PlanKeyHash> plans_df;
    std::unordered_map<PlanKey, std::unique_ptr<sycl_nufft::Nufft1Plan>,   PlanKeyHash> plans_d;
    size_t cached_plan_bytes = 0;

    explicit StreamSlot(const sycl::context& ctx, const sycl::device& dev)
        : q(ctx, dev, sycl::property::queue::in_order{}) {}

    size_t slab_bytes() const noexcept {
        return d_x.capacity() + d_src.capacity() + d_trig.capacity() +
               d_ihs_acc.capacity() + d_a_scratch.capacity() +
               d_power.capacity() + d_cond.capacity();
    }

    void clear_plan_caches() noexcept {
        try { q.wait(); } catch (...) {}
        plans_h.clear();
        plans_f.clear();
        plans_df.clear();
        plans_d.clear();
        cached_plan_bytes = 0;
    }

    // Reclaim unused/oversized device slabs (and, if needed, cached plans) so total
    // device memory owned by this slot stays within slot_budget (75% VRAM / streams).
    void enforce_budget(
        size_t slot_budget,
        size_t need_x,
        size_t need_src,
        size_t need_trig,
        size_t need_acc,
        size_t need_scr,
        size_t need_pow,
        size_t need_cond) {
        auto proj_one = [](const GrowableDeviceBuf& b, size_t need) -> size_t {
            return (need <= b.capacity()) ? b.capacity() : need;
        };
        const size_t projected =
            proj_one(d_x, need_x) + proj_one(d_src, need_src) + proj_one(d_trig, need_trig) +
            proj_one(d_ihs_acc, need_acc) + proj_one(d_a_scratch, need_scr) +
            proj_one(d_power, need_pow) + proj_one(d_cond, need_cond) + cached_plan_bytes;
        if (projected <= slot_budget) return;

        q.wait();
        auto trim = [](GrowableDeviceBuf& b, size_t need) {
            if (need == 0 || b.capacity() > need * 2) {
                b.reset();
            }
        };
        trim(d_x, need_x);
        trim(d_src, need_src);
        trim(d_trig, need_trig);
        trim(d_ihs_acc, need_acc);
        trim(d_a_scratch, need_scr);
        trim(d_power, need_pow);
        trim(d_cond, need_cond);

        const size_t after_trim =
            proj_one(d_x, need_x) + proj_one(d_src, need_src) + proj_one(d_trig, need_trig) +
            proj_one(d_ihs_acc, need_acc) + proj_one(d_a_scratch, need_scr) +
            proj_one(d_power, need_pow) + proj_one(d_cond, need_cond) + cached_plan_bytes;
        if (after_trim > slot_budget) {
            clear_plan_caches();
        }
    }

    template <typename PlanT>
    PlanT& get_or_create_plan(
        std::unordered_map<PlanKey, std::unique_ptr<PlanT>, PlanKeyHash>& cache,
        std::mutex& plan_create_mtx,
        size_t slot_budget,
        int M_hint,
        int N_block,
        int w,
        size_t plan_max_bs,
        sycl_nufft::PswfMode mode,
        bool geo_fp16) {
        const PlanKey key{N_block, static_cast<int>(mode), geo_fp16};
        auto it = cache.find(key);
        if (it != cache.end()) {
            return *it->second;
        }
        // Estimate device footprint of the new plan; evict stale plans if cache grows too large
        const size_t cbytes = sizeof(typename PlanT::ComplexT);
        const size_t sbytes = sizeof(typename PlanT::ScalarT);
        const size_t init_M = static_cast<size_t>(std::max(M_hint, 256));
        const size_t est_bytes =
            size_t{2} * plan_max_bs * static_cast<size_t>(2 * N_block) * cbytes +
            init_M * (size_t{4} + sbytes + cbytes + static_cast<size_t>(w) * sbytes) * size_t{3};
        if (slab_bytes() + cached_plan_bytes + est_bytes > slot_budget && cached_plan_bytes > 0) {
            clear_plan_caches();
        }

        // Serialize cold plan creation (and first-time autotuning) across slots on the same device
        std::lock_guard<std::mutex> create_lk(plan_create_mtx);
        sycl_nufft::Nufft1LaunchConfig cfg{};
        cfg.use_geo_scale_fp16 = geo_fp16;
        auto plan = std::make_unique<PlanT>(q, static_cast<int>(init_M), N_block, w, mode, plan_max_bs, cfg);
        PlanT& ref = *plan;
        cache.emplace(key, std::move(plan));
        cached_plan_bytes += est_bytes;
        return ref;
    }
};

struct DeviceRuntime {
    sycl_fft::SyclDeviceDescriptor desc;
    sycl_fft::DeviceFftConfig      dev_cfg;
    sycl::context                  ctx;
    int                            num_slots = 1;
    size_t                         slot_mem_budget = 0;

    std::vector<std::unique_ptr<StreamSlot>> slots;
    std::mutex                               pool_mtx;
    std::condition_variable                  pool_cv;
    std::vector<StreamSlot*>                 idle;
    std::mutex                               plan_create_mtx;

    std::mutex                               fp64_probe_mtx;
    std::atomic<bool>                        fp64_probed{false};
    double                                   fp64_to_fp32_ratio = 0.0;

    explicit DeviceRuntime(const sycl_fft::SyclDeviceDescriptor& d)
        : desc(d),
          dev_cfg(sycl_fft::DeviceFftConfig::query(d.device)),
          ctx(d.device) {
        const int env_k = env_int_or("TLS_SYCL_STREAMS", 0);
        if (env_k > 0) {
            num_slots = std::clamp(env_k, 1, 32);
        } else {
            num_slots = static_cast<int>(dev_cfg.recommended_stream_count());
        }
        slot_mem_budget = dev_cfg.per_stream_mem_budget(static_cast<size_t>(num_slots));
        slots.reserve(static_cast<size_t>(num_slots));
        idle.reserve(static_cast<size_t>(num_slots));
        for (int i = 0; i < num_slots; ++i) {
            auto s = std::make_unique<StreamSlot>(ctx, desc.device);
            idle.push_back(s.get());
            slots.push_back(std::move(s));
        }
    }

    double query_fp64_ratio(StreamSlot& slot) {
        if (fp64_probed.load(std::memory_order_acquire)) {
            return fp64_to_fp32_ratio;
        }
        std::lock_guard<std::mutex> lk(fp64_probe_mtx);
        if (!fp64_probed.load(std::memory_order_relaxed)) {
            const auto probe = sycl_fft::probe_fp64_ratio(slot.q, dev_cfg);
            fp64_to_fp32_ratio = probe.fp64_to_fp32_ratio;
            fp64_probed.store(true, std::memory_order_release);
        }
        return fp64_to_fp32_ratio;
    }

    double cached_fp64_ratio() const noexcept {
        return fp64_probed.load(std::memory_order_acquire) ? fp64_to_fp32_ratio : 0.0;
    }

    StreamSlot* acquire() {
        std::unique_lock<std::mutex> lk(pool_mtx);
        pool_cv.wait(lk, [&]() { return !idle.empty(); });
        StreamSlot* s = idle.back();
        idle.pop_back();
        return s;
    }

    void release(StreamSlot* s) noexcept {
        {
            std::lock_guard<std::mutex> lk(pool_mtx);
            idle.push_back(s);
        }
        pool_cv.notify_one();
    }
};

struct SlotGuard {
    DeviceRuntime* rt = nullptr;
    StreamSlot*    slot = nullptr;
    SlotGuard(DeviceRuntime& r) : rt(&r), slot(r.acquire()) {}
    ~SlotGuard() {
        if (rt && slot) rt->release(slot);
    }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;
};

class DeviceManager {
public:
    static DeviceManager& instance() {
        // Intentionally leaked singleton so USM handles are never destroyed after
        // CUDA/AdaptiveCpp runtime unload during process exit.
        static DeviceManager* mgr = new DeviceManager();
        return *mgr;
    }

    DeviceRuntime& get(int device_id) {
        const auto& devs = sycl_fft::enumerate_sycl_devices();
        if (device_id < 0 || static_cast<size_t>(device_id) >= devs.size()) {
            throw std::out_of_range(
                "Invalid SYCL device ID " + std::to_string(device_id) +
                " (available devices: " + std::to_string(devs.size()) + ")");
        }
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = runtimes_.find(device_id);
        if (it != runtimes_.end()) {
            return *it->second;
        }
        auto rt = std::make_unique<DeviceRuntime>(devs[static_cast<size_t>(device_id)]);
        DeviceRuntime& ref = *rt;
        runtimes_.emplace(device_id, std::move(rt));
        return ref;
    }

private:
    std::mutex mtx_;
    std::unordered_map<int, std::unique_ptr<DeviceRuntime>> runtimes_;
};

int parse_backend(int backend, sycl_nufft::PswfMode& out_mode) {
    if (backend == TLS_SYCL_BACKEND_PSWF43) {
        out_mode = sycl_nufft::PswfMode::PSWF43;
        return TLS_SYCL_OK;
    }
    if (backend == TLS_SYCL_BACKEND_PSWF21) {
        out_mode = sycl_nufft::PswfMode::PSWF21;
        return TLS_SYCL_OK;
    }
    return TLS_SYCL_ERR_BACKEND;
}

int parse_fastchi2_norm(int normalization, sycl_linalg::NormKind& out_norm) {
    if (normalization < 0 || normalization > 3) return TLS_SYCL_ERR_ARGUMENT;
    out_norm = static_cast<sycl_linalg::NormKind>(normalization);
    return TLS_SYCL_OK;
}

int parse_ihs_norm(int normalization, sycl_linalg::NormKind& out_norm) {
    if (normalization != 0 && normalization != 2 && normalization != 4) return TLS_SYCL_ERR_ARGUMENT;
    out_norm = static_cast<sycl_linalg::NormKind>(normalization);
    return TLS_SYCL_OK;
}

// ============================================================================
// Precision Traits bridging Engine -> NuFFT + Subgrid + Levinson/GLS/IHS
// ============================================================================
struct EngineHalfTraits {
    static constexpr PrecKind kPrec = TLS_SYCL_PREC_HALF;
    using PlanT      = sycl_nufft::Nufft1Planh;
    using XCoordT    = sycl_fft::df_t;
    using ScalarT    = sycl_fft::f16_scalar_t;
    using ComplexT   = sycl_fft::Complexh;
    using SolveRealT = sycl_fft::f16x2_t;
    using DevOutT    = float;
    using HostOutT   = float;
    using IhsAccT    = float;

    static auto& plan_cache(StreamSlot& s) { return s.plans_h; }

    static void build_subgrid(
        sycl::queue& q, const XCoordT* dx, const ScalarT* dw, const ScalarT* dyw,
        ComplexT* dsrc, int M, int b_start, int b_cnt, double qf0, double qbdf, size_t wg) {
        detail::launch_build_subgrid_sourcesh(q, dx, dw, dyw, dsrc, M, b_start, b_cnt, qf0, qbdf, wg);
    }

    static void solve_fastchi2(
        sycl::queue& q, const ComplexT* dtrig, const ScalarT* dphi, const ScalarT* ddec,
        SolveRealT* dscr, DevOutT* dpow, DevOutT* dcond, const sycl_linalg::FastChi2SolveParams& p) {
        sycl_linalg::detail::launch_fastchi2_solveh(q, dtrig, dphi, ddec, dscr, dpow, dcond, p);
    }

    static void ihs_accum(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, IhsAccT* dacc,
        bool first, const sycl_linalg::IhsSolveParams& p) {
        const float* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? plan.d_slot_deconv_f32() : nullptr;
        sycl_linalg::detail::launch_ihs_accumh(q, dz, ddec, dacc, first, p);
    }

    static void ihs_finalize(
        sycl::queue& q, const IhsAccT* dacc, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        const float* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? nullptr : plan.d_slot_deconv_f32();
        sycl_linalg::detail::launch_ihs_finalizeh(q, dacc, ddec, dpow, p);
    }

    static void ihs_single(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        sycl_linalg::detail::launch_ihs_singleh(q, dz, plan.d_slot_deconv_f32(), dpow, p);
    }

    static void copy_out(const DevOutT* src, HostOutT* dst, int N) {
        std::memcpy(dst, src, static_cast<size_t>(N) * sizeof(float));
    }
};

struct EngineFloatTraits {
    static constexpr PrecKind kPrec = TLS_SYCL_PREC_FLOAT;
    using PlanT      = sycl_nufft::Nufft1Planf;
    using XCoordT    = sycl_fft::df_t;
    using ScalarT    = float;
    using ComplexT   = sycl_fft::Complexf;
    using SolveRealT = float;
    using DevOutT    = float;
    using HostOutT   = float;
    using IhsAccT    = float;

    static auto& plan_cache(StreamSlot& s) { return s.plans_f; }

    static void build_subgrid(
        sycl::queue& q, const XCoordT* dx, const ScalarT* dw, const ScalarT* dyw,
        ComplexT* dsrc, int M, int b_start, int b_cnt, double qf0, double qbdf, size_t wg) {
        detail::launch_build_subgrid_sourcesf(q, dx, dw, dyw, dsrc, M, b_start, b_cnt, qf0, qbdf, wg);
    }

    static void solve_fastchi2(
        sycl::queue& q, const ComplexT* dtrig, const ScalarT* dphi, const ScalarT* ddec,
        SolveRealT* dscr, DevOutT* dpow, DevOutT* dcond, const sycl_linalg::FastChi2SolveParams& p) {
        sycl_linalg::detail::launch_fastchi2_solvef(q, dtrig, dphi, ddec, dscr, dpow, dcond, p);
    }

    static void ihs_accum(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, IhsAccT* dacc,
        bool first, const sycl_linalg::IhsSolveParams& p) {
        const float* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? plan.d_slot_deconv() : nullptr;
        sycl_linalg::detail::launch_ihs_accumf(q, dz, ddec, dacc, first, p);
    }

    static void ihs_finalize(
        sycl::queue& q, const IhsAccT* dacc, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        const float* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? nullptr : plan.d_slot_deconv();
        sycl_linalg::detail::launch_ihs_finalizef(q, dacc, ddec, dpow, p);
    }

    static void ihs_single(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        sycl_linalg::detail::launch_ihs_singlef(q, dz, plan.d_slot_deconv(), dpow, p);
    }

    static void copy_out(const DevOutT* src, HostOutT* dst, int N) {
        std::memcpy(dst, src, static_cast<size_t>(N) * sizeof(float));
    }
};

struct EngineDfTraits {
    static constexpr PrecKind kPrec = TLS_SYCL_PREC_DF;
    using PlanT      = sycl_nufft::Nufft1Plandf;
    using XCoordT    = sycl_fft::df_t;
    using ScalarT    = sycl_fft::df_t;
    using ComplexT   = sycl_fft::Complexdf;
    using SolveRealT = sycl_fft::df_t;
    using DevOutT    = sycl_fft::df_t;
    using HostOutT   = double;
    using IhsAccT    = sycl_fft::df_t;

    static auto& plan_cache(StreamSlot& s) { return s.plans_df; }

    static void build_subgrid(
        sycl::queue& q, const XCoordT* dx, const ScalarT* dw, const ScalarT* dyw,
        ComplexT* dsrc, int M, int b_start, int b_cnt, double qf0, double qbdf, size_t wg) {
        detail::launch_build_subgrid_sourcesdf(q, dx, dw, dyw, dsrc, M, b_start, b_cnt, qf0, qbdf, wg);
    }

    static void solve_fastchi2(
        sycl::queue& q, const ComplexT* dtrig, const ScalarT* dphi, const ScalarT* ddec,
        SolveRealT* dscr, DevOutT* dpow, DevOutT* dcond, const sycl_linalg::FastChi2SolveParams& p) {
        sycl_linalg::detail::launch_fastchi2_solvedf(q, dtrig, dphi, ddec, dscr, dpow, dcond, p);
    }

    static void ihs_accum(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, IhsAccT* dacc,
        bool first, const sycl_linalg::IhsSolveParams& p) {
        const sycl_fft::df_t* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? plan.d_slot_deconv() : nullptr;
        sycl_linalg::detail::launch_ihs_accumdf(q, dz, ddec, dacc, first, p);
    }

    static void ihs_finalize(
        sycl::queue& q, const IhsAccT* dacc, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        const sycl_fft::df_t* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? nullptr : plan.d_slot_deconv();
        sycl_linalg::detail::launch_ihs_finalizedf(q, dacc, ddec, dpow, p);
    }

    static void ihs_single(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        sycl_linalg::detail::launch_ihs_singledf(q, dz, plan.d_slot_deconv(), dpow, p);
    }

    static void copy_out(const DevOutT* src, HostOutT* dst, int N) {
        for (int i = 0; i < N; ++i) {
            dst[i] = static_cast<double>(src[i].hi) + static_cast<double>(src[i].lo);
        }
    }
};

struct EngineDoubleTraits {
    static constexpr PrecKind kPrec = TLS_SYCL_PREC_DOUBLE;
    using PlanT      = sycl_nufft::Nufft1Plan;
    using XCoordT    = double;
    using ScalarT    = double;
    using ComplexT   = sycl_fft::Complex;
    using SolveRealT = double;
    using DevOutT    = double;
    using HostOutT   = double;
    using IhsAccT    = double;

    static auto& plan_cache(StreamSlot& s) { return s.plans_d; }

    static void build_subgrid(
        sycl::queue& q, const XCoordT* dx, const ScalarT* dw, const ScalarT* dyw,
        ComplexT* dsrc, int M, int b_start, int b_cnt, double qf0, double qbdf, size_t wg) {
        detail::launch_build_subgrid_sources(q, dx, dw, dyw, dsrc, M, b_start, b_cnt, qf0, qbdf, wg);
    }

    static void solve_fastchi2(
        sycl::queue& q, const ComplexT* dtrig, const ScalarT* dphi, const ScalarT* ddec,
        SolveRealT* dscr, DevOutT* dpow, DevOutT* dcond, const sycl_linalg::FastChi2SolveParams& p) {
        sycl_linalg::detail::launch_fastchi2_solve(q, dtrig, dphi, ddec, dscr, dpow, dcond, p);
    }

    static void ihs_accum(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, IhsAccT* dacc,
        bool first, const sycl_linalg::IhsSolveParams& p) {
        const double* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? plan.d_slot_deconv() : nullptr;
        sycl_linalg::detail::launch_ihs_accum(q, dz, ddec, dacc, first, p);
    }

    static void ihs_finalize(
        sycl::queue& q, const IhsAccT* dacc, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        const double* ddec = (p.norm == sycl_linalg::NormKind::Nll && p.degree > 1) ? nullptr : plan.d_slot_deconv();
        sycl_linalg::detail::launch_ihs_finalize(q, dacc, ddec, dpow, p);
    }

    static void ihs_single(
        sycl::queue& q, const ComplexT* dz, PlanT& plan, DevOutT* dpow,
        const sycl_linalg::IhsSolveParams& p) {
        sycl_linalg::detail::launch_ihs_single(q, dz, plan.d_slot_deconv(), dpow, p);
    }

    static void copy_out(const DevOutT* src, HostOutT* dst, int N) {
        std::memcpy(dst, src, static_cast<size_t>(N) * sizeof(double));
    }
};

// ============================================================================
// Core FastChi2 Execution Template
// ============================================================================
template <typename Traits>
int execute_fastchi2_common(
    int M, double f0, double df, int Nf, int degree,
    sycl_nufft::PswfMode mode, sycl_linalg::NormKind norm, int device_id,
    typename Traits::HostOutT* power_out, typename Traits::HostOutT* cond_out,
    const std::function<int(int, FastChi2HostData<typename Traits::XCoordT, typename Traits::ScalarT>&)>& preproc_fn) {
    using XCoordT    = typename Traits::XCoordT;
    using ScalarT    = typename Traits::ScalarT;
    using ComplexT   = typename Traits::ComplexT;
    using SolveRealT = typename Traits::SolveRealT;
    using DevOutT    = typename Traits::DevOutT;
    using PlanT      = typename Traits::PlanT;

    DeviceRuntime& rt = DeviceManager::instance().get(device_id);
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_HALF) {
        if (!rt.desc.has_fp16) {
            set_last_error("Selected SYCL device does not support native FP16; switch to FP32 (tlsf) instead");
            return TLS_SYCL_ERR_DEVICE;
        }
    }
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_DOUBLE) {
        if (!rt.dev_cfg.has_fp64) {
            set_last_error("Selected SYCL device does not support native FP64; use Dekker double-float (tlsdf) instead");
            return TLS_SYCL_ERR_DEVICE;
        }
    }

    // Lazily measure native FP64/FP32 ratio once per device when FP64 is first used
    double fp64_ratio = 0.0;
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_DOUBLE) {
        if (!rt.fp64_probed.load(std::memory_order_acquire)) {
            SlotGuard probe_guard(rt);
            fp64_ratio = rt.query_fp64_ratio(*probe_guard.slot);
        } else {
            fp64_ratio = rt.cached_fp64_ratio();
        }
    }

    // 1. Choose hardware-generalized block size & super-chunking within 75% VRAM / streams budget
    const bool has_cond = (cond_out != nullptr);
    const GpuPlanChoice choice = choose_gpu_periodogram_plan(
        Nf, M, degree, /*is_ihs=*/false, has_cond, mode, Traits::kPrec,
        rt.dev_cfg, rt.num_slots, fp64_ratio);
    if (!choice.fits_budget) {
        char buf[256];
        std::snprintf(
            buf, sizeof(buf),
            "Lightcurve exceeds per-stream device memory budget (required %.2f MiB > budget %.2f MiB; "
            "75%% of device memory / %d streams)",
            static_cast<double>(choice.required_min_bytes) / (1024.0 * 1024.0),
            static_cast<double>(choice.slot_budget_bytes) / (1024.0 * 1024.0),
            rt.num_slots);
        set_last_error(buf);
        return TLS_SYCL_ERR_ALLOC;
    }

    // 2. Lock-free CPU preprocessing on calling thread (reusable thread_local buffers)
    thread_local FastChi2HostData<XCoordT, ScalarT> host_data;
    const int pre_status = preproc_fn(choice.nfft, host_data);
    if (pre_status != TLS_SYCL_OK) return pre_status;

    // 3. Acquire a reusable GPU stream slot
    SlotGuard guard(rt);
    StreamSlot& slot = *guard.slot;
    sycl::queue& q = slot.q;

    // 4. Compute exact buffer sizes & enforce per-slot 75% / streams device memory cap
    constexpr bool kIsHalf = (Traits::kPrec == TLS_SYCL_PREC_HALF);
    auto align64 = [](size_t n) -> size_t { return (n + 63u) & ~size_t{63u}; };
    const size_t x_bytes   = static_cast<size_t>(M) * sizeof(XCoordT);
    const size_t w_bytes   = static_cast<size_t>(M) * sizeof(ScalarT);
    const size_t off_w     = align64(x_bytes);
    const size_t off_yw    = off_w + align64(w_bytes);
    const size_t in_bytes  = off_yw + w_bytes;
    const size_t src_bytes = size_t{2} * static_cast<size_t>(choice.B_chunk) * static_cast<size_t>(M) * sizeof(ComplexT);
    const size_t chunk_bins_max = static_cast<size_t>(choice.B_chunk) * static_cast<size_t>(choice.nout);
    const size_t trig_bytes = static_cast<size_t>(3 * degree) * chunk_bins_max * sizeof(ComplexT);
    const size_t out_bytes = static_cast<size_t>(Nf) * sizeof(DevOutT);
    const size_t cond_bytes = has_cond ? out_bytes : size_t{0};
    const size_t scr_bytes = (degree > 12)
        ? (size_t{2} * static_cast<size_t>(2 * degree + 1) *
           (kIsHalf ? (chunk_bins_max >> 1) : chunk_bins_max) * sizeof(SolveRealT))
        : size_t{0};
    const size_t max_alloc = rt.dev_cfg.safe_max_alloc_bytes();

    slot.enforce_budget(
        rt.slot_mem_budget, in_bytes, src_bytes, trig_bytes, 0, scr_bytes, out_bytes, cond_bytes);

    // 5. Fetch cached M-resizable Nufft1Plan (cold creation serialized per device)
    PlanT& plan = slot.get_or_create_plan<PlanT>(
        Traits::plan_cache(slot), rt.plan_create_mtx, rt.slot_mem_budget,
        M, choice.N_block, choice.w, choice.plan_max_bs, mode, /*geo_fp16=*/false);

    // FP16 FastChi2 uses fused FFT-epilogue deconvolution with row_scale folded in FP32;
    // FP32 / DF / FP64 use SlotBitReversedRaw deferred deconvolution!
    const sycl_nufft::NufftOutOrder nufft_order =
        kIsHalf ? sycl_nufft::NufftOutOrder::SlotBitReversed : sycl_nufft::NufftOutOrder::SlotBitReversedRaw;
    if constexpr (kIsHalf) {
        plan.set_output_scale(host_data.row_scale);
    }

    const ScalarT* d_phi_hat = kIsHalf ? nullptr : plan.d_slot_phi_hat();
    const ScalarT* d_deconv  = kIsHalf ? nullptr : plan.d_slot_deconv();

    // 6. Ensure pinned host staging + reusable device workspace capacities
    //    Pack (x, w, yw) into one 64B-aligned contiguous slab -> single async H2D memcpy!
    uint8_t* h_in = slot.h_x.ensure(q, in_bytes);
    std::memcpy(h_in,          host_data.x.data(),  x_bytes);
    std::memcpy(h_in + off_w,  host_data.w.data(),  w_bytes);
    std::memcpy(h_in + off_yw, host_data.yw.data(), w_bytes);

    const size_t cap_limit = std::min(rt.slot_mem_budget, max_alloc);
    uint8_t* d_in = slot.d_x.ensure(q, in_bytes, cap_limit);
    auto* d_x     = reinterpret_cast<XCoordT*>(d_in);
    auto* d_w     = reinterpret_cast<ScalarT*>(d_in + off_w);
    auto* d_yw    = reinterpret_cast<ScalarT*>(d_in + off_yw);
    auto* d_src   = reinterpret_cast<ComplexT*>(slot.d_src.ensure(q, src_bytes, cap_limit));
    auto* d_trig  = reinterpret_cast<ComplexT*>(slot.d_trig.ensure(q, trig_bytes, cap_limit));
    auto* d_power = reinterpret_cast<DevOutT*>(slot.d_power.ensure(q, out_bytes, cap_limit));
    DevOutT* d_cond = has_cond ? reinterpret_cast<DevOutT*>(slot.d_cond.ensure(q, cond_bytes, cap_limit)) : nullptr;

    SolveRealT* d_a_scratch = (scr_bytes > 0)
        ? reinterpret_cast<SolveRealT*>(slot.d_a_scratch.ensure(q, scr_bytes, cap_limit))
        : nullptr;

    // 7. Single async H2D memcpy of packed O(M) vectors
    q.memcpy(d_in, h_in, in_bytes);

    const XCoordT df_coord = [&]() {
        if constexpr (std::is_same_v<XCoordT, sycl_fft::df_t>) {
            return sycl_fft::df_from_double(df);
        } else {
            return df;
        }
    }();
    // Padded bounds-checked kernels must NOT restrict wg_size to divisors of M!
    const size_t subgrid_wg = rt.dev_cfg.default_padded_wg_size(256);
    const size_t solve_wg   = rt.dev_cfg.default_padded_wg_size(128);
    const double block_df   = static_cast<double>(choice.nout) * df;

    // 8. Execute super-chunks
    for (int b_start = 0; b_start < choice.B_total; b_start += choice.B_chunk) {
        const int b_cnt = std::min(choice.B_chunk, choice.B_total - b_start);
        const size_t chunk_bins = static_cast<size_t>(b_cnt) * static_cast<size_t>(choice.nout);

        for (int harm = 1; harm <= 2 * degree; ++harm) {
            plan.set_points(d_x, M, df_coord, harm);
            const bool need_yw = (harm <= degree);
            const double qf0  = static_cast<double>(harm) * f0;
            const double qbdf = static_cast<double>(harm) * block_df;
            Traits::build_subgrid(
                q, d_x, d_w, need_yw ? d_yw : nullptr, d_src,
                M, b_start, b_cnt, qf0, qbdf, subgrid_wg);

            const size_t row_idx = need_yw
                ? static_cast<size_t>(2 * (harm - 1))
                : static_cast<size_t>(degree + harm - 1);
            const size_t bs = static_cast<size_t>(need_yw ? (2 * b_cnt) : b_cnt);
            plan.execute(d_src, d_trig + row_idx * chunk_bins, bs, nufft_order);
        }

        sycl_linalg::FastChi2SolveParams sp{};
        sp.degree       = degree;
        sp.M            = M;
        sp.Nf           = Nf;
        sp.nfft         = choice.nfft;
        sp.log_nfft     = plan.log_nfft();
        sp.nout         = choice.nout;
        sp.output_shift = plan.output_shift();
        sp.b_start      = b_start;
        sp.b_count      = b_cnt;
        sp.mode         = mode;
        sp.norm         = norm;
        sp.ws_fft       = host_data.ws_fft;
        sp.yws_fft      = host_data.yws_fft;
        sp.chi2_ref     = host_data.chi2_ref;
        sp.b_param      = host_data.b_param;
        sp.cyb          = host_data.cyb;
        sp.wg_size      = solve_wg;

        Traits::solve_fastchi2(q, d_trig, d_phi_hat, d_deconv, d_a_scratch, d_power, d_cond, sp);
    }

    // 9. Async D2H into pinned host staging, wait on stream, copy out to caller
    auto* h_power = reinterpret_cast<DevOutT*>(slot.h_power.ensure(q, out_bytes));
    q.memcpy(h_power, d_power, out_bytes);
    DevOutT* h_cond = nullptr;
    if (cond_out) {
        h_cond = reinterpret_cast<DevOutT*>(slot.h_cond.ensure(q, out_bytes));
        q.memcpy(h_cond, d_cond, out_bytes);
    }
    q.wait_and_throw();

    Traits::copy_out(h_power, power_out, Nf);
    if (cond_out) {
        Traits::copy_out(h_cond, cond_out, Nf);
    }
    return TLS_SYCL_OK;
}

// ============================================================================
// Core IHS Execution Template
// ============================================================================
template <typename Traits>
int execute_ihs_common(
    int M, double f0, double df, int Nf, int degree,
    sycl_nufft::PswfMode mode, sycl_linalg::NormKind norm, int device_id,
    typename Traits::HostOutT* power_out,
    const std::function<int(int, IhsHostData<typename Traits::XCoordT, typename Traits::ScalarT>&)>& preproc_fn) {
    using XCoordT  = typename Traits::XCoordT;
    using ScalarT  = typename Traits::ScalarT;
    using ComplexT = typename Traits::ComplexT;
    using DevOutT  = typename Traits::DevOutT;
    using IhsAccT  = typename Traits::IhsAccT;
    using PlanT    = typename Traits::PlanT;

    DeviceRuntime& rt = DeviceManager::instance().get(device_id);
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_HALF) {
        if (!rt.desc.has_fp16) {
            set_last_error("Selected SYCL device does not support native FP16; switch to FP32 (ihsf) instead");
            return TLS_SYCL_ERR_DEVICE;
        }
    }
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_DOUBLE) {
        if (!rt.dev_cfg.has_fp64) {
            set_last_error("Selected SYCL device does not support native FP64; use Dekker double-float (ihsdf) instead");
            return TLS_SYCL_ERR_DEVICE;
        }
    }

    double fp64_ratio = 0.0;
    if constexpr (Traits::kPrec == TLS_SYCL_PREC_DOUBLE) {
        if (!rt.fp64_probed.load(std::memory_order_acquire)) {
            SlotGuard probe_guard(rt);
            fp64_ratio = rt.query_fp64_ratio(*probe_guard.slot);
        } else {
            fp64_ratio = rt.cached_fp64_ratio();
        }
    }

    const GpuPlanChoice choice = choose_gpu_periodogram_plan(
        Nf, M, degree, /*is_ihs=*/true, /*has_cond=*/false, mode, Traits::kPrec,
        rt.dev_cfg, rt.num_slots, fp64_ratio);
    if (!choice.fits_budget) {
        char buf[256];
        std::snprintf(
            buf, sizeof(buf),
            "Lightcurve exceeds per-stream device memory budget (required %.2f MiB > budget %.2f MiB; "
            "75%% of device memory / %d streams)",
            static_cast<double>(choice.required_min_bytes) / (1024.0 * 1024.0),
            static_cast<double>(choice.slot_budget_bytes) / (1024.0 * 1024.0),
            rt.num_slots);
        set_last_error(buf);
        return TLS_SYCL_ERR_ALLOC;
    }

    thread_local IhsHostData<XCoordT, ScalarT> host_data;
    const int pre_status = preproc_fn(choice.nfft, host_data);
    if (pre_status != TLS_SYCL_OK) return pre_status;

    SlotGuard guard(rt);
    StreamSlot& slot = *guard.slot;
    sycl::queue& q = slot.q;

    auto align64 = [](size_t n) -> size_t { return (n + 63u) & ~size_t{63u}; };
    const size_t x_bytes   = static_cast<size_t>(M) * sizeof(XCoordT);
    const size_t w_bytes   = static_cast<size_t>(M) * sizeof(ScalarT);
    const size_t off_wy    = align64(x_bytes);
    const size_t in_bytes  = off_wy + w_bytes;
    const size_t src_bytes = static_cast<size_t>(choice.B_chunk) * static_cast<size_t>(M) * sizeof(ComplexT);
    const size_t chunk_bins_max = static_cast<size_t>(choice.B_chunk) * static_cast<size_t>(choice.nout);
    const size_t z_bytes   = chunk_bins_max * sizeof(ComplexT);
    // Fused single-harmonic IHS (degree == 1) skips d_ihs_acc allocation & round-trip entirely!
    const size_t acc_bytes = (degree > 1) ? (chunk_bins_max * sizeof(IhsAccT)) : size_t{0};
    const size_t out_bytes = static_cast<size_t>(Nf) * sizeof(DevOutT);
    const size_t max_alloc = rt.dev_cfg.safe_max_alloc_bytes();

    slot.enforce_budget(rt.slot_mem_budget, in_bytes, src_bytes, z_bytes, acc_bytes, 0, out_bytes, 0);

    // FP16 IHS uses geometric-mean-1 scaling + SlotBitReversedRaw deferred deconvolution!
    constexpr bool kGeoFp16 = (Traits::kPrec == TLS_SYCL_PREC_HALF);
    PlanT& plan = slot.get_or_create_plan<PlanT>(
        Traits::plan_cache(slot), rt.plan_create_mtx, rt.slot_mem_budget,
        M, choice.N_block, choice.w, choice.plan_max_bs, mode, kGeoFp16);

    uint8_t* h_in = slot.h_x.ensure(q, in_bytes);
    std::memcpy(h_in,          host_data.x.data(),  x_bytes);
    std::memcpy(h_in + off_wy, host_data.wy.data(), w_bytes);

    const size_t cap_limit = std::min(rt.slot_mem_budget, max_alloc);
    uint8_t* d_in = slot.d_x.ensure(q, in_bytes, cap_limit);
    auto* d_x     = reinterpret_cast<XCoordT*>(d_in);
    auto* d_wy    = reinterpret_cast<ScalarT*>(d_in + off_wy);
    auto* d_src   = reinterpret_cast<ComplexT*>(slot.d_src.ensure(q, src_bytes, cap_limit));
    auto* d_z     = reinterpret_cast<ComplexT*>(slot.d_trig.ensure(q, z_bytes, cap_limit));
    IhsAccT* d_acc = (acc_bytes > 0)
        ? reinterpret_cast<IhsAccT*>(slot.d_ihs_acc.ensure(q, acc_bytes, cap_limit))
        : nullptr;
    auto* d_power = reinterpret_cast<DevOutT*>(slot.d_power.ensure(q, out_bytes, cap_limit));

    q.memcpy(d_in, h_in, in_bytes);

    const XCoordT df_coord = [&]() {
        if constexpr (std::is_same_v<XCoordT, sycl_fft::df_t>) {
            return sycl_fft::df_from_double(df);
        } else {
            return df;
        }
    }();
    const size_t subgrid_wg = rt.dev_cfg.default_padded_wg_size(256);
    const size_t solve_wg   = rt.dev_cfg.default_padded_wg_size(128);
    const double block_df   = static_cast<double>(choice.nout) * df;

    if (degree == 1) {
        // Point geometry for harmonic 1 is invariant across super-chunks!
        plan.set_points(d_x, M, df_coord, 1);
    }

    for (int b_start = 0; b_start < choice.B_total; b_start += choice.B_chunk) {
        const int b_cnt = std::min(choice.B_chunk, choice.B_total - b_start);

        sycl_linalg::IhsSolveParams sp{};
        sp.degree         = degree;
        sp.Nf             = Nf;
        sp.nfft           = choice.nfft;
        sp.log_nfft       = plan.log_nfft();
        sp.nout           = choice.nout;
        sp.output_shift   = plan.output_shift();
        sp.b_start        = b_start;
        sp.b_count        = b_cnt;
        sp.mode           = mode;
        sp.norm           = norm;
        sp.standard_scale = host_data.standard_scale;
        sp.nll_scale      = host_data.nll_scale;
        sp.n_eff          = host_data.norm_neff;
        sp.wg_size        = solve_wg;

        if (degree == 1) {
            Traits::build_subgrid(
                q, d_x, d_wy, nullptr, d_src, M, b_start, b_cnt, f0, block_df, subgrid_wg);
            plan.execute(d_src, d_z, static_cast<size_t>(b_cnt), sycl_nufft::NufftOutOrder::SlotBitReversedRaw);
            Traits::ihs_single(q, d_z, plan, d_power, sp);
        } else {
            for (int harm = 1; harm <= degree; ++harm) {
                plan.set_points(d_x, M, df_coord, harm);
                const double qf0  = static_cast<double>(harm) * f0;
                const double qbdf = static_cast<double>(harm) * block_df;
                Traits::build_subgrid(
                    q, d_x, d_wy, nullptr, d_src, M, b_start, b_cnt, qf0, qbdf, subgrid_wg);
                plan.execute(d_src, d_z, static_cast<size_t>(b_cnt), sycl_nufft::NufftOutOrder::SlotBitReversedRaw);
                Traits::ihs_accum(q, d_z, plan, d_acc, harm == 1, sp);
            }
            Traits::ihs_finalize(q, d_acc, plan, d_power, sp);
        }
    }

    auto* h_power = reinterpret_cast<DevOutT*>(slot.h_power.ensure(q, out_bytes));
    q.memcpy(h_power, d_power, out_bytes);
    q.wait_and_throw();

    Traits::copy_out(h_power, power_out, Nf);
    return TLS_SYCL_OK;
}

template <typename Fn>
int guarded_call(Fn&& fn) noexcept {
    clear_last_error();
    try {
        return fn();
    } catch (const std::out_of_range& e) {
        set_last_error(e.what());
        return TLS_SYCL_ERR_DEVICE;
    } catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return TLS_SYCL_ERR_ARGUMENT;
    } catch (const std::bad_alloc& e) {
        set_last_error(e.what());
        return TLS_SYCL_ERR_ALLOC;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return TLS_SYCL_ERR_RUNTIME;
    } catch (...) {
        set_last_error("Unknown SYCL runtime exception");
        return TLS_SYCL_ERR_RUNTIME;
    }
}

int validate_common_args(const void* t, const void* y, const void* dy, const void* power,
                         int M, double f0, double df, int N, int degree) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 ||
        !std::isfinite(f0) || !std::isfinite(df) || f0 < 0.0 || df <= 0.0) {
        return TLS_SYCL_ERR_ARGUMENT;
    }
    if (M < 2 * degree + 2) return TLS_SYCL_ERR_DEGENERATE;
    return TLS_SYCL_OK;
}

} // namespace

int get_device_count() noexcept {
    try {
        return static_cast<int>(sycl_fft::enumerate_sycl_devices().size());
    } catch (...) {
        return 0;
    }
}

int get_device_info(int device_id, CDeviceInfo* out) noexcept {
    if (!out) return TLS_SYCL_ERR_ARGUMENT;
    try {
        const auto& devs = sycl_fft::enumerate_sycl_devices();
        if (device_id < 0 || static_cast<size_t>(device_id) >= devs.size()) {
            return TLS_SYCL_ERR_DEVICE;
        }
        const auto& d = devs[static_cast<size_t>(device_id)];
        std::memset(out, 0, sizeof(*out));
        out->id = d.id;
        std::snprintf(out->name, sizeof(out->name), "%s", d.name.c_str());
        std::snprintf(out->backend, sizeof(out->backend), "%s", d.backend.c_str());
        std::snprintf(out->device_type, sizeof(out->device_type), "%s", d.device_type.c_str());
        out->max_compute_units = static_cast<uint64_t>(d.max_compute_units);
        out->global_mem_size = static_cast<uint64_t>(d.global_mem_size);
        out->is_gpu = d.is_gpu ? 1 : 0;
        out->has_fp64 = d.has_fp64 ? 1 : 0;
        out->has_fp16 = d.has_fp16 ? 1 : 0;
        return TLS_SYCL_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return TLS_SYCL_ERR_RUNTIME;
    } catch (...) {
        return TLS_SYCL_ERR_RUNTIME;
    }
}

const char* get_last_error() noexcept {
    return tls_last_err.c_str();
}

int run_fastchi2_half(
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power, float* cond) noexcept {
    return guarded_call([&]() {
        const void* y_any  = y_d ? static_cast<const void*>(y_d) : static_cast<const void*>(y_f);
        const void* dy_any = dy_d ? static_cast<const void*>(dy_d) : static_cast<const void*>(dy_f);
        int st = validate_common_args(t, y_any, dy_any, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_fastchi2_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_fastchi2_common<EngineHalfTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power, cond,
            [&](int nfft, auto& out) {
                return preprocess_fastchi2_half(t, y_f, y_d, dy_f, dy_d, M, degree, nfft, out);
            });
    });
}

int run_fastchi2_float(
    const double* t, const float* y, const float* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power, float* cond) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_fastchi2_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_fastchi2_common<EngineFloatTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power, cond,
            [&](int nfft, auto& out) {
                return preprocess_fastchi2_generic<float, sycl_fft::df_t, float>(t, y, dy, M, degree, nfft, out);
            });
    });
}

int run_fastchi2_df(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power, double* cond) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_fastchi2_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_fastchi2_common<EngineDfTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power, cond,
            [&](int nfft, auto& out) {
                return preprocess_fastchi2_generic<double, sycl_fft::df_t, sycl_fft::df_t>(t, y, dy, M, degree, nfft, out);
            });
    });
}

int run_fastchi2_double(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power, double* cond) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_fastchi2_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_fastchi2_common<EngineDoubleTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power, cond,
            [&](int nfft, auto& out) {
                return preprocess_fastchi2_generic<double, double, double>(t, y, dy, M, degree, nfft, out);
            });
    });
}

int run_ihs_half(
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power) noexcept {
    return guarded_call([&]() {
        const void* y_any  = y_d ? static_cast<const void*>(y_d) : static_cast<const void*>(y_f);
        const void* dy_any = dy_d ? static_cast<const void*>(dy_d) : static_cast<const void*>(dy_f);
        int st = validate_common_args(t, y_any, dy_any, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_ihs_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_ihs_common<EngineHalfTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power,
            [&](int nfft, auto& out) {
                return preprocess_ihs_impl<sycl_fft::df_t, sycl_fft::f16_scalar_t, true>(
                    t, y_f, y_d, dy_f, dy_d, M, nfft, out);
            });
    });
}

int run_ihs_float(
    const double* t, const float* y, const float* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_ihs_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_ihs_common<EngineFloatTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power,
            [&](int nfft, auto& out) {
                return preprocess_ihs_impl<sycl_fft::df_t, float, false>(
                    t, y, nullptr, dy, nullptr, M, nfft, out);
            });
    });
}

int run_ihs_df(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_ihs_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_ihs_common<EngineDfTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power,
            [&](int nfft, auto& out) {
                return preprocess_ihs_impl<sycl_fft::df_t, sycl_fft::df_t, false>(
                    t, nullptr, y, nullptr, dy, M, nfft, out);
            });
    });
}

int run_ihs_double(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power) noexcept {
    return guarded_call([&]() {
        int st = validate_common_args(t, y, dy, power, M, f0, df, N, degree);
        if (st != TLS_SYCL_OK) return st;
        sycl_nufft::PswfMode mode{};
        sycl_linalg::NormKind norm{};
        if ((st = parse_backend(backend, mode)) != TLS_SYCL_OK) return st;
        if ((st = parse_ihs_norm(normalization, norm)) != TLS_SYCL_OK) return st;
        return execute_ihs_common<EngineDoubleTraits>(
            M, f0, df, N, degree, mode, norm, device_id, power,
            [&](int nfft, auto& out) {
                return preprocess_ihs_impl<double, double, false>(
                    t, nullptr, y, nullptr, dy, M, nfft, out);
            });
    });
}

} // namespace sycl_periodogram
