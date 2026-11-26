#pragma once

// Quadruple-Precision SYCL DIF (Sande-Tukey) Radix-2 FFT Header (Declarations Only):
//   FP16 (2x):   h suffix   (Complexh,  DifFftPlanh,  ...)
//   FP32:        f suffix   (Complexf,  DifFftPlanf,  ...)
//   FP64:        no suffix  (Complex,   DifFftPlan,   ...)
//   DoubleFloat: df suffix  (Complexdf, DifFftPlandf, ...)
//
// All FFT kernels and DifFftPlan implementations live in fft/dif.cpp (compiled per
// precision and small/large split into isolated HCF modules).
// All DifFftPlan constructors require an in-order sycl::queue (enforced at runtime).
//
// Sign convention compile-time controls (default: +1 only):
//   SYCL_FFT_ENABLE_POS_SIGN (0 or 1, default 1)
//   SYCL_FFT_ENABLE_NEG_SIGN (0 or 1, default 0)
//   SYCL_FFT_DEFAULT_SIGN    (+1 or -1; defaults to +1 if POS enabled, else -1)
#if !defined(SYCL_FFT_ENABLE_POS_SIGN) && !defined(SYCL_FFT_ENABLE_NEG_SIGN)
#define SYCL_FFT_ENABLE_POS_SIGN 1
#define SYCL_FFT_ENABLE_NEG_SIGN 0
#else
#ifndef SYCL_FFT_ENABLE_POS_SIGN
#define SYCL_FFT_ENABLE_POS_SIGN 0
#endif
#ifndef SYCL_FFT_ENABLE_NEG_SIGN
#define SYCL_FFT_ENABLE_NEG_SIGN 0
#endif
#endif

#if !SYCL_FFT_ENABLE_POS_SIGN && !SYCL_FFT_ENABLE_NEG_SIGN
#error "At least one of SYCL_FFT_ENABLE_POS_SIGN or SYCL_FFT_ENABLE_NEG_SIGN must be 1"
#endif

#ifndef SYCL_FFT_DEFAULT_SIGN
#if SYCL_FFT_ENABLE_POS_SIGN
#define SYCL_FFT_DEFAULT_SIGN (+1)
#else
#define SYCL_FFT_DEFAULT_SIGN (-1)
#endif
#endif

#include "common/math.hpp"
#include "common/layout.hpp"
#include "common/device.hpp"
#include "bitrev/bitrev.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sycl_fft {

#if SYCL_FFT_ENABLE_NEG_SIGN
#define SYCL_FFT_EXTERN_DECL_NEG(PlanName) extern template class PlanName<-1>;
#else
#define SYCL_FFT_EXTERN_DECL_NEG(PlanName)
#endif

#if SYCL_FFT_ENABLE_POS_SIGN
#define SYCL_FFT_EXTERN_DECL_POS(PlanName) extern template class PlanName<+1>;
#else
#define SYCL_FFT_EXTERN_DECL_POS(PlanName)
#endif

// Optional fused compact-slot filtering + deconvolution epilogue for SinglePass / TwoPass DIF FFTs.
//   compact_mode == 0: full N-element DIF slot output (standard FFT)
//   compact_mode == 1: PSWF21 kept slots (s % 4 in {0, 3}, compact stride N/2)
//   compact_mode == 2: PSWF43 kept slots (s % 8 in {0, 2, 3, 4, 5, 7}, compact stride 3N/4)
struct FftCompactParams {
    int         compact_mode  = 0;
    int         log_n1        = 0;       // 0 for SinglePass; log_n1 for TwoPass Pass-1 sub-FFTs
    const void* d_slot_deconv = nullptr; // ScalarT[full_nout] compact-slot deconv LUT (or non-null dummy)
    bool        apply_deconv  = false;   // true iff d_slot_deconv should be loaded and multiplied
};

#define SYCL_FFT_DECLARE_DIF_PLAN(CandName, PlanName, ComplexType, SmemType)                  \
struct CandName {                                                                             \
    using ComplexT = ComplexType;                                                             \
    using LaunchFn = void (*)(sycl::queue&, const ComplexT*, ComplexT*, size_t, bool, bool,   \
                              const FftCompactParams&);                                       \
    using ProbeFn  = KernelProbeInfo (*)(sycl::queue&, const sycl::device&);                  \
                                                                                              \
    const char*   variant_id;                                                                 \
    int           logN;                                                                       \
    size_t        wg_size;                                                                    \
    size_t        smem_bytes;                                                                 \
    size_t        req_sg;                                                                     \
    int           pinned_sg;                                                                  \
    bool          uses_wide_2t;                                                               \
    bool          uses_stash_8192;                                                            \
    BankModelKind bank_kind;                                                                  \
    LaunchFn      launch;                                                                     \
    ProbeFn       probe_bundle;                                                               \
};                                                                                            \
                                                                                              \
template <int Sign = SYCL_FFT_DEFAULT_SIGN>                                                   \
class PlanName {                                                                              \
    static_assert(Sign == -1 || Sign == +1, "DifFftPlan Sign must be -1 or +1");              \
    static_assert((Sign == +1 && SYCL_FFT_ENABLE_POS_SIGN) ||                                 \
                      (Sign == -1 && SYCL_FFT_ENABLE_NEG_SIGN),                               \
                  "Requested DifFftPlan<Sign> is disabled at compile time (build with SIGN=pos|neg|both)"); \
public:                                                                                       \
    static constexpr bool kPosSign = (Sign > 0);                                              \
    using ComplexT  = ComplexType;                                                            \
    using SmemT     = SmemType;                                                               \
    using Candidate = CandName;                                                               \
                                                                                              \
    PlanName(sycl::queue& q, int logN, size_t max_bs, const FftLaunchConfig& cfg = {});       \
    PlanName(sycl::queue& q, int logN, size_t max_bs, size_t l2_chunk_bytes);                 \
    ~PlanName();                                                                              \
                                                                                              \
    PlanName(const PlanName&) = delete;                                                       \
    PlanName& operator=(const PlanName&) = delete;                                            \
                                                                                              \
    void execute_raw(const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override = 0); \
    /* Fused compact-slot + optional deconvolution epilogue (SinglePass & TwoPass). */        \
    /* Note: `in` (full length bs*N) may be overwritten as scratch when StashChunk1 runs. */  \
    void execute_raw_compact(ComplexT* in, ComplexT* out, size_t bs, int compact_mode,        \
                             const void* d_slot_deconv = nullptr, size_t chunk_bs_override = 0); \
    void execute(const ComplexT* in, ComplexT* out, size_t bs, size_t chunk_bs_override = 0); \
    void bit_reverse(const ComplexT* src, ComplexT* dst, size_t bs);                          \
    void bit_reverse(ComplexT* buf, size_t bs);                                               \
    const char* bitrev_variant_name(bool in_place = true);                                    \
    bool bitrev_cache_hit(bool in_place = true);                                              \
    BitRevPlan<ComplexT>& bitrev_plan();                                                      \
                                                                                              \
    size_t chunk_bs() const { return chunk_bs_; }                                             \
    size_t chunk_bytes() const { return chunk_bs_ * N_ * sizeof(ComplexT); }                  \
    size_t pass0_wg_size() const { return pass0_wg_size_; }                                   \
    size_t bitrev_wg_size() const { return bitrev_wg_size_; }                                 \
    int    two_pass_min_logN() const { return two_pass_min_logN_; }                           \
    bool   is_two_pass() const { return exec_mode_ == PlanExecMode::TwoPass; }                \
    bool   is_multistage() const { return exec_mode_ == PlanExecMode::MultiStageGlobal; }     \
    PlanExecMode exec_mode() const { return exec_mode_; }                                     \
    int    log_n1() const { return log_n1_; }                                                 \
    int    log_n2() const { return log_n2_; }                                                 \
    const char* variant_name() const { return variant_name_.c_str(); }                        \
    size_t effective_sg_size() const { return effective_sg_size_; }                           \
    double fp64_to_fp32_ratio() const { return fp64_to_fp32_ratio_; }                         \
    bool   is_compute_heavy() const { return is_compute_heavy_; }                             \
    bool   cache_hit() const { return cache_hit_; }                                           \
    const DeviceFftConfig& device_config() const { return dev_cfg_; }                         \
    const FftLaunchConfig& launch_config() const { return launch_cfg_; }                      \
                                                                                              \
private:                                                                                      \
    struct FilteredCandidate {                                                                \
        Candidate       cand;                                                                 \
        KernelProbeInfo kinfo;                                                                \
    };                                                                                        \
                                                                                              \
    static FftLaunchConfig make_chunk_config(size_t l2_chunk_bytes);                          \
    static std::string format_multistage_name(int logN);                                      \
    bool fits_candidate(const Candidate& c, bool apply_variant_override,                      \
                        KernelProbeInfo& out_kinfo) const;                                    \
    std::vector<FilteredCandidate> collect_candidates(int n, bool apply_variant_override) const; \
    static bool find_by_id(const std::vector<FilteredCandidate>& valid,                       \
                           const std::string& id, Candidate& out);                            \
    static Candidate pick_heuristic(const std::vector<FilteredCandidate>& valid);             \
    std::vector<FilteredCandidate> shortlist_single_pass(                                     \
        const std::vector<FilteredCandidate>& valid) const;                                   \
    size_t representative_tune_bs(int n) const;                                               \
    bool tune_single_pass(int n, const std::vector<FilteredCandidate>& valid,                 \
                          Candidate& out_winner, double& out_best_ms);                        \
    bool resolve_single_pass_for(int n, bool apply_variant_override, bool allow_cache,        \
                                 bool force_tune, Candidate& out_cand, bool& out_cache_hit,   \
                                 double* out_ms = nullptr);                                   \
    void setup_multistage(size_t max_bs);                                                     \
    void configure_plan(size_t max_bs);                                                       \
    BitRevPlan<ComplexT>& ensure_bitrev_plan();                                               \
                                                                                              \
    sycl::queue&    q_;                                                                       \
    int             logN_;                                                                    \
    size_t          N_;                                                                       \
    DeviceFftConfig dev_cfg_;                                                                 \
    FftLaunchConfig launch_cfg_;                                                              \
    std::string     cache_path_;                                                              \
                                                                                              \
    PlanExecMode    exec_mode_ = PlanExecMode::SinglePass;                                    \
    Candidate       single_pass_cand_{};                                                      \
    std::string     variant_name_;                                                            \
    int             two_pass_min_logN_ = 0;                                                   \
    int             log_n1_ = 0;                                                              \
    int             log_n2_ = 0;                                                              \
    size_t          pass0_wg_size_ = 256;                                                     \
    size_t          bitrev_wg_size_ = 256;                                                    \
    size_t          chunk_bs_ = 0;                                                            \
    size_t          scratch_elems_ = 0;                                                       \
    ComplexT*       scratch_ = nullptr;                                                       \
    UsmPtr<ComplexT> d_dummy_deconv_;                                                         \
                                                                                              \
    size_t          effective_sg_size_ = 1;                                                   \
    bool            sg_linear_ok_ = false;                                                    \
    bool            sg_tile_sync_ok_ = false;                                                 \
    double          fp64_to_fp32_ratio_ = 0.0;                                                \
    bool            is_compute_heavy_ = false;                                                \
    bool            cache_hit_ = false;                                                       \
    std::unique_ptr<BitRevPlan<ComplexT>> bitrev_plan_;                                       \
};                                                                                            \
                                                                                              \
SYCL_FFT_EXTERN_DECL_NEG(PlanName)                                                            \
SYCL_FFT_EXTERN_DECL_POS(PlanName)

SYCL_FFT_DECLARE_DIF_PLAN(SinglePassCandidateh,  DifFftPlanh,  Complexh,  uint32_t)
SYCL_FFT_DECLARE_DIF_PLAN(SinglePassCandidatef,  DifFftPlanf,  Complexf,  float)
SYCL_FFT_DECLARE_DIF_PLAN(SinglePassCandidate,   DifFftPlan,   Complex,   double)
SYCL_FFT_DECLARE_DIF_PLAN(SinglePassCandidatedf, DifFftPlandf, Complexdf, df_t)

#undef SYCL_FFT_DECLARE_DIF_PLAN
#undef SYCL_FFT_EXTERN_DECL_NEG
#undef SYCL_FFT_EXTERN_DECL_POS

} // namespace sycl_fft
