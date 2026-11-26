#pragma once

// Fused SYCL GPU Periodogram Solvers:
//   - Generalized Lomb-Scargle (degree == 1)
//   - In-place Hermitian Toeplitz Levinson-Durbin recursion (degree > 1)
//   - Incoherent Harmonic Sum (IHS) accumulation & Wilkie/Erlang finalization
//
// Every solver operates directly on compact DIF slot-order trig rows, applies
// deferred deconvolution (or pre-deconvolved FP16 fused-epilogue rows), computes
// normalization + singularity NaN gating, and writes directly to natural frequency
// order (eliminating any separate bit-reversal pass).

#include "nufft/nufft1.hpp"

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>

namespace sycl_linalg {

enum class NormKind : int {
    Standard    = 0,
    Anova       = 1,
    Nll         = 2,
    Regularized = 3,
    IhsDefault  = 4
};

// Host-computed Cybenko reflection-product distribution model (mirrors src/chi2per.c)
struct CybenkoModel {
    double var_S      = 0.0;
    double mean_S     = 0.0;
    double k_eff      = 0.0;
    double theta_eff  = 0.0;
    double lgamma_k   = 0.0;
    double sum_N_eff  = 0.0;
    double inv_2d     = 0.0;
};

struct CybenkoModelF {
    float k_eff      = 0.0f;
    float theta_eff  = 0.0f;
    float lgamma_k   = 0.0f;
    float sum_N_eff  = 0.0f;
    float inv_2d     = 0.0f;
};

inline CybenkoModelF to_cybenko_f(const CybenkoModel& m) {
    return CybenkoModelF{
        static_cast<float>(m.k_eff),
        static_cast<float>(m.theta_eff),
        static_cast<float>(m.lgamma_k),
        static_cast<float>(m.sum_N_eff),
        static_cast<float>(m.inv_2d)
    };
}

// FastChi2 solver parameters shared across GLS (d == 1) and Levinson (d > 1).
// Unified trig layout:
//   chunk_bins = b_count * nout
//   For q in 1..degree:
//     w_q  at row (2*(q-1))     * chunk_bins
//     yw_q at row (2*(q-1) + 1) * chunk_bins
//   For q in degree+1..2*degree:
//     w_q  at row (degree + q - 1) * chunk_bins
struct FastChi2SolveParams {
    int                   degree        = 1;
    int                   M             = 0;
    int                   Nf            = 0;
    int                   nfft          = 0;
    int                   log_nfft      = 0;
    int                   nout          = 0;
    int                   output_shift  = 0;
    int                   b_start       = 0;
    int                   b_count       = 0;
    sycl_nufft::PswfMode  mode          = sycl_nufft::PswfMode::PSWF43;
    NormKind              norm          = NormKind::Standard;
    double                ws_fft        = 0.0;
    double                yws_fft       = 0.0;
    double                chi2_ref      = 0.0;
    double                b_param       = 0.0;
    CybenkoModel          cyb{};
    size_t                wg_size       = 128;
};

struct IhsSolveParams {
    int                   degree         = 1;
    int                   Nf             = 0;
    int                   nfft           = 0;
    int                   log_nfft       = 0;
    int                   nout           = 0;
    int                   output_shift   = 0;
    int                   b_start        = 0;
    int                   b_count        = 0;
    sycl_nufft::PswfMode  mode           = sycl_nufft::PswfMode::PSWF43;
    NormKind              norm           = NormKind::IhsDefault;
    double                standard_scale = 0.0;
    double                nll_scale      = 0.0;
    double                n_eff          = 0.0;
    size_t                wg_size        = 128;
};

namespace detail {

// --- FastChi2 (GLS d=1 + Levinson d>1) per-precision launchers ---
// d_slot_phi_hat & d_slot_deconv are non-null for deferred deconvolution (FP32/DF/FP64)
// and null when NuFFT outputs are already deconvolved (FP16 fused FFT epilogue).
// d_a_scratch is only needed when degree > 12 (2 * (2*degree+1) * chunk_bins elements).
void launch_fastchi2_solveh(
    sycl::queue& q,
    const sycl_fft::Complexh* d_trig,
    const sycl_fft::f16_scalar_t* d_slot_phi_hat,
    const sycl_fft::f16_scalar_t* d_slot_deconv,
    sycl_fft::f16x2_t* d_a_scratch,
    float* d_power,
    float* d_cond,
    const FastChi2SolveParams& p);

void launch_fastchi2_solvef(
    sycl::queue& q,
    const sycl_fft::Complexf* d_trig,
    const float* d_slot_phi_hat,
    const float* d_slot_deconv,
    float* d_a_scratch,
    float* d_power,
    float* d_cond,
    const FastChi2SolveParams& p);

void launch_fastchi2_solve(
    sycl::queue& q,
    const sycl_fft::Complex* d_trig,
    const double* d_slot_phi_hat,
    const double* d_slot_deconv,
    double* d_a_scratch,
    double* d_power,
    double* d_cond,
    const FastChi2SolveParams& p);

void launch_fastchi2_solvedf(
    sycl::queue& q,
    const sycl_fft::Complexdf* d_trig,
    const sycl_fft::df_t* d_slot_phi_hat,
    const sycl_fft::df_t* d_slot_deconv,
    sycl_fft::df_t* d_a_scratch,
    sycl_fft::df_t* d_power,
    sycl_fft::df_t* d_cond,
    const FastChi2SolveParams& p);

// --- IHS per-harmonic accumulation & finalization launchers ---
// `first_harmonic == true` initializes d_acc instead of adding into it.
void launch_ihs_accumh(
    sycl::queue& q,
    const sycl_fft::Complexh* d_z,
    const float* d_slot_deconv_f32,
    float* d_acc,
    bool first_harmonic,
    const IhsSolveParams& p);

void launch_ihs_finalizeh(
    sycl::queue& q,
    const float* d_acc,
    const float* d_slot_deconv_f32,
    float* d_power,
    const IhsSolveParams& p);

void launch_ihs_accumf(
    sycl::queue& q,
    const sycl_fft::Complexf* d_z,
    const float* d_slot_deconv,
    float* d_acc,
    bool first_harmonic,
    const IhsSolveParams& p);

void launch_ihs_finalizef(
    sycl::queue& q,
    const float* d_acc,
    const float* d_slot_deconv,
    float* d_power,
    const IhsSolveParams& p);

void launch_ihs_accum(
    sycl::queue& q,
    const sycl_fft::Complex* d_z,
    const double* d_slot_deconv,
    double* d_acc,
    bool first_harmonic,
    const IhsSolveParams& p);

void launch_ihs_finalize(
    sycl::queue& q,
    const double* d_acc,
    const double* d_slot_deconv,
    double* d_power,
    const IhsSolveParams& p);

void launch_ihs_accumdf(
    sycl::queue& q,
    const sycl_fft::Complexdf* d_z,
    const sycl_fft::df_t* d_slot_deconv,
    sycl_fft::df_t* d_acc,
    bool first_harmonic,
    const IhsSolveParams& p);

void launch_ihs_finalizedf(
    sycl::queue& q,
    const sycl_fft::df_t* d_acc,
    const sycl_fft::df_t* d_slot_deconv,
    sycl_fft::df_t* d_power,
    const IhsSolveParams& p);

// --- Fused single-harmonic (degree == 1) IHS kernel (skips d_acc round-trip) ---
void launch_ihs_singleh(
    sycl::queue& q,
    const sycl_fft::Complexh* d_z,
    const float* d_slot_deconv_f32,
    float* d_power,
    const IhsSolveParams& p);

void launch_ihs_singlef(
    sycl::queue& q,
    const sycl_fft::Complexf* d_z,
    const float* d_slot_deconv,
    float* d_power,
    const IhsSolveParams& p);

void launch_ihs_single(
    sycl::queue& q,
    const sycl_fft::Complex* d_z,
    const double* d_slot_deconv,
    double* d_power,
    const IhsSolveParams& p);

void launch_ihs_singledf(
    sycl::queue& q,
    const sycl_fft::Complexdf* d_z,
    const sycl_fft::df_t* d_slot_deconv,
    sycl_fft::df_t* d_power,
    const IhsSolveParams& p);

} // namespace detail
} // namespace sycl_linalg
