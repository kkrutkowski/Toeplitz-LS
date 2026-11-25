#include "sycl_fft_dif.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Deterministic SplitMix64 PRNG producing full 53-bit IEEE-754 doubles in [-1, 1]
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_double() {
        // 53-bit uniform in [-1, 1]
        const double u = static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
        return 2.0 * u - 1.0;
    }
};

// 80-bit long double CPU reference implementation of Sande-Tukey (DIF) Radix-2 FFT
// (64-bit significand ~19 digits -> exact gold reference for FP32, FP64, and DF).
void cpu_dif_radix2_ref(
    std::vector<long double>& re, std::vector<long double>& im, int logN, int sign, bool do_bitrev) {
    const size_t N = size_t{1} << logN;
    constexpr long double kTwoPiLD = 6.28318530717958647692528676655900576839L;

    for (size_t step = N; step > 1; step >>= 1) {
        const size_t half = step >> 1;
        const long double ang_step = static_cast<long double>(sign) * kTwoPiLD / static_cast<long double>(step);
        for (size_t i = 0; i < N; i += step) {
            for (size_t j = 0; j < half; ++j) {
                const long double ar = 0.5L * re[i + j], ai = 0.5L * im[i + j];
                const long double br = 0.5L * re[i + j + half], bi = 0.5L * im[i + j + half];
                re[i + j] = ar + br;
                im[i + j] = ai + bi;
                const long double dr = ar - br;
                const long double di = ai - bi;
                const long double ang = ang_step * static_cast<long double>(j);
                const long double wr = std::cos(ang);
                const long double wi = std::sin(ang);
                re[i + j + half] = dr * wr - di * wi;
                im[i + j + half] = dr * wi + di * wr;
            }
        }
    }

    if (do_bitrev) {
        for (size_t i = 0; i < N; ++i) {
            const size_t j = sycl_fft::bit_reverse(static_cast<uint32_t>(i), logN);
            if (j > i) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
    }
}

// Precision traits for templated verification and benchmarking
struct HalfTraits {
    using ComplexT = sycl_fft::Complexh;
    template <int Sign> using PlanT = sycl_fft::DifFftPlanh<Sign>;
    static constexpr const char* kName = "FP16 (2xFP16)";
    static constexpr const char* kPrecTag = "h";
    static constexpr int kDefaultLogTotal = 26;   // 256 MiB / buffer
    static constexpr double kTolRel = 1.5e-2;

    static ComplexT from_doubles(double r, double i) {
        return sycl_fft::complexh_from_doubles(r, i);
    }
    static void to_long_doubles(const ComplexT& v, long double& r, long double& i) {
        sycl_fft::complexh_to_long_doubles(v, r, i);
    }
};

struct FloatTraits {
    using ComplexT = sycl_fft::Complexf;
    template <int Sign> using PlanT = sycl_fft::DifFftPlanf<Sign>;
    static constexpr const char* kName = "FP32";
    static constexpr const char* kPrecTag = "f";
    static constexpr int kDefaultLogTotal = 25;   // 256 MiB / buffer
    static constexpr double kTolRel = 2e-5;

    static ComplexT from_doubles(double r, double i) {
        return ComplexT{static_cast<float>(r), static_cast<float>(i)};
    }
    static void to_long_doubles(const ComplexT& v, long double& r, long double& i) {
        r = static_cast<long double>(v.x());
        i = static_cast<long double>(v.y());
    }
};

struct DoubleTraits {
    using ComplexT = sycl_fft::Complex;
    template <int Sign> using PlanT = sycl_fft::DifFftPlan<Sign>;
    static constexpr const char* kName = "FP64";
    static constexpr const char* kPrecTag = "d";
    static constexpr int kDefaultLogTotal = 24;   // 256 MiB / buffer
    static constexpr double kTolRel = 5e-14;

    static ComplexT from_doubles(double r, double i) {
        return ComplexT{r, i};
    }
    static void to_long_doubles(const ComplexT& v, long double& r, long double& i) {
        r = static_cast<long double>(v.x());
        i = static_cast<long double>(v.y());
    }
};

struct DfTraits {
    using ComplexT = sycl_fft::Complexdf;
    template <int Sign> using PlanT = sycl_fft::DifFftPlandf<Sign>;
    static constexpr const char* kName = "DF (2xFP32)";
    static constexpr const char* kPrecTag = "df";
    static constexpr int kDefaultLogTotal = 24;   // 256 MiB / buffer
    static constexpr double kTolRel = 5e-12;

    static ComplexT from_doubles(double r, double i) {
        return ComplexT{sycl_fft::df_from_double(r), sycl_fft::df_from_double(i)};
    }
    static void to_long_doubles(const ComplexT& v, long double& r, long double& i) {
        r = static_cast<long double>(v.re.hi) + static_cast<long double>(v.re.lo);
        i = static_cast<long double>(v.im.hi) + static_cast<long double>(v.im.lo);
    }
};
// Standalone verification of branchless polynomial cos2pi / sin2pi across all 4 precisions
// on negative, >1, quadrant, and octant boundary values (executed on the SYCL device).
bool verify_standalone_trig(sycl::queue& q) {
    constexpr long double kTwoPiLD = 6.28318530717958647692528676655900576839L;
    std::vector<double> h_x = {
        0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5,
        0.625, 0.75, 0.875, 1.0, 1.125, 2.345678901234567,
        -0.0625, -0.125, -0.25, -0.375, -0.5, -0.75, -1.0, -3.141592653589793,
        0.125 - 1e-7, 0.125 + 1e-7, 0.25 - 1e-7, 0.25 + 1e-7, 0.5 - 1e-7, 0.5 + 1e-7,
        -1e-12, 1.0 - 1e-12
    };
    SplitMix64 rng(777);
    for (int i = 0; i < 1000; ++i) {
        h_x.push_back(rng.next_double() * 8.0);
    }
    const size_t M = h_x.size();

    auto* d_x = sycl::malloc_device<double>(M, q);
    auto* d_ch = sycl::malloc_device<sycl::half>(M, q);
    auto* d_sh = sycl::malloc_device<sycl::half>(M, q);
    auto* d_cf = sycl::malloc_device<float>(M, q);
    auto* d_sf = sycl::malloc_device<float>(M, q);
    auto* d_cd = sycl::malloc_device<double>(M, q);
    auto* d_sd = sycl::malloc_device<double>(M, q);
    auto* d_cdf = sycl::malloc_device<sycl_fft::df_t>(M, q);
    auto* d_sdf = sycl::malloc_device<sycl_fft::df_t>(M, q);

    q.memcpy(d_x, h_x.data(), M * sizeof(double)).wait();
    q.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
        const size_t i = idx[0];
        const double xd = d_x[i];
        const float xf = static_cast<float>(xd);
        const sycl::half xh = static_cast<sycl::half>(xf);
        const sycl_fft::df_t xdf = sycl_fft::df_from_double(xd);
        d_ch[i] = sycl_fft::cos2pih(xh);
        d_sh[i] = sycl_fft::sin2pih(xh);
        d_cf[i] = sycl_fft::cos2pif(xf);
        d_sf[i] = sycl_fft::sin2pif(xf);
        d_cd[i] = sycl_fft::cos2pi(xd);
        d_sd[i] = sycl_fft::sin2pi(xd);
        d_cdf[i] = sycl_fft::cos2pidf(xdf);
        d_sdf[i] = sycl_fft::sin2pidf(xdf);
    }).wait();

    std::vector<sycl::half> h_ch(M), h_sh(M);
    std::vector<float> h_cf(M), h_sf(M);
    std::vector<double> h_cd(M), h_sd(M);
    std::vector<sycl_fft::df_t> h_cdf(M), h_sdf(M);
    q.memcpy(h_ch.data(), d_ch, M * sizeof(sycl::half));
    q.memcpy(h_sh.data(), d_sh, M * sizeof(sycl::half));
    q.memcpy(h_cf.data(), d_cf, M * sizeof(float));
    q.memcpy(h_sf.data(), d_sf, M * sizeof(float));
    q.memcpy(h_cd.data(), d_cd, M * sizeof(double));
    q.memcpy(h_sd.data(), d_sd, M * sizeof(double));
    q.memcpy(h_cdf.data(), d_cdf, M * sizeof(sycl_fft::df_t));
    q.memcpy(h_sdf.data(), d_sdf, M * sizeof(sycl_fft::df_t));
    q.wait();

    sycl::free(d_x, q);
    sycl::free(d_ch, q);
    sycl::free(d_sh, q);
    sycl::free(d_cf, q);
    sycl::free(d_sf, q);
    sycl::free(d_cd, q);
    sycl::free(d_sd, q);
    sycl::free(d_cdf, q);
    sycl::free(d_sdf, q);

    double err_h = 0.0, err_f = 0.0, err_d = 0.0, err_df = 0.0;
    for (size_t i = 0; i < M; ++i) {
        const long double xh_ld = static_cast<long double>(static_cast<float>(static_cast<sycl::half>(static_cast<float>(h_x[i]))));
        err_h = std::max(err_h, static_cast<double>(std::fabsl(static_cast<long double>(static_cast<float>(h_ch[i])) - std::cos(kTwoPiLD * xh_ld))));
        err_h = std::max(err_h, static_cast<double>(std::fabsl(static_cast<long double>(static_cast<float>(h_sh[i])) - std::sin(kTwoPiLD * xh_ld))));

        const long double xf_ld = static_cast<long double>(static_cast<float>(h_x[i]));
        err_f = std::max(err_f, static_cast<double>(std::fabsl(static_cast<long double>(h_cf[i]) - std::cos(kTwoPiLD * xf_ld))));
        err_f = std::max(err_f, static_cast<double>(std::fabsl(static_cast<long double>(h_sf[i]) - std::sin(kTwoPiLD * xf_ld))));

        const long double xd_ld = static_cast<long double>(h_x[i]);
        err_d = std::max(err_d, static_cast<double>(std::fabsl(static_cast<long double>(h_cd[i]) - std::cos(kTwoPiLD * xd_ld))));
        err_d = std::max(err_d, static_cast<double>(std::fabsl(static_cast<long double>(h_sd[i]) - std::sin(kTwoPiLD * xd_ld))));

        const sycl_fft::df_t xdf = sycl_fft::df_from_double(h_x[i]);
        const long double xdf_ld = static_cast<long double>(xdf.hi) + static_cast<long double>(xdf.lo);
        const long double cdf_ld = static_cast<long double>(h_cdf[i].hi) + static_cast<long double>(h_cdf[i].lo);
        const long double sdf_ld = static_cast<long double>(h_sdf[i].hi) + static_cast<long double>(h_sdf[i].lo);
        err_df = std::max(err_df, static_cast<double>(std::fabsl(cdf_ld - std::cos(kTwoPiLD * xdf_ld))));
        err_df = std::max(err_df, static_cast<double>(std::fabsl(sdf_ld - std::sin(kTwoPiLD * xdf_ld))));
    }

    const bool ok = (err_h < 2e-3 && err_f < 2e-6 && err_d < 5e-15 && err_df < 1e-13);
    std::printf("=== Standalone Device Trig Verification (%zu pts incl. boundaries/neg/>1) ===\n", M);
    std::printf("  FP16 max_err = %.3e | FP32 max_err = %.3e | FP64 max_err = %.3e | DF max_err = %.3e | %s\n",
                err_h, err_f, err_d, err_df, ok ? "PASS" : "FAIL");
    return ok;
}

template <typename Traits>
void compute_errors_prec(
    const typename Traits::ComplexT* got,
    const std::vector<long double>& ref_re,
    const std::vector<long double>& ref_im,
    size_t N,
    double& max_abs,
    double& max_rel) {
    long double m_abs = 0.0L;
    long double m_ref = 0.0L;
    for (size_t i = 0; i < N; ++i) {
        long double gr, gi;
        Traits::to_long_doubles(got[i], gr, gi);
        const long double dx = gr - ref_re[i];
        const long double dy = gi - ref_im[i];
        const long double err = std::hypotl(dx, dy);
        const long double mag = std::hypotl(ref_re[i], ref_im[i]);
        if (err > m_abs) m_abs = err;
        if (mag > m_ref) m_ref = mag;
    }
    max_abs = static_cast<double>(m_abs);
    max_rel = static_cast<double>(m_abs / (m_ref + 1e-30L));
}

template <typename Traits, int Sign>
bool run_verification_prec(
    sycl::queue& q, int min_logN, int max_logN, const sycl_fft::FftLaunchConfig& cfg) {
    using ComplexT = typename Traits::ComplexT;
    using PlanT = typename Traits::template PlanT<Sign>;

    std::printf("\n=== Correctness Verification (%s DIF Radix-2, Sign=%+d vs 80-bit long double Ref) ===\n",
                Traits::kName, Sign);
    std::printf("%-5s %-8s %-4s | %-12s %-12s %-12s | %-12s | %s\n",
                "logN", "N", "bs", "Raw_MaxRel", "InPl_MaxRel", "BitRev_MaxRel", "Raw_MaxAbs", "Status");

    bool all_ok = true;
    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = (logN == 10) ? 1 : ((logN <= 14) ? 5 : 3);
        const size_t total = N * bs;
        std::vector<ComplexT> h_in(total), h_raw(total), h_inpl(total), h_nat(total);
        SplitMix64 rng(0x12345678ULL + static_cast<uint64_t>(logN) * 997ULL);
        for (size_t i = 0; i < total; ++i) {
            h_in[i] = Traits::from_doubles(rng.next_double(), rng.next_double());
        }

        auto* d_in = sycl::malloc_device<ComplexT>(total, q);
        auto* d_out = sycl::malloc_device<ComplexT>(total, q);
        q.memcpy(d_in, h_in.data(), total * sizeof(ComplexT)).wait();

        PlanT plan(q, logN, bs, cfg);

        // 1) Bit-reversal-free out-of-place (natural-to-reversed)
        plan.execute_raw(d_in, d_out, bs);
        q.memcpy(h_raw.data(), d_out, total * sizeof(ComplexT)).wait();

        // 2) Bit-reversal-free in-place (verifies in-place safety incl. 8192 chunk-1 stash)
        q.memcpy(d_out, h_in.data(), total * sizeof(ComplexT)).wait();
        plan.execute_raw(d_out, d_out, bs);
        q.memcpy(h_inpl.data(), d_out, total * sizeof(ComplexT)).wait();

        // 3) Full with GPU bit-reversal (natural-to-natural)
        plan.execute(d_in, d_out, bs);
        q.memcpy(h_nat.data(), d_out, total * sizeof(ComplexT)).wait();

        sycl::free(d_in, q);
        sycl::free(d_out, q);

        double worst_raw_rel = 0.0, worst_raw_abs = 0.0;
        double worst_inpl_rel = 0.0;
        double worst_nat_rel = 0.0;
        for (size_t b : {size_t{0}, bs - 1}) {
            std::vector<long double> ref_re(N), ref_im(N);
            for (size_t i = 0; i < N; ++i) {
                Traits::to_long_doubles(h_in[b * N + i], ref_re[i], ref_im[i]);
            }
            cpu_dif_radix2_ref(ref_re, ref_im, logN, Sign, false);
            double a = 0.0, r = 0.0;
            compute_errors_prec<Traits>(h_raw.data() + b * N, ref_re, ref_im, N, a, r);
            worst_raw_abs = std::max(worst_raw_abs, a);
            worst_raw_rel = std::max(worst_raw_rel, r);

            compute_errors_prec<Traits>(h_inpl.data() + b * N, ref_re, ref_im, N, a, r);
            worst_inpl_rel = std::max(worst_inpl_rel, r);

            for (size_t i = 0; i < N; ++i) {
                const size_t j = sycl_fft::bit_reverse(static_cast<uint32_t>(i), logN);
                if (j > i) {
                    std::swap(ref_re[i], ref_re[j]);
                    std::swap(ref_im[i], ref_im[j]);
                }
            }
            compute_errors_prec<Traits>(h_nat.data() + b * N, ref_re, ref_im, N, a, r);
            worst_nat_rel = std::max(worst_nat_rel, r);
        }

        const bool ok = (worst_raw_rel < Traits::kTolRel &&
                         worst_inpl_rel < Traits::kTolRel &&
                         worst_nat_rel < Traits::kTolRel);
        if (!ok) all_ok = false;
        std::printf("%-5d %-8zu %-4zu | %12.3e %12.3e %12.3e | %12.3e | %s\n",
                    logN, N, bs, worst_raw_rel, worst_inpl_rel, worst_nat_rel, worst_raw_abs,
                    ok ? "PASS" : "FAIL");
    }
    return all_ok;
}

struct BenchRow {
    int logN;
    double elapsed_ms;
    double gflops;
    double gbs;
};

template <typename Traits, int Sign = SYCL_FFT_DEFAULT_SIGN>
std::vector<BenchRow> run_benchmark_prec(
    sycl::queue& q,
    int log_total,
    int min_logN,
    int max_logN,
    const sycl_fft::FftLaunchConfig& cfg,
    int ntest,
    bool in_place) {
    using ComplexT = typename Traits::ComplexT;
    using PlanT = typename Traits::template PlanT<Sign>;
    const size_t total = size_t{1} << log_total;
    std::vector<ComplexT> h_in(total);
    SplitMix64 rng(42);
    for (size_t i = 0; i < total; ++i) {
        h_in[i] = Traits::from_doubles(rng.next_double(), rng.next_double());
    }

    auto* d_in = sycl::malloc_device<ComplexT>(total, q);
    auto* d_out = in_place ? d_in : sycl::malloc_device<ComplexT>(total, q);
    q.memcpy(d_in, h_in.data(), total * sizeof(ComplexT)).wait();

    // Warm up GPU clocks to P0 before timing
    {
        PlanT wplan(q, 12, total >> 12, cfg);
        const int warm_iters = (sizeof(ComplexT) == 8) ? 80 : 40;
        for (int w = 0; w < warm_iters; ++w) wplan.execute_raw(d_in, d_out, total >> 12);
        q.wait();
    }

    std::printf("\n=== Throughput Benchmark [%s]: Bit-Reversal-Free SYCL DIF Radix-2 (Sign=%+d, logTotal=%d, %.1f MiB/buf, %s) ===\n",
                Traits::kName,
                Sign,
                log_total,
                static_cast<double>(total * sizeof(ComplexT)) / (1024.0 * 1024.0),
                in_place ? "in-place" : "out-of-place");
    std::printf("%-5s %-8s %-7s | %-9s %-10s %-10s | %-34s %-7s %-8s\n",
                "logN", "N", "bs", "Time(ms)", "GFLOP/s", "Eff_GB/s", "Dispatch", "ChunkBS", "ChunkMiB");

    std::vector<BenchRow> rows;
    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = total / N;
        PlanT plan(q, logN, bs, cfg);

        for (int w = 0; w < 8; ++w) {
            plan.execute_raw(d_in, d_out, bs);
        }
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < ntest; ++it) {
            plan.execute_raw(d_in, d_out, bs);
        }
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ntest;
        const double flops = 5.0 * static_cast<double>(N) * static_cast<double>(logN) * static_cast<double>(bs);
        const double bytes = static_cast<double>(N) * static_cast<double>(bs) * sizeof(ComplexT) * 2.0;
        const double gflops = flops / (elapsed_ms * 1e6);
        const double gbs = bytes / (elapsed_ms * 1e6);

        char disp[64];
        const char* mark = (cfg.autotune != sycl_fft::AutotuneMode::Off && !plan.cache_hit()) ? "*" : "";
        if (plan.is_multistage()) {
            std::snprintf(disp, sizeof(disp), "ms:%s/w%zu",
                          plan.variant_name(), plan.pass0_wg_size());
        } else if (plan.is_two_pass()) {
            std::snprintf(disp, sizeof(disp), "2p:%dx%d/%s/w%zu%s",
                          1 << plan.log_n1(), 1 << plan.log_n2(),
                          plan.variant_name(), plan.pass0_wg_size(), mark);
        } else {
            std::snprintf(disp, sizeof(disp), "1p:%s%s", plan.variant_name(), mark);
        }
        const double chunk_mib = static_cast<double>(plan.chunk_bytes()) / (1024.0 * 1024.0);

        std::printf("%-5d %-8zu %-7zu | %9.3f %10.1f %10.1f | %-34s %-7zu %7.1f\n",
                    logN, N, bs, elapsed_ms, gflops, gbs, disp, plan.chunk_bs(), chunk_mib);
        rows.push_back(BenchRow{logN, elapsed_ms, gflops, gbs});
    }

    if (!in_place) sycl::free(d_out, q);
    sycl::free(d_in, q);
    return rows;
}

// Standalone bit-reversal benchmark timed strictly separately from FFT execution.
template <typename Traits>
void run_bitrev_benchmark_prec(
    sycl::queue& q,
    int log_total,
    int min_logN,
    int max_logN,
    const sycl_fft::FftLaunchConfig& cfg,
    int ntest,
    bool bitrev_in_place,
    const std::vector<BenchRow>& fft_rows) {
    using ComplexT = typename Traits::ComplexT;

    const auto dev_cfg = sycl_fft::DeviceFftConfig::query(q.get_device());
    const size_t total = size_t{1} << log_total;
    auto* d_src = sycl::malloc_device<ComplexT>(total, q);
    auto* d_dst = bitrev_in_place ? d_src : sycl::malloc_device<ComplexT>(total, q);
    q.memset(d_src, 0, total * sizeof(ComplexT)).wait();
    if (!bitrev_in_place) q.memset(d_dst, 0, total * sizeof(ComplexT)).wait();

    std::printf("\n=== Bit-Reversal Permutation Benchmark [%s] (logTotal=%d, %.1f MiB/buf, %s) ===\n",
                Traits::kName,
                log_total,
                static_cast<double>(total * sizeof(ComplexT)) / (1024.0 * 1024.0),
                bitrev_in_place ? "in-place" : "out-of-place");
    std::printf("%-5s %-8s %-7s | %-9s %-10s %-9s | %-9s | %s\n",
                "logN", "N", "bs", "Time(ms)", "Eff_GB/s", "ns/elem", "vs_FFT(%)", "Strategy");

    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = total / N;
        sycl_fft::BitRevPlan<ComplexT> br_plan(q, logN, Traits::kPrecTag, dev_cfg, cfg);
        br_plan.resolve(bitrev_in_place);

        for (int w = 0; w < 6; ++w) {
            if (bitrev_in_place) br_plan.execute_in_place(d_dst, bs);
            else br_plan.execute(d_src, d_dst, bs);
        }
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < ntest; ++it) {
            if (bitrev_in_place) br_plan.execute_in_place(d_dst, bs);
            else br_plan.execute(d_src, d_dst, bs);
        }
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ntest;
        const double bytes = static_cast<double>(total) * sizeof(ComplexT) * 2.0;
        const double gbs = bytes / (elapsed_ms * 1e6);
        const double ns_per_elem = (elapsed_ms * 1e6) / static_cast<double>(total);

        char pct_str[24] = "n/a";
        for (const auto& r : fft_rows) {
            if (r.logN == logN && r.elapsed_ms > 0.0) {
                std::snprintf(pct_str, sizeof(pct_str), "%7.1f%%", 100.0 * elapsed_ms / r.elapsed_ms);
                break;
            }
        }
        const char* mark = (cfg.autotune != sycl_fft::AutotuneMode::Off && !br_plan.cache_hit(bitrev_in_place)) ? "*" : "";
        std::printf("%-5d %-8zu %-7zu | %9.3f %10.1f %9.3f | %9s | %s%s\n",
                    logN, N, bs, elapsed_ms, gbs, ns_per_elem, pct_str,
                    br_plan.variant_name(bitrev_in_place), mark);
    }

    if (!bitrev_in_place) sycl::free(d_dst, q);
    sycl::free(d_src, q);
}

// Head-to-head interleaved comparison of ALL valid bit-reversal strategies per logN.
template <typename Traits>
bool run_bitrev_compare_prec(
    sycl::queue& q,
    int log_total,
    int min_logN,
    int max_logN,
    const sycl_fft::FftLaunchConfig& cfg,
    int ntest,
    bool bitrev_in_place) {
    using ComplexT = typename Traits::ComplexT;

    const auto dev_cfg = sycl_fft::DeviceFftConfig::query(q.get_device());
    const size_t total = size_t{1} << log_total;
    auto* d_src = sycl::malloc_device<ComplexT>(total, q);
    auto* d_dst = bitrev_in_place ? d_src : sycl::malloc_device<ComplexT>(total, q);
    q.memset(d_src, 0, total * sizeof(ComplexT)).wait();
    if (!bitrev_in_place) q.memset(d_dst, 0, total * sizeof(ComplexT)).wait();

    sycl_fft::FftLaunchConfig cmp_cfg = cfg;
    cmp_cfg.autotune = sycl_fft::AutotuneMode::Off;
    cmp_cfg.bitrev_variant_override.clear();

    std::printf("\n=== Bit-Reversal Strategy Comparison [%s] (logTotal=%d, %.1f MiB/buf, %s, %d interleaved rounds) ===\n",
                Traits::kName,
                log_total,
                static_cast<double>(total * sizeof(ComplexT)) / (1024.0 * 1024.0),
                bitrev_in_place ? "in-place" : "out-of-place",
                ntest);

    bool all_ok = true;
    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = total / N;
        sycl_fft::BitRevPlan<ComplexT> br_plan(q, logN, Traits::kPrecTag, dev_cfg, cmp_cfg);
        const auto& cands = br_plan.valid_candidates(bitrev_in_place);

        // Exact bit-for-bit verification on odd batch bs=5 against CPU bit_reverse
        const size_t ver_bs = 5;
        const size_t ver_total = ver_bs * N;
        std::vector<ComplexT> h_in(ver_total), h_ref(ver_total), h_got(ver_total);
        SplitMix64 rng(0xB17A0000ULL + static_cast<uint64_t>(logN));
        for (size_t i = 0; i < ver_total; ++i) {
            h_in[i] = Traits::from_doubles(rng.next_double(), rng.next_double());
        }
        for (size_t b = 0; b < ver_bs; ++b) {
            for (size_t i = 0; i < N; ++i) {
                const size_t r = sycl_fft::bit_reverse(static_cast<uint32_t>(i), logN);
                h_ref[b * N + i] = h_in[b * N + r];
            }
        }
        auto* dv_in = sycl::malloc_device<ComplexT>(ver_total, q);
        auto* dv_out = sycl::malloc_device<ComplexT>(ver_total, q);
        std::vector<bool> cand_ok(cands.size(), false);
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            if (bitrev_in_place) {
                q.memcpy(dv_out, h_in.data(), ver_total * sizeof(ComplexT)).wait();
                br_plan.execute_candidate(cands[ci], dv_out, dv_out, ver_bs);
            } else {
                q.memcpy(dv_in, h_in.data(), ver_total * sizeof(ComplexT)).wait();
                q.memset(dv_out, 0, ver_total * sizeof(ComplexT)).wait();
                br_plan.execute_candidate(cands[ci], dv_in, dv_out, ver_bs);
            }
            q.memcpy(h_got.data(), dv_out, ver_total * sizeof(ComplexT)).wait();
            cand_ok[ci] = (std::memcmp(h_got.data(), h_ref.data(), ver_total * sizeof(ComplexT)) == 0);
            if (!cand_ok[ci]) all_ok = false;
        }
        sycl::free(dv_in, q);
        sycl::free(dv_out, q);

        // Warmup every candidate (triggers JIT + lazy LUT init)
        for (const auto& c : cands) {
            for (int w = 0; w < 2; ++w) {
                br_plan.execute_candidate(c, d_src, d_dst, bs);
            }
            q.wait();
        }

        // Interleaved timed rounds
        std::vector<std::vector<double>> samples(cands.size());
        for (auto& s : samples) s.reserve(static_cast<size_t>(ntest));
        for (int r = 0; r < ntest; ++r) {
            for (size_t ci = 0; ci < cands.size(); ++ci) {
                const auto t0 = std::chrono::steady_clock::now();
                br_plan.execute_candidate(cands[ci], d_src, d_dst, bs);
                q.wait();
                const auto t1 = std::chrono::steady_clock::now();
                samples[ci].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }

        std::vector<double> med_ms(cands.size(), 0.0);
        double naive_ms = 0.0;
        size_t best_ci = 0;
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            std::sort(samples[ci].begin(), samples[ci].end());
            med_ms[ci] = samples[ci][samples[ci].size() / 2];
            if (std::strcmp(cands[ci].variant_id, "naive_w256") == 0) naive_ms = med_ms[ci];
            if (med_ms[ci] < med_ms[best_ci]) best_ci = ci;
        }
        if (naive_ms <= 0.0 && !med_ms.empty()) naive_ms = med_ms.back();

        std::printf("\n  -- logN=%d (N=%zu, bs=%zu) --\n", logN, N, bs);
        std::printf("  %-24s | %-9s %-10s %-9s | %-9s | %s\n",
                    "Strategy", "Time(ms)", "Eff_GB/s", "ns/elem", "vs_naive", "Exact");
        const double bytes = static_cast<double>(total) * sizeof(ComplexT) * 2.0;
        for (size_t ci = 0; ci < cands.size(); ++ci) {
            const double ms = med_ms[ci];
            const double gbs = bytes / (ms * 1e6);
            const double ns_e = (ms * 1e6) / static_cast<double>(total);
            const double spd = naive_ms / ms;
            std::printf("  %-24s | %9.3f %10.1f %9.3f | %8.2fx | %s%s\n",
                        cands[ci].variant_id, ms, gbs, ns_e, spd,
                        cand_ok[ci] ? "PASS" : "FAIL",
                        (ci == best_ci) ? "  <-- BEST" : "");
        }
    }

    if (!bitrev_in_place) sycl::free(d_dst, q);
    sycl::free(d_src, q);
    return all_ok;
}

template <typename Traits>
bool verify_bitrev_all_strategies(
    sycl::queue& q,
    const sycl_fft::DeviceFftConfig& dev_cfg,
    int min_logN,
    int max_logN) {
    using ComplexT = typename Traits::ComplexT;
    sycl_fft::FftLaunchConfig cfg{};
    cfg.autotune = sycl_fft::AutotuneMode::Off;

    const int top_logN = std::min(max_logN, 14);
    size_t checked = 0;
    for (int logN = min_logN; logN <= top_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = 5; // odd batch
        const size_t total = bs * N;
        std::vector<ComplexT> h_in(total), h_ref(total), h_got(total);
        SplitMix64 rng(0x5E1F0000ULL + static_cast<uint64_t>(logN) * 31ULL);
        for (size_t i = 0; i < total; ++i) {
            h_in[i] = Traits::from_doubles(rng.next_double(), rng.next_double());
        }
        for (size_t b = 0; b < bs; ++b) {
            for (size_t i = 0; i < N; ++i) {
                const size_t r = sycl_fft::bit_reverse(static_cast<uint32_t>(i), logN);
                h_ref[b * N + i] = h_in[b * N + r];
            }
        }
        auto* d_in = sycl::malloc_device<ComplexT>(total, q);
        auto* d_out = sycl::malloc_device<ComplexT>(total, q);
        sycl_fft::BitRevPlan<ComplexT> plan(q, logN, Traits::kPrecTag, dev_cfg, cfg);
        for (bool ip : {true, false}) {
            for (const auto& c : plan.valid_candidates(ip)) {
                if (ip) {
                    q.memcpy(d_out, h_in.data(), total * sizeof(ComplexT)).wait();
                    plan.execute_candidate(c, d_out, d_out, bs);
                } else {
                    q.memcpy(d_in, h_in.data(), total * sizeof(ComplexT)).wait();
                    q.memset(d_out, 0, total * sizeof(ComplexT)).wait();
                    plan.execute_candidate(c, d_in, d_out, bs);
                }
                q.memcpy(h_got.data(), d_out, total * sizeof(ComplexT)).wait();
                if (std::memcmp(h_got.data(), h_ref.data(), total * sizeof(ComplexT)) != 0) {
                    std::printf("    FAIL [%s] logN=%d %s strategy=%s mismatch!\n",
                                Traits::kName, logN, ip ? "in-place" : "out-of-place", c.variant_id);
                    sycl::free(d_in, q);
                    sycl::free(d_out, q);
                    return false;
                }
                ++checked;
            }
        }
        sycl::free(d_in, q);
        sycl::free(d_out, q);
    }
    std::printf("    [%s] %zu (logN x mode x strategy) bit-reversal checks bit-exact -> PASS\n",
                Traits::kName, checked);
    return true;
}

void list_sycl_devices() {
    const auto devs = sycl::device::get_devices();
    std::printf("Available SYCL devices (%zu):\n", devs.size());
    for (size_t i = 0; i < devs.size(); ++i) {
        const auto cfg = sycl_fft::DeviceFftConfig::query(devs[i]);
        std::printf("  [%zu] backend=%-10s type=%-3s name=%s (vendor=0x%x CUs=%zu NativeLocal=%d FP64=%d)\n",
                    i, cfg.backend_tag.c_str(), cfg.is_gpu ? "GPU" : "CPU",
                    cfg.device_name.c_str(), cfg.vendor_id, cfg.max_compute_units,
                    cfg.has_native_local_mem ? 1 : 0, cfg.has_fp64 ? 1 : 0);
    }
}

bool select_sycl_device(const std::string& spec, sycl::device& out_dev) {
    if (spec == "default") {
        out_dev = sycl::device{sycl::default_selector_v};
        return true;
    }
    const auto devs = sycl::device::get_devices();
    for (const auto& d : devs) {
        const std::string tag = sycl_fft::backend_to_tag(d);
        if ((spec == "gpu" && d.is_gpu()) ||
            (spec == "cpu" && d.is_cpu()) ||
            (spec == "cuda" && tag == "cuda") ||
            (spec == "hip" && tag == "hip") ||
            ((spec == "ocl" || spec == "opencl") && tag == "opencl") ||
            ((spec == "l0" || spec == "level_zero") && tag == "level_zero") ||
            (spec == "omp" && tag == "omp") ||
            (spec == "metal" && tag == "metal")) {
            out_dev = d;
            return true;
        }
    }
    return false;
}

bool run_self_tests(
    sycl::queue& q,
    const sycl_fft::DeviceFftConfig& dev_cfg,
    int min_logN,
    int max_logN,
    bool run_h,
    bool run_f,
    bool run_d,
    bool run_df) {
    std::printf("\n=== Hardware Generalization Self-Tests ===\n");

    const auto sg_probe = sycl_fft::probe_subgroup_topology(q, dev_cfg);
    std::printf("  [1] Sub-group topology probe: raw=%zu eff=%zu linear=%d xor=%d -> PASS\n",
                sg_probe.raw_sg_size, sg_probe.effective_sg_size,
                sg_probe.sg_linear_ok ? 1 : 0, sg_probe.sg_xor_ok ? 1 : 0);

    if (run_df) {
        const auto kat = sycl_fft::probe_df_kat(q, dev_cfg);
        std::printf("  [2a] Double-Float KAT: %s%s\n",
                    kat.df_kat_passed ? "PASS" : "FAIL: ",
                    kat.df_kat_passed ? "" : kat.df_kat_error.c_str());
        if (!kat.df_kat_passed) return false;
    }

    if (run_h) {
        const auto hkat = sycl_fft::probe_fp16_kat(q, dev_cfg);
        std::printf("  [2b] FP16 (2xFP16) KAT: %s%s\n",
                    hkat.fp16_kat_passed ? "PASS" : "FAIL: ",
                    hkat.fp16_kat_passed ? "" : hkat.fp16_kat_error.c_str());
        if (!hkat.fp16_kat_passed) return false;
    }

    // [3] Forced zero-smem multi-stage global fallback verification
    {
        sycl_fft::FftLaunchConfig ms_cfg{};
        ms_cfg.force_multistage = true;
        ms_cfg.autotune = sycl_fft::AutotuneMode::Off;
        std::printf("  [3] Forced zero-smem multi-stage fallback (logN=%d..%d):\n", min_logN, max_logN);
        if (run_h && !run_verification_prec<HalfTraits, SYCL_FFT_DEFAULT_SIGN>(q, min_logN, max_logN, ms_cfg)) return false;
        if (run_f && !run_verification_prec<FloatTraits, SYCL_FFT_DEFAULT_SIGN>(q, min_logN, max_logN, ms_cfg)) return false;
        if (run_d && dev_cfg.has_fp64 &&
            !run_verification_prec<DoubleTraits, SYCL_FFT_DEFAULT_SIGN>(q, min_logN, max_logN, ms_cfg)) return false;
        if (run_df && !run_verification_prec<DfTraits, SYCL_FFT_DEFAULT_SIGN>(q, min_logN, max_logN, ms_cfg)) return false;
    }

    // [4] Persistent disk cache round-trip (FFT + BitRev) or non-native refusal check
    if (dev_cfg.has_native_local_mem) {
        const std::string tmp_cache = "/tmp/sycl_fft_selftest_cache.txt";
        std::error_code ec;
        std::filesystem::remove(tmp_cache, ec);

        sycl_fft::FftLaunchConfig c_force{};
        c_force.autotune = sycl_fft::AutotuneMode::Force;
        c_force.cache_path = tmp_cache;
        sycl_fft::DifFftPlanf<SYCL_FFT_DEFAULT_SIGN> p1(q, 12, 64, c_force);
        const std::string v1 = p1.variant_name();
        const std::string br1 = p1.bitrev_variant_name(true);

        sycl_fft::AutotuneCache::clear_memory_cache();

        sycl_fft::FftLaunchConfig c_auto{};
        c_auto.autotune = sycl_fft::AutotuneMode::Auto;
        c_auto.cache_path = tmp_cache;
        sycl_fft::DifFftPlanf<SYCL_FFT_DEFAULT_SIGN> p2(q, 12, 64, c_auto);
        const std::string v2 = p2.variant_name();
        const std::string br2 = p2.bitrev_variant_name(true);

        std::filesystem::remove(tmp_cache, ec);
        const bool ok = p2.cache_hit() && (v1 == v2) && p2.bitrev_cache_hit(true) && (br1 == br2);
        std::printf("  [4] Disk autotune cache round-trip: fft=%s/%s(hit=%d) bitrev=%s/%s(hit=%d) -> %s\n",
                    v1.c_str(), v2.c_str(), p2.cache_hit() ? 1 : 0,
                    br1.c_str(), br2.c_str(), p2.bitrev_cache_hit(true) ? 1 : 0,
                    ok ? "PASS" : "FAIL");
        if (!ok) return false;
    } else {
        sycl_fft::FftLaunchConfig c_refuse{};
        c_refuse.allow_multistage_fallback = false;
        c_refuse.autotune = sycl_fft::AutotuneMode::Off;
        bool threw = false;
        try {
            sycl_fft::DifFftPlanf<SYCL_FFT_DEFAULT_SIGN> p_bad(q, 10, 4, c_refuse);
            (void)p_bad;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::printf("  [4] Non-native local-mem refusal when fallback disabled: threw=%d -> %s\n",
                    threw ? 1 : 0, threw ? "PASS" : "FAIL");
        if (!threw) return false;
    }

    // [5] Exact bit-for-bit cross-strategy bit-reversal equivalence (in-place & out-of-place, odd bs=5)
    std::printf("  [5] Bit-reversal cross-strategy bit-exact equivalence (logN=%d..%d, bs=5):\n",
                min_logN, std::min(max_logN, 14));
    if (run_h && !verify_bitrev_all_strategies<HalfTraits>(q, dev_cfg, min_logN, max_logN)) return false;
    if (run_f && !verify_bitrev_all_strategies<FloatTraits>(q, dev_cfg, min_logN, max_logN)) return false;
    if (run_d && dev_cfg.has_fp64 &&
        !verify_bitrev_all_strategies<DoubleTraits>(q, dev_cfg, min_logN, max_logN)) return false;
    if (run_df && !verify_bitrev_all_strategies<DfTraits>(q, dev_cfg, min_logN, max_logN)) return false;

    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    int log_total_override = -1;
    int min_logN = 10;
    int max_logN = 18;
    int ntest = 20;
    int chunk_mb_override = -1;
    sycl_fft::FftLaunchConfig cfg{};
    bool in_place = false;
    bool do_verify = true;
    bool do_bench = true;
    bool do_bitrev_bench = true;
    bool do_bitrev_compare = false;
    bool bitrev_oop = false;
    int active_sign = SYCL_FFT_DEFAULT_SIGN;
    bool verify_both_signs = false;
    bool skip_self_test = false;
    bool do_list_devices = false;
    std::string device_spec = "default";
    std::string prec = "all";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--log-total") == 0 && i + 1 < argc) {
            log_total_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--min-logN") == 0 && i + 1 < argc) {
            min_logN = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-logN") == 0 && i + 1 < argc) {
            max_logN = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--wg-size") == 0 && i + 1 < argc) {
            const size_t wg = static_cast<size_t>(std::atoi(argv[++i]));
            cfg.pass0_wg_size = wg;
            cfg.bitrev_wg_size = wg;
        } else if (std::strcmp(argv[i], "--two-pass-min") == 0 && i + 1 < argc) {
            cfg.two_pass_min_logN = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--two-pass-n1") == 0 && i + 1 < argc) {
            cfg.two_pass_log_n1 = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--l2-div") == 0 && i + 1 < argc) {
            cfg.l2_budget_den = static_cast<size_t>(std::max(1, std::atoi(argv[++i])));
        } else if (std::strcmp(argv[i], "--min-waves") == 0 && i + 1 < argc) {
            cfg.min_waves_per_cu = static_cast<size_t>(std::max(0, std::atoi(argv[++i])));
        } else if (std::strcmp(argv[i], "--no-wide") == 0) {
            cfg.allow_wide_kernels = false;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_spec = argv[++i];
        } else if (std::strcmp(argv[i], "--list-devices") == 0) {
            do_list_devices = true;
        } else if (std::strcmp(argv[i], "--no-autotune") == 0) {
            cfg.autotune = sycl_fft::AutotuneMode::Off;
        } else if (std::strcmp(argv[i], "--retune") == 0) {
            cfg.autotune = sycl_fft::AutotuneMode::Force;
        } else if (std::strcmp(argv[i], "--cache-path") == 0 && i + 1 < argc) {
            cfg.cache_path = argv[++i];
        } else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            cfg.variant_override = argv[++i];
        } else if (std::strcmp(argv[i], "--force-multistage") == 0) {
            cfg.force_multistage = true;
        } else if (std::strcmp(argv[i], "--no-multistage") == 0) {
            cfg.allow_multistage_fallback = false;
        } else if (std::strcmp(argv[i], "--allow-emulated-smem") == 0) {
            cfg.allow_emulated_local_mem = true;
        } else if (std::strcmp(argv[i], "--fp64-heavy") == 0 && i + 1 < argc) {
            cfg.fp64_compute_heavy_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--stash-8192") == 0 && i + 1 < argc) {
            cfg.stash_8192_override = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--skip-self-test") == 0) {
            skip_self_test = true;
        } else if (std::strcmp(argv[i], "--ntest") == 0 && i + 1 < argc) {
            ntest = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
            prec = argv[++i];
        } else if (std::strcmp(argv[i], "--in-place") == 0) {
            in_place = true;
        } else if (std::strcmp(argv[i], "--verify-only") == 0) {
            do_bench = false;
            do_bitrev_bench = false;
        } else if (std::strcmp(argv[i], "--bench-only") == 0) {
            do_verify = false;
        } else if (std::strcmp(argv[i], "--bitrev-only") == 0) {
            do_verify = false;
            do_bench = false;
            do_bitrev_bench = true;
        } else if (std::strcmp(argv[i], "--bitrev-compare") == 0) {
            do_verify = false;
            do_bench = false;
            do_bitrev_bench = false;
            do_bitrev_compare = true;
        } else if (std::strcmp(argv[i], "--bitrev-variant") == 0 && i + 1 < argc) {
            cfg.bitrev_variant_override = argv[++i];
        } else if (std::strcmp(argv[i], "--bitrev-oop") == 0) {
            bitrev_oop = true;
        } else if (std::strcmp(argv[i], "--skip-bitrev-bench") == 0) {
            do_bitrev_bench = false;
        } else if (std::strcmp(argv[i], "--verify-both-signs") == 0 ||
                   std::strcmp(argv[i], "--verify-pos-sign") == 0 ||
                   std::strcmp(argv[i], "--verify-neg-sign") == 0) {
            verify_both_signs = true;
        } else if (std::strcmp(argv[i], "--sign") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            if (std::strcmp(s, "+1") == 0 || std::strcmp(s, "1") == 0 || std::strcmp(s, "pos") == 0) {
                active_sign = +1;
            } else if (std::strcmp(s, "-1") == 0 || std::strcmp(s, "neg") == 0) {
                active_sign = -1;
            } else {
                std::cerr << "Unknown --sign value: " << s << " (use +1|pos|-1|neg)\n";
                return 1;
            }
        }
    }

    if (do_list_devices) {
        list_sycl_devices();
        return 0;
    }

    if (chunk_mb_override > 0) {
        cfg.l2_chunk_bytes = static_cast<size_t>(chunk_mb_override) * 1024 * 1024;
    }

    const bool run_h = (prec == "all" || prec == "h" || prec == "fp16");
    const bool run_f = (prec == "all" || prec == "f" || prec == "fp32");
    const bool run_d = (prec == "all" || prec == "d" || prec == "fp64");
    const bool run_df = (prec == "all" || prec == "df");
    if (!run_h && !run_f && !run_d && !run_df) {
        std::cerr << "Unknown --prec value: " << prec << " (use all|h|fp16|f|fp32|d|fp64|df)\n";
        return 1;
    }

    if ((active_sign == +1 && !SYCL_FFT_ENABLE_POS_SIGN) ||
        (active_sign == -1 && !SYCL_FFT_ENABLE_NEG_SIGN)) {
        std::cerr << "Requested Sign=" << active_sign
                  << " is not compiled in this binary (rebuild with SIGN=pos|neg|both).\n";
        return 1;
    }

    sycl::device dev;
    if (!select_sycl_device(device_spec, dev)) {
        std::cerr << "Requested --device '" << device_spec << "' not found.\n";
        list_sycl_devices();
        return 1;
    }

    sycl::queue q{dev, sycl::property::queue::in_order{}};
    const auto dev_cfg = sycl_fft::DeviceFftConfig::query(dev);
    const auto sg_probe = sycl_fft::probe_subgroup_topology(q, dev_cfg);
    sycl_fft::DeviceProbeResult fp64_probe{};
    if (dev_cfg.has_fp64 && run_d) {
        fp64_probe = sycl_fft::probe_fp64_ratio(q, dev_cfg);
    }
    sycl_fft::DeviceProbeResult df_probe{};
    if (run_df) {
        df_probe = sycl_fft::probe_df_kat(q, dev_cfg);
    }
    sycl_fft::DeviceProbeResult fp16_probe{};
    if (run_h) {
        fp16_probe = sycl_fft::probe_fp16_kat(q, dev_cfg);
    }

    const char* compiled_signs =
#if SYCL_FFT_ENABLE_POS_SIGN && SYCL_FFT_ENABLE_NEG_SIGN
        "+1,-1";
#elif SYCL_FFT_ENABLE_POS_SIGN
        "+1";
#else
        "-1";
#endif

    std::printf(
        "Device: %s | Backend=%s | Vendor=0x%x | Driver=%s\n"
        "  CUs=%zu | L2=%.1f MiB | LocalMem=%.0f KiB (native=%d) | BankModel=%s | MaxWG=%zu\n"
        "  SG: reported=%zu..%zu verified=%zu (lin=%d xor=%d) | FP64=%d (ratio=%.3f heavy=%d) | FP16_KAT=%s | DF_KAT=%s | Signs=%s (active=%+d)\n",
        dev_cfg.device_name.c_str(),
        dev_cfg.backend_tag.c_str(),
        dev_cfg.vendor_id,
        dev_cfg.driver_version.c_str(),
        dev_cfg.max_compute_units,
        static_cast<double>(dev_cfg.l2_cache_size) / (1024.0 * 1024.0),
        static_cast<double>(dev_cfg.local_mem_size) / 1024.0,
        dev_cfg.has_native_local_mem ? 1 : 0,
        sycl_fft::bank_model_name(dev_cfg.bank_model),
        dev_cfg.max_work_group_size,
        dev_cfg.min_sub_group_size,
        dev_cfg.sub_group_size,
        sg_probe.effective_sg_size,
        sg_probe.sg_linear_ok ? 1 : 0,
        sg_probe.sg_xor_ok ? 1 : 0,
        dev_cfg.has_fp64 ? 1 : 0,
        fp64_probe.fp64_to_fp32_ratio,
        fp64_probe.fp64_compute_heavy ? 1 : 0,
        run_h ? (fp16_probe.fp16_kat_passed ? "PASS" : "FAIL") : "skip",
        run_df ? (df_probe.df_kat_passed ? "PASS" : "FAIL") : "skip",
        compiled_signs,
        active_sign);

    if (do_verify) {
        if (!verify_standalone_trig(q)) {
            std::cerr << "STANDALONE TRIG VERIFICATION FAILED!\n";
            return 1;
        }
        auto verify_enabled_signs = [&](auto tag) -> bool {
            using Traits = decltype(tag);
#if SYCL_FFT_ENABLE_POS_SIGN && SYCL_FFT_ENABLE_NEG_SIGN
            if (verify_both_signs) {
                if (!run_verification_prec<Traits, +1>(q, min_logN, max_logN, cfg)) return false;
                if (!run_verification_prec<Traits, -1>(q, min_logN, max_logN, cfg)) return false;
                return true;
            }
            if (active_sign == +1) {
                return run_verification_prec<Traits, +1>(q, min_logN, max_logN, cfg);
            }
            return run_verification_prec<Traits, -1>(q, min_logN, max_logN, cfg);
#elif SYCL_FFT_ENABLE_POS_SIGN
            (void)verify_both_signs;
            return run_verification_prec<Traits, +1>(q, min_logN, max_logN, cfg);
#else
            (void)verify_both_signs;
            return run_verification_prec<Traits, -1>(q, min_logN, max_logN, cfg);
#endif
        };
        if (run_h && !verify_enabled_signs(HalfTraits{})) return 1;
        if (run_f && !verify_enabled_signs(FloatTraits{})) return 1;
        if (run_d) {
            if (!dev_cfg.has_fp64) {
                std::printf("\n[FP64] Skipping on device without sycl::aspect::fp64 (verifying expected throw)...\n");
                bool threw = false;
                try {
                    sycl_fft::DifFftPlan<SYCL_FFT_DEFAULT_SIGN> p(q, min_logN, 1, cfg);
                    (void)p;
                } catch (const std::runtime_error&) {
                    threw = true;
                }
                if (!threw) {
                    std::cerr << "Expected FP64 plan to throw on !has_fp64 device!\n";
                    return 1;
                }
            } else {
                if (!verify_enabled_signs(DoubleTraits{})) return 1;
            }
        }
        if (run_df && !verify_enabled_signs(DfTraits{})) return 1;
        if (!skip_self_test && !cfg.has_plan_overrides()) {
            if (!run_self_tests(q, dev_cfg, min_logN, max_logN, run_h, run_f, run_d, run_df)) return 1;
        }
    }

    std::vector<BenchRow> rows_h, rows_f, rows_d, rows_df;
    if (do_bench) {
        auto bench_active_sign = [&](auto tag, int lt) -> std::vector<BenchRow> {
            using Traits = decltype(tag);
#if SYCL_FFT_ENABLE_POS_SIGN && SYCL_FFT_ENABLE_NEG_SIGN
            if (active_sign == +1) {
                return run_benchmark_prec<Traits, +1>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
            }
            return run_benchmark_prec<Traits, -1>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
#elif SYCL_FFT_ENABLE_POS_SIGN
            return run_benchmark_prec<Traits, +1>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
#else
            return run_benchmark_prec<Traits, -1>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
#endif
        };
        if (run_h) {
            const int lt = (log_total_override > 0) ? log_total_override : HalfTraits::kDefaultLogTotal;
            rows_h = bench_active_sign(HalfTraits{}, lt);
        }
        if (run_f) {
            const int lt = (log_total_override > 0) ? log_total_override : FloatTraits::kDefaultLogTotal;
            rows_f = bench_active_sign(FloatTraits{}, lt);
        }
        if (run_d && dev_cfg.has_fp64) {
            const int lt = (log_total_override > 0) ? log_total_override : DoubleTraits::kDefaultLogTotal;
            rows_d = bench_active_sign(DoubleTraits{}, lt);
        }
        if (run_df) {
            const int lt = (log_total_override > 0) ? log_total_override : DfTraits::kDefaultLogTotal;
            rows_df = bench_active_sign(DfTraits{}, lt);
        }
        if (!rows_h.empty() && rows_h.size() == rows_f.size()) {
            std::printf("\n=== Summary: FP16 (2xFP16) vs FP32 Speedup ===\n");
            std::printf("%-5s | %-10s %-10s | %-10s\n", "logN", "FP32 GF/s", "FP16 GF/s", "FP16/FP32");
            for (size_t i = 0; i < rows_f.size(); ++i) {
                const double ratio = rows_h[i].gflops / rows_f[i].gflops;
                std::printf("%-5d | %10.1f %10.1f | %9.2fx\n",
                            rows_f[i].logN, rows_f[i].gflops, rows_h[i].gflops, ratio);
            }
        }
        if (!rows_d.empty() && rows_d.size() == rows_df.size()) {
            std::printf("\n=== Summary: DF (2xFP32) vs Native FP64 Speedup ===\n");
            std::printf("%-5s | %-10s %-10s | %-10s\n", "logN", "FP64 GF/s", "DF GF/s", "DF/FP64");
            for (size_t i = 0; i < rows_d.size(); ++i) {
                const double ratio = rows_df[i].gflops / rows_d[i].gflops;
                std::printf("%-5d | %10.1f %10.1f | %9.2fx\n",
                            rows_d[i].logN, rows_d[i].gflops, rows_df[i].gflops, ratio);
            }
        }
    }

    if (do_bitrev_bench) {
        const bool br_ip = !bitrev_oop;
        if (run_h) {
            const int lt = (log_total_override > 0) ? log_total_override : HalfTraits::kDefaultLogTotal;
            run_bitrev_benchmark_prec<HalfTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip, rows_h);
        }
        if (run_f) {
            const int lt = (log_total_override > 0) ? log_total_override : FloatTraits::kDefaultLogTotal;
            run_bitrev_benchmark_prec<FloatTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip, rows_f);
        }
        if (run_d && dev_cfg.has_fp64) {
            const int lt = (log_total_override > 0) ? log_total_override : DoubleTraits::kDefaultLogTotal;
            run_bitrev_benchmark_prec<DoubleTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip, rows_d);
        }
        if (run_df) {
            const int lt = (log_total_override > 0) ? log_total_override : DfTraits::kDefaultLogTotal;
            run_bitrev_benchmark_prec<DfTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip, rows_df);
        }
    }

    if (do_bitrev_compare) {
        const bool br_ip = !bitrev_oop;
        bool ok = true;
        if (run_h) {
            const int lt = (log_total_override > 0) ? log_total_override : HalfTraits::kDefaultLogTotal;
            ok = run_bitrev_compare_prec<HalfTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip) && ok;
        }
        if (run_f) {
            const int lt = (log_total_override > 0) ? log_total_override : FloatTraits::kDefaultLogTotal;
            ok = run_bitrev_compare_prec<FloatTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip) && ok;
        }
        if (run_d && dev_cfg.has_fp64) {
            const int lt = (log_total_override > 0) ? log_total_override : DoubleTraits::kDefaultLogTotal;
            ok = run_bitrev_compare_prec<DoubleTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip) && ok;
        }
        if (run_df) {
            const int lt = (log_total_override > 0) ? log_total_override : DfTraits::kDefaultLogTotal;
            ok = run_bitrev_compare_prec<DfTraits>(q, lt, min_logN, max_logN, cfg, ntest, br_ip) && ok;
        }
        if (!ok) return 1;
    }

    return 0;
}
