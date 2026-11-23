#include "sycl_fft_dif.hpp"

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
                const long double ar = re[i + j], ai = im[i + j];
                const long double br = re[i + j + half], bi = im[i + j + half];
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
struct FloatTraits {
    using ComplexT = sycl_fft::Complexf;
    template <int Sign> using PlanT = sycl_fft::DifFftPlanf<Sign>;
    static constexpr const char* kName = "FP32";
    static constexpr int kDefaultLogTotal = 25;   // 256 MiB / buffer
    static constexpr size_t kDefaultChunkMB = 6;
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
    static constexpr int kDefaultLogTotal = 24;   // 256 MiB / buffer
    static constexpr size_t kDefaultChunkMB = 16;
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
    static constexpr int kDefaultLogTotal = 24;   // 256 MiB / buffer
    static constexpr size_t kDefaultChunkMB = 16;
    static constexpr double kTolRel = 5e-12;

    static ComplexT from_doubles(double r, double i) {
        return ComplexT{sycl_fft::df_from_double(r), sycl_fft::df_from_double(i)};
    }
    static void to_long_doubles(const ComplexT& v, long double& r, long double& i) {
        r = static_cast<long double>(v.re.hi) + static_cast<long double>(v.re.lo);
        i = static_cast<long double>(v.im.hi) + static_cast<long double>(v.im.lo);
    }
};

// Standalone verification of branchless polynomial cos2pi / sin2pi across all 3 precisions
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
        const sycl_fft::df_t xdf = sycl_fft::df_from_double(xd);
        d_cf[i] = sycl_fft::cos2pif(xf);
        d_sf[i] = sycl_fft::sin2pif(xf);
        d_cd[i] = sycl_fft::cos2pi(xd);
        d_sd[i] = sycl_fft::sin2pi(xd);
        d_cdf[i] = sycl_fft::cos2pidf(xdf);
        d_sdf[i] = sycl_fft::sin2pidf(xdf);
    }).wait();

    std::vector<float> h_cf(M), h_sf(M);
    std::vector<double> h_cd(M), h_sd(M);
    std::vector<sycl_fft::df_t> h_cdf(M), h_sdf(M);
    q.memcpy(h_cf.data(), d_cf, M * sizeof(float));
    q.memcpy(h_sf.data(), d_sf, M * sizeof(float));
    q.memcpy(h_cd.data(), d_cd, M * sizeof(double));
    q.memcpy(h_sd.data(), d_sd, M * sizeof(double));
    q.memcpy(h_cdf.data(), d_cdf, M * sizeof(sycl_fft::df_t));
    q.memcpy(h_sdf.data(), d_sdf, M * sizeof(sycl_fft::df_t));
    q.wait();

    sycl::free(d_x, q);
    sycl::free(d_cf, q);
    sycl::free(d_sf, q);
    sycl::free(d_cd, q);
    sycl::free(d_sd, q);
    sycl::free(d_cdf, q);
    sycl::free(d_sdf, q);

    double err_f = 0.0, err_d = 0.0, err_df = 0.0;
    for (size_t i = 0; i < M; ++i) {
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

    const bool ok = (err_f < 2e-6 && err_d < 5e-15 && err_df < 1e-13);
    std::printf("=== Standalone Device Trig Verification (%zu pts incl. boundaries/neg/>1) ===\n", M);
    std::printf("  FP32 max_err = %.3e | FP64 max_err = %.3e | DF max_err = %.3e | %s\n",
                err_f, err_d, err_df, ok ? "PASS" : "FAIL");
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
bool run_verification_prec(sycl::queue& q, int min_logN, int max_logN) {
    using ComplexT = typename Traits::ComplexT;
    using PlanT = typename Traits::template PlanT<Sign>;

    std::printf("\n=== Correctness Verification (%s DIF Radix-2, Sign=%+d vs 80-bit long double Ref) ===\n",
                Traits::kName, Sign);
    std::printf("%-5s %-8s %-4s | %-12s %-12s | %-12s %-12s | %s\n",
                "logN", "N", "bs", "Raw_MaxRel", "BitRev_MaxRel", "Raw_MaxAbs", "BitRev_MaxAbs", "Status");

    bool all_ok = true;
    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = (logN == 10) ? 1 : ((logN <= 14) ? 5 : 3);
        const size_t total = N * bs;
        std::vector<ComplexT> h_in(total), h_raw(total), h_nat(total);
        SplitMix64 rng(0x12345678ULL + static_cast<uint64_t>(logN) * 997ULL);
        for (size_t i = 0; i < total; ++i) {
            h_in[i] = Traits::from_doubles(rng.next_double(), rng.next_double());
        }

        auto* d_in = sycl::malloc_device<ComplexT>(total, q);
        auto* d_out = sycl::malloc_device<ComplexT>(total, q);
        q.memcpy(d_in, h_in.data(), total * sizeof(ComplexT)).wait();

        PlanT plan(q, logN, bs, 4 * 1024 * 1024);

        // 1) Bit-reversal-free (natural-to-reversed)
        plan.execute_raw(d_in, d_out, bs);
        q.memcpy(h_raw.data(), d_out, total * sizeof(ComplexT)).wait();

        // 2) Full with GPU bit-reversal (natural-to-natural)
        plan.execute(d_in, d_out, bs);
        q.memcpy(h_nat.data(), d_out, total * sizeof(ComplexT)).wait();

        sycl::free(d_in, q);
        sycl::free(d_out, q);

        double worst_raw_rel = 0.0, worst_raw_abs = 0.0;
        double worst_nat_rel = 0.0, worst_nat_abs = 0.0;
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

            for (size_t i = 0; i < N; ++i) {
                const size_t j = sycl_fft::bit_reverse(static_cast<uint32_t>(i), logN);
                if (j > i) {
                    std::swap(ref_re[i], ref_re[j]);
                    std::swap(ref_im[i], ref_im[j]);
                }
            }
            compute_errors_prec<Traits>(h_nat.data() + b * N, ref_re, ref_im, N, a, r);
            worst_nat_abs = std::max(worst_nat_abs, a);
            worst_nat_rel = std::max(worst_nat_rel, r);
        }

        const bool ok = (worst_raw_rel < Traits::kTolRel && worst_nat_rel < Traits::kTolRel);
        if (!ok) all_ok = false;
        std::printf("%-5d %-8zu %-4zu | %12.3e %12.3e | %12.3e %12.3e | %s\n",
                    logN, N, bs, worst_raw_rel, worst_nat_rel, worst_raw_abs, worst_nat_abs,
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

template <typename Traits>
std::vector<BenchRow> run_benchmark_prec(
    sycl::queue& q, int log_total, int min_logN, int max_logN, size_t chunk_bytes, int ntest, bool in_place) {
    using ComplexT = typename Traits::ComplexT;
    using PlanT = typename Traits::template PlanT<-1>;

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
        PlanT wplan(q, 12, total >> 12, chunk_bytes);
        const int warm_iters = (sizeof(ComplexT) == 8) ? 80 : 15;
        for (int w = 0; w < warm_iters; ++w) wplan.execute_raw(d_in, d_out, total >> 12);
        q.wait();
    }

    std::printf("\n=== Throughput Benchmark [%s]: Bit-Reversal-Free SYCL DIF Radix-2 (logTotal=%d, %.1f MiB/buf, L2Chunk=%.1f MiB, %s) ===\n",
                Traits::kName,
                log_total,
                static_cast<double>(total * sizeof(ComplexT)) / (1024.0 * 1024.0),
                static_cast<double>(chunk_bytes) / (1024.0 * 1024.0),
                in_place ? "in-place" : "out-of-place");
    std::printf("%-5s %-8s %-7s | %-9s %-10s %-10s | %-8s\n",
                "logN", "N", "bs", "Time(ms)", "GFLOP/s", "Eff_GB/s", "ChunkBS");

    std::vector<BenchRow> rows;
    for (int logN = min_logN; logN <= max_logN; ++logN) {
        const size_t N = size_t{1} << logN;
        const size_t bs = total / N;
        PlanT plan(q, logN, bs, chunk_bytes);

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

        std::printf("%-5d %-8zu %-7zu | %9.3f %10.1f %10.1f | %-8zu\n",
                    logN, N, bs, elapsed_ms, gflops, gbs, plan.chunk_bs());
        rows.push_back(BenchRow{logN, elapsed_ms, gflops, gbs});
    }

    if (!in_place) sycl::free(d_out, q);
    sycl::free(d_in, q);
    return rows;
}

} // namespace

int main(int argc, char** argv) {
    int log_total_override = -1;
    int min_logN = 10;
    int max_logN = 18;
    int ntest = 20;
    int chunk_mb_override = -1;
    bool in_place = false;
    bool do_verify = true;
    bool do_bench = true;
    bool verify_pos_sign = false;
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
        } else if (std::strcmp(argv[i], "--ntest") == 0 && i + 1 < argc) {
            ntest = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
            prec = argv[++i];
        } else if (std::strcmp(argv[i], "--in-place") == 0) {
            in_place = true;
        } else if (std::strcmp(argv[i], "--verify-only") == 0) {
            do_bench = false;
        } else if (std::strcmp(argv[i], "--bench-only") == 0) {
            do_verify = false;
        } else if (std::strcmp(argv[i], "--verify-pos-sign") == 0) {
            verify_pos_sign = true;
        }
    }

    const bool run_f = (prec == "all" || prec == "f" || prec == "fp32");
    const bool run_d = (prec == "all" || prec == "d" || prec == "fp64");
    const bool run_df = (prec == "all" || prec == "df");
    if (!run_f && !run_d && !run_df) {
        std::cerr << "Unknown --prec value: " << prec << " (use all|f|fp32|d|fp64|df)\n";
        return 1;
    }

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << '\n';

    if (do_verify) {
        if (!verify_standalone_trig(q)) {
            std::cerr << "STANDALONE TRIG VERIFICATION FAILED!\n";
            return 1;
        }
        if (run_f) {
            if (!run_verification_prec<FloatTraits, -1>(q, min_logN, max_logN)) return 1;
            if (verify_pos_sign && !run_verification_prec<FloatTraits, +1>(q, min_logN, max_logN)) return 1;
        }
        if (run_d) {
            if (!run_verification_prec<DoubleTraits, -1>(q, min_logN, max_logN)) return 1;
            if (verify_pos_sign && !run_verification_prec<DoubleTraits, +1>(q, min_logN, max_logN)) return 1;
        }
        if (run_df) {
            if (!run_verification_prec<DfTraits, -1>(q, min_logN, max_logN)) return 1;
            if (verify_pos_sign && !run_verification_prec<DfTraits, +1>(q, min_logN, max_logN)) return 1;
        }
    }

    if (do_bench) {
        std::vector<BenchRow> rows_d, rows_df;
        if (run_f) {
            const int lt = (log_total_override > 0) ? log_total_override : FloatTraits::kDefaultLogTotal;
            const size_t cmb = (chunk_mb_override > 0) ? static_cast<size_t>(chunk_mb_override) : FloatTraits::kDefaultChunkMB;
            run_benchmark_prec<FloatTraits>(q, lt, min_logN, max_logN, cmb * 1024 * 1024, ntest, in_place);
        }
        if (run_d) {
            const int lt = (log_total_override > 0) ? log_total_override : DoubleTraits::kDefaultLogTotal;
            const size_t cmb = (chunk_mb_override > 0) ? static_cast<size_t>(chunk_mb_override) : DoubleTraits::kDefaultChunkMB;
            rows_d = run_benchmark_prec<DoubleTraits>(q, lt, min_logN, max_logN, cmb * 1024 * 1024, ntest, in_place);
        }
        if (run_df) {
            const int lt = (log_total_override > 0) ? log_total_override : DfTraits::kDefaultLogTotal;
            const size_t cmb = (chunk_mb_override > 0) ? static_cast<size_t>(chunk_mb_override) : DfTraits::kDefaultChunkMB;
            rows_df = run_benchmark_prec<DfTraits>(q, lt, min_logN, max_logN, cmb * 1024 * 1024, ntest, in_place);
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

    return 0;
}
