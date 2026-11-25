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
struct FloatTraits {
    using ComplexT = sycl_fft::Complexf;
    template <int Sign> using PlanT = sycl_fft::DifFftPlanf<Sign>;
    static constexpr const char* kName = "FP32";
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

template <typename Traits>
std::vector<BenchRow> run_benchmark_prec(
    sycl::queue& q,
    int log_total,
    int min_logN,
    int max_logN,
    const sycl_fft::FftLaunchConfig& cfg,
    int ntest,
    bool in_place) {
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
        PlanT wplan(q, 12, total >> 12, cfg);
        const int warm_iters = (sizeof(ComplexT) == 8) ? 80 : 40;
        for (int w = 0; w < warm_iters; ++w) wplan.execute_raw(d_in, d_out, total >> 12);
        q.wait();
    }

    std::printf("\n=== Throughput Benchmark [%s]: Bit-Reversal-Free SYCL DIF Radix-2 (logTotal=%d, %.1f MiB/buf, %s) ===\n",
                Traits::kName,
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
        std::printf("  [2] Double-Float KAT: %s%s\n",
                    kat.df_kat_passed ? "PASS" : "FAIL: ",
                    kat.df_kat_passed ? "" : kat.df_kat_error.c_str());
        if (!kat.df_kat_passed) return false;
    }

    // [3] Forced zero-smem multi-stage global fallback verification
    {
        sycl_fft::FftLaunchConfig ms_cfg{};
        ms_cfg.force_multistage = true;
        ms_cfg.autotune = sycl_fft::AutotuneMode::Off;
        std::printf("  [3] Forced zero-smem multi-stage fallback (logN=%d..%d):\n", min_logN, max_logN);
        if (run_f && !run_verification_prec<FloatTraits, -1>(q, min_logN, max_logN, ms_cfg)) return false;
        if (run_d && dev_cfg.has_fp64 &&
            !run_verification_prec<DoubleTraits, -1>(q, min_logN, max_logN, ms_cfg)) return false;
        if (run_df && !run_verification_prec<DfTraits, -1>(q, min_logN, max_logN, ms_cfg)) return false;
    }

    // [4] Persistent disk cache round-trip (on native-local-mem devices) or refusal check
    if (dev_cfg.has_native_local_mem) {
        const std::string tmp_cache = "/tmp/sycl_fft_selftest_cache.txt";
        std::error_code ec;
        std::filesystem::remove(tmp_cache, ec);

        sycl_fft::FftLaunchConfig c_force{};
        c_force.autotune = sycl_fft::AutotuneMode::Force;
        c_force.cache_path = tmp_cache;
        sycl_fft::DifFftPlanf<-1> p1(q, 12, 64, c_force);
        const std::string v1 = p1.variant_name();

        sycl_fft::AutotuneCache::clear_memory_cache();

        sycl_fft::FftLaunchConfig c_auto{};
        c_auto.autotune = sycl_fft::AutotuneMode::Auto;
        c_auto.cache_path = tmp_cache;
        sycl_fft::DifFftPlanf<-1> p2(q, 12, 64, c_auto);
        const std::string v2 = p2.variant_name();

        std::filesystem::remove(tmp_cache, ec);
        const bool ok = p2.cache_hit() && (v1 == v2);
        std::printf("  [4] Disk autotune cache round-trip: v1=%s v2=%s cache_hit=%d -> %s\n",
                    v1.c_str(), v2.c_str(), p2.cache_hit() ? 1 : 0, ok ? "PASS" : "FAIL");
        if (!ok) return false;
    } else {
        sycl_fft::FftLaunchConfig c_refuse{};
        c_refuse.allow_multistage_fallback = false;
        c_refuse.autotune = sycl_fft::AutotuneMode::Off;
        bool threw = false;
        try {
            sycl_fft::DifFftPlanf<-1> p_bad(q, 10, 4, c_refuse);
            (void)p_bad;
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::printf("  [4] Non-native local-mem refusal when fallback disabled: threw=%d -> %s\n",
                    threw ? 1 : 0, threw ? "PASS" : "FAIL");
        if (!threw) return false;
    }

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
    bool verify_pos_sign = false;
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
        } else if (std::strcmp(argv[i], "--bench-only") == 0) {
            do_verify = false;
        } else if (std::strcmp(argv[i], "--verify-pos-sign") == 0) {
            verify_pos_sign = true;
        }
    }

    if (do_list_devices) {
        list_sycl_devices();
        return 0;
    }

    if (chunk_mb_override > 0) {
        cfg.l2_chunk_bytes = static_cast<size_t>(chunk_mb_override) * 1024 * 1024;
    }

    const bool run_f = (prec == "all" || prec == "f" || prec == "fp32");
    const bool run_d = (prec == "all" || prec == "d" || prec == "fp64");
    const bool run_df = (prec == "all" || prec == "df");
    if (!run_f && !run_d && !run_df) {
        std::cerr << "Unknown --prec value: " << prec << " (use all|f|fp32|d|fp64|df)\n";
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

    std::printf(
        "Device: %s | Backend=%s | Vendor=0x%x | Driver=%s\n"
        "  CUs=%zu | L2=%.1f MiB | LocalMem=%.0f KiB (native=%d) | BankModel=%s | MaxWG=%zu\n"
        "  SG: reported=%zu..%zu verified=%zu (lin=%d xor=%d) | FP64=%d (ratio=%.3f heavy=%d) | DF_KAT=%s\n",
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
        run_df ? (df_probe.df_kat_passed ? "PASS" : "FAIL") : "skip");

    if (do_verify) {
        if (!verify_standalone_trig(q)) {
            std::cerr << "STANDALONE TRIG VERIFICATION FAILED!\n";
            return 1;
        }
        if (run_f) {
            if (!run_verification_prec<FloatTraits, -1>(q, min_logN, max_logN, cfg)) return 1;
            if (verify_pos_sign && !run_verification_prec<FloatTraits, +1>(q, min_logN, max_logN, cfg)) return 1;
        }
        if (run_d) {
            if (!dev_cfg.has_fp64) {
                std::printf("\n[FP64] Skipping on device without sycl::aspect::fp64 (verifying expected throw)...\n");
                bool threw = false;
                try {
                    sycl_fft::DifFftPlan<-1> p(q, min_logN, 1, cfg);
                    (void)p;
                } catch (const std::runtime_error&) {
                    threw = true;
                }
                if (!threw) {
                    std::cerr << "Expected FP64 plan to throw on !has_fp64 device!\n";
                    return 1;
                }
            } else {
                if (!run_verification_prec<DoubleTraits, -1>(q, min_logN, max_logN, cfg)) return 1;
                if (verify_pos_sign && !run_verification_prec<DoubleTraits, +1>(q, min_logN, max_logN, cfg)) return 1;
            }
        }
        if (run_df) {
            if (!run_verification_prec<DfTraits, -1>(q, min_logN, max_logN, cfg)) return 1;
            if (verify_pos_sign && !run_verification_prec<DfTraits, +1>(q, min_logN, max_logN, cfg)) return 1;
        }
        if (!skip_self_test && !cfg.has_plan_overrides()) {
            if (!run_self_tests(q, dev_cfg, min_logN, max_logN, run_f, run_d, run_df)) return 1;
        }
    }

    if (do_bench) {
        std::vector<BenchRow> rows_d, rows_df;
        if (run_f) {
            const int lt = (log_total_override > 0) ? log_total_override : FloatTraits::kDefaultLogTotal;
            run_benchmark_prec<FloatTraits>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
        }
        if (run_d && dev_cfg.has_fp64) {
            const int lt = (log_total_override > 0) ? log_total_override : DoubleTraits::kDefaultLogTotal;
            rows_d = run_benchmark_prec<DoubleTraits>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
        }
        if (run_df) {
            const int lt = (log_total_override > 0) ? log_total_override : DfTraits::kDefaultLogTotal;
            rows_df = run_benchmark_prec<DfTraits>(q, lt, min_logN, max_logN, cfg, ntest, in_place);
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
