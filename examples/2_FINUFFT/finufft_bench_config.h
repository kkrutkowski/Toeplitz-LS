#ifndef FINUFFT_BENCH_CONFIG_H
#define FINUFFT_BENCH_CONFIG_H

constexpr int kBenchmarkM = 1e5;
constexpr int kBenchmarkNMin = 1 << 8;
constexpr int kBenchmarkNMax = 1 << 26;
// constexpr int kBenchmarkNMax = 1 << 23;

#ifndef MAX_TWIDDLE_REUSE
#define MAX_TWIDDLE_REUSE 16
#endif

constexpr int kMaxTwiddleReuse = MAX_TWIDDLE_REUSE;
static_assert(kMaxTwiddleReuse >= 2, "MAX_TWIDDLE_REUSE must be at least 2");
static_assert((kMaxTwiddleReuse & (kMaxTwiddleReuse - 1)) == 0,
              "MAX_TWIDDLE_REUSE must be a power of two");

#endif
