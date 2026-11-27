#pragma once

// Pure C++ host interface for the SYCL multi-stream periodogram engine.
// Intentionally free of <sycl/sycl.hpp> so impl.cpp compiles with plain CXX (0 HCF).

#include <cstddef>
#include <cstdint>

namespace sycl_periodogram {

struct CDeviceInfo {
    int id;
    char name[256];
    char backend[32];
    char device_type[32];
    uint64_t max_compute_units;
    uint64_t global_mem_size;
    int is_gpu;
    int has_fp64;
    int has_fp16;
};

int get_device_count() noexcept;
int get_device_info(int device_id, CDeviceInfo* out) noexcept;
const char* get_last_error() noexcept;

// FastChi2 entrypoints (GLS for degree == 1, Levinson for degree > 1)
int run_fastchi2_half(const double* t, const float* y_f, const double* y_d, const float* dy_f, const double* dy_d, int M, double f0, double df, int N,
                      int degree, int backend, int normalization, int device_id, float* power, float* cond) noexcept;

int run_fastchi2_float(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                       int device_id, float* power, float* cond) noexcept;

int run_fastchi2_df(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                    int device_id, double* power, double* cond) noexcept;

int run_fastchi2_double(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                        int device_id, double* power, double* cond) noexcept;

// IHS entrypoints
int run_ihs_half(const double* t, const float* y_f, const double* y_d, const float* dy_f, const double* dy_d, int M, double f0, double df, int N, int degree,
                 int backend, int normalization, int device_id, float* power) noexcept;

int run_ihs_float(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                  int device_id, float* power) noexcept;

int run_ihs_df(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
               int device_id, double* power) noexcept;

int run_ihs_double(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                   int device_id, double* power) noexcept;

}  // namespace sycl_periodogram
