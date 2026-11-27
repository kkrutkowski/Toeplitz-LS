// Public C ABI entrypoints for the Toeplitz-LS SYCL offload module (tls_sycl.so).
//
// All internal C++ SYCL kernels and plans are compiled with -fvisibility=hidden.
// Only symbols explicitly annotated with TLS_SYCL_API (matching tls_sycl_* in
// tls_sycl.map) are exported from the compiled shared library for consumption
// by toeplitz_ls.offload.sycl.

#include <cstdint>

#include "periodogram/engine.hpp"

#if defined(_WIN32)
#    define TLS_SYCL_API __declspec(dllexport)
#else
#    define TLS_SYCL_API __attribute__((visibility("default")))
#endif

extern "C" {

typedef struct tls_sycl_device_info_t {
    int id;
    char name[256];
    char backend[32];
    char device_type[32];
    uint64_t max_compute_units;
    uint64_t global_mem_size;
    int is_gpu;
    int has_fp64;
    int has_fp16;
} tls_sycl_device_info_t;

static_assert(sizeof(tls_sycl_device_info_t) == sizeof(sycl_periodogram::CDeviceInfo), "C ABI device info struct layout mismatch");

TLS_SYCL_API int tls_sycl_abi_version(void) { return 2; }

TLS_SYCL_API int tls_sycl_device_count(void) { return sycl_periodogram::get_device_count(); }

TLS_SYCL_API int tls_sycl_device_info(int device_id, tls_sycl_device_info_t* out) {
    return sycl_periodogram::get_device_info(device_id, reinterpret_cast<sycl_periodogram::CDeviceInfo*>(out));
}

TLS_SYCL_API const char* tls_sycl_last_error(void) { return sycl_periodogram::get_last_error(); }

// --- FastChi2 periodograms (GLS for degree == 1, Levinson for degree > 1) ---
TLS_SYCL_API int tls_sycl_h_fastchi2(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend,
                                     int normalization, int device_id, float* power, float* cond) {
    return sycl_periodogram::run_fastchi2_half(t, y, nullptr, dy, nullptr, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_SYCL_API int tls_sycl_h_fastchi2_d(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                       int normalization, int device_id, float* power, float* cond) {
    return sycl_periodogram::run_fastchi2_half(t, nullptr, y, nullptr, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_SYCL_API int tls_sycl_f_fastchi2(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend,
                                     int normalization, int device_id, float* power, float* cond) {
    return sycl_periodogram::run_fastchi2_float(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_SYCL_API int tls_sycl_df_fastchi2(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                      int normalization, int device_id, double* power, double* cond) {
    return sycl_periodogram::run_fastchi2_df(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_SYCL_API int tls_sycl_d_fastchi2(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                     int normalization, int device_id, double* power, double* cond) {
    return sycl_periodogram::run_fastchi2_double(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

// --- IHS periodograms ---
TLS_SYCL_API int tls_sycl_h_ihs(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend,
                                int normalization, int device_id, float* power) {
    return sycl_periodogram::run_ihs_half(t, y, nullptr, dy, nullptr, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_SYCL_API int tls_sycl_h_ihs_d(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                  int normalization, int device_id, float* power) {
    return sycl_periodogram::run_ihs_half(t, nullptr, y, nullptr, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_SYCL_API int tls_sycl_f_ihs(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend,
                                int normalization, int device_id, float* power) {
    return sycl_periodogram::run_ihs_float(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_SYCL_API int tls_sycl_df_ihs(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                 int normalization, int device_id, double* power) {
    return sycl_periodogram::run_ihs_df(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_SYCL_API int tls_sycl_d_ihs(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                int normalization, int device_id, double* power) {
    return sycl_periodogram::run_ihs_double(t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

}  // extern "C"
