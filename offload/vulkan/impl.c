/* Public C23 ABI entrypoints for Toeplitz-LS Vulkan offload module (tls_vulkan.so).
 * Compiled with -fvisibility=hidden; only tls_vk_* symbols are exported. */
#include "periodogram/vk_engine.h"

#include <stdint.h>

#if defined(_WIN32)
#  define TLS_VK_API __declspec(dllexport)
#else
#  define TLS_VK_API __attribute__((visibility("default")))
#endif

typedef struct tls_vk_device_info_t {
    int      id;
    char     name[256];
    char     backend[32];
    char     device_type[32];
    uint64_t max_compute_units;
    uint64_t global_mem_size;
    int      is_gpu;
    int      has_fp64;
    int      has_fp16;
} tls_vk_device_info_t;

static_assert(sizeof(tls_vk_device_info_t) == sizeof(VkCDeviceInfo),
              "C ABI device info struct layout mismatch");

TLS_VK_API int tls_vk_abi_version(void) {
    return 1;
}

TLS_VK_API int tls_vk_device_count(void) {
    return vk_engine_device_count();
}

TLS_VK_API int tls_vk_device_info(int device_id, tls_vk_device_info_t* out) {
    return vk_engine_device_info(device_id, (VkCDeviceInfo*)out);
}

TLS_VK_API const char* tls_vk_last_error(void) {
    return vk_engine_last_error();
}

/* --- FastChi2 periodograms --- */
TLS_VK_API int tls_vk_h_fastchi2(
    const double* t, const float* y, const float* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power, float* cond) {
    return vk_engine_run_fastchi2_half(
        t, y, nullptr, dy, nullptr, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_VK_API int tls_vk_h_fastchi2_d(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power, float* cond) {
    return vk_engine_run_fastchi2_half(
        t, nullptr, y, nullptr, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_VK_API int tls_vk_f_fastchi2(
    const double* t, const float* y, const float* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power, float* cond) {
    return vk_engine_run_fastchi2_float(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_VK_API int tls_vk_df_fastchi2(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, double* power, double* cond) {
    return vk_engine_run_fastchi2_df(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

TLS_VK_API int tls_vk_d_fastchi2(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, double* power, double* cond) {
    return vk_engine_run_fastchi2_double(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

/* --- IHS periodograms --- */
TLS_VK_API int tls_vk_h_ihs(
    const double* t, const float* y, const float* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power) {
    return vk_engine_run_ihs_half(
        t, y, nullptr, dy, nullptr, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_VK_API int tls_vk_h_ihs_d(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power) {
    return vk_engine_run_ihs_half(
        t, nullptr, y, nullptr, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_VK_API int tls_vk_f_ihs(
    const double* t, const float* y, const float* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, float* power) {
    return vk_engine_run_ihs_float(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_VK_API int tls_vk_df_ihs(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, double* power) {
    return vk_engine_run_ihs_df(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}

TLS_VK_API int tls_vk_d_ihs(
    const double* t, const double* y, const double* dy, int M,
    double f0, double df, int N, int degree, int backend, int normalization,
    int device_id, double* power) {
    return vk_engine_run_ihs_double(
        t, y, dy, M, f0, df, N, degree, backend, normalization, device_id, power);
}
