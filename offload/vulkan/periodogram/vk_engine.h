#ifndef TLS_VK_ENGINE_H
#define TLS_VK_ENGINE_H

/* Plain C23 Multi-Stream Vulkan Periodogram Engine (FastChi2 & IHS). */

#include <stdint.h>

typedef struct VkCDeviceInfo {
    int id;
    char name[256];
    char backend[32];
    char device_type[32];
    uint64_t max_compute_units;
    uint64_t global_mem_size;
    int is_gpu;
    int has_fp64;
    int has_fp16;
} VkCDeviceInfo;

int vk_engine_device_count(void);
int vk_engine_device_info(int device_id, VkCDeviceInfo* out);
const char* vk_engine_last_error(void);

int vk_engine_run_fastchi2_half(const double* t, const float* y_f, const double* y_d, const float* dy_f, const double* dy_d, int M, double f0, double df, int N,
                                int degree, int backend, int normalization, int device_id, float* power, float* cond);

int vk_engine_run_fastchi2_float(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend,
                                 int normalization, int device_id, float* power, float* cond);

int vk_engine_run_fastchi2_df(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                              int normalization, int device_id, double* power, double* cond);

int vk_engine_run_fastchi2_double(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend,
                                  int normalization, int device_id, double* power, double* cond);

int vk_engine_run_ihs_half(const double* t, const float* y_f, const double* y_d, const float* dy_f, const double* dy_d, int M, double f0, double df, int N,
                           int degree, int backend, int normalization, int device_id, float* power);

int vk_engine_run_ihs_float(const double* t, const float* y, const float* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                            int device_id, float* power);

int vk_engine_run_ihs_df(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                         int device_id, double* power);

int vk_engine_run_ihs_double(const double* t, const double* y, const double* dy, int M, double f0, double df, int N, int degree, int backend, int normalization,
                             int device_id, double* power);

#endif /* TLS_VK_ENGINE_H */
