#ifndef TLS_VK_DEVICE_H
#define TLS_VK_DEVICE_H

/* Plain C23 Vulkan 1.1+ Device Discovery, Capability Querying, Persistent
 * AutotuneCache, and Mirrored Interleaved Micro-Benchmarker. */

#include <stddef.h>
#include <stdint.h>

#include "common/layout.h"
#include "common/vk_loader.h"

typedef enum VkAutotuneMode { VK_AUTOTUNE_AUTO = 0, VK_AUTOTUNE_OFF = 1, VK_AUTOTUNE_FORCE = 2 } VkAutotuneMode;

typedef enum VkPlanExecMode { VK_PLAN_SINGLE_PASS = 0, VK_PLAN_TWO_PASS = 1, VK_PLAN_MULTISTAGE_GLOBAL = 2 } VkPlanExecMode;

typedef enum VkPrecKind { VK_PREC_HALF = 0, VK_PREC_FLOAT = 1, VK_PREC_DF = 2, VK_PREC_DOUBLE = 3 } VkPrecKind;

static inline const char* vk_prec_tag(VkPrecKind p) {
    switch (p) {
        case VK_PREC_HALF:
            return "h";
        case VK_PREC_FLOAT:
            return "f";
        case VK_PREC_DF:
            return "df";
        case VK_PREC_DOUBLE:
            return "d";
    }
    return "f";
}

void vk_sanitize_cache_token(const char* raw, char* out, size_t out_cap);
const char* vk_kernel_code_hash(void);
VkInstance vk_get_instance(void);
double vk_now_ms(void);

typedef struct VkDeviceDescriptor {
    int id;
    char name[256];
    char backend[32];
    char device_type[32];
    size_t max_compute_units;
    size_t global_mem_size;
    bool is_gpu;
    bool is_discrete_gpu;
    bool has_fp64;
    bool has_fp16;
    uint32_t api_version;
    uint32_t vendor_id;
    uint32_t device_id;
    char driver_version[64];
    VkPhysicalDevice phys_dev;
} VkDeviceDescriptor;

/* Enumerates Vulkan physical devices sorted:
 *   Discrete GPU -> Integrated GPU -> Virtual GPU -> CPU (llvmpipe)
 * tie-broken by device-local heap size descending. */
size_t vk_enumerate_devices(const VkDeviceDescriptor** out_list);

/* Returns 0 if supported, -8 if FP16 missing (recommends tlsf/ihsf),
 * -9 if FP64 missing (recommends tlsdf/ihsdf). */
int vk_validate_device_precision(const VkDeviceDescriptor* desc, VkPrecKind prec, bool is_ihs, char* err_out, size_t err_cap);

typedef struct VkDeviceFftConfig {
    uint32_t api_version;
    uint32_t vendor_id;
    uint32_t device_id;
    VkPhysicalDeviceType vk_device_type;

    size_t max_compute_units;
    size_t max_work_group_size;
    uint32_t max_wg_size_xyz[3];
    uint32_t max_wg_count_xyz[3];
    size_t sub_group_size;
    size_t min_sub_group_size;
    size_t max_sub_group_size;
    size_t sub_group_sizes[8];
    size_t num_sub_group_sizes;
    size_t local_mem_size;
    size_t l2_cache_size;
    size_t cache_line_size;
    size_t global_mem_size;
    size_t max_mem_alloc_size;
    size_t max_storage_buffer_range;
    size_t min_storage_buffer_offset_alignment;
    size_t compute_queue_count;
    uint32_t compute_queue_family;

    bool is_gpu;
    bool is_discrete_gpu;
    bool has_native_local_mem;
    bool has_host_unified_memory;
    bool has_uma_cached;
    bool has_push_descriptor;
    bool has_fp64;
    bool has_fp16;
    bool has_subgroup_basic;
    bool has_subgroup_shuffle;
    bool has_subgroup_arithmetic;
    bool has_subgroup_size_control;

    char device_name[256];
    char driver_version[64];
    char backend_tag[32];
    VkBankModelKind bank_model;
} VkDeviceFftConfig;

void vk_device_fft_config_query(const VkDeviceDescriptor* desc, VkDeviceFftConfig* out);
void vk_device_identity_key(const VkDeviceFftConfig* cfg, char* out, size_t out_cap);
bool vk_device_supports_sg_size(const VkDeviceFftConfig* cfg, size_t s);
size_t vk_device_pick_sg_size(const VkDeviceFftConfig* cfg, size_t min_req);
size_t vk_device_clamp_pow2_wg(const VkDeviceFftConfig* cfg, size_t total_threads, size_t req_wg, size_t hard_cap);
size_t vk_device_default_padded_wg(const VkDeviceFftConfig* cfg, size_t reg_wg_cap);
size_t vk_device_effective_l2_bytes(const VkDeviceFftConfig* cfg);
size_t vk_device_safe_max_alloc_bytes(const VkDeviceFftConfig* cfg);
size_t vk_device_recommended_stream_count(const VkDeviceFftConfig* cfg);
size_t vk_device_per_stream_mem_budget(const VkDeviceFftConfig* cfg, size_t num_streams);
size_t vk_device_compute_l2_chunk_bytes(const VkDeviceFftConfig* cfg, size_t fft_bytes, size_t n1, bool is_compute_heavy, size_t l2_num, size_t l2_den,
                                        size_t min_waves_per_cu);

typedef struct VkFftLaunchConfig {
    size_t l2_chunk_bytes;
    size_t pass0_wg_size;
    size_t bitrev_wg_size;
    int two_pass_min_logN;
    int two_pass_log_n1;
    size_t l2_budget_num;
    size_t l2_budget_den;
    size_t min_waves_per_cu;
    bool allow_wide_kernels;

    VkAutotuneMode autotune;
    char cache_path[512];
    bool allow_emulated_local_mem;
    bool allow_multistage_fallback;
    bool force_multistage;
    int fp64_compute_heavy_override;
    int stash_8192_override;
    char variant_override[64];
    bool skip_df_kat;
    char bitrev_variant_override[64];
    bool skip_fp16_kat;
} VkFftLaunchConfig;

static inline VkFftLaunchConfig vk_fft_launch_config_default(void) {
    VkFftLaunchConfig c = {0};
    c.l2_budget_num = 1;
    c.l2_budget_den = 4;
    c.allow_wide_kernels = true;
    c.autotune = VK_AUTOTUNE_AUTO;
    c.allow_multistage_fallback = true;
    c.fp64_compute_heavy_override = -1;
    c.stash_8192_override = -1;
    return c;
}

static inline bool vk_fft_launch_has_plan_overrides(const VkFftLaunchConfig* c) {
    return c->l2_chunk_bytes > 0 || c->pass0_wg_size > 0 || c->two_pass_min_logN > 0 || c->two_pass_log_n1 > 0 || !c->allow_wide_kernels ||
           c->force_multistage || c->allow_emulated_local_mem || c->fp64_compute_heavy_override >= 0 || c->stash_8192_override >= 0 ||
           c->variant_override[0] != '\0';
}

static inline bool vk_fft_launch_has_bitrev_overrides(const VkFftLaunchConfig* c) {
    return c->bitrev_wg_size > 0 || c->bitrev_variant_override[0] != '\0' || c->allow_emulated_local_mem;
}

/* Persistent Disk + In-Memory AutotuneCache (v1 Vulkan) */
typedef struct VkAutotuneEntry {
    VkPlanExecMode mode;
    char variant_id[64];
    int log_n1;
    size_t pass0_wg;
    size_t l2_chunk_bytes;
    double margin_pct;
    char runner_up_id[64];
} VkAutotuneEntry;

void vk_autotune_resolve_cache_path(const char* explicit_path, char* out, size_t out_cap);
void vk_autotune_make_key(const VkDeviceFftConfig* cfg, const char* prec_tag_str, int logN, const char* slot_tag, int sign, char* out, size_t out_cap);
bool vk_autotune_cache_get(const char* path, const char* key, VkAutotuneEntry* out);
void vk_autotune_cache_put(const char* path, const char* key, const VkAutotuneEntry* val);
void vk_autotune_cache_clear_memory(void);

/* Mirrored Interleaved Micro-Benchmarker */
typedef bool (*VkBenchRunFn)(size_t cand_idx, int reps, void* user_data);
typedef bool (*VkBenchConservFn)(size_t a, size_t b, void* user_data);

typedef struct VkInterleavedBenchResult {
    bool ok;
    size_t chosen_idx;
    size_t fastest_idx;
    size_t runner_up_idx;
    bool has_runner_up;
    double chosen_ms;
    double fastest_ms;
    double runner_up_ms;
    double margin_pct;
    bool conservative_fallback;
} VkInterleavedBenchResult;

VkInterleavedBenchResult vk_run_interleaved_bench(size_t num_cands, VkBenchRunFn run_fn, VkBenchConservFn conserv_fn, void* user_data, int num_rounds,
                                                  double target_slice_ms, int max_batch_reps, double min_noise_frac, double max_noise_frac);

#endif /* TLS_VK_DEVICE_H */
