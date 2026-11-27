#ifndef TLS_VK_NUFFT1_H
#define TLS_VK_NUFFT1_H

/* Plain C23 Quadruple-Precision Vulkan Type-1 NuFFT (NuFFT-1) Engine.
 * Supports FP16 (w in {4,5}), FP32 (w in {8,9}), Dekker DF (w in {14,15,16}),
 * and FP64 (w in {16,17,18}) across PSWF21 and PSWF43 modes, with hardware-derived
 * windowed SMEM / register-resident / direct-CPU spreading and fused DIF FFT
 * compact-slot + deconvolution epilogue. */

#include "fft/vk_dif.h"
#include "nufft/pswf_coeffs.h"

#include <stddef.h>
#include <stdint.h>

typedef enum VkNufftOutOrder {
    VK_NUFFT_ORDER_SLOT_BIT_REVERSED     = 0, /* Primary fast path: compact DIF slot order, deconvolved */
    VK_NUFFT_ORDER_NATURAL               = 1, /* Verification path: natural k in [0, N_out), deconvolved */
    VK_NUFFT_ORDER_SLOT_BIT_REVERSED_RAW = 2  /* Deferred deconv: compact DIF slot order, no 1/phi_hat */
} VkNufftOutOrder;

typedef struct VkNufft1LaunchConfig {
    VkFftLaunchConfig fft_cfg;
    char              spread_variant_override[64];
    char              deconv_variant_override[64];
    size_t            chunk_bytes_override;
    bool              use_geo_scale_fp16; /* true = geo-mean-1 FP16 table (FP16 IHS) */
} VkNufft1LaunchConfig;

static inline VkNufft1LaunchConfig vk_nufft1_launch_config_default(void) {
    VkNufft1LaunchConfig c = {0};
    c.fft_cfg = vk_fft_launch_config_default();
    return c;
}

typedef struct VkSpreadConfig {
    bool use_smem;
    bool use_reg_acc;
    bool use_cpu_direct;
    int  wg_size;
    int  t_j;
    int  b_per_wg;
    int  win_elems;
    int  num_wins;
    int  grid_stride;
    int  tile_m;
} VkSpreadConfig;

typedef struct VkDeconvConfig {
    int wg_i;
    int batch_tile;
} VkDeconvConfig;

static inline int vk_nufft1_nout(int N, VkPswfMode mode) {
    return (mode == VK_PSWF43) ? (N + (N >> 1)) : N;
}

static inline int vk_nufft1_output_shift(int N, VkPswfMode mode) {
    return (mode == VK_PSWF43) ? (3 * (N >> 2)) : (N >> 1);
}

static inline int vk_nufft1_compact_idx_to_slot(int i, VkPswfMode mode) {
    if (mode == VK_PSWF21) {
        return ((i >> 1) << 2) + ((i & 1) ? 3 : 0);
    }
    int q = i / 6;
    int r = i - 6 * q;
    int off = r + (r >= 1 ? 1 : 0) + (r >= 5 ? 1 : 0);
    return (q << 3) + off;
}

static inline int vk_nufft1_slot_to_k_prime(int slot_p, int log_nfft, int nfft, int output_shift) {
    int bin = (int)vk_bit_reverse((uint32_t)slot_p, log_nfft);
    int first_bin = nfft - output_shift;
    return (bin >= first_bin) ? (bin - nfft) : bin;
}

static inline int vk_nufft1_slot_to_natural_idx(int slot_p, int log_nfft, int nfft, int output_shift) {
    return vk_nufft1_slot_to_k_prime(slot_p, log_nfft, nfft, output_shift) + output_shift;
}

typedef struct VkNufft1Plan {
    VkDeviceContext*     ctx;
    VkPrecKind           prec;
    int                  sign;
    int                  M;
    int                  M_cap;
    int                  N;
    int                  w;
    VkPswfMode           mode;
    int                  log_nfft;
    int                  nfft;
    int                  nout;
    int                  output_shift;
    size_t               max_bs;
    size_t               chunk_bs;
    size_t               scalar_bytes;
    size_t               complex_bytes;
    size_t               real_bytes;
    size_t               lanes;
    VkNufft1LaunchConfig launch_cfg;
    char                 cache_path[512];

    VkDifFftPlan         fft_plan;

    VkSpreadConfig       spread_cfg;
    VkDeconvConfig       deconv_cfg;
    char                 spread_variant_name[64];
    char                 deconv_variant_name[64];
    bool                 spread_cache_hit;
    bool                 deconv_cache_hit;
    size_t               precompute_wg;
    double               output_scale;

    VkBufferAlloc        d_pswf_tbl;
    VkBufferAlloc        d_weights;
    VkBufferAlloc        d_slot_phi_hat;
    VkBufferAlloc        d_slot_deconv;
    VkBufferAlloc        d_nat_deconv;
    VkBufferAlloc        d_slot_deconv_f32;
    VkBufferAlloc        d_first;
    VkBufferAlloc        d_u;
    VkBufferAlloc        d_shift;
    VkBufferAlloc        d_bucket_counts;
    VkBufferAlloc        d_bucket_off;
    VkBufferAlloc        d_bucket_m;
    VkBufferAlloc        d_b_first;
    VkBufferAlloc        d_b_shift;
    VkBufferAlloc        d_b_weights;
    VkBufferAlloc        d_fft_buf;
} VkNufft1Plan;

bool vk_nufft1_plan_init(
    VkNufft1Plan* plan,
    VkDeviceContext* ctx,
    VkPrecKind prec,
    int sign,
    int M,
    int N,
    int w,
    VkPswfMode mode,
    size_t max_bs,
    const VkNufft1LaunchConfig* cfg);

void vk_nufft1_plan_destroy(VkNufft1Plan* plan);

/* Reallocate point buffers on host if M > M_cap (call BEFORE vk_batch_begin). */
bool vk_nufft1_ensure_point_capacity(VkNufft1Plan* plan, int M);

/* Record x-axis gridding, weight precomputation, and optional window bucketing into batch.
 * Assumes vk_nufft1_ensure_point_capacity(plan, M) was already satisfied. */
bool vk_nufft1_record_set_points(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    VkBufferSlice d_x,
    int M,
    double df,
    int freq_factor);

/* Record scale change & in-place rebuild of any allocated deconv LUTs into batch. */
bool vk_nufft1_record_set_output_scale(VkNufft1Plan* plan, VkCmdBatch* batch, double scale);

/* Lazy read-only device LUTs (allocated and recorded into batch on first access). */
VkBufferSlice vk_nufft1_get_slot_phi_hat(VkNufft1Plan* plan, VkCmdBatch* batch);
VkBufferSlice vk_nufft1_get_slot_deconv(VkNufft1Plan* plan, VkCmdBatch* batch);
VkBufferSlice vk_nufft1_get_nat_deconv(VkNufft1Plan* plan, VkCmdBatch* batch);
VkBufferSlice vk_nufft1_get_slot_deconv_f32(VkNufft1Plan* plan, VkCmdBatch* batch);

/* Record full Type-1 NuFFT (spread + DIF FFT + fused or standalone deconv) into batch. */
bool vk_nufft1_record_execute(
    VkNufft1Plan* plan,
    VkCmdBatch* batch,
    VkBufferSlice d_y,
    uint32_t y_offset_elems,
    VkBufferSlice d_out,
    uint32_t out_offset_elems,
    size_t bs,
    VkNufftOutOrder order);

/* Host verification helper: unpermute compact SlotBitReversed [bs][nout] -> Natural [bs][nout]. */
void vk_nufft1_unpermute_slots_to_natural_host(
    const void* h_slots,
    void* h_nat,
    size_t elem_bytes,
    int N,
    VkPswfMode mode,
    size_t bs);

#endif /* TLS_VK_NUFFT1_H */
