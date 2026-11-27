#ifndef TLS_VK_DIF_H
#define TLS_VK_DIF_H

/* Plain C23 Quadruple-Precision Vulkan DIF (Sande-Tukey) Radix-2 FFT Engine.
 * Supports logN = 10..18, Sign = +1/-1, SinglePass / TwoPass / MultiStageGlobal,
 * fused compact-slot + deconvolution epilogue, and persistent autotuning. */

#include <stddef.h>
#include <stdint.h>

#include "bitrev/vk_bitrev.h"
#include "common/vk_context.h"

typedef struct VkFftCompactParams {
    int compact_mode;            /* 0=full, 1=PSWF21 (N/2), 2=PSWF43 (3N/4) */
    int log_n1;                  /* 0 for SinglePass; log_n1 for TwoPass Pass-1 */
    VkBufferSlice d_slot_deconv; /* optional compact-slot deconv LUT */
    bool apply_deconv;
} VkFftCompactParams;

typedef struct VkDifCandidate {
    const char* variant_id;
    VkShaderId shader_id;
    int logN;
    int logE;
    int logChunks;
    int B;
    VkStagingPolicy staging;
    VkT1LayoutPolicy t1_layout;
    VkPhase3Policy phase3;
    bool stash_chunk1;

    /* Derived for active device BankModel */
    size_t wg_size;
    size_t row_stride;
    size_t stride2;
    size_t smem_elems_per_buf;
    size_t smem_bytes;
    size_t req_sg;
} VkDifCandidate;

typedef struct VkDifFftPlan {
    VkDeviceContext* ctx;
    VkPrecKind prec;
    int sign; /* +1 or -1 */
    int logN;
    size_t N;
    size_t complex_bytes;
    size_t smem_elem_bytes;
    size_t packed_batch;
    VkFftLaunchConfig launch_cfg;
    char cache_path[512];

    VkPlanExecMode exec_mode;
    VkDifCandidate sp_cand;
    char variant_name[64];
    int log_n1;
    int log_n2;
    size_t pass0_wg;
    size_t chunk_bs;
    VkBufferAlloc scratch;

    size_t effective_sg_size;
    bool df_use_veltkamp;
    double fp64_to_fp32_ratio;
    bool is_compute_heavy;
    bool cache_hit;

    bool has_bitrev;
    VkBitRevEngine bitrev;
} VkDifFftPlan;

bool vk_dif_plan_init(VkDifFftPlan* plan, VkDeviceContext* ctx, VkPrecKind prec, int sign, int logN, size_t max_bs, const VkFftLaunchConfig* cfg);

void vk_dif_plan_destroy(VkDifFftPlan* plan);

/* Record raw DIF FFT (output in bit-reversed slot order) into caller's VkCmdBatch. */
bool vk_dif_record_execute_raw(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                               uint32_t out_offset_elems, size_t chunk_bs_override);

/* Record fused compact-slot + optional deconv epilogue DIF FFT (SinglePass & TwoPass). */
bool vk_dif_record_execute_raw_compact(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                                       uint32_t out_offset_elems, int compact_mode, VkBufferSlice d_slot_deconv, bool apply_deconv, size_t chunk_bs_override);

/* Record full natural-order DIF FFT (execute_raw + bit-reversal) into caller's VkCmdBatch. */
bool vk_dif_record_execute(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice in_buf, VkBufferSlice out_buf, size_t bs, uint32_t in_offset_elems,
                           uint32_t out_offset_elems, size_t chunk_bs_override);

bool vk_dif_record_bit_reverse(VkDifFftPlan* plan, VkCmdBatch* batch, VkBufferSlice src, VkBufferSlice dst, size_t bs, uint32_t batch_offset);

#endif /* TLS_VK_DIF_H */
