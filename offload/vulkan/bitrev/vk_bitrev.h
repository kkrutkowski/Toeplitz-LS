#ifndef TLS_VK_BITREV_H
#define TLS_VK_BITREV_H

/* Plain C23 Multi-Strategy Vulkan Bit-Reversal Engine (4B / 8B / 16B). */

#include <stddef.h>
#include <stdint.h>

#include "common/vk_context.h"

typedef enum VkBitRevLutKind {
    VK_BITREV_LUT_NONE = 0,
    VK_BITREV_LUT_PAIRS = 1,
    VK_BITREV_LUT_IDX = 2,
    VK_BITREV_LUT_COBRA_W4 = 3,
    VK_BITREV_LUT_COBRA_W5 = 4
} VkBitRevLutKind;

typedef struct VkBitRevCandidate {
    const char* variant_id;
    size_t wg_size;
    size_t smem_bytes;
    bool uses_smem;
    bool cpu_preferred;
    VkBitRevLutKind lut_kind;
    int strategy; /* 0=naive,1=lut_pairs,2=lut_idx,3=whole_smem,4=cobra_w4,5=cobra_w5,6=scalar_cobra */
    bool use_lut;
} VkBitRevCandidate;

typedef struct VkBitRevLutState {
    VkDeviceContext* ctx;
    int logN;

    VkBufferAlloc pairs_buf;
    size_t npairs;
    bool pairs_packed_u16;

    VkBufferAlloc idx_buf;
    bool idx_packed_u16;

    VkBufferAlloc rev_w[2];
    VkBufferAlloc b_pairs[2];
    size_t nb_pairs[2];
    VkBufferAlloc b_all_rev[2];
} VkBitRevLutState;

constexpr size_t VK_BITREV_MAX_CANDS = 16;

typedef struct VkBitRevEngine {
    VkDeviceContext* ctx;
    int logN;
    size_t elem_bytes;
    char prec_tag[8];
    VkFftLaunchConfig launch_cfg;
    char cache_path[512];
    int xor_k;

    VkBitRevCandidate valid_ip[VK_BITREV_MAX_CANDS];
    size_t num_ip;
    VkBitRevCandidate valid_oop[VK_BITREV_MAX_CANDS];
    size_t num_oop;

    bool resolved_ip;
    bool resolved_oop;
    int chosen_ip;
    int chosen_oop;
    bool cache_hit_ip;
    bool cache_hit_oop;

    VkBitRevLutState lut;
} VkBitRevEngine;

bool vk_bitrev_init(VkBitRevEngine* eng, VkDeviceContext* ctx, int logN, size_t elem_bytes, const char* prec_tag_str, const VkFftLaunchConfig* launch_cfg);
void vk_bitrev_destroy(VkBitRevEngine* eng);

bool vk_bitrev_resolve(VkBitRevEngine* eng, bool in_place);
bool vk_bitrev_record_in_place(VkBitRevEngine* eng, VkCmdBatch* batch, VkBufferSlice buf, size_t bs, uint32_t batch_offset);
bool vk_bitrev_record_out_of_place(VkBitRevEngine* eng, VkCmdBatch* batch, VkBufferSlice src, VkBufferSlice dst, size_t bs, uint32_t batch_offset);
bool vk_bitrev_record_candidate(VkBitRevEngine* eng, VkCmdBatch* batch, bool in_place, const char* variant_id, VkBufferSlice src, VkBufferSlice dst, size_t bs,
                                uint32_t batch_offset);

const char* vk_bitrev_variant_name(VkBitRevEngine* eng, bool in_place);
bool vk_bitrev_cache_hit(VkBitRevEngine* eng, bool in_place);

#endif /* TLS_VK_BITREV_H */
