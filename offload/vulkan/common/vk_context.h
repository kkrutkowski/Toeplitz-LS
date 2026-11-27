#ifndef TLS_VK_CONTEXT_H
#define TLS_VK_CONTEXT_H

/* Plain C23 Vulkan Device Context, Persistent Pipeline Cache, Buffer Allocator,
 * Batched Single-Submit Command Recorder, and Hardware Probes. */

#include <stddef.h>
#include <stdint.h>

#include "common/vk_device.h"
#include "common/vk_os.h"
#include "shaders_spv.h"

typedef enum VkBufferMemKind {
    VK_MEM_DEVICE_LOCAL = 0,
    VK_MEM_HOST_UPLOAD = 1,
    VK_MEM_HOST_READBACK = 2,
    VK_MEM_UMA_UPLOAD = 3,
    VK_MEM_UMA_READBACK = 4
} VkBufferMemKind;

typedef struct VkBufferSlice {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkBufferSlice;

static inline bool vk_slice_valid(VkBufferSlice s) { return s.buffer != VK_NULL_HANDLE && s.range > 0; }

typedef struct VkDeviceContext VkDeviceContext;

typedef struct VkBufferAlloc {
    VkDeviceContext* ctx;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    void* mapped;
    VkBufferMemKind kind;
} VkBufferAlloc;

bool vk_buf_create(VkDeviceContext* ctx, size_t bytes, VkBufferMemKind kind, VkBufferAlloc* out);
void vk_buf_destroy(VkBufferAlloc* buf);
VkBufferSlice vk_buf_slice(const VkBufferAlloc* buf, VkDeviceSize off, VkDeviceSize len);

typedef struct VkSpecConstEntry {
    uint32_t constant_id;
    uint32_t value;
} VkSpecConstEntry;

typedef struct VkCompiledPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout ds_layout;
    uint32_t num_bindings;
    uint32_t push_const_bytes;
} VkCompiledPipeline;

typedef struct VkDeviceProbeResult {
    bool sg_probed;
    size_t raw_sg_size;
    size_t effective_sg_size;
    bool sg_linear_ok;
    bool sg_xor_ok;

    bool df_kat_probed;
    bool df_kat_passed;
    bool df_use_veltkamp;
    char df_kat_error[256];

    bool fp16_kat_probed;
    bool fp16_kat_passed;
    char fp16_kat_error[256];

    bool fp64_probed;
    double fp64_to_fp32_ratio;
    bool fp64_compute_heavy;
} VkDeviceProbeResult;

typedef struct VkCmdBatch {
    VkDeviceContext* ctx;
    uint32_t queue_index;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd_buf;
    VkFence fence;
    VkDescriptorPool ds_pool;
    uint32_t ds_used;
    uint32_t ds_cap;
    VkPipeline last_pipeline;
    bool recording;
    VkBufferAlloc staging_up;
    VkBufferAlloc staging_dn;
} VkCmdBatch;

bool vk_batch_init(VkCmdBatch* b, VkDeviceContext* ctx, uint32_t queue_index);
void vk_batch_destroy(VkCmdBatch* b);
bool vk_batch_begin(VkCmdBatch* b);
void vk_batch_copy(VkCmdBatch* b, VkBufferSlice src, VkBufferSlice dst, VkDeviceSize bytes);
void vk_batch_fill(VkCmdBatch* b, VkBufferSlice dst, VkDeviceSize bytes, uint32_t val);
void vk_batch_barrier_t2c(VkCmdBatch* b);
void vk_batch_barrier_c2c(VkCmdBatch* b);
void vk_batch_barrier_tc2c(VkCmdBatch* b);
void vk_batch_barrier_c2t(VkCmdBatch* b);
void vk_batch_barrier_t2t(VkCmdBatch* b);
bool vk_batch_dispatch(VkCmdBatch* b, const VkCompiledPipeline* pipe, const VkBufferSlice* bindings, uint32_t num_bindings, const void* push_data,
                       uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz);
bool vk_batch_submit_and_wait(VkCmdBatch* b);
bool vk_batch_upload_sync(VkCmdBatch* b, const void* host_src, VkBufferSlice dev_dst, size_t bytes);
bool vk_batch_download_sync(VkCmdBatch* b, VkBufferSlice dev_src, void* host_dst, size_t bytes);

/* Device Context API */
VkDeviceContext* vk_ctx_get_shared(int device_id, char* err_out, size_t err_cap);
const VkDeviceDescriptor* vk_ctx_desc(const VkDeviceContext* ctx);
const VkDeviceFftConfig* vk_ctx_config(const VkDeviceContext* ctx);
VkDevice vk_ctx_device(const VkDeviceContext* ctx);
uint32_t vk_ctx_queue_family(const VkDeviceContext* ctx);
uint32_t vk_ctx_queue_count(const VkDeviceContext* ctx);
int vk_ctx_find_memory_type(const VkDeviceContext* ctx, uint32_t type_bits, VkBufferMemKind kind);
VkBufferSlice vk_ctx_dummy_buffer(const VkDeviceContext* ctx);
bool vk_ctx_submit_queue(VkDeviceContext* ctx, uint32_t queue_idx, VkCommandBuffer cmd, VkFence fence);

const VkCompiledPipeline* vk_ctx_get_pipeline(VkDeviceContext* ctx, VkShaderId shader_id, uint32_t num_bindings, uint32_t push_const_bytes,
                                              const VkSpecConstEntry* specs, size_t num_specs, uint32_t required_subgroup_size);

VkDeviceProbeResult vk_ctx_probe_subgroup(VkDeviceContext* ctx);
VkDeviceProbeResult vk_ctx_probe_df_kat(VkDeviceContext* ctx);
VkDeviceProbeResult vk_ctx_probe_fp16_kat(VkDeviceContext* ctx);
VkDeviceProbeResult vk_ctx_probe_fp64_ratio(VkDeviceContext* ctx);

#endif /* TLS_VK_CONTEXT_H */
