#define _POSIX_C_SOURCE 200809L

#include "common/vk_context.h"
#include "common/math.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct DslEntry {
    uint32_t              num_bindings;
    VkDescriptorSetLayout layout;
} DslEntry;

typedef struct PlEntry {
    uint64_t         key;
    VkPipelineLayout layout;
} PlEntry;

typedef struct VkPipeKey {
    uint32_t shader_id;
    uint32_t num_bindings;
    uint32_t push_const_bytes;
    uint32_t required_subgroup_size;
    uint32_t num_specs;
    VkSpecConstEntry specs[16];
} VkPipeKey;

typedef struct PipeCacheEntry {
    uint64_t            hash;
    VkPipeKey           key;
    VkCompiledPipeline* pipe;
    bool                occupied;
} PipeCacheEntry;

static uint64_t pipe_key_hash(const VkPipeKey* k) {
    uint64_t h = 14695981039346656037ull;
    const uint32_t head[5] = {
        k->shader_id, k->num_bindings, k->push_const_bytes,
        k->required_subgroup_size, k->num_specs
    };
    for (size_t i = 0; i < 5; ++i) {
        h ^= (uint64_t)head[i];
        h *= 1099511628211ull;
    }
    for (uint32_t i = 0; i < k->num_specs; ++i) {
        uint64_t pair = ((uint64_t)k->specs[i].constant_id << 32) | k->specs[i].value;
        h ^= pair;
        h *= 1099511628211ull;
    }
    return h ? h : 1ull;
}

static bool pipe_key_eq(const VkPipeKey* a, const VkPipeKey* b) {
    if (a->shader_id != b->shader_id ||
        a->num_bindings != b->num_bindings ||
        a->push_const_bytes != b->push_const_bytes ||
        a->required_subgroup_size != b->required_subgroup_size ||
        a->num_specs != b->num_specs) {
        return false;
    }
    for (uint32_t i = 0; i < a->num_specs; ++i) {
        if (a->specs[i].constant_id != b->specs[i].constant_id ||
            a->specs[i].value != b->specs[i].value) {
            return false;
        }
    }
    return true;
}

struct VkDeviceContext {
    int                              device_id;
    VkDeviceDescriptor               desc;
    VkDeviceFftConfig                cfg;
    VkDevice                         device;
    PFN_vkCmdPushDescriptorSetKHR    pfn_push_desc;
    VkPipelineCache                  pipeline_cache;
    char                             pipeline_cache_path[512];
    VkPhysicalDeviceMemoryProperties mem_props;

    VkQueue         queues[8];
    pthread_mutex_t queue_mtxs[8];
    uint32_t        num_queues;

    pthread_rwlock_t pipe_rwlock;
    VkShaderModule   modules[VK_SHADER_COUNT];
    DslEntry         dsls[24];
    size_t           num_dsls;
    PlEntry          pls[64];
    size_t           num_pls;
    PipeCacheEntry*  pipe_buckets; /* open-addressed power-of-2 hash table */
    size_t           pipe_mask;
    size_t           num_pipes;

    VkBufferAlloc dummy_buf;

    pthread_mutex_t     probe_mtx;
    VkDeviceProbeResult probe_res;
};

static bool has_dev_ext(VkPhysicalDevice pdev, const char* name) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n, nullptr) != VK_SUCCESS || n == 0) {
        return false;
    }
    VkExtensionProperties* v = (VkExtensionProperties*)malloc(n * sizeof(*v));
    if (!v) return false;
    bool ok = false;
    if (vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n, v) == VK_SUCCESS) {
        for (uint32_t i = 0; i < n; ++i) {
            if (strcmp(v[i].extensionName, name) == 0) { ok = true; break; }
        }
    }
    free(v);
    return ok;
}

static void mkdir_p_parent_ctx(const char* filepath) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", filepath);
    char* slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) return;
    *slash = '\0';
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

static void resolve_pipe_cache_path(const VkDeviceFftConfig* cfg, char* out, size_t out_cap) {
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    char dir[384];
    if (xdg && xdg[0] != '\0') snprintf(dir, sizeof(dir), "%s/toeplitz_ls", xdg);
    else if (home && home[0] != '\0') snprintf(dir, sizeof(dir), "%s/.cache/toeplitz_ls", home);
    else snprintf(dir, sizeof(dir), "/tmp/toeplitz_ls_vk_cache");
    snprintf(out, out_cap, "%s/vk_pipe_%x_%x_%s.bin",
             dir, cfg->vendor_id, cfg->device_id, vk_kernel_code_hash());
}

/* ============================================================================
 * VkBufferAlloc
 * ============================================================================ */
bool vk_buf_create(VkDeviceContext* ctx, size_t bytes, VkBufferMemKind kind, VkBufferAlloc* out) {
    memset(out, 0, sizeof(*out));
    if (!ctx || bytes == 0) return false;

    VkDeviceSize alloc_sz = (bytes < 16u) ? 16u : (VkDeviceSize)bytes;
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = alloc_sz;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(ctx->device, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req = {0};
    vkGetBufferMemoryRequirements(ctx->device, buf, &req);
    int mem_idx = vk_ctx_find_memory_type(ctx, req.memoryTypeBits, kind);
    if (mem_idx < 0) {
        vkDestroyBuffer(ctx->device, buf, nullptr);
        return false;
    }

    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = (uint32_t)mem_idx;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(ctx->device, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, buf, nullptr);
        return false;
    }
    if (vkBindBufferMemory(ctx->device, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, buf, nullptr);
        vkFreeMemory(ctx->device, mem, nullptr);
        return false;
    }

    void* ptr = nullptr;
    if (kind != VK_MEM_DEVICE_LOCAL) {
        if (vkMapMemory(ctx->device, mem, 0, alloc_sz, 0, &ptr) != VK_SUCCESS) {
            vkDestroyBuffer(ctx->device, buf, nullptr);
            vkFreeMemory(ctx->device, mem, nullptr);
            return false;
        }
    }

    out->ctx = ctx;
    out->buffer = buf;
    out->memory = mem;
    out->size = alloc_sz;
    out->mapped = ptr;
    out->kind = kind;
    return true;
}

void vk_buf_destroy(VkBufferAlloc* buf) {
    if (!buf) return;
    if (buf->ctx && buf->ctx->device != VK_NULL_HANDLE) {
        if (buf->mapped && buf->memory != VK_NULL_HANDLE) {
            vkUnmapMemory(buf->ctx->device, buf->memory);
        }
        if (buf->buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(buf->ctx->device, buf->buffer, nullptr);
        }
        if (buf->memory != VK_NULL_HANDLE) {
            vkFreeMemory(buf->ctx->device, buf->memory, nullptr);
        }
    }
    memset(buf, 0, sizeof(*buf));
}

VkBufferSlice vk_buf_slice(const VkBufferAlloc* buf, VkDeviceSize off, VkDeviceSize len) {
    VkBufferSlice s = {VK_NULL_HANDLE, 0, 0};
    if (!buf || buf->buffer == VK_NULL_HANDLE) return s;
    s.buffer = buf->buffer;
    s.offset = off;
    s.range = (len == VK_WHOLE_SIZE) ? (buf->size - off) : len;
    return s;
}

/* ============================================================================
 * VkCmdBatch
 * ============================================================================ */
bool vk_batch_init(VkCmdBatch* b, VkDeviceContext* ctx, uint32_t queue_index) {
    memset(b, 0, sizeof(*b));
    b->ctx = ctx;
    b->queue_index = queue_index % (ctx->num_queues > 0 ? ctx->num_queues : 1u);

    VkCommandPoolCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = ctx->cfg.compute_queue_family;
    if (vkCreateCommandPool(ctx->device, &cpci, nullptr, &b->cmd_pool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo cbai = {0};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = b->cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx->device, &cbai, &b->cmd_buf) != VK_SUCCESS) {
        vk_batch_destroy(b);
        return false;
    }

    VkFenceCreateInfo fci = {0};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(ctx->device, &fci, nullptr, &b->fence) != VK_SUCCESS) {
        vk_batch_destroy(b);
        return false;
    }

    if (!ctx->pfn_push_desc) {
        b->ds_cap = 2048;
        VkDescriptorPoolSize psz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b->ds_cap * 12u};
        VkDescriptorPoolCreateInfo dpci = {0};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = b->ds_cap;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psz;
        if (vkCreateDescriptorPool(ctx->device, &dpci, nullptr, &b->ds_pool) != VK_SUCCESS) {
            vk_batch_destroy(b);
            return false;
        }
    }
    return true;
}

void vk_batch_destroy(VkCmdBatch* b) {
    if (!b) return;
    vk_buf_destroy(&b->staging_up);
    vk_buf_destroy(&b->staging_dn);
    if (b->ctx && b->ctx->device != VK_NULL_HANDLE) {
        if (b->ds_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(b->ctx->device, b->ds_pool, nullptr);
        if (b->fence != VK_NULL_HANDLE) vkDestroyFence(b->ctx->device, b->fence, nullptr);
        if (b->cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(b->ctx->device, b->cmd_pool, nullptr);
    }
    memset(b, 0, sizeof(*b));
}

bool vk_batch_begin(VkCmdBatch* b) {
    if (vkResetCommandBuffer(b->cmd_buf, 0) != VK_SUCCESS) return false;
    if (b->ds_pool != VK_NULL_HANDLE) {
        if (vkResetDescriptorPool(b->ctx->device, b->ds_pool, 0) != VK_SUCCESS) return false;
    }
    b->ds_used = 0;
    b->last_pipeline = VK_NULL_HANDLE;

    VkCommandBufferBeginInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(b->cmd_buf, &bi) != VK_SUCCESS) return false;
    b->recording = true;
    return true;
}

void vk_batch_copy(VkCmdBatch* b, VkBufferSlice src, VkBufferSlice dst, VkDeviceSize bytes) {
    if (bytes == 0) return;
    VkBufferCopy region = {src.offset, dst.offset, bytes};
    vkCmdCopyBuffer(b->cmd_buf, src.buffer, dst.buffer, 1, &region);
}

void vk_batch_fill(VkCmdBatch* b, VkBufferSlice dst, VkDeviceSize bytes, uint32_t val) {
    if (bytes == 0) return;
    VkDeviceSize aligned = (bytes + 3u) & ~(VkDeviceSize)3u;
    vkCmdFillBuffer(b->cmd_buf, dst.buffer, dst.offset, aligned, val);
}

static void emit_mem_barrier(
    VkCommandBuffer cmd,
    VkPipelineStageFlags src_stage, VkAccessFlags src_acc,
    VkPipelineStageFlags dst_stage, VkAccessFlags dst_acc) {
    VkMemoryBarrier mb = {0};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src_acc;
    mb.dstAccessMask = dst_acc;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void vk_batch_barrier_t2c(VkCmdBatch* b) {
    emit_mem_barrier(b->cmd_buf,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void vk_batch_barrier_c2c(VkCmdBatch* b) {
    emit_mem_barrier(b->cmd_buf,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void vk_batch_barrier_tc2c(VkCmdBatch* b) {
    emit_mem_barrier(b->cmd_buf,
                     VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void vk_batch_barrier_c2t(VkCmdBatch* b) {
    emit_mem_barrier(b->cmd_buf,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

void vk_batch_barrier_t2t(VkCmdBatch* b) {
    emit_mem_barrier(b->cmd_buf,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

bool vk_batch_dispatch(
    VkCmdBatch* b,
    const VkCompiledPipeline* pipe,
    const VkBufferSlice* bindings,
    uint32_t num_bindings,
    const void* push_data,
    uint32_t push_bytes,
    uint32_t gx,
    uint32_t gy,
    uint32_t gz) {
    if (!pipe || gx == 0 || gy == 0 || gz == 0) return true;
    const bool use_push = (b->ctx->pfn_push_desc != nullptr);
    if (!use_push && b->ds_used >= b->ds_cap) {
        if (!vk_batch_submit_and_wait(b)) return false;
        if (!vk_batch_begin(b)) return false;
    }

    uint32_t nb = (num_bindings <= 16u) ? num_bindings : 16u;
    VkDescriptorBufferInfo binfos[16];
    VkWriteDescriptorSet writes[16];
    if (nb > 0) {
        VkBufferSlice dummy = vk_ctx_dummy_buffer(b->ctx);
        for (uint32_t i = 0; i < nb; ++i) {
            VkBufferSlice s = (bindings && vk_slice_valid(bindings[i])) ? bindings[i] : dummy;
            binfos[i] = (VkDescriptorBufferInfo){s.buffer, s.offset, s.range};
            writes[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = i,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &binfos[i],
                .pTexelBufferView = nullptr,
            };
        }
    }

    if (b->last_pipeline != pipe->pipeline) {
        vkCmdBindPipeline(b->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
        b->last_pipeline = pipe->pipeline;
    }

    if (nb > 0) {
        if (use_push) {
            b->ctx->pfn_push_desc(b->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipe->layout, 0, nb, writes);
        } else {
            VkDescriptorSet dset = VK_NULL_HANDLE;
            VkDescriptorSetAllocateInfo dsai = {0};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = b->ds_pool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &pipe->ds_layout;
            if (vkAllocateDescriptorSets(b->ctx->device, &dsai, &dset) != VK_SUCCESS) return false;
            ++b->ds_used;
            for (uint32_t i = 0; i < nb; ++i) writes[i].dstSet = dset;
            vkUpdateDescriptorSets(b->ctx->device, nb, writes, 0, nullptr);
            vkCmdBindDescriptorSets(b->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe->layout, 0, 1, &dset, 0, nullptr);
        }
    }
    if (push_data && push_bytes > 0) {
        vkCmdPushConstants(b->cmd_buf, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_bytes, push_data);
    }
    vkCmdDispatch(b->cmd_buf, gx, gy, gz);
    return true;
}

bool vk_batch_submit_and_wait(VkCmdBatch* b) {
    if (!b->recording) return true;
    if (vkEndCommandBuffer(b->cmd_buf) != VK_SUCCESS) return false;
    b->recording = false;
    if (vkResetFences(b->ctx->device, 1, &b->fence) != VK_SUCCESS) return false;
    if (!vk_ctx_submit_queue(b->ctx, b->queue_index, b->cmd_buf, b->fence)) return false;
    return vkWaitForFences(b->ctx->device, 1, &b->fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

static bool batch_ensure_staging(VkCmdBatch* b, size_t bytes, bool upload) {
    VkBufferAlloc* buf = upload ? &b->staging_up : &b->staging_dn;
    if (buf->buffer != VK_NULL_HANDLE && buf->size >= bytes) return true;
    vk_buf_destroy(buf);
    size_t sz = (bytes < 4096u) ? 4096u : bytes;
    return vk_buf_create(b->ctx, sz, upload ? VK_MEM_HOST_UPLOAD : VK_MEM_HOST_READBACK, buf);
}

bool vk_batch_upload_sync(VkCmdBatch* b, const void* host_src, VkBufferSlice dev_dst, size_t bytes) {
    if (bytes == 0) return true;
    if (!batch_ensure_staging(b, bytes, true)) return false;
    memcpy(b->staging_up.mapped, host_src, bytes);
    if (!vk_batch_begin(b)) return false;
    vk_batch_copy(b, vk_buf_slice(&b->staging_up, 0, bytes), dev_dst, bytes);
    return vk_batch_submit_and_wait(b);
}

bool vk_batch_download_sync(VkCmdBatch* b, VkBufferSlice dev_src, void* host_dst, size_t bytes) {
    if (bytes == 0) return true;
    if (!batch_ensure_staging(b, bytes, false)) return false;
    if (!vk_batch_begin(b)) return false;
    vk_batch_copy(b, dev_src, vk_buf_slice(&b->staging_dn, 0, bytes), bytes);
    if (!vk_batch_submit_and_wait(b)) return false;
    memcpy(host_dst, b->staging_dn.mapped, bytes);
    return true;
}

/* ============================================================================
 * VkDeviceContext Implementation
 * ============================================================================ */
static void load_pipe_cache(VkDeviceContext* ctx) {
    void* blob = nullptr;
    size_t sz = 0;
    FILE* fp = fopen(ctx->pipeline_cache_path, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        long len = ftell(fp);
        if (len > 0) {
            fseek(fp, 0, SEEK_SET);
            blob = malloc((size_t)len);
            if (blob && fread(blob, 1, (size_t)len, fp) == (size_t)len) sz = (size_t)len;
            else { free(blob); blob = nullptr; }
        }
        fclose(fp);
    }
    VkPipelineCacheCreateInfo pci = {0};
    pci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    pci.initialDataSize = sz;
    pci.pInitialData = blob;
    if (vkCreatePipelineCache(ctx->device, &pci, nullptr, &ctx->pipeline_cache) != VK_SUCCESS) {
        pci.initialDataSize = 0;
        pci.pInitialData = nullptr;
        vkCreatePipelineCache(ctx->device, &pci, nullptr, &ctx->pipeline_cache);
    }
    free(blob);
}

static void save_pipe_cache(VkDeviceContext* ctx) {
    if (ctx->pipeline_cache == VK_NULL_HANDLE || ctx->pipeline_cache_path[0] == '\0') return;
    size_t sz = 0;
    if (vkGetPipelineCacheData(ctx->device, ctx->pipeline_cache, &sz, nullptr) != VK_SUCCESS || sz == 0) return;
    void* blob = malloc(sz);
    if (!blob) return;
    if (vkGetPipelineCacheData(ctx->device, ctx->pipeline_cache, &sz, blob) == VK_SUCCESS) {
        mkdir_p_parent_ctx(ctx->pipeline_cache_path);
        char tmp[560];
        snprintf(tmp, sizeof(tmp), "%s.tmp.%ld", ctx->pipeline_cache_path, (long)getpid());
        FILE* fp = fopen(tmp, "wb");
        if (fp) {
            fwrite(blob, 1, sz, fp);
            fclose(fp);
            if (rename(tmp, ctx->pipeline_cache_path) != 0) unlink(tmp);
        }
    }
    free(blob);
}

static VkDeviceContext* create_vk_context(int device_id, char* err_out, size_t err_cap) {
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (device_id < 0 || (size_t)device_id >= ndevs) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap, "Invalid Vulkan device ID %d (available: %zu)", device_id, ndevs);
        }
        return nullptr;
    }

    VkDeviceContext* ctx = (VkDeviceContext*)calloc(1, sizeof(*ctx));
    if (!ctx) return nullptr;
    ctx->device_id = device_id;
    ctx->desc = devs[device_id];
    vk_device_fft_config_query(&ctx->desc, &ctx->cfg);
    vkGetPhysicalDeviceMemoryProperties(ctx->desc.phys_dev, &ctx->mem_props);

    uint32_t nq = (ctx->cfg.compute_queue_count < 8u) ? (uint32_t)ctx->cfg.compute_queue_count : 8u;
    if (nq == 0) nq = 1;
    float prio[8];
    for (uint32_t i = 0; i < nq; ++i) prio[i] = 1.0f;

    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = ctx->cfg.compute_queue_family;
    qci.queueCount = nq;
    qci.pQueuePriorities = prio;

    const char* exts[8];
    uint32_t nexts = 0;

    VkPhysicalDeviceFeatures2 f2 = {0};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (ctx->cfg.has_fp64) f2.features.shaderFloat64 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12 = {0};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8 = {0};
    f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    VkPhysicalDevice16BitStorageFeatures f16s = {0};
    f16s.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceSubgroupSizeControlFeatures sg_ctrl = {0};
    sg_ctrl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;

    void** tail = &f2.pNext;
    if (ctx->cfg.has_fp16) {
        if (ctx->cfg.api_version >= VK_API_VERSION_1_2) {
            f12.shaderFloat16 = VK_TRUE;
            *tail = &f12; tail = &f12.pNext;
            f16s.storageBuffer16BitAccess = VK_TRUE;
            *tail = &f16s; tail = &f16s.pNext;
        } else {
            exts[nexts++] = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
            exts[nexts++] = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;
            f16i8.shaderFloat16 = VK_TRUE;
            f16s.storageBuffer16BitAccess = VK_TRUE;
            *tail = &f16i8; tail = &f16i8.pNext;
            *tail = &f16s; tail = &f16s.pNext;
        }
    }
    if (ctx->cfg.has_subgroup_size_control) {
        if (ctx->cfg.api_version < VK_API_VERSION_1_3 &&
            has_dev_ext(ctx->desc.phys_dev, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
            exts[nexts++] = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
        }
        sg_ctrl.subgroupSizeControl = VK_TRUE;
        sg_ctrl.computeFullSubgroups = VK_TRUE;
        *tail = &sg_ctrl; tail = &sg_ctrl.pNext;
    }
    if (ctx->cfg.has_push_descriptor) {
        exts[nexts++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    }

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nexts;
    dci.ppEnabledExtensionNames = (nexts > 0) ? exts : nullptr;

    VkResult r = vkCreateDevice(ctx->desc.phys_dev, &dci, nullptr, &ctx->device);
    if (r != VK_SUCCESS) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap, "vkCreateDevice failed on '%s' (VkResult=%d)", ctx->desc.name, (int)r);
        }
        free(ctx);
        return nullptr;
    }

    if (ctx->cfg.has_push_descriptor) {
        ctx->pfn_push_desc = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(
            ctx->device, "vkCmdPushDescriptorSetKHR");
    }

    ctx->num_queues = nq;
    for (uint32_t i = 0; i < nq; ++i) {
        vkGetDeviceQueue(ctx->device, ctx->cfg.compute_queue_family, i, &ctx->queues[i]);
        pthread_mutex_init(&ctx->queue_mtxs[i], nullptr);
    }
    pthread_rwlock_init(&ctx->pipe_rwlock, nullptr);
    pthread_mutex_init(&ctx->probe_mtx, nullptr);
    ctx->pipe_buckets = (PipeCacheEntry*)calloc(256u, sizeof(PipeCacheEntry));
    ctx->pipe_mask = ctx->pipe_buckets ? 255u : 0u;

    resolve_pipe_cache_path(&ctx->cfg, ctx->pipeline_cache_path, sizeof(ctx->pipeline_cache_path));
    load_pipe_cache(ctx);
    vk_buf_create(ctx, 64, VK_MEM_DEVICE_LOCAL, &ctx->dummy_buf);
    return ctx;
}

static pthread_mutex_t  g_ctx_mtx = PTHREAD_MUTEX_INITIALIZER;
static VkDeviceContext* g_ctxs[32] = {nullptr};

VkDeviceContext* vk_ctx_get_shared(int device_id, char* err_out, size_t err_cap) {
    if (device_id < 0 || device_id >= 32) {
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap, "Invalid Vulkan device ID %d", device_id);
        }
        return nullptr;
    }
    pthread_mutex_lock(&g_ctx_mtx);
    VkDeviceContext* ctx = g_ctxs[device_id];
    if (!ctx) {
        ctx = create_vk_context(device_id, err_out, err_cap);
        g_ctxs[device_id] = ctx;
    }
    pthread_mutex_unlock(&g_ctx_mtx);
    return ctx;
}

const VkDeviceDescriptor* vk_ctx_desc(const VkDeviceContext* ctx) { return &ctx->desc; }
const VkDeviceFftConfig*  vk_ctx_config(const VkDeviceContext* ctx) { return &ctx->cfg; }
VkDevice                  vk_ctx_device(const VkDeviceContext* ctx) { return ctx->device; }
uint32_t                  vk_ctx_queue_family(const VkDeviceContext* ctx) { return ctx->cfg.compute_queue_family; }
uint32_t                  vk_ctx_queue_count(const VkDeviceContext* ctx) { return ctx->num_queues; }
VkBufferSlice             vk_ctx_dummy_buffer(const VkDeviceContext* ctx) { return vk_buf_slice(&ctx->dummy_buf, 0, VK_WHOLE_SIZE); }

int vk_ctx_find_memory_type(const VkDeviceContext* ctx, uint32_t type_bits, VkBufferMemKind kind) {
    const VkPhysicalDeviceMemoryProperties* mp = &ctx->mem_props;
    VkMemoryPropertyFlags req_list[3];
    VkMemoryPropertyFlags avoid_list[3];
    int n_tries = 0;

    if (kind == VK_MEM_DEVICE_LOCAL) {
        if (ctx->cfg.is_discrete_gpu) {
            req_list[n_tries] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            avoid_list[n_tries++] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        }
        req_list[n_tries] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        avoid_list[n_tries++] = 0;
        req_list[n_tries] = 0;
        avoid_list[n_tries++] = 0;
    } else if (kind == VK_MEM_UMA_UPLOAD) {
        req_list[n_tries] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = 0;
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = 0;
    } else if (kind == VK_MEM_UMA_READBACK) {
        req_list[n_tries] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        avoid_list[n_tries++] = 0;
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        avoid_list[n_tries++] = 0;
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = 0;
    } else if (kind == VK_MEM_HOST_READBACK) {
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        avoid_list[n_tries++] = 0;
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = 0;
    } else {
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        req_list[n_tries] = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        avoid_list[n_tries++] = 0;
    }

    for (int t = 0; t < n_tries; ++t) {
        for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
            if (!(type_bits & (1u << i))) continue;
            VkMemoryPropertyFlags f = mp->memoryTypes[i].propertyFlags;
            if ((f & req_list[t]) == req_list[t] && (f & avoid_list[t]) == 0) {
                return (int)i;
            }
        }
    }
    return -1;
}

bool vk_ctx_submit_queue(VkDeviceContext* ctx, uint32_t queue_idx, VkCommandBuffer cmd, VkFence fence) {
    uint32_t qi = queue_idx % ctx->num_queues;
    pthread_mutex_lock(&ctx->queue_mtxs[qi]);
    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkResult r = vkQueueSubmit(ctx->queues[qi], 1, &si, fence);
    pthread_mutex_unlock(&ctx->queue_mtxs[qi]);
    return r == VK_SUCCESS;
}

static VkShaderModule get_or_create_mod_unlocked(VkDeviceContext* ctx, VkShaderId id) {
    if ((unsigned)id >= (unsigned)VK_SHADER_COUNT) return VK_NULL_HANDLE;
    if (ctx->modules[id] != VK_NULL_HANDLE) return ctx->modules[id];
    const VkEmbeddedSpirv* spv = vk_get_embedded_spirv(id);
    if (!spv) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv->word_count * sizeof(uint32_t);
    ci.pCode = spv->words;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx->device, &ci, nullptr, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    ctx->modules[id] = mod;
    return mod;
}

static VkDescriptorSetLayout get_or_create_dsl_unlocked(VkDeviceContext* ctx, uint32_t nb) {
    for (size_t i = 0; i < ctx->num_dsls; ++i) {
        if (ctx->dsls[i].num_bindings == nb) return ctx->dsls[i].layout;
    }
    VkDescriptorSetLayoutBinding binds[16];
    uint32_t n = (nb <= 16u) ? nb : 16u;
    for (uint32_t i = 0; i < n; ++i) {
        memset(&binds[i], 0, sizeof(binds[i]));
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    if (ctx->pfn_push_desc) {
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }
    ci.bindingCount = n;
    ci.pBindings = (n > 0) ? binds : nullptr;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(ctx->device, &ci, nullptr, &dsl) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (ctx->num_dsls < 24) {
        ctx->dsls[ctx->num_dsls++] = (DslEntry){nb, dsl};
    }
    return dsl;
}

static VkPipelineLayout get_or_create_pl_unlocked(VkDeviceContext* ctx, uint32_t nb, uint32_t pcb) {
    uint64_t key = ((uint64_t)nb << 32) | pcb;
    for (size_t i = 0; i < ctx->num_pls; ++i) {
        if (ctx->pls[i].key == key) return ctx->pls[i].layout;
    }
    VkDescriptorSetLayout dsl = get_or_create_dsl_unlocked(ctx, nb);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcb};
    VkPipelineLayoutCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = (nb > 0) ? 1u : 0u;
    ci.pSetLayouts = (nb > 0) ? &dsl : nullptr;
    ci.pushConstantRangeCount = (pcb > 0) ? 1u : 0u;
    ci.pPushConstantRanges = (pcb > 0) ? &pcr : nullptr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(ctx->device, &ci, nullptr, &pl) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (ctx->num_pls < 64) {
        ctx->pls[ctx->num_pls++] = (PlEntry){key, pl};
    }
    return pl;
}

static const VkCompiledPipeline* pipe_table_find(const VkDeviceContext* ctx, uint64_t h, const VkPipeKey* k) {
    if (!ctx->pipe_buckets) return nullptr;
    size_t idx = (size_t)h & ctx->pipe_mask;
    for (;;) {
        const PipeCacheEntry* e = &ctx->pipe_buckets[idx];
        if (!e->occupied) return nullptr;
        if (e->hash == h && pipe_key_eq(&e->key, k)) return e->pipe;
        idx = (idx + 1u) & ctx->pipe_mask;
    }
}

static bool pipe_table_grow_unlocked(VkDeviceContext* ctx) {
    size_t old_cap = ctx->pipe_buckets ? (ctx->pipe_mask + 1u) : 0u;
    size_t new_cap = (old_cap == 0u) ? 256u : (old_cap * 2u);
    PipeCacheEntry* nb = (PipeCacheEntry*)calloc(new_cap, sizeof(PipeCacheEntry));
    if (!nb) return false;
    size_t nmask = new_cap - 1u;
    for (size_t i = 0; i < old_cap; ++i) {
        if (ctx->pipe_buckets[i].occupied) {
            size_t idx = (size_t)ctx->pipe_buckets[i].hash & nmask;
            while (nb[idx].occupied) idx = (idx + 1u) & nmask;
            nb[idx] = ctx->pipe_buckets[i];
        }
    }
    free(ctx->pipe_buckets);
    ctx->pipe_buckets = nb;
    ctx->pipe_mask = nmask;
    return true;
}

const VkCompiledPipeline* vk_ctx_get_pipeline(
    VkDeviceContext* ctx,
    VkShaderId shader_id,
    uint32_t num_bindings,
    uint32_t push_const_bytes,
    const VkSpecConstEntry* specs,
    size_t num_specs,
    uint32_t required_subgroup_size) {
    VkPipeKey key = {0};
    key.shader_id = (uint32_t)shader_id;
    key.num_bindings = num_bindings;
    key.push_const_bytes = push_const_bytes;
    key.required_subgroup_size = required_subgroup_size;
    size_t ns = (num_specs <= 16u) ? num_specs : 16u;
    key.num_specs = (uint32_t)ns;
    for (size_t i = 0; i < ns; ++i) key.specs[i] = specs[i];
    uint64_t h = pipe_key_hash(&key);

    /* Fast path: concurrent O(1) read-locked lookup (zero snprintf/strcmp/mutex contention) */
    pthread_rwlock_rdlock(&ctx->pipe_rwlock);
    const VkCompiledPipeline* hit = pipe_table_find(ctx, h, &key);
    pthread_rwlock_unlock(&ctx->pipe_rwlock);
    if (hit) return hit;

    /* Slow path: compile pipeline once under write lock */
    pthread_rwlock_wrlock(&ctx->pipe_rwlock);
    hit = pipe_table_find(ctx, h, &key);
    if (hit) {
        pthread_rwlock_unlock(&ctx->pipe_rwlock);
        return hit;
    }

    VkShaderModule mod = get_or_create_mod_unlocked(ctx, shader_id);
    VkDescriptorSetLayout dsl = get_or_create_dsl_unlocked(ctx, num_bindings);
    VkPipelineLayout pl = get_or_create_pl_unlocked(ctx, num_bindings, push_const_bytes);
    if (mod == VK_NULL_HANDLE || pl == VK_NULL_HANDLE) {
        pthread_rwlock_unlock(&ctx->pipe_rwlock);
        return nullptr;
    }

    VkSpecializationMapEntry entries[16];
    uint32_t vals[16];
    for (size_t i = 0; i < ns; ++i) {
        entries[i].constantID = key.specs[i].constant_id;
        entries[i].offset = (uint32_t)(i * sizeof(uint32_t));
        entries[i].size = sizeof(uint32_t);
        vals[i] = key.specs[i].value;
    }
    VkSpecializationInfo spec_info = {0};
    spec_info.mapEntryCount = (uint32_t)ns;
    spec_info.pMapEntries = (ns > 0) ? entries : nullptr;
    spec_info.dataSize = ns * sizeof(uint32_t);
    spec_info.pData = (ns > 0) ? vals : nullptr;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req_sg = {0};
    req_sg.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    req_sg.requiredSubgroupSize = required_subgroup_size;

    VkPipelineShaderStageCreateInfo stage = {0};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = mod;
    stage.pName = "main";
    stage.pSpecializationInfo = (ns > 0) ? &spec_info : nullptr;
    if (required_subgroup_size > 0 && ctx->cfg.has_subgroup_size_control &&
        vk_device_supports_sg_size(&ctx->cfg, required_subgroup_size)) {
        stage.pNext = &req_sg;
        stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }

    VkComputePipelineCreateInfo cpci = {0};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = pl;

    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(ctx->device, ctx->pipeline_cache, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) {
        pthread_rwlock_unlock(&ctx->pipe_rwlock);
        return nullptr;
    }

    VkCompiledPipeline* cp = (VkCompiledPipeline*)malloc(sizeof(VkCompiledPipeline));
    if (!cp) {
        vkDestroyPipeline(ctx->device, pipe, nullptr);
        pthread_rwlock_unlock(&ctx->pipe_rwlock);
        return nullptr;
    }
    *cp = (VkCompiledPipeline){pipe, pl, dsl, num_bindings, push_const_bytes};

    size_t cap = ctx->pipe_buckets ? (ctx->pipe_mask + 1u) : 0u;
    if (cap == 0u || (ctx->num_pipes + 1u) * 2u > cap) {
        if (!pipe_table_grow_unlocked(ctx)) {
            free(cp);
            vkDestroyPipeline(ctx->device, pipe, nullptr);
            pthread_rwlock_unlock(&ctx->pipe_rwlock);
            return nullptr;
        }
    }

    size_t idx = (size_t)h & ctx->pipe_mask;
    while (ctx->pipe_buckets[idx].occupied) idx = (idx + 1u) & ctx->pipe_mask;
    PipeCacheEntry* slot = &ctx->pipe_buckets[idx];
    slot->hash = h;
    slot->key = key;
    slot->pipe = cp;
    slot->occupied = true;
    ctx->num_pipes++;
    bool should_save = ((ctx->num_pipes % 16u) == 1u);
    const VkCompiledPipeline* out = cp;

    pthread_rwlock_unlock(&ctx->pipe_rwlock);
    if (should_save) save_pipe_cache(ctx);
    return out;
}

/* ============================================================================
 * Hardware Probes
 * ============================================================================ */
VkDeviceProbeResult vk_ctx_probe_subgroup(VkDeviceContext* ctx) {
    pthread_mutex_lock(&ctx->probe_mtx);
    if (ctx->probe_res.sg_probed) {
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }
    ctx->probe_res.sg_probed = true;
    ctx->probe_res.raw_sg_size = ctx->cfg.sub_group_size;
    ctx->probe_res.effective_sg_size = 1;

    if (ctx->cfg.has_subgroup_shuffle) {
        size_t wg = ctx->cfg.sub_group_size > 64u ? ctx->cfg.sub_group_size : 64u;
        if (wg > ctx->cfg.max_work_group_size) wg = ctx->cfg.max_work_group_size;
        uint32_t probe_wg = (uint32_t)wg;
        VkBufferAlloc out_buf = {0};
        VkCmdBatch batch = {0};
        if (vk_buf_create(ctx, 16, VK_MEM_DEVICE_LOCAL, &out_buf) && vk_batch_init(&batch, ctx, 0)) {
            uint32_t init_vals[4] = {0u, 1u, 1u, 0u};
            VkSpecConstEntry sp = {0u, probe_wg};
            const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(
                ctx, VK_SHADER_probe_subgroup, 1, sizeof(uint32_t), &sp, 1, 0);
            if (pipe && vk_batch_upload_sync(&batch, init_vals, vk_buf_slice(&out_buf, 0, 16), 16)) {
                VkBufferSlice binds[1] = {vk_buf_slice(&out_buf, 0, 16)};
                if (vk_batch_begin(&batch) &&
                    vk_batch_dispatch(&batch, pipe, binds, 1, &probe_wg, sizeof(uint32_t), 1, 1, 1) &&
                    vk_batch_submit_and_wait(&batch)) {
                    uint32_t res_vals[4] = {0};
                    if (vk_batch_download_sync(&batch, vk_buf_slice(&out_buf, 0, 16), res_vals, 16)) {
                        ctx->probe_res.raw_sg_size = res_vals[0];
                        ctx->probe_res.sg_linear_ok = (res_vals[1] == 1u);
                        ctx->probe_res.sg_xor_ok = (res_vals[2] == 1u);
                        if (ctx->probe_res.sg_linear_ok && ctx->probe_res.sg_xor_ok && res_vals[0] >= 4u) {
                            ctx->probe_res.effective_sg_size = res_vals[0];
                        }
                    }
                }
            }
        }
        vk_batch_destroy(&batch);
        vk_buf_destroy(&out_buf);
    }

    VkDeviceProbeResult r = ctx->probe_res;
    pthread_mutex_unlock(&ctx->probe_mtx);
    return r;
}

VkDeviceProbeResult vk_ctx_probe_df_kat(VkDeviceContext* ctx) {
    pthread_mutex_lock(&ctx->probe_mtx);
    if (ctx->probe_res.df_kat_probed) {
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }
    ctx->probe_res.df_kat_probed = true;

    double x_ang = 1.0 / 7.0;
    VkDf ang_df = vk_df_from_double(x_ang);
    float h_in[10] = {
        1.0f, 3.0f * ldexpf(1.0f, -24),
        1.5f, 3.0f * ldexpf(1.0f, -24),
        1.0f + ldexpf(1.0f, -12), 1.0f + ldexpf(1.0f, -13),
        ldexpf(1.0f, -25), -ldexpf(1.0f, -26),
        ang_df.hi, ang_df.lo
    };

    VkBufferAlloc in_buf = {0}, out_buf = {0};
    VkCmdBatch batch = {0};
    bool ok = false;
    bool used_veltkamp = false;
    if (vk_buf_create(ctx, sizeof(h_in), VK_MEM_DEVICE_LOCAL, &in_buf) &&
        vk_buf_create(ctx, 12 * sizeof(float), VK_MEM_DEVICE_LOCAL, &out_buf) &&
        vk_batch_init(&batch, ctx, 0) &&
        vk_batch_upload_sync(&batch, h_in, vk_buf_slice(&in_buf, 0, sizeof(h_in)), sizeof(h_in))) {
        VkBufferSlice binds[2] = {
            vk_buf_slice(&in_buf, 0, sizeof(h_in)),
            vk_buf_slice(&out_buf, 0, 12 * sizeof(float))
        };
        for (uint32_t vmode = 0u; vmode <= 1u && !ok; ++vmode) {
            VkSpecConstEntry sp = {15u, vmode};
            const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(
                ctx, VK_SHADER_probe_df_kat, 2, 0, &sp, 1, 0);
            float h_out[12] = {0};
            if (pipe && vk_batch_begin(&batch) &&
                vk_batch_dispatch(&batch, pipe, binds, 2, nullptr, 0, 1, 1, 1) &&
                vk_batch_submit_and_wait(&batch) &&
                vk_batch_download_sync(&batch, binds[1], h_out, sizeof(h_out))) {
                double s0 = (double)h_out[0] + (double)h_out[1];
                double e0 = (double)h_in[0] + (double)h_in[1];
                double s1 = (double)h_out[2] + (double)h_out[3];
                double e1 = (double)h_in[2] - (double)h_in[3];
                double s2 = (double)h_out[4] + (double)h_out[5];
                double e2 = (double)h_in[4] * (double)h_in[5];
                double s3 = (double)h_out[6] + (double)h_out[7];
                double e3 = ((double)h_in[4] + (double)h_in[6]) * (-((double)h_in[5] + (double)h_in[7]));
                double gc = (double)h_out[8] + (double)h_out[9];
                double gs = (double)h_out[10] + (double)h_out[11];
                double ec = cos(2.0 * M_PI * x_ang);
                double es = sin(2.0 * M_PI * x_ang);

                ok = isfinite(s0) && fabs(s0 - e0) <= 1e-14 && h_out[1] != 0.0f &&
                     isfinite(s1) && fabs(s1 - e1) <= 1e-14 && h_out[3] != 0.0f &&
                     isfinite(s2) && fabs(s2 - e2) <= 1e-14 && h_out[5] != 0.0f &&
                     isfinite(s3) && fabs(s3 - e3) <= 1e-13 &&
                     isfinite(gc) && fabs(gc - ec) <= 1e-13 &&
                     isfinite(gs) && fabs(gs - es) <= 1e-13;
                if (ok) {
                    used_veltkamp = (vmode != 0u);
                } else if (vmode == 1u) {
                    snprintf(ctx->probe_res.df_kat_error, sizeof(ctx->probe_res.df_kat_error),
                             "DF KAT failed on '%.80s' (err_terms: %.3e %.3e %.3e, s3=%.12g vs %.12g)",
                             ctx->cfg.device_name, h_out[1], h_out[3], h_out[5], s3, e3);
                }
            }
        }
    }
    vk_batch_destroy(&batch);
    vk_buf_destroy(&in_buf);
    vk_buf_destroy(&out_buf);

    ctx->probe_res.df_kat_passed = ok;
    ctx->probe_res.df_use_veltkamp = used_veltkamp;
    VkDeviceProbeResult r = ctx->probe_res;
    pthread_mutex_unlock(&ctx->probe_mtx);
    return r;
}

VkDeviceProbeResult vk_ctx_probe_fp16_kat(VkDeviceContext* ctx) {
    pthread_mutex_lock(&ctx->probe_mtx);
    if (ctx->probe_res.fp16_kat_probed) {
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }
    ctx->probe_res.fp16_kat_probed = true;
    if (!ctx->cfg.has_fp16) {
        ctx->probe_res.fp16_kat_passed = false;
        snprintf(ctx->probe_res.fp16_kat_error, sizeof(ctx->probe_res.fp16_kat_error),
                 "Device lacks native FP16");
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }

    uint16_t h_in[6] = {
        vk_f32_to_f16_bits(1.5f), vk_f32_to_f16_bits(-2.25f),
        vk_f32_to_f16_bits(3.0f), vk_f32_to_f16_bits(4.0f),
        vk_f32_to_f16_bits(0.5f), vk_f32_to_f16_bits(1.25f)
    };
    VkBufferAlloc in_buf = {0}, out_buf = {0};
    VkCmdBatch batch = {0};
    bool ok = false;
    if (vk_buf_create(ctx, sizeof(h_in), VK_MEM_DEVICE_LOCAL, &in_buf) &&
        vk_buf_create(ctx, 2 * sizeof(float), VK_MEM_DEVICE_LOCAL, &out_buf) &&
        vk_batch_init(&batch, ctx, 0)) {
        const VkCompiledPipeline* pipe = vk_ctx_get_pipeline(ctx, VK_SHADER_probe_fp16_kat, 2, 0, nullptr, 0, 0);
        if (pipe && vk_batch_upload_sync(&batch, h_in, vk_buf_slice(&in_buf, 0, sizeof(h_in)), sizeof(h_in))) {
            VkBufferSlice binds[2] = {
                vk_buf_slice(&in_buf, 0, sizeof(h_in)),
                vk_buf_slice(&out_buf, 0, 2 * sizeof(float))
            };
            float h_out[2] = {0};
            if (vk_batch_begin(&batch) &&
                vk_batch_dispatch(&batch, pipe, binds, 2, nullptr, 0, 1, 1, 1) &&
                vk_batch_submit_and_wait(&batch) &&
                vk_batch_download_sync(&batch, binds[1], h_out, sizeof(h_out))) {
                ok = fabsf(h_out[0] - 5.0f) < 1e-3f && fabsf(h_out[1] - (-7.75f)) < 1e-3f;
                if (!ok) {
                    snprintf(ctx->probe_res.fp16_kat_error, sizeof(ctx->probe_res.fp16_kat_error),
                             "FP16 KAT mismatch (got %.4f, %.4f)", h_out[0], h_out[1]);
                }
            }
        }
    }
    vk_batch_destroy(&batch);
    vk_buf_destroy(&in_buf);
    vk_buf_destroy(&out_buf);

    ctx->probe_res.fp16_kat_passed = ok;
    VkDeviceProbeResult r = ctx->probe_res;
    pthread_mutex_unlock(&ctx->probe_mtx);
    return r;
}

VkDeviceProbeResult vk_ctx_probe_fp64_ratio(VkDeviceContext* ctx) {
    pthread_mutex_lock(&ctx->probe_mtx);
    if (ctx->probe_res.fp64_probed) {
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }
    ctx->probe_res.fp64_probed = true;
    ctx->probe_res.fp64_to_fp32_ratio = 1.0 / 16.0;
    ctx->probe_res.fp64_compute_heavy = true;
    if (!ctx->cfg.has_fp64) {
        VkDeviceProbeResult r = ctx->probe_res;
        pthread_mutex_unlock(&ctx->probe_mtx);
        return r;
    }

    struct { uint32_t iters; uint32_t total; } pc = {
        ctx->cfg.is_gpu ? 512u : 128u,
        ctx->cfg.is_gpu ? 65536u : 4096u
    };
    uint32_t groups = (pc.total + 255u) / 256u;
    VkBufferAlloc b32 = {0}, b64 = {0};
    VkCmdBatch batch = {0};
    if (vk_buf_create(ctx, pc.total * sizeof(float), VK_MEM_DEVICE_LOCAL, &b32) &&
        vk_buf_create(ctx, pc.total * sizeof(double), VK_MEM_DEVICE_LOCAL, &b64) &&
        vk_batch_init(&batch, ctx, 0)) {
        const VkCompiledPipeline* p32 = vk_ctx_get_pipeline(ctx, VK_SHADER_probe_fp64_ratio_f32, 1, sizeof(pc), nullptr, 0, 0);
        const VkCompiledPipeline* p64 = vk_ctx_get_pipeline(ctx, VK_SHADER_probe_fp64_ratio_f64, 1, sizeof(pc), nullptr, 0, 0);
        if (p32 && p64) {
            VkBufferSlice s32[1] = {vk_buf_slice(&b32, 0, VK_WHOLE_SIZE)};
            VkBufferSlice s64[1] = {vk_buf_slice(&b64, 0, VK_WHOLE_SIZE)};
            vk_batch_begin(&batch);
            vk_batch_dispatch(&batch, p32, s32, 1, &pc, sizeof(pc), groups, 1, 1);
            vk_batch_dispatch(&batch, p64, s64, 1, &pc, sizeof(pc), groups, 1, 1);
            vk_batch_submit_and_wait(&batch);

            double t0 = vk_now_ms();
            vk_batch_begin(&batch);
            for (int i = 0; i < 3; ++i) {
                vk_batch_dispatch(&batch, p32, s32, 1, &pc, sizeof(pc), groups, 1, 1);
                vk_batch_barrier_c2c(&batch);
            }
            vk_batch_submit_and_wait(&batch);
            double t32 = vk_now_ms() - t0;
            if (t32 < 1e-4) t32 = 1e-4;

            t0 = vk_now_ms();
            vk_batch_begin(&batch);
            for (int i = 0; i < 3; ++i) {
                vk_batch_dispatch(&batch, p64, s64, 1, &pc, sizeof(pc), groups, 1, 1);
                vk_batch_barrier_c2c(&batch);
            }
            vk_batch_submit_and_wait(&batch);
            double t64 = vk_now_ms() - t0;
            if (t64 < 1e-4) t64 = 1e-4;

            ctx->probe_res.fp64_to_fp32_ratio = t32 / t64;
            ctx->probe_res.fp64_compute_heavy = (ctx->probe_res.fp64_to_fp32_ratio < 0.25);
        }
    }
    vk_batch_destroy(&batch);
    vk_buf_destroy(&b32);
    vk_buf_destroy(&b64);

    VkDeviceProbeResult r = ctx->probe_res;
    pthread_mutex_unlock(&ctx->probe_mtx);
    return r;
}
