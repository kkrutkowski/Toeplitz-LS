#define _POSIX_C_SOURCE 200809L

#include "common/vk_device.h"
#include "shaders_spv.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

double vk_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

void vk_sanitize_cache_token(const char* raw, char* out, size_t out_cap) {
    if (!out || out_cap == 0) return;
    if (!raw || raw[0] == '\0') {
        snprintf(out, out_cap, "unknown");
        return;
    }
    size_t w = 0;
    for (size_t i = 0; raw[i] != '\0' && w + 1 < out_cap; ++i) {
        unsigned char c = (unsigned char)raw[i];
        if (c == '\t' || c == '\n' || c == '\r') {
            out[w++] = ' ';
        } else if (c >= 0x20 && c < 0x7f) {
            out[w++] = (char)c;
        }
    }
    while (w > 0 && out[w - 1] == ' ') --w;
    size_t lead = 0;
    while (lead < w && out[lead] == ' ') ++lead;
    if (lead > 0) {
        memmove(out, out + lead, w - lead);
        w -= lead;
    }
    out[w] = '\0';
    if (w == 0) snprintf(out, out_cap, "unknown");
}

const char* vk_kernel_code_hash(void) {
    return vk_shader_bundle_hash();
}

static pthread_once_t g_inst_once = PTHREAD_ONCE_INIT;
static VkInstance     g_instance  = VK_NULL_HANDLE;

static void create_global_instance(void) {
    uint32_t loader_ver = VK_API_VERSION_1_0;
    uint32_t api_ver = VK_API_VERSION_1_1;
    if (vkEnumerateInstanceVersion(&loader_ver) == VK_SUCCESS) {
        if (loader_ver >= VK_API_VERSION_1_3) api_ver = VK_API_VERSION_1_3;
        else if (loader_ver >= VK_API_VERSION_1_2) api_ver = VK_API_VERSION_1_2;
    }

    VkApplicationInfo app = {0};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "toeplitz_ls_vulkan";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "toeplitz_ls";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = api_ver;

    VkInstanceCreateInfo ci = {0};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    VkResult r = vkCreateInstance(&ci, nullptr, &g_instance);
    if (r != VK_SUCCESS && api_ver > VK_API_VERSION_1_1) {
        app.apiVersion = VK_API_VERSION_1_1;
        r = vkCreateInstance(&ci, nullptr, &g_instance);
    }
    if (r != VK_SUCCESS) g_instance = VK_NULL_HANDLE;
}

VkInstance vk_get_instance(void) {
    pthread_once(&g_inst_once, create_global_instance);
    return g_instance;
}

static int device_type_rank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 0;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 3;
        default:                                     return 4;
    }
}

static const char* device_type_str(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete_gpu";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated_gpu";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual_gpu";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
        default:                                     return "other";
    }
}

static const char* driver_id_tag(VkDriverId id, uint32_t vendor_id) {
    switch (id) {
        case VK_DRIVER_ID_NVIDIA_PROPRIETARY:         return "nvidia";
        case VK_DRIVER_ID_MESA_RADV:                  return "radv";
        case VK_DRIVER_ID_AMD_PROPRIETARY:            return "amdvlk";
        case VK_DRIVER_ID_AMD_OPEN_SOURCE:            return "amdvlk";
        case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:     return "anv";
        case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:  return "intel";
        case VK_DRIVER_ID_MESA_LLVMPIPE:              return "llvmpipe";
        case VK_DRIVER_ID_MOLTENVK:                   return "moltenvk";
        case VK_DRIVER_ID_MESA_NVK:                   return "nvk";
        case VK_DRIVER_ID_MESA_TURNIP:                return "turnip";
        case VK_DRIVER_ID_MESA_PANVK:                 return "panvk";
        case VK_DRIVER_ID_ARM_PROPRIETARY:            return "mali";
        case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:       return "adreno";
        default: break;
    }
    if (vendor_id == 0x10deu) return "nvidia";
    if (vendor_id == 0x1002u) return "amd";
    if (vendor_id == 0x8086u) return "intel";
    return "vk";
}

static bool env_flag_set(const char* name) {
    const char* v = getenv(name);
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
}

static bool has_device_ext(VkPhysicalDevice pdev, const char* ext_name) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n, nullptr) != VK_SUCCESS || n == 0) {
        return false;
    }
    VkExtensionProperties* exts = (VkExtensionProperties*)malloc(n * sizeof(*exts));
    if (!exts) return false;
    bool found = false;
    if (vkEnumerateDeviceExtensionProperties(pdev, nullptr, &n, exts) == VK_SUCCESS) {
        for (uint32_t i = 0; i < n; ++i) {
            if (strcmp(exts[i].extensionName, ext_name) == 0) {
                found = true;
                break;
            }
        }
    }
    free(exts);
    return found;
}

static uint64_t query_device_local_heap_bytes(VkPhysicalDevice pdev) {
    VkPhysicalDeviceMemoryProperties mem = {0};
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem);
    uint64_t dev_local_max = 0;
    uint64_t any_max = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        uint64_t sz = (uint64_t)mem.memoryHeaps[i].size;
        if (sz > any_max) any_max = sz;
        if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && sz > dev_local_max) {
            dev_local_max = sz;
        }
    }
    return (dev_local_max > 0) ? dev_local_max : any_max;
}

typedef struct RawDevEntry {
    int                rank;
    uint64_t           mem;
    size_t             orig_idx;
    VkDeviceDescriptor desc;
} RawDevEntry;

static int cmp_raw_dev(const void* a, const void* b) {
    const RawDevEntry* x = (const RawDevEntry*)a;
    const RawDevEntry* y = (const RawDevEntry*)b;
    if (x->rank != y->rank) return (x->rank < y->rank) ? -1 : 1;
    if (x->mem != y->mem) return (x->mem > y->mem) ? -1 : 1;
    return (x->orig_idx < y->orig_idx) ? -1 : 1;
}

static pthread_once_t      g_devs_once  = PTHREAD_ONCE_INIT;
static VkDeviceDescriptor* g_devs       = nullptr;
static size_t              g_dev_count  = 0;

static void enumerate_devices_once(void) {
    VkInstance inst = vk_get_instance();
    if (inst == VK_NULL_HANDLE) return;

    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(inst, &count, nullptr) != VK_SUCCESS || count == 0) return;

    VkPhysicalDevice* phys = (VkPhysicalDevice*)malloc(count * sizeof(*phys));
    RawDevEntry* raw = (RawDevEntry*)calloc(count, sizeof(*raw));
    if (!phys || !raw) {
        free(phys);
        free(raw);
        return;
    }

    if (vkEnumeratePhysicalDevices(inst, &count, phys) != VK_SUCCESS) {
        free(phys);
        free(raw);
        return;
    }

    size_t n_valid = 0;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDevice pdev = phys[i];
        VkPhysicalDeviceProperties props = {0};
        vkGetPhysicalDeviceProperties(pdev, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) continue;

        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, nullptr);
        if (qf_count == 0) continue;
        VkQueueFamilyProperties* qfs = (VkQueueFamilyProperties*)malloc(qf_count * sizeof(*qfs));
        if (!qfs) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qfs);
        uint32_t best_q = 0;
        for (uint32_t q = 0; q < qf_count; ++q) {
            if ((qfs[q].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[q].queueCount > best_q) {
                best_q = qfs[q].queueCount;
            }
        }
        free(qfs);
        if (best_q == 0) continue;

        VkPhysicalDeviceFeatures2 f2 = {0};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        VkPhysicalDeviceVulkan12Features f12 = {0};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8 = {0};
        f16i8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
        VkPhysicalDevice16BitStorageFeatures f16s = {0};
        f16s.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

        if (props.apiVersion >= VK_API_VERSION_1_2) {
            f2.pNext = &f12;
            f12.pNext = &f16s;
        } else {
            f2.pNext = &f16i8;
            f16i8.pNext = &f16s;
        }
        vkGetPhysicalDeviceFeatures2(pdev, &f2);

        bool has_fp16 = false;
        if (props.apiVersion >= VK_API_VERSION_1_2) {
            has_fp16 = (f12.shaderFloat16 == VK_TRUE) && (f16s.storageBuffer16BitAccess == VK_TRUE);
        } else {
            has_fp16 = (f16i8.shaderFloat16 == VK_TRUE) &&
                       (f16s.storageBuffer16BitAccess == VK_TRUE) &&
                       has_device_ext(pdev, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) &&
                       has_device_ext(pdev, VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        }
        bool has_fp64 = (f2.features.shaderFloat64 == VK_TRUE);

        VkPhysicalDeviceProperties2 p2 = {0};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        VkPhysicalDeviceDriverProperties dp = {0};
        dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        void** p2_tail = &p2.pNext;
        if (props.apiVersion >= VK_API_VERSION_1_2 ||
            has_device_ext(pdev, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
            *p2_tail = &dp;
            p2_tail = &dp.pNext;
        }
#ifdef VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME
        VkPhysicalDeviceShaderCorePropertiesAMD amd_core = {0};
        amd_core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
        bool has_amd_core = has_device_ext(pdev, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME);
        if (has_amd_core) {
            *p2_tail = &amd_core;
            p2_tail = &amd_core.pNext;
        }
#endif
        vkGetPhysicalDeviceProperties2(pdev, &p2);

        uint64_t mem_bytes = query_device_local_heap_bytes(pdev);
        RawDevEntry* e = &raw[n_valid];
        e->rank = device_type_rank(props.deviceType);
        e->mem = mem_bytes;
        e->orig_idx = i;

        VkDeviceDescriptor* d = &e->desc;
        vk_sanitize_cache_token(props.deviceName, d->name, sizeof(d->name));
        snprintf(d->backend, sizeof(d->backend), "vulkan/%s",
                 driver_id_tag(dp.driverID, props.vendorID));
        snprintf(d->device_type, sizeof(d->device_type), "%s", device_type_str(props.deviceType));
        size_t est_cus = best_q;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            if (ncpu > 0) est_cus = (size_t)ncpu;
        }
#ifdef VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME
        if (has_amd_core && amd_core.shaderEngineCount > 0 &&
            amd_core.shaderArraysPerEngineCount > 0 && amd_core.computeUnitsPerShaderArray > 0) {
            est_cus = (size_t)amd_core.shaderEngineCount *
                      (size_t)amd_core.shaderArraysPerEngineCount *
                      (size_t)amd_core.computeUnitsPerShaderArray;
        }
#endif
        d->max_compute_units = est_cus;
        d->global_mem_size = (size_t)mem_bytes;
        d->is_gpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                     props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
                     props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU);
        d->is_discrete_gpu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        if (env_flag_set("TLS_VK_TEST_DISABLE_FP16")) has_fp16 = false;
        if (env_flag_set("TLS_VK_TEST_DISABLE_FP64")) has_fp64 = false;
        d->has_fp64 = has_fp64;
        d->has_fp16 = has_fp16;
        d->api_version = props.apiVersion;
        d->vendor_id = props.vendorID;
        d->device_id = props.deviceID;
        snprintf(d->driver_version, sizeof(d->driver_version), "%u.%u.%u_%u",
                 VK_API_VERSION_MAJOR(props.apiVersion),
                 VK_API_VERSION_MINOR(props.apiVersion),
                 VK_API_VERSION_PATCH(props.apiVersion),
                 props.driverVersion);
        d->phys_dev = pdev;
        ++n_valid;
    }

    qsort(raw, n_valid, sizeof(RawDevEntry), cmp_raw_dev);
    if (n_valid > 0) {
        g_devs = (VkDeviceDescriptor*)calloc(n_valid, sizeof(*g_devs));
        if (g_devs) {
            for (size_t i = 0; i < n_valid; ++i) {
                g_devs[i] = raw[i].desc;
                g_devs[i].id = (int)i;
            }
            g_dev_count = n_valid;
        }
    }
    free(phys);
    free(raw);
}

size_t vk_enumerate_devices(const VkDeviceDescriptor** out_list) {
    pthread_once(&g_devs_once, enumerate_devices_once);
    if (out_list) *out_list = g_devs;
    return g_dev_count;
}

int vk_validate_device_precision(
    const VkDeviceDescriptor* desc,
    VkPrecKind prec,
    bool is_ihs,
    char* err_out,
    size_t err_cap) {
    bool fp16_ok = desc->has_fp16 && !env_flag_set("TLS_VK_TEST_DISABLE_FP16");
    bool fp64_ok = desc->has_fp64 && !env_flag_set("TLS_VK_TEST_DISABLE_FP64");
    if (prec == VK_PREC_HALF && !fp16_ok) {
        const char* alt = is_ihs ? "ihsf" : "tlsf";
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap,
                     "Vulkan device [%d] '%s' does not support native FP16 compute/storage; "
                     "switch to FP32 (%s) instead.",
                     desc->id, desc->name, alt);
        }
        return -8;
    }
    if (prec == VK_PREC_DOUBLE && !fp64_ok) {
        const char* alt = is_ihs ? "ihsdf" : "tlsdf";
        if (err_out && err_cap > 0) {
            snprintf(err_out, err_cap,
                     "Vulkan device [%d] '%s' does not support native FP64; "
                     "use Dekker double-float (%s) instead.",
                     desc->id, desc->name, alt);
        }
        return -9;
    }
    return 0;
}

void vk_device_fft_config_query(const VkDeviceDescriptor* desc, VkDeviceFftConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    VkPhysicalDevice pdev = desc->phys_dev;

    VkPhysicalDeviceProperties2 p2 = {0};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    VkPhysicalDeviceSubgroupProperties sg_props = {0};
    sg_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceMaintenance3Properties m3_props = {0};
    m3_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlProperties sg_ctrl_props = {0};
    sg_ctrl_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDevicePushDescriptorPropertiesKHR push_desc_props = {0};
    push_desc_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;

    p2.pNext = &sg_props;
    sg_props.pNext = &m3_props;
    void** ptail = &m3_props.pNext;
    bool has_sg_ctrl_ext = (desc->api_version >= VK_API_VERSION_1_3) ||
        has_device_ext(pdev, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (has_sg_ctrl_ext) { *ptail = &sg_ctrl_props; ptail = &sg_ctrl_props.pNext; }
    bool has_push_desc_ext = has_device_ext(pdev, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    if (has_push_desc_ext) { *ptail = &push_desc_props; ptail = &push_desc_props.pNext; }
    vkGetPhysicalDeviceProperties2(pdev, &p2);
    cfg->has_push_descriptor = has_push_desc_ext && (push_desc_props.maxPushDescriptors >= 16u);

    const VkPhysicalDeviceProperties* props = &p2.properties;
    cfg->api_version = props->apiVersion;
    cfg->vendor_id = props->vendorID;
    cfg->device_id = props->deviceID;
    cfg->vk_device_type = props->deviceType;
    snprintf(cfg->device_name, sizeof(cfg->device_name), "%s", desc->name);
    snprintf(cfg->driver_version, sizeof(cfg->driver_version), "%s", desc->driver_version);
    snprintf(cfg->backend_tag, sizeof(cfg->backend_tag), "%s", desc->backend);
    cfg->is_gpu = desc->is_gpu;
    cfg->is_discrete_gpu = desc->is_discrete_gpu;
    cfg->has_fp64 = desc->has_fp64;
    cfg->has_fp16 = desc->has_fp16;
    cfg->has_native_local_mem = (props->deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU);

    cfg->local_mem_size = (size_t)props->limits.maxComputeSharedMemorySize;
    cfg->max_work_group_size = props->limits.maxComputeWorkGroupInvocations > 0
        ? (size_t)props->limits.maxComputeWorkGroupInvocations : 256u;
    for (int i = 0; i < 3; ++i) {
        cfg->max_wg_size_xyz[i] = props->limits.maxComputeWorkGroupSize[i];
        cfg->max_wg_count_xyz[i] = props->limits.maxComputeWorkGroupCount[i];
    }
    cfg->max_storage_buffer_range = (size_t)props->limits.maxStorageBufferRange;
    cfg->min_storage_buffer_offset_alignment = props->limits.minStorageBufferOffsetAlignment > 16
        ? (size_t)props->limits.minStorageBufferOffsetAlignment : 16u;
    cfg->cache_line_size = 128;
    cfg->global_mem_size = desc->global_mem_size;
    cfg->max_mem_alloc_size = (m3_props.maxMemoryAllocationSize > 0)
        ? (size_t)m3_props.maxMemoryAllocationSize : cfg->global_mem_size;

    cfg->sub_group_size = sg_props.subgroupSize > 0 ? (size_t)sg_props.subgroupSize : 1u;
    cfg->has_subgroup_basic =
        (sg_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT);
    cfg->has_subgroup_shuffle = cfg->has_subgroup_basic &&
        (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    cfg->has_subgroup_arithmetic = cfg->has_subgroup_basic &&
        (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);

    VkPhysicalDeviceFeatures2 f2 = {0};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceSubgroupSizeControlFeatures sg_ctrl_feat = {0};
    sg_ctrl_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    if (has_sg_ctrl_ext) f2.pNext = &sg_ctrl_feat;
    vkGetPhysicalDeviceFeatures2(pdev, &f2);
    cfg->has_subgroup_size_control =
        has_sg_ctrl_ext &&
        (sg_ctrl_feat.subgroupSizeControl == VK_TRUE) &&
        (sg_ctrl_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);

    cfg->num_sub_group_sizes = 0;
    if (cfg->has_subgroup_size_control && sg_ctrl_props.minSubgroupSize > 0 &&
        sg_ctrl_props.maxSubgroupSize >= sg_ctrl_props.minSubgroupSize) {
        cfg->min_sub_group_size = sg_ctrl_props.minSubgroupSize;
        cfg->max_sub_group_size = sg_ctrl_props.maxSubgroupSize;
        for (size_t s = cfg->min_sub_group_size;
             s <= cfg->max_sub_group_size && cfg->num_sub_group_sizes < 8; s <<= 1) {
            cfg->sub_group_sizes[cfg->num_sub_group_sizes++] = s;
        }
    } else {
        cfg->min_sub_group_size = cfg->sub_group_size;
        cfg->max_sub_group_size = cfg->sub_group_size;
        cfg->sub_group_sizes[cfg->num_sub_group_sizes++] = cfg->sub_group_size;
    }

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, nullptr);
    VkQueueFamilyProperties* qfs = (VkQueueFamilyProperties*)malloc(qf_count * sizeof(*qfs));
    int chosen_dedicated = -1;
    int chosen_general = -1;
    if (qfs) {
        vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qfs);
        for (uint32_t i = 0; i < qf_count; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[i].queueCount > 0) {
                if (!(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    if (chosen_dedicated < 0 || qfs[i].queueCount > qfs[chosen_dedicated].queueCount) {
                        chosen_dedicated = (int)i;
                    }
                } else {
                    if (chosen_general < 0 || qfs[i].queueCount > qfs[chosen_general].queueCount) {
                        chosen_general = (int)i;
                    }
                }
            }
        }
        uint32_t chosen_qf = (chosen_dedicated >= 0) ? (uint32_t)chosen_dedicated
                                                     : (uint32_t)(chosen_general >= 0 ? chosen_general : 0);
        cfg->compute_queue_family = chosen_qf;
        cfg->compute_queue_count = (chosen_qf < qf_count && qfs[chosen_qf].queueCount > 0)
            ? (size_t)qfs[chosen_qf].queueCount : 1u;
        free(qfs);
    } else {
        cfg->compute_queue_family = 0;
        cfg->compute_queue_count = 1;
    }
    cfg->max_compute_units = desc->max_compute_units;

    VkPhysicalDeviceMemoryProperties mem = {0};
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem);
    VkMemoryPropertyFlags uma_req = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkMemoryPropertyFlags uma_cached_req = uma_req | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mem.memoryTypes[i].propertyFlags;
        uint32_t hi = mem.memoryTypes[i].heapIndex;
        if (hi < mem.memoryHeapCount && mem.memoryHeaps[hi].size >= cfg->global_mem_size / 2) {
            if ((f & uma_req) == uma_req) {
                cfg->has_host_unified_memory = true;
            }
            if ((f & uma_cached_req) == uma_cached_req) {
                cfg->has_uma_cached = true;
            }
        }
    }

    if (!cfg->has_native_local_mem || !cfg->is_gpu) {
        cfg->bank_model = VK_BANK_NONE;
    } else if (cfg->vendor_id == 0x8086u && cfg->max_work_group_size <= 256) {
        cfg->bank_model = VK_BANK_16x4;
    } else {
        cfg->bank_model = VK_BANK_32x4;
    }
}

void vk_device_identity_key(const VkDeviceFftConfig* cfg, char* out, size_t out_cap) {
    snprintf(out, out_cap, "%s|0x%x_0x%x|sg%zu_smem%zu|%s|%s",
             cfg->device_name, cfg->vendor_id, cfg->device_id,
             cfg->sub_group_size, cfg->local_mem_size,
             cfg->driver_version, cfg->backend_tag);
}

bool vk_device_supports_sg_size(const VkDeviceFftConfig* cfg, size_t s) {
    for (size_t i = 0; i < cfg->num_sub_group_sizes; ++i) {
        if (cfg->sub_group_sizes[i] == s) return true;
    }
    return false;
}

size_t vk_device_pick_sg_size(const VkDeviceFftConfig* cfg, size_t min_req) {
    for (size_t i = 0; i < cfg->num_sub_group_sizes; ++i) {
        size_t s = cfg->sub_group_sizes[i];
        if (s >= min_req && (s & (s - 1)) == 0) return s;
    }
    return 0;
}

size_t vk_device_clamp_pow2_wg(const VkDeviceFftConfig* cfg, size_t total_threads, size_t req_wg, size_t hard_cap) {
    size_t cap = (req_wg < cfg->max_work_group_size) ? req_wg : cfg->max_work_group_size;
    if (hard_cap > 0 && hard_cap < cap) cap = hard_cap;
    if (total_threads == 0) return (cap > 0) ? cap : 1u;
    if (total_threads < cap) cap = total_threads;
    size_t wg = 1;
    while ((wg << 1) <= cap) wg <<= 1;
    while (wg > 1 && (total_threads % wg) != 0) wg >>= 1;
    return wg;
}

size_t vk_device_default_padded_wg(const VkDeviceFftConfig* cfg, size_t reg_wg_cap) {
    size_t sg_hint = cfg->is_gpu ? (cfg->sub_group_size < 64 ? cfg->sub_group_size : 64u) : 8u;
    size_t pref = sg_hint * 8;
    if (pref < 64u) pref = 64u;
    size_t cap = (pref < cfg->max_work_group_size) ? pref : cfg->max_work_group_size;
    if (reg_wg_cap > 0 && reg_wg_cap < cap) cap = reg_wg_cap;
    size_t wg = 1;
    while ((wg << 1) <= cap) wg <<= 1;
    return wg;
}

size_t vk_device_effective_l2_bytes(const VkDeviceFftConfig* cfg) {
    if (cfg->l2_cache_size > 0) return cfg->l2_cache_size;
    size_t fb = cfg->is_gpu ? ((size_t)16 << 20) : ((size_t)4 << 20);
    if (cfg->global_mem_size > 0) {
        fb = cfg->global_mem_size / 256u;
        if (fb < ((size_t)2 << 20)) fb = ((size_t)2 << 20);
        if (fb > ((size_t)32 << 20)) fb = ((size_t)32 << 20);
    }
    return fb;
}

size_t vk_device_safe_max_alloc_bytes(const VkDeviceFftConfig* cfg) {
    size_t cap = SIZE_MAX;
    if (cfg->global_mem_size > 0 && cfg->global_mem_size < cap) cap = cfg->global_mem_size;
    if (cfg->max_mem_alloc_size > 0 && cfg->max_mem_alloc_size < (SIZE_MAX >> 2) &&
        cfg->max_mem_alloc_size < cap) {
        cap = cfg->max_mem_alloc_size;
    }
    if (cfg->max_storage_buffer_range > 0 && cfg->max_storage_buffer_range < cap) {
        cap = cfg->max_storage_buffer_range;
    }
    return (cap == SIZE_MAX) ? ((size_t)128 << 20) : cap;
}

size_t vk_device_recommended_stream_count(const VkDeviceFftConfig* cfg) {
    if (!cfg->is_gpu) return 1u;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    size_t host_threads = (ncpu > 2) ? (size_t)ncpu : 2u;
    size_t hw_cap = cfg->is_discrete_gpu ? ((host_threads >= 12u) ? 16u : 8u) : 4u;
    size_t streams = (cfg->compute_queue_count > 2u) ? cfg->compute_queue_count : 2u;
    if (cfg->is_discrete_gpu && host_threads >= 12u && streams < hw_cap) {
        streams = host_threads;
    }
    if (host_threads < streams) streams = host_threads;
    if (hw_cap < streams) streams = hw_cap;
    if (cfg->global_mem_size > 0) {
        size_t usable_mem = (cfg->global_mem_size / 4u) * 3u;
        size_t min_slice = vk_device_effective_l2_bytes(cfg) * 4u;
        size_t floor_slice = (streams > 8u) ? ((size_t)256 << 20) : ((size_t)64 << 20);
        if (min_slice < floor_slice) min_slice = floor_slice;
        size_t mem_cap = usable_mem / (min_slice > 0 ? min_slice : 1u);
        if (mem_cap < 1u) mem_cap = 1u;
        if (mem_cap < streams) streams = mem_cap;
    }
    return (streams > 0) ? streams : 1u;
}

size_t vk_device_per_stream_mem_budget(const VkDeviceFftConfig* cfg, size_t num_streams) {
    size_t k = (num_streams > 0) ? num_streams : 1u;
    if (cfg->global_mem_size > 0) {
        size_t total_75 = (cfg->global_mem_size / 4u) * 3u + ((cfg->global_mem_size % 4u) * 3u) / 4u;
        size_t slice = total_75 / k;
        return (slice >= ((size_t)1 << 20)) ? slice : ((size_t)1 << 20);
    }
    return (size_t)256 << 20;
}

size_t vk_device_compute_l2_chunk_bytes(
    const VkDeviceFftConfig* cfg,
    size_t fft_bytes,
    size_t n1,
    bool is_compute_heavy,
    size_t l2_num,
    size_t l2_den,
    size_t min_waves_per_cu) {
    size_t eff_l2 = vk_device_effective_l2_bytes(cfg);
    size_t den = (l2_den == 0) ? 4u : l2_den;
    size_t num = (l2_num == 0) ? 1u : l2_num;
    size_t base_budget = (eff_l2 * num) / den;
    if (base_budget < ((size_t)256 << 10)) base_budget = ((size_t)256 << 10);
    size_t max_l2_budget = (eff_l2 / 2u > base_budget) ? (eff_l2 / 2u) : base_budget;

    size_t target_waves = (min_waves_per_cu > 0)
        ? min_waves_per_cu
        : (!cfg->is_gpu ? 1u : (is_compute_heavy ? 8u : 2u));
    size_t est_cus = cfg->is_discrete_gpu ? 24u : 4u;
    size_t target_wgs = est_cus * target_waves;
    size_t occ_chunk_bs = (n1 > 0) ? ((target_wgs + n1 - 1u) / n1) : 1u;
    size_t occ_budget = occ_chunk_bs * fft_bytes;
    if (occ_budget > max_l2_budget) occ_budget = max_l2_budget;

    size_t budget = (base_budget > occ_budget) ? base_budget : occ_budget;
    size_t align = (fft_bytes > cfg->cache_line_size) ? fft_bytes : cfg->cache_line_size;
    if (align > 0) budget = ((budget + align - 1u) / align) * align;
    size_t smax = vk_device_safe_max_alloc_bytes(cfg);
    size_t cap = (smax / 4u > fft_bytes) ? (smax / 4u) : fft_bytes;
    if (budget > cap) budget = cap;
    return (budget > fft_bytes) ? budget : fft_bytes;
}

/* ============================================================================
 * Persistent AutotuneCache (plain C23 dynamic table + atomic file rename)
 * ============================================================================ */
typedef struct CacheRow {
    char            key[384];
    VkAutotuneEntry val;
} CacheRow;

typedef struct CacheFileTable {
    char      path[512];
    CacheRow* rows;
    size_t    count;
    size_t    cap;
} CacheFileTable;

static pthread_mutex_t g_cache_mtx = PTHREAD_MUTEX_INITIALIZER;
static CacheFileTable* g_tables    = nullptr;
static size_t          g_ntables   = 0;

static void mkdir_p_parent(const char* filepath) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", filepath);
    char* slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) return;
    *slash = '\0';
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static bool parse_cache_line(const char* line, char* key_out, size_t key_cap, VkAutotuneEntry* val) {
    if (!line || line[0] == '\0' || line[0] == '#') return false;
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", line);
    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';

    char* cols[8] = {nullptr};
    int nc = 0;
    char* p = buf;
    while (nc < 8) {
        cols[nc++] = p;
        char* tab = strchr(p, '\t');
        if (!tab) break;
        *tab = '\0';
        p = tab + 1;
    }
    if (nc < 6) return false;
    memset(val, 0, sizeof(*val));
    snprintf(key_out, key_cap, "%s", cols[0]);
    int m = atoi(cols[1]);
    if (m < 0) m = 0;
    if (m > 2) m = 2;
    val->mode = (VkPlanExecMode)m;
    snprintf(val->variant_id, sizeof(val->variant_id), "%s", cols[2]);
    val->log_n1 = atoi(cols[3]);
    val->pass0_wg = (size_t)strtoull(cols[4], nullptr, 10);
    val->l2_chunk_bytes = (size_t)strtoull(cols[5], nullptr, 10);
    if (nc >= 7) val->margin_pct = atof(cols[6]);
    if (nc >= 8) snprintf(val->runner_up_id, sizeof(val->runner_up_id), "%s", cols[7]);
    return true;
}

static void table_upsert(CacheFileTable* t, const char* key, const VkAutotuneEntry* val) {
    for (size_t i = 0; i < t->count; ++i) {
        if (strcmp(t->rows[i].key, key) == 0) {
            t->rows[i].val = *val;
            return;
        }
    }
    if (t->count == t->cap) {
        size_t ncap = (t->cap == 0) ? 32u : (t->cap * 2u);
        CacheRow* nr = (CacheRow*)realloc(t->rows, ncap * sizeof(*nr));
        if (!nr) return;
        t->rows = nr;
        t->cap = ncap;
    }
    snprintf(t->rows[t->count].key, sizeof(t->rows[t->count].key), "%s", key);
    t->rows[t->count].val = *val;
    ++t->count;
}

static CacheFileTable* get_or_load_table_unlocked(const char* path) {
    for (size_t i = 0; i < g_ntables; ++i) {
        if (strcmp(g_tables[i].path, path) == 0) return &g_tables[i];
    }
    CacheFileTable* nt = (CacheFileTable*)realloc(g_tables, (g_ntables + 1) * sizeof(*nt));
    if (!nt) return nullptr;
    g_tables = nt;
    CacheFileTable* t = &g_tables[g_ntables++];
    memset(t, 0, sizeof(*t));
    snprintf(t->path, sizeof(t->path), "%s", path);

    FILE* fp = fopen(path, "r");
    if (fp) {
        char line[1024];
        while (fgets(line, sizeof(line), fp)) {
            char k[384];
            VkAutotuneEntry e;
            if (parse_cache_line(line, k, sizeof(k), &e)) table_upsert(t, k, &e);
        }
        fclose(fp);
    }
    return t;
}

static void save_table_unlocked(const CacheFileTable* t) {
    mkdir_p_parent(t->path);
    char tmp[600];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%ld.%llu",
             t->path, (long)getpid(), (unsigned long long)(vk_now_ms() * 1e6));
    FILE* fp = fopen(tmp, "w");
    if (!fp) return;
    fprintf(fp, "# toeplitz_ls vulkan autotune cache v1\n");
    for (size_t i = 0; i < t->count; ++i) {
        const CacheRow* r = &t->rows[i];
        char vbuf[64], rbuf[64];
        vk_sanitize_cache_token(r->val.variant_id, vbuf, sizeof(vbuf));
        vk_sanitize_cache_token(r->val.runner_up_id, rbuf, sizeof(rbuf));
        fprintf(fp, "%s\t%d\t%s\t%d\t%zu\t%zu\t%.4f\t%s\n",
                r->key, (int)r->val.mode, vbuf, r->val.log_n1,
                r->val.pass0_wg, r->val.l2_chunk_bytes, r->val.margin_pct, rbuf);
    }
    fclose(fp);
    if (rename(tmp, t->path) != 0) unlink(tmp);
}

void vk_autotune_resolve_cache_path(const char* explicit_path, char* out, size_t out_cap) {
    if (explicit_path && explicit_path[0] != '\0') {
        snprintf(out, out_cap, "%s", explicit_path);
        return;
    }
    const char* env_p = getenv("TLS_VK_CACHE_PATH");
    if (env_p && env_p[0] != '\0') {
        snprintf(out, out_cap, "%s", env_p);
        return;
    }
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] != '\0') {
        snprintf(out, out_cap, "%s/toeplitz_ls/vulkan_autotune_v1.txt", xdg);
        return;
    }
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        snprintf(out, out_cap, "%s/.cache/toeplitz_ls/vulkan_autotune_v1.txt", home);
        return;
    }
    snprintf(out, out_cap, "/tmp/toeplitz_ls_vulkan_autotune_v1.txt");
}

void vk_autotune_make_key(
    const VkDeviceFftConfig* cfg,
    const char* prec_tag_str,
    int logN,
    const char* slot_tag,
    int sign,
    char* out,
    size_t out_cap) {
    char id_key[512];
    vk_device_identity_key(cfg, id_key, sizeof(id_key));
    snprintf(out, out_cap, "v1|%s|%.220s|%s|s%d|%s|logN=%d",
             vk_kernel_code_hash(), id_key,
             prec_tag_str ? prec_tag_str : "f", sign,
             slot_tag ? slot_tag : "any", logN);
}

bool vk_autotune_cache_get(const char* path, const char* key, VkAutotuneEntry* out) {
    pthread_mutex_lock(&g_cache_mtx);
    CacheFileTable* t = get_or_load_table_unlocked(path);
    bool found = false;
    if (t) {
        for (size_t i = 0; i < t->count; ++i) {
            if (strcmp(t->rows[i].key, key) == 0) {
                *out = t->rows[i].val;
                found = true;
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_cache_mtx);
    return found;
}

void vk_autotune_cache_put(const char* path, const char* key, const VkAutotuneEntry* val) {
    pthread_mutex_lock(&g_cache_mtx);
    CacheFileTable* t = get_or_load_table_unlocked(path);
    if (t) {
        table_upsert(t, key, val);
        save_table_unlocked(t);
    }
    pthread_mutex_unlock(&g_cache_mtx);
}

void vk_autotune_cache_clear_memory(void) {
    pthread_mutex_lock(&g_cache_mtx);
    for (size_t i = 0; i < g_ntables; ++i) free(g_tables[i].rows);
    free(g_tables);
    g_tables = nullptr;
    g_ntables = 0;
    pthread_mutex_unlock(&g_cache_mtx);
}

/* ============================================================================
 * Mirrored Interleaved Micro-Benchmarker
 * ============================================================================ */
static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a;
    double y = *(const double*)b;
    return (x < y) ? -1 : (x > y ? 1 : 0);
}

VkInterleavedBenchResult vk_run_interleaved_bench(
    size_t num_cands,
    VkBenchRunFn run_fn,
    VkBenchConservFn conserv_fn,
    void* user_data,
    int num_rounds,
    double target_slice_ms,
    int max_batch_reps,
    double min_noise_frac,
    double max_noise_frac) {
    VkInterleavedBenchResult res = {0};
    if (num_cands == 0 || !run_fn) return res;

    int rounds = (num_rounds < 3) ? 3 : (num_rounds | 1);
    bool* alive = (bool*)calloc(num_cands, sizeof(bool));
    int* batch_reps = (int*)calloc(num_cands, sizeof(int));
    double* samples = (double*)calloc(num_cands * (size_t)rounds, sizeof(double));
    size_t* sample_cnt = (size_t*)calloc(num_cands, sizeof(size_t));
    double* med = (double*)malloc(num_cands * sizeof(double));
    double* iqr = (double*)calloc(num_cands, sizeof(double));
    if (!alive || !batch_reps || !samples || !sample_cnt || !med || !iqr) {
        free(alive); free(batch_reps); free(samples); free(sample_cnt); free(med); free(iqr);
        return res;
    }

    size_t first_alive = num_cands;
    size_t alive_count = 0;
    for (size_t c = 0; c < num_cands; ++c) {
        med[c] = 1e300;
        if (!run_fn(c, 1, user_data)) continue;
        double t0 = vk_now_ms();
        if (!run_fn(c, 1, user_data)) continue;
        double est = vk_now_ms() - t0;
        if (est < 0.01) est = 0.01;
        int reps = (int)(target_slice_ms / est) + 1;
        if (reps < 1) reps = 1;
        if (reps > max_batch_reps) reps = max_batch_reps;
        batch_reps[c] = reps;
        alive[c] = true;
        ++alive_count;
        if (first_alive == num_cands) first_alive = c;
    }

    if (first_alive == num_cands) goto cleanup;
    if (alive_count == 1) {
        res.ok = true;
        res.chosen_idx = first_alive;
        res.fastest_idx = first_alive;
        goto cleanup;
    }

    (void)run_fn(first_alive, 2, user_data);

    for (int r = 0; r < rounds; ++r) {
        bool rev = ((r & 1) != 0);
        for (size_t step = 0; step < num_cands; ++step) {
            size_t c = rev ? (num_cands - 1u - step) : step;
            if (!alive[c]) continue;
            int reps = batch_reps[c];
            double t0 = vk_now_ms();
            if (!run_fn(c, reps, user_data)) {
                alive[c] = false;
                continue;
            }
            double dt = (vk_now_ms() - t0) / (double)reps;
            samples[c * (size_t)rounds + sample_cnt[c]++] = dt;
        }
    }

    size_t fastest = num_cands;
    double fastest_ms = 1e300;
    for (size_t c = 0; c < num_cands; ++c) {
        size_t n = sample_cnt[c];
        if (!alive[c] || n == 0) continue;
        double* s = &samples[c * (size_t)rounds];
        qsort(s, n, sizeof(double), cmp_double);
        med[c] = s[n / 2];
        if (n >= 4) iqr[c] = s[n - 2] - s[1];
        else if (n >= 2) iqr[c] = s[n - 1] - s[0];
        if (med[c] < fastest_ms) {
            fastest_ms = med[c];
            fastest = c;
        }
    }
    if (fastest == num_cands || !isfinite(fastest_ms)) goto cleanup;

    size_t runner_up = num_cands;
    double runner_up_ms = 1e300;
    for (size_t c = 0; c < num_cands; ++c) {
        if (c == fastest || !alive[c] || sample_cnt[c] == 0) continue;
        if (med[c] < runner_up_ms) {
            runner_up_ms = med[c];
            runner_up = c;
        }
    }

    size_t chosen = fastest;
    for (size_t c = 0; c < num_cands; ++c) {
        if (!alive[c] || sample_cnt[c] == 0 || c == chosen) continue;
        double emp = 0.5 * (iqr[fastest] + iqr[c]) / (fastest_ms > 1e-9 ? fastest_ms : 1e-9);
        double nf = (emp > min_noise_frac) ? emp : min_noise_frac;
        if (nf > max_noise_frac) nf = max_noise_frac;
        if (med[c] <= fastest_ms * (1.0 + nf) && conserv_fn && conserv_fn(c, chosen, user_data)) {
            chosen = c;
        }
    }
    if (chosen != fastest) {
        runner_up = fastest;
        runner_up_ms = fastest_ms;
    }

    res.ok = true;
    res.chosen_idx = chosen;
    res.fastest_idx = fastest;
    res.chosen_ms = med[chosen];
    res.fastest_ms = fastest_ms;
    res.conservative_fallback = (chosen != fastest);
    if (runner_up != num_cands && isfinite(runner_up_ms)) {
        res.has_runner_up = true;
        res.runner_up_idx = runner_up;
        res.runner_up_ms = runner_up_ms;
        res.margin_pct = ((runner_up_ms - med[chosen]) / (med[chosen] > 1e-9 ? med[chosen] : 1e-9)) * 100.0;
    }

cleanup:
    free(alive);
    free(batch_reps);
    free(samples);
    free(sample_cnt);
    free(med);
    free(iqr);
    return res;
}
