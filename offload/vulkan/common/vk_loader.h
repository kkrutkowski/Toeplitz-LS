#ifndef TLS_VK_LOADER_H
#define TLS_VK_LOADER_H

/* Runtime Vulkan loader (VK_NO_PROTOTYPES) for zero link-time libvulkan dependency.
 * Loads libvulkan.so.1 (Linux), vulkan-1.dll (Windows), or bundled libMoltenVK.dylib
 * / system libvulkan.1.dylib (macOS) on first use. */

#ifndef VK_NO_PROTOTYPES
#    define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

/* Global entry points (available immediately after vk_loader_init()) */
extern PFN_vkGetInstanceProcAddr tls_pfn_vkGetInstanceProcAddr;
extern PFN_vkEnumerateInstanceVersion tls_pfn_vkEnumerateInstanceVersion;
extern PFN_vkCreateInstance tls_pfn_vkCreateInstance;

/* Instance & device entry points (available after vk_loader_load_instance()) */
#define TLS_VK_INSTANCE_FUNCS(X)                \
    X(vkEnumeratePhysicalDevices)               \
    X(vkEnumerateDeviceExtensionProperties)     \
    X(vkGetPhysicalDeviceProperties)            \
    X(vkGetPhysicalDeviceProperties2)           \
    X(vkGetPhysicalDeviceFeatures2)             \
    X(vkGetPhysicalDeviceMemoryProperties)      \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkCreateDevice)                           \
    X(vkGetDeviceProcAddr)                      \
    X(vkGetDeviceQueue)                         \
    X(vkCreateBuffer)                           \
    X(vkDestroyBuffer)                          \
    X(vkGetBufferMemoryRequirements)            \
    X(vkAllocateMemory)                         \
    X(vkFreeMemory)                             \
    X(vkBindBufferMemory)                       \
    X(vkMapMemory)                              \
    X(vkUnmapMemory)                            \
    X(vkCreateCommandPool)                      \
    X(vkDestroyCommandPool)                     \
    X(vkAllocateCommandBuffers)                 \
    X(vkBeginCommandBuffer)                     \
    X(vkEndCommandBuffer)                       \
    X(vkResetCommandBuffer)                     \
    X(vkCreateFence)                            \
    X(vkDestroyFence)                           \
    X(vkResetFences)                            \
    X(vkWaitForFences)                          \
    X(vkQueueSubmit)                            \
    X(vkCreateDescriptorPool)                   \
    X(vkDestroyDescriptorPool)                  \
    X(vkResetDescriptorPool)                    \
    X(vkAllocateDescriptorSets)                 \
    X(vkUpdateDescriptorSets)                   \
    X(vkCreateDescriptorSetLayout)              \
    X(vkCreatePipelineLayout)                   \
    X(vkCreatePipelineCache)                    \
    X(vkGetPipelineCacheData)                   \
    X(vkCreateShaderModule)                     \
    X(vkCreateComputePipelines)                 \
    X(vkDestroyPipeline)                        \
    X(vkCmdBindPipeline)                        \
    X(vkCmdBindDescriptorSets)                  \
    X(vkCmdPushConstants)                       \
    X(vkCmdDispatch)                            \
    X(vkCmdCopyBuffer)                          \
    X(vkCmdFillBuffer)                          \
    X(vkCmdPipelineBarrier)

#define TLS_VK_DECL_PFN(name) extern PFN_##name tls_pfn_##name;
TLS_VK_INSTANCE_FUNCS(TLS_VK_DECL_PFN)
#undef TLS_VK_DECL_PFN

/* Redirect direct vk* calls to the dynamically loaded function pointers. */
#define vkGetInstanceProcAddr tls_pfn_vkGetInstanceProcAddr
#define vkEnumerateInstanceVersion tls_pfn_vkEnumerateInstanceVersion
#define vkCreateInstance tls_pfn_vkCreateInstance
#define vkEnumeratePhysicalDevices tls_pfn_vkEnumeratePhysicalDevices
#define vkEnumerateDeviceExtensionProperties tls_pfn_vkEnumerateDeviceExtensionProperties
#define vkGetPhysicalDeviceProperties tls_pfn_vkGetPhysicalDeviceProperties
#define vkGetPhysicalDeviceProperties2 tls_pfn_vkGetPhysicalDeviceProperties2
#define vkGetPhysicalDeviceFeatures2 tls_pfn_vkGetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceMemoryProperties tls_pfn_vkGetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceQueueFamilyProperties tls_pfn_vkGetPhysicalDeviceQueueFamilyProperties
#define vkCreateDevice tls_pfn_vkCreateDevice
#define vkGetDeviceProcAddr tls_pfn_vkGetDeviceProcAddr
#define vkGetDeviceQueue tls_pfn_vkGetDeviceQueue
#define vkCreateBuffer tls_pfn_vkCreateBuffer
#define vkDestroyBuffer tls_pfn_vkDestroyBuffer
#define vkGetBufferMemoryRequirements tls_pfn_vkGetBufferMemoryRequirements
#define vkAllocateMemory tls_pfn_vkAllocateMemory
#define vkFreeMemory tls_pfn_vkFreeMemory
#define vkBindBufferMemory tls_pfn_vkBindBufferMemory
#define vkMapMemory tls_pfn_vkMapMemory
#define vkUnmapMemory tls_pfn_vkUnmapMemory
#define vkCreateCommandPool tls_pfn_vkCreateCommandPool
#define vkDestroyCommandPool tls_pfn_vkDestroyCommandPool
#define vkAllocateCommandBuffers tls_pfn_vkAllocateCommandBuffers
#define vkBeginCommandBuffer tls_pfn_vkBeginCommandBuffer
#define vkEndCommandBuffer tls_pfn_vkEndCommandBuffer
#define vkResetCommandBuffer tls_pfn_vkResetCommandBuffer
#define vkCreateFence tls_pfn_vkCreateFence
#define vkDestroyFence tls_pfn_vkDestroyFence
#define vkResetFences tls_pfn_vkResetFences
#define vkWaitForFences tls_pfn_vkWaitForFences
#define vkQueueSubmit tls_pfn_vkQueueSubmit
#define vkCreateDescriptorPool tls_pfn_vkCreateDescriptorPool
#define vkDestroyDescriptorPool tls_pfn_vkDestroyDescriptorPool
#define vkResetDescriptorPool tls_pfn_vkResetDescriptorPool
#define vkAllocateDescriptorSets tls_pfn_vkAllocateDescriptorSets
#define vkUpdateDescriptorSets tls_pfn_vkUpdateDescriptorSets
#define vkCreateDescriptorSetLayout tls_pfn_vkCreateDescriptorSetLayout
#define vkCreatePipelineLayout tls_pfn_vkCreatePipelineLayout
#define vkCreatePipelineCache tls_pfn_vkCreatePipelineCache
#define vkGetPipelineCacheData tls_pfn_vkGetPipelineCacheData
#define vkCreateShaderModule tls_pfn_vkCreateShaderModule
#define vkCreateComputePipelines tls_pfn_vkCreateComputePipelines
#define vkDestroyPipeline tls_pfn_vkDestroyPipeline
#define vkCmdBindPipeline tls_pfn_vkCmdBindPipeline
#define vkCmdBindDescriptorSets tls_pfn_vkCmdBindDescriptorSets
#define vkCmdPushConstants tls_pfn_vkCmdPushConstants
#define vkCmdDispatch tls_pfn_vkCmdDispatch
#define vkCmdCopyBuffer tls_pfn_vkCmdCopyBuffer
#define vkCmdFillBuffer tls_pfn_vkCmdFillBuffer
#define vkCmdPipelineBarrier tls_pfn_vkCmdPipelineBarrier

bool vk_loader_init(void);
bool vk_loader_load_instance(VkInstance instance);
const char* vk_loader_error(void);

#endif /* TLS_VK_LOADER_H */
