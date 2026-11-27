#if !defined(_WIN32) && !defined(__APPLE__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>

#include "common/vk_loader.h"
#include "common/vk_os.h"

PFN_vkGetInstanceProcAddr tls_pfn_vkGetInstanceProcAddr = nullptr;
PFN_vkEnumerateInstanceVersion tls_pfn_vkEnumerateInstanceVersion = nullptr;
PFN_vkCreateInstance tls_pfn_vkCreateInstance = nullptr;

#define TLS_VK_DEF_PFN(name) PFN_##name tls_pfn_##name = nullptr;
TLS_VK_INSTANCE_FUNCS(TLS_VK_DEF_PFN)
#undef TLS_VK_DEF_PFN

static vk_once_t g_loader_once = VK_ONCE_INIT;
static vk_os_lib_t g_vk_lib = nullptr;
static bool g_loader_ok = false;
static char g_loader_err[256] = "";

static bool try_open_loader(const char* path) {
    if (!path || path[0] == '\0') return false;
    vk_os_lib_t lib = vk_os_dlopen(path);
    if (!lib) return false;
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)vk_os_dlsym(lib, "vkGetInstanceProcAddr");
    if (!gipa) {
        vk_os_dlclose(lib);
        return false;
    }
    PFN_vkCreateInstance ci = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!ci) {
        /* Some direct ICDs (e.g. older MoltenVK builds) export vkCreateInstance directly. */
        ci = (PFN_vkCreateInstance)vk_os_dlsym(lib, "vkCreateInstance");
    }
    if (!ci) {
        vk_os_dlclose(lib);
        return false;
    }
    PFN_vkEnumerateInstanceVersion eiv = (PFN_vkEnumerateInstanceVersion)gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (!eiv) {
        eiv = (PFN_vkEnumerateInstanceVersion)vk_os_dlsym(lib, "vkEnumerateInstanceVersion");
    }
    g_vk_lib = lib;
    tls_pfn_vkGetInstanceProcAddr = gipa;
    tls_pfn_vkCreateInstance = ci;
    tls_pfn_vkEnumerateInstanceVersion = eiv;
    g_loader_ok = true;
    return true;
}

static void loader_init_once(void) {
    const char* env_loader = getenv("TOEPLITZ_LS_VULKAN_LOADER");
    if (env_loader && env_loader[0] != '\0') {
        if (try_open_loader(env_loader)) return;
        snprintf(g_loader_err, sizeof(g_loader_err), "failed to load TOEPLITZ_LS_VULKAN_LOADER='%s'", env_loader);
        return;
    }

#if defined(_WIN32)
    if (try_open_loader("vulkan-1.dll")) return;
    snprintf(g_loader_err, sizeof(g_loader_err), "vulkan-1.dll not found (install a Vulkan-capable GPU driver)");
#elif defined(__APPLE__)
    char self_dir[512];
    if (vk_os_self_dir((const void*)vk_loader_init, self_dir, sizeof(self_dir))) {
        char sibling[600];
        snprintf(sibling, sizeof(sibling), "%s/libMoltenVK.dylib", self_dir);
        if (try_open_loader(sibling)) return;
    }
    const char* candidates[] = {
        "libMoltenVK.dylib", "libvulkan.1.dylib", "libvulkan.dylib", "/usr/local/lib/libMoltenVK.dylib", "/opt/homebrew/lib/libMoltenVK.dylib",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (try_open_loader(candidates[i])) return;
    }
    snprintf(g_loader_err, sizeof(g_loader_err), "libMoltenVK.dylib / libvulkan.1.dylib not found");
#else
    if (try_open_loader("libvulkan.so.1")) return;
    if (try_open_loader("libvulkan.so")) return;
    snprintf(g_loader_err, sizeof(g_loader_err), "libvulkan.so.1 not found (install the system Vulkan loader / ICD)");
#endif
}

bool vk_loader_init(void) {
    vk_call_once(&g_loader_once, loader_init_once);
    return g_loader_ok;
}

bool vk_loader_load_instance(VkInstance instance) {
    if (!g_loader_ok || instance == VK_NULL_HANDLE || !tls_pfn_vkGetInstanceProcAddr) return false;

    bool all_ok = true;
#define TLS_VK_LOAD_PFN(name)                                                                 \
    do {                                                                                      \
        tls_pfn_##name = (PFN_##name)tls_pfn_vkGetInstanceProcAddr(instance, #name);          \
        if (!tls_pfn_##name && g_vk_lib) {                                                    \
            tls_pfn_##name = (PFN_##name)vk_os_dlsym(g_vk_lib, #name);                        \
        }                                                                                     \
        if (!tls_pfn_##name) {                                                                \
            snprintf(g_loader_err, sizeof(g_loader_err), "missing Vulkan symbol: %s", #name); \
            all_ok = false;                                                                   \
        }                                                                                     \
    } while (0);

    TLS_VK_INSTANCE_FUNCS(TLS_VK_LOAD_PFN)
#undef TLS_VK_LOAD_PFN

    return all_ok;
}

const char* vk_loader_error(void) { return g_loader_err; }
