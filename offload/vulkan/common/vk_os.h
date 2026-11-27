#ifndef TLS_VK_OS_H
#define TLS_VK_OS_H

/* Thin cross-platform OS shim for offload/vulkan (POSIX + Win32). */

#if !defined(_WIN32) && !defined(__APPLE__)
#    ifndef _GNU_SOURCE
#        define _GNU_SOURCE
#    endif
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <direct.h>
#    include <process.h>
#    include <windows.h>

typedef SRWLOCK vk_mutex_t;
#    define VK_MUTEX_INITIALIZER SRWLOCK_INIT

static inline void vk_mutex_init(vk_mutex_t* m) { InitializeSRWLock(m); }
static inline void vk_mutex_lock(vk_mutex_t* m) { AcquireSRWLockExclusive(m); }
static inline void vk_mutex_unlock(vk_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static inline void vk_mutex_destroy(vk_mutex_t* m) { (void)m; }

typedef SRWLOCK vk_rwlock_t;
static inline void vk_rwlock_init(vk_rwlock_t* l) { InitializeSRWLock(l); }
static inline void vk_rwlock_rdlock(vk_rwlock_t* l) { AcquireSRWLockShared(l); }
static inline void vk_rwlock_wrlock(vk_rwlock_t* l) { AcquireSRWLockExclusive(l); }
static inline void vk_rwlock_rdunlock(vk_rwlock_t* l) { ReleaseSRWLockShared(l); }
static inline void vk_rwlock_wrunlock(vk_rwlock_t* l) { ReleaseSRWLockExclusive(l); }
static inline void vk_rwlock_destroy(vk_rwlock_t* l) { (void)l; }

typedef CONDITION_VARIABLE vk_cond_t;
static inline void vk_cond_init(vk_cond_t* c) { InitializeConditionVariable(c); }
static inline void vk_cond_wait(vk_cond_t* c, vk_mutex_t* m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static inline void vk_cond_signal(vk_cond_t* c) { WakeConditionVariable(c); }
static inline void vk_cond_destroy(vk_cond_t* c) { (void)c; }

typedef INIT_ONCE vk_once_t;
#    define VK_ONCE_INIT INIT_ONCE_STATIC_INIT

static inline BOOL CALLBACK vk_once_trampoline(PINIT_ONCE once, PVOID param, PVOID* ctx) {
    (void)once;
    (void)ctx;
    void (*fn)(void) = (void (*)(void))param;
    fn();
    return TRUE;
}

static inline void vk_call_once(vk_once_t* once, void (*fn)(void)) { InitOnceExecuteOnce(once, vk_once_trampoline, (PVOID)fn, nullptr); }

static inline double vk_os_now_ms(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}

static inline long vk_os_getpid(void) { return (long)_getpid(); }

static inline long vk_os_cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)(si.dwNumberOfProcessors > 0 ? si.dwNumberOfProcessors : 1);
}

static inline void vk_os_mkdir(const char* path) { (void)_mkdir(path); }

static inline bool vk_os_replace_file(const char* src, const char* dst) {
    if (MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING)) return true;
    DeleteFileA(src);
    return false;
}

static inline void vk_os_mkdir_p_parent(const char* filepath) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", filepath);
    char* slash = nullptr;
    for (char* p = tmp; *p; ++p) {
        if (*p == '/' || *p == '\\') slash = p;
    }
    if (!slash || slash == tmp) return;
    *slash = '\0';
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/' || *p == '\\') {
            if (p == tmp + 2 && tmp[1] == ':') continue;
            char saved = *p;
            *p = '\0';
            vk_os_mkdir(tmp);
            *p = saved;
        }
    }
    vk_os_mkdir(tmp);
}

static inline void vk_os_default_cache_dir(char* out, size_t out_cap) {
    const char* local = getenv("LOCALAPPDATA");
    if (local && local[0] != '\0') {
        snprintf(out, out_cap, "%s\\toeplitz_ls", local);
        return;
    }
    const char* temp = getenv("TEMP");
    if (temp && temp[0] != '\0') {
        snprintf(out, out_cap, "%s\\toeplitz_ls", temp);
        return;
    }
    snprintf(out, out_cap, ".\\toeplitz_ls_cache");
}

typedef HMODULE vk_os_lib_t;
static inline vk_os_lib_t vk_os_dlopen(const char* name) { return LoadLibraryA(name); }
static inline void* vk_os_dlsym(vk_os_lib_t lib, const char* sym) { return (void*)GetProcAddress(lib, sym); }
static inline void vk_os_dlclose(vk_os_lib_t lib) {
    if (lib) FreeLibrary(lib);
}

static inline bool vk_os_self_dir(const void* addr_in_module, char* out, size_t out_cap) {
    HMODULE hm = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)addr_in_module, &hm)) {
        return false;
    }
    DWORD n = GetModuleFileNameA(hm, out, (DWORD)out_cap);
    if (n == 0 || n >= out_cap) return false;
    char* slash = nullptr;
    for (char* p = out; *p; ++p) {
        if (*p == '/' || *p == '\\') slash = p;
    }
    if (!slash) return false;
    *slash = '\0';
    return true;
}

#else /* POSIX (Linux / macOS) */

#    include <dlfcn.h>
#    include <pthread.h>
#    include <sys/stat.h>
#    include <time.h>
#    include <unistd.h>

typedef pthread_mutex_t vk_mutex_t;
#    define VK_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

static inline void vk_mutex_init(vk_mutex_t* m) { pthread_mutex_init(m, nullptr); }
static inline void vk_mutex_lock(vk_mutex_t* m) { pthread_mutex_lock(m); }
static inline void vk_mutex_unlock(vk_mutex_t* m) { pthread_mutex_unlock(m); }
static inline void vk_mutex_destroy(vk_mutex_t* m) { pthread_mutex_destroy(m); }

typedef pthread_rwlock_t vk_rwlock_t;
static inline void vk_rwlock_init(vk_rwlock_t* l) { pthread_rwlock_init(l, nullptr); }
static inline void vk_rwlock_rdlock(vk_rwlock_t* l) { pthread_rwlock_rdlock(l); }
static inline void vk_rwlock_wrlock(vk_rwlock_t* l) { pthread_rwlock_wrlock(l); }
static inline void vk_rwlock_rdunlock(vk_rwlock_t* l) { pthread_rwlock_unlock(l); }
static inline void vk_rwlock_wrunlock(vk_rwlock_t* l) { pthread_rwlock_unlock(l); }
static inline void vk_rwlock_destroy(vk_rwlock_t* l) { pthread_rwlock_destroy(l); }

typedef pthread_cond_t vk_cond_t;
static inline void vk_cond_init(vk_cond_t* c) { pthread_cond_init(c, nullptr); }
static inline void vk_cond_wait(vk_cond_t* c, vk_mutex_t* m) { pthread_cond_wait(c, m); }
static inline void vk_cond_signal(vk_cond_t* c) { pthread_cond_signal(c); }
static inline void vk_cond_destroy(vk_cond_t* c) { pthread_cond_destroy(c); }

typedef pthread_once_t vk_once_t;
#    define VK_ONCE_INIT PTHREAD_ONCE_INIT
static inline void vk_call_once(vk_once_t* once, void (*fn)(void)) { pthread_once(once, fn); }

static inline double vk_os_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static inline long vk_os_getpid(void) { return (long)getpid(); }

static inline long vk_os_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1;
}

static inline void vk_os_mkdir(const char* path) { (void)mkdir(path, 0755); }

static inline bool vk_os_replace_file(const char* src, const char* dst) {
    if (rename(src, dst) == 0) return true;
    (void)unlink(src);
    return false;
}

static inline void vk_os_mkdir_p_parent(const char* filepath) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", filepath);
    char* slash = strrchr(tmp, '/');
    if (!slash || slash == tmp) return;
    *slash = '\0';
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            vk_os_mkdir(tmp);
            *p = '/';
        }
    }
    vk_os_mkdir(tmp);
}

static inline void vk_os_default_cache_dir(char* out, size_t out_cap) {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0] != '\0') {
        snprintf(out, out_cap, "%s/toeplitz_ls", xdg);
        return;
    }
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        snprintf(out, out_cap, "%s/.cache/toeplitz_ls", home);
        return;
    }
    snprintf(out, out_cap, "/tmp/toeplitz_ls_vk_cache");
}

typedef void* vk_os_lib_t;
static inline vk_os_lib_t vk_os_dlopen(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
static inline void* vk_os_dlsym(vk_os_lib_t lib, const char* sym) { return dlsym(lib, sym); }
static inline void vk_os_dlclose(vk_os_lib_t lib) {
    if (lib) dlclose(lib);
}

static inline bool vk_os_self_dir(const void* addr_in_module, char* out, size_t out_cap) {
    Dl_info info;
    if (!dladdr(addr_in_module, &info) || !info.dli_fname) return false;
    snprintf(out, out_cap, "%s", info.dli_fname);
    char* slash = strrchr(out, '/');
    if (!slash) return false;
    *slash = '\0';
    return true;
}

#endif /* _WIN32 */

#endif /* TLS_VK_OS_H */
