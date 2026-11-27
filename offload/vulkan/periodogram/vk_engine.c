/* Plain C23 Multi-Stream Vulkan Periodogram Engine (FastChi2 & IHS). */
#include "periodogram/vk_engine.h"
#include "periodogram/vk_planner.h"
#include "periodogram/vk_preprocess.h"
#include "periodogram/vk_subgrid.h"
#include "linalg/vk_levinson.h"
#include "nufft/vk_nufft1.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local char g_tls_vk_last_err[512] = "";
static _Thread_local VkHostFastChi2Data g_tls_fc2_host = {0};
static _Thread_local VkHostIhsData      g_tls_ihs_host = {0};

static void set_last_error(const char* msg) {
    if (!msg) {
        g_tls_vk_last_err[0] = '\0';
        return;
    }
    snprintf(g_tls_vk_last_err, sizeof(g_tls_vk_last_err), "%.511s", msg);
}

static void clear_last_error(void) {
    g_tls_vk_last_err[0] = '\0';
}

/* ============================================================================
 * Growable Vulkan Buffer Slab
 * ============================================================================ */

typedef struct VkGrowBuf {
    VkBufferAlloc   buf;
    VkBufferMemKind kind;
} VkGrowBuf;

static bool grow_buf_ensure(VkGrowBuf* g, VkDeviceContext* ctx, size_t bytes, size_t max_cap) {
    if (bytes == 0u) return true;
    if (g->buf.buffer != VK_NULL_HANDLE && g->buf.size >= bytes) return true;
    size_t nc = (size_t)g->buf.size + ((size_t)g->buf.size >> 1);
    if (nc < bytes) nc = bytes;
    if (max_cap >= bytes && nc > max_cap) nc = max_cap;
    vk_buf_destroy(&g->buf);
    return vk_buf_create(ctx, nc, g->kind, &g->buf);
}

static void grow_buf_reset(VkGrowBuf* g) {
    vk_buf_destroy(&g->buf);
}

/* ============================================================================
 * Per-Slot Cached NuFFT-1 Plan Table
 * ============================================================================ */

typedef struct VkPlanCacheEntry {
    VkPrecKind   prec;
    int          N_block;
    VkPswfMode   mode;
    bool         geo_fp16;
    size_t       est_bytes;
    VkNufft1Plan plan;
} VkPlanCacheEntry;

typedef struct VkStreamSlot {
    VkDeviceContext*  ctx;
    VkCmdBatch        batch;

    VkGrowBuf         h_in;
    VkGrowBuf         h_power;
    VkGrowBuf         h_cond;

    VkGrowBuf         d_in;
    VkGrowBuf         d_src;
    VkGrowBuf         d_trig;
    VkGrowBuf         d_ihs_acc;
    VkGrowBuf         d_a_scratch;
    VkGrowBuf         d_power;
    VkGrowBuf         d_cond;

    VkPlanCacheEntry* plans;
    size_t            plan_count;
    size_t            plan_cap;
    size_t            cached_plan_bytes;
} VkStreamSlot;

static size_t slot_slab_bytes(const VkStreamSlot* s) {
    return (size_t)s->d_in.buf.size + (size_t)s->d_src.buf.size + (size_t)s->d_trig.buf.size +
           (size_t)s->d_ihs_acc.buf.size + (size_t)s->d_a_scratch.buf.size +
           (size_t)s->d_power.buf.size + (size_t)s->d_cond.buf.size;
}

static void slot_clear_plans(VkStreamSlot* s) {
    for (size_t i = 0; i < s->plan_count; ++i) {
        vk_nufft1_plan_destroy(&s->plans[i].plan);
    }
    s->plan_count = 0;
    s->cached_plan_bytes = 0;
}

static size_t proj_one(const VkGrowBuf* b, size_t need) {
    return (need <= (size_t)b->buf.size) ? (size_t)b->buf.size : need;
}

static void trim_one(VkGrowBuf* b, size_t need) {
    if (need == 0u || (size_t)b->buf.size > need * 2u) {
        grow_buf_reset(b);
    }
}

static void slot_enforce_budget(
    VkStreamSlot* s, size_t slot_budget,
    size_t need_in, size_t need_src, size_t need_trig,
    size_t need_acc, size_t need_scr, size_t need_pow, size_t need_cond) {
    size_t proj = proj_one(&s->d_in, need_in) + proj_one(&s->d_src, need_src) +
                  proj_one(&s->d_trig, need_trig) + proj_one(&s->d_ihs_acc, need_acc) +
                  proj_one(&s->d_a_scratch, need_scr) + proj_one(&s->d_power, need_pow) +
                  proj_one(&s->d_cond, need_cond) + s->cached_plan_bytes;
    if (proj <= slot_budget) return;

    trim_one(&s->d_in, need_in);
    trim_one(&s->d_src, need_src);
    trim_one(&s->d_trig, need_trig);
    trim_one(&s->d_ihs_acc, need_acc);
    trim_one(&s->d_a_scratch, need_scr);
    trim_one(&s->d_power, need_pow);
    trim_one(&s->d_cond, need_cond);

    size_t after = proj_one(&s->d_in, need_in) + proj_one(&s->d_src, need_src) +
                   proj_one(&s->d_trig, need_trig) + proj_one(&s->d_ihs_acc, need_acc) +
                   proj_one(&s->d_a_scratch, need_scr) + proj_one(&s->d_power, need_pow) +
                   proj_one(&s->d_cond, need_cond) + s->cached_plan_bytes;
    if (after > slot_budget) {
        slot_clear_plans(s);
    }
}

static VkNufft1Plan* slot_get_or_create_plan(
    VkStreamSlot* s,
    pthread_mutex_t* plan_create_mtx,
    size_t slot_budget,
    VkPrecKind prec,
    int M_hint,
    int N_block,
    int w,
    size_t plan_max_bs,
    VkPswfMode mode,
    bool geo_fp16) {
    for (size_t i = 0; i < s->plan_count; ++i) {
        VkPlanCacheEntry* e = &s->plans[i];
        if (e->prec == prec && e->N_block == N_block && e->mode == mode && e->geo_fp16 == geo_fp16) {
            return &e->plan;
        }
    }

    size_t cbytes = vk_complex_bytes_for(prec);
    size_t sbytes = vk_scalar_bytes_for(prec);
    int init_M = (M_hint > 256) ? M_hint : 256;
    size_t est_bytes = 2u * plan_max_bs * (size_t)(2 * N_block) * cbytes +
                       (size_t)init_M * (4u + sbytes + cbytes + (size_t)w * sbytes) * 3u;
    if (slot_slab_bytes(s) + s->cached_plan_bytes + est_bytes > slot_budget && s->cached_plan_bytes > 0u) {
        slot_clear_plans(s);
    }

    if (s->plan_count == s->plan_cap) {
        size_t nc = (s->plan_cap == 0u) ? 8u : (s->plan_cap * 2u);
        VkPlanCacheEntry* np = (VkPlanCacheEntry*)realloc(s->plans, nc * sizeof(VkPlanCacheEntry));
        if (!np) return nullptr;
        s->plans = np;
        s->plan_cap = nc;
    }

    pthread_mutex_lock(plan_create_mtx);
    VkNufft1LaunchConfig cfg = vk_nufft1_launch_config_default();
    cfg.use_geo_scale_fp16 = geo_fp16;
    VkPlanCacheEntry* e = &s->plans[s->plan_count];
    memset(e, 0, sizeof(*e));
    bool ok = vk_nufft1_plan_init(&e->plan, s->ctx, prec, +1, init_M, N_block, w, mode, plan_max_bs, &cfg);
    pthread_mutex_unlock(plan_create_mtx);
    if (!ok) return nullptr;

    e->prec = prec;
    e->N_block = N_block;
    e->mode = mode;
    e->geo_fp16 = geo_fp16;
    e->est_bytes = est_bytes;
    s->plan_count++;
    s->cached_plan_bytes += est_bytes;
    return &e->plan;
}

/* ============================================================================
 * Per-Device Multi-Stream Runtime Pool
 * ============================================================================ */

typedef struct VkDeviceRuntime {
    VkDeviceContext* ctx;
    int              num_slots;
    size_t           slot_mem_budget;
    VkStreamSlot*    slots;

    pthread_mutex_t  pool_mtx;
    pthread_cond_t   pool_cv;
    int*             idle_stack;
    int              idle_top;

    pthread_mutex_t  plan_create_mtx;
    pthread_mutex_t  fp64_probe_mtx;
    bool             fp64_probed;
    double           fp64_to_fp32_ratio;
} VkDeviceRuntime;

static pthread_mutex_t g_rt_mgr_mtx = PTHREAD_MUTEX_INITIALIZER;
static VkDeviceRuntime* g_runtimes[32] = {nullptr};

static VkDeviceRuntime* get_device_runtime(int device_id) {
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (device_id < 0 || (size_t)device_id >= ndevs || device_id >= 32) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Invalid Vulkan device ID %d (available devices: %zu)", device_id, ndevs);
        set_last_error(buf);
        return nullptr;
    }

    pthread_mutex_lock(&g_rt_mgr_mtx);
    if (g_runtimes[device_id]) {
        VkDeviceRuntime* rt = g_runtimes[device_id];
        pthread_mutex_unlock(&g_rt_mgr_mtx);
        return rt;
    }

    char err[256] = {0};
    VkDeviceContext* ctx = vk_ctx_get_shared(device_id, err, sizeof(err));
    if (!ctx) {
        pthread_mutex_unlock(&g_rt_mgr_mtx);
        set_last_error(err[0] ? err : "Failed to initialize Vulkan device context");
        return nullptr;
    }

    VkDeviceRuntime* rt = (VkDeviceRuntime*)calloc(1, sizeof(VkDeviceRuntime));
    if (!rt) {
        pthread_mutex_unlock(&g_rt_mgr_mtx);
        set_last_error("Out of host memory allocating Vulkan device runtime");
        return nullptr;
    }
    rt->ctx = ctx;
    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(ctx);
    int env_k = vk_env_int_or("TLS_VK_STREAMS", 0);
    if (env_k > 0) {
        rt->num_slots = (env_k > 32) ? 32 : env_k;
    } else {
        rt->num_slots = (int)vk_device_recommended_stream_count(dev_cfg);
        if (rt->num_slots < 1) rt->num_slots = 1;
    }
    rt->slot_mem_budget = vk_device_per_stream_mem_budget(dev_cfg, (size_t)rt->num_slots);

    pthread_mutex_init(&rt->pool_mtx, nullptr);
    pthread_cond_init(&rt->pool_cv, nullptr);
    pthread_mutex_init(&rt->plan_create_mtx, nullptr);
    pthread_mutex_init(&rt->fp64_probe_mtx, nullptr);

    rt->slots = (VkStreamSlot*)calloc((size_t)rt->num_slots, sizeof(VkStreamSlot));
    rt->idle_stack = (int*)malloc((size_t)rt->num_slots * sizeof(int));
    if (!rt->slots || !rt->idle_stack) {
        free(rt->slots); free(rt->idle_stack); free(rt);
        pthread_mutex_unlock(&g_rt_mgr_mtx);
        set_last_error("Out of host memory allocating Vulkan stream slots");
        return nullptr;
    }

    uint32_t nqueues = vk_ctx_queue_count(ctx);
    const VkBufferMemKind in_kind  = dev_cfg->has_host_unified_memory ? VK_MEM_UMA_UPLOAD   : VK_MEM_DEVICE_LOCAL;
    const VkBufferMemKind out_kind = dev_cfg->has_uma_cached          ? VK_MEM_UMA_READBACK : VK_MEM_DEVICE_LOCAL;
    for (int i = 0; i < rt->num_slots; ++i) {
        VkStreamSlot* s = &rt->slots[i];
        s->ctx = ctx;
        s->h_in.kind = VK_MEM_HOST_UPLOAD;
        s->h_power.kind = VK_MEM_HOST_READBACK;
        s->h_cond.kind = VK_MEM_HOST_READBACK;
        s->d_in.kind = in_kind;
        s->d_src.kind = VK_MEM_DEVICE_LOCAL;
        s->d_trig.kind = VK_MEM_DEVICE_LOCAL;
        s->d_ihs_acc.kind = VK_MEM_DEVICE_LOCAL;
        s->d_a_scratch.kind = VK_MEM_DEVICE_LOCAL;
        s->d_power.kind = out_kind;
        s->d_cond.kind = out_kind;
        if (!vk_batch_init(&s->batch, ctx, (uint32_t)i % (nqueues > 0u ? nqueues : 1u))) {
            for (int j = 0; j < i; ++j) vk_batch_destroy(&rt->slots[j].batch);
            free(rt->slots); free(rt->idle_stack); free(rt);
            pthread_mutex_unlock(&g_rt_mgr_mtx);
            set_last_error("Failed to initialize Vulkan stream command batch");
            return nullptr;
        }
        rt->idle_stack[i] = i;
    }
    rt->idle_top = rt->num_slots;
    g_runtimes[device_id] = rt;
    pthread_mutex_unlock(&g_rt_mgr_mtx);
    return rt;
}

static int rt_acquire_slot(VkDeviceRuntime* rt) {
    pthread_mutex_lock(&rt->pool_mtx);
    while (rt->idle_top == 0) {
        pthread_cond_wait(&rt->pool_cv, &rt->pool_mtx);
    }
    int idx = rt->idle_stack[--rt->idle_top];
    pthread_mutex_unlock(&rt->pool_mtx);
    return idx;
}

static void rt_release_slot(VkDeviceRuntime* rt, int idx) {
    pthread_mutex_lock(&rt->pool_mtx);
    rt->idle_stack[rt->idle_top++] = idx;
    pthread_cond_signal(&rt->pool_cv);
    pthread_mutex_unlock(&rt->pool_mtx);
}

static double rt_query_fp64_ratio(VkDeviceRuntime* rt) {
    pthread_mutex_lock(&rt->fp64_probe_mtx);
    if (!rt->fp64_probed) {
        VkDeviceProbeResult pr = vk_ctx_probe_fp64_ratio(rt->ctx);
        rt->fp64_to_fp32_ratio = pr.fp64_to_fp32_ratio;
        rt->fp64_probed = true;
    }
    double r = rt->fp64_to_fp32_ratio;
    pthread_mutex_unlock(&rt->fp64_probe_mtx);
    return r;
}

/* ============================================================================
 * Validation & Output Copy Helpers
 * ============================================================================ */

static int validate_common_args(
    const void* t, const void* y, const void* dy, const void* power,
    int M, double f0, double df, int N, int degree) {
    if (!t || !y || !dy || !power || M <= 0 || N <= 0 || degree <= 0 ||
        !isfinite(f0) || !isfinite(df) || f0 < 0.0 || df <= 0.0) {
        set_last_error("Invalid periodogram argument");
        return TLS_VK_ERR_ARGUMENT;
    }
    if (M < 2 * degree + 2) {
        set_last_error("Degenerate input: M < 2 * degree + 2");
        return TLS_VK_ERR_DEGENERATE;
    }
    return TLS_VK_OK;
}

static int parse_backend(int backend, VkPswfMode* out_mode) {
    if (backend == TLS_VK_BACKEND_PSWF43) { *out_mode = VK_PSWF43; return TLS_VK_OK; }
    if (backend == TLS_VK_BACKEND_PSWF21) { *out_mode = VK_PSWF21; return TLS_VK_OK; }
    set_last_error("Unsupported backend; Vulkan offload supports 'pswf43' and 'pswf21'");
    return TLS_VK_ERR_BACKEND;
}

static void copy_dev_out_to_host(VkPrecKind prec, const void* restrict src, void* restrict dst, int Nf) {
    if (prec == VK_PREC_HALF || prec == VK_PREC_FLOAT) {
        memcpy(dst, src, (size_t)Nf * sizeof(float));
    } else if (prec == VK_PREC_DOUBLE) {
        memcpy(dst, src, (size_t)Nf * sizeof(double));
    } else {
        const VkDf* restrict s = (const VkDf*)src;
        double* restrict d = (double*)dst;
        for (int i = 0; i < Nf; ++i) {
            d[i] = (double)s[i].hi + (double)s[i].lo;
        }
    }
}

/* ============================================================================
 * Core FastChi2 Execution (All 4 Precisions)
 * ============================================================================ */

static int execute_fastchi2_common(
    VkPrecKind prec,
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int Nf, int degree,
    int backend, int normalization, int device_id,
    void* power_out, void* cond_out) {
    clear_last_error();
    const void* y_any  = y_d  ? (const void*)y_d  : (const void*)y_f;
    const void* dy_any = dy_d ? (const void*)dy_d : (const void*)dy_f;
    int st = validate_common_args(t, y_any, dy_any, power_out, M, f0, df, Nf, degree);
    if (st != TLS_VK_OK) return st;

    VkPswfMode mode = VK_PSWF43;
    if ((st = parse_backend(backend, &mode)) != TLS_VK_OK) return st;
    if (normalization < 0 || normalization > 3) {
        set_last_error("Invalid FastChi2 normalization");
        return TLS_VK_ERR_ARGUMENT;
    }
    VkNormKind norm = (VkNormKind)normalization;

    /* Validate device ID & native precision support BEFORE initializing plans */
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (device_id < 0 || (size_t)device_id >= ndevs) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Invalid Vulkan device ID %d (available devices: %zu)", device_id, ndevs);
        set_last_error(buf);
        return TLS_VK_ERR_DEVICE;
    }
    char perr[256] = {0};
    int pv = vk_validate_device_precision(&devs[device_id], prec, false, perr, sizeof(perr));
    if (pv != 0) {
        set_last_error(perr);
        return pv;
    }

    VkDeviceRuntime* rt = get_device_runtime(device_id);
    if (!rt) return TLS_VK_ERR_DEVICE;

    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(rt->ctx);
    double fp64_ratio = (prec == VK_PREC_DOUBLE) ? rt_query_fp64_ratio(rt) : 0.0;

    bool has_cond = (cond_out != nullptr);
    VkGpuPlanChoice choice = vk_choose_gpu_periodogram_plan(
        Nf, M, degree, false, has_cond, mode, prec, dev_cfg, rt->num_slots, fp64_ratio);
    if (!choice.fits_budget) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Lightcurve exceeds per-stream Vulkan device memory budget (required %.2f MiB > budget %.2f MiB; "
                 "75%% of device memory / %d streams)",
                 (double)choice.required_min_bytes / (1024.0 * 1024.0),
                 (double)choice.slot_budget_bytes / (1024.0 * 1024.0),
                 rt->num_slots);
        set_last_error(buf);
        return TLS_VK_ERR_ALLOC;
    }

    /* Lock-free CPU preprocessing on calling thread */
    st = vk_preprocess_fastchi2(prec, t, y_f, y_d, dy_f, dy_d, M, degree, choice.nfft, &g_tls_fc2_host);
    if (st != TLS_VK_OK) return st;

    int slot_idx = rt_acquire_slot(rt);
    VkStreamSlot* slot = &rt->slots[slot_idx];
    int ret = TLS_VK_OK;

    const bool is_half = (prec == VK_PREC_HALF);
    const size_t xbytes = (size_t)M * 8u;
    const size_t sbytes = vk_scalar_bytes_for(prec);
    const size_t cbytes = vk_complex_bytes_for(prec);
    const size_t obytes = vk_dev_out_bytes_for(prec);
    const size_t wbytes = (size_t)M * sbytes;
    const size_t off_w  = (xbytes + 63u) & ~(size_t)63u;
    const size_t off_yw = off_w + ((wbytes + 63u) & ~(size_t)63u);
    const size_t in_bytes = off_yw + wbytes;
    const size_t src_bytes = 2u * (size_t)choice.B_chunk * (size_t)M * cbytes;
    const size_t chunk_bins_max = (size_t)choice.B_chunk * (size_t)choice.nout;
    const size_t trig_bytes = (size_t)(3 * degree) * chunk_bins_max * cbytes;
    const size_t out_bytes = (size_t)Nf * obytes;
    const size_t cond_bytes = has_cond ? out_bytes : 0u;
    const size_t chunk_items_max = is_half ? (chunk_bins_max >> 1) : chunk_bins_max;
    const size_t chunk_items_pad = (chunk_items_max + 63u) & ~(size_t)63u;
    const size_t scr_bytes = vk_fastchi2_needs_scratch(dev_cfg, prec, degree)
        ? (2u * (size_t)(2 * degree - 1) * chunk_items_pad * vk_solve_real_bytes_for(prec))
        : 0u;
    const size_t max_alloc = vk_device_safe_max_alloc_bytes(dev_cfg);
    const size_t cap_limit = (rt->slot_mem_budget < max_alloc) ? rt->slot_mem_budget : max_alloc;

    slot_enforce_budget(slot, rt->slot_mem_budget, in_bytes, src_bytes, trig_bytes, 0u, scr_bytes, out_bytes, cond_bytes);

    VkNufft1Plan* plan = slot_get_or_create_plan(
        slot, &rt->plan_create_mtx, rt->slot_mem_budget,
        prec, M, choice.N_block, choice.w, choice.plan_max_bs, mode, false);
    if (!plan || !vk_nufft1_ensure_point_capacity(plan, M)) {
        set_last_error("Failed to create or resize Vulkan NuFFT-1 plan");
        ret = TLS_VK_ERR_ALLOC;
        goto cleanup;
    }

    const bool uma_up = (slot->d_in.kind == VK_MEM_UMA_UPLOAD);
    const bool uma_dn = (slot->d_power.kind == VK_MEM_UMA_READBACK);
    if ((!uma_up && !grow_buf_ensure(&slot->h_in, rt->ctx, in_bytes, 0u)) ||
        (!uma_dn && !grow_buf_ensure(&slot->h_power, rt->ctx, out_bytes, 0u)) ||
        (!uma_dn && has_cond && !grow_buf_ensure(&slot->h_cond, rt->ctx, cond_bytes, 0u)) ||
        !grow_buf_ensure(&slot->d_in, rt->ctx, in_bytes, cap_limit) ||
        !grow_buf_ensure(&slot->d_src, rt->ctx, src_bytes, cap_limit) ||
        !grow_buf_ensure(&slot->d_trig, rt->ctx, trig_bytes, cap_limit) ||
        !grow_buf_ensure(&slot->d_power, rt->ctx, out_bytes, cap_limit) ||
        (has_cond && !grow_buf_ensure(&slot->d_cond, rt->ctx, cond_bytes, cap_limit)) ||
        (scr_bytes > 0u && !grow_buf_ensure(&slot->d_a_scratch, rt->ctx, scr_bytes, cap_limit))) {
        set_last_error("Vulkan buffer workspace allocation failed");
        ret = TLS_VK_ERR_ALLOC;
        goto cleanup;
    }

    /* Pack (x, w, yw) directly into UMA d_in (zero-copy) or mapped upload slab -> single H2D copy */
    uint8_t* h_in_ptr = (uint8_t*)(uma_up ? slot->d_in.buf.mapped : slot->h_in.buf.mapped);
    memcpy(h_in_ptr,          g_tls_fc2_host.x,  xbytes);
    memcpy(h_in_ptr + off_w,  g_tls_fc2_host.w,  wbytes);
    memcpy(h_in_ptr + off_yw, g_tls_fc2_host.yw, wbytes);

    VkBufferSlice d_in_s   = vk_buf_slice(&slot->d_in.buf, 0, in_bytes);
    VkBufferSlice d_x_s    = vk_buf_slice(&slot->d_in.buf, 0, xbytes);
    VkBufferSlice d_src_s  = vk_buf_slice(&slot->d_src.buf, 0, VK_WHOLE_SIZE);
    VkBufferSlice d_trig_s = vk_buf_slice(&slot->d_trig.buf, 0, VK_WHOLE_SIZE);
    VkBufferSlice d_pow_s  = vk_buf_slice(&slot->d_power.buf, 0, VK_WHOLE_SIZE);
    VkBufferSlice d_cnd_s  = has_cond ? vk_buf_slice(&slot->d_cond.buf, 0, VK_WHOLE_SIZE) : (VkBufferSlice){0};
    VkBufferSlice d_scr_s  = (scr_bytes > 0u) ? vk_buf_slice(&slot->d_a_scratch.buf, 0, VK_WHOLE_SIZE) : (VkBufferSlice){0};

    uint32_t w_off_elems  = (uint32_t)(off_w / sbytes);
    uint32_t yw_off_elems = (uint32_t)(off_yw / sbytes);

    VkNufftOutOrder nufft_order = is_half ? VK_NUFFT_ORDER_SLOT_BIT_REVERSED : VK_NUFFT_ORDER_SLOT_BIT_REVERSED_RAW;
    size_t subgrid_wg = vk_device_default_padded_wg(dev_cfg, 256u);
    size_t solve_wg   = vk_device_default_padded_wg(dev_cfg, 128u);
    double block_df   = (double)choice.nout * df;
    bool df_vk        = plan->fft_plan.df_use_veltkamp;

    /* Single batched command submission across all super-chunks (zero inter-chunk CPU stalls;
     * vk_batch_dispatch auto-flushes only if ds_cap is exceeded). */
    if (!vk_batch_begin(&slot->batch)) { ret = TLS_VK_ERR_RUNTIME; goto cleanup; }
    if (!uma_up) {
        vk_batch_copy(&slot->batch, vk_buf_slice(&slot->h_in.buf, 0, in_bytes), d_in_s, in_bytes);
        vk_batch_barrier_t2c(&slot->batch);
    }
    if (is_half) {
        if (!vk_nufft1_record_set_output_scale(plan, &slot->batch, g_tls_fc2_host.row_scale)) {
            ret = TLS_VK_ERR_RUNTIME; goto cleanup;
        }
    }

    VkBufferSlice d_phi = is_half ? (VkBufferSlice){0} : vk_nufft1_get_slot_phi_hat(plan, &slot->batch);
    VkBufferSlice d_dec = is_half ? (VkBufferSlice){0} : vk_nufft1_get_slot_deconv(plan, &slot->batch);
    if (!is_half && (!vk_slice_valid(d_phi) || !vk_slice_valid(d_dec))) {
        ret = TLS_VK_ERR_ALLOC; goto cleanup;
    }

    for (int b_start = 0; b_start < choice.B_total; b_start += choice.B_chunk) {
        if (b_start > 0) vk_batch_barrier_c2c(&slot->batch);
        int b_cnt = (choice.B_total - b_start < choice.B_chunk) ? (choice.B_total - b_start) : choice.B_chunk;
        size_t chunk_bins = (size_t)b_cnt * (size_t)choice.nout;

        for (int harm = 1; harm <= 2 * degree; ++harm) {
            if (harm > 1) vk_batch_barrier_c2c(&slot->batch);
            bool need_yw = (harm <= degree);
            double qf0 = (double)harm * f0;
            double qbdf = (double)harm * block_df;
            /* build_subgrid_sources writes only d_src and reads only d_in (disjoint from
             * set_points buffers), so placing it before set_points overlaps both on the GPU
             * and uses set_points' trailing c2c barrier for both! */
            if (!vk_record_build_subgrid_sources(
                    rt->ctx, &slot->batch, prec, df_vk,
                    d_in_s, 0u, d_in_s, w_off_elems, d_in_s, yw_off_elems, need_yw,
                    d_src_s, M, b_start, b_cnt, qf0, qbdf, subgrid_wg)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
            if (!vk_nufft1_record_set_points(plan, &slot->batch, d_x_s, M, df, harm)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
            size_t row_idx = need_yw ? (size_t)(2 * (harm - 1)) : (size_t)(degree + harm - 1);
            size_t bs = (size_t)(need_yw ? (2 * b_cnt) : b_cnt);
            if (!vk_nufft1_record_execute(
                    plan, &slot->batch, d_src_s, 0u, d_trig_s,
                    (uint32_t)(row_idx * chunk_bins), bs, nufft_order)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
        }
        vk_batch_barrier_c2c(&slot->batch);

        VkFastChi2SolveParams sp = {
            degree, M, Nf, choice.nfft, plan->log_nfft, choice.nout, plan->output_shift,
            b_start, b_cnt, mode, norm,
            g_tls_fc2_host.ws_fft, g_tls_fc2_host.yws_fft, g_tls_fc2_host.chi2_ref,
            g_tls_fc2_host.b_param, g_tls_fc2_host.cyb, solve_wg,
        };
        if (!vk_record_fastchi2_solve(
                rt->ctx, &slot->batch, prec, df_vk,
                d_trig_s, d_phi, d_dec, d_scr_s, d_pow_s, d_cnd_s,
                has_cond, !is_half, &sp)) {
            ret = TLS_VK_ERR_RUNTIME; goto cleanup;
        }
    }

    if (!uma_dn) {
        vk_batch_barrier_c2t(&slot->batch);
        vk_batch_copy(&slot->batch, vk_buf_slice(&slot->d_power.buf, 0, out_bytes),
                      vk_buf_slice(&slot->h_power.buf, 0, out_bytes), out_bytes);
        if (has_cond) {
            vk_batch_copy(&slot->batch, vk_buf_slice(&slot->d_cond.buf, 0, cond_bytes),
                          vk_buf_slice(&slot->h_cond.buf, 0, cond_bytes), cond_bytes);
        }
    }
    if (!vk_batch_submit_and_wait(&slot->batch)) {
        set_last_error("Vulkan command submission failed during FastChi2 execution");
        ret = TLS_VK_ERR_RUNTIME;
        goto cleanup;
    }

    copy_dev_out_to_host(prec, uma_dn ? slot->d_power.buf.mapped : slot->h_power.buf.mapped, power_out, Nf);
    if (has_cond) {
        copy_dev_out_to_host(prec, uma_dn ? slot->d_cond.buf.mapped : slot->h_cond.buf.mapped, cond_out, Nf);
    }

cleanup:
    rt_release_slot(rt, slot_idx);
    return ret;
}

/* ============================================================================
 * Core IHS Execution (All 4 Precisions)
 * ============================================================================ */

static int execute_ihs_common(
    VkPrecKind prec,
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int Nf, int degree,
    int backend, int normalization, int device_id,
    void* power_out) {
    clear_last_error();
    const void* y_any  = y_d  ? (const void*)y_d  : (const void*)y_f;
    const void* dy_any = dy_d ? (const void*)dy_d : (const void*)dy_f;
    int st = validate_common_args(t, y_any, dy_any, power_out, M, f0, df, Nf, degree);
    if (st != TLS_VK_OK) return st;

    VkPswfMode mode = VK_PSWF43;
    if ((st = parse_backend(backend, &mode)) != TLS_VK_OK) return st;
    if (normalization != 0 && normalization != 2 && normalization != 4) {
        set_last_error("Invalid IHS normalization");
        return TLS_VK_ERR_ARGUMENT;
    }
    VkNormKind norm = (VkNormKind)normalization;

    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (device_id < 0 || (size_t)device_id >= ndevs) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Invalid Vulkan device ID %d (available devices: %zu)", device_id, ndevs);
        set_last_error(buf);
        return TLS_VK_ERR_DEVICE;
    }
    char perr[256] = {0};
    int pv = vk_validate_device_precision(&devs[device_id], prec, true, perr, sizeof(perr));
    if (pv != 0) {
        set_last_error(perr);
        return pv;
    }

    VkDeviceRuntime* rt = get_device_runtime(device_id);
    if (!rt) return TLS_VK_ERR_DEVICE;

    const VkDeviceFftConfig* dev_cfg = vk_ctx_config(rt->ctx);
    double fp64_ratio = (prec == VK_PREC_DOUBLE) ? rt_query_fp64_ratio(rt) : 0.0;

    VkGpuPlanChoice choice = vk_choose_gpu_periodogram_plan(
        Nf, M, degree, true, false, mode, prec, dev_cfg, rt->num_slots, fp64_ratio);
    if (!choice.fits_budget) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Lightcurve exceeds per-stream Vulkan device memory budget (required %.2f MiB > budget %.2f MiB; "
                 "75%% of device memory / %d streams)",
                 (double)choice.required_min_bytes / (1024.0 * 1024.0),
                 (double)choice.slot_budget_bytes / (1024.0 * 1024.0),
                 rt->num_slots);
        set_last_error(buf);
        return TLS_VK_ERR_ALLOC;
    }

    st = vk_preprocess_ihs(prec, t, y_f, y_d, dy_f, dy_d, M, choice.nfft, &g_tls_ihs_host);
    if (st != TLS_VK_OK) return st;

    int slot_idx = rt_acquire_slot(rt);
    VkStreamSlot* slot = &rt->slots[slot_idx];
    int ret = TLS_VK_OK;

    const bool is_half = (prec == VK_PREC_HALF);
    const size_t xbytes = (size_t)M * 8u;
    const size_t sbytes = vk_scalar_bytes_for(prec);
    const size_t cbytes = vk_complex_bytes_for(prec);
    const size_t obytes = vk_dev_out_bytes_for(prec);
    const size_t wbytes = (size_t)M * sbytes;
    const size_t off_wy = (xbytes + 63u) & ~(size_t)63u;
    const size_t in_bytes = off_wy + wbytes;
    const size_t src_bytes = (size_t)choice.B_chunk * (size_t)M * cbytes;
    const size_t chunk_bins_max = (size_t)choice.B_chunk * (size_t)choice.nout;
    const size_t z_bytes = chunk_bins_max * cbytes;
    const size_t acc_bytes = (degree > 1) ? (chunk_bins_max * obytes) : 0u;
    const size_t out_bytes = (size_t)Nf * obytes;
    const size_t max_alloc = vk_device_safe_max_alloc_bytes(dev_cfg);
    const size_t cap_limit = (rt->slot_mem_budget < max_alloc) ? rt->slot_mem_budget : max_alloc;

    slot_enforce_budget(slot, rt->slot_mem_budget, in_bytes, src_bytes, z_bytes, acc_bytes, 0u, out_bytes, 0u);

    VkNufft1Plan* plan = slot_get_or_create_plan(
        slot, &rt->plan_create_mtx, rt->slot_mem_budget,
        prec, M, choice.N_block, choice.w, choice.plan_max_bs, mode, is_half);
    if (!plan || !vk_nufft1_ensure_point_capacity(plan, M)) {
        set_last_error("Failed to create or resize Vulkan NuFFT-1 plan");
        ret = TLS_VK_ERR_ALLOC;
        goto cleanup;
    }

    const bool uma_up = (slot->d_in.kind == VK_MEM_UMA_UPLOAD);
    const bool uma_dn = (slot->d_power.kind == VK_MEM_UMA_READBACK);
    if ((!uma_up && !grow_buf_ensure(&slot->h_in, rt->ctx, in_bytes, 0u)) ||
        (!uma_dn && !grow_buf_ensure(&slot->h_power, rt->ctx, out_bytes, 0u)) ||
        !grow_buf_ensure(&slot->d_in, rt->ctx, in_bytes, cap_limit) ||
        !grow_buf_ensure(&slot->d_src, rt->ctx, src_bytes, cap_limit) ||
        !grow_buf_ensure(&slot->d_trig, rt->ctx, z_bytes, cap_limit) ||
        (acc_bytes > 0u && !grow_buf_ensure(&slot->d_ihs_acc, rt->ctx, acc_bytes, cap_limit)) ||
        !grow_buf_ensure(&slot->d_power, rt->ctx, out_bytes, cap_limit)) {
        set_last_error("Vulkan buffer workspace allocation failed");
        ret = TLS_VK_ERR_ALLOC;
        goto cleanup;
    }

    uint8_t* h_in_ptr = (uint8_t*)(uma_up ? slot->d_in.buf.mapped : slot->h_in.buf.mapped);
    memcpy(h_in_ptr,          g_tls_ihs_host.x,  xbytes);
    memcpy(h_in_ptr + off_wy, g_tls_ihs_host.wy, wbytes);

    VkBufferSlice d_in_s  = vk_buf_slice(&slot->d_in.buf, 0, in_bytes);
    VkBufferSlice d_x_s   = vk_buf_slice(&slot->d_in.buf, 0, xbytes);
    VkBufferSlice d_src_s = vk_buf_slice(&slot->d_src.buf, 0, VK_WHOLE_SIZE);
    VkBufferSlice d_z_s   = vk_buf_slice(&slot->d_trig.buf, 0, VK_WHOLE_SIZE);
    VkBufferSlice d_acc_s = (acc_bytes > 0u) ? vk_buf_slice(&slot->d_ihs_acc.buf, 0, VK_WHOLE_SIZE) : (VkBufferSlice){0};
    VkBufferSlice d_pow_s = vk_buf_slice(&slot->d_power.buf, 0, VK_WHOLE_SIZE);

    uint32_t wy_off_elems = (uint32_t)(off_wy / sbytes);
    size_t subgrid_wg = vk_device_default_padded_wg(dev_cfg, 256u);
    size_t solve_wg   = vk_device_default_padded_wg(dev_cfg, 128u);
    double block_df   = (double)choice.nout * df;
    bool df_vk        = plan->fft_plan.df_use_veltkamp;

    if (!vk_batch_begin(&slot->batch)) { ret = TLS_VK_ERR_RUNTIME; goto cleanup; }
    if (!uma_up) {
        vk_batch_copy(&slot->batch, vk_buf_slice(&slot->h_in.buf, 0, in_bytes), d_in_s, in_bytes);
        vk_batch_barrier_t2c(&slot->batch);
    }
    if (degree == 1) {
        if (!vk_nufft1_record_set_points(plan, &slot->batch, d_x_s, M, df, 1)) {
            ret = TLS_VK_ERR_RUNTIME; goto cleanup;
        }
    }

    VkBufferSlice d_dec = is_half
        ? vk_nufft1_get_slot_deconv_f32(plan, &slot->batch)
        : vk_nufft1_get_slot_deconv(plan, &slot->batch);
    if (!vk_slice_valid(d_dec)) { ret = TLS_VK_ERR_ALLOC; goto cleanup; }

    for (int b_start = 0; b_start < choice.B_total; b_start += choice.B_chunk) {
        if (b_start > 0) vk_batch_barrier_c2c(&slot->batch);
        int b_cnt = (choice.B_total - b_start < choice.B_chunk) ? (choice.B_total - b_start) : choice.B_chunk;

        VkIhsSolveParams sp = {
            degree, Nf, choice.nfft, plan->log_nfft, choice.nout, plan->output_shift,
            b_start, b_cnt, mode, norm,
            g_tls_ihs_host.standard_scale, g_tls_ihs_host.nll_scale, g_tls_ihs_host.norm_neff,
            solve_wg,
        };

        if (degree == 1) {
            if (!vk_record_build_subgrid_sources(
                    rt->ctx, &slot->batch, prec, df_vk,
                    d_in_s, 0u, d_in_s, wy_off_elems, (VkBufferSlice){0}, 0u, false,
                    d_src_s, M, b_start, b_cnt, f0, block_df, subgrid_wg)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
            vk_batch_barrier_c2c(&slot->batch);
            if (!vk_nufft1_record_execute(
                    plan, &slot->batch, d_src_s, 0u, d_z_s, 0u,
                    (size_t)b_cnt, VK_NUFFT_ORDER_SLOT_BIT_REVERSED_RAW)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
            vk_batch_barrier_c2c(&slot->batch);
            if (!vk_record_ihs_single(rt->ctx, &slot->batch, prec, df_vk, d_z_s, d_dec, d_pow_s, &sp)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
        } else {
            bool nll_multi = (norm == VK_NORM_NLL);
            for (int harm = 1; harm <= degree; ++harm) {
                if (harm > 1) vk_batch_barrier_c2c(&slot->batch);
                double qf0 = (double)harm * f0;
                double qbdf = (double)harm * block_df;
                if (!vk_record_build_subgrid_sources(
                        rt->ctx, &slot->batch, prec, df_vk,
                        d_in_s, 0u, d_in_s, wy_off_elems, (VkBufferSlice){0}, 0u, false,
                        d_src_s, M, b_start, b_cnt, qf0, qbdf, subgrid_wg)) {
                    ret = TLS_VK_ERR_RUNTIME; goto cleanup;
                }
                if (!vk_nufft1_record_set_points(plan, &slot->batch, d_x_s, M, df, harm)) {
                    ret = TLS_VK_ERR_RUNTIME; goto cleanup;
                }
                if (!vk_nufft1_record_execute(
                        plan, &slot->batch, d_src_s, 0u, d_z_s, 0u,
                        (size_t)b_cnt, VK_NUFFT_ORDER_SLOT_BIT_REVERSED_RAW)) {
                    ret = TLS_VK_ERR_RUNTIME; goto cleanup;
                }
                vk_batch_barrier_c2c(&slot->batch);
                if (!vk_record_ihs_accum(
                        rt->ctx, &slot->batch, prec, df_vk, d_z_s,
                        nll_multi ? d_dec : (VkBufferSlice){0}, d_acc_s, harm == 1, &sp)) {
                    ret = TLS_VK_ERR_RUNTIME; goto cleanup;
                }
            }
            vk_batch_barrier_c2c(&slot->batch);
            if (!vk_record_ihs_finalize(
                    rt->ctx, &slot->batch, prec, df_vk, d_acc_s,
                    nll_multi ? (VkBufferSlice){0} : d_dec, d_pow_s, &sp)) {
                ret = TLS_VK_ERR_RUNTIME; goto cleanup;
            }
        }
    }

    if (!uma_dn) {
        vk_batch_barrier_c2t(&slot->batch);
        vk_batch_copy(&slot->batch, vk_buf_slice(&slot->d_power.buf, 0, out_bytes),
                      vk_buf_slice(&slot->h_power.buf, 0, out_bytes), out_bytes);
    }
    if (!vk_batch_submit_and_wait(&slot->batch)) {
        set_last_error("Vulkan command submission failed during IHS execution");
        ret = TLS_VK_ERR_RUNTIME;
        goto cleanup;
    }

    copy_dev_out_to_host(prec, uma_dn ? slot->d_power.buf.mapped : slot->h_power.buf.mapped, power_out, Nf);

cleanup:
    rt_release_slot(rt, slot_idx);
    return ret;
}

/* ============================================================================
 * Public Engine Entrypoints
 * ============================================================================ */

int vk_engine_device_count(void) {
    const VkDeviceDescriptor* devs = nullptr;
    return (int)vk_enumerate_devices(&devs);
}

int vk_engine_device_info(int device_id, VkCDeviceInfo* out) {
    if (!out) return TLS_VK_ERR_ARGUMENT;
    const VkDeviceDescriptor* devs = nullptr;
    size_t ndevs = vk_enumerate_devices(&devs);
    if (device_id < 0 || (size_t)device_id >= ndevs) return TLS_VK_ERR_DEVICE;
    const VkDeviceDescriptor* d = &devs[device_id];
    memset(out, 0, sizeof(*out));
    out->id = d->id;
    snprintf(out->name, sizeof(out->name), "%.255s", d->name);
    snprintf(out->backend, sizeof(out->backend), "%.31s", d->backend);
    snprintf(out->device_type, sizeof(out->device_type), "%.31s", d->device_type);
    out->max_compute_units = (uint64_t)d->max_compute_units;
    out->global_mem_size = (uint64_t)d->global_mem_size;
    out->is_gpu = d->is_gpu ? 1 : 0;
    out->has_fp64 = d->has_fp64 ? 1 : 0;
    out->has_fp16 = d->has_fp16 ? 1 : 0;
    return TLS_VK_OK;
}

const char* vk_engine_last_error(void) {
    return g_tls_vk_last_err;
}

int vk_engine_run_fastchi2_half(
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power, float* cond) {
    return execute_fastchi2_common(VK_PREC_HALF, t, y_f, y_d, dy_f, dy_d,
                                   M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

int vk_engine_run_fastchi2_float(
    const double* t, const float* y, const float* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power, float* cond) {
    return execute_fastchi2_common(VK_PREC_FLOAT, t, y, nullptr, dy, nullptr,
                                   M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

int vk_engine_run_fastchi2_df(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power, double* cond) {
    return execute_fastchi2_common(VK_PREC_DF, t, nullptr, y, nullptr, dy,
                                   M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

int vk_engine_run_fastchi2_double(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power, double* cond) {
    return execute_fastchi2_common(VK_PREC_DOUBLE, t, nullptr, y, nullptr, dy,
                                   M, f0, df, N, degree, backend, normalization, device_id, power, cond);
}

int vk_engine_run_ihs_half(
    const double* t, const float* y_f, const double* y_d,
    const float* dy_f, const double* dy_d,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power) {
    return execute_ihs_common(VK_PREC_HALF, t, y_f, y_d, dy_f, dy_d,
                              M, f0, df, N, degree, backend, normalization, device_id, power);
}

int vk_engine_run_ihs_float(
    const double* t, const float* y, const float* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    float* power) {
    return execute_ihs_common(VK_PREC_FLOAT, t, y, nullptr, dy, nullptr,
                              M, f0, df, N, degree, backend, normalization, device_id, power);
}

int vk_engine_run_ihs_df(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power) {
    return execute_ihs_common(VK_PREC_DF, t, nullptr, y, nullptr, dy,
                              M, f0, df, N, degree, backend, normalization, device_id, power);
}

int vk_engine_run_ihs_double(
    const double* t, const double* y, const double* dy,
    int M, double f0, double df, int N, int degree,
    int backend, int normalization, int device_id,
    double* power) {
    return execute_ihs_common(VK_PREC_DOUBLE, t, nullptr, y, nullptr, dy,
                              M, f0, df, N, degree, backend, normalization, device_id, power);
}
