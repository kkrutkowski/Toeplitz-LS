/* Plain C23 Vulkan Subgrid Phase-Modulation Recorder. */
#include "common/math.h"
#include "periodogram/vk_subgrid.h"

typedef struct VkSubgridDfPush {
    float f0_hi, f0_lo;
    float step_hi, step_lo;
    int32_t M;
    int32_t b_count;
    int32_t strip_b;
    int32_t has_yw;
    uint32_t x_off;
    uint32_t w_off;
    uint32_t yw_off;
} VkSubgridDfPush;

typedef struct VkSubgridDPush {
    double f0_base;
    double f_step;
    int32_t M;
    int32_t b_count;
    int32_t strip_b;
    int32_t has_yw;
    uint32_t x_off;
    uint32_t w_off;
    uint32_t yw_off;
    uint32_t _pad;
} VkSubgridDPush;

static_assert(sizeof(VkSubgridDfPush) == 44);
static_assert(sizeof(VkSubgridDPush) == 48);

bool vk_record_build_subgrid_sources(VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, bool df_use_veltkamp, VkBufferSlice d_x, uint32_t x_off_elems,
                                     VkBufferSlice d_w, uint32_t w_off_elems, VkBufferSlice d_yw, uint32_t yw_off_elems, bool has_yw, VkBufferSlice d_src,
                                     int M, int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size) {
    if (M <= 0 || b_count <= 0) return true;

    uint32_t wg_m = (uint32_t)(wg_size > 0 ? wg_size : 64u);
    if (wg_m > 256u) wg_m = 256u;
    int strip_b = (b_count >= 4) ? 8 : 1;
    uint32_t gx = ((uint32_t)M + wg_m - 1u) / wg_m;
    uint32_t gy = (uint32_t)((b_count + strip_b - 1) / strip_b);

    VkShaderId sid = VK_SHADER_COUNT;
    switch (prec) {
        case VK_PREC_HALF:
            sid = VK_SHADER_subgrid_h;
            break;
        case VK_PREC_FLOAT:
            sid = VK_SHADER_subgrid_f;
            break;
        case VK_PREC_DF:
            sid = VK_SHADER_subgrid_df;
            break;
        case VK_PREC_DOUBLE:
            sid = VK_SHADER_subgrid_d;
            break;
    }

    VkSpecConstEntry specs[2] = {
        {0u, wg_m},
        {15u, df_use_veltkamp ? 1u : 0u},
    };
    size_t nspec = (prec == VK_PREC_DOUBLE) ? 1u : 2u;

    VkBufferSlice binds[4] = {
        d_x,
        d_w,
        (has_yw && vk_slice_valid(d_yw)) ? d_yw : vk_ctx_dummy_buffer(ctx),
        d_src,
    };

    double f0_base = q_f0 + (double)b_start * q_block_df;
    if (prec == VK_PREC_DOUBLE) {
        VkSubgridDPush pc = {f0_base, q_block_df, M, b_count, strip_b, has_yw ? 1 : 0, x_off_elems, w_off_elems, yw_off_elems, 0u};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(ctx, sid, 4u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, binds, 4u, &pc, sizeof(pc), gx, gy, 1u);
    } else {
        VkDf fb = vk_df_from_double(f0_base);
        VkDf fs = vk_df_from_double(q_block_df);
        VkSubgridDfPush pc = {fb.hi, fb.lo, fs.hi, fs.lo, M, b_count, strip_b, has_yw ? 1 : 0, x_off_elems, w_off_elems, yw_off_elems};
        const VkCompiledPipeline* p = vk_ctx_get_pipeline(ctx, sid, 4u, sizeof(pc), specs, nspec, 0u);
        if (!p) return false;
        return vk_batch_dispatch(batch, p, binds, 4u, &pc, sizeof(pc), gx, gy, 1u);
    }
}
