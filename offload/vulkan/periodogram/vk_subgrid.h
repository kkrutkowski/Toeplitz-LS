#ifndef TLS_VK_SUBGRID_H
#define TLS_VK_SUBGRID_H

/* Plain C23 Vulkan On-the-Fly Subgrid Phase-Modulation Recorder (K_b = 8 strip-recurrence). */

#include <stddef.h>
#include <stdint.h>

#include "common/vk_context.h"

bool vk_record_build_subgrid_sources(VkDeviceContext* ctx, VkCmdBatch* batch, VkPrecKind prec, bool df_use_veltkamp, VkBufferSlice d_x, uint32_t x_off_elems,
                                     VkBufferSlice d_w, uint32_t w_off_elems, VkBufferSlice d_yw, uint32_t yw_off_elems, bool has_yw, VkBufferSlice d_src,
                                     int M, int b_start, int b_count, double q_f0, double q_block_df, size_t wg_size);

#endif /* TLS_VK_SUBGRID_H */
