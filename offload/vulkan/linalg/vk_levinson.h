#ifndef TLS_VK_LEVINSON_H
#define TLS_VK_LEVINSON_H

/* Plain C23 Vulkan Fused GLS (d=1), Levinson-Durbin (d>=2), and IHS Solver Recorders. */

#include "common/vk_context.h"
#include "nufft/pswf_coeffs.h"
#include "periodogram/vk_preprocess.h"

#include <stddef.h>
#include <stdint.h>

typedef struct VkFastChi2SolveParams {
    int            degree;
    int            M;
    int            Nf;
    int            nfft;
    int            log_nfft;
    int            nout;
    int            output_shift;
    int            b_start;
    int            b_count;
    VkPswfMode     mode;
    VkNormKind     norm;
    double         ws_fft;
    double         yws_fft;
    double         chi2_ref;
    double         b_param;
    VkCybenkoModel cyb;
    size_t         wg_size;
} VkFastChi2SolveParams;

typedef struct VkIhsSolveParams {
    int        degree;
    int        Nf;
    int        nfft;
    int        log_nfft;
    int        nout;
    int        output_shift;
    int        b_start;
    int        b_count;
    VkPswfMode mode;
    VkNormKind norm;
    double     standard_scale;
    double     nll_scale;
    double     n_eff;
    size_t     wg_size;
} VkIhsSolveParams;

bool vk_record_fastchi2_solve(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkBufferSlice d_trig,
    VkBufferSlice d_slot_phi_hat,
    VkBufferSlice d_slot_deconv,
    VkBufferSlice d_a_scratch,
    VkBufferSlice d_power,
    VkBufferSlice d_cond,
    bool has_cond,
    bool use_raw_deconv,
    const VkFastChi2SolveParams* p);

bool vk_record_ihs_single(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkBufferSlice d_z,
    VkBufferSlice d_slot_deconv,
    VkBufferSlice d_power,
    const VkIhsSolveParams* p);

bool vk_record_ihs_accum(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkBufferSlice d_z,
    VkBufferSlice d_slot_deconv,
    VkBufferSlice d_acc,
    bool first_harmonic,
    const VkIhsSolveParams* p);

bool vk_record_ihs_finalize(
    VkDeviceContext* ctx,
    VkCmdBatch* batch,
    VkPrecKind prec,
    bool df_use_veltkamp,
    VkBufferSlice d_acc,
    VkBufferSlice d_slot_deconv,
    VkBufferSlice d_power,
    const VkIhsSolveParams* p);

#endif /* TLS_VK_LEVINSON_H */
