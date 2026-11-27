#ifndef TLS_VK_LAYOUT_H
#define TLS_VK_LAYOUT_H

/* Plain C11 Shared-Memory Bank Model & Conflict-Free Layout Solver. */

#include <stdint.h>

typedef struct VkBankModel {
    int banks;
    int bank_width_bytes;
} VkBankModel;

typedef enum VkBankModelKind { VK_BANK_32x4 = 0, VK_BANK_16x4 = 1, VK_BANK_32x8 = 2, VK_BANK_NONE = 3 } VkBankModelKind;

typedef enum VkStagingPolicy { VK_STAGING_SPLIT = 0, VK_STAGING_SEQ = 1 } VkStagingPolicy;

typedef enum VkT1LayoutPolicy { VK_T1_AFFINE = 0, VK_T1_XOR = 1 } VkT1LayoutPolicy;

typedef enum VkPhase3Policy { VK_PHASE3_SHUFFLE = 0, VK_PHASE3_2T_XOR = 1, VK_PHASE3_2T_STRIDE = 2 } VkPhase3Policy;

static inline VkBankModel vk_bank_model_from_kind(VkBankModelKind kind) {
    switch (kind) {
        case VK_BANK_32x4:
            return (VkBankModel){32, 4};
        case VK_BANK_16x4:
            return (VkBankModel){16, 4};
        case VK_BANK_32x8:
            return (VkBankModel){32, 8};
        case VK_BANK_NONE:
            return (VkBankModel){1, 1};
    }
    return (VkBankModel){32, 4};
}

static inline const char* vk_bank_model_name(VkBankModelKind kind) {
    switch (kind) {
        case VK_BANK_32x4:
            return "32x4";
        case VK_BANK_16x4:
            return "16x4";
        case VK_BANK_32x8:
            return "32x8";
        case VK_BANK_NONE:
            return "none";
    }
    return "32x4";
}

static inline int vk_gcd_int(int a, int b) {
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return (a < 0) ? -a : a;
}

static inline int vk_eval_bank_conflicts(VkBankModel b, int elem_bytes, int total_lanes, const int* elem_indices) {
    if (b.banks <= 1) return 1;
    int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    int window = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    int lanes = (total_lanes < 32) ? total_lanes : 32;
    int worst = 1;

    for (int phase_start = 0; phase_start < lanes; phase_start += window) {
        int phase_end = (phase_start + window < lanes) ? (phase_start + window) : lanes;
        int seen_elems[32];
        int n_elems = 0;
        int bank_hits[64];
        for (int i = 0; i < 64; ++i) bank_hits[i] = 0;

        for (int lane = phase_start; lane < phase_end; ++lane) {
            int elem_idx = elem_indices[lane];
            int dup = 0;
            for (int j = 0; j < n_elems; ++j) {
                if (seen_elems[j] == elem_idx) {
                    dup = 1;
                    break;
                }
            }
            if (!dup && n_elems < 32) {
                seen_elems[n_elems++] = elem_idx;
                for (int sb = 0; sb < banks_per_elem; ++sb) {
                    int bank = (elem_idx * banks_per_elem + sb) % b.banks;
                    if (bank < 0) bank += b.banks;
                    if (bank < 64) {
                        ++bank_hits[bank];
                        if (bank_hits[bank] > worst) worst = bank_hits[bank];
                    }
                }
            }
        }
    }
    return worst;
}

static inline int vk_t1_affine_conflicts(VkBankModel b, int elem_bytes, int T, int W, int row_stride) {
    int w_idx[32], r_idx[32];
    int lanes = (T < 32) ? T : 32;
    for (int t = 0; t < lanes; ++t) {
        w_idx[t] = t;
        r_idx[t] = (t / W) * row_stride + (t % W);
    }
    int dw = vk_eval_bank_conflicts(b, elem_bytes, T, w_idx);
    int dr = vk_eval_bank_conflicts(b, elem_bytes, T, r_idx);
    return (dw > dr) ? dw : dr;
}

static inline int vk_t2_stride_conflicts(VkBankModel b, int elem_bytes, int T, int W, int row_stride, int stride2) {
    int w_idx[32], r_idx[32];
    int lanes = (T < 32) ? T : 32;
    for (int t = 0; t < lanes; ++t) {
        int sub = t / W;
        int l = t % W;
        w_idx[t] = sub * row_stride + l;
        r_idx[t] = sub * row_stride + l * stride2;
    }
    int dw = vk_eval_bank_conflicts(b, elem_bytes, T, w_idx);
    int dr = vk_eval_bank_conflicts(b, elem_bytes, T, r_idx);
    return (dw > dr) ? dw : dr;
}

static inline int vk_find_min_conflict_stride2(VkBankModel b, int elem_bytes, int W) {
    if (b.banks <= 1) return W;
    int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    int elem_banks = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    for (int s = W; s <= W + elem_banks; ++s) {
        if (vk_gcd_int(s, elem_banks) == 1) return s;
    }
    return W + 1;
}

static inline int vk_find_min_conflict_row_stride(VkBankModel b, int elem_bytes, int T, int W, int min_row, int stride2, int chunk) {
    if (b.banks <= 1) return min_row;
    int best_stride = min_row;
    int best_deg = 1000000;
    for (int pad = 0; pad <= b.banks; ++pad) {
        int rs = min_row + pad;
        int deg = vk_t1_affine_conflicts(b, elem_bytes, T, W, rs);
        if (stride2 > 0 && chunk > 0) {
            int d2 = vk_t2_stride_conflicts(b, elem_bytes, T, W, rs, stride2);
            if (d2 > deg) deg = d2;
        }
        if (deg < best_deg) {
            best_deg = deg;
            best_stride = rs;
            if (best_deg == 1) break;
        }
    }
    return best_stride;
}

static inline int vk_whole_smem_xor_k(VkBankModel b, int elem_bytes) {
    if (b.banks <= 1) return 0;
    int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    int elem_banks = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    int k = 0;
    while ((1 << (k + 1)) <= elem_banks) ++k;
    return k;
}

#endif /* TLS_VK_LAYOUT_H */
