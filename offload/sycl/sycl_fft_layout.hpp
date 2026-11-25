#pragma once

#include <cstdint>

namespace sycl_fft {

// ============================================================================
// Compile-Time Shared / Local Memory Bank Model & Conflict-Free Layout Solver
//   Models GPU shared/local memory bank geometry {banks, bank_width_bytes} and
//   evaluates physical per-phase bank conflict degree for affine and XOR
//   transpose maps at compile time (C++17 compatible).
// ============================================================================
struct BankModel {
    int banks;             // number of shared/local memory banks (1 = bankless/CPU)
    int bank_width_bytes;  // bytes per bank (typically 4; 8 for 64-bit bank mode)

    constexpr bool operator==(const BankModel& o) const {
        return banks == o.banks && bank_width_bytes == o.bank_width_bytes;
    }
    constexpr bool operator!=(const BankModel& o) const { return !(*this == o); }
};

inline constexpr BankModel kBank32x4{32, 4}; // NVIDIA Volta-Blackwell, AMD RDNA/CDNA, Intel Xe-HPG/PVC
inline constexpr BankModel kBank16x4{16, 4}; // Intel Gen9/Gen11 16-bank SLM / mobile GPUs
inline constexpr BankModel kBank32x8{32, 8}; // 64-bit wide-bank GPUs
inline constexpr BankModel kBankNone{1, 1};  // CPU / emulated local memory / bankless

enum class BankModelKind : uint8_t {
    B32x4 = 0,
    B16x4 = 1,
    B32x8 = 2,
    None  = 3
};

constexpr BankModel bank_model_from_kind(BankModelKind kind) {
    switch (kind) {
        case BankModelKind::B32x4: return kBank32x4;
        case BankModelKind::B16x4: return kBank16x4;
        case BankModelKind::B32x8: return kBank32x8;
        case BankModelKind::None:  return kBankNone;
    }
    return kBank32x4;
}

inline const char* bank_model_name(BankModelKind kind) {
    switch (kind) {
        case BankModelKind::B32x4: return "32x4";
        case BankModelKind::B16x4: return "16x4";
        case BankModelKind::B32x8: return "32x8";
        case BankModelKind::None:  return "none";
    }
    return "32x4";
}

// ============================================================================
// Single-Pass Kernel Policy Enums (shared across all precisions)
// ============================================================================
enum class StagingPolicy : uint8_t {
    Split = 0, // Two local buffers (smem_re, smem_im), 1 barrier per transpose
    Seq   = 1  // Single local buffer reused sequentially for re then im
};

enum class T1LayoutPolicy : uint8_t {
    Affine = 0, // Strided rows: M * RowStride + t
    XorT1  = 1  // Zero-pad XOR swizzle: M * T + ((sub ^ M) << LogW) + l
};

enum class Phase3Policy : uint8_t {
    SubgroupShuffle    = 0, // Warp/sub-group XOR shuffle DIF (requires sg_size >= W)
    TwoTransposeXor    = 1, // In-smem XOR transpose + reg_base<LogW> + coalesced store (Chunk == W, req_sg = 1)
    TwoTransposeStride = 2  // In-smem strided transpose + reg_base<LogW> + coalesced store (Chunk >= W, req_sg = 1)
};

namespace detail {

constexpr int gcd_int(int a, int b) {
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return (a < 0) ? -a : a;
}

// Evaluate maximum bank conflict degree across all hardware issue phases for a
// warp/wavefront window of min(32, total_lanes) lanes.
// Identical word addresses broadcast cleanly (do not conflict).
template <typename IdxFn>
constexpr int eval_window_conflicts(BankModel b, int elem_bytes, int total_lanes, IdxFn idx_fn) {
    if (b.banks <= 1) {
        return 1;
    }
    const int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    const int window = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    const int lanes = (total_lanes < 32) ? total_lanes : 32;
    int worst = 1;

    for (int phase_start = 0; phase_start < lanes; phase_start += window) {
        const int phase_end = (phase_start + window < lanes) ? (phase_start + window) : lanes;
        int seen_elems[32] = {};
        int n_elems = 0;
        int bank_hits[64] = {};

        for (int lane = phase_start; lane < phase_end; ++lane) {
            const int elem_idx = idx_fn(lane);
            bool dup = false;
            for (int j = 0; j < n_elems; ++j) {
                if (seen_elems[j] == elem_idx) {
                    dup = true;
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

} // namespace detail

// Transpose 1 affine write (M * row_stride + t) and read ((t / W) * row_stride + M * W + (t % W)).
// Constant additive shifts (M * row_stride, M * W) permute banks bijectively, so M = 0 is exact.
constexpr int t1_affine_conflicts(BankModel b, int elem_bytes, int T, int W, int row_stride) {
    const int dw = detail::eval_window_conflicts(
        b, elem_bytes, T, [=](int t) { return t; });
    const int dr = detail::eval_window_conflicts(
        b, elem_bytes, T, [=](int t) { return (t / W) * row_stride + (t % W); });
    return (dw > dr) ? dw : dr;
}

// Transpose 1 zero-pad XOR write/read (used when W < Chunk).
constexpr int t1_xor_conflicts(BankModel b, int elem_bytes, int T, int W, int Chunk) {
    const int check_m = (Chunk < 8) ? Chunk : 8;
    int worst = 1;
    for (int m = 0; m < check_m; ++m) {
        const int dw = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
            const int sub = t / W;
            const int l = t % W;
            return m * T + ((sub ^ m) * W) + l;
        });
        const int dr = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
            const int sub = t / W;
            const int l = t % W;
            return sub * T + ((m ^ sub) * W) + l;
        });
        if (dw > worst) worst = dw;
        if (dr > worst) worst = dr;
    }
    return worst;
}

// Transpose 2/3 XOR write/read (used when Chunk == W).
constexpr int t2_xor_conflicts(BankModel b, int elem_bytes, int T, int W) {
    const int check = (W < 8) ? W : 8;
    int worst = 1;
    for (int m = 0; m < check; ++m) {
        const int dw = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
            const int sub = t / W;
            const int l = t % W;
            return sub * T + m * W + (l ^ m);
        });
        const int dr = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
            const int sub = t / W;
            const int l = t % W;
            return sub * T + l * W + (m ^ l);
        });
        if (dw > worst) worst = dw;
        if (dr > worst) worst = dr;
    }
    return worst;
}

// Transpose 2/3 strided write/read (used when Chunk >= W).
// Additive offsets (M * stride2, g * W * stride2 + K) are bijective mod banks, so M=0, g=0, K=0 is exact.
constexpr int t2_stride_conflicts(BankModel b, int elem_bytes, int T, int W, int Chunk, int row_stride, int stride2) {
    (void)Chunk;
    const int dw = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
        const int sub = t / W;
        const int l = t % W;
        return sub * row_stride + l;
    });
    const int dr = detail::eval_window_conflicts(b, elem_bytes, T, [=](int t) {
        const int sub = t / W;
        const int l = t % W;
        return sub * row_stride + l * stride2;
    });
    return (dw > dr) ? dw : dr;
}

// Find smallest inner stride >= W coprime to the per-phase element bank count.
constexpr int find_min_conflict_stride2(BankModel b, int elem_bytes, int W) {
    if (b.banks <= 1) {
        return W;
    }
    const int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    const int elem_banks = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    for (int s = W; s <= W + elem_banks; ++s) {
        if (detail::gcd_int(s, elem_banks) == 1) {
            return s;
        }
    }
    return W + 1;
}

// Find smallest row_stride >= min_row achieving conflict degree 1 (or minimum achievable)
// for Transpose 1 affine access and (when stride2 > 0) Transpose 2 strided access.
constexpr int find_min_conflict_row_stride(
    BankModel b, int elem_bytes, int T, int W, int min_row, int stride2 = 0, int Chunk = 0) {
    if (b.banks <= 1) {
        return min_row;
    }
    int best_stride = min_row;
    int best_deg = 1000000;
    for (int pad = 0; pad <= b.banks; ++pad) {
        const int rs = min_row + pad;
        int deg = t1_affine_conflicts(b, elem_bytes, T, W, rs);
        if (stride2 > 0 && Chunk > 0) {
            const int d2 = t2_stride_conflicts(b, elem_bytes, T, W, Chunk, rs, stride2);
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

// ============================================================================
// Compile-Time Regression Suite: Verify Existing NVIDIA 32x4 & Intel 16x4 Layouts
// ============================================================================
namespace layout_checks {

// FP32 split_smem on 32x4
static_assert(t1_affine_conflicts(kBank32x4, 4, 128, 16, 144) == 1,
              "FP32 logN=10 E=8 T=128 W=16 row_stride=144 must be conflict-free on 32x4");
static_assert(find_min_conflict_row_stride(kBank32x4, 4, 128, 16, 128) == 144,
              "Derived row_stride for FP32 T=128 W=16 must match 144");
static_assert(find_min_conflict_row_stride(kBank32x4, 4, 256, 32, 256) == 256,
              "Derived row_stride for FP32 T=256 W=32 must be unpadded 256");
static_assert(find_min_conflict_row_stride(kBank32x4, 4, 256, 16, 256) == 272,
              "Derived row_stride for FP32 T=256 W=16 must match 272");

// Improved compact conflict-free padding for narrow W=4 and W=8 (works on both 32x4 and 16x4)
static_assert(find_min_conflict_row_stride(kBank32x4, 4, 64, 4, 64) == 68,
              "Derived row_stride for FP32 T=64 W=4 must be 68 (pad=4)");
static_assert(t1_affine_conflicts(kBank16x4, 4, 64, 4, 68) == 1,
              "FP32 T=64 W=4 row_stride=68 must also be conflict-free on 16x4");
static_assert(find_min_conflict_row_stride(kBank32x4, 4, 128, 8, 128) == 136,
              "Derived row_stride for FP32 T=128 W=8 must be 136 (pad=8)");
static_assert(t1_affine_conflicts(kBank16x4, 4, 128, 8, 136) == 1,
              "FP32 T=128 W=8 row_stride=136 must also be conflict-free on 16x4");

// Wide (8B) 1024 XOR T1
static_assert(t1_xor_conflicts(kBank32x4, 8, 64, 4, 16) == 1,
              "Wide logN=10 XOR T1 must be conflict-free on 32x4");
static_assert(t1_xor_conflicts(kBank16x4, 8, 64, 4, 16) == 1,
              "Wide logN=10 XOR T1 must be conflict-free on 16x4");

// Wide (8B) 2048 stride-9 / row_stride=152
static_assert(find_min_conflict_stride2(kBank32x4, 8, 8) == 9,
              "Wide logN=11 Stride2 must derive to 9");
static_assert(find_min_conflict_row_stride(kBank32x4, 8, 128, 8, 144, 9, 16) == 152,
              "Wide logN=11 RowStride must derive to 152");
static_assert(t2_stride_conflicts(kBank32x4, 8, 128, 8, 16, 152, 9) == 1,
              "Wide logN=11 T2 stride-9 must be conflict-free on 32x4");
static_assert(t2_stride_conflicts(kBank16x4, 8, 128, 8, 16, 152, 9) == 1,
              "Wide logN=11 T2 stride-9 must be conflict-free on 16x4");

// Wide (8B) 4096 / 8192 XOR T2
static_assert(t2_xor_conflicts(kBank32x4, 8, 256, 16) == 1,
              "Wide logN=12/13 XOR T2 must be conflict-free on 32x4");
static_assert(t2_xor_conflicts(kBank16x4, 8, 256, 16) == 1,
              "Wide logN=12/13 XOR T2 must be conflict-free on 16x4");

// 32x8 wide-bank model derivation sanity
static_assert(find_min_conflict_stride2(kBank32x8, 8, 8) == 9,
              "Wide logN=11 Stride2 on 32x8 must derive to 9");
static_assert(t2_stride_conflicts(kBank32x8, 8, 128, 8, 16,
                  find_min_conflict_row_stride(kBank32x8, 8, 128, 8, 144, 9, 16), 9) == 1,
              "Wide logN=11 on 32x8 must derive conflict-free row_stride");

} // namespace layout_checks

} // namespace sycl_fft
