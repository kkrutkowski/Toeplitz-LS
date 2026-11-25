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
// Bit-Reversal Shared-Memory Bank-Conflict Models (Whole-FFT XOR & COBRA Tile)
// ============================================================================
namespace detail {

constexpr int bit_reverse_const(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | ((x >> i) & 1);
    }
    return r;
}

} // namespace detail

// Number of low address bits matching one conflict-free bank window for elem_bytes.
constexpr int whole_smem_xor_k(BankModel b, int elem_bytes) {
    if (b.banks <= 1) {
        return 0;
    }
    const int banks_per_elem = (elem_bytes + b.bank_width_bytes - 1) / b.bank_width_bytes;
    const int elem_banks = (b.banks / banks_per_elem > 0) ? (b.banks / banks_per_elem) : 1;
    int k = 0;
    while ((1 << (k + 1)) <= elem_banks) {
        ++k;
    }
    return k;
}

// Zero-pad XOR swizzle for whole-FFT shared-memory bit-reversal.
constexpr int whole_smem_xor_slot(int idx, int logN, int k) {
    if (k <= 0 || logN <= k) {
        return idx;
    }
    const int mask = (1 << k) - 1;
    return idx ^ ((idx >> (logN - k)) & mask);
}

// Evaluate bank-conflict degree for whole-FFT XOR-swizzled write and bit-reversed read.
constexpr int whole_smem_xor_conflicts(BankModel b, int elem_bytes, int logN) {
    const int k = whole_smem_xor_k(b, elem_bytes);
    const int N = 1 << logN;
    const int bases[3] = {0, 32, N >> 1};
    int worst = 1;
    for (int bi = 0; bi < 3; ++bi) {
        const int base = bases[bi];
        if (base + 32 > N) continue;
        const int dw = detail::eval_window_conflicts(b, elem_bytes, 32, [=](int lane) {
            return whole_smem_xor_slot(base + lane, logN, k);
        });
        const int dr = detail::eval_window_conflicts(b, elem_bytes, 32, [=](int lane) {
            const int r = detail::bit_reverse_const(base + lane, logN);
            return whole_smem_xor_slot(r, logN, k);
        });
        if (dw > worst) worst = dw;
        if (dr > worst) worst = dr;
    }
    return worst;
}

// Strided COBRA tile stride (coprime to element bank count) and evaluator.
constexpr int cobra_tile_stride(BankModel b, int elem_bytes, int w) {
    return find_min_conflict_stride2(b, elem_bytes, 1 << w);
}

constexpr int cobra_tile_conflicts(BankModel b, int elem_bytes, int w, int stride) {
    const int W = 1 << w;
    const int dw = detail::eval_window_conflicts(b, elem_bytes, W, [=](int lane) {
        return lane;
    });
    const int dr = detail::eval_window_conflicts(b, elem_bytes, W, [=](int lane) {
        const int r = detail::bit_reverse_const(lane, w);
        return r * stride;
    });
    return (dw > dr) ? dw : dr;
}

// Zero-padding conflict-free XOR swizzle for W x W COBRA tiles (w = 4 or 5).
// Conflict degree == 1 for both coalesced row writes and bit-reversed reads
// across 4B, 8B, and 16B elements on both 32x4 and 16x4 bank models.
constexpr int cobra_tile_xor_slot(int a, int c, int w) {
    if (w == 4) {
        return (a << 4) + (c ^ a ^ (a >> 3) ^ ((a >> 1) & 2));
    }
    if (w == 5) {
        return (a << 5) + (c ^ a ^ (a >> 4) ^ ((a >> 2) & 2));
    }
    return (a << w) + c;
}

constexpr int cobra_tile_xor_conflicts(BankModel b, int elem_bytes, int w) {
    const int W = 1 << w;
    const int rows_per_warp = (32 >= W) ? (32 / W) : 1;
    int worst = 1;
    for (int row0 = 0; row0 < W; row0 += rows_per_warp) {
        const int dw = detail::eval_window_conflicts(b, elem_bytes, 32, [=](int lane) {
            const int a = row0 + (lane >> w);
            const int c = lane & (W - 1);
            return cobra_tile_xor_slot(a, c, w);
        });
        const int dr = detail::eval_window_conflicts(b, elem_bytes, 32, [=](int lane) {
            const int a_out = row0 + (lane >> w);
            const int c_out = lane & (W - 1);
            const int a_src = detail::bit_reverse_const(c_out, w);
            const int c_src = detail::bit_reverse_const(a_out, w);
            return cobra_tile_xor_slot(a_src, c_src, w);
        });
        if (dw > worst) worst = dw;
        if (dr > worst) worst = dr;
    }
    return worst;
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


// Bit-reversal Whole-FFT XOR swizzle: 4B, 8B, 16B on 32x4 and 16x4 for logN=10..14
static_assert(whole_smem_xor_conflicts(kBank32x4, 4, 10) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 4, 12) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 4, 14) == 1,
              "Whole-FFT XOR bit-rev 4B must be conflict-free on 32x4");
static_assert(whole_smem_xor_conflicts(kBank32x4, 8, 10) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 8, 12) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 8, 13) == 1,
              "Whole-FFT XOR bit-rev 8B must be conflict-free on 32x4");
static_assert(whole_smem_xor_conflicts(kBank32x4, 16, 10) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 16, 11) == 1 &&
              whole_smem_xor_conflicts(kBank32x4, 16, 12) == 1,
              "Whole-FFT XOR bit-rev 16B must be conflict-free on 32x4");
static_assert(whole_smem_xor_conflicts(kBank16x4, 4, 10) == 1 &&
              whole_smem_xor_conflicts(kBank16x4, 8, 10) == 1 &&
              whole_smem_xor_conflicts(kBank16x4, 16, 10) == 1,
              "Whole-FFT XOR bit-rev must be conflict-free on 16x4");

// Bit-reversal COBRA tile: strided (when W <= elem_banks) and zero-pad XOR (all sizes)
static_assert(cobra_tile_conflicts(kBank32x4, 4, 4, cobra_tile_stride(kBank32x4, 4, 4)) == 1 &&
              cobra_tile_conflicts(kBank32x4, 8, 4, cobra_tile_stride(kBank32x4, 8, 4)) == 1 &&
              cobra_tile_conflicts(kBank32x4, 4, 5, cobra_tile_stride(kBank32x4, 4, 5)) == 1 &&
              cobra_tile_conflicts(kBank16x4, 4, 4, cobra_tile_stride(kBank16x4, 4, 4)) == 1,
              "COBRA strided tile must be conflict-free whenever W <= elem_banks");
static_assert(cobra_tile_xor_conflicts(kBank32x4, 4, 4) == 1 &&
              cobra_tile_xor_conflicts(kBank32x4, 8, 4) == 1 &&
              cobra_tile_xor_conflicts(kBank32x4, 16, 4) == 1,
              "COBRA zero-pad XOR tile w=4 must be conflict-free on 32x4");
static_assert(cobra_tile_xor_conflicts(kBank32x4, 4, 5) == 1 &&
              cobra_tile_xor_conflicts(kBank32x4, 8, 5) == 1 &&
              cobra_tile_xor_conflicts(kBank32x4, 16, 5) == 1,
              "COBRA zero-pad XOR tile w=5 must be conflict-free on 32x4");
static_assert(cobra_tile_xor_conflicts(kBank16x4, 4, 4) == 1 &&
              cobra_tile_xor_conflicts(kBank16x4, 8, 4) == 1 &&
              cobra_tile_xor_conflicts(kBank16x4, 16, 4) == 1 &&
              cobra_tile_xor_conflicts(kBank16x4, 4, 5) == 1 &&
              cobra_tile_xor_conflicts(kBank16x4, 8, 5) == 1 &&
              cobra_tile_xor_conflicts(kBank16x4, 16, 5) == 1,
              "COBRA zero-pad XOR tile w=4,5 must be conflict-free on 16x4");
} // namespace layout_checks

} // namespace sycl_fft
