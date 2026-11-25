// Hardware-Generalized Multi-Strategy Bit-Reversal Engine Implementation.
// Deduplicated by raw element size (4B uint32_t, 8B uint64_t, 16B sycl::uint4)
// into a single isolated HCF module:
//   1. whole_smem_xor_w256 / _w128    : 1 WG/FFT zero-pad XOR-swizzled smem
//   2. cobra_w5_lut_wg256 / wg256/128 : 32x32 zero-pad conflict-free XOR smem COBRA
//   3. cobra_w4_lut_wg256 / wg256/128 : 16x16 zero-pad conflict-free XOR smem COBRA
//   4. scalar_cobra_w4                : zero-smem private L1-tile COBRA (CPU fallback)
//   5. lut_pairs_w256 / lut_idx_w256  : precomputed compact pair/index device LUT
//   6. naive_w256 / naive_w128        : direct global swap / gather baseline
#include "bitrev/bitrev.hpp"
#include "common/math.hpp"
#include "common/layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace sycl_fft {

struct BitRevLutState {
    sycl::queue q;
    int logN = 0;

    uint16_t* pairs16 = nullptr;
    uint32_t* pairs32 = nullptr;
    size_t npairs = 0;

    uint16_t* idx16 = nullptr;
    uint32_t* idx32 = nullptr;

    uint16_t* rev_w[2] = {nullptr, nullptr};
    uint16_t* b_pairs[2] = {nullptr, nullptr};
    size_t nb_pairs[2] = {0, 0};
    uint16_t* b_all_rev[2] = {nullptr, nullptr};

    BitRevLutState(sycl::queue queue, int log_n) : q(std::move(queue)), logN(log_n) {}

    BitRevLutState(const BitRevLutState&) = delete;
    BitRevLutState& operator=(const BitRevLutState&) = delete;

    void release() {
        auto free_ptr = [&](auto*& p) {
            if (p) {
                sycl::free(p, q);
                p = nullptr;
            }
        };
        free_ptr(pairs16);
        free_ptr(pairs32);
        npairs = 0;
        free_ptr(idx16);
        free_ptr(idx32);
        for (int wi = 0; wi < 2; ++wi) {
            free_ptr(rev_w[wi]);
            free_ptr(b_pairs[wi]);
            nb_pairs[wi] = 0;
            free_ptr(b_all_rev[wi]);
        }
    }

    ~BitRevLutState() { release(); }

    void ensure(BitRevLutKind kind) {
        if (kind == BitRevLutKind::None || logN <= 0) return;
        const size_t N = size_t{1} << logN;

        if (kind == BitRevLutKind::Pairs) {
            if (pairs16 || pairs32) return;
            if (logN <= 16) {
                std::vector<uint16_t> host;
                host.reserve(N);
                for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
                    const uint32_t j = bit_reverse(i, logN);
                    if (i < j) {
                        host.push_back(static_cast<uint16_t>(i));
                        host.push_back(static_cast<uint16_t>(j));
                    }
                }
                npairs = host.size() / 2;
                pairs16 = sycl::malloc_device<uint16_t>(host.size(), q);
                if (!pairs16) throw std::bad_alloc();
                q.memcpy(pairs16, host.data(), host.size() * sizeof(uint16_t)).wait();
            } else {
                std::vector<uint32_t> host;
                host.reserve(N);
                for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
                    const uint32_t j = bit_reverse(i, logN);
                    if (i < j) {
                        host.push_back(i);
                        host.push_back(j);
                    }
                }
                npairs = host.size() / 2;
                pairs32 = sycl::malloc_device<uint32_t>(host.size(), q);
                if (!pairs32) throw std::bad_alloc();
                q.memcpy(pairs32, host.data(), host.size() * sizeof(uint32_t)).wait();
            }
            return;
        }

        if (kind == BitRevLutKind::Idx) {
            if (idx16 || idx32) return;
            if (logN <= 16) {
                std::vector<uint16_t> host(N);
                for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
                    host[i] = static_cast<uint16_t>(bit_reverse(i, logN));
                }
                idx16 = sycl::malloc_device<uint16_t>(N, q);
                if (!idx16) throw std::bad_alloc();
                q.memcpy(idx16, host.data(), N * sizeof(uint16_t)).wait();
            } else {
                std::vector<uint32_t> host(N);
                for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
                    host[i] = bit_reverse(i, logN);
                }
                idx32 = sycl::malloc_device<uint32_t>(N, q);
                if (!idx32) throw std::bad_alloc();
                q.memcpy(idx32, host.data(), N * sizeof(uint32_t)).wait();
            }
            return;
        }

        if (kind == BitRevLutKind::CobraW4 || kind == BitRevLutKind::CobraW5) {
            const int w = (kind == BitRevLutKind::CobraW4) ? 4 : 5;
            const int wi = w - 4;
            if (rev_w[wi] && b_pairs[wi] && b_all_rev[wi]) return;
            const int m = logN - 2 * w;
            if (m < 0 || m > 16) return;
            const uint32_t W = 1u << w;
            const uint32_t num_b = 1u << m;

            std::vector<uint16_t> h_rev_w(W);
            for (uint32_t x = 0; x < W; ++x) {
                h_rev_w[x] = static_cast<uint16_t>(bit_reverse(x, w));
            }

            std::vector<uint16_t> h_b_all(num_b);
            std::vector<uint16_t> h_b_pairs;
            h_b_pairs.reserve(num_b + 64);
            for (uint32_t b = 0; b < num_b; ++b) {
                const uint32_t b_rev = (m == 0) ? 0u : bit_reverse(b, m);
                h_b_all[b] = static_cast<uint16_t>(b_rev);
                if (b <= b_rev) {
                    h_b_pairs.push_back(static_cast<uint16_t>(b));
                    h_b_pairs.push_back(static_cast<uint16_t>(b_rev));
                }
            }
            nb_pairs[wi] = h_b_pairs.size() / 2;

            rev_w[wi] = sycl::malloc_device<uint16_t>(W, q);
            b_pairs[wi] = sycl::malloc_device<uint16_t>(h_b_pairs.size(), q);
            b_all_rev[wi] = sycl::malloc_device<uint16_t>(num_b, q);
            if (!rev_w[wi] || !b_pairs[wi] || !b_all_rev[wi]) throw std::bad_alloc();

            q.memcpy(rev_w[wi], h_rev_w.data(), W * sizeof(uint16_t)).wait();
            q.memcpy(b_pairs[wi], h_b_pairs.data(), h_b_pairs.size() * sizeof(uint16_t)).wait();
            q.memcpy(b_all_rev[wi], h_b_all.data(), num_b * sizeof(uint16_t)).wait();
        }
    }
};

// ============================================================================
// Strategy 1: Naive Direct Global Swap (In-Place) / Gather (Out-Of-Place)
// ============================================================================
template <typename RawT>
struct BitRevNaiveFunctor {
    const RawT* src;
    RawT* dst;
    size_t total;
    int logN;
    bool in_place;

    void operator()(sycl::nd_item<1> item) const {
        const size_t gid = item.get_global_linear_id();
        if (gid >= total) return;
        const size_t mask = (size_t{1} << logN) - 1;
        const size_t batch_off = gid & ~mask;
        const uint32_t idx = static_cast<uint32_t>(gid & mask);
        const uint32_t rev = bit_reverse(idx, logN);
        if (in_place) {
            if (idx < rev) {
                const RawT a = dst[batch_off + idx];
                const RawT b = dst[batch_off + rev];
                dst[batch_off + idx] = b;
                dst[batch_off + rev] = a;
            }
        } else {
            dst[batch_off + idx] = src[batch_off + rev];
        }
    }
};

// ============================================================================
// Strategy 2: Precomputed Device LUT (Compact Pairs In-Place / Index Out-Of-Place)
// ============================================================================
template <typename RawT, typename IdxT>
struct BitRevLutPairsFunctor {
    RawT* buf;
    const IdxT* pairs;
    size_t npairs;
    size_t total_pairs;
    int logN;

    void operator()(sycl::nd_item<1> item) const {
        const size_t gid = item.get_global_linear_id();
        if (gid >= total_pairs) return;
        const size_t batch = gid / npairs;
        const size_t p = gid - batch * npairs;
        const size_t base = batch << logN;
        const size_t i = static_cast<size_t>(pairs[2 * p]);
        const size_t j = static_cast<size_t>(pairs[2 * p + 1]);
        const RawT a = buf[base + i];
        const RawT b = buf[base + j];
        buf[base + i] = b;
        buf[base + j] = a;
    }
};

template <typename RawT, typename IdxT>
struct BitRevLutIdxFunctor {
    const RawT* src;
    RawT* dst;
    const IdxT* rev_idx;
    size_t total;
    int logN;

    void operator()(sycl::nd_item<1> item) const {
        const size_t gid = item.get_global_linear_id();
        if (gid >= total) return;
        const size_t mask = (size_t{1} << logN) - 1;
        const size_t base = gid & ~mask;
        const size_t i = gid & mask;
        const size_t j = static_cast<size_t>(rev_idx[i]);
        dst[base + i] = src[base + j];
    }
};

// ============================================================================
// Strategy 3: Whole-FFT Zero-Pad XOR-Swizzled Shared-Memory Bit-Reversal
//   1 work-group per FFT. Unified for both in-place (src == dst) and out-of-place,
//   with runtime xor_k derived from BankModel -> 4x fewer kernel instantiations.
// ============================================================================
template <typename RawT>
struct BitRevWholeSmemFunctor {
    const RawT* src;
    RawT* dst;
    int logN;
    int wg;
    int xor_k;
    sycl::local_accessor<RawT, 1> smem;

    void operator()(sycl::nd_item<1> item) const {
        const size_t batch = item.get_group_linear_id();
        const int tid = static_cast<int>(item.get_local_linear_id());
        const int n = 1 << logN;
        const size_t base = batch << logN;
        const RawT* g_src = src + base;
        RawT* g_dst = dst + base;
        const int shift = logN - xor_k;
        const int mask = (xor_k > 0) ? ((1 << xor_k) - 1) : 0;

        #pragma unroll 4
        for (int idx = tid; idx < n; idx += wg) {
            const int slot = idx ^ ((idx >> shift) & mask);
            smem[slot] = g_src[idx];
        }

        sycl::group_barrier(item.get_group());

        #pragma unroll 4
        for (int idx = tid; idx < n; idx += wg) {
            const int r = static_cast<int>(bit_reverse(static_cast<uint32_t>(idx), logN));
            const int slot = r ^ ((r >> shift) & mask);
            g_dst[idx] = smem[slot];
        }
    }
};

// ============================================================================
// Strategy 4 & 5: Shared-Memory Tiled COBRA (Zero-Pad Conflict-Free XOR Tile)
//   Non-layout axes (WG, InPlace, UseLut) are runtime arguments -> 6x fewer kernels.
// ============================================================================
template <typename RawT, int WBits>
struct BitRevCobraSmemFunctor {
    static constexpr int kW = 1 << WBits;
    static constexpr int kMask = kW - 1;
    static constexpr int kTileElems = kW * kW;

    const RawT* src;
    RawT* dst;
    const uint16_t* rev_w;
    const uint16_t* b_lut;
    size_t work_per_batch;
    int logN;
    int wg;
    bool in_place;
    bool use_lut;
    sycl::local_accessor<RawT, 1> smem;

    void operator()(sycl::nd_item<1> item) const {
        const size_t wg_id = item.get_group_linear_id();
        const int tid = static_cast<int>(item.get_local_linear_id());
        const size_t batch = wg_id / work_per_batch;
        const size_t b_idx = wg_id - batch * work_per_batch;
        const int m = logN - 2 * WBits;
        const int a_shift = m + WBits;

        uint32_t b = 0;
        uint32_t b_rev = 0;
        if (use_lut) {
            if (in_place) {
                b = static_cast<uint32_t>(b_lut[2 * b_idx]);
                b_rev = static_cast<uint32_t>(b_lut[2 * b_idx + 1]);
            } else {
                b = static_cast<uint32_t>(b_idx);
                b_rev = static_cast<uint32_t>(b_lut[b_idx]);
            }
        } else {
            b = static_cast<uint32_t>(b_idx);
            b_rev = (m == 0) ? 0u : bit_reverse(b, m);
            if (in_place && b > b_rev) return;
        }

        const size_t base = batch << logN;
        const size_t off_b = base + (static_cast<size_t>(b) << WBits);
        const size_t off_brev = base + (static_cast<size_t>(b_rev) << WBits);
        const RawT* g_src = in_place ? dst : src;

        #pragma unroll 4
        for (int lin = tid; lin < kTileElems; lin += wg) {
            const int a = lin >> WBits;
            const int c = lin & kMask;
            smem[cobra_tile_xor_slot(a, c, WBits)] =
                g_src[off_b + (static_cast<size_t>(a) << a_shift) + c];
        }

        if (in_place && b != b_rev) {
            #pragma unroll 4
            for (int lin = tid; lin < kTileElems; lin += wg) {
                const int a = lin >> WBits;
                const int c = lin & kMask;
                smem[kTileElems + cobra_tile_xor_slot(a, c, WBits)] =
                    g_src[off_brev + (static_cast<size_t>(a) << a_shift) + c];
            }
        }

        sycl::group_barrier(item.get_group());

        #pragma unroll 4
        for (int lin = tid; lin < kTileElems; lin += wg) {
            const int a_out = lin >> WBits;
            const int c_out = lin & kMask;
            const int a_src = use_lut
                ? static_cast<int>(rev_w[c_out])
                : static_cast<int>(bit_reverse(static_cast<uint32_t>(c_out), WBits));
            const int c_src = use_lut
                ? static_cast<int>(rev_w[a_out])
                : static_cast<int>(bit_reverse(static_cast<uint32_t>(a_out), WBits));
            dst[off_brev + (static_cast<size_t>(a_out) << a_shift) + c_out] =
                smem[cobra_tile_xor_slot(a_src, c_src, WBits)];
        }

        if (in_place && b != b_rev) {
            #pragma unroll 4
            for (int lin = tid; lin < kTileElems; lin += wg) {
                const int a_out = lin >> WBits;
                const int c_out = lin & kMask;
                const int a_src = use_lut
                    ? static_cast<int>(rev_w[c_out])
                    : static_cast<int>(bit_reverse(static_cast<uint32_t>(c_out), WBits));
                const int c_src = use_lut
                    ? static_cast<int>(rev_w[a_out])
                    : static_cast<int>(bit_reverse(static_cast<uint32_t>(a_out), WBits));
                dst[off_b + (static_cast<size_t>(a_out) << a_shift) + c_out] =
                    smem[kTileElems + cobra_tile_xor_slot(a_src, c_src, WBits)];
            }
        }
    }
};

// ============================================================================
// Strategy 6: Zero-Smem Private L1-Tile Scalar COBRA (w = 4, 16x16 tiles, CPU)
// ============================================================================
template <typename RawT>
struct BitRevScalarCobraFunctor {
    static constexpr int kWBits = 4;
    static constexpr int kW = 16;

    const RawT* src;
    RawT* dst;
    const uint16_t* rev_w;
    const uint16_t* b_lut;
    size_t work_per_batch;
    size_t total_work;
    int logN;
    bool in_place;

    void operator()(sycl::nd_item<1> item) const {
        const size_t gid = item.get_global_linear_id();
        if (gid >= total_work) return;

        const size_t batch = gid / work_per_batch;
        const size_t b_idx = gid - batch * work_per_batch;
        const int m = logN - 2 * kWBits;
        const int a_shift = m + kWBits;

        uint32_t b = 0;
        uint32_t b_rev = 0;
        if (in_place) {
            b = static_cast<uint32_t>(b_lut[2 * b_idx]);
            b_rev = static_cast<uint32_t>(b_lut[2 * b_idx + 1]);
        } else {
            b = static_cast<uint32_t>(b_idx);
            b_rev = static_cast<uint32_t>(b_lut[b_idx]);
        }

        const size_t base = batch << logN;
        const size_t off_b = base + (static_cast<size_t>(b) << kWBits);
        const size_t off_brev = base + (static_cast<size_t>(b_rev) << kWBits);
        const RawT* g_src = in_place ? dst : src;
        RawT tileA[kW * kW];
        for (int a = 0; a < kW; ++a) {
            const RawT* row = g_src + off_b + (static_cast<size_t>(a) << a_shift);
            for (int c = 0; c < kW; ++c) {
                tileA[a * kW + c] = row[c];
            }
        }

        if (in_place && b != b_rev) {
            RawT tileB[kW * kW];
            for (int a = 0; a < kW; ++a) {
                const RawT* row = g_src + off_brev + (static_cast<size_t>(a) << a_shift);
                for (int c = 0; c < kW; ++c) {
                    tileB[a * kW + c] = row[c];
                }
            }
            for (int a_out = 0; a_out < kW; ++a_out) {
                const int c_src = static_cast<int>(rev_w[a_out]);
                RawT* row_rev = dst + off_brev + (static_cast<size_t>(a_out) << a_shift);
                RawT* row_b = dst + off_b + (static_cast<size_t>(a_out) << a_shift);
                for (int c_out = 0; c_out < kW; ++c_out) {
                    const int a_src = static_cast<int>(rev_w[c_out]);
                    row_rev[c_out] = tileA[a_src * kW + c_src];
                    row_b[c_out] = tileB[a_src * kW + c_src];
                }
            }
            return;
        }

        for (int a_out = 0; a_out < kW; ++a_out) {
            const int c_src = static_cast<int>(rev_w[a_out]);
            RawT* row_rev = dst + off_brev + (static_cast<size_t>(a_out) << a_shift);
            for (int c_out = 0; c_out < kW; ++c_out) {
                const int a_src = static_cast<int>(rev_w[c_out]);
                row_rev[c_out] = tileA[a_src * kW + c_src];
            }
        }
    }
};

namespace detail {

template <typename RawT, bool InPlace, size_t DefaultWG>
void launch_bitrev_naive(
    sycl::queue& q, const RawT* src, RawT* dst, size_t bs, int logN,
    BitRevLutState&, size_t wg_override, int) {
    const size_t total = bs << logN;
    if (total == 0) return;
    const size_t wg = (wg_override > 0) ? wg_override : DefaultWG;
    const size_t global = ((total + wg - 1) / wg) * wg;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
            BitRevNaiveFunctor<RawT>{src, dst, total, logN, InPlace});
    });
}

template <typename RawT, size_t DefaultWG>
void launch_bitrev_lut_pairs(
    sycl::queue& q, const RawT*, RawT* dst, size_t bs, int logN,
    BitRevLutState& lut, size_t wg_override, int) {
    lut.ensure(BitRevLutKind::Pairs);
    const size_t total_pairs = bs * lut.npairs;
    if (total_pairs == 0) return;
    const size_t wg = (wg_override > 0) ? wg_override : DefaultWG;
    const size_t global = ((total_pairs + wg - 1) / wg) * wg;
    if (lut.pairs16) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                BitRevLutPairsFunctor<RawT, uint16_t>{dst, lut.pairs16, lut.npairs, total_pairs, logN});
        });
    } else {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                BitRevLutPairsFunctor<RawT, uint32_t>{dst, lut.pairs32, lut.npairs, total_pairs, logN});
        });
    }
}

template <typename RawT, size_t DefaultWG>
void launch_bitrev_lut_idx(
    sycl::queue& q, const RawT* src, RawT* dst, size_t bs, int logN,
    BitRevLutState& lut, size_t wg_override, int) {
    lut.ensure(BitRevLutKind::Idx);
    const size_t total = bs << logN;
    if (total == 0) return;
    const size_t wg = (wg_override > 0) ? wg_override : DefaultWG;
    const size_t global = ((total + wg - 1) / wg) * wg;
    if (lut.idx16) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                BitRevLutIdxFunctor<RawT, uint16_t>{src, dst, lut.idx16, total, logN});
        });
    } else {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
                BitRevLutIdxFunctor<RawT, uint32_t>{src, dst, lut.idx32, total, logN});
        });
    }
}

template <typename RawT, int WG>
void launch_bitrev_whole_smem(
    sycl::queue& q, const RawT* src, RawT* dst, size_t bs, int logN,
    BitRevLutState&, size_t, int xor_k) {
    if (bs == 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<RawT, 1> smem(sycl::range<1>(size_t{1} << logN), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(bs * WG), sycl::range<1>(WG)),
            BitRevWholeSmemFunctor<RawT>{src, dst, logN, WG, xor_k, smem});
    });
}

template <typename RawT, int WBits, int WG, bool InPlace, bool UseLut>
void launch_bitrev_cobra_smem(
    sycl::queue& q, const RawT* src, RawT* dst, size_t bs, int logN,
    BitRevLutState& lut, size_t, int) {
    if (bs == 0) return;
    const int m = logN - 2 * WBits;
    const uint16_t* rev_w_ptr = nullptr;
    const uint16_t* b_lut_ptr = nullptr;
    size_t work_per_batch = size_t{1} << m;

    if constexpr (UseLut) {
        const BitRevLutKind kind = (WBits == 4) ? BitRevLutKind::CobraW4 : BitRevLutKind::CobraW5;
        lut.ensure(kind);
        const int wi = WBits - 4;
        rev_w_ptr = lut.rev_w[wi];
        if constexpr (InPlace) {
            b_lut_ptr = lut.b_pairs[wi];
            work_per_batch = lut.nb_pairs[wi];
        } else {
            b_lut_ptr = lut.b_all_rev[wi];
            work_per_batch = size_t{1} << m;
        }
    }

    const size_t total_wg = bs * work_per_batch;
    const size_t smem_elems = (size_t{1} << (2 * WBits)) * (InPlace ? 2 : 1);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<RawT, 1> smem(sycl::range<1>(smem_elems), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(total_wg * WG), sycl::range<1>(WG)),
            BitRevCobraSmemFunctor<RawT, WBits>{
                src, dst, rev_w_ptr, b_lut_ptr, work_per_batch, logN, WG, InPlace, UseLut, smem});
    });
}

template <typename RawT, bool InPlace>
void launch_bitrev_scalar_cobra(
    sycl::queue& q, const RawT* src, RawT* dst, size_t bs, int logN,
    BitRevLutState& lut, size_t wg_override, int) {
    if (bs == 0) return;
    lut.ensure(BitRevLutKind::CobraW4);
    const int m = logN - 8;
    const size_t work_per_batch = InPlace ? lut.nb_pairs[0] : (size_t{1} << m);
    const uint16_t* b_lut_ptr = InPlace ? lut.b_pairs[0] : lut.b_all_rev[0];
    const size_t total_work = bs * work_per_batch;
    if (total_work == 0) return;
    const size_t wg = (wg_override > 0) ? wg_override : size_t{32};
    const size_t global = ((total_work + wg - 1) / wg) * wg;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
            BitRevScalarCobraFunctor<RawT>{
                src, dst, lut.rev_w[0], b_lut_ptr, work_per_batch, total_work, logN, InPlace});
    });
}

template <typename RawT>
void append_whole_smem_candidates(int logN, std::vector<BitRevCandidate<RawT>>& out) {
    auto add = [&](const char* name, size_t wg, size_t smem_b,
                   typename BitRevCandidate<RawT>::LaunchFn fn) {
        out.push_back(BitRevCandidate<RawT>{
            name, wg, smem_b, true, false, BitRevLutKind::None, fn});
    };
    const int max_whole_logN = (sizeof(RawT) <= 4) ? 14 : (sizeof(RawT) <= 8 ? 13 : 12);
    if (logN >= 10 && logN <= max_whole_logN) {
        const size_t smem_b = (size_t{1} << logN) * sizeof(RawT);
        add("whole_smem_xor_w256", 256, smem_b, &launch_bitrev_whole_smem<RawT, 256>);
        add("whole_smem_xor_w128", 128, smem_b, &launch_bitrev_whole_smem<RawT, 128>);
    }
}

template <typename RawT, bool InPlace>
std::vector<BitRevCandidate<RawT>> build_bitrev_candidates(int logN) {
    std::vector<BitRevCandidate<RawT>> out;

    // 1. Whole-FFT zero-pad XOR shared memory (highest priority when it fits)
    append_whole_smem_candidates<RawT>(logN, out);

    // 2. Shared-memory COBRA w=5 (32x32 tiles) + LUT variant
    if (logN >= 10) {
        const size_t smem_w5 = (size_t{1} << 10) * (InPlace ? 2 : 1) * sizeof(RawT);
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w5_lut_wg256", 256, smem_w5, true, false, BitRevLutKind::CobraW5,
            &launch_bitrev_cobra_smem<RawT, 5, 256, InPlace, true>});
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w5_wg256", 256, smem_w5, true, false, BitRevLutKind::None,
            &launch_bitrev_cobra_smem<RawT, 5, 256, InPlace, false>});
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w5_wg128", 128, smem_w5, true, false, BitRevLutKind::None,
            &launch_bitrev_cobra_smem<RawT, 5, 128, InPlace, false>});
    }

    // 3. Shared-memory COBRA w=4 (16x16 tiles) + LUT variant
    if (logN >= 8) {
        const size_t smem_w4 = (size_t{1} << 8) * (InPlace ? 2 : 1) * sizeof(RawT);
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w4_lut_wg256", 256, smem_w4, true, false, BitRevLutKind::CobraW4,
            &launch_bitrev_cobra_smem<RawT, 4, 256, InPlace, true>});
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w4_wg256", 256, smem_w4, true, false, BitRevLutKind::None,
            &launch_bitrev_cobra_smem<RawT, 4, 256, InPlace, false>});
        out.push_back(BitRevCandidate<RawT>{
            "cobra_w4_wg128", 128, smem_w4, true, false, BitRevLutKind::None,
            &launch_bitrev_cobra_smem<RawT, 4, 128, InPlace, false>});
    }

    // 4. Zero-smem scalar COBRA w=4 (CPU / non-native-local-mem fast path)
    if (logN >= 8) {
        out.push_back(BitRevCandidate<RawT>{
            "scalar_cobra_w4", 32, 0, false, true, BitRevLutKind::CobraW4,
            &launch_bitrev_scalar_cobra<RawT, InPlace>});
    }

    // 5. Precomputed device LUT (pairs for in-place, rev index for out-of-place)
    if constexpr (InPlace) {
        out.push_back(BitRevCandidate<RawT>{
            "lut_pairs_w256", 256, 0, false, false, BitRevLutKind::Pairs,
            &launch_bitrev_lut_pairs<RawT, 256>});
    } else {
        out.push_back(BitRevCandidate<RawT>{
            "lut_idx_w256", 256, 0, false, false, BitRevLutKind::Idx,
            &launch_bitrev_lut_idx<RawT, 256>});
    }

    // 6. Naive global scatter/gather baseline
    out.push_back(BitRevCandidate<RawT>{
        "naive_w256", 256, 0, false, false, BitRevLutKind::None,
        &launch_bitrev_naive<RawT, InPlace, 256>});
    out.push_back(BitRevCandidate<RawT>{
        "naive_w128", 128, 0, false, false, BitRevLutKind::None,
        &launch_bitrev_naive<RawT, InPlace, 128>});

    return out;
}

} // namespace detail

// ============================================================================
// BitRevEngine<RawT> Member Implementations
// ============================================================================
template <typename RawT>
BitRevEngine<RawT>::BitRevEngine() = default;

template <typename RawT>
BitRevEngine<RawT>::BitRevEngine(
    sycl::queue q, int logN, const char* prec_tag,
    const DeviceFftConfig& dev_cfg, const FftLaunchConfig& launch_cfg)
    : q_(std::move(q)),
      logN_(logN),
      prec_tag_(prec_tag ? prec_tag : "f"),
      dev_cfg_(dev_cfg),
      launch_cfg_(launch_cfg),
      xor_k_(whole_smem_xor_k(bank_model_from_kind(dev_cfg.bank_model), sizeof(RawT))),
      lut_(std::make_unique<BitRevLutState>(q_, logN)) {
    cache_path_ = AutotuneCache::resolve_cache_path(launch_cfg_.cache_path);
    valid_ip_ = filter_candidates(detail::build_bitrev_candidates<RawT, true>(logN_));
    valid_oop_ = filter_candidates(detail::build_bitrev_candidates<RawT, false>(logN_));
}

template <typename RawT>
BitRevEngine<RawT>::~BitRevEngine() = default;

template <typename RawT>
BitRevEngine<RawT>::BitRevEngine(BitRevEngine&&) noexcept = default;

template <typename RawT>
BitRevEngine<RawT>& BitRevEngine<RawT>::operator=(BitRevEngine&&) noexcept = default;

template <typename RawT>
void BitRevEngine<RawT>::resolve(bool in_place) {
    if (in_place && resolved_ip_) return;
    if (!in_place && resolved_oop_) return;

    const auto& cands = in_place ? valid_ip_ : valid_oop_;
    if (cands.empty()) {
        throw std::runtime_error("BitRevEngine: no valid bit-reversal candidate for device");
    }

    const char* slot_tag = in_place ? "bitrev_ip" : "bitrev_oop";
    int chosen = 0;

    if (!launch_cfg_.bitrev_variant_override.empty()) {
        bool found = false;
        for (size_t i = 0; i < cands.size(); ++i) {
            if (launch_cfg_.bitrev_variant_override == cands[i].variant_id) {
                chosen = static_cast<int>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                std::string("BitRevEngine: requested --bitrev-variant '") +
                launch_cfg_.bitrev_variant_override + "' is not valid for logN=" +
                std::to_string(logN_) + " (" + slot_tag + ")");
        }
        set_resolved(in_place, chosen, false);
        return;
    }

    const bool allow_cache = (launch_cfg_.autotune == AutotuneMode::Auto);
    const bool force_tune = (launch_cfg_.autotune == AutotuneMode::Force);
    const std::string cache_key = AutotuneCache::make_key(dev_cfg_, prec_tag_, logN_, slot_tag);

    if (allow_cache && !launch_cfg_.has_bitrev_overrides()) {
        AutotuneEntry cached{};
        if (AutotuneCache::get(cache_path_, cache_key, cached) &&
            cached.mode == PlanExecMode::SinglePass) {
            for (size_t i = 0; i < cands.size(); ++i) {
                if (cached.variant_id == cands[i].variant_id) {
                    set_resolved(in_place, static_cast<int>(i), true);
                    return;
                }
            }
        }
    }

    chosen = 0;
    if (launch_cfg_.autotune != AutotuneMode::Off && cands.size() > 1) {
        chosen = tune_candidates(in_place, cands, allow_cache || force_tune, cache_key);
    }
    set_resolved(in_place, chosen, false);
}

template <typename RawT>
void BitRevEngine<RawT>::execute_in_place(RawT* buf, size_t bs) {
    if (bs == 0 || logN_ <= 0) return;
    resolve(true);
    const auto& cand = valid_ip_[static_cast<size_t>(chosen_ip_)];
    cand.launch(q_, buf, buf, bs, logN_, *lut_, effective_wg_override(cand), xor_k_);
}

template <typename RawT>
void BitRevEngine<RawT>::execute(const RawT* src, RawT* dst, size_t bs) {
    if (bs == 0 || logN_ <= 0) return;
    if (src == dst) {
        execute_in_place(dst, bs);
        return;
    }
    resolve(false);
    const auto& cand = valid_oop_[static_cast<size_t>(chosen_oop_)];
    cand.launch(q_, src, dst, bs, logN_, *lut_, effective_wg_override(cand), xor_k_);
}

template <typename RawT>
void BitRevEngine<RawT>::execute_candidate(
    const BitRevCandidate<RawT>& cand, const RawT* src, RawT* dst, size_t bs) {
    if (bs == 0 || logN_ <= 0) return;
    cand.launch(q_, src, dst, bs, logN_, *lut_, effective_wg_override(cand), xor_k_);
}

template <typename RawT>
const char* BitRevEngine<RawT>::variant_name(bool in_place) {
    resolve(in_place);
    const auto& cands = in_place ? valid_ip_ : valid_oop_;
    const int idx = in_place ? chosen_ip_ : chosen_oop_;
    return cands[static_cast<size_t>(idx)].variant_id;
}

template <typename RawT>
bool BitRevEngine<RawT>::cache_hit(bool in_place) {
    resolve(in_place);
    return in_place ? cache_hit_ip_ : cache_hit_oop_;
}

template <typename RawT>
size_t BitRevEngine<RawT>::effective_wg_override(const BitRevCandidate<RawT>& cand) const {
    if (cand.smem_bytes > 0) return 0;
    size_t wg = launch_cfg_.bitrev_wg_size > 0 ? launch_cfg_.bitrev_wg_size : cand.wg_size;
    if (wg > dev_cfg_.max_work_group_size) {
        wg = dev_cfg_.max_work_group_size;
    }
    return std::max<size_t>(1, wg);
}

template <typename RawT>
std::vector<BitRevCandidate<RawT>> BitRevEngine<RawT>::filter_candidates(
    const std::vector<BitRevCandidate<RawT>>& all) const {
    std::vector<BitRevCandidate<RawT>> out;
    for (const auto& c : all) {
        if (c.requires_native_local_mem &&
            !dev_cfg_.has_native_local_mem &&
            !launch_cfg_.allow_emulated_local_mem) {
            continue;
        }
        if (c.cpu_only && dev_cfg_.has_native_local_mem && dev_cfg_.is_gpu) {
            continue;
        }
        if (c.smem_bytes > dev_cfg_.local_mem_size) {
            continue;
        }
        if (c.smem_bytes > 0 && c.wg_size > dev_cfg_.max_work_group_size) {
            continue;
        }
        out.push_back(c);
    }
    return out;
}

template <typename RawT>
void BitRevEngine<RawT>::set_resolved(bool in_place, int idx, bool hit) {
    if (in_place) {
        chosen_ip_ = idx;
        cache_hit_ip_ = hit;
        resolved_ip_ = true;
    } else {
        chosen_oop_ = idx;
        cache_hit_oop_ = hit;
        resolved_oop_ = true;
    }
}

template <typename RawT>
std::vector<size_t> BitRevEngine<RawT>::shortlist_for_tuning(
    bool in_place, const std::vector<BitRevCandidate<RawT>>& cands) const {
    std::vector<size_t> picks;
    auto add_by_name = [&](const char* name) {
        for (size_t i = 0; i < cands.size(); ++i) {
            if (std::strcmp(cands[i].variant_id, name) == 0) {
                if (std::find(picks.begin(), picks.end(), i) == picks.end()) {
                    picks.push_back(i);
                }
                return true;
            }
        }
        return false;
    };

    if (!dev_cfg_.has_native_local_mem) {
        add_by_name("scalar_cobra_w4");
        add_by_name(in_place ? "lut_pairs_w256" : "lut_idx_w256");
    } else {
        const bool has_whole = add_by_name("whole_smem_xor_w256");
        if (has_whole) {
            add_by_name("whole_smem_xor_w128");
            const size_t whole_bytes = (size_t{1} << logN_) * sizeof(RawT);
            if (whole_bytes * 2 >= dev_cfg_.local_mem_size) {
                if (sizeof(RawT) <= 8 && logN_ >= 12) {
                    if (!add_by_name("cobra_w5_lut_wg256")) {
                        add_by_name("cobra_w5_wg256");
                    }
                } else if (!add_by_name("cobra_w4_lut_wg256")) {
                    add_by_name("cobra_w5_wg256");
                }
            }
        } else {
            const int m5 = logN_ - 10;
            if (m5 >= 2) {
                add_by_name("cobra_w5_lut_wg256");
            }
            add_by_name("cobra_w5_wg256");
            if (picks.size() < 2) {
                add_by_name("cobra_w5_wg128");
            }
            if (picks.size() < 3) {
                add_by_name("cobra_w4_lut_wg256");
            }
            if (picks.size() < 3) {
                add_by_name("cobra_w4_wg128");
            }
        }
    }

    if (picks.empty()) {
        const size_t lim = std::min<size_t>(cands.size(), 2);
        for (size_t i = 0; i < lim; ++i) picks.push_back(i);
    }
    return picks;
}

template <typename RawT>
size_t BitRevEngine<RawT>::representative_tune_bs() const {
    const size_t fft_bytes = (size_t{1} << logN_) * sizeof(RawT);
    const size_t target_bytes = dev_cfg_.is_gpu ? (size_t{16} * 1024 * 1024)
                                                : (size_t{2} * 1024 * 1024);
    size_t bs = std::max<size_t>(4, target_bytes / std::max<size_t>(1, fft_bytes));
    if (dev_cfg_.max_mem_alloc_size > 0) {
        const size_t max_bs = std::max<size_t>(2, (dev_cfg_.max_mem_alloc_size / 8) / fft_bytes);
        bs = std::min(bs, max_bs);
    }
    return bs;
}

template <typename RawT>
int BitRevEngine<RawT>::tune_candidates(
    bool in_place, const std::vector<BitRevCandidate<RawT>>& cands,
    bool persist_cache, const std::string& cache_key) {
    const auto shortlist = shortlist_for_tuning(in_place, cands);
    if (shortlist.size() == 1) {
        const int idx = static_cast<int>(shortlist[0]);
        if (persist_cache && !launch_cfg_.has_bitrev_overrides()) {
            AutotuneEntry ent{};
            ent.mode = PlanExecMode::SinglePass;
            ent.variant_id = cands[shortlist[0]].variant_id;
            AutotuneCache::put(cache_path_, cache_key, ent);
        }
        return idx;
    }

    const size_t N = size_t{1} << logN_;
    const size_t tune_bs = representative_tune_bs();
    const size_t total = tune_bs * N;

    RawT* buf_a = nullptr;
    RawT* buf_b = nullptr;
    int best_idx = static_cast<int>(shortlist[0]);
    double best_ms = std::numeric_limits<double>::max();

    try {
        buf_a = sycl::malloc_device<RawT>(total, q_);
        if (!in_place) {
            buf_b = sycl::malloc_device<RawT>(total, q_);
        }
        if (!buf_a || (!in_place && !buf_b)) {
            if (buf_a) sycl::free(buf_a, q_);
            if (buf_b) sycl::free(buf_b, q_);
            return best_idx;
        }
        q_.memset(buf_a, 0, total * sizeof(RawT)).wait();
        if (buf_b) q_.memset(buf_b, 0, total * sizeof(RawT)).wait();

        try {
            const auto& c0 = cands[shortlist[0]];
            const size_t wg0 = effective_wg_override(c0);
            if (in_place) c0.launch(q_, buf_a, buf_a, tune_bs, logN_, *lut_, wg0, xor_k_);
            else c0.launch(q_, buf_a, buf_b, tune_bs, logN_, *lut_, wg0, xor_k_);
            q_.wait_and_throw();
        } catch (...) {
        }

        constexpr int kTimedRuns = 3;
        for (size_t idx : shortlist) {
            try {
                const auto& cand = cands[idx];
                const size_t wg_ov = effective_wg_override(cand);
                if (in_place) cand.launch(q_, buf_a, buf_a, tune_bs, logN_, *lut_, wg_ov, xor_k_);
                else cand.launch(q_, buf_a, buf_b, tune_bs, logN_, *lut_, wg_ov, xor_k_);
                q_.wait_and_throw();

                double samples[kTimedRuns];
                for (int r = 0; r < kTimedRuns; ++r) {
                    const auto t0 = std::chrono::steady_clock::now();
                    if (in_place) cand.launch(q_, buf_a, buf_a, tune_bs, logN_, *lut_, wg_ov, xor_k_);
                    else cand.launch(q_, buf_a, buf_b, tune_bs, logN_, *lut_, wg_ov, xor_k_);
                    q_.wait_and_throw();
                    const auto t1 = std::chrono::steady_clock::now();
                    samples[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
                std::sort(samples, samples + kTimedRuns);
                const double med_ms = samples[kTimedRuns / 2];
                if (med_ms < best_ms) {
                    best_ms = med_ms;
                    best_idx = static_cast<int>(idx);
                }
            } catch (...) {
            }
        }
    } catch (...) {
        best_idx = static_cast<int>(shortlist[0]);
    }

    if (buf_a) sycl::free(buf_a, q_);
    if (buf_b) sycl::free(buf_b, q_);

    if (persist_cache && !launch_cfg_.has_bitrev_overrides() &&
        best_ms < std::numeric_limits<double>::max()) {
        AutotuneEntry ent{};
        ent.mode = PlanExecMode::SinglePass;
        ent.variant_id = cands[static_cast<size_t>(best_idx)].variant_id;
        AutotuneCache::put(cache_path_, cache_key, ent);
    }
    return best_idx;
}

// Explicit instantiations for deduplicated 4B/8B/16B BitRevEngine
template class BitRevEngine<uint32_t>;
template class BitRevEngine<uint64_t>;
template class BitRevEngine<sycl::uint4>;

} // namespace sycl_fft
