#pragma once

#include "common/device.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace sycl_fft {

// ============================================================================
// Hardware-Generalized Multi-Strategy Bit-Reversal Engine (Declarations Only)
//   Deduplicated by raw element size (4B uint32_t, 8B uint64_t, 16B sycl::uint4)
//   so Complex (FP64) and Complexdf (DF) share identical compiled/JIT'd kernels.
//   All kernels, LUT state, and planning/autotuning logic live in bitrev/bitrev.cpp.
// ============================================================================
template <size_t Bytes>
struct RawBitRevStorage;
template <> struct RawBitRevStorage<4>  { using type = uint32_t; };
template <> struct RawBitRevStorage<8>  { using type = uint64_t; };
template <> struct RawBitRevStorage<16> { using type = sycl::uint4; };

template <typename ComplexT>
using RawBitRevStorageT = typename RawBitRevStorage<sizeof(ComplexT)>::type;

enum class BitRevLutKind : uint8_t {
    None = 0,
    Pairs = 1,   // in-place (i, j) pairs with i < j
    Idx = 2,     // out-of-place full rev[N] index table
    CobraW4 = 3, // rev_w (w=4) + compact b_pairs (in-place) / b_all_rev (out-of-place)
    CobraW5 = 4  // rev_w (w=5) + compact b_pairs (in-place) / b_all_rev (out-of-place)
};

// Opaque device LUT state (defined in bitrev/bitrev.cpp)
struct BitRevLutState;

template <typename RawT>
struct BitRevCandidate {
    using LaunchFn = void (*)(
        sycl::queue&, const RawT*, RawT*, size_t, int, BitRevLutState&, size_t, int);

    const char*   variant_id;
    size_t        wg_size;
    size_t        smem_bytes;
    bool          requires_native_local_mem;
    bool          cpu_only;
    BitRevLutKind lut_kind;
    LaunchFn      launch;
};

// Capability-filtered, shortlist-autotuned bit-reversal engine
// (instantiated for uint32_t, uint64_t, and sycl::uint4 in bitrev/bitrev.cpp).
template <typename RawT>
class BitRevEngine {
public:
    BitRevEngine();
    BitRevEngine(sycl::queue q, int logN, const char* prec_tag,
                 const DeviceFftConfig& dev_cfg, const FftLaunchConfig& launch_cfg);
    ~BitRevEngine();

    BitRevEngine(const BitRevEngine&) = delete;
    BitRevEngine& operator=(const BitRevEngine&) = delete;
    BitRevEngine(BitRevEngine&&) noexcept;
    BitRevEngine& operator=(BitRevEngine&&) noexcept;

    const std::vector<BitRevCandidate<RawT>>& valid_candidates(bool in_place) const {
        return in_place ? valid_ip_ : valid_oop_;
    }

    void resolve(bool in_place);
    void execute_in_place(RawT* buf, size_t bs);
    void execute(const RawT* src, RawT* dst, size_t bs);
    void execute_candidate(const BitRevCandidate<RawT>& cand,
                           const RawT* src, RawT* dst, size_t bs);
    const char* variant_name(bool in_place = true);
    bool cache_hit(bool in_place = true);

private:
    size_t effective_wg_override(const BitRevCandidate<RawT>& cand) const;
    std::vector<BitRevCandidate<RawT>> filter_candidates(
        const std::vector<BitRevCandidate<RawT>>& all) const;
    void set_resolved(bool in_place, int idx, bool hit);
    std::vector<size_t> shortlist_for_tuning(
        bool in_place, const std::vector<BitRevCandidate<RawT>>& cands) const;
    size_t representative_tune_bs() const;
    int tune_candidates(bool in_place, const std::vector<BitRevCandidate<RawT>>& cands,
                        bool persist_cache, const std::string& cache_key);

    sycl::queue q_;
    int logN_ = 0;
    const char* prec_tag_ = "f";
    DeviceFftConfig dev_cfg_{};
    FftLaunchConfig launch_cfg_{};
    int xor_k_ = 0;
    std::string cache_path_;
    std::unique_ptr<BitRevLutState> lut_;

    std::vector<BitRevCandidate<RawT>> valid_ip_;
    std::vector<BitRevCandidate<RawT>> valid_oop_;
    bool resolved_ip_ = false;
    bool resolved_oop_ = false;
    int chosen_ip_ = 0;
    int chosen_oop_ = 0;
    bool cache_hit_ip_ = false;
    bool cache_hit_oop_ = false;
};

extern template class BitRevEngine<uint32_t>;
extern template class BitRevEngine<uint64_t>;
extern template class BitRevEngine<sycl::uint4>;

// Zero-overhead typed wrapper mapping ComplexT -> RawBitRevStorageT<ComplexT>.
template <typename ComplexT>
class BitRevPlan {
public:
    using RawT = RawBitRevStorageT<ComplexT>;
    using CandidateT = BitRevCandidate<RawT>;

    static_assert(sizeof(ComplexT) == sizeof(RawT) && std::is_trivially_copyable_v<ComplexT>,
                  "ComplexT must match raw bit-reversal storage size and be trivially copyable");

    BitRevPlan() = default;

    BitRevPlan(sycl::queue q, int logN, const char* prec_tag,
               const DeviceFftConfig& dev_cfg, const FftLaunchConfig& launch_cfg)
        : engine_(std::move(q), logN, prec_tag, dev_cfg, launch_cfg) {}

    const std::vector<CandidateT>& valid_candidates(bool in_place) const {
        return engine_.valid_candidates(in_place);
    }

    void resolve(bool in_place) { engine_.resolve(in_place); }

    void execute_in_place(ComplexT* buf, size_t bs) {
        engine_.execute_in_place(reinterpret_cast<RawT*>(buf), bs);
    }

    void execute(const ComplexT* src, ComplexT* dst, size_t bs) {
        engine_.execute(reinterpret_cast<const RawT*>(src), reinterpret_cast<RawT*>(dst), bs);
    }

    void execute_candidate(const CandidateT& cand, const ComplexT* src, ComplexT* dst, size_t bs) {
        engine_.execute_candidate(cand, reinterpret_cast<const RawT*>(src), reinterpret_cast<RawT*>(dst), bs);
    }

    const char* variant_name(bool in_place = true) { return engine_.variant_name(in_place); }
    bool cache_hit(bool in_place = true) { return engine_.cache_hit(in_place); }

private:
    BitRevEngine<RawT> engine_{};
};

} // namespace sycl_fft
