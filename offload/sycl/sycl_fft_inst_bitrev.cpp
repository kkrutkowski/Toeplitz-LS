// Explicit instantiations for deduplicated 4B/8B/16B BitRevEngine (separate HCF module)
#include "sycl_fft_bitrev.hpp"

namespace sycl_fft {
template class BitRevEngine<uint32_t>;
template class BitRevEngine<uint64_t>;
template class BitRevEngine<sycl::uint4>;
} // namespace sycl_fft
