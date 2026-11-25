// Dedicated compilation unit for SYCL hardware probes and persistent AutotuneCache.
// Keeps the 5 hardware probe kernels in a single tiny HCF module instead of duplicating
// them across main.o and every precision-specific FFT translation unit.
#define SYCL_FFT_DEVICE_IMPLEMENTATION_TU 1
#include "common/device.hpp"
