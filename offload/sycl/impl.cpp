// Public C ABI entrypoints for the Toeplitz-LS SYCL offload module (tls_sycl.so).
//
// All internal C++ SYCL kernels and plans are compiled with -fvisibility=hidden.
// Only symbols explicitly annotated with TLS_SYCL_API are exported from the
// compiled shared library for consumption by toeplitz_ls.offload.sycl.

#if defined(_WIN32)
#  define TLS_SYCL_API __declspec(dllexport)
#else
#  define TLS_SYCL_API __attribute__((visibility("default")))
#endif

extern "C" {

// Placeholder ABI version hook so tls_sycl.so links and exports cleanly prior
// to exposing the full PSWF21 / PSWF43 periodogram + Levinson solver C API.
TLS_SYCL_API int tls_sycl_abi_version(void) {
    return 1;
}

} // extern "C"
