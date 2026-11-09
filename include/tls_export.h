#ifndef TLS_EXPORT_H
#define TLS_EXPORT_H

/*
 * Shared-library export decoration.
 *
 * The library is built three ways:
 *
 *   static           TLS_API is empty; every symbol keeps its default linkage.
 *   shared, building TLS_BUILD_SHARED is defined, so the public entry points
 *                    are pinned to default visibility on ELF/Mach-O (where the
 *                    translation units are compiled with -fvisibility=hidden)
 *                    and declared __declspec(dllexport) on Windows.
 *   shared, using    TLS_USE_SHARED is defined by a consumer that links the
 *                    DLL, which needs __declspec(dllimport) on Windows only.
 *
 * The annotation is the export list for the Windows DLL: x64 and arm64 export
 * undecorated names, so the Python ctypes wrapper sees the plain identifiers.
 */

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(TLS_BUILD_SHARED)
#        define TLS_API __declspec(dllexport)
#    elif defined(TLS_USE_SHARED)
#        define TLS_API __declspec(dllimport)
#    else
#        define TLS_API
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    if defined(TLS_BUILD_SHARED)
#        define TLS_API __attribute__((visibility("default")))
#    else
#        define TLS_API
#    endif
#else
#    define TLS_API
#endif

#endif /* TLS_EXPORT_H */
