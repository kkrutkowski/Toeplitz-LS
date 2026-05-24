cmake_minimum_required(VERSION 3.15)
project(NUFFT_Benchmark CXX C)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# C Standard (Equivalant to -std=gnu11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)

# Optimize for performance across both CXX and C
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math")

# ==============================================================================
# 1. Output Directory Configuration
# ==============================================================================
# Force the final executable to be placed in the project root directory
# even when building inside a 'tmp/' build folder.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")

# ==============================================================================
# 2. Automatically fetch and configure CPM
# ==============================================================================
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# ==============================================================================
# 3. Add FINUFFT via CPM (Strict Single-Threaded)
# ==============================================================================
set(FINUFFT_USE_OPENMP OFF CACHE BOOL "Disable OpenMP in FINUFFT" FORCE)
set(CMAKE_DISABLE_FIND_PACKAGE_OpenMP ON CACHE BOOL "Disable OpenMP globally" FORCE)

#CPMAddPackage("gh:flatironinstitute/finufft@2.4.1")
CPMAddPackage("gh:flatironinstitute/finufft@2.5.1")

# ==============================================================================
# 4. Compile Microlibrary Objects
# ==============================================================================
# Setting the path to your source files
set(TLS_SRC_DIR "../../src")

set(TLS_MAX_TWIDDLE_REUSE 16 CACHE STRING "Twiddle ladder reuse factor")

add_library(tls_utils OBJECT ${TLS_SRC_DIR}/utils.c)
target_compile_definitions(tls_utils PRIVATE MAX_TWIDDLE_REUSE=${TLS_MAX_TWIDDLE_REUSE})
target_include_directories(tls_utils PRIVATE ../../include)

# Variant 1: DOUBLE_DOUBLE (-D DOUBLE_DOUBLE)
add_library(tlsdd_nufft1 OBJECT ${TLS_SRC_DIR}/nanofft.c ${TLS_SRC_DIR}/nufft1.c)
target_compile_definitions(tlsdd_nufft1 PRIVATE DOUBLE_DOUBLE)
target_include_directories(tlsdd_nufft1 PRIVATE ../../include)

# Variant 2: DOUBLE (-D DOUBLE)
add_library(tls_nufft1 OBJECT ${TLS_SRC_DIR}/nanofft.c ${TLS_SRC_DIR}/nufft1.c)
target_compile_definitions(tls_nufft1 PRIVATE DOUBLE)
target_include_directories(tls_nufft1 PRIVATE ../../include)

# Variant 3: FLOAT (No custom definitions)
add_library(tlsf_nufft1 OBJECT ${TLS_SRC_DIR}/nanofft.c ${TLS_SRC_DIR}/nufft1.c)
target_include_directories(tlsf_nufft1 PRIVATE ../../include)

# Native scaling generators. These mirror the root Makefile flow: build one
# scaling executable per precision and run them in sequence so each appends its
# macro group to a build-directory scaling.h.
add_executable(tlsf_nufft_scaling EXCLUDE_FROM_ALL
    ${TLS_SRC_DIR}/scaling.c
    $<TARGET_OBJECTS:tlsf_nufft1>
)
target_compile_definitions(tlsf_nufft_scaling PRIVATE SAVE)
target_include_directories(tlsf_nufft_scaling PRIVATE ../../include)
target_link_libraries(tlsf_nufft_scaling PRIVATE m)

add_executable(tls_nufft_scaling EXCLUDE_FROM_ALL
    ${TLS_SRC_DIR}/scaling.c
    $<TARGET_OBJECTS:tls_nufft1>
)
target_compile_definitions(tls_nufft_scaling PRIVATE DOUBLE SAVE)
target_include_directories(tls_nufft_scaling PRIVATE ../../include)
target_link_libraries(tls_nufft_scaling PRIVATE m)

add_executable(tlsdd_nufft_scaling EXCLUDE_FROM_ALL
    ${TLS_SRC_DIR}/scaling.c
    $<TARGET_OBJECTS:tlsdd_nufft1>
)
target_compile_definitions(tlsdd_nufft_scaling PRIVATE DOUBLE_DOUBLE SAVE)
target_include_directories(tlsdd_nufft_scaling PRIVATE ../../include)
target_link_libraries(tlsdd_nufft_scaling PRIVATE m)

set_target_properties(
    tlsf_nufft_scaling
    tls_nufft_scaling
    tlsdd_nufft_scaling
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

set(NATIVE_SCALING_HEADER "${CMAKE_BINARY_DIR}/scaling.h")
add_custom_command(
    OUTPUT "${NATIVE_SCALING_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${NATIVE_SCALING_HEADER}"
    COMMAND tlsf_nufft_scaling
    COMMAND tls_nufft_scaling
    COMMAND tlsdd_nufft_scaling
    DEPENDS tlsf_nufft_scaling tls_nufft_scaling tlsdd_nufft_scaling
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Generating native scaling.h"
    VERBATIM
)
add_custom_target(generate_native_scaling_header
    DEPENDS "${NATIVE_SCALING_HEADER}"
)

# ==============================================================================
# 5. Setup Benchmark Executable
# ==============================================================================
# Link the benchmark code with the object files generated above
add_executable(finufft_bench
    finufft_bench.cpp
    $<TARGET_OBJECTS:tlsdd_nufft1>
    $<TARGET_OBJECTS:tls_nufft1>
    $<TARGET_OBJECTS:tlsf_nufft1>
)

add_executable(finufft_bench_block
    finufft_bench_block.cpp
    $<TARGET_OBJECTS:tls_utils>
    $<TARGET_OBJECTS:tlsdd_nufft1>
    $<TARGET_OBJECTS:tls_nufft1>
    $<TARGET_OBJECTS:tlsf_nufft1>
)

# Include your microlibrary headers
target_include_directories(finufft_bench PRIVATE ../../include)
target_include_directories(finufft_bench_block PRIVATE
    "${CMAKE_BINARY_DIR}"
    ../../include
)
target_compile_definitions(finufft_bench_block PRIVATE MAX_TWIDDLE_REUSE=${TLS_MAX_TWIDDLE_REUSE})
add_dependencies(finufft_bench_block generate_native_scaling_header)

# Link FINUFFT and math libraries
target_link_libraries(finufft_bench PRIVATE
    finufft
    quadmath
    m
)

target_link_libraries(finufft_bench_block PRIVATE
    finufft
    m
)
