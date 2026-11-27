# Toeplitz-LS SYCL Offload Module (tls_sycl.so) Release Build System
#
# Modular codelet hierarchy isolating hardware probes, bit-reversal, small/large
# radix-2 DIF FFTs, NuFFT-1 aux/spread/deconv kernels, zero-HCF host planning,
# linear algebra solvers (linalg/, upcoming), and the extern "C" ABI (impl.cpp).
#
# Release builds produce build/tls_sycl.so and do NOT build tests/.
# Use `make -C tests` to build and run the verification/benchmark suite.
#
# Usage:
#   make -j$(nproc)             # Build build/tls_sycl.so (Sign = +1 default)
#   make SIGN=neg -j$(nproc)    # Build with Sign = -1 only
#   make SIGN=both -j$(nproc)   # Build with both Sign = +1 and Sign = -1
#   make objs -j$(nproc)        # Build core C++ object files only (reused by tests/)
#   make clean                  # Remove build artifacts

ACPP ?= acpp
CXX ?= clang++
STRIP ?= strip
ACPP_TARGETS ?= generic
SIGN ?= pos

ifeq ($(SIGN),pos)
SIGN_FLAGS := -DSYCL_FFT_ENABLE_POS_SIGN=1 -DSYCL_FFT_ENABLE_NEG_SIGN=0 -DSYCL_FFT_DEFAULT_SIGN=1
else ifeq ($(SIGN),neg)
SIGN_FLAGS := -DSYCL_FFT_ENABLE_POS_SIGN=0 -DSYCL_FFT_ENABLE_NEG_SIGN=1 -DSYCL_FFT_DEFAULT_SIGN=-1
else ifeq ($(SIGN),both)
SIGN_FLAGS := -DSYCL_FFT_ENABLE_POS_SIGN=1 -DSYCL_FFT_ENABLE_NEG_SIGN=1 -DSYCL_FFT_DEFAULT_SIGN=1
else
$(error Invalid SIGN='$(SIGN)'. Expected pos, neg, or both)
endif

CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -fno-fast-math -ffp-contract=fast $(SIGN_FLAGS)
PICFLAGS := -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections
INCLUDES := -I.
VERSION_SCRIPT := tls_sycl.map

ACPP_COMPILE_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS) $(PICFLAGS) $(INCLUDES)
ACPP_LINK_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS) -Wl,--version-script=$(VERSION_SCRIPT) -Wl,--gc-sections -Wl,-O1 -Wl,--as-needed

BUILD_DIR ?= build
OBJDIR ?= $(BUILD_DIR)/obj
LIBDIR ?= $(BUILD_DIR)
TARGET_SO := $(LIBDIR)/tls_sycl.so

COMMON_HEADERS := \
	common/math.hpp \
	common/layout.hpp \
	common/device.hpp

FFT_HEADERS := \
	$(COMMON_HEADERS) \
	bitrev/bitrev.hpp \
	fft/dif.hpp

NUFFT_BASE_HEADERS := \
	$(FFT_HEADERS) \
	nufft/nufft1.hpp

PERIODOGRAM_HEADERS := \
	$(NUFFT_BASE_HEADERS) \
	linalg/levinson.hpp \
	periodogram/subgrid.hpp \
	periodogram/preprocess.hpp \
	periodogram/planner.hpp \
	periodogram/engine.hpp

CORE_OBJS := \
	$(OBJDIR)/common/device.o \
	$(OBJDIR)/bitrev/bitrev.o \
	$(OBJDIR)/fft/dif_fp16_small.o \
	$(OBJDIR)/fft/dif_fp16_large.o \
	$(OBJDIR)/fft/dif_fp32_small.o \
	$(OBJDIR)/fft/dif_fp32_large.o \
	$(OBJDIR)/fft/dif_fp64_small.o \
	$(OBJDIR)/fft/dif_fp64_large.o \
	$(OBJDIR)/fft/dif_df_small.o \
	$(OBJDIR)/fft/dif_df_large.o \
	$(OBJDIR)/nufft/auxiliary.o \
	$(OBJDIR)/nufft/spread_fp16.o \
	$(OBJDIR)/nufft/spread_fp32.o \
	$(OBJDIR)/nufft/spread_fp64.o \
	$(OBJDIR)/nufft/spread_df.o \
	$(OBJDIR)/nufft/deconv_fp16.o \
	$(OBJDIR)/nufft/deconv_fp32.o \
	$(OBJDIR)/nufft/deconv_fp64.o \
	$(OBJDIR)/nufft/deconv_df.o \
	$(OBJDIR)/nufft/plan.o \
	$(OBJDIR)/periodogram/subgrid_fp16.o \
	$(OBJDIR)/periodogram/subgrid_fp32.o \
	$(OBJDIR)/periodogram/subgrid_fp64.o \
	$(OBJDIR)/periodogram/subgrid_df.o \
	$(OBJDIR)/linalg/levinson_fp16.o \
	$(OBJDIR)/linalg/levinson_fp32.o \
	$(OBJDIR)/linalg/levinson_fp64.o \
	$(OBJDIR)/linalg/levinson_df.o \
	$(OBJDIR)/periodogram/engine.o

IMPL_OBJS := \
	$(OBJDIR)/impl.o

LIB_OBJS := $(CORE_OBJS) $(IMPL_OBJS)

.PHONY: all objs clean

all: $(TARGET_SO)

objs: $(CORE_OBJS)

$(TARGET_SO): $(LIB_OBJS) $(VERSION_SCRIPT)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_LINK_FLAGS) -shared $(LIB_OBJS) -o $@
	$(STRIP) -s $@

$(OBJDIR)/impl.o: impl.cpp periodogram/engine.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(PICFLAGS) $(INCLUDES) -c $< -o $@

KERNEL_HASH_INPUTS := \
	common/math.hpp \
	common/layout.hpp \
	common/device.hpp \
	common/precision.hpp \
	bitrev/bitrev.hpp \
	bitrev/bitrev.cpp \
	fft/dif.hpp \
	fft/dif.cpp \
	nufft/pswf_coeffs.hpp \
	nufft/nufft1.hpp \
	nufft/auxiliary.cpp \
	nufft/spread.cpp \
	nufft/deconv.cpp \
	nufft/plan.cpp \
	periodogram/subgrid.hpp \
	periodogram/subgrid.cpp \
	linalg/levinson.hpp \
	linalg/levinson.cpp

KERNEL_CODE_HASH := $(shell (command -v sha256sum >/dev/null 2>&1 && sha256sum $(KERNEL_HASH_INPUTS) | sha256sum | cut -c1-16) || (command -v shasum >/dev/null 2>&1 && shasum -a 256 $(KERNEL_HASH_INPUTS) | shasum -a 256 | cut -c1-16) || echo "unknown")

$(OBJDIR)/common/device.o: common/device.cpp $(COMMON_HEADERS) $(KERNEL_HASH_INPUTS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -DSYCL_FFT_KERNEL_HASH=\"$(KERNEL_CODE_HASH)\" -c $< -o $@

$(OBJDIR)/bitrev/bitrev.o: bitrev/bitrev.cpp $(COMMON_HEADERS) bitrev/bitrev.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

PREC_FLAG_fp16 := -DSYCL_FFT_HALF
PREC_FLAG_fp32 := -DSYCL_FFT_FLOAT
PREC_FLAG_fp64 := -DSYCL_FFT_DOUBLE
PREC_FLAG_df   := -DSYCL_FFT_DF

$(OBJDIR)/fft/dif_%_small.o: fft/dif.cpp $(FFT_HEADERS) common/precision.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -DSYCL_FFT_INST_SMALL_ONLY -c $< -o $@

$(OBJDIR)/fft/dif_%_large.o: fft/dif.cpp $(FFT_HEADERS) common/precision.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -DSYCL_FFT_INST_LARGE_ONLY -c $< -o $@

$(OBJDIR)/nufft/auxiliary.o: nufft/auxiliary.cpp $(NUFFT_BASE_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/nufft/spread_%.o: nufft/spread.cpp $(NUFFT_BASE_HEADERS) common/precision.hpp nufft/pswf_coeffs.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/nufft/deconv_%.o: nufft/deconv.cpp $(NUFFT_BASE_HEADERS) common/precision.hpp nufft/pswf_coeffs.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/nufft/plan.o: nufft/plan.cpp $(NUFFT_BASE_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/periodogram/subgrid_%.o: periodogram/subgrid.cpp $(COMMON_HEADERS) common/precision.hpp periodogram/subgrid.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/linalg/levinson_%.o: linalg/levinson.cpp $(NUFFT_BASE_HEADERS) common/precision.hpp linalg/levinson.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/periodogram/engine.o: periodogram/engine.cpp $(PERIODOGRAM_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

clean:
	rm -f $(LIB_OBJS) $(TARGET_SO)
