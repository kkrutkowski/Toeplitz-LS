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
PICFLAGS := -fPIC -fvisibility=hidden
INCLUDES := -I.

ACPP_COMPILE_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS) $(PICFLAGS) $(INCLUDES)
ACPP_LINK_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS)

BUILD_DIR ?= build
OBJDIR ?= $(BUILD_DIR)/obj
LIBDIR ?= $(BUILD_DIR)
TARGET_SO := $(LIBDIR)/tls_sycl.so

COMMON_HEADERS := \
	common/math.hpp \
	common/layout.hpp \
	common/device.hpp \
	common/precision.hpp

FFT_HEADERS := \
	$(COMMON_HEADERS) \
	bitrev/bitrev.hpp \
	fft/dif.hpp \
	fft/dif_impl.hpp

NUFFT_BASE_HEADERS := \
	$(FFT_HEADERS) \
	nufft/pswf_coeffs.hpp \
	nufft/nufft1.hpp

CORE_OBJS := \
	$(OBJDIR)/common/device.o \
	$(OBJDIR)/bitrev/inst.o \
	$(OBJDIR)/fft/inst_fp16_small.o \
	$(OBJDIR)/fft/inst_fp16_large.o \
	$(OBJDIR)/fft/inst_fp32_small.o \
	$(OBJDIR)/fft/inst_fp32_large.o \
	$(OBJDIR)/fft/inst_fp64_small.o \
	$(OBJDIR)/fft/inst_fp64_large.o \
	$(OBJDIR)/fft/inst_df_small.o \
	$(OBJDIR)/fft/inst_df_large.o \
	$(OBJDIR)/nufft/aux.o \
	$(OBJDIR)/nufft/spread_fp16.o \
	$(OBJDIR)/nufft/spread_fp32.o \
	$(OBJDIR)/nufft/spread_fp64.o \
	$(OBJDIR)/nufft/spread_df.o \
	$(OBJDIR)/nufft/deconv_fp16.o \
	$(OBJDIR)/nufft/deconv_fp32.o \
	$(OBJDIR)/nufft/deconv_fp64.o \
	$(OBJDIR)/nufft/deconv_df.o \
	$(OBJDIR)/nufft/plan.o

IMPL_OBJS := \
	$(OBJDIR)/impl.o

LIB_OBJS := $(CORE_OBJS) $(IMPL_OBJS)

.PHONY: all objs clean

all: $(TARGET_SO)

objs: $(CORE_OBJS)

$(TARGET_SO): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_LINK_FLAGS) -shared $(LIB_OBJS) -o $@

$(OBJDIR)/impl.o: impl.cpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/common/device.o: common/device.cpp $(COMMON_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/bitrev/inst.o: bitrev/inst.cpp $(COMMON_HEADERS) bitrev/bitrev.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

PREC_FLAG_fp16 := -DSYCL_FFT_HALF
PREC_FLAG_fp32 := -DSYCL_FFT_FLOAT
PREC_FLAG_fp64 := -DSYCL_FFT_DOUBLE
PREC_FLAG_df   := -DSYCL_FFT_DF

$(OBJDIR)/fft/inst_%_small.o: fft/inst.cpp $(FFT_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -DSYCL_FFT_INST_SMALL_ONLY -c $< -o $@

$(OBJDIR)/fft/inst_%_large.o: fft/inst.cpp $(FFT_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -DSYCL_FFT_INST_LARGE_ONLY -c $< -o $@

$(OBJDIR)/nufft/aux.o: nufft/aux.cpp common/math.hpp nufft/nufft1.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

$(OBJDIR)/nufft/spread_%.o: nufft/spread.cpp common/math.hpp common/precision.hpp nufft/pswf_coeffs.hpp nufft/nufft1.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/nufft/deconv_%.o: nufft/deconv.cpp common/math.hpp common/precision.hpp nufft/pswf_coeffs.hpp nufft/nufft1.hpp
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) $(PREC_FLAG_$*) -c $< -o $@

$(OBJDIR)/nufft/plan.o: nufft/plan.cpp $(NUFFT_BASE_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_COMPILE_FLAGS) -c $< -o $@

clean:
	rm -f $(LIB_OBJS) $(TARGET_SO)
