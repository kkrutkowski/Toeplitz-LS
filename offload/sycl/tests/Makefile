# Toeplitz-LS SYCL Offload Module — Verification & Benchmark Harness
#
# Reuses compiled C++ module object files from ../build/obj/ (via `make -C .. objs`)
# so SYCL translation units are never double-compiled between release and test builds.
#
# Usage:
#   make -j$(nproc)             # Build build/sycl_bench (Sign = +1 default)
#   make SIGN=both -j$(nproc)   # Build with both Sign = +1 and Sign = -1
#   make run                    # Run on default SYCL device (GPU if available)
#   make run-cpu                # Force OpenMP host CPU backend
#   make run-gpu                # Force CUDA/HIP/Level-Zero GPU backend
#   make clean                  # Remove test build artifacts (leaves ../build/ intact)

ROOT := ..
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
ACPP_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS) -I$(ROOT)

OBJDIR ?= $(ROOT)/build/obj
TEST_BUILD ?= build
TARGET := $(TEST_BUILD)/sycl_bench

COMMON_HEADERS := \
	$(ROOT)/common/math.hpp \
	$(ROOT)/common/layout.hpp \
	$(ROOT)/common/device.hpp

FFT_HEADERS := \
	$(COMMON_HEADERS) \
	$(ROOT)/bitrev/bitrev.hpp \
	$(ROOT)/fft/dif.hpp

NUFFT_BASE_HEADERS := \
	$(FFT_HEADERS) \
	$(ROOT)/nufft/nufft1.hpp

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
	$(OBJDIR)/nufft/aux.o \
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

TEST_OBJS := $(TEST_BUILD)/main.o

.PHONY: all core_objs clean run run-cpu run-gpu

all: core_objs $(TEST_OBJS)
	@$(MAKE) --no-print-directory $(TARGET) ACPP="$(ACPP)" ACPP_TARGETS="$(ACPP_TARGETS)" SIGN="$(SIGN)" OBJDIR="$(OBJDIR)" TEST_BUILD="$(TEST_BUILD)"

core_objs:
	@$(MAKE) -C $(ROOT) ACPP="$(ACPP)" ACPP_TARGETS="$(ACPP_TARGETS)" SIGN="$(SIGN)" OBJDIR="$(abspath $(OBJDIR))" objs

$(TARGET): $(TEST_OBJS) $(CORE_OBJS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_FLAGS) $(TEST_OBJS) $(CORE_OBJS) -o $@

$(TEST_BUILD)/main.o: main.cpp $(NUFFT_BASE_HEADERS)
	@mkdir -p $(dir $@)
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

run: all
	$(TARGET)

run-cpu: all
	ACPP_VISIBILITY_MASK="omp" $(TARGET)

run-gpu: all
	ACPP_VISIBILITY_MASK="cuda;hip;ocl;ze" $(TARGET)

clean:
	rm -f $(TEST_OBJS) $(TARGET)
