# AdaptiveCpp (SYCL) Quadruple-Precision DIF Radix-2 FFT + NuFFT-1 Build System
# Supports FP16 (2xFP16), FP32, FP64, and Dekker Double-Float (DF = 2xFP32)
# Modular 21-TU layout isolating device probes, bit-reversal, small/large FFTs,
# NuFFT-1 aux/spread/deconv execution modules, and zero-HCF host planning.
#
# Usage:
#   make -j$(nproc)             # Parallel build (Sign = +1 default, fastest compile)
#   make SIGN=neg -j$(nproc)    # Build with Sign = -1 only
#   make SIGN=both -j$(nproc)   # Build with both Sign = +1 and Sign = -1
#   make run                    # Run on default SYCL device (GPU if available)
#   make run-cpu                # Force OpenMP host CPU backend
#   make run-gpu                # Force CUDA/HIP/Level-Zero GPU backend

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
ACPP_FLAGS := --acpp-targets=$(ACPP_TARGETS) $(CXXFLAGS)

OBJDIR ?= .
BINDIR ?= .
TARGET := $(BINDIR)/sycl_fft_bench

COMMON_HEADERS := \
	sycl_fft_math.hpp \
	sycl_fft_layout.hpp \
	sycl_fft_device.hpp \
	sycl_fft_precision.hpp

FFT_HEADERS := \
	$(COMMON_HEADERS) \
	sycl_fft_bitrev.hpp \
	sycl_fft_dif.hpp \
	sycl_fft_dif_impl.hpp

NUFFT_BASE_HEADERS := \
	$(FFT_HEADERS) \
	sycl_nufft_pswf_coeffs.hpp \
	sycl_nufft1.hpp

OBJS := \
	$(OBJDIR)/main.o \
	$(OBJDIR)/sycl_fft_device.o \
	$(OBJDIR)/sycl_fft_inst_bitrev.o \
	$(OBJDIR)/sycl_fft_inst_fp16_small.o \
	$(OBJDIR)/sycl_fft_inst_fp16_large.o \
	$(OBJDIR)/sycl_fft_inst_fp32_small.o \
	$(OBJDIR)/sycl_fft_inst_fp32_large.o \
	$(OBJDIR)/sycl_fft_inst_fp64_small.o \
	$(OBJDIR)/sycl_fft_inst_fp64_large.o \
	$(OBJDIR)/sycl_fft_inst_df_small.o \
	$(OBJDIR)/sycl_fft_inst_df_large.o \
	$(OBJDIR)/sycl_nufft1_aux.o \
	$(OBJDIR)/sycl_nufft1_spread_fp16.o \
	$(OBJDIR)/sycl_nufft1_spread_fp32.o \
	$(OBJDIR)/sycl_nufft1_spread_fp64.o \
	$(OBJDIR)/sycl_nufft1_spread_df.o \
	$(OBJDIR)/sycl_nufft1_deconv_fp16.o \
	$(OBJDIR)/sycl_nufft1_deconv_fp32.o \
	$(OBJDIR)/sycl_nufft1_deconv_fp64.o \
	$(OBJDIR)/sycl_nufft1_deconv_df.o \
	$(OBJDIR)/sycl_nufft1_plan.o

.PHONY: all clean run run-cpu run-gpu

all: $(TARGET)

$(TARGET): $(OBJS)
	$(ACPP) $(ACPP_FLAGS) $(OBJS) -o $(TARGET)

$(OBJDIR)/main.o: main.cpp $(NUFFT_BASE_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -DSYCL_FFT_USE_EXTERN_TEMPLATES=1 -DSYCL_FFT_EXTERN_DEVICE=1 -c main.cpp -o $@

$(OBJDIR)/sycl_fft_device.o: sycl_fft_device.cpp $(COMMON_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -c sycl_fft_device.cpp -o $@

$(OBJDIR)/sycl_fft_inst_bitrev.o: sycl_fft_inst_bitrev.cpp $(COMMON_HEADERS) sycl_fft_bitrev.hpp
	$(ACPP) $(ACPP_FLAGS) -c sycl_fft_inst_bitrev.cpp -o $@

$(OBJDIR)/sycl_fft_inst_%.o: sycl_fft_inst_%.cpp $(FFT_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

$(OBJDIR)/sycl_nufft1_aux.o: sycl_nufft1_aux.cpp sycl_fft_math.hpp sycl_nufft1.hpp
	$(ACPP) $(ACPP_FLAGS) -c sycl_nufft1_aux.cpp -o $@

$(OBJDIR)/sycl_nufft1_spread_%.o: sycl_nufft1_spread_%.cpp sycl_fft_math.hpp sycl_fft_precision.hpp sycl_nufft_pswf_coeffs.hpp sycl_nufft1.hpp sycl_nufft1_spread_impl.hpp
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

$(OBJDIR)/sycl_nufft1_deconv_%.o: sycl_nufft1_deconv_%.cpp sycl_fft_math.hpp sycl_fft_precision.hpp sycl_nufft_pswf_coeffs.hpp sycl_nufft1.hpp sycl_nufft1_deconv_impl.hpp
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

$(OBJDIR)/sycl_nufft1_plan.o: sycl_nufft1_plan.cpp $(NUFFT_BASE_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -c sycl_nufft1_plan.cpp -o $@

run: $(TARGET)
	$(TARGET)

run-cpu: $(TARGET)
	ACPP_VISIBILITY_MASK="omp" $(TARGET)

run-gpu: $(TARGET)
	ACPP_VISIBILITY_MASK="cuda;hip;ocl;ze" $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
