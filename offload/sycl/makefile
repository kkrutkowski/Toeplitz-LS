# AdaptiveCpp (SYCL) Quadruple-Precision DIF Radix-2 FFT + NuFFT-1 Build System
# Supports FP16 (2xFP16), FP32, FP64, and Dekker Double-Float (DF = 2xFP32)
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

TARGET := sycl_fft_bench

FFT_HEADERS := \
	sycl_fft_dif.hpp \
	sycl_fft_dif_impl.hpp \
	sycl_fft_bitrev.hpp \
	sycl_fft_math.hpp \
	sycl_fft_layout.hpp \
	sycl_fft_device.hpp \
	sycl_fft_precision.hpp

NUFFT_HEADERS := \
	$(FFT_HEADERS) \
	sycl_nufft_pswf_coeffs.hpp \
	sycl_nufft1.hpp \
	sycl_nufft1_impl.hpp

OBJS := \
	main.o \
	sycl_fft_inst_bitrev.o \
	sycl_fft_inst_fp16.o \
	sycl_fft_inst_fp32.o \
	sycl_fft_inst_fp64.o \
	sycl_fft_inst_df.o \
	sycl_nufft1_inst_fp16.o \
	sycl_nufft1_inst_fp32.o \
	sycl_nufft1_inst_fp64.o \
	sycl_nufft1_inst_df.o

.PHONY: all clean run run-cpu run-gpu

all: $(TARGET)

$(TARGET): $(OBJS)
	$(ACPP) $(ACPP_FLAGS) $(OBJS) -o $(TARGET)

main.o: main.cpp $(NUFFT_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -DSYCL_FFT_USE_EXTERN_TEMPLATES=1 -c main.cpp -o $@

sycl_fft_inst_%.o: sycl_fft_inst_%.cpp $(FFT_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

sycl_nufft1_inst_%.o: sycl_nufft1_inst_%.cpp $(NUFFT_HEADERS)
	$(ACPP) $(ACPP_FLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

run-cpu: $(TARGET)
	ACPP_VISIBILITY_MASK="omp" ./$(TARGET)

run-gpu: $(TARGET)
	ACPP_VISIBILITY_MASK="cuda;hip;ocl;ze" ./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)
