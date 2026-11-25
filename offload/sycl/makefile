.PHONY: generic blackwell check clean

ACPP        ?= acpp
CXXFLAGS    ?= -O3 -Wall -Wextra -fexcess-precision=16
# Sign convention build selector (Sign is a runtime kernel argument, so SIGN=both
# compiles the exact same kernel set, HCF footprint, and cold JIT cost as SIGN=pos):
#   SIGN=pos  (default): expose +1 sign in DifFftPlan/CLI
#   SIGN=neg           : expose -1 sign in DifFftPlan/CLI
#   SIGN=both          : expose both +1 and -1 signs in DifFftPlan/CLI
SIGN        ?= pos

ifeq ($(SIGN),pos)
SIGN_FLAGS  := -DSYCL_FFT_ENABLE_POS_SIGN=1 -DSYCL_FFT_ENABLE_NEG_SIGN=0
CHECK_SIGN_ARG :=
else ifeq ($(SIGN),neg)
SIGN_FLAGS  := -DSYCL_FFT_ENABLE_POS_SIGN=0 -DSYCL_FFT_ENABLE_NEG_SIGN=1
CHECK_SIGN_ARG :=
else ifeq ($(SIGN),both)
SIGN_FLAGS  := -DSYCL_FFT_ENABLE_POS_SIGN=1 -DSYCL_FFT_ENABLE_NEG_SIGN=1
CHECK_SIGN_ARG := --verify-both-signs
else
$(error Invalid SIGN='$(SIGN)' (expected pos, neg, or both))
endif

BUILD_ROOT  := build
BUILD_DIR   := $(BUILD_ROOT)/$(SIGN)
HDRS        := sycl_fft_math.hpp sycl_fft_layout.hpp sycl_fft_device.hpp \
               sycl_fft_bitrev.hpp sycl_fft_precision.hpp sycl_fft_dif_impl.hpp sycl_fft_dif.hpp
INST_SRCS   := sycl_fft_inst_bitrev.cpp sycl_fft_inst_fp16.cpp sycl_fft_inst_fp32.cpp \
               sycl_fft_inst_fp64.cpp sycl_fft_inst_df.cpp
INST_OBJS   := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(INST_SRCS))
MAIN_OBJ    := $(BUILD_DIR)/main.o

# Default parallel build: compiles 6 independent TUs concurrently (each with its own small HCF module)
generic:
	@$(MAKE) SIGN=$(SIGN) -j$$(nproc) $(BUILD_DIR)/main_generic
	@cp $(BUILD_DIR)/main_generic main

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main.o: main.cpp $(HDRS) | $(BUILD_DIR)
	$(ACPP) --acpp-targets=generic $(CXXFLAGS) $(SIGN_FLAGS) -DSYCL_FFT_USE_EXTERN_TEMPLATES -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(HDRS) | $(BUILD_DIR)
	$(ACPP) --acpp-targets=generic $(CXXFLAGS) $(SIGN_FLAGS) -c $< -o $@

$(BUILD_DIR)/main_generic: $(MAIN_OBJ) $(INST_OBJS)
	$(ACPP) --acpp-targets=generic $(CXXFLAGS) $^ -o $@

blackwell:
	PATH="$(ACPP_CUDA_PATH)/bin:$$PATH" $(ACPP) --acpp-targets=cuda:sm_120 $(CXXFLAGS) $(SIGN_FLAGS) main.cpp -o main

clean:
	rm -rf $(BUILD_ROOT) main
check: generic
	./main --list-devices
	@for dev in cuda opencl omp; do \
		if ./main --list-devices | grep -q "backend=$$dev "; then \
			if [ "$$dev" = "opencl" ] && ! command -v llvm-spirv >/dev/null 2>&1; then \
				echo "=== Skipping opencl (llvm-spirv not installed) ==="; \
				continue; \
			fi; \
			echo "=== Testing backend: $$dev ==="; \
			if [ "$$dev" = "cuda" ]; then \
				./main --device $$dev --verify-only $(CHECK_SIGN_ARG) --prec all --min-logN 10 --max-logN 18 || exit 1; \
			else \
				./main --device $$dev --verify-only --prec all --min-logN 10 --max-logN 14 --no-autotune || exit 1; \
			fi; \
		fi; \
	done
