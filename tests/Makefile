CC ?= gcc

ROOT := ..
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
# Fused-twiddle FFT depth per precision.  Left unset on purpose so nanofft.h
# selects the per-precision default; override any of them to pin a depth.
# Double-double is compensated arithmetic; see the top-level Makefile.
DD_FLAGS := -fno-fast-math
FUSE_CFLAGS := $(if $(FFT_FUSE_FLOAT),-DNANOFFT_FFT_FUSE_FLOAT=$(FFT_FUSE_FLOAT)) \
               $(if $(FFT_FUSE_DOUBLE),-DNANOFFT_FFT_FUSE_DOUBLE=$(FFT_FUSE_DOUBLE)) \
               $(if $(FFT_FUSE_DD),-DNANOFFT_FFT_FUSE_DD=$(FFT_FUSE_DD)) \
               $(if $(FFT_FUSE_HALF),-DNANOFFT_FFT_FUSE_HALF=$(FFT_FUSE_HALF))
BUILD := build
INC := -I$(ROOT)/include -I$(BUILD)
LDLIBS := -lm

HOST_ARCH := $(shell uname -m)
ifeq ($(HOST_ARCH),x86_64)
NATIVE_FP16_FLAGS := -march=x86-64-v4 -mavx512fp16
CC_HAS_NATIVE_FP16 := $(shell printf '_Float16 f(_Float16 x){return x;}\n' | $(CC) -x c -c $(NATIVE_FP16_FLAGS) - -o /dev/null 2>/dev/null && echo 1 || echo 0)
else ifeq ($(HOST_ARCH),aarch64)
NATIVE_FP16_FLAGS := -march=armv8.2-a+fp16
CC_HAS_NATIVE_FP16 := $(shell printf '_Float16 f(_Float16 x){return x;}\n' | $(CC) -x c -c $(NATIVE_FP16_FLAGS) - -o /dev/null 2>/dev/null && echo 1 || echo 0)
else
NATIVE_FP16_FLAGS :=
CC_HAS_NATIVE_FP16 := 0
endif
TEST_ALREADY_FP16 := $(shell $(CC) -dM -E $(CFLAGS) -x c /dev/null 2>/dev/null | grep -qE '__AVX512FP16__|__ARM_FEATURE_FP16_VECTOR_ARITHMETIC' && echo 1 || echo 0)
TEST_EMBED_FP16 := $(if $(filter 1,$(TEST_ALREADY_FP16)),0,$(CC_HAS_NATIVE_FP16))

OBJS := \
	$(BUILD)/nanofftf.o \
	$(BUILD)/nanofft.o \
	$(BUILD)/nanofftdd.o \
	$(BUILD)/nanoffth.o \
	$(BUILD)/tlsf_nufft1.o \
	$(BUILD)/tls_nufft1.o \
	$(BUILD)/tlsdd_nufft1.o \
	$(BUILD)/tlsh_nufft1.o \
	$(BUILD)/tlsf_linalg.o \
	$(BUILD)/tls_linalg.o \
	$(BUILD)/tlsdd_linalg.o \
	$(BUILD)/tlsh_linalg.o \
	$(BUILD)/tlsf_chi2per.o \
	$(BUILD)/tls_chi2per.o \
	$(BUILD)/tlsdd_chi2per.o \
	$(BUILD)/tlsh_chi2per.o \
	$(BUILD)/tlsf_utils.o \
	$(BUILD)/tls_utils.o \
	$(BUILD)/tlsdd_utils.o \
	$(BUILD)/tuning.o \
	$(BUILD)/cpu_features.o

ifeq ($(TEST_EMBED_FP16),1)
OBJS += \
	$(BUILD)/nanoffth_nat.o \
	$(BUILD)/tlsh_nat_nufft1.o \
	$(BUILD)/tlsh_nat_linalg.o \
	$(BUILD)/tlsh_nat_chi2per.o
TEST_HALF_DISPATCH_FLAG := -DTLS_HALF_HAS_NATIVE_EMBED
else
TEST_HALF_DISPATCH_FLAG :=
endif
TARGETS := $(BUILD)/check_nanofft $(BUILD)/check_vecmath $(BUILD)/check_periodogram $(BUILD)/dump_periodogram $(BUILD)/bench_fft $(BUILD)/bench_permute $(BUILD)/bench_fuse $(BUILD)/bench_periodogram $(BUILD)/bench_ihs $(BUILD)/bench_public $(BUILD)/bench_spread $(BUILD)/bench_vectorize

.PHONY: all check bench bench_permute permute bench_fuse fuse tune clean
.DELETE_ON_ERROR:

all: $(TARGETS)

# The cost model in chi2per.c includes a generated scaling.h; the benchmark
# builds reuse the generic calibration instead of running the slow native
# calibration (see the top-level Makefile).
$(BUILD)/scaling.h: $(ROOT)/include/scaling_generic.h | $(BUILD)
	cp $< $@

$(BUILD):
	mkdir -p $@

$(BUILD)/nanofftf.o: $(ROOT)/src/nanofft.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/nanofft.o: $(ROOT)/src/nanofft.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/nanofftdd.o: $(ROOT)/src/nanofft.c | $(BUILD)
	$(CC) $(CFLAGS) $(DD_FLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@
$(BUILD)/nanoffth.o: $(ROOT)/src/nanofft.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -c $< -o $@
$(BUILD)/nanoffth_nat.o: $(ROOT)/src/nanofft.c | $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE_FP16_FLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -DHALF_NATIVE_EMBED -c $< -o $@

$(BUILD)/tlsf_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(DD_FLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@
$(BUILD)/tlsh_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -c $< -o $@
$(BUILD)/tlsh_nat_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE_FP16_FLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -DHALF_NATIVE_EMBED -c $< -o $@

$(BUILD)/tlsf_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(DD_FLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@
$(BUILD)/tlsh_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -c $< -o $@
$(BUILD)/tlsh_nat_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE_FP16_FLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -DHALF_NATIVE_EMBED -c $< -o $@

$(BUILD)/tlsf_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(DD_FLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@
$(BUILD)/tlsh_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DHALF $(TEST_HALF_DISPATCH_FLAG) -c $< -o $@
$(BUILD)/tlsh_nat_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(NATIVE_FP16_FLAGS) $(FUSE_CFLAGS) $(INC) -DHALF -DHALF_NATIVE_EMBED -c $< -o $@

$(BUILD)/tlsf_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(DD_FLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tuning.o: $(ROOT)/src/tuning.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/cpu_features.o: $(ROOT)/src/cpu_features.c | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

$(BUILD)/check_nanofft: check_nanofft.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

# Header-only: it exercises the inline vector sqrt/exp helpers directly.
$(BUILD)/check_vecmath: check_vecmath.c $(ROOT)/include/nanofft.h $(ROOT)/include/nanofft_vecmath.h $(ROOT)/include/nanofft_exp_coeffs.h
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< -o $@ $(LDLIBS)

$(BUILD)/check_periodogram: check_periodogram.c $(BUILD)/scaling.h $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/dump_periodogram: dump_periodogram.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_fft: bench_fft.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_fuse: bench_fuse.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_permute: bench_permute.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_periodogram: bench_periodogram.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_ihs: bench_ihs.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_public: bench_public.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_spread: bench_spread.c $(BUILD)/scaling.h $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_vectorize: bench_vectorize.c $(BUILD)/scaling.h $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

bench_fuse: $(BUILD)/bench_fuse

fuse: $(BUILD)/bench_fuse
	./$(BUILD)/bench_fuse

# Per-grid fusion-depth table.  TUNE_OUT points at the directory the library
# build compiles against; the tuner itself must be built without the table, so
# it measures the depths instead of reading them back.
TUNE_OUT ?= ../tmp/native/nanofft_depth_table.h
TUNE_LAYOUTS ?= 16
TUNE_REPS ?= 8
TUNE_NOTE ?= $(CC) $(CFLAGS)
tune: $(BUILD)/bench_fuse
	./$(BUILD)/bench_fuse --tune --layouts $(TUNE_LAYOUTS) --reps $(TUNE_REPS) --emit $(TUNE_OUT) --note "$(TUNE_NOTE)"

bench_permute: $(BUILD)/bench_permute

permute: $(BUILD)/bench_permute
	./$(BUILD)/bench_permute

check: $(BUILD)/check_nanofft $(BUILD)/check_vecmath $(BUILD)/check_periodogram
	./$(BUILD)/check_vecmath
	./$(BUILD)/check_nanofft
	./$(BUILD)/check_periodogram

bench: all
	./$(BUILD)/bench_fft
	./$(BUILD)/bench_permute
	./$(BUILD)/bench_periodogram

clean:
	rm -rf $(BUILD)
