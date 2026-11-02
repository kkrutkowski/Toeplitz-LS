CC ?= gcc

ROOT := ..
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
# Fused-twiddle FFT depth per precision (1 = no fusion .. 4 = radix-16 pass).
FFT_FUSE_FLOAT ?= 3
FFT_FUSE_DOUBLE ?= 3
FFT_FUSE_DD ?= 3
FUSE_CFLAGS := -DNANOFFT_FFT_FUSE_FLOAT=$(FFT_FUSE_FLOAT) -DNANOFFT_FFT_FUSE_DOUBLE=$(FFT_FUSE_DOUBLE) -DNANOFFT_FFT_FUSE_DD=$(FFT_FUSE_DD)
BUILD := build
INC := -I$(ROOT)/include -I$(BUILD)
LDLIBS := -lm

OBJS := \
	$(BUILD)/nanofftf.o \
	$(BUILD)/nanofft.o \
	$(BUILD)/nanofftdd.o \
	$(BUILD)/tlsf_nufft1.o \
	$(BUILD)/tls_nufft1.o \
	$(BUILD)/tlsdd_nufft1.o \
	$(BUILD)/tlsf_linalg.o \
	$(BUILD)/tls_linalg.o \
	$(BUILD)/tlsdd_linalg.o \
	$(BUILD)/tlsf_chi2per.o \
	$(BUILD)/tls_chi2per.o \
	$(BUILD)/tlsdd_chi2per.o \
	$(BUILD)/tlsf_utils.o \
	$(BUILD)/tls_utils.o \
	$(BUILD)/tlsdd_utils.o \
	$(BUILD)/tuning.o

TARGETS := $(BUILD)/check_nanofft $(BUILD)/check_periodogram $(BUILD)/dump_periodogram $(BUILD)/bench_fft $(BUILD)/bench_periodogram $(BUILD)/bench_public

.PHONY: all check bench clean
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
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tlsf_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_nufft1.o: $(ROOT)/src/nufft1.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tlsf_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_linalg.o: $(ROOT)/src/linalg.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tlsf_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_chi2per.o: $(ROOT)/src/chi2per.c $(BUILD)/scaling.h | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tlsf_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@
$(BUILD)/tls_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE -c $< -o $@
$(BUILD)/tlsdd_utils.o: $(ROOT)/src/utils.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -DDOUBLE_DOUBLE -c $< -o $@

$(BUILD)/tuning.o: $(ROOT)/src/tuning.c | $(BUILD)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) -c $< -o $@

$(BUILD)/check_nanofft: check_nanofft.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/check_periodogram: check_periodogram.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/dump_periodogram: dump_periodogram.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_fft: bench_fft.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_periodogram: bench_periodogram.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

$(BUILD)/bench_public: bench_public.c $(OBJS)
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) $(INC) $< $(OBJS) -o $@ $(LDLIBS)

check: $(BUILD)/check_nanofft $(BUILD)/check_periodogram
	./$(BUILD)/check_nanofft
	./$(BUILD)/check_periodogram

bench: all
	./$(BUILD)/bench_fft
	./$(BUILD)/bench_periodogram

clean:
	rm -rf $(BUILD)
