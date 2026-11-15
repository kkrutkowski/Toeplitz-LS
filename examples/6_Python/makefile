CC ?= gcc
AR ?= ar

ROOT := ../..
BUILD_DIR := build
INCLUDES := -I$(ROOT)/include -I$(BUILD_DIR)
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -flto -fno-semantic-interposition -Wall -Wextra
PICFLAGS := -fPIC
# Double-double is compensated arithmetic; see the top-level Makefile.
DD_FLAGS := -fno-fast-math
SHARED_FLAGS := -DTLS_BUILD_SHARED -fvisibility=hidden
LDFLAGS_SHARED ?= -flto -shared
LDLIBS ?= -lm

COMMON_CFLAGS := $(CFLAGS) $(SHARED_FLAGS) $(PICFLAGS) $(INCLUDES)
SCALING_TEMPLATE := $(ROOT)/include/scaling_generic.h
SCALING_HEADER := $(BUILD_DIR)/scaling.h

NANOFFT_OBJS := nanofftf.o nanofft.o nanofftdd.o
NUFFT_OBJS := tlsf_nufft1.o tls_nufft1.o tlsdd_nufft1.o
LINALG_OBJS := tlsf_linalg.o tls_linalg.o tlsdd_linalg.o
CHI2_OBJS := tlsf_chi2per.o tls_chi2per.o tlsdd_chi2per.o
UTIL_OBJS := utils.o
OBJS := $(NANOFFT_OBJS) $(NUFFT_OBJS) $(LINALG_OBJS) $(CHI2_OBJS) $(UTIL_OBJS) tuning.o

.PHONY: all static shared clean

all: static shared
	rm -f $(OBJS)
	rm -rf $(BUILD_DIR)

static: tls.a

shared: tls.so

tls.a: $(OBJS)
	$(AR) rcs $@ $^

tls.so: $(OBJS)
	$(CC) $(LDFLAGS_SHARED) -o $@ $^ $(LDLIBS)
	cp $@ $(ROOT)/toeplitz-ls/toeplitz_ls/tls.so

$(SCALING_HEADER): $(SCALING_TEMPLATE)
	mkdir -p $(BUILD_DIR)
	cp $< $@

nanofftf.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

nanofft.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

nanofftdd.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

tlsf_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

tuning.o: $(ROOT)/src/tuning.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tlsf_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

tlsf_chi2per.o: $(ROOT)/src/chi2per.c $(SCALING_HEADER)
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_chi2per.o: $(ROOT)/src/chi2per.c $(SCALING_HEADER)
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_chi2per.o: $(ROOT)/src/chi2per.c $(SCALING_HEADER)
	$(CC) $(COMMON_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

utils.o: $(ROOT)/src/utils.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) tls.a tls.so
	rm -rf $(BUILD_DIR)
