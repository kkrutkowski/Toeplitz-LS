CC ?= gcc
AR ?= ar

SRC_DIR := src
INC_DIR := include
LIB_DIR := lib
TMP_DIR := tmp
PY_DIST_DIR := toeplitz-ls
PY_PACKAGE_DIR := $(PY_DIST_DIR)/toeplitz_ls

TLS_SO := $(LIB_DIR)/tls.so
TLS_A := $(LIB_DIR)/tls.a
PY_TLS_SO := $(PY_PACKAGE_DIR)/tls.so
OLD_VARIANT_LIBS := $(LIB_DIR)/generic.so $(LIB_DIR)/native.so

GENERIC_FLAGS := -std=gnu11 -O3 -ffast-math -Wall -Wextra
NATIVE_FLAGS := $(GENERIC_FLAGS) -march=native -mtune=native
PICFLAGS := -fPIC
LDFLAGS_SHARED ?= -shared
LDLIBS := -lm
MAX_TWIDDLE_REUSE ?= 16
TWIDDLE_CFLAGS := -DMAX_TWIDDLE_REUSE=$(MAX_TWIDDLE_REUSE)

SCALING_GENERIC := $(INC_DIR)/scaling_generic.h
LIB_SOURCES := $(SRC_DIR)/nanofft.c $(SRC_DIR)/nufft1.c $(SRC_DIR)/linalg.c $(SRC_DIR)/chi2per.c $(SRC_DIR)/utils.c
SCALING_SOURCES := $(SRC_DIR)/nanofft.c $(SRC_DIR)/nufft1.c $(SRC_DIR)/scaling.c
PUBLIC_HEADERS := \
	$(INC_DIR)/nanofft.h \
	$(INC_DIR)/nanofft_precision.h \
	$(INC_DIR)/nufft1.h \
	$(INC_DIR)/linalg.h \
	$(INC_DIR)/tls.h \
	$(INC_DIR)/utils.h \
	$(SCALING_GENERIC)

GENERIC_DIR := $(TMP_DIR)/generic
NATIVE_DIR := $(TMP_DIR)/native
NATIVE_SCALING_DIR := $(TMP_DIR)/native-scaling

GENERIC_SCALING_HEADER := $(GENERIC_DIR)/scaling.h
NATIVE_SCALING_HEADER := $(NATIVE_DIR)/scaling.h

GENERIC_CFLAGS := $(GENERIC_FLAGS) $(PICFLAGS) $(TWIDDLE_CFLAGS) -I$(INC_DIR) -I$(GENERIC_DIR)
NATIVE_CFLAGS := $(NATIVE_FLAGS) $(PICFLAGS) $(TWIDDLE_CFLAGS) -I$(INC_DIR) -I$(NATIVE_DIR)

GENERIC_OBJS := \
	$(GENERIC_DIR)/nanofftf.o \
	$(GENERIC_DIR)/nanofft.o \
	$(GENERIC_DIR)/nanofftdd.o \
	$(GENERIC_DIR)/tlsf_nufft1.o \
	$(GENERIC_DIR)/tls_nufft1.o \
	$(GENERIC_DIR)/tlsdd_nufft1.o \
	$(GENERIC_DIR)/tlsf_linalg.o \
	$(GENERIC_DIR)/tls_linalg.o \
	$(GENERIC_DIR)/tlsdd_linalg.o \
	$(GENERIC_DIR)/tlsf_chi2per.o \
	$(GENERIC_DIR)/tls_chi2per.o \
	$(GENERIC_DIR)/tlsdd_chi2per.o \
	$(GENERIC_DIR)/tlsf_utils.o \
	$(GENERIC_DIR)/tls_utils.o \
	$(GENERIC_DIR)/tlsdd_utils.o

NATIVE_OBJS := \
	$(NATIVE_DIR)/nanofftf.o \
	$(NATIVE_DIR)/nanofft.o \
	$(NATIVE_DIR)/nanofftdd.o \
	$(NATIVE_DIR)/tlsf_nufft1.o \
	$(NATIVE_DIR)/tls_nufft1.o \
	$(NATIVE_DIR)/tlsdd_nufft1.o \
	$(NATIVE_DIR)/tlsf_linalg.o \
	$(NATIVE_DIR)/tls_linalg.o \
	$(NATIVE_DIR)/tlsdd_linalg.o \
	$(NATIVE_DIR)/tlsf_chi2per.o \
	$(NATIVE_DIR)/tls_chi2per.o \
	$(NATIVE_DIR)/tlsdd_chi2per.o \
	$(NATIVE_DIR)/tlsf_utils.o \
	$(NATIVE_DIR)/tls_utils.o \
	$(NATIVE_DIR)/tlsdd_utils.o

.PHONY: all generic native clean
.DELETE_ON_ERROR:

all: generic

generic: $(LIB_SOURCES) $(PUBLIC_HEADERS) | $(LIB_DIR) $(PY_PACKAGE_DIR)
	$(MAKE) $(GENERIC_OBJS)
	$(AR) rcs $(TLS_A) $(GENERIC_OBJS)
	$(CC) $(LDFLAGS_SHARED) -o $(TLS_SO) $(GENERIC_OBJS) $(LDLIBS)
	cp $(TLS_SO) $(PY_TLS_SO)
	rm -rf $(GENERIC_DIR)
	rmdir $(TMP_DIR) 2>/dev/null || true

native: $(LIB_SOURCES) $(SCALING_SOURCES) $(PUBLIC_HEADERS) | $(LIB_DIR) $(PY_PACKAGE_DIR)
	$(MAKE) $(NATIVE_OBJS)
	$(AR) rcs $(TLS_A) $(NATIVE_OBJS)
	$(CC) $(LDFLAGS_SHARED) -o $(TLS_SO) $(NATIVE_OBJS) $(LDLIBS)
	cp $(TLS_SO) $(PY_TLS_SO)
	rm -rf $(NATIVE_DIR)
	rmdir $(TMP_DIR) 2>/dev/null || true

$(GENERIC_SCALING_HEADER): $(SCALING_GENERIC) | $(GENERIC_DIR)
	cp $< $@

$(NATIVE_SCALING_HEADER): | $(NATIVE_DIR) $(NATIVE_SCALING_DIR)
	rm -f $@
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -c $(SRC_DIR)/nanofft.c -o $(NATIVE_SCALING_DIR)/nanofftf.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE -c $(SRC_DIR)/nanofft.c -o $(NATIVE_SCALING_DIR)/nanofft.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE_DOUBLE -c $(SRC_DIR)/nanofft.c -o $(NATIVE_SCALING_DIR)/nanofftdd.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -c $(SRC_DIR)/nufft1.c -o $(NATIVE_SCALING_DIR)/tlsf_nufft1.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE -c $(SRC_DIR)/nufft1.c -o $(NATIVE_SCALING_DIR)/tls_nufft1.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE_DOUBLE -c $(SRC_DIR)/nufft1.c -o $(NATIVE_SCALING_DIR)/tlsdd_nufft1.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofftf.o $(NATIVE_SCALING_DIR)/tlsf_nufft1.o -o $(NATIVE_SCALING_DIR)/tlsf_nufft_scaling $(LDLIBS)
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofft.o $(NATIVE_SCALING_DIR)/tls_nufft1.o -o $(NATIVE_SCALING_DIR)/tls_nufft_scaling $(LDLIBS)
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE_DOUBLE -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofftdd.o $(NATIVE_SCALING_DIR)/tlsdd_nufft1.o -o $(NATIVE_SCALING_DIR)/tlsdd_nufft_scaling $(LDLIBS)
	cd $(NATIVE_DIR) && ../native-scaling/tlsf_nufft_scaling
	cd $(NATIVE_DIR) && ../native-scaling/tls_nufft_scaling
	cd $(NATIVE_DIR) && ../native-scaling/tlsdd_nufft_scaling
	rm -rf $(NATIVE_SCALING_DIR)

$(GENERIC_DIR)/nanofftf.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/nanofft.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/nanofftdd.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/nanofftf.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/nanofft.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/nanofftdd.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(LIB_DIR) $(PY_PACKAGE_DIR) $(GENERIC_DIR) $(NATIVE_DIR) $(NATIVE_SCALING_DIR):
	mkdir -p $@

clean:
	rm -rf $(TMP_DIR)
	rm -f $(TLS_SO) $(TLS_A) $(PY_TLS_SO) $(OLD_VARIANT_LIBS)
