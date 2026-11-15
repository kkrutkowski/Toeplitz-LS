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

# Double-double is compensated arithmetic: -ffast-math would let the compiler
# reassociate the two-sum / two-product chains it depends on, so those
# translation units are built without it (the float and double ones keep it).
DD_FLAGS := -fno-fast-math
# The shipped shared library exports exactly the TLS_API surface and nothing
# else; the same objects also go into tls.a, where hidden visibility is inert.
SHARED_FLAGS := -DTLS_BUILD_SHARED -fvisibility=hidden
PICFLAGS := -fPIC
LDFLAGS_SHARED ?= -shared
LDLIBS := -lm
CLANG_FORMAT ?= clang-format
RUFF ?= ruff
C_SOURCES := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(INC_DIR)/*.h) $(wildcard tests/*.c) $(wildcard tests/*.h)

MAX_TWIDDLE_REUSE ?= 16
TWIDDLE_CFLAGS := -DMAX_TWIDDLE_REUSE=$(MAX_TWIDDLE_REUSE)

# Fused-twiddle FFT depth per precision.  Left unset so nanofft.h picks the
# per-precision default (floor(log2(VEC_LEN)) stages, one less for float, 2 for
# double-double); pin any of them to override:
#   make FFT_FUSE_FLOAT=2 FFT_FUSE_DOUBLE=4
FUSE_CFLAGS := $(if $(FFT_FUSE_FLOAT),-DNANOFFT_FFT_FUSE_FLOAT=$(FFT_FUSE_FLOAT)) \
               $(if $(FFT_FUSE_DOUBLE),-DNANOFFT_FFT_FUSE_DOUBLE=$(FFT_FUSE_DOUBLE)) \
               $(if $(FFT_FUSE_DD),-DNANOFFT_FFT_FUSE_DD=$(FFT_FUSE_DD))

SCALING_GENERIC := $(INC_DIR)/scaling_generic.h
LIB_SOURCES := $(SRC_DIR)/nanofft.c $(SRC_DIR)/nufft1.c $(SRC_DIR)/linalg.c $(SRC_DIR)/chi2per.c $(SRC_DIR)/utils.c $(SRC_DIR)/tuning.c $(SRC_DIR)/cpu_features.c $(SRC_DIR)/probe_arm64.c
SCALING_SOURCES := $(SRC_DIR)/nanofft.c $(SRC_DIR)/nufft1.c $(SRC_DIR)/scaling.c
PUBLIC_HEADERS := \
	$(INC_DIR)/nanofft.h \
	$(INC_DIR)/nanofft_exp_coeffs.h \
	$(INC_DIR)/nanofft_precision.h \
	$(INC_DIR)/nanofft_vecmath.h \
	$(INC_DIR)/nufft1.h \
	$(INC_DIR)/linalg.h \
	$(INC_DIR)/tls.h \
	$(INC_DIR)/tls_cpu.h \
	$(INC_DIR)/tls_export.h \
	$(INC_DIR)/tls_internal.h \
	$(INC_DIR)/tls_platform.h \
	$(INC_DIR)/tls_tuning.h \
	$(INC_DIR)/utils.h \
	$(SCALING_GENERIC)

# `make native` tunes the per-grid fusion depth on this machine unless TUNE=0.
TUNE ?= 1
TUNE_CPPFLAGS = $(if $(filter 1,$(TUNE)),-DNANOFFT_DEPTH_TABLE -I$(NATIVE_DIR))

GENERIC_DIR := $(TMP_DIR)/generic
NATIVE_DIR := $(TMP_DIR)/native
NATIVE_DEPTH_HEADER := $(NATIVE_DIR)/nanofft_depth_table.h
NATIVE_SCALING_DIR := $(TMP_DIR)/native-scaling

GENERIC_SCALING_HEADER := $(GENERIC_DIR)/scaling.h
NATIVE_SCALING_HEADER := $(NATIVE_DIR)/scaling.h

GENERIC_CFLAGS := $(GENERIC_FLAGS) $(SHARED_FLAGS) $(PICFLAGS) $(TWIDDLE_CFLAGS) $(FUSE_CFLAGS) -I$(INC_DIR) -I$(GENERIC_DIR)
NATIVE_CFLAGS := $(NATIVE_FLAGS) $(SHARED_FLAGS) $(PICFLAGS) $(TWIDDLE_CFLAGS) $(FUSE_CFLAGS) -I$(INC_DIR) -I$(NATIVE_DIR) $(TUNE_CPPFLAGS)

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
	$(GENERIC_DIR)/tlsdd_utils.o \
	$(GENERIC_DIR)/tuning.o \
	$(GENERIC_DIR)/cpu_features.o \
	$(GENERIC_DIR)/probe_arm64.o

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
	$(NATIVE_DIR)/tlsdd_utils.o \
	$(NATIVE_DIR)/tuning.o \
	$(NATIVE_DIR)/cpu_features.o \
	$(NATIVE_DIR)/probe_arm64.o

.PHONY: all generic native clean wheels wheels-check sdist tune dispatch-list toolchains format format-check

# --- release wheels ------------------------------------------------------
# Every target the 1.0 release ships.  Each one is built and ABI-checked
# independently, so `make wheels` can be pointed at a subset on a smaller
# machine (see dispatch/README.md).
WHEEL_TARGETS ?= linux-x86_64-glibc linux-x86_64-musl linux-aarch64-glibc linux-aarch64-musl win-amd64 win-arm64 macos-x86_64 macos-arm64

wheels: $(addprefix wheel-,$(WHEEL_TARGETS))

wheel-%:
	python3 -m dispatch.build --target $*
	python3 -m dispatch.wheel --target $*

wheels-check:
	python3 -m dispatch.check --all

# Source distribution: the C sources plus a PEP 517 backend that compiles the
# baseline library at install time.  Only used on platforms without a wheel.
sdist:
	python3 -m dispatch.sdist

dispatch-list:
	python3 -m dispatch.build --list

toolchains:
	python3 -m dispatch.toolchains

# Regenerate the per-grid fusion-depth table for this machine only.
tune:
	$(MAKE) -C tests BUILD=build-tune CFLAGS="$(NATIVE_FLAGS)" TUNE_OUT=$(abspath $(TMP_DIR))/nanofft_depth_table.h tune
.DELETE_ON_ERROR:

all: generic

generic: $(LIB_SOURCES) $(PUBLIC_HEADERS) | $(LIB_DIR) $(PY_PACKAGE_DIR)
	$(MAKE) $(GENERIC_OBJS)
	$(AR) rcs $(TLS_A) $(GENERIC_OBJS)
	$(CC) $(LDFLAGS_SHARED) -o $(TLS_SO) $(GENERIC_OBJS) $(LDLIBS)
	cp $(TLS_SO) $(PY_TLS_SO)
	rm -rf $(GENERIC_DIR)
	rmdir $(TMP_DIR) 2>/dev/null || true

native: $(LIB_SOURCES) $(SCALING_SOURCES) $(PUBLIC_HEADERS) $(NATIVE_DEPTH_HEADER) | $(LIB_DIR) $(PY_PACKAGE_DIR)
	$(MAKE) $(NATIVE_OBJS)
	$(AR) rcs $(TLS_A) $(NATIVE_OBJS)
	$(CC) $(LDFLAGS_SHARED) -o $(TLS_SO) $(NATIVE_OBJS) $(LDLIBS)
	cp $(TLS_SO) $(PY_TLS_SO)
	rm -rf $(NATIVE_DIR)
	rmdir $(TMP_DIR) 2>/dev/null || true

# Build the tuner with this build's flags, run it, and leave the table where
# the objects can include it.
$(NATIVE_DEPTH_HEADER): $(SRC_DIR)/nanofft.c $(SRC_DIR)/tuning.c $(INC_DIR)/nanofft.h | $(NATIVE_DIR)
ifeq ($(TUNE),1)
	$(MAKE) -C tests BUILD=build-tune CFLAGS="$(NATIVE_FLAGS)" TUNE_OUT=$(abspath $@) TUNE_NOTE="$(CC) $(NATIVE_FLAGS)" tune
else
	@echo "tune disabled: using the per-precision defaults"
	@printf '/* tuning disabled */\n' > $@
endif

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
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -c $(SRC_DIR)/tuning.c -o $(NATIVE_SCALING_DIR)/tuning.o
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofftf.o $(NATIVE_SCALING_DIR)/tlsf_nufft1.o $(NATIVE_SCALING_DIR)/tuning.o -o $(NATIVE_SCALING_DIR)/tlsf_nufft_scaling $(LDLIBS)
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofft.o $(NATIVE_SCALING_DIR)/tls_nufft1.o $(NATIVE_SCALING_DIR)/tuning.o -o $(NATIVE_SCALING_DIR)/tls_nufft_scaling $(LDLIBS)
	$(CC) $(NATIVE_FLAGS) -I$(INC_DIR) -D DOUBLE_DOUBLE -DSAVE $(SRC_DIR)/scaling.c $(NATIVE_SCALING_DIR)/nanofftdd.o $(NATIVE_SCALING_DIR)/tlsdd_nufft1.o $(NATIVE_SCALING_DIR)/tuning.o -o $(NATIVE_SCALING_DIR)/tlsdd_nufft_scaling $(LDLIBS)
	cd $(NATIVE_DIR) && ../native-scaling/tlsf_nufft_scaling
	cd $(NATIVE_DIR) && ../native-scaling/tls_nufft_scaling
	cd $(NATIVE_DIR) && ../native-scaling/tlsdd_nufft_scaling
	rm -rf $(NATIVE_SCALING_DIR)

$(GENERIC_DIR)/nanofftf.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/nanofft.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/nanofftdd.o: $(SRC_DIR)/nanofft.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_nufft1.o: $(SRC_DIR)/nufft1.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_linalg.o: $(SRC_DIR)/linalg.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_chi2per.o: $(SRC_DIR)/chi2per.c $(GENERIC_SCALING_HEADER)
	$(CC) $(GENERIC_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsf_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/tls_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -D DOUBLE -c $< -o $@

$(GENERIC_DIR)/tlsdd_utils.o: $(SRC_DIR)/utils.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(GENERIC_DIR)/tuning.o: $(SRC_DIR)/tuning.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/cpu_features.o: $(SRC_DIR)/cpu_features.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(GENERIC_DIR)/probe_arm64.o: $(SRC_DIR)/probe_arm64.c | $(GENERIC_DIR)
	$(CC) $(GENERIC_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/nanofftf.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/nanofft.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/nanofftdd.o: $(SRC_DIR)/nanofft.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_nufft1.o: $(SRC_DIR)/nufft1.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_linalg.o: $(SRC_DIR)/linalg.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_chi2per.o: $(SRC_DIR)/chi2per.c $(NATIVE_SCALING_HEADER)
	$(CC) $(NATIVE_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsf_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/tls_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -D DOUBLE -c $< -o $@

$(NATIVE_DIR)/tlsdd_utils.o: $(SRC_DIR)/utils.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) $(DD_FLAGS) -D DOUBLE_DOUBLE -c $< -o $@

$(NATIVE_DIR)/tuning.o: $(SRC_DIR)/tuning.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/cpu_features.o: $(SRC_DIR)/cpu_features.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(NATIVE_DIR)/probe_arm64.o: $(SRC_DIR)/probe_arm64.c | $(NATIVE_DIR)
	$(CC) $(NATIVE_CFLAGS) -c $< -o $@

$(LIB_DIR) $(PY_PACKAGE_DIR) $(GENERIC_DIR) $(NATIVE_DIR) $(NATIVE_SCALING_DIR):
	mkdir -p $@

clean:
	rm -rf $(TMP_DIR) .build/sdist
	rm -f $(TLS_SO) $(TLS_A) $(PY_TLS_SO) $(OLD_VARIANT_LIBS)

# --- formatting -----------------------------------------------------------
# One style for both halves of the tree: .clang-format (C) and ruff.toml
# (Python) share the 160-column budget and 4-space indentation.
format:
	$(CLANG_FORMAT) -i $(C_SOURCES)
	$(RUFF) format $(PY_DIST_DIR) dispatch
	$(RUFF) check --fix $(PY_DIST_DIR) dispatch

format-check:
	$(CLANG_FORMAT) --dry-run --Werror $(C_SOURCES)
	$(RUFF) format --check $(PY_DIST_DIR) dispatch
	$(RUFF) check $(PY_DIST_DIR) dispatch
