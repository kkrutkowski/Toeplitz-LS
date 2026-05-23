CC ?= cc
PYTHON ?= python3
CFLAGS ?= -O3 -ffast-math -march=native -flto -fPIC -Wall -Wextra
LDFLAGS_SHARED ?= -shared

ROOT := ../..
BUILD_DIR := build
OUT_DIR ?= out
PHOT_DIR ?= phot
INDEX_PATH ?= index.tsv
BENCH_ARGS ?=

AOV_URL ?= http://users.camk.edu.pl/alex/soft/aovdist.tgz
FASTCHI2_URL ?= https://web.archive.org/web/20250525051459/http://public.lanl.gov/palmer/fastchi/FastChi2-1.03.tar.gz
AOV_ARCHIVE := aovdist.tgz
FASTCHI2_ARCHIVE := FastChi2-1.03.tar.gz
AOV_DIR := aovdist
FASTCHI2_DIR := FastChi2-1.03
AOV_STAMP := .unpacked-aovdist
FASTCHI2_STAMP := .unpacked-fastchi2_103

AOV_LIB := $(BUILD_DIR)/libaovdist.so
FASTCHI2_LIB := $(BUILD_DIR)/libfastchi2_103.so
PYTHONPATH := $(abspath $(ROOT)/toeplitz-ls):$(CURDIR)

.PHONY: all download unpack native check benchmark accuracy clean distclean
.DELETE_ON_ERROR:

all: native

download: $(AOV_ARCHIVE) $(FASTCHI2_ARCHIVE)

$(AOV_ARCHIVE):
	curl --fail --location --output $@ $(AOV_URL)

$(FASTCHI2_ARCHIVE):
	curl --fail --location --output $@ $(FASTCHI2_URL)

unpack: $(AOV_STAMP) $(FASTCHI2_STAMP)

$(AOV_STAMP): $(AOV_ARCHIVE)
	test -f $(AOV_DIR)/c/AovPer.c || tar -xzf $<
	test -f $(AOV_DIR)/c/AovPer.c
	touch $@

$(FASTCHI2_STAMP): $(FASTCHI2_ARCHIVE)
	test -f $(FASTCHI2_DIR)/fastchi2.c || tar -xzf $<
	test -f $(FASTCHI2_DIR)/fastchi2.c
	touch $@

native: $(AOV_LIB) $(FASTCHI2_LIB)

$(AOV_LIB): $(AOV_STAMP) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) -I$(AOV_DIR)/c \
		-o $@ $(AOV_DIR)/c/AovPer.c $(AOV_DIR)/c/AovSub.c -lm

$(FASTCHI2_LIB): $(FASTCHI2_STAMP) | $(BUILD_DIR)
	@command -v gsl-config >/dev/null || { \
		echo "FastChi2-1.03 requires GSL: install the GSL development package so gsl-config is available." >&2; \
		exit 1; \
	}
	$(CC) $(CFLAGS) $(LDFLAGS_SHARED) $$(gsl-config --cflags) \
		-o $@ $(FASTCHI2_DIR)/fastchi2.c $$(gsl-config --libs)

$(BUILD_DIR):
	mkdir -p $@

check: native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

benchmark: native
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) run_benchmark.py \
		--phot-dir $(abspath $(PHOT_DIR)) --index-path $(abspath $(INDEX_PATH)) \
		--output-dir $(abspath $(OUT_DIR)) \
		$(BENCH_ARGS)

accuracy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) compare.py \
		--index-path $(abspath $(INDEX_PATH)) --result-dir $(abspath $(OUT_DIR))

clean:
	rm -rf $(BUILD_DIR) $(OUT_DIR) __pycache__ astropy_impl/__pycache__ tests/__pycache__
	rm -f $(AOV_STAMP) $(FASTCHI2_STAMP)

distclean: clean
	rm -rf $(AOV_DIR) $(FASTCHI2_DIR)
	rm -f $(AOV_ARCHIVE) $(FASTCHI2_ARCHIVE)
