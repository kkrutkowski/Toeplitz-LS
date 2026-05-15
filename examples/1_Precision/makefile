CC ?= gcc

ROOT := ../..
INCLUDES := -I$(ROOT)/include
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
LDLIBS ?= -lm -lquadmath

COMMON_CFLAGS := $(CFLAGS) $(INCLUDES)

NANOFFT_OBJS := nanofftf.o nanofft.o nanofftdd.o
NUFFT_OBJS := tlsf_nufft1.o tls_nufft1.o tlsdd_nufft1.o
OBJS := $(NANOFFT_OBJS) $(NUFFT_OBJS)
TARGET := nufft_bench

.PHONY: all bench clean

all: $(TARGET)
	rm -f $(OBJS)

$(TARGET): nufft1_bench.c $(OBJS)
	$(CC) $(COMMON_CFLAGS) $^ -o $@ $(LDLIBS)

nanofftf.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

nanofft.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

nanofftdd.o: $(ROOT)/src/nanofft.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

tlsf_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_nufft1.o: $(ROOT)/src/nufft1.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

bench: all
	timeout 90s ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
