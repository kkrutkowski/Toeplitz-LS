CC ?= gcc

ROOT := ../..
INCLUDES := -I$(ROOT)/include
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
LDLIBS ?= -lm
SAVE_FLAGS ?= -DSAVE

COMMON_CFLAGS := $(CFLAGS) $(INCLUDES)

NANOFFT_OBJS := nanofftf.o nanofft.o nanofftdd.o
NUFFT_OBJS := tlsf_nufft1.o tls_nufft1.o tlsdd_nufft1.o
OBJS := $(NANOFFT_OBJS) $(NUFFT_OBJS)
TARGETS := tlsf_nufft_scaling tls_nufft_scaling tlsdd_nufft_scaling

.PHONY: all bench clean save

all: $(TARGETS)
	rm -f $(OBJS)

tlsf_nufft_scaling: $(ROOT)/src/scaling.c nanofftf.o tlsf_nufft1.o
	$(CC) $(COMMON_CFLAGS) $(SAVE_FLAGS) $^ -o $@ $(LDLIBS)

tls_nufft_scaling: $(ROOT)/src/scaling.c nanofft.o tls_nufft1.o
	$(CC) $(COMMON_CFLAGS) -D DOUBLE $(SAVE_FLAGS) $^ -o $@ $(LDLIBS)

tlsdd_nufft_scaling: $(ROOT)/src/scaling.c nanofftdd.o tlsdd_nufft1.o
	$(CC) $(COMMON_CFLAGS) -D DOUBLE_DOUBLE $(SAVE_FLAGS) $^ -o $@ $(LDLIBS)

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
	./tlsf_nufft_scaling
	./tls_nufft_scaling
	./tlsdd_nufft_scaling

save: all

clean:
	rm -f $(OBJS) $(TARGETS)
