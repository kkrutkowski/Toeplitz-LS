CC ?= gcc

ROOT := ../..
INCLUDES := -I$(ROOT)/include
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
LDFLAGS ?= -static
LDLIBS ?= -lm
SAVE_FLAGS ?=

COMMON_CFLAGS := $(CFLAGS) $(INCLUDES)

LINALG_OBJS := tlsf_linalg.o tls_linalg.o tlsdd_linalg.o
OBJS := $(LINALG_OBJS)
TARGET := linalg_test

.PHONY: all clean test save

all: $(TARGET)
	rm -f $(OBJS)

$(TARGET): linalg_test.c $(OBJS)
	$(CC) $(COMMON_CFLAGS) $(SAVE_FLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

tlsf_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

test: all
	./$(TARGET)
	rm -f ./$(TARGET)

save:
	$(MAKE) SAVE_FLAGS=-DSAVE all
	./$(TARGET)
	rm -f ./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
