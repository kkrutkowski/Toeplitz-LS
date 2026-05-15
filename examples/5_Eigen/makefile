CC ?= gcc
CXX ?= g++

ROOT := ../..
EIGEN_INC ?= /usr/include/eigen3
INCLUDES := -I$(ROOT)/include
EIGEN_INCLUDES := $(INCLUDES) -I$(EIGEN_INC)
CFLAGS ?= -std=gnu11 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -march=native -mtune=native -Wall -Wextra
LDLIBS ?= -lm

COMMON_CFLAGS := $(CFLAGS) $(INCLUDES)
COMMON_CXXFLAGS := $(CXXFLAGS) $(EIGEN_INCLUDES)

LINALG_OBJS := tlsf_linalg.o tls_linalg.o tlsdd_linalg.o
OBJS := $(LINALG_OBJS)
TARGET := eigen_zohar_bench

.PHONY: all clean test

all: $(TARGET)
	rm -f $(OBJS)

$(TARGET): eigen_zohar_bench.cpp $(OBJS)
	$(CXX) $(COMMON_CXXFLAGS) $^ -o $@ $(LDLIBS)

tlsf_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

tls_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE -c $< -o $@

tlsdd_linalg.o: $(ROOT)/src/linalg.c
	$(CC) $(COMMON_CFLAGS) -D DOUBLE_DOUBLE -c $< -o $@

test: all
	./$(TARGET) 12 256

clean:
	rm -f $(OBJS) $(TARGET)
