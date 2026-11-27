# Plain C23 Vulkan 1.1+ / GLSL Compute Offload Shared Library (tls_vulkan.so)
# Zero C++ / libstdc++ dependencies; links only -lvulkan -lm -lpthread.

CC       ?= gcc
PYTHON   ?= python3
STRIP    ?= strip
BUILD    := build
GEN_DIR  := $(BUILD)/gen
OBJ_DIR  := $(BUILD)/obj
TARGET   := $(BUILD)/tls_vulkan.so

CFLAGS   := -O3 -std=c23 -D_GNU_SOURCE -Wall -Wextra -fno-math-errno -fno-trapping-math -ffp-contract=fast -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections -I. -I$(GEN_DIR)
LDFLAGS  := -shared -Wl,--version-script=tls_vulkan.map -Wl,--gc-sections -Wl,-O1 -Wl,--as-needed -ldl -lm -lpthread

CODEGEN_SCRIPTS := $(wildcard codegen/*.py)
PSWF_HDR        := nufft/pswf_coeffs.h
SPV_HDR         := $(GEN_DIR)/shaders_spv.h
SPV_SRC         := $(GEN_DIR)/shaders_spv.c

C_SRCS := \
	common/vk_loader.c \
	common/vk_device.c \
	common/vk_context.c \
	bitrev/vk_bitrev.c \
	fft/vk_dif.c \
	nufft/vk_nufft1.c \
	periodogram/vk_subgrid.c \
	linalg/vk_levinson.c \
	periodogram/vk_engine.c \
	impl.c

C_HDRS := $(wildcard common/*.h bitrev/*.h fft/*.h nufft/*.h periodogram/*.h linalg/*.h)
C_OBJS  := $(patsubst %.c,$(OBJ_DIR)/%.o,$(C_SRCS))
SPV_OBJ := $(OBJ_DIR)/shaders_spv.o

.PHONY: all clean shaders

all: $(TARGET)

shaders: $(PSWF_HDR) $(SPV_HDR) $(SPV_SRC)

$(PSWF_HDR): codegen/pswf_tables.py ../sycl/nufft/pswf_coeffs.hpp
	$(PYTHON) codegen/pswf_tables.py

$(SPV_SRC): $(CODEGEN_SCRIPTS)
	$(PYTHON) codegen/compile_shaders.py --out-dir $(GEN_DIR)

$(SPV_HDR): $(SPV_SRC)
	@test -f $@ || $(PYTHON) codegen/compile_shaders.py --out-dir $(GEN_DIR)

# Compile large static SPIR-V byte arrays at -Os (fast build, compact lookup helpers)
$(SPV_OBJ): $(SPV_SRC) $(SPV_HDR)
	@mkdir -p $(dir $@)
	$(CC) -Os -std=c23 -fPIC -fvisibility=hidden -ffunction-sections -fdata-sections -fno-asynchronous-unwind-tables -I$(GEN_DIR) -c $< -o $@

$(OBJ_DIR)/%.o: %.c $(C_HDRS) $(SPV_HDR) $(PSWF_HDR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(C_OBJS) $(SPV_OBJ) tls_vulkan.map
	@mkdir -p $(dir $@)
	$(CC) $(C_OBJS) $(SPV_OBJ) $(LDFLAGS) -o $@
	$(STRIP) -s $@

clean:
	rm -rf $(BUILD)
