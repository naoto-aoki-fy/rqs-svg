-include config.mk

LDLIBS ?= -lcurand
NVCC ?= nvcc --forward-unknown-to-host-compiler
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 -O3 -std=c++17 -Xcompiler -fopenmp -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
QCS_BIN ?= bin/qcs

.PHONY: target
target: $(QCS_BIN)

$(QCS_BIN): main.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) $< $(NVCC_LDFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(QCS_BIN)
