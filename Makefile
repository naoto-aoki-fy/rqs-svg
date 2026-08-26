-include config.mk

NVCC ?= nvcc --forward-unknown-to-host-compiler
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 -O3 -std=c++17 -Xcompiler -fopenmp -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
QCS_BIN ?= bin/qcs
GENGETOPT ?= gengetopt

.PHONY: target
target: $(QCS_BIN)

cmdline.c cmdline.h &: args.ggo
	$(GENGETOPT) --input=$< --file-name=cmdline --output-dir=.

$(QCS_BIN): main.cu cmdline.c cmdline.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) main.cu cmdline.c $(NVCC_LDFLAGS) -o $@

.PHONY: clean
clean:
	$(RM) $(QCS_BIN)
