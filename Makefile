-include config.mk
LDLIBS ?= -lcurand -lnccl -lssl -lcrypto -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) -L./lib $(LDLIBS) --cudart=shared
QCS_BIN ?= bin/qcs
LIBQCS_SO ?= lib/libqcs.so

.PHONY: target
target: $(QCS_BIN)

.PHONY: sharedlibrary
sharedlibrary: $(LIBQCS_SO)

$(QCS_BIN): src/qcs_main.cpp src/qcs_args.c src/qcs_args.h include/qcs.h $(LIBQCS_SO)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) src/qcs_main.cpp src/qcs_args.c -lqcs $(NVCC_LDFLAGS) -o $@

$(LIBQCS_SO): src/qcs.cu include/qcs.h
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) -fPIC $(NVCC_LDFLAGS) -shared src/qcs.cu -o $@

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: clean
clean:
	$(RM) $(QCS_BIN) $(LIBQCS_SO)
