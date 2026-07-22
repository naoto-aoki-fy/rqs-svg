-include config.mk
LDLIBS ?= -lcurand -lnccl -lssl -lcrypto -ldl
NVCC ?= nvcc --forward-unknown-to-host-compiler
INCLUDE ?= -I./atlc/include -I./include
NVCC_CFLAGS = $(CFLAGS_VENDOR) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(GENCODE_FLAGS)
NVCC_LDFLAGS = $(LDFLAGS_VENDOR) $(LDLIBS) --cudart=shared
NVCC_SO_CFLAGS = $(NVCC_CFLAGS) -DQCS_SHARED_LIBRARY -Xcompiler -fPIC
NVCC_SO_LDFLAGS = $(NVCC_LDFLAGS) -shared
MPIRUN ?= mpirun
MPIRUN_FLAGS ?= -np $(shell nvidia-smi -L 2>/dev/null | wc -l)

.PHONY: target
target: qcs

.PHONY: sharedlibrary
sharedlibrary: libqcs.so

qcs: src/qcs.cu src/qcs_args.c src/qcs_args.h include/qcs.h
	$(NVCC) $(NVCC_CFLAGS) $(NVCC_LDFLAGS) src/qcs.cu src/qcs_args.c -o $@

libqcs.so: src/qcs.cu include/qcs.h
	$(NVCC) $(NVCC_SO_CFLAGS) $(NVCC_SO_LDFLAGS) src/qcs.cu -o $@

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs libqcs.so
