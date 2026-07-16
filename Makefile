-include config.mk
ifndef CFLAGS
	$(error CFLAGS not defined)
endif
ifndef LDFLAGS
	$(error LDFLAGS not defined)
endif
ifndef NVCC_LDFLAGS
	$(error NVCC_LDFLAGS not defined)
endif
ifndef NVCC_GENCODE_FLAGS
	$(error NVCC_GENCODE_FLAGS not defined)
endif

LDLIBS ?= -lcurand -lnccl -lssl -lcrypto -ldl
NVCC ?= nvcc
INCLUDE ?= -I./atlc/include -I./include
# CC_CFLAGS = $(CFLAGS) -Wformat=2 $(INCLUDE) -O3 -rdynamic -std=c++17
NVCC_CFLAGS = $(CFLAGS) -Xcompiler -Wformat=2 $(INCLUDE) -Xcompiler -rdynamic -std=c++17 -Wno-deprecated-gpu-targets $(NVCC_GENCODE_FLAGS)
# LDFLAGS += -Bdynamic $(LDLIBS) -lcudart
NVCC_LDFLAGS += $(LDLIBS) --cudart=shared
MPIRUN ?= mpirun
MPIRUN_FLAGS ?= -np $(shell nvidia-smi -L 2>/dev/null | wc -l)

.PHONY: target
target: qcs

qcs: src/qcs.cu src/qcs_args.c src/qcs_args.h include/qcs.hpp
	$(NVCC) $(NVCC_CFLAGS) $(NVCC_LDFLAGS) src/qcs.cu src/qcs_args.c -o $@

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs
