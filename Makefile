-include config.mk
ifndef NVCC_GENCODE_FLAGS
  $(error NVCC_GENCODE_FLAGS not defined)
endif
ifndef NVCCOPTIONS
  $(error NVCCOPTIONS not defined)
endif

NVCC = nvcc
NVCCFLAGS = $(NVCCOPTIONS) -Xcompiler -Wformat=2 -I./atlc/include -I./include -lcurand -lnccl -lssl -lcrypto -ldl --cudart=shared -O3 -Xcompiler -fopenmp -Xcompiler -rdynamic -std=c++17 -rdc=true -Wno-deprecated-gpu-targets $(NVCC_GENCODE_FLAGS)
MPIRUN = mpirun
MPIRUN_FLAGS ?= -np $(shell nvidia-smi -L 2>/dev/null | wc -l)

.PHONY: target
target: qcs

qcs: src/qcs.cu src/qcs_args.c src/qcs_args.h include/qcs.hpp
	$(NVCC) $(NVCCFLAGS) src/qcs.cu src/qcs_args.c -o $@

.PHONY: gengetopt
gengetopt: src/qcs_args.ggo
	gengetopt --input=$< --unamed-opts --file-name=qcs_args --output-dir=src

.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs
