SM_VER ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | awk '{print $$1*10;}')
NVCC = nvcc
NVCCFLAGS = $(shell ./nvccoptions/get_nvccopts.sh) -Xcompiler -Wformat=2 -I./atlc/include -I./include -I./cxxopts/include -lcurand -lnccl -lssl -lcrypto --cudart=shared -O3 -Xcompiler -fopenmp -Xcompiler -rdynamic -std=c++17 -rdc=true -Wno-deprecated-gpu-targets -gencode=arch=compute_$(SM_VER),code=sm_$(SM_VER)
MPIRUN = mpirun
MPIRUN_FLAGS ?= -np $(shell nvidia-smi -L 2>/dev/null | wc -l)

.PHONY: target
target: qcs

qcs: src/qcs.cu include/qcs.hpp
	$(NVCC) $(NVCCFLAGS) $< -o $@

.PHONY: run
run: qcs
	$(MPIRUN) $(MPIRUN_FLAGS) ./$<

.PHONY: clean
clean:
	$(RM) qcs
