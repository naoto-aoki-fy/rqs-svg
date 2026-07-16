# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.hpp`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `qcs` executable.

## Building Simulator

The project requires NVIDIA's CUDA toolkit as well as NCCL and MPI.

Specify the target GPU architecture by passing `NVCC_GENCODE_FLAGS` to `make`. Also the options required when invoking `nvcc` through the MPI C++ compiler wrapper must be passed to `make` using `NVCCOPTIONS`. For example:

```sh
make qcs NVCC_GENCODE_FLAGS=-gencode=arch=compute_100,code=sm_100 NVCCOPTIONS='(NVCC options)'
```

Alternatively, you can define `NVCC_GENCODE_FLAGS` and `NVCCOPTIONS` in `config.mk`, which is automatically included by the Makefile:

```make
NVCC_GENCODE_FLAGS = -gencode=arch=compute_100,code=sm_100
NVCCOPTIONS = (NVCC options)
```

These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility.

## Compiling Circuit

Use the helper script:

```sh
./compile_circuit.sh user_circuit.cpp -o user_circuit.so
```

The script resolves `DIR_QCS_HPP` automatically from `BASH_SOURCE` and executes:

```sh
g++ -fPIC -shared -I(DIR_QCS_HPP) -std=c++17 user_circuit.cpp -o user_circuit.so
```

## Running

You can execute the built simulator via:

```sh
mpirun -np (NUM_GPUS) ./qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./qcs --output-statevector data.bin user_circuit.so
```

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
