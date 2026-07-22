# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.h`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `qcs` executable.

## Building Simulator

The project requires NVIDIA's CUDA toolkit as well as NCCL and MPI.

Before building, set up the
[`atlc`](https://github.com/naoto-aoki-fy/atlc)
repository, which is required during the build process.

Define `CFLAGS_VENDOR`, `LDFLAGS_VENDOR`, and `GENCODE_FLAGS` in `config.mk`, which is automatically included by the Makefile:

```make
CFLAGS_VENDOR = -I/foo/bar
LDFLAGS_VENDOR = -L/foo/bar -lfoobar
GENCODE_FLAGS = -gencode=arch=compute_xx,code=sm_xx
```

These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility.

Then, `make` will build `qcs`.

```sh
make
```

### CUDA 10.x Note

For CUDA 10.x, it is recommended using
[nvcc-wrapper](https://github.com/naoto-aoki-fy/nvcc-wrapper)
and setting
`NVCC="python3 /path/to/nvcc_wrapper.py"`.


## Compiling Circuit

Use the helper script:

```sh
./compile_circuit.sh user_circuit.c -o user_circuit.so
```

The script resolves `DIR_QCS_H` automatically from `BASH_SOURCE` and executes:

```sh
gcc -fPIC -shared -I(DIR_QCS_H) -std=c11 user_circuit.c -o user_circuit.so
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
