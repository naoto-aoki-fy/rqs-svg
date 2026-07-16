# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.hpp`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `qcs` executable.

## Building Simulator

The project requires NVIDIA's CUDA toolkit as well as NCCL and MPI.

Specify the target GPU architecture by passing `SM_VER` to `make`. For example:

```sh
make qcs SM_VER=100
```

Alternatively, you can define `SM_VER` in `config.mk`, which is automatically included by the Makefile:

```make
SM_VER = 100
```

You can obtain the appropriate `SM_VER` value from your GPU's compute capability with:

```sh
nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
  | awk '{ print $1 * 10 }'
```

For example, a compute capability of `10.0` corresponds to `SM_VER=100`.

The options required when invoking `nvcc` through the MPI C++ compiler wrapper must be passed to `make` using `NVCCOPTIONS`:

```sh
make qcs SM_VER=100 NVCCOPTIONS='(NVCC options)'
```

`NVCCOPTIONS` may also be defined in `config.mk`:

```make
SM_VER = 100
NVCCOPTIONS = (NVCC options)
```

When using the NVIDIA HPC SDK, `NVCCOPTIONS` should contain the options that `mpicxx` passes to `nvcc`. These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility:

```sh
git clone https://github.com/naoto-aoki-fy/nvccoptions.git
cd nvccoptions
./get_nvccopts.sh
```

The utility prints the options passed to `nvcc` when NVIDIA HPC SDK's `mpicxx` compiles CUDA code. Pass the resulting option string to `make` or assign it to `NVCCOPTIONS` in `config.mk`.

Because `NVCCOPTIONS` is processed by `make`, each dollar sign (`$`) in the option string must be escaped as `$$`. For example, if the generated options contain `$ORIGIN`, specify it as `$$ORIGIN`:

```make
NVCCOPTIONS = -Xlinker -rpath -Xlinker '$$ORIGIN/../lib'
```

The same escaping is required when passing the value on the `make` command line:

```sh
make qcs SM_VER=100 \
  NVCCOPTIONS="-Xlinker -rpath -Xlinker '\$$ORIGIN/../lib'"
```

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
