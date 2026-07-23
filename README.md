# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.h`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `bin/qcs` executable and can also build `lib/libqcs.so` for programs that link against RQS-SVG directly.

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

Then, `make` will build `bin/qcs`.

```sh
make
```

To build RQS-SVG as a shared library at `lib/libqcs.so`:

```sh
make sharedlibrary
```

If you will run programs that load or link against `lib/libqcs.so`, source the provided shell setup file from the repository root or from any other directory:

```sh
source ./env.bash
```

The setup file appends this repository's absolute `lib` directory path to `LD_LIBRARY_PATH`.

## Compiling Circuit

Use the helper script:

```sh
./compile_circuit.sh user_circuit.c -o user_circuit.so
```

Alternatively:

```sh
source ./env.bash
gcc -fPIC -shared user_circuit.c -o user_circuit.so
```

The loadable circuit examples live under `examples/standalone/`.

## Using the Shared Library from C

The shared library exposes the same C API through `include/qcs.h`, plus
`qcs_simulator_create` and `qcs_simulator_destroy` helpers for C programs that
cannot allocate the opaque `qcs_simulator` type directly.

Build the shared library and then compile the C example:

```sh
make sharedlibrary
source ./env.bash
gcc examples/sharedlibrary/ghz_from_c.c -lqcs -o ghz_from_c
mpirun -np (NUM_GPUS) ./ghz_from_c
```

Code examples for linking against `libqcs.so` live under
`examples/sharedlibrary/`.

Python bindings are maintained separately at
[naoto-aoki-fy/rqs-svg-py](https://github.com/naoto-aoki-fy/rqs-svg-py).

## Running

You can execute the built simulator via:

```sh
source ./env.bash

mpirun -np (NUM_GPUS) ./bin/qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./bin/qcs --output-statevector data.bin user_circuit.so
```

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
