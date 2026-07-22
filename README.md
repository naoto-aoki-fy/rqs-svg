# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.h`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds a standalone `qcs` executable and can also build `libqcs.so` for programs that link against RQS-SVG directly.

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

To build RQS-SVG as a shared library:

```sh
make sharedlibrary
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

The loadable circuit examples live under `examples/standalone/`.

## Using the Shared Library from C

The shared library exposes the same C API through `include/qcs.h`, plus
`qcs_simulator_create` and `qcs_simulator_destroy` helpers for C programs that
cannot allocate the opaque `qcs_simulator` type directly.

Build the shared library and then compile the C example:

```sh
make sharedlibrary
gcc -I./include -L. -Wl,-rpath,"$(realpath .)" examples/sharedlibrary/ghz_from_c.c -lqcs -o ghz_from_c
mpirun -np (NUM_GPUS) ./ghz_from_c
```

Code examples for linking against `libqcs.so` live under
`examples/sharedlibrary/`.


## Using the Shared Library from Python

The `python/qcs_ctypes.py` module loads `libqcs.so` with Python's standard
`ctypes` package and wraps simulator allocation, gate calls, measurement, and
destruction in a small `Simulator` class. Build the shared library first, then
run the Python GHZ example with MPI just like the C example:

```sh
make sharedlibrary
mpirun -np (NUM_GPUS) python3 examples/python/ghz_ctypes.py --num-qubits 3
```

If `libqcs.so` is not in the repository root, pass `--library /path/to/libqcs.so`
or set `QCS_LIBRARY_PATH=/path/to/libqcs.so`.

## Running

You can execute the built simulator via:

```sh
mpirun -np (NUM_GPUS) ./qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./qcs --output-statevector data.bin user_circuit.so
```

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
