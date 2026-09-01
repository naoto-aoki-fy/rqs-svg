# RQS-SVG OpenMP CPU Version

This branch runs the quantum-circuit benchmark entirely on the CPU. The CUDA
kernel launch has been replaced by an OpenMP parallel loop, CUDA complex values
by `std::complex`, device allocations by standard C++ storage, and CUDA event
timing by `std::chrono`. The split-state gate indexing remains close to the
CUDA-UVA implementation.

## Building

The build requires a C++17 compiler with OpenMP support and GNU Gengetopt.
Optional platform-specific flags can be placed in `config.mk`:

```make
CFLAGS_VENDOR = -I/foo/bar/include
LDFLAGS_VENDOR = -L/foo/bar/lib
```

Build the simulator with:

```sh
make
```

The resulting executable is written to `bin/qcs`. No CUDA toolkit, GPU, or
`atlc` installation is required.

## Running

By default, the simulator uses the OpenMP runtime's default thread count,
simulates 24 qubits, and runs 64 benchmark samples. Use `-t`/`--threads` to set
the CPU thread count, `-q`/`--num-qubits` to set the qubit count, and
`-s`/`--num-samples` to set the sample count:

```sh
bin/qcs -t 8 -q 24 -s 10
# Equivalent: bin/qcs --threads 8 --num-qubits 24 --num-samples 10
```

Passing `--threads 0` leaves thread selection to the OpenMP runtime. Run
`bin/qcs -h` to display all command-line options.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017,
commissioned by the New Energy and Industrial Technology Development
Organization (NEDO).
