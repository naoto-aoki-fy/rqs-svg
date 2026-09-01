# RQS-SVG OpenMP benchmark

This branch provides a CPU-only, shared-memory quantum state-vector benchmark.
It applies a Hadamard gate to every qubit using a persistent OpenMP team and a
flat, first-touch-initialized state vector.

## Building

The build requires a C++17 compiler with OpenMP support and GNU Gengetopt when
regenerating the checked-in command-line parser. Platform-specific options can
be supplied in `config.mk`:

```make
CFLAGS_VENDOR = -march=native
LDFLAGS_VENDOR =
```

Build the simulator and run the correctness tests with:

```sh
make
make test
```

The executable is written to `bin/qcs`.

## Running

By default, the simulator uses the OpenMP runtime's thread count, simulates 24
qubits, and runs 64 benchmark samples. Each stdout line is the elapsed time in
seconds for one complete pass of Hadamard gates over all qubits.

```sh
bin/qcs -q 30 -s 100 -t 32
# Equivalent: bin/qcs --num-qubits 30 --num-samples 100 --threads 32
```

`--threads=0` leaves thread selection to the OpenMP runtime. This makes normal
affinity configuration available, for example:

```sh
OMP_NUM_THREADS=32 OMP_PROC_BIND=spread OMP_PLACES=cores bin/qcs -q 30 -s 64
```

Parallel first-touch initialization helps distribute the state across NUMA
nodes. Benchmark both `OMP_PROC_BIND=close` and `spread` on multi-socket hosts.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017,
commissioned by the New Energy and Industrial Technology Development
Organization (NEDO).
