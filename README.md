# RQS-SVG

The RQS-SVG is a GPU-accelerated testbed for quantum circuit simulation. The code provides a `simulator` API (see `qcs.hpp`) implementing common gate operations, state preparation, and measurement using CUDA, MPI, and NCCL. The Makefile builds either a standalone `qcs` executable or the shared library `libqcs.so`.

## Building Simulator

The project requires NVIDIA's CUDA toolkit as well as NCCL, and MPI. Example build targets:

```
make qcs      # build standalone simulator
```

## Compiling Circuit

```
g++ -fPIC -shared -I(DIR_QCS_HPP) -std=c++17 user_circuit.cpp -o user_circuit.so
```

## Running

You can execute the built simulator via:

```
mpirun -np (NUM_GPUS) ./qcs [--num-samples NUM_SAMPLES|-s NUM_SAMPLES] user_circuit.so

# Save the final statevector to a binary file
mpirun -np (NUM_GPUS) ./qcs --output-statevector data.bin user_circuit.so
```

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
