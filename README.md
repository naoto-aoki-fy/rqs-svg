# RQS-SVG Memory Sharing Branch (experimental)

## Building

The build requires the NVIDIA CUDA toolkit and the header-only
[`atlc`](https://github.com/naoto-aoki-fy/atlc) utility library.

Define any platform-specific compiler, linker, and GPU architecture flags in `config.mk`:

```make
CFLAGS_VENDOR = -I/foo/bar/include
LDFLAGS_VENDOR = -L/foo/bar/lib
GENCODE_FLAGS = -gencode=arch=compute_90,code=sm_90
```

These options can be obtained using the [`nvccoptions`](https://github.com/naoto-aoki-fy/nvccoptions) utility.

Then build the simulator with:

```sh
make
```

The resulting executable is written to `bin/qcs`.

## Running

By default, the simulator uses GPUs 0 through 7 and simulates 24 qubits. Use
`-g`/`--gpu_list` with a comma-separated GPU list and `-q`/`--num_qubits` with
a qubit count to override these defaults:

```sh
bin/qcs -g 0,1,2,3 -q 30
# Equivalent: bin/qcs --gpu_list 0,1,2,3 --num_qubits 30
```

The GPU list must contain 1, 2, 4, or 8 entries. Run `bin/qcs -h` to display
the command-line help.

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
