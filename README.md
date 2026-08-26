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

## Acknowledgments

This repository is based on results obtained from a project, JPNP20017, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
