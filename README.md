# GPU simple hadamard

## Building

The build requires the NVIDIA CUDA toolkit and OpenSSL. Define any
platform-specific compiler, linker, and GPU architecture flags in `config.mk`:

```make
CFLAGS_VENDOR = -I/foo/bar/include
LDFLAGS_VENDOR = -L/foo/bar/lib
GENCODE_FLAGS = -gencode=arch=compute_90,code=sm_90
```

Then build the simulator with:

```sh
make
```

The resulting executable is written to `bin/qcs`. The output path can be
overridden when necessary, for example with `make QCS_BIN=main.exe`.
