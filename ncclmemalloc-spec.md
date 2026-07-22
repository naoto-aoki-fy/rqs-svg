## Replacement Targets

In this repository, there are effectively two places where replacing `cudaMallocAsync` with `ncclMemAlloc` should be considered.

| Buffer                   |                               Replacement Decision | Reason                                                                    |
| ------------------------ | -------------------------------------------------: | ------------------------------------------------------------------------- |
| `state_data_device`      | **Candidate for replacement (benchmark required)** | Serves as the NCCL send buffer during global qubit exchange               |
| `swap_buffer`            |                    **Recommended for replacement** | Used as the NCCL receive buffer and exchange buffer                       |
| `cub_temp_buffer_device` |                                 **Do not replace** | Temporary workspace for CUB, not an NCCL communication buffer             |
| `measure_norm_device`    |                                 **Do not replace** | Small temporary buffer for measurement calculations; never passed to NCCL |

Currently, all four buffers are allocated in `allocate_memory()`.

---

# 1. `swap_buffer`

## Current

```cpp
ATLC_CHECK_CUDA(
    cudaMallocAsync,
    &swap_buffer,
    swap_buffer_total_length * sizeof(qcs::complex_t),
    stream
);
```

## After the change

```cpp
ATLC_CHECK_NCCL(
    ncclMemAlloc,
    reinterpret_cast<void**>(&swap_buffer),
    swap_buffer_total_length * sizeof(qcs::complex_t)
);
```

## Deallocation

Replace the current

```cpp
ATLC_CHECK_CUDA(
    cudaFreeAsync,
    swap_buffer,
    stream
);
```

with

```cpp
ATLC_CHECK_NCCL(
    ncclMemFree,
    swap_buffer
);
```

The current deallocation is performed in `free_memory()`.

## Recommendation

`swap_buffer` is the clearest replacement target.

This buffer is used almost exclusively for data exchange when localizing global qubits, making it an ideal match for NCCL's VMM-based allocation model. The NCCL documentation also recommends using `ncclMemAlloc` (or another VMM-compatible allocator) for communication buffers that will be registered. ([NVIDIA Docs][1])

---

# 2. `state_data_device`

## Current

```cpp
if (use_unified_memory) {
    ATLC_CHECK_CUDA(
        cudaMallocManaged,
        &state_data_device,
        num_states_local * sizeof(*state_data_device)
    );
} else {
    ATLC_CHECK_CUDA(
        cudaMallocAsync,
        &state_data_device,
        num_states_local * sizeof(*state_data_device),
        stream
    );
}
```

## After the change

If you replace the allocation only when `use_unified_memory == false`, the code becomes:

```cpp
if (use_unified_memory) {
    ATLC_CHECK_CUDA(
        cudaMallocManaged,
        &state_data_device,
        num_states_local * sizeof(*state_data_device)
    );
} else {
    ATLC_CHECK_NCCL(
        ncclMemAlloc,
        reinterpret_cast<void**>(&state_data_device),
        num_states_local * sizeof(*state_data_device)
    );
}
```

The deallocation should also branch according to the allocation method:

```cpp
if (state_data_device) {
    if (use_unified_memory) {
        ATLC_CHECK_CUDA(
            cudaFree,
            state_data_device
        );
    } else {
        ATLC_CHECK_NCCL(
            ncclMemFree,
            state_data_device
        );
    }

    state_data_device = nullptr;
}
```

## Recommendation

If the goal is to reliably use NCCL user-buffer registration via `ncclCommRegister`, `state_data_device` should also be replaced.

According to NCCL, establishing the registered communication path generally requires **both the sender and receiver buffers** to use registration-compatible allocations. If `state_data_device` is the sender and `swap_buffer` is the receiver, replacing only `swap_buffer` with `ncclMemAlloc` is insufficient. ([NVIDIA Docs][1])

However, `state_data_device` requires additional consideration. Unlike `swap_buffer`, it is not dedicated to NCCL communication—it is the primary state vector that CUDA kernels continuously read from and write to. The code accesses `state_data_device` directly from CUDA kernels.

The NCCL documentation describes `ncclMemAlloc` as an allocator intended for NCCL-related use cases and does not recommend using it indiscriminately for all application memory. Therefore, before replacing `state_data_device`, its impact on CUDA kernel performance, memory usage, and allocation time should be benchmarked. ([NVIDIA Docs][1])

---

# 3. `cub_temp_buffer_device`

## Current

```cpp
ATLC_CHECK_CUDA(
    cudaMallocAsync,
    &cub_temp_buffer_device,
    1,
    stream
);
cub_temp_buffer_device_size = 1;
```

## Recommendation

**Do not replace this with `ncclMemAlloc`.**

This buffer is a temporary workspace for CUB operations such as reductions and scans and is never passed to NCCL as a communication buffer.

Therefore, it should remain:

```cpp
cudaMallocAsync(...)
cudaFreeAsync(...)
```

This buffer may also be reallocated once its required size becomes known, making the stream-ordered behavior of `cudaMallocAsync` better suited to this use case.

---

# 4. `measure_norm_device`

## Current

```cpp
ATLC_CHECK_CUDA(
    cudaMallocAsync,
    &measure_norm_device,
    sizeof(qcs::complex_t),
    stream
);
```

## Recommendation

**Do not replace this with `ncclMemAlloc`.**

This is a small temporary GPU buffer used to store intermediate results for measurement probabilities or norm calculations. Since it is never used as an NCCL send or receive buffer, there is no benefit in switching to a VMM-based NCCL allocation.

Moreover, allocating only a few bytes (or even a few dozen bytes) with `ncclMemAlloc` may incur unnecessary memory consumption and allocation overhead due to VMM allocation granularity.

---

# Recommended Final Configuration

```cpp
// Directly used for NCCL communication
state_data_device  -> ncclMemAlloc
swap_buffer        -> ncclMemAlloc

// Not directly used for NCCL communication
cub_temp_buffer_device -> cudaMallocAsync
measure_norm_device     -> cudaMallocAsync
```

For a phased rollout, the following order is recommended.

## Phase 1

Replace only `swap_buffer` with `ncclMemAlloc`.

```cpp
state_data_device       = cudaMallocAsync
swap_buffer             = ncclMemAlloc
cub_temp_buffer_device  = cudaMallocAsync
measure_norm_device     = cudaMallocAsync
```

This minimizes compatibility risks and any impact on the quantum gate kernels while validating the allocator change itself. However, this configuration alone may not enable the benefits of registered communication, since both the sender and receiver buffers typically need to be registration-compatible.

## Phase 2

Replace `state_data_device` as well.

```cpp
state_data_device       = ncclMemAlloc
swap_buffer             = ncclMemAlloc
```

Then register both buffers.

```cpp
void* state_reg_handle = nullptr;
void* swap_reg_handle = nullptr;

ATLC_CHECK_NCCL(
    ncclCommRegister,
    nccl_comm,
    state_data_device,
    num_states_local * sizeof(*state_data_device),
    &state_reg_handle
);

ATLC_CHECK_NCCL(
    ncclCommRegister,
    nccl_comm,
    swap_buffer,
    swap_buffer_total_length * sizeof(*swap_buffer),
    &swap_reg_handle
);
```

For NCCL general buffer registration, the documentation recommends buffers allocated via the CUDA VMM API, a VMM-based allocator, or `ncclMemAlloc`. Registration of buffers allocated with traditional CUDA allocators is subject to safety restrictions and may be disabled by default in current NCCL configurations. ([NVIDIA Docs][1])

---

# Implementation Notes

## `ncclMemAlloc` is not an asynchronous allocator

Unlike `cudaMallocAsync`, which is ordered on a specified stream, `ncclMemAlloc` does not take a stream parameter.

Therefore,

```cpp
cudaMallocAsync(&ptr, size, stream);
```

becomes

```cpp
ncclMemAlloc(&ptr, size);
```

which changes the allocation execution model.

In this repository, however, `allocate_memory()` performs long-lived allocations once before circuit execution, so this difference is generally not expected to be an issue.

---

## Ensure communication completes before freeing

Before calling `ncclMemFree`, all NCCL operations using the registered buffers must have completed.

```cpp
ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

ATLC_CHECK_NCCL(
    ncclCommDeregister,
    nccl_comm,
    state_reg_handle
);

ATLC_CHECK_NCCL(
    ncclCommDeregister,
    nccl_comm,
    swap_reg_handle
);

ATLC_CHECK_NCCL(ncclMemFree, state_data_device);
ATLC_CHECK_NCCL(ncclMemFree, swap_buffer);
```

The official examples likewise synchronize the communication stream, deregister the buffers, and only then call `ncclMemFree`. ([NVIDIA Docs][1])

---

## Unified memory mode

`state_data_device` currently supports a `cudaMallocManaged` allocation path.

If migrating to `ncclMemAlloc`, one of the following approaches should be chosen:

```cpp
use_unified_memory == true
    -> Keep cudaMallocManaged
    -> Do not expect the optimized ncclCommRegister path
```

or

```cpp
Use NCCL buffer registration
    -> Disable unified memory
    -> Use ncclMemAlloc
```

Memory allocated with `ncclMemAlloc` is **not** CUDA managed memory. Therefore, it is preferable to treat the existing `use_unified_memory` mode and the `ncclMemAlloc` mode as mutually exclusive allocation modes.

---

# Summary

```text
Replace:
  1. state_data_device
  2. swap_buffer

Do not replace:
  3. cub_temp_buffer_device
  4. measure_norm_device
```

From a practical implementation perspective, the recommended order is:

```text
swap_buffer
    ↓
state_data_device
```

However, to realize the benefits of `ncclCommRegister`, **both `state_data_device` and `swap_buffer` ultimately need to be migrated to `ncclMemAlloc`.**

[1]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html?utm_source=chatgpt.com "User Buffer Registration — NCCL 2.30.7 documentation"
