## Recommended Approach

The recommended initial approach is to **perform one NCCL point-to-point communication with each communication peer before starting measurement, without modifying the quantum state**.

The current global-qubit implementation performs the following communication:

* Source: `state_data_device`
* Destination: `swap_buffer`
* Communication size: up to `swap_buffer_total_length`
* Communicator: `nccl_comm`
* Stream: `stream`
* `ncclSend` and `ncclRecv` are issued within the same NCCL group

In the current implementation, global/local qubit exchange is completed by copying `swap_buffer` back into the quantum state after the receive operation.

For the warm-up, **omit this final copy and simply discard the received data**.

```text
state_data_device ──send──> peer rank
swap_buffer       <─recv── peer rank

After receiving:
Discard the contents of swap_buffer.
Do not modify state_data_device.
```

This initializes the actual communication path, including:

* NCCL internal communication buffers
* Required proxy processing
* Transport-specific memory registration or mapping
* Send/receive channels
* Communication-size-dependent NCCL resources

---

# Example Implementation

Add the following method to `simulator_core`.

```cpp
void warmup_nccl_communication()
{
    if (num_procs <= 1) {
        return;
    }

    if (state_data_device == nullptr || swap_buffer == nullptr) {
        throw std::runtime_error(
            "warmup_nccl_communication called before allocate_memory");
    }

    /*
     * Production code uses:
     *
     *   count = swap_buffer_length * 2
     *   datatype = ncclDouble
     *
     * because complex_t consists of two doubles.
     */
    uint64_t const warmup_length =
        std::min(swap_buffer_total_length, num_states_local);

    size_t const warmup_count = warmup_length * 2;

    /*
     * Make sure cudaMallocAsync and initialization operations previously
     * submitted to this stream have completed.
     */
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

    /*
     * Global-qubit communication normally changes one or more bits of the
     * rank number. Communicating once along every rank dimension warms the
     * basic peer set used by one-global-qubit swaps.
     */
    for (int global_bit = 0; global_bit < log_num_procs; ++global_bit) {
        int const peer = proc_num ^ (1 << global_bit);

        ATLC_CHECK_NCCL(ncclGroupStart);

        ATLC_CHECK_NCCL(
            ncclSend,
            state_data_device,
            warmup_count,
            ncclDouble,
            peer,
            nccl_comm,
            stream);

        ATLC_CHECK_NCCL(
            ncclRecv,
            swap_buffer,
            warmup_count,
            ncclDouble,
            peer,
            nccl_comm,
            stream);

        ATLC_CHECK_NCCL(ncclGroupEnd);

        /*
         * Complete the communication before moving to the next peer.
         * This is intentionally outside the measured circuit region.
         */
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    }

    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
}
```

For NCCL point-to-point communication, every `ncclSend` must have a matching `ncclRecv` on the peer rank with the same `count` and `datatype`. It is also appropriate to group the corresponding send/receive operations within `ncclGroupStart`/`ncclGroupEnd`.

---

# Where to Insert It

The current execution flow allocates memory, creates CUDA events, and then starts timing inside the sample loop.

Therefore, the warm-up should be inserted here:

```cpp
qcs_simulator_allocate_memory(&sim);

/*
 * Add here.
 * This must be completed before event_1 is recorded.
 */
qcs_simulator_warmup_nccl_communication(&sim);

int const event_1 = qcs_simulator_event_create(&sim);
int const event_2 = qcs_simulator_event_create(&sim);

for (int sample_num = 0; sample_num < num_samples; sample_num++) {
    qcs_simulator_event_record(&sim, event_1);

    circuit_run(&sim);

    // ...
}
```

Since `swap_buffer` is allocated inside `allocate_memory()`, the warm-up must always run afterward. Currently, both `state_data_device` and `swap_buffer` are allocated using `cudaMallocAsync`.

---

# Why This Does Not Corrupt the Quantum State

The normal communication sequence is:

```cpp
ncclSend(state_data_device + offset, ...);
ncclRecv(swap_buffer, ...);

cudaMemcpyAsync(
    state_data_device + offset,
    swap_buffer,
    ...);
```

The quantum state is modified **not by the NCCL communication itself**, but by the final copy from `swap_buffer` back into `state_data_device`.

During warm-up, this copy is omitted. As a result:

* `state_data_device`: read-only
* `swap_buffer`: overwritten
* Qubit mapping: unchanged
* Measurement state: unchanged
* Classical bits: unchanged

Therefore, there is generally no need to reinitialize the quantum state before executing the circuit.

---

# Peer Selection

## Minimal Configuration: One Peer per Global Rank Bit

Use the following expression:

```cpp
peer = proc_num ^ (1 << global_bit);
```

For an 8-rank system:

| Rank | Global bit 0 | Global bit 1 | Global bit 2 |
| ---: | -----------: | -----------: | -----------: |
|    0 |            1 |            2 |            4 |
|    1 |            0 |            3 |            5 |
|    2 |            3 |            0 |            6 |
|    3 |            2 |            1 |            7 |

This covers the basic peer set used when swapping a single global qubit with a local qubit.

The advantage is that the number of communications scales as

```text
log2(num_procs)
```

rather than the total number of ranks.

```text
2 ranks  → 1 communication
4 ranks  → 2 communications
8 ranks  → 3 communications
16 ranks → 4 communications
```

---

## Supporting Multi-Global-Qubit Gates

In the current `ensure_local_qubits()` implementation, when multiple global qubits are localized simultaneously, peers may be generated by combining multiple rank bits. The peer rank is computed from the rank bits corresponding to the target global qubits together with the local-region index.

For example, when two global qubits are involved, peers such as

```text
rank ^ 01
rank ^ 10
rank ^ 11
```

may be required.

Accordingly, there are two possible warm-up strategies.

### Lightweight Version

Execute only

```cpp
peer = rank ^ (1 << bit)
```

Advantages:

* `log2(num_procs)` communications
* Covers single-global-qubit operations
* Simple implementation
* Recommended as the initial implementation

### Complete Version

Warm up every possible peer:

```cpp
for (int mask = 1; mask < num_procs; ++mask) {
    int peer = proc_num ^ mask;
    // sendrecv
}
```

Advantages:

* `num_procs - 1` communications
* Covers every peer that may arise from multi-global-qubit operations

Disadvantage:

* Warm-up cost increases with the number of ranks

The lightweight version is recommended initially. If first-use latency remains only for circuits containing multiple global qubits, it can later be extended to either the complete version or a circuit-dependent peer list.

---

# Communication Size

## Use a Production-Sized Transfer, Not a Single Element

Avoid warm-up calls such as

```cpp
ncclSend(..., 1, ncclDouble, ...);
ncclRecv(..., 1, ncclDouble, ...);
```

NCCL may select different:

* protocols
* channel counts
* chunk sizes
* transport buffers
* proxy behavior
* registration ranges

depending on the communication size.

Therefore, the warm-up should use the same upper-bound transfer size as production:

```cpp
warmup_length = swap_buffer_total_length;
warmup_count  = warmup_length * 2;
```

The production implementation also uses the smaller of `swap_buffer_total_length` and the target local-region length as the communication size.

If `swap_buffer_total_length` is extremely large, however, the warm-up traffic itself may become excessive. In that case, an upper limit can be applied:

```cpp
constexpr uint64_t warmup_max_bytes =
    UINT64_C(64) * 1024 * 1024;

uint64_t warmup_length =
    std::min(swap_buffer_total_length, num_states_local);

warmup_length = std::min(
    warmup_length,
    warmup_max_bytes / sizeof(qcs::complex_t));
```

A practical approach is to first verify that using the full production-sized transfer eliminates the first-run overhead, then gradually reduce the warm-up size to determine the minimum effective value.

---

# Should `ncclCommRegister()` Be Used?

## Prefer Not to Use It Initially

The objective here is to execute, before measurement,

> the one-time processing associated with NCCL internal buffers, proxy setup, and memory registration.

To achieve this, performing an actual `ncclSend`/`ncclRecv` once is the most direct approach.

`ncclCommRegister()` is **not merely a warm-up API**. Rather, it enables NCCL to use user buffers directly. Changing the registration mechanism may alter the communication path used during normal execution.

Furthermore, the current buffers are allocated with `cudaMallocAsync`. According to the current NCCL documentation, general buffer registration is primarily intended for VMM-based allocators or `ncclMemAlloc()`. Registering buffers allocated with the legacy CUDA allocator requires additional care and may even be disabled by default in some configurations.

Accordingly, a staged approach is recommended.

### Stage 1

```text
Perform a non-destructive warm-up using actual ncclSend/ncclRecv operations.
```

First, verify whether this eliminates the first-run latency.

### Stage 2

If the latency persists and is clearly attributable to user-buffer registration, then consider:

* allocating `state_data_device` and `swap_buffer` with `ncclMemAlloc()`
* explicitly registering them with `ncclCommRegister()`
* deregistering them with `ncclCommDeregister()` at shutdown
* releasing them with `ncclMemFree()`

Since this requires a substantially larger change to the memory management model, it should be treated as a separate enhancement rather than part of the initial warm-up implementation.

---

# Recommended Initial Implementation

For the smallest code change, the following implementation is recommended.

```cpp
void warmup_nccl_communication()
{
    if (num_procs <= 1) {
        return;
    }

    uint64_t const length =
        std::min(swap_buffer_total_length, num_states_local);

    size_t const count = length * 2;

    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

    for (int bit = 0; bit < log_num_procs; ++bit) {
        int const peer = proc_num ^ (1 << bit);

        ATLC_CHECK_NCCL(ncclGroupStart);

        ATLC_CHECK_NCCL(
            ncclSend,
            state_data_device,
            count,
            ncclDouble,
            peer,
            nccl_comm,
            stream);

        ATLC_CHECK_NCCL(
            ncclRecv,
            swap_buffer,
            count,
            ncclDouble,
            peer,
            nccl_comm,
            stream);

        ATLC_CHECK_NCCL(ncclGroupEnd);
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    }

    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
}
```

## Validation Criteria

After adding the warm-up, verify the following:

1. The execution time difference between the first and second circuit runs disappears.
2. The initial quantum state is identical before and after introducing the warm-up.
3. The latency disappears for circuits containing a single global qubit.
4. The latency also disappears for circuits containing multiple global qubits.
5. No first-use latency reappears when a previously unused peer is encountered.
6. The warm-up time roughly matches the original first-run penalty.

If latency remains only for multi-global-qubit circuits, the next step is to extend peer enumeration from `1 << bit` to the actual set of rank masks used by the circuit.
