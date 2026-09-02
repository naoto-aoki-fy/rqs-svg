#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <climits>
#include <stdint.h>

#include "cmdline.h"

#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <atlc/cuda.hpp>

#include <atlc/check_cuda.hpp>
#include <atlc/log2_int.hpp>

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0/SQRT2)

typedef double my_float_t;
typedef cuda::std::complex<my_float_t> my_complex_t;

const int max_num_gpus = 8;
__constant__ my_complex_t* state_data_device_list_constmem[max_num_gpus];

static bool parse_gpu_list(char const* value, std::vector<int>* gpu_list) {
    gpu_list->clear();
    char const* current = value;
    while (*current != '\0') {
        char* end;
        errno = 0;
        long const gpu_id = strtol(current, &end, 10);
        if (errno != 0 || end == current || gpu_id < 0 || gpu_id > INT_MAX ||
            (*end != ',' && *end != '\0')) {
            return false;
        }
        gpu_list->push_back(static_cast<int>(gpu_id));
        if (*end == '\0') {
            return true;
        }
        current = end + 1;
    }
    return false;
}

class hadamard_local
{
public:
    static __device__ __host__ void apply(
        int const num_split_areas,
        int const log_num_split_areas,
        int64_t const thread_num,
        int64_t const num_qubits,
        int64_t const target_qubit_num,
        my_complex_t **const state_data)
    {
        (void)num_split_areas;
        uint64_t const num_qubits_local = num_qubits - log_num_split_areas;
        uint64_t const local_pair_mask = (((uint64_t)1) << (num_qubits_local - 1)) - 1;
        uint64_t const split_num = ((uint64_t)thread_num) >> (num_qubits_local - 1);
        uint64_t const local_pair_num = ((uint64_t)thread_num) & local_pair_mask;
        uint64_t const target_mask = ((uint64_t)1) << target_qubit_num;
        uint64_t const lower_mask = target_mask - 1;
        uint64_t const address_0 = (local_pair_num & lower_mask) | ((local_pair_num & ~lower_mask) << 1);
        uint64_t const address_1 = address_0 | target_mask;
        my_complex_t *const state = state_data[split_num];
        my_complex_t const amp_state_0 = state[address_0];
        my_complex_t const amp_state_1 = state[address_1];
        state[address_0] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state[address_1] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

class hadamard_global_naive
{
public:
    static __device__ __host__ void apply(
        int const num_split_areas,
        int const log_num_split_areas,
        int64_t const thread_num,
        int64_t const num_qubits,
        int64_t const target_qubit_num,
        my_complex_t **const state_data)
    {
        (void)num_split_areas;
        uint64_t const num_qubits_local = num_qubits - log_num_split_areas;
        uint64_t const split_mask = (((uint64_t)1) << num_qubits_local) - 1;
        uint64_t const address = ((uint64_t)thread_num) & split_mask;
        uint64_t const split_pair_num = ((uint64_t)thread_num) >> num_qubits_local;
        uint64_t const target_split_bit = target_qubit_num - num_qubits_local;
        uint64_t const target_split_mask = ((uint64_t)1) << target_split_bit;
        uint64_t const lower_split_mask = target_split_mask - 1;
        uint64_t const split_0 = (split_pair_num & lower_split_mask) | ((split_pair_num & ~lower_split_mask) << 1);
        uint64_t const split_1 = split_0 | target_split_mask;
        my_complex_t const amp_state_0 = state_data[split_0][address];
        my_complex_t const amp_state_1 = state_data[split_1][address];
        state_data[split_0][address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[split_1][address] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

class hadamard_global_proposed {
public:
    static __device__ __host__ void apply(int const num_split_areas, int const log_num_split_areas, int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t **const state_data)
    {
        int64_t const num_qubits_local = num_qubits - log_num_split_areas;
        uint64_t const num_threads_local = ((uint64_t)1 << (num_qubits_local - 1));
        uint64_t const local_mask = num_threads_local - 1;
        uint64_t const split_num = thread_num >> (num_qubits_local - 1);
        uint64_t const local_thread_num = thread_num & local_mask;
        uint64_t const target_split_bit = target_qubit_num - num_qubits_local;
        // uint64_t const peer_split_num = split_num ^ (((uint64_t)1) << target_split_bit);
        uint64_t const address_high_bit = (split_num >> target_split_bit) & 1;
        uint64_t const address = local_thread_num | (address_high_bit << (num_qubits_local - 1));
        uint64_t const split_0 = split_num & ~(((uint64_t)1) << target_split_bit);
        uint64_t const split_1 = split_0 | (((uint64_t)1) << target_split_bit);
        my_complex_t const amp_state_0 = state_data[split_0][address];
        my_complex_t const amp_state_1 = state_data[split_1][address];
        state_data[split_0][address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[split_1][address] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

typedef hadamard_global_proposed hadamard_global;

template<class Gate>
__global__ void cuda_gate(int const num_split_areas, int const log_num_split_areas, int64_t const split_num, int64_t const num_qubits, int64_t const target_qubit_num) {
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits, target_qubit_num, state_data_device_list_constmem);
}

int main(int argc, char** argv) {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    MPI_Init(&argc, &argv);
    int rank;
    int num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    struct gengetopt_args_info args_info;
    if (cmdline_parser(argc, argv, &args_info) != 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    std::vector<int> gpu_list;
    if (!parse_gpu_list(args_info.gpus_arg, &gpu_list)) {
        fprintf(stderr, "[error] invalid GPU list: %s\n", args_info.gpus_arg);
        cmdline_parser_print_help();
        cmdline_parser_free(&args_info);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int const num_qubits = args_info.num_qubits_arg;
    if (num_qubits <= 0 || num_qubits >= 63) {
        fprintf(stderr, "[error] --num-qubits must be between 1 and 62: %d\n", num_qubits);
        cmdline_parser_print_help();
        cmdline_parser_free(&args_info);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int const num_samples = args_info.num_samples_arg;
    if (num_samples <= 0) {
        fprintf(stderr, "[error] --num-samples must be greater than 0: %d\n", num_samples);
        cmdline_parser_print_help();
        cmdline_parser_free(&args_info);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    cmdline_parser_free(&args_info);

    if (gpu_list.size() > max_num_gpus || (gpu_list.size() & (gpu_list.size() - 1)) != 0 ||
        static_cast<int>(gpu_list.size()) != num_ranks) {
        if (rank == 0) {
            fprintf(stderr, "[error] GPU list must contain one GPU ID per MPI rank (1, 2, 4, or 8)\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int const num_gpus = gpu_list.size();
    int const log_num_gpus = atlc::log2_int(num_gpus);
    if (num_qubits <= log_num_gpus) {
        fprintf(stderr, "[error] num_qubits must be greater than log2(num_gpus)\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (rank == 0) {
        fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            fprintf(stderr, "%d, ", gpu_list[gpu_num]);
        }
        fprintf(stderr, ")\n");
    }

    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);

    fprintf(stderr, "[info] num_samples=%d\n", num_samples);
    int const log_block_size = 8;
    fprintf(stderr, "[info] log_block_size=%d\n", log_block_size);
    int const target_qubit_num_begin = 0;
    int const target_qubit_num_end = num_qubits;
    // int const target_qubit_num_end = 1;

    int const gpu_id = gpu_list[rank];
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
    cudaStream_t stream;
    cudaEvent_t event_1;
    cudaEvent_t event_2;
    ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
    ATLC_CHECK_CUDA(cudaEventCreate, &event_1);
    ATLC_CHECK_CUDA(cudaEventCreate, &event_2);
    ATLC_DEFER_CODE({
        ATLC_CHECK_CUDA(cudaEventDestroy, event_2);
        ATLC_CHECK_CUDA(cudaEventDestroy, event_1);
        ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
    });

    my_complex_t** state_data_device_list_constmem_addr;
    ATLC_CHECK_CUDA(cudaGetSymbolAddress<decltype(state_data_device_list_constmem)>,
        (void**)&state_data_device_list_constmem_addr, state_data_device_list_constmem);

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    fprintf(stderr, "[info] malloc device memory\n");

    my_complex_t* state_data_device;
    ATLC_CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));

    cudaIpcMemHandle_t local_handle;
    ATLC_CHECK_CUDA(cudaIpcGetMemHandle, &local_handle, state_data_device);
    std::vector<cudaIpcMemHandle_t> handles(num_ranks);
    MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE,
        handles.data(), sizeof(local_handle), MPI_BYTE, MPI_COMM_WORLD);

    std::vector<my_complex_t*> state_data_device_list(num_ranks);
    state_data_device_list[rank] = state_data_device;
    for (int peer = 0; peer < num_ranks; ++peer) {
        if (peer != rank) {
            ATLC_CHECK_CUDA(cudaIpcOpenMemHandle, &state_data_device_list[peer],
                handles[peer], cudaIpcMemLazyEnablePeerAccess);
        }
    }
    ATLC_DEFER_CODE({
        for (int peer = 0; peer < num_ranks; ++peer) {
            if (peer != rank) {
                ATLC_CHECK_CUDA(cudaIpcCloseMemHandle, state_data_device_list[peer]);
            }
        }
        ATLC_CHECK_CUDA(cudaFree, state_data_device);
    });

    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device_list_constmem_addr,
        state_data_device_list.data(), state_data_device_list.size() * sizeof(state_data_device_list[0]),
        cudaMemcpyHostToDevice, stream);


    fprintf(stderr, "[info] initializing state to |0...0>\n");
    ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device, 0,
        num_states_local * sizeof(*state_data_device), stream);

    my_complex_t const zero_state_amplitude(1.0, 0.0);
    if (rank == 0) {
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &zero_state_amplitude,
            sizeof(zero_state_amplitude), cudaMemcpyHostToDevice, stream);
    }
    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stderr, "[info] gpu_hadamard\n");

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        ATLC_CHECK_CUDA(cudaEventRecord, event_1, stream);

        for(int target_qubit_num = target_qubit_num_begin; target_qubit_num < target_qubit_num_end; target_qubit_num++) {

            if (target_qubit_num < num_qubits - log_num_gpus) {
                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<hadamard_local>, num_blocks,
                    block_size, 0, stream, num_gpus, log_num_gpus, rank, num_qubits, target_qubit_num);
            } else {
                if (target_qubit_num == num_qubits - log_num_gpus) {
                    // The first global gate may read a peer's state split, so
                    // all preceding local gates must have completed first.
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<hadamard_global>, num_blocks,
                    block_size, 0, stream, num_gpus, log_num_gpus, rank, num_qubits, target_qubit_num);
                // Remote state is consumed by the next global gate.  Complete
                // CUDA work on every rank before proceeding collectively.
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        ATLC_CHECK_CUDA(cudaEventRecord, event_2, stream);
        ATLC_CHECK_CUDA(cudaEventSynchronize, event_2);

        float elapsed_ms;
        ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, event_1, event_2);
        double const elapsed_local = elapsed_ms * 1e-3;
        double elapsed_max;
        MPI_Reduce(&elapsed_local, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            fprintf(stderr, "[info] elapsed_gpu=%lf\n", elapsed_max);
            fprintf(stdout, "%lf\n", elapsed_max);
        }
    }

    MPI_Finalize();
    return 0;

}
