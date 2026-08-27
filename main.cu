#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <climits>
#include <stdint.h>

#include "cmdline.h"

#include <unordered_set>
#include <vector>

#include <omp.h>
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

    struct gengetopt_args_info args_info;
    if (cmdline_parser(argc, argv, &args_info) != 0) {
        return EXIT_FAILURE;
    }

    std::vector<int> gpu_list;
    if (!parse_gpu_list(args_info.gpus_arg, &gpu_list)) {
        fprintf(stderr, "[error] invalid GPU list: %s\n", args_info.gpus_arg);
        cmdline_parser_print_help();
        cmdline_parser_free(&args_info);
        return EXIT_FAILURE;
    }

    int const num_qubits = args_info.num_qubits_arg;
    if (num_qubits <= 0 || num_qubits >= 63) {
        fprintf(stderr, "[error] --num-qubits must be between 1 and 62: %d\n", num_qubits);
        cmdline_parser_print_help();
        cmdline_parser_free(&args_info);
        return EXIT_FAILURE;
    }
    cmdline_parser_free(&args_info);

    if (gpu_list.size() > max_num_gpus || (gpu_list.size() & (gpu_list.size() - 1)) != 0) {
        fprintf(stderr, "[error] GPU list must contain 1, 2, 4, or 8 GPU IDs\n");
        return EXIT_FAILURE;
    }

    int const num_gpus = gpu_list.size();
    int const log_num_gpus = atlc::log2_int(num_gpus);
    if (num_qubits <= log_num_gpus) {
        fprintf(stderr, "[error] num_qubits must be greater than log2(num_gpus)\n");
        return EXIT_FAILURE;
    }
    fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        fprintf(stderr, "%d, ", gpu_list[gpu_num]);
    }
    fprintf(stderr, ")\n");

    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);

    int const num_samples = 64;
    int const log_block_size = 8;
    fprintf(stderr, "[info] log_block_size=%d\n", log_block_size);
    int const target_qubit_num_begin = 0;
    int const target_qubit_num_end = num_qubits;
    // int const target_qubit_num_end = 1;

    std::vector<int> gpu_list_dedup;
    {
        std::unordered_set<int> gpu_set(gpu_list.begin(), gpu_list.end());
        gpu_list_dedup = {gpu_set.begin(), gpu_set.end()};
    }

    std::vector<cudaStream_t> stream(num_gpus);
    std::vector<cudaEvent_t> event_1(num_gpus);
    std::vector<cudaEvent_t> event_2(num_gpus);
    std::vector<cudaEvent_t> done(num_gpus);
    cudaEvent_t layer_done;

    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

        int const gpu_id = gpu_list[gpu_num]; 
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        ATLC_CHECK_CUDA(cudaStreamCreate, &stream[gpu_num]);

        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1[gpu_num], cudaEventDefault);

        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2[gpu_num], cudaEventDefault);

        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &done[gpu_num], cudaEventDisableTiming);

    }
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &layer_done, cudaEventDisableTiming);
    ATLC_DEFER_CODE({
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
        ATLC_CHECK_CUDA(cudaEventDestroy, layer_done);
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventDestroy, done[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventDestroy, event_2[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventDestroy, event_1[gpu_num]);
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream[gpu_num]);
        }
    });

    // Join all GPU streams without blocking the host. Events may be waited on
    // from a stream belonging to another device, so stream 0 can coordinate a
    // layer and fan the resulting dependency back out to every GPU.
    auto enqueue_inter_gpu_barrier = [&]() {
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventRecord, done[gpu_num], stream[gpu_num]);
        }

        ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream[0], done[gpu_num], 0);
        }
        ATLC_CHECK_CUDA(cudaEventRecord, layer_done, stream[0]);

        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream[gpu_num], layer_done, 0);
        }
    };

    std::vector<my_complex_t**> state_data_device_list_constmem_addr(num_gpus);
    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        my_complex_t** addr;
        ATLC_CHECK_CUDA(cudaGetSymbolAddress<decltype(state_data_device_list_constmem)>, (void**)&addr, state_data_device_list_constmem);

        state_data_device_list_constmem_addr[gpu_num] = addr;
    }

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    fprintf(stderr, "[info] malloc device memory\n");

    std::vector<my_complex_t*> state_data_device_list(num_gpus);

    for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

        int const gpu_id = gpu_list[gpu_num]; 
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        my_complex_t* state_data_device;
        ATLC_CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));
        state_data_device_list[gpu_num] = state_data_device;

    }
    ATLC_DEFER_CODE({
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaFree, state_data_device_list[gpu_num]);
        }
    });

    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device_list_constmem_addr[gpu_num], &state_data_device_list[0], state_data_device_list.size() * sizeof(state_data_device_list[0]), cudaMemcpyHostToDevice, stream[gpu_num]);
    }

    for(int gpu_num = 0; gpu_num < gpu_list_dedup.size(); gpu_num++) {
        int const gpu_id = gpu_list_dedup[gpu_num]; 
        for(int gpu_num_2 = 0; gpu_num_2 < gpu_list_dedup.size(); gpu_num_2++) {
            if(gpu_num == gpu_num_2) continue;
            int const gpu_id_2 = gpu_list_dedup[gpu_num_2]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaDeviceEnablePeerAccess, gpu_id_2, 0);
        }
    }

    for(int gpu_num = 0; gpu_num < gpu_list_dedup.size(); gpu_num++) {
        int const gpu_id = gpu_list_dedup[gpu_num]; 
        for(int gpu_num_2 = 0; gpu_num_2 < gpu_list_dedup.size(); gpu_num_2++) {
            if(gpu_num == gpu_num_2) continue;
            int const gpu_id_2 = gpu_list_dedup[gpu_num_2]; 
            int canAccessPeer;
            ATLC_CHECK_CUDA(cudaDeviceCanAccessPeer, &canAccessPeer, gpu_id, gpu_id_2);
            if (!canAccessPeer) {
                fprintf(stderr, "[error] GPU%d can not access GPU%d\n", gpu_id, gpu_id_2);
            }
        }
    }


    fprintf(stderr, "[info] initializing state to |0...0>\n");
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
        ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device_list[gpu_num], 0,
            num_states_local * sizeof(*state_data_device_list[gpu_num]), stream[gpu_num]);
    }

    my_complex_t const zero_state_amplitude(1.0, 0.0);
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device_list[0], &zero_state_amplitude,
        sizeof(zero_state_amplitude), cudaMemcpyHostToDevice, stream[0]);
    enqueue_inter_gpu_barrier();
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
    ATLC_CHECK_CUDA(cudaEventSynchronize, layer_done);

    fprintf(stderr, "[info] gpu_hadamard\n");

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaEventRecord, event_1[gpu_num], stream[gpu_num]);
        }

        for(int target_qubit_num = target_qubit_num_begin; target_qubit_num < target_qubit_num_end; target_qubit_num++) {

            // The first global gate reads state owned by other GPUs.  Wait for
            // every GPU's preceding local gates before allowing those reads.
            if (target_qubit_num == num_qubits - log_num_gpus) {
                enqueue_inter_gpu_barrier();
            }

            for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

                int const gpu_id = gpu_list[gpu_num]; 
                ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

                if (target_qubit_num < num_qubits - log_num_gpus) {
                    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<hadamard_local>, num_blocks, block_size, 0, stream[gpu_num], num_gpus, log_num_gpus, gpu_num, num_qubits, target_qubit_num);
                } else {
                    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<hadamard_global>, num_blocks, block_size, 0, stream[gpu_num], num_gpus, log_num_gpus, gpu_num, num_qubits, target_qubit_num);
                }
            }

            if (target_qubit_num >= num_qubits - log_num_gpus && target_qubit_num < target_qubit_num_end - 1) {
                enqueue_inter_gpu_barrier();
            }

        }

        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaEventRecord, event_2[gpu_num], stream[gpu_num]);
        }

        enqueue_inter_gpu_barrier();
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[0]);
        ATLC_CHECK_CUDA(cudaEventSynchronize, layer_done);

        double elapsed_gpu = 0;
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

            float elapsed_i_ms;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_i_ms, event_1[gpu_num], event_2[gpu_num]);
            double const elapsed_i = elapsed_i_ms * 1e-3;

            if(elapsed_i > elapsed_gpu) {
                elapsed_gpu = elapsed_i;
            }
        }
        fprintf(stderr, "[info] elapsed_gpu=%lf\n", elapsed_gpu);
        fprintf(stdout, "%lf\n", elapsed_gpu);

    }

    return 0;

}
