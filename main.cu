#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdint.h>

#include <unordered_set>

#include <omp.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <atlc/cuda.hpp>

#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/block_reduce_sum.cuh>
#include <atlc/log2_int.hpp>

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0/SQRT2)

typedef double my_float_t;
typedef cuda::std::complex<my_float_t> my_complex_t;

const int max_num_gpus = 8;
__constant__ my_complex_t* state_data_device_list_constmem[max_num_gpus];

__global__ void norm_sum_reduce_kernel(my_complex_t const* const input_global, my_float_t* const output_global)
{
    extern __shared__ my_float_t warp_partials[];
    int64_t const idx =  blockDim.x * blockIdx.x + threadIdx.x;
    my_float_t sum = cuda::std::norm(input_global[idx]);
    atlc::block_reduce_sum_core(&sum, 1, warp_partials);
    if (threadIdx.x == 0) {
        output_global[blockIdx.x] = sum;
    }
}

__global__ void sum_reduce_kernel(my_float_t const* const input_global, my_float_t* const output_global)
{
    extern __shared__ my_float_t warp_partials[];
    int64_t const idx =  blockDim.x * blockIdx.x + threadIdx.x;
    my_float_t sum = input_global[idx];
    atlc::block_reduce_sum_core(&sum, 1, warp_partials);
    if (threadIdx.x == 0) {
        output_global[blockIdx.x] = sum;
    }
}

__global__ void normalize_kernel(my_float_t* const data_global, my_float_t const factor)
{
    int64_t const idx = blockDim.x * blockIdx.x + threadIdx.x;
    data_global[idx] *= factor;
}

class hadamard_naive { public:
    static __device__ __host__ void apply(int const num_split_areas, int const log_num_split_areas, int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t** const state_data) {

        uint64_t const lower_mask = (((uint64_t)1)<<target_qubit_num) - (uint64_t)1;
        uint64_t const split_mask = (((uint64_t)1)<<((uint64_t)(num_qubits - log_num_split_areas))) - (uint64_t)1;

        int64_t const index_state_lower = thread_num & lower_mask;
        int64_t const index_state_higher = (thread_num & ~lower_mask) << ((int64_t)1);

        int64_t const index_state_0 = index_state_lower | index_state_higher;
        int64_t const index_state_1 = index_state_0 | (((int64_t)1)<<target_qubit_num);

        int64_t const index_state_0_split_num = index_state_0 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_0_split_address = index_state_0 & split_mask;

        int64_t const index_state_1_split_num = index_state_1 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_1_split_address = index_state_1 & split_mask;

        my_complex_t const amp_state_0 = state_data[index_state_0_split_num][index_state_0_split_address];
        my_complex_t const amp_state_1 = state_data[index_state_1_split_num][index_state_1_split_address];

        state_data[index_state_0_split_num][index_state_0_split_address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[index_state_1_split_num][index_state_1_split_address] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

class hadamard_proposed { public:
    static __device__ __host__ void apply(int const num_split_areas, int const log_num_split_areas, int64_t const thread_num, int64_t const num_qubits, int64_t const target_qubit_num, my_complex_t** const state_data) {

        uint64_t const split_mask = (((uint64_t)1)<<((uint64_t)(num_qubits - log_num_split_areas))) - (uint64_t)1;

        uint64_t const local_mask = (((uint64_t)1)<<((uint64_t)(num_qubits - log_num_split_areas - 1))) - (uint64_t)1;

        uint64_t const index_global = thread_num & ~local_mask;
        uint64_t const index_local = thread_num & local_mask;
        uint64_t const target_index = ((thread_num >> target_qubit_num) & 1) << (num_qubits - log_num_split_areas - 1);

        int64_t const index_state_0 =
            (
                (index_global << 1)
                | target_index
                | index_local
            ) & ~(((uint64_t)1)<<target_qubit_num);

        int64_t const index_state_1 = index_state_0 | (((uint64_t)1)<<target_qubit_num);



        int64_t const index_state_0_split_num = index_state_0 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_0_split_address = index_state_0 & split_mask;

        int64_t const index_state_1_split_num = index_state_1 >> (num_qubits - log_num_split_areas);
        int64_t const index_state_1_split_address = index_state_1 & split_mask;

        my_complex_t const amp_state_0 = state_data[index_state_0_split_num][index_state_0_split_address];
        my_complex_t const amp_state_1 = state_data[index_state_1_split_num][index_state_1_split_address];

        state_data[index_state_0_split_num][index_state_0_split_address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[index_state_1_split_num][index_state_1_split_address] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

typedef hadamard_naive hadamard;

template<class Gate>
__global__ void cuda_gate(int const num_split_areas, int const log_num_split_areas, int64_t const split_num, int64_t const num_qubits, int64_t const target_qubit_num) {
    int64_t const num_qubits_local = num_qubits - log_num_split_areas;
    int64_t const num_threads_local = ((int64_t)1) << (num_qubits_local-1);

    int64_t const thread_num = threadIdx.x + blockIdx.x * blockDim.x + num_threads_local * split_num;
    Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits, target_qubit_num, state_data_device_list_constmem);
}

int main(int argc, char** argv) {

    setvbuf(stdout, NULL, _IOLBF, 1024 * 512);

    bool const do_normalization = false;
    std::vector<int> gpu_list{0, 1, 2, 3, 4, 5, 6, 7};
    // std::vector<int> gpu_list{0, 1, 2, 3};
    int const num_gpus = gpu_list.size();
    int const log_num_gpus = atlc::log2_int(num_gpus);
    fprintf(stderr, "[info] num_gpus=%d (", num_gpus);
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        fprintf(stderr, "%d, ", gpu_list[gpu_num]);
    }
    fprintf(stderr, ")\n");

    int const num_qubits = 24;
    fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);

    int const num_samples = 64;
    int const rng_seed = 12345;

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

    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

        int const gpu_id = gpu_list[gpu_num]; 
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        ATLC_CHECK_CUDA(cudaStreamCreate, &stream[gpu_num]);

        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1[gpu_num], cudaEventDefault);

        ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2[gpu_num], cudaEventDefault);

    }
    ATLC_DEFER_CODE({
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventDestroy, event_2[gpu_num]);
            ATLC_CHECK_CUDA(cudaEventDestroy, event_1[gpu_num]);
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream[gpu_num]);
        }
    });

    std::vector<my_complex_t**> state_data_device_list_constmem_addr(num_gpus);
    for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        my_complex_t** addr;
        ATLC_CHECK_CUDA(cudaGetSymbolAddress<decltype(state_data_device_list_constmem)>, (void**)&addr, state_data_device_list_constmem);

        state_data_device_list_constmem_addr[gpu_num] = addr;
    }

    int64_t const num_states = INT64_C(1) << num_qubits;

    int const num_qubits_local = num_qubits - log_num_gpus;
    int64_t const num_states_local = ((int64_t)1) << ((int64_t)num_qubits_local);
    int const block_size = 1 << log_block_size;
    int64_t const num_blocks = ((int64_t)1) << ((int64_t)(num_qubits_local - 1 - log_block_size));

    fprintf(stderr, "[info] malloc device memory\n");

    std::vector<my_complex_t*> state_data_device_list(num_gpus);

    std::vector<my_float_t*> norm_sum_device_list(num_gpus);

    for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

        int const gpu_id = gpu_list[gpu_num]; 
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

        my_complex_t* state_data_device;
        ATLC_CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));
        state_data_device_list[gpu_num] = state_data_device;

        my_float_t* norm_sum_device;
        ATLC_CHECK_CUDA(cudaMalloc, &norm_sum_device, (INT64_C(1) << (num_qubits - log_block_size)) * sizeof(my_float_t));
        norm_sum_device_list[gpu_num] = norm_sum_device;

    }
    ATLC_DEFER_CODE({
        for (int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_list[gpu_num]);
            ATLC_CHECK_CUDA(cudaFree, norm_sum_device_list[gpu_num]);
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


    fprintf(stderr, "[info] generating random state\n");
    std::vector<curandGenerator_t> rng_device_list(num_gpus);
    for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
        int const gpu_id = gpu_list[gpu_num];
        ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
        ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device_list[gpu_num], CURAND_RNG_PSEUDO_DEFAULT);
        ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device_list[gpu_num], rng_seed + gpu_num);
        ATLC_CHECK_CURAND(curandSetStream, rng_device_list[gpu_num], stream[gpu_num]);
    }

    if (do_normalization) {

        fprintf(stderr, "[info] gpu reduce\n");
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaEventRecord, event_1[gpu_num], stream[gpu_num]);
        }

        std::vector<my_float_t> norm_sum_list(num_gpus);
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

            ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device_list[gpu_num], (my_float_t*)(void*)state_data_device_list[gpu_num], num_states_local * 2, 0.0, 1.0);

            int64_t data_length = num_states_local;
            int64_t num_blocks_reduce;
            int block_size_reduce;

            if (data_length > block_size) {
                block_size_reduce = block_size;
                num_blocks_reduce = data_length >> log_block_size;
            } else {
                block_size_reduce = data_length;
                num_blocks_reduce = 1;
            }

            ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, norm_sum_reduce_kernel, num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream[gpu_num], state_data_device_list[gpu_num], norm_sum_device_list[gpu_num]);

            data_length = num_blocks_reduce;

            while (data_length>1) {
                if (data_length > block_size) {
                    block_size_reduce = block_size;
                    num_blocks_reduce = data_length >> log_block_size;
                } else {
                    block_size_reduce = data_length;
                    num_blocks_reduce = 1;
                }

                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, sum_reduce_kernel, num_blocks_reduce, block_size_reduce, sizeof(my_float_t) * block_size_reduce, stream[gpu_num], norm_sum_device_list[gpu_num], norm_sum_device_list[gpu_num]);

                data_length = num_blocks_reduce;
            }

            ATLC_CHECK_CUDA(cudaMemcpyAsync, &norm_sum_list[gpu_num], norm_sum_device_list[gpu_num], sizeof(my_float_t), cudaMemcpyDeviceToHost, stream[gpu_num]);
        }

        for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
        }

        my_float_t norm_sum_gpu = 0;
        for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
            norm_sum_gpu += norm_sum_list[gpu_num];
        }
        fprintf(stderr, "[info] norm_sum_gpu=%lf\n", norm_sum_gpu);

        fprintf(stderr, "[info] normalize\n");
        my_float_t const normalize_factor = 1.0 / sqrt(norm_sum_gpu);
        for(int gpu_num=0; gpu_num<num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

            int64_t data_length = num_states_local * 2;
            int64_t num_blocks_reduce;
            int block_size_reduce;

            if (data_length > block_size) {
                block_size_reduce = block_size;
                num_blocks_reduce = data_length >> log_block_size;
            } else {
                block_size_reduce = data_length;
                num_blocks_reduce = 1;
            }

            ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, normalize_kernel, 1ULL<<(num_qubits_local+1-log_block_size), block_size, 0, stream[gpu_num], (my_float_t*)(void*)state_data_device_list[gpu_num], normalize_factor);

            ATLC_CHECK_CUDA(cudaEventRecord, event_2[gpu_num], stream[gpu_num]);
        }
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num];
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
        }

        double elapsed_rng = 0;
        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

            float elapsed_i_ms;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_i_ms, event_1[gpu_num], event_2[gpu_num]);
            double const elapsed_i = elapsed_i_ms * 1e-3;

            if (elapsed_i > elapsed_rng) {
                elapsed_rng = elapsed_i;
            }
        }

        fprintf(stderr, "[info] rng elapsed=%lf\n", elapsed_rng);
    }
    fprintf(stderr, "[info] gpu_hadamard\n");

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaEventRecord, event_1[gpu_num], stream[gpu_num]);
        }

        for(int target_qubit_num = target_qubit_num_begin; target_qubit_num < target_qubit_num_end; target_qubit_num++) {

            for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {

                int const gpu_id = gpu_list[gpu_num]; 
                ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<hadamard>, num_blocks, block_size, 0, stream[gpu_num], num_gpus, log_num_gpus, gpu_num, num_qubits, target_qubit_num);
            }

            if (target_qubit_num >= num_qubits - log_num_gpus) {
                for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
                    int const gpu_id = gpu_list[gpu_num]; 
                    ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
                }
            }

        }

        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaEventRecord, event_2[gpu_num], stream[gpu_num]);
        }

        for(int gpu_num = 0; gpu_num < num_gpus; gpu_num++) {
            int const gpu_id = gpu_list[gpu_num]; 
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream[gpu_num]);
        }

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
