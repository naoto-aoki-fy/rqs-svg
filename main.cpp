#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <vector>

#include <omp.h>

#include "cmdline.h"

#define SQRT2 (1.41421356237309504880168872420969807856967187537694)
#define INV_SQRT2 (1.0 / SQRT2)

using my_float_t = double;
using my_complex_t = std::complex<my_float_t>;

class hadamard_local {
public:
    static void apply(int const num_split_areas, int const log_num_split_areas,
                      int64_t const thread_num, int64_t const num_qubits,
                      int64_t const target_qubit_num, my_complex_t **const state_data) {
        (void)num_split_areas;
        uint64_t const num_qubits_local = num_qubits - log_num_split_areas;
        uint64_t const local_pair_mask = (UINT64_C(1) << (num_qubits_local - 1)) - 1;
        uint64_t const split_num = static_cast<uint64_t>(thread_num) >> (num_qubits_local - 1);
        uint64_t const local_pair_num = static_cast<uint64_t>(thread_num) & local_pair_mask;
        uint64_t const target_mask = UINT64_C(1) << target_qubit_num;
        uint64_t const lower_mask = target_mask - 1;
        uint64_t const address_0 =
            (local_pair_num & lower_mask) | ((local_pair_num & ~lower_mask) << 1);
        uint64_t const address_1 = address_0 | target_mask;
        my_complex_t *const state = state_data[split_num];
        my_complex_t const amp_state_0 = state[address_0];
        my_complex_t const amp_state_1 = state[address_1];
        state[address_0] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state[address_1] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

class hadamard_global {
public:
    static void apply(int const num_split_areas, int const log_num_split_areas,
                      int64_t const thread_num, int64_t const num_qubits,
                      int64_t const target_qubit_num, my_complex_t **const state_data) {
        (void)num_split_areas;
        int64_t const num_qubits_local = num_qubits - log_num_split_areas;
        uint64_t const num_threads_local = UINT64_C(1) << (num_qubits_local - 1);
        uint64_t const local_mask = num_threads_local - 1;
        uint64_t const split_num = static_cast<uint64_t>(thread_num) >> (num_qubits_local - 1);
        uint64_t const local_thread_num = static_cast<uint64_t>(thread_num) & local_mask;
        uint64_t const target_split_bit = target_qubit_num - num_qubits_local;
        uint64_t const address_high_bit = (split_num >> target_split_bit) & 1;
        uint64_t const address =
            local_thread_num | (address_high_bit << (num_qubits_local - 1));
        uint64_t const split_0 = split_num & ~(UINT64_C(1) << target_split_bit);
        uint64_t const split_1 = split_0 | (UINT64_C(1) << target_split_bit);
        my_complex_t const amp_state_0 = state_data[split_0][address];
        my_complex_t const amp_state_1 = state_data[split_1][address];
        state_data[split_0][address] = (amp_state_0 + amp_state_1) * INV_SQRT2;
        state_data[split_1][address] = (amp_state_0 - amp_state_1) * INV_SQRT2;
    }
};

// CPU replacement for the CUDA kernel. OpenMP assigns the logical CUDA thread
// numbers to CPU threads while the gate address calculations remain unchanged.
template<class Gate>
static void cpu_gate(int const num_split_areas, int const log_num_split_areas,
                     int64_t const num_qubits, int64_t const target_qubit_num,
                     my_complex_t **const state_data) {
    int64_t const num_gate_threads = INT64_C(1) << (num_qubits - 1);
#pragma omp parallel for schedule(static)
    for (int64_t thread_num = 0; thread_num < num_gate_threads; ++thread_num) {
        Gate::apply(num_split_areas, log_num_split_areas, thread_num, num_qubits,
                    target_qubit_num, state_data);
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 1024 * 512);

    gengetopt_args_info args_info;
    if (cmdline_parser(argc, argv, &args_info) != 0) {
        return EXIT_FAILURE;
    }

    int const num_qubits = args_info.num_qubits_arg;
    int const num_samples = args_info.num_samples_arg;
    int const requested_threads = args_info.threads_arg;
    if (num_qubits <= 0 || num_qubits >= 63) {
        std::fprintf(stderr, "[error] --num-qubits must be between 1 and 62: %d\n", num_qubits);
        cmdline_parser_free(&args_info);
        return EXIT_FAILURE;
    }
    if (num_samples <= 0 || requested_threads < 0) {
        std::fprintf(stderr, "[error] --num-samples must be positive and --threads nonnegative\n");
        cmdline_parser_free(&args_info);
        return EXIT_FAILURE;
    }
    cmdline_parser_free(&args_info);

    if (requested_threads > 0) {
        omp_set_num_threads(requested_threads);
    }
    int const num_threads = requested_threads > 0 ? requested_threads : omp_get_max_threads();

    // Preserve the original split-state layout. A power-of-two number of CPU
    // areas lets the unchanged local/global gate indexing address every state.
    int num_split_areas = 1;
    int log_num_split_areas = 0;
    while (num_split_areas <= num_threads / 2 && log_num_split_areas < num_qubits - 1) {
        num_split_areas *= 2;
        ++log_num_split_areas;
    }
    int const num_qubits_local = num_qubits - log_num_split_areas;
    uint64_t const num_states_local = UINT64_C(1) << num_qubits_local;

    std::fprintf(stderr, "[info] num_threads=%d\n", num_threads);
    std::fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);
    std::fprintf(stderr, "[info] num_samples=%d\n", num_samples);
    std::fprintf(stderr, "[info] num_split_areas=%d\n", num_split_areas);
    std::fprintf(stderr, "[info] allocating CPU memory\n");

    std::vector<std::vector<my_complex_t>> state_storage;
    try {
        state_storage.resize(num_split_areas);
        for (auto &area : state_storage) {
            area.resize(num_states_local);
        }
    } catch (std::bad_alloc const &) {
        std::fprintf(stderr, "[error] unable to allocate state vector\n");
        return EXIT_FAILURE;
    }
    std::vector<my_complex_t *> state_data(num_split_areas);
    for (int area = 0; area < num_split_areas; ++area) {
        state_data[area] = state_storage[area].data();
    }
    state_data[0][0] = my_complex_t(1.0, 0.0);

    std::fprintf(stderr, "[info] cpu_hadamard\n");
    for (int sample_num = 0; sample_num < num_samples; ++sample_num) {
        auto const begin = std::chrono::steady_clock::now();
        for (int target_qubit_num = 0; target_qubit_num < num_qubits; ++target_qubit_num) {
            if (target_qubit_num < num_qubits_local) {
                cpu_gate<hadamard_local>(num_split_areas, log_num_split_areas,
                                         num_qubits, target_qubit_num, state_data.data());
            } else {
                cpu_gate<hadamard_global>(num_split_areas, log_num_split_areas,
                                          num_qubits, target_qubit_num, state_data.data());
            }
        }
        double const elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        std::fprintf(stderr, "[info] elapsed_cpu=%lf\n", elapsed);
        std::fprintf(stdout, "%lf\n", elapsed);
    }
    return EXIT_SUCCESS;
}
