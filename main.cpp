#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <omp.h>

#include "cmdline.h"

struct complex_t {
    double real;
    double imag;
};

namespace {

constexpr double INV_SQRT2 = 0.707106781186547524400844362104849039;

complex_t add(complex_t const a, complex_t const b) {
    return {a.real + b.real, a.imag + b.imag};
}

complex_t subtract(complex_t const a, complex_t const b) {
    return {a.real - b.real, a.imag - b.imag};
}

complex_t scale(complex_t const value, double const factor) {
    return {value.real * factor, value.imag * factor};
}

void apply_hadamard_pair(complex_t* const state, uint64_t const pair,
                         int const target) {
    uint64_t const target_mask = UINT64_C(1) << target;
    uint64_t const lower_mask = target_mask - 1;
    uint64_t const i0 = (pair & lower_mask) | ((pair & ~lower_mask) << 1);
    uint64_t const i1 = i0 | target_mask;
    complex_t const a = state[i0];
    complex_t const b = state[i1];
    state[i0] = scale(add(a, b), INV_SQRT2);
    state[i1] = scale(subtract(a, b), INV_SQRT2);
}

[[maybe_unused]] void hadamard_serial(complex_t* const state,
                                      uint64_t const num_states,
                                      int const target) {
    for (uint64_t pair = 0; pair < num_states / 2; ++pair) {
        apply_hadamard_pair(state, pair, target);
    }
}

[[maybe_unused]] void hadamard_openmp(complex_t* const state,
                                      uint64_t const num_states,
                                      int const target) {
    int64_t const num_pairs = static_cast<int64_t>(num_states / 2);
#pragma omp parallel for schedule(static)
    for (int64_t pair = 0; pair < num_pairs; ++pair) {
        apply_hadamard_pair(state, static_cast<uint64_t>(pair), target);
    }
}

}  // namespace

#ifndef QCS_TESTING
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 512 * 1024);

    gengetopt_args_info args_info;
    if (cmdline_parser(argc, argv, &args_info) != 0) {
        return EXIT_FAILURE;
    }

    int const num_qubits = args_info.num_qubits_arg;
    int const num_samples = args_info.num_samples_arg;
    int const threads = args_info.threads_arg;
    cmdline_parser_free(&args_info);

    if (num_qubits <= 0 || num_qubits >= 63) {
        std::fprintf(stderr,
                     "[error] --num-qubits must be between 1 and 62: %d\n",
                     num_qubits);
        return EXIT_FAILURE;
    }
    if (num_samples <= 0) {
        std::fprintf(stderr,
                     "[error] --num-samples must be greater than 0: %d\n",
                     num_samples);
        return EXIT_FAILURE;
    }
    if (threads < 0) {
        std::fprintf(stderr, "[error] --threads must be non-negative: %d\n",
                     threads);
        return EXIT_FAILURE;
    }
    if (threads > 0) {
        omp_set_num_threads(threads);
    }

    uint64_t const num_states = UINT64_C(1) << num_qubits;
    if (num_states > std::numeric_limits<size_t>::max() / sizeof(complex_t) ||
        num_states > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        std::fprintf(stderr,
                     "[error] state vector size overflows this platform\n");
        return EXIT_FAILURE;
    }
    size_t const state_bytes = static_cast<size_t>(num_states) * sizeof(complex_t);
    auto* const state = static_cast<complex_t*>(std::malloc(state_bytes));
    if (state == nullptr) {
        std::fprintf(stderr, "[error] unable to allocate %zu bytes for state\n",
                     state_bytes);
        return EXIT_FAILURE;
    }

    int64_t const signed_num_states = static_cast<int64_t>(num_states);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < signed_num_states; ++i) {
        state[i] = {0.0, 0.0};
    }
    state[0] = {1.0, 0.0};

    std::fprintf(stderr, "[info] num_threads=%d\n", omp_get_max_threads());
    std::fprintf(stderr, "[info] num_qubits=%d\n", num_qubits);
    std::fprintf(stderr, "[info] num_samples=%d\n", num_samples);

    std::vector<double> elapsed(static_cast<size_t>(num_samples));
    double sample_start = 0.0;
    int64_t const num_pairs = signed_num_states / 2;

#pragma omp parallel shared(sample_start, elapsed, state)
    {
        for (int sample = 0; sample < num_samples; ++sample) {
#pragma omp single
            sample_start = omp_get_wtime();

            for (int target = 0; target < num_qubits; ++target) {
#pragma omp for schedule(static)
                for (int64_t pair = 0; pair < num_pairs; ++pair) {
                    apply_hadamard_pair(state, static_cast<uint64_t>(pair),
                                        target);
                }
            }

#pragma omp single
            elapsed[static_cast<size_t>(sample)] =
                omp_get_wtime() - sample_start;
        }
    }

    for (double const seconds : elapsed) {
        std::fprintf(stderr, "[info] elapsed_cpu=%lf\n", seconds);
        std::fprintf(stdout, "%lf\n", seconds);
    }

    std::free(state);
    return EXIT_SUCCESS;
}
#endif
