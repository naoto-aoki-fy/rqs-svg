#define QCS_TESTING
#include "../main.cpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

bool close(complex_t const a, complex_t const b) {
    return std::abs(a.real - b.real) < 1e-12 &&
           std::abs(a.imag - b.imag) < 1e-12;
}

void check_pair_mapping(int const qubits, int const target) {
    uint64_t const states = UINT64_C(1) << qubits;
    uint64_t const target_mask = UINT64_C(1) << target;
    uint64_t const lower_mask = target_mask - 1;
    std::vector<bool> visited(states, false);
    for (uint64_t pair = 0; pair < states / 2; ++pair) {
        uint64_t const i0 = (pair & lower_mask) | ((pair & ~lower_mask) << 1);
        uint64_t const i1 = i0 | target_mask;
        assert((i0 ^ i1) == target_mask);
        assert(!visited[i0] && !visited[i1]);
        visited[i0] = visited[i1] = true;
    }
    assert(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }));
}

void check_global_proposed(int const qubits, int const log_splits,
                           int const target) {
    uint64_t const states = UINT64_C(1) << qubits;
    uint64_t const split_size = states >> log_splits;
    int const splits = 1 << log_splits;
    std::vector<complex_t> expected(states);
    std::vector<complex_t> actual(states);
    for (uint64_t i = 0; i < states; ++i) {
        expected[i] = actual[i] = {static_cast<double>(i + 1),
                                   -static_cast<double>(i)};
    }

    hadamard_serial(expected.data(), states, target);
    std::vector<complex_t*> split_data(static_cast<size_t>(splits));
    for (int split = 0; split < splits; ++split) {
        split_data[static_cast<size_t>(split)] =
            actual.data() + static_cast<uint64_t>(split) * split_size;
    }
    for (uint64_t thread = 0; thread < states / 2; ++thread) {
        hadamard_global_proposed::apply(
            splits, log_splits, static_cast<int64_t>(thread), qubits, target,
            split_data.data());
    }
    for (uint64_t i = 0; i < states; ++i) {
        assert(close(expected[i], actual[i]));
    }
}

}  // namespace

int main() {
    for (int qubits = 2; qubits <= 12; ++qubits) {
        for (int log_splits = 1; log_splits < qubits; ++log_splits) {
            for (int target = qubits - log_splits; target < qubits;
                 ++target) {
                check_global_proposed(qubits, log_splits, target);
            }
        }
    }

    for (int threads : {1, 2, 3, 4, 8}) {
        omp_set_num_threads(threads);
        for (int qubits = 1; qubits <= 16; ++qubits) {
            uint64_t const states = UINT64_C(1) << qubits;
            std::vector<complex_t> serial(states, {0.0, 0.0});
            std::vector<complex_t> parallel(states, {0.0, 0.0});
            serial[0] = parallel[0] = {1.0, 0.0};

            for (int target = 0; target < qubits; ++target) {
                check_pair_mapping(qubits, target);
                hadamard_serial(serial.data(), states, target);
                hadamard_openmp(parallel.data(), states, target);
            }

            double const expected = 1.0 / std::sqrt(static_cast<double>(states));
            double norm = 0.0;
            for (uint64_t i = 0; i < states; ++i) {
                assert(close(serial[i], parallel[i]));
                assert(std::abs(parallel[i].real - expected) < 1e-12);
                norm += parallel[i].real * parallel[i].real +
                        parallel[i].imag * parallel[i].imag;
            }
            assert(std::abs(norm - 1.0) < 1e-12);

            for (int target = 0; target < qubits; ++target) {
                hadamard_openmp(parallel.data(), states, target);
            }
            assert(close(parallel[0], {1.0, 0.0}));
            for (uint64_t i = 1; i < states; ++i) {
                assert(close(parallel[i], {0.0, 0.0}));
            }
        }
    }
    std::puts("Hadamard tests passed");
}
