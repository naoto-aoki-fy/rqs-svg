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

}  // namespace

int main() {
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
