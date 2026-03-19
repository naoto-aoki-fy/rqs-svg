#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <unistd.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>
#include <unordered_set>
#include <string_view>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <tuple>
#include <type_traits>

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <cuda/std/array>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <nccl.h>
#include <openssl/evp.h>

#include <atlc/format.hpp>
#include <atlc/log2_int.hpp>
#include <atlc/mpi.hpp>
#include <atlc/cuda.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_nccl.hpp>

#include <qcs.hpp>

namespace qcs {

typedef double float_t;
typedef cuda::std::complex<qcs::float_t> complex_t;
typedef cuda::std::array<qcs::float_t, 2> float2_t;

__device__ __host__ complex_t multiply_i(complex_t input) {
    return complex_t(- input.imag(), input.real());
}

__device__ __host__ complex_t multiply_i_m(complex_t input) {
    return complex_t(input.imag(), - input.real());
}

__device__ __host__ complex_t multiply_i_real(complex_t input, float_t multiplier) {
    return complex_t(multiplier * input.imag(), - multiplier * input.real());
}

constexpr uint64_t kernel_input_max_size = 256;

__global__ void initstate_sequential_kernel(qcs::complex_t* const data_global, int proc_num)
{
    uint64_t const num_threads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
    uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    data_global[idx] = idx + num_threads * proc_num;
}

__global__ void initstate_flat_kernel(qcs::complex_t* const data_global)
{
    uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
    data_global[idx] = 1;
}

struct kernel_common_struct {
    int num_qubits;
    qcs::complex_t* state_data_device;
};

__constant__ qcs::complex_t zero_constant;

__constant__ qcs::kernel_common_struct kernel_common_constant;

__constant__ unsigned char kernel_input_constant[qcs::kernel_input_max_size];

struct kernel_input_qnlist_struct {
    int num_target_qubits;
    int num_positive_control_qubits;
    int num_negative_control_qubits;
    uint64_t is_measured_bits;
    uint64_t measured_value_bits;
    int qubit_num_list[1];

    static __host__ __device__ uint64_t needed_size(
        int const num_positive_control_qubits,
        int const num_negative_control_qubits,
        int const num_target_qubits
    ) {
        return
            sizeof(qcs::kernel_input_qnlist_struct)
            - sizeof(qubit_num_list)
            + sizeof(int) * (
                2 * num_positive_control_qubits
                + num_negative_control_qubits
                + 2 * num_target_qubits
            );
    }

    __host__ __device__ uint64_t byte_size() const {
        return needed_size(this->num_positive_control_qubits, this->num_negative_control_qubits, this->num_target_qubits);
    }

    __host__ __device__ int get_num_operand_qubits() const {
        return
            this->num_positive_control_qubits
            + this->num_negative_control_qubits
            + this->num_target_qubits;
    }

    __host__ __device__ int const* get_operand_qubit_num_list_sorted() const {
        return this->qubit_num_list;
    }

    __host__ __device__ int* get_operand_qubit_num_list_sorted() {
        return this->qubit_num_list;
    }

    __host__ __device__ int const* get_positive_control_qubit_num_list() const {
        return this->qubit_num_list + this->get_num_operand_qubits();
    }

    __host__ __device__ int* get_positive_control_qubit_num_list() {
        return this->qubit_num_list + this->get_num_operand_qubits();
    }

    __host__ __device__ int const* get_target_qubit_num_list() const {
        return this->qubit_num_list
            + 2 * this->num_positive_control_qubits
            + this->num_negative_control_qubits
            + this->num_target_qubits;
    }

    __host__ __device__ int* get_target_qubit_num_list() {
        return this->qubit_num_list
            + 2 * this->num_positive_control_qubits
            + this->num_negative_control_qubits
            + this->num_target_qubits;
    }
}; /* kernel_input_qnlist_struct */

static __device__ void thread_num_to_state_index_q0(uint64_t thread_num, uint64_t& index_state) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state = 0;

    int const num_operand_qubits = args->get_num_operand_qubits();
    int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

    // generate index_state
    // ignoring positive control qubits
    uint64_t lower_mask = 0;
    for(int i = 0; i < num_operand_qubits; i++) {
        uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
        uint64_t const upper_mask = mask & ~lower_mask;
        lower_mask = mask;
        index_state |= (thread_num & upper_mask) << i;
    }
    index_state |= (thread_num & ~lower_mask) << num_operand_qubits;

    // update index_state
    // considering positive control qubits
    int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
    for(int i = 0; i < args->num_positive_control_qubits; i++) {
        index_state |= UINT64_C(1) << positive_control_qubit_num_list[i];
    }

} /* thread_num_to_state_index_q0 */

static __device__ void thread_num_to_state_index_q1(uint64_t thread_num, uint64_t& index_state_0, uint64_t& index_state_1, int& is_measured_bits, int& measured_value_bits) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state_0 = 0;

    int const num_operand_qubits = args->get_num_operand_qubits();
    int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

    // generate index_state_0
    // ignoring positive control qubits
    uint64_t lower_mask = 0;
    for(int i = 0; i < num_operand_qubits; i++) {
        uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
        uint64_t const upper_mask = mask & ~lower_mask;
        lower_mask = mask;
        index_state_0 |= (thread_num & upper_mask) << i;
    }
    index_state_0 |= (thread_num & ~lower_mask) << num_operand_qubits;

    // update index_state_0
    // considering positive control qubits
    int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
    for(int i = 0; i < args->num_positive_control_qubits; i++) {
        index_state_0 |= UINT64_C(1) << positive_control_qubit_num_list[i];
    }

    // generate index_state_1
    // num_target_qubits == 1
    auto const target_qubit_num = args->get_target_qubit_num_list()[0];
    index_state_1 = index_state_0 | (UINT64_C(1) << target_qubit_num);

    is_measured_bits = args->is_measured_bits;
    measured_value_bits = args->measured_value_bits;

} /* thread_num_to_state_index_q1 */

static __device__ void thread_num_to_state_index_q2(uint64_t thread_num, uint64_t& index_state_00, uint64_t& index_state_01, uint64_t& index_state_10, uint64_t& index_state_11, int& is_measured_bits, int& measured_value_bits) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state_00 = 0;

    int const num_operand_qubits = args->get_num_operand_qubits();
    int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

    // generate index_state_00
    // ignoring positive control qubits
    uint64_t lower_mask = 0;
    for(int i = 0; i < num_operand_qubits; i++) {
        uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
        uint64_t const upper_mask = mask & ~lower_mask;
        lower_mask = mask;
        index_state_00 |= (thread_num & upper_mask) << i;
    }
    index_state_00 |= (thread_num & ~lower_mask) << num_operand_qubits;

    // update index_state_0
    // considering positive control qubits
    int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
    for(int i = 0; i < args->num_positive_control_qubits; i++) {
        index_state_00 |= UINT64_C(1) << positive_control_qubit_num_list[i];
    }

    // generate index_state_1
    auto target_qubit_num_list = args->get_target_qubit_num_list();
    index_state_01 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_10 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_11 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[0]) | (UINT64_C(1) << target_qubit_num_list[1]);

    is_measured_bits = args->is_measured_bits;
    measured_value_bits = args->measured_value_bits;

} /* thread_num_to_state_index_q2 */

static __device__ void thread_num_to_state_index_q3(
    uint64_t thread_num,
    uint64_t& index_state_000, uint64_t& index_state_001, uint64_t& index_state_010, uint64_t& index_state_011,
    uint64_t& index_state_100, uint64_t& index_state_101, uint64_t& index_state_110, uint64_t& index_state_111,
    int& is_measured_bits, int& measured_value_bits
) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state_000 = 0;

    int const num_operand_qubits = args->get_num_operand_qubits();
    int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

    // generate index_state_000
    // ignoring positive control qubits
    uint64_t lower_mask = 0;
    for(int i = 0; i < num_operand_qubits; i++) {
        uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
        uint64_t const upper_mask = mask & ~lower_mask;
        lower_mask = mask;
        index_state_000 |= (thread_num & upper_mask) << i;
    }
    index_state_000 |= (thread_num & ~lower_mask) << num_operand_qubits;

    // update index_state_000
    // considering positive control qubits
    int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
    for(int i = 0; i < args->num_positive_control_qubits; i++) {
        index_state_000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
    }

    // generate other indices
    auto target_qubit_num_list = args->get_target_qubit_num_list();

    index_state_001 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_010 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_011 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[0]) | (UINT64_C(1) << target_qubit_num_list[1]);

    index_state_100 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]);
    index_state_101 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_110 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_111 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

    is_measured_bits = args->is_measured_bits;
    measured_value_bits = args->measured_value_bits;

} /* thread_num_to_state_index_q3 */

static __device__ void thread_num_to_state_index_q4(
    uint64_t thread_num,
    uint64_t& index_state_0000, uint64_t& index_state_0001, uint64_t& index_state_0010, uint64_t& index_state_0011,
    uint64_t& index_state_0100, uint64_t& index_state_0101, uint64_t& index_state_0110, uint64_t& index_state_0111,
    uint64_t& index_state_1000, uint64_t& index_state_1001, uint64_t& index_state_1010, uint64_t& index_state_1011,
    uint64_t& index_state_1100, uint64_t& index_state_1101, uint64_t& index_state_1110, uint64_t& index_state_1111,
    int& is_measured_bits, int& measured_value_bits
) {
    auto args = (qcs::kernel_input_qnlist_struct const*)(void*)qcs::kernel_input_constant;

    index_state_0000 = 0;

    int const num_operand_qubits = args->get_num_operand_qubits();
    int const* const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

    // generate index_state_0000
    // ignoring positive control qubits
    uint64_t lower_mask = 0;
    for(int i = 0; i < num_operand_qubits; i++) {
        uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
        uint64_t const upper_mask = mask & ~lower_mask;
        lower_mask = mask;
        index_state_0000 |= (thread_num & upper_mask) << i;
    }
    index_state_0000 |= (thread_num & ~lower_mask) << num_operand_qubits;

    // update index_state_0000
    // considering positive control qubits
    int const* const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
    for(int i = 0; i < args->num_positive_control_qubits; i++) {
        index_state_0000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
    }

    // generate other indices
    auto target_qubit_num_list = args->get_target_qubit_num_list();

    index_state_0001 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_0010 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_0011 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

    index_state_0100 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]);
    index_state_0101 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_0110 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_0111 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

    index_state_1000 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]);
    index_state_1001 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_1010 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_1011 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

    index_state_1100 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
    index_state_1101 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
    index_state_1110 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
    index_state_1111 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

    is_measured_bits = args->is_measured_bits;
    measured_value_bits = args->measured_value_bits;

} /* thread_num_to_state_index_q4 */

namespace gate {

    struct u4 {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        qcs::complex_t exp_i_phi;
        qcs::complex_t exp_i_lambda;
        qcs::complex_t exp_i_gamma;
        u4(double theta, double phi, double lambda, double gamma = 0.0) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
            exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
            exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
            exp_i_gamma = qcs::complex_t(cos(gamma), sin(gamma));
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            qcs::complex_t const s0_in_copy = s0_in;
            qcs::complex_t const s1_in_copy = s1_in;
            s0_out = exp_i_gamma * (cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy);
            s1_out = exp_i_gamma * (exp_i_phi * (sin_theta_2 * s0_in_copy - exp_i_lambda * cos_theta_2 * s1_in_copy));
        }
    };

    struct u3 {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        qcs::complex_t exp_i_phi;
        qcs::complex_t exp_i_lambda;
        qcs::complex_t exp_i_phi_plus_lambda;
        u3(double theta, double phi, double lambda) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
            exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
            exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
            qcs::float_t const phi_plus_lambda = phi + lambda;
            exp_i_phi_plus_lambda = qcs::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            qcs::complex_t const s0_in_copy = s0_in;
            qcs::complex_t const s1_in_copy = s1_in;
            s0_out = cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy;
            s1_out = exp_i_phi * sin_theta_2 * s0_in_copy + exp_i_phi_plus_lambda * cos_theta_2 * s1_in_copy;
        }
    };

    struct u2 {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::complex_t exp_i_phi;
        qcs::complex_t exp_i_lambda;
        qcs::complex_t exp_i_phi_plus_lambda;
        u2(double phi, double lambda) {
            exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
            exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
            qcs::float_t const phi_plus_lambda = phi + lambda;
            exp_i_phi_plus_lambda = qcs::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            qcs::complex_t const s0_in_copy = s0_in;
            qcs::complex_t const s1_in_copy = s1_in;
            s0_out = M_SQRT1_2 * (s0_in_copy - exp_i_lambda * s1_in_copy);
            s1_out = M_SQRT1_2 * (exp_i_phi * s0_in_copy + exp_i_phi_plus_lambda * s1_in_copy);
        }
    };

    struct u1 {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::complex_t exp_i_lambda;
        u1(double lambda) {
            exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s0_out = s0_in;
            s1_out = exp_i_lambda * s1_in;
        }
    };

    typedef u3 u;
    typedef u1 p;

    struct global_phase {
        static constexpr unsigned int num_target_qubits = 0;
        qcs::complex_t exp_i_theta;
        global_phase(double theta) {
            exp_i_theta.real(cos(theta));
            exp_i_theta.imag(sin(theta));
        }
        __device__ void apply(qcs::complex_t const& s_in, qcs::complex_t& s_out) const {
            s_out = exp_i_theta * s_in;
        }
    };

    struct hadamard {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            // break unitarity intentionally: a hack to prevent amplitude attenuation
            s0_out = s0_in_copy + s1_in_copy;
            s1_out = s0_in_copy - s1_in_copy;
        }
    };

    struct identity {
        static constexpr unsigned int num_target_qubits = 0;
        __device__ void apply() const {
        }
    };

    struct x {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out = s1_in_copy;
            s1_out = s0_in_copy;
        }
    };

    struct y {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out.real(s1_in_copy.imag());
            s0_out.imag(- s1_in_copy.real());
            s1_out.real(- s0_in_copy.imag());
            s1_out.imag(s0_in_copy.real());
        }
    };

    struct z {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out = - s0_in_copy;
            s1_out = - s1_in_copy;
        }
    };

    struct s {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s1_in_copy = s1_in;
            s1_out.real(- s1_in_copy.imag());
            s1_out.imag(s1_in_copy.real());
        }
    };

    struct sdg {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s1_in_copy = s1_in;
            s1_out.real(s1_in_copy.imag());
            s1_out.imag(- s1_in_copy.real());
        }
    };

    struct t {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s1_out = qcs::complex_t(M_SQRT1_2, M_SQRT1_2) * s1_in;
        }
    };

    struct tdg {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s1_out = qcs::complex_t(M_SQRT1_2, - M_SQRT1_2) * s1_in;
        }
    };

    struct sx {
        static constexpr unsigned int num_target_qubits = 1;
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            // break unitarity intentionally: a hack to prevent amplitude attenuation
            // auto const s0_in_copy = s0_in;
            // auto const s1_in_copy = s1_in;
            // s0_out = qcs::complex_t(M_SQRT1_2, M_SQRT1_2) * s0_in_copy + qcs::complex_t(M_SQRT1_2, - M_SQRT1_2) * s1_in_copy;
            // s1_out = qcs::complex_t(M_SQRT1_2, - M_SQRT1_2) * s0_in_copy - qcs::complex_t(M_SQRT1_2, M_SQRT1_2) * s1_in_copy;

            // break unitarity intentionally: a hack to prevent amplitude attenuation
            auto const a = M_SQRT1_2 * (s0_in + s1_in);
            auto const b = M_SQRT1_2 * multiply_i(s0_in - s1_in);
            s0_out = a + b;
            s1_out = a - b;
        }
    };

    struct sxdg {
        static constexpr unsigned int num_target_qubits = 1;

        __device__ void apply(qcs::complex_t const& s0_in,
                            qcs::complex_t const& s1_in,
                            qcs::complex_t& s0_out,
                            qcs::complex_t& s1_out) const {
            // break unitarity intentionally: a hack to prevent amplitude attenuation
            auto const a = M_SQRT1_2 * (s0_in + s1_in);
            auto const b = -M_SQRT1_2 * multiply_i(s0_in - s1_in);
            s0_out = a + b;
            s1_out = a - b;
        }
    };

    struct rx {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        rx(double theta) {
            cos_theta_2 = cos(0.5 * theta);
            cos_theta_2 = sin(0.5 * theta);
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out = cos_theta_2 * s0_in_copy + multiply_i_real(s1_in_copy, - sin_theta_2);
            s1_out = multiply_i_real(s0_in_copy, - sin_theta_2) + cos_theta_2 * s1_in_copy;
        }
    };

    struct ry {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        ry(double theta) {
            cos_theta_2 = cos(0.5 * theta);
            cos_theta_2 = sin(0.5 * theta);
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out = cos_theta_2 * s0_in_copy - sin_theta_2 * s1_in_copy;
            s1_out = sin_theta_2 * s0_in_copy + cos_theta_2 * s1_in_copy;
        }
    };

    struct rz {
        static constexpr unsigned int num_target_qubits = 1;
        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        rz(double theta) {
            cos_theta_2 = cos(0.5 * theta);
            cos_theta_2 = sin(0.5 * theta);
        }
        __device__ void apply(qcs::complex_t const& s0_in, qcs::complex_t const& s1_in, qcs::complex_t& s0_out, qcs::complex_t& s1_out) const {
            s0_out = qcs::complex_t(cos_theta_2, - sin_theta_2) * s0_in;
            s1_out = qcs::complex_t(cos_theta_2, sin_theta_2) * s1_in;
        }
    };

    struct swap {
        static constexpr unsigned int num_target_qubits = 2;
        __device__ void apply(qcs::complex_t const& s00_in, qcs::complex_t const& s01_in, qcs::complex_t const& s10_in, qcs::complex_t const& s11_in, qcs::complex_t& s00_out, qcs::complex_t& s01_out, qcs::complex_t& s10_out, qcs::complex_t& s11_out) const {
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;
            // s00_out = s00_in;
            s01_out = s10_in_copy;
            s10_out = s01_in_copy;
            // s11_out = s11_in;
        }
    };

    struct iswap {
        static constexpr unsigned int num_target_qubits = 2;
        __device__ void apply(qcs::complex_t const& s00_in, qcs::complex_t const& s01_in, qcs::complex_t const& s10_in, qcs::complex_t const& s11_in, qcs::complex_t& s00_out, qcs::complex_t& s01_out, qcs::complex_t& s10_out, qcs::complex_t& s11_out) const {
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;
            // s00_out = s00_in;
            s01_out.real(- s10_in_copy.imag());
            s01_out.imag(s10_in_copy.real());
            s10_out.real(- s01_in_copy.imag());
            s10_out.imag(s01_in_copy.real());
            // s11_out = s11_in;
        }
    };

    struct id {
        static constexpr unsigned int num_target_qubits = 1;

        __device__ void apply(qcs::complex_t const& s0_in,
                            qcs::complex_t const& s1_in,
                            qcs::complex_t& s0_out,
                            qcs::complex_t& s1_out) const {
            // s0_out = s0_in;
            // s1_out = s1_in;
        }
    };

    struct r {
        static constexpr unsigned int num_target_qubits = 1;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        qcs::complex_t exp_i_phi;
        qcs::complex_t exp_minus_i_phi;

        r(double theta, double phi) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
            exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
            exp_minus_i_phi = qcs::complex_t(cos(phi), -sin(phi));
        }

        __device__ void apply(qcs::complex_t const& s0_in,
                            qcs::complex_t const& s1_in,
                            qcs::complex_t& s0_out,
                            qcs::complex_t& s1_out) const {
            auto const s0_in_copy = s0_in;
            auto const s1_in_copy = s1_in;
            s0_out = cos_theta_2 * s0_in_copy + sin_theta_2 * multiply_i_real(exp_minus_i_phi * s1_in_copy, 1.0);
            s1_out = multiply_i_m(exp_i_phi * s0_in_copy) * sin_theta_2 + cos_theta_2 * s1_in_copy;
        }
    };

    struct rzz {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::complex_t phase_minus;
        qcs::complex_t phase_plus;

        rzz(double theta) {
            qcs::float_t const theta_2 = 0.5 * theta;
            phase_minus = qcs::complex_t(cos(theta_2), -sin(theta_2));
            phase_plus = qcs::complex_t(cos(theta_2),  sin(theta_2));
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            s00_out = phase_minus * s00_in;
            s01_out = phase_plus * s01_in;
            s10_out = phase_plus * s10_in;
            s11_out = phase_minus * s11_in;
        }
    };

    struct rxx {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;

        rxx(double theta) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            auto const s00_in_copy = s00_in;
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;
            auto const s11_in_copy = s11_in;

            s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, sin_theta_2);
            s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, sin_theta_2);
            s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, sin_theta_2);
            s11_out = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, sin_theta_2);
        }
    };

    struct ryy {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;

        ryy(double theta) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            auto const s00_in_copy = s00_in;
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;
            auto const s11_in_copy = s11_in;

            s00_out = cos_theta_2 * s00_in_copy + multiply_i(s11_in_copy) * sin_theta_2;
            s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, sin_theta_2);
            s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, sin_theta_2);
            s11_out = cos_theta_2 * s11_in_copy + multiply_i(s00_in_copy) * sin_theta_2;
        }
    };

    struct rzx {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;

        rzx(double theta) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            auto const s00_in_copy = s00_in;
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;
            auto const s11_in_copy = s11_in;

            s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(s10_in_copy, sin_theta_2);
            s01_out = cos_theta_2 * s01_in_copy + multiply_i(s11_in_copy) * sin_theta_2;
            s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s00_in_copy, sin_theta_2);
            s11_out = cos_theta_2 * s11_in_copy + multiply_i(s01_in_copy) * sin_theta_2;
        }
    };

    struct xx_plus_yy {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        qcs::complex_t exp_i_beta;
        qcs::complex_t exp_minus_i_beta;

        xx_plus_yy(double theta, double beta = 0.0) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
            exp_i_beta = qcs::complex_t(cos(beta), sin(beta));
            exp_minus_i_beta = qcs::complex_t(cos(beta), -sin(beta));
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            auto const s01_in_copy = s01_in;
            auto const s10_in_copy = s10_in;

            // s00_out = s00_in;
            s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(exp_i_beta * s10_in_copy, sin_theta_2);
            s10_out = multiply_i_m(exp_minus_i_beta * s01_in_copy) * sin_theta_2 + cos_theta_2 * s10_in_copy;
            // s11_out = s11_in;
        }
    };

    struct xx_minus_yy {
        static constexpr unsigned int num_target_qubits = 2;

        qcs::float_t cos_theta_2;
        qcs::float_t sin_theta_2;
        qcs::complex_t exp_i_beta;
        qcs::complex_t exp_minus_i_beta;

        xx_minus_yy(double theta, double beta = 0.0) {
            qcs::float_t const theta_2 = 0.5 * theta;
            cos_theta_2 = cos(theta_2);
            sin_theta_2 = sin(theta_2);
            exp_i_beta = qcs::complex_t(cos(beta), sin(beta));
            exp_minus_i_beta = qcs::complex_t(cos(beta), -sin(beta));
        }

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            auto const s00_in_copy = s00_in;
            auto const s11_in_copy = s11_in;

            s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(exp_i_beta * s11_in_copy, sin_theta_2);
            // s01_out = s01_in;
            // s10_out = s10_in;
            s11_out = multiply_i_m(exp_minus_i_beta * s00_in_copy) * sin_theta_2 + cos_theta_2 * s11_in_copy;
        }
    };

    struct dcx {
        static constexpr unsigned int num_target_qubits = 2;

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            // s00_out = s00_in;
            s01_out = s10_in;
            s10_out = s11_in;
            s11_out = s01_in;
        }
    };

    struct ecr {
        static constexpr unsigned int num_target_qubits = 2;

        __device__ void apply(qcs::complex_t const& s00_in,
                            qcs::complex_t const& s01_in,
                            qcs::complex_t const& s10_in,
                            qcs::complex_t const& s11_in,
                            qcs::complex_t& s00_out,
                            qcs::complex_t& s01_out,
                            qcs::complex_t& s10_out,
                            qcs::complex_t& s11_out) const {
            s00_out = M_SQRT1_2 * s01_in + multiply_i(s11_in) * M_SQRT1_2;
            s01_out = M_SQRT1_2 * s00_in + multiply_i_real(s10_in, M_SQRT1_2);
            s10_out = multiply_i(s01_in) * M_SQRT1_2 + M_SQRT1_2 * s11_in;
            s11_out = multiply_i_real(s00_in, M_SQRT1_2) + M_SQRT1_2 * s10_in;
        }
    };

    struct rccx {
        static constexpr unsigned int num_target_qubits = 3;

        __device__ void apply(qcs::complex_t const& s000_in,
                            qcs::complex_t const& s001_in,
                            qcs::complex_t const& s010_in,
                            qcs::complex_t const& s011_in,
                            qcs::complex_t const& s100_in,
                            qcs::complex_t const& s101_in,
                            qcs::complex_t const& s110_in,
                            qcs::complex_t const& s111_in,
                            qcs::complex_t& s000_out,
                            qcs::complex_t& s001_out,
                            qcs::complex_t& s010_out,
                            qcs::complex_t& s011_out,
                            qcs::complex_t& s100_out,
                            qcs::complex_t& s101_out,
                            qcs::complex_t& s110_out,
                            qcs::complex_t& s111_out) const {
            // s000_out = s000_in;
            // s001_out = s001_in;
            // s010_out = s010_in;
            s011_out = multiply_i_m(s111_in);
            // s100_out = s100_in;
            s101_out = -s101_in;
            // s110_out = s110_in;
            s111_out = multiply_i(s011_in);
        }
    };

    struct rcccx {
        static constexpr unsigned int num_target_qubits = 4;

        __device__ void apply(qcs::complex_t const& s0000_in,
                            qcs::complex_t const& s0001_in,
                            qcs::complex_t const& s0010_in,
                            qcs::complex_t const& s0011_in,
                            qcs::complex_t const& s0100_in,
                            qcs::complex_t const& s0101_in,
                            qcs::complex_t const& s0110_in,
                            qcs::complex_t const& s0111_in,
                            qcs::complex_t const& s1000_in,
                            qcs::complex_t const& s1001_in,
                            qcs::complex_t const& s1010_in,
                            qcs::complex_t const& s1011_in,
                            qcs::complex_t const& s1100_in,
                            qcs::complex_t const& s1101_in,
                            qcs::complex_t const& s1110_in,
                            qcs::complex_t const& s1111_in,
                            qcs::complex_t& s0000_out,
                            qcs::complex_t& s0001_out,
                            qcs::complex_t& s0010_out,
                            qcs::complex_t& s0011_out,
                            qcs::complex_t& s0100_out,
                            qcs::complex_t& s0101_out,
                            qcs::complex_t& s0110_out,
                            qcs::complex_t& s0111_out,
                            qcs::complex_t& s1000_out,
                            qcs::complex_t& s1001_out,
                            qcs::complex_t& s1010_out,
                            qcs::complex_t& s1011_out,
                            qcs::complex_t& s1100_out,
                            qcs::complex_t& s1101_out,
                            qcs::complex_t& s1110_out,
                            qcs::complex_t& s1111_out) const {
            // s0000_out = s0000_in;
            // s0001_out = s0001_in;
            // s0010_out = s0010_in;
            s0011_out = multiply_i(s0011_in);
            // s0100_out = s0100_in;
            // s0101_out = s0101_in;
            // s0110_out = s0110_in;
            s0111_out = s1111_in;
            // s1000_out = s1000_in;
            // s1001_out = s1001_in;
            // s1010_out = s1010_in;
            s1011_out = multiply_i_m(s1011_in);
            // s1100_out = s1100_in;
            // s1101_out = s1101_in;
            // s1110_out = s1110_in;
            s1111_out = -s0111_in;
        }
    };

}

__global__ void clear_alternative_state(int target_qubit_num, int measured_value) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    // generate index_state_0
    uint64_t lower_mask = 0;
    uint64_t const mask = (UINT64_C(1) << (target_qubit_num)) - 1;
    uint64_t const upper_mask = mask & ~lower_mask;
    lower_mask = mask;
    uint64_t index_state = (thread_num & upper_mask) | ((thread_num & ~lower_mask) << 1);

    if (!measured_value) {
        index_state = index_state | (UINT64_C(1) << target_qubit_num);
    }

    qcs::kernel_common_constant.state_data_device[index_state] = 0.0;

}

template<typename GateType>
__global__
typename std::enable_if<GateType::num_target_qubits==0>::type
cuda_gate(GateType const gateobj) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state;
    thread_num_to_state_index_q0(thread_num, index_state);

    qcs::complex_t const* ps_in = &qcs::kernel_common_constant.state_data_device[index_state];
    gateobj.apply(*ps_in, qcs::kernel_common_constant.state_data_device[index_state]);
}

template<typename GateType>
__global__
typename std::enable_if<GateType::num_target_qubits==1>::type
cuda_gate(GateType const gateobj) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state_0, index_state_1;
    int is_measured_bits, measured_value_bits;
    thread_num_to_state_index_q1(thread_num, index_state_0, index_state_1, is_measured_bits, measured_value_bits);

    qcs::complex_t const* ps0_in;
    qcs::complex_t const* ps1_in;

    if (
        (!(is_measured_bits) || ((is_measured_bits) && (measured_value_bits)==0))
    ) {
        ps0_in = &qcs::kernel_common_constant.state_data_device[index_state_0];
    } else {
        ps0_in = &zero_constant;
    }

    if (
        (!(is_measured_bits) || ((is_measured_bits) && (measured_value_bits)==1))
    ) {
        ps1_in = &qcs::kernel_common_constant.state_data_device[index_state_1];
    } else {
        ps1_in = &zero_constant;
    }

    gateobj.apply(*ps0_in, *ps1_in, qcs::kernel_common_constant.state_data_device[index_state_0], qcs::kernel_common_constant.state_data_device[index_state_1]);
}

template<typename GateType>
__global__
typename std::enable_if<GateType::num_target_qubits==2>::type
cuda_gate(GateType const gateobj) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state_00, index_state_01, index_state_10, index_state_11;
    int is_measured_bits, measured_value_bits;
    thread_num_to_state_index_q2(thread_num, index_state_00, index_state_01, index_state_10, index_state_11, is_measured_bits, measured_value_bits);

    // qcs::complex_t s00_in, s01_in, s10_in, s11_in;
    qcs::complex_t const* ps00_in;
    qcs::complex_t const* ps01_in;
    qcs::complex_t const* ps10_in;
    qcs::complex_t const* ps11_in;

    if (
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps00_in = &qcs::kernel_common_constant.state_data_device[index_state_00];
    } else {
        ps00_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps01_in = &qcs::kernel_common_constant.state_data_device[index_state_01];
    } else {
        ps01_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps10_in = &qcs::kernel_common_constant.state_data_device[index_state_10];
    } else {
        ps10_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps11_in = &qcs::kernel_common_constant.state_data_device[index_state_11];
    } else {
        ps11_in = &zero_constant;
    }

    gateobj.apply(*ps00_in, *ps01_in, *ps10_in, *ps11_in, qcs::kernel_common_constant.state_data_device[index_state_00], qcs::kernel_common_constant.state_data_device[index_state_01], qcs::kernel_common_constant.state_data_device[index_state_10], qcs::kernel_common_constant.state_data_device[index_state_11]);
}

template<typename GateType>
__global__
typename std::enable_if<GateType::num_target_qubits==3>::type
cuda_gate(GateType const gateobj) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state_000, index_state_001, index_state_010, index_state_011;
    uint64_t index_state_100, index_state_101, index_state_110, index_state_111;
    int is_measured_bits, measured_value_bits;
    thread_num_to_state_index_q3(
        thread_num,
        index_state_000, index_state_001, index_state_010, index_state_011,
        index_state_100, index_state_101, index_state_110, index_state_111,
        is_measured_bits, measured_value_bits
    );

    qcs::complex_t const* ps000_in;
    qcs::complex_t const* ps001_in;
    qcs::complex_t const* ps010_in;
    qcs::complex_t const* ps011_in;
    qcs::complex_t const* ps100_in;
    qcs::complex_t const* ps101_in;
    qcs::complex_t const* ps110_in;
    qcs::complex_t const* ps111_in;

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps000_in = &qcs::kernel_common_constant.state_data_device[index_state_000];
    } else {
        ps000_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps001_in = &qcs::kernel_common_constant.state_data_device[index_state_001];
    } else {
        ps001_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps010_in = &qcs::kernel_common_constant.state_data_device[index_state_010];
    } else {
        ps010_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps011_in = &qcs::kernel_common_constant.state_data_device[index_state_011];
    } else {
        ps011_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps100_in = &qcs::kernel_common_constant.state_data_device[index_state_100];
    } else {
        ps100_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps101_in = &qcs::kernel_common_constant.state_data_device[index_state_101];
    } else {
        ps101_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps110_in = &qcs::kernel_common_constant.state_data_device[index_state_110];
    } else {
        ps110_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps111_in = &qcs::kernel_common_constant.state_data_device[index_state_111];
    } else {
        ps111_in = &zero_constant;
    }

    gateobj.apply(
        *ps000_in, *ps001_in, *ps010_in, *ps011_in,
        *ps100_in, *ps101_in, *ps110_in, *ps111_in,
        qcs::kernel_common_constant.state_data_device[index_state_000],
        qcs::kernel_common_constant.state_data_device[index_state_001],
        qcs::kernel_common_constant.state_data_device[index_state_010],
        qcs::kernel_common_constant.state_data_device[index_state_011],
        qcs::kernel_common_constant.state_data_device[index_state_100],
        qcs::kernel_common_constant.state_data_device[index_state_101],
        qcs::kernel_common_constant.state_data_device[index_state_110],
        qcs::kernel_common_constant.state_data_device[index_state_111]
    );
}

template<typename GateType>
__global__
typename std::enable_if<GateType::num_target_qubits==4>::type
cuda_gate(GateType const gateobj) {
    int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

    uint64_t index_state_0000, index_state_0001, index_state_0010, index_state_0011;
    uint64_t index_state_0100, index_state_0101, index_state_0110, index_state_0111;
    uint64_t index_state_1000, index_state_1001, index_state_1010, index_state_1011;
    uint64_t index_state_1100, index_state_1101, index_state_1110, index_state_1111;
    int is_measured_bits, measured_value_bits;
    thread_num_to_state_index_q4(
        thread_num,
        index_state_0000, index_state_0001, index_state_0010, index_state_0011,
        index_state_0100, index_state_0101, index_state_0110, index_state_0111,
        index_state_1000, index_state_1001, index_state_1010, index_state_1011,
        index_state_1100, index_state_1101, index_state_1110, index_state_1111,
        is_measured_bits, measured_value_bits
    );

    qcs::complex_t const* ps0000_in;
    qcs::complex_t const* ps0001_in;
    qcs::complex_t const* ps0010_in;
    qcs::complex_t const* ps0011_in;
    qcs::complex_t const* ps0100_in;
    qcs::complex_t const* ps0101_in;
    qcs::complex_t const* ps0110_in;
    qcs::complex_t const* ps0111_in;
    qcs::complex_t const* ps1000_in;
    qcs::complex_t const* ps1001_in;
    qcs::complex_t const* ps1010_in;
    qcs::complex_t const* ps1011_in;
    qcs::complex_t const* ps1100_in;
    qcs::complex_t const* ps1101_in;
    qcs::complex_t const* ps1110_in;
    qcs::complex_t const* ps1111_in;

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps0000_in = &qcs::kernel_common_constant.state_data_device[index_state_0000];
    } else {
        ps0000_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps0001_in = &qcs::kernel_common_constant.state_data_device[index_state_0001];
    } else {
        ps0001_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps0010_in = &qcs::kernel_common_constant.state_data_device[index_state_0010];
    } else {
        ps0010_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps0011_in = &qcs::kernel_common_constant.state_data_device[index_state_0011];
    } else {
        ps0011_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps0100_in = &qcs::kernel_common_constant.state_data_device[index_state_0100];
    } else {
        ps0100_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps0101_in = &qcs::kernel_common_constant.state_data_device[index_state_0101];
    } else {
        ps0101_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps0110_in = &qcs::kernel_common_constant.state_data_device[index_state_0110];
    } else {
        ps0110_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==0)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps0111_in = &qcs::kernel_common_constant.state_data_device[index_state_0111];
    } else {
        ps0111_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps1000_in = &qcs::kernel_common_constant.state_data_device[index_state_1000];
    } else {
        ps1000_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps1001_in = &qcs::kernel_common_constant.state_data_device[index_state_1001];
    } else {
        ps1001_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps1010_in = &qcs::kernel_common_constant.state_data_device[index_state_1010];
    } else {
        ps1010_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==0)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps1011_in = &qcs::kernel_common_constant.state_data_device[index_state_1011];
    } else {
        ps1011_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps1100_in = &qcs::kernel_common_constant.state_data_device[index_state_1100];
    } else {
        ps1100_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==0)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps1101_in = &qcs::kernel_common_constant.state_data_device[index_state_1101];
    } else {
        ps1101_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==0))
    ) {
        ps1110_in = &qcs::kernel_common_constant.state_data_device[index_state_1110];
    } else {
        ps1110_in = &zero_constant;
    }

    if (
        (!(is_measured_bits&8) || ((is_measured_bits&8) && (measured_value_bits&8)==1)) &&
        (!(is_measured_bits&4) || ((is_measured_bits&4) && (measured_value_bits&4)==1)) &&
        (!(is_measured_bits&2) || ((is_measured_bits&2) && (measured_value_bits&2)==1)) &&
        (!(is_measured_bits&1) || ((is_measured_bits&1) && (measured_value_bits&1)==1))
    ) {
        ps1111_in = &qcs::kernel_common_constant.state_data_device[index_state_1111];
    } else {
        ps1111_in = &zero_constant;
    }

    gateobj.apply(
        *ps0000_in, *ps0001_in, *ps0010_in, *ps0011_in,
        *ps0100_in, *ps0101_in, *ps0110_in, *ps0111_in,
        *ps1000_in, *ps1001_in, *ps1010_in, *ps1011_in,
        *ps1100_in, *ps1101_in, *ps1110_in, *ps1111_in,
        qcs::kernel_common_constant.state_data_device[index_state_0000],
        qcs::kernel_common_constant.state_data_device[index_state_0001],
        qcs::kernel_common_constant.state_data_device[index_state_0010],
        qcs::kernel_common_constant.state_data_device[index_state_0011],
        qcs::kernel_common_constant.state_data_device[index_state_0100],
        qcs::kernel_common_constant.state_data_device[index_state_0101],
        qcs::kernel_common_constant.state_data_device[index_state_0110],
        qcs::kernel_common_constant.state_data_device[index_state_0111],
        qcs::kernel_common_constant.state_data_device[index_state_1000],
        qcs::kernel_common_constant.state_data_device[index_state_1001],
        qcs::kernel_common_constant.state_data_device[index_state_1010],
        qcs::kernel_common_constant.state_data_device[index_state_1011],
        qcs::kernel_common_constant.state_data_device[index_state_1100],
        qcs::kernel_common_constant.state_data_device[index_state_1101],
        qcs::kernel_common_constant.state_data_device[index_state_1110],
        qcs::kernel_common_constant.state_data_device[index_state_1111]
    );
}

namespace cubUtility {

    struct float2Add {
        __device__ qcs::float2_t operator()(const qcs::float2_t& a, const qcs::float2_t& b) const {
            return {a[0] + b[0], a[1] + b[1]};
        }
    };

    struct IndirectLoad
    {
        __device__ qcs::float2_t operator()(uint64_t thread_num) const
        {
            uint64_t index_state_0, index_state_1;
            int is_measured_bits, measured_value_bits;
            thread_num_to_state_index_q1(thread_num, index_state_0, index_state_1, is_measured_bits, measured_value_bits);

            // since target_qubit must be unmeasured, branching is not necessary.
            return qcs::float2_t{
                cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]),
                cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1])
            };

            // return qcs::float2_t{
            //     (measured_state != 1)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]): 0,
            //     (measured_state != 0)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1]): 0
            // };

        }
    };

} /* cubUtility */

enum class initstate_enum {
    sequential,
    flat,
    zero,
    entangled,
    use_curand,
    laod_statevector,
};

struct simulator_core {

/* begin simulator variables */

int num_qubits;

bool use_unified_memory;

initstate_enum initstate_choice;

float elapsed_ms, elapsed_ms_2;

int num_procs, proc_num;

int num_rand_areas;

std::string my_hostname;
int my_node_number;
int my_node_local_rank;
int node_count;

int gpu_id;

ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
int nccl_rank;

std::vector<int> perm_p2l;
std::vector<int> perm_l2p;

int num_samples;
unsigned int rng_seed;
std::mt19937_64 engine;
int num_rand_areas_times_num_procs;

int log_num_procs;
int log_block_size_max;
int block_size_max;
int target_qubit_num_begin;
int target_qubit_num_end;

cudaStream_t stream;
cudaEvent_t event_1;
cudaEvent_t event_2;

uint64_t num_states;
int num_qubits_local;
uint64_t num_states_local;

uint64_t num_blocks_gateop;
uint64_t block_size_gateop;
uint64_t num_operand_qubits;

qcs::complex_t* state_data_device;

qcs::complex_t* zero_constant_addr;
qcs::kernel_common_struct* qcs_kernel_common_constant_addr;
qcs::kernel_common_struct qcs_kernel_common_host;
qcs::kernel_input_qnlist_struct* qcs_kernel_input_constant_addr;
std::vector<char> qcs_kernel_input_host_buffer;
int log_swap_buffer_total_length;
uint64_t swap_buffer_total_length;
qcs::complex_t* swap_buffer;
qcs::float2_t* measure_norm_device;

void* cub_temp_buffer_device;
uint64_t cub_temp_buffer_device_size;

std::vector<int> operand_qubit_num_list;
std::vector<int> target_qubit_num_physical_list;
std::vector<int> swap_target_global_list;
std::vector<int> swap_target_local_list;
std::vector<int> swap_target_local_logical_list;
std::vector<int> swap_target_global_logical_list;
std::vector<int> positive_control_qubit_num_physical_list;
std::vector<int> positive_control_qubit_num_physical_global_list;
std::vector<int> positive_control_qubit_num_physical_local_list;
std::vector<int> negative_control_qubit_num_physical_list;
std::vector<int> negative_control_qubit_num_physical_global_list;
std::vector<int> negative_control_qubit_num_physical_local_list;

std::vector<int> measured_1_qubit_num_logical_list;
std::vector<int> measured_0_qubit_num_logical_list;
std::vector<int> measured_1_qubit_num_logical_list_copy;
std::vector<int> measured_0_qubit_num_logical_list_copy;

std::vector<int> target_qubit_num_logical_list;
std::vector<int> positive_control_qubit_num_logical_list;
std::vector<int> negative_control_qubit_num_logical_list;

bool measured_control_condition;
bool proc_num_control_condition;

/* end simulator variables */

simulator_core() :
num_qubits(0),
qubit_allocated_var(false),
use_unified_memory(false),
num_rand_areas_times_num_procs(8)
{
}

void setup() {

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (num_rand_areas_times_num_procs < num_procs) {
        throw std::runtime_error(atlc::format("num_rand_areas_times_num_procs %d < num_procs %d", num_rand_areas_times_num_procs, num_procs));
    }
    num_rand_areas = num_rand_areas_times_num_procs / num_procs;

    atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
    // fprintf(stderr, "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n", proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    gpu_id = my_node_local_rank;
    ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

    if (proc_num == 0) {
        ATLC_CHECK_NCCL(ncclGetUniqueId, &nccl_id);
    }

    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    nccl_rank = proc_num;
    ATLC_CHECK_NCCL(ncclCommInitRank, &nccl_comm, num_procs, nccl_id, nccl_rank);

    log_num_procs = atlc::log2_int(num_procs);

    log_block_size_max = 9;

    ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);

    block_size_max = 1 << log_block_size_max;

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_common_constant_addr, qcs::kernel_common_constant);

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&qcs_kernel_input_constant_addr, qcs::kernel_input_constant);

    ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void**)&zero_constant_addr, qcs::zero_constant);

    qcs::complex_t const complex_zero = 0;
    ATLC_CHECK_CUDA(cudaMemcpyAsync, zero_constant_addr, &complex_zero, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);

    if (proc_num==0) {
        std::random_device rng;
        this->rng_seed = rng(); // & ((UINT64_C(1)<<12)-1);
    }
    ATLC_CHECK_MPI(MPI_Bcast, &this->rng_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    engine = std::mt19937_64(rng_seed);

    cub_temp_buffer_device = NULL;
    state_data_device = NULL;
    swap_buffer = NULL;

} /* setup */

bool qubit_allocated_var;
bool is_qubit_allocated() const {
    return qubit_allocated_var;
}

void allocate_memory(int num_qubits) {

    std::vector<std::string> exmes_list;
    if (this->num_qubits > 0) {
        exmes_list.push_back(atlc::format("num_qubits is already set %d > 0", this->num_qubits));
    }
    if (cub_temp_buffer_device) {
        exmes_list.push_back(atlc::format("cub_temp_buffer_device is %p not NULL", cub_temp_buffer_device));
    }
    if (state_data_device) {
        exmes_list.push_back(atlc::format("state_data_device is %p not NULL", state_data_device));
    }
    if (swap_buffer) {
        exmes_list.push_back(atlc::format("swap_buffer is %p not NULL", swap_buffer));
    }
    if (exmes_list.size()>0) {
        std::string exmes_all = std::move(exmes_list[0]);
        for (int i = 1; i < exmes_list.size(); i++) {
            exmes_all += "\n" + exmes_list[i];
        }
        throw std::runtime_error(exmes_all);
    }

    this->num_qubits = num_qubits;

    perm_p2l.resize(num_qubits);
    perm_l2p.resize(num_qubits);

    for(int qubit_num=0; qubit_num<num_qubits; qubit_num++) {
        perm_p2l[qubit_num] = qubit_num;
        perm_l2p[qubit_num] = qubit_num;
    }

    num_states = UINT64_C(1) << num_qubits;

    num_qubits_local = num_qubits - log_num_procs;

    num_states_local = UINT64_C(1) << num_qubits_local;

    if (use_unified_memory) {
        ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
        ATLC_CHECK_CUDA(cudaMemAdvise, state_data_device, num_states_local * sizeof(*state_data_device), cudaMemAdviseSetPreferredLocation, gpu_id);
    } else {
        ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
    }
    initialize_zero();

    qcs_kernel_common_host.num_qubits = num_qubits;
    qcs_kernel_common_host.state_data_device = state_data_device;
    ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_common_constant_addr, &qcs_kernel_common_host, sizeof(qcs::kernel_common_struct), cudaMemcpyHostToDevice, stream);

    log_swap_buffer_total_length = (num_qubits_local>30)? num_qubits_local - 3 : num_qubits_local;
    swap_buffer_total_length = UINT64_C(1) << log_swap_buffer_total_length;
    ATLC_CHECK_CUDA(cudaMallocAsync, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t), stream);

    ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, 1, stream);
    cub_temp_buffer_device_size = 1;

    ATLC_CHECK_CUDA(cudaMallocAsync, &measure_norm_device, sizeof(qcs::complex_t), stream);

    qubit_allocated_var = true;

}

void free_memory() {
    if (cub_temp_buffer_device) {
        ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
        cub_temp_buffer_device = NULL;
    }
    if (state_data_device) {
        ATLC_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);
        state_data_device = NULL;
    }
    if (swap_buffer) {
        ATLC_CHECK_CUDA(cudaFreeAsync, swap_buffer, stream);
        swap_buffer = NULL;
    }
    if (measure_norm_device) {
        ATLC_CHECK_CUDA(cudaFreeAsync, measure_norm_device, stream);
        measure_norm_device = NULL;
    }
    this->num_qubits = 0;
    qubit_allocated_var = false;
}

void initialize_sequential() {
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();
    uint64_t num_blocks_init;
    uint64_t block_size_init;
    if (num_qubits_local >= log_block_size_max) {
        num_blocks_init = num_states_local >> log_block_size_max;
        block_size_init = block_size_max;
    } else {
        num_blocks_init = 1;
        block_size_init = num_states_local;
    }

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_sequential_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device, proc_num);
} /* initialize_sequential */

void initialize_flat() {
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();
    uint64_t num_blocks_init;
    uint64_t block_size_init;
    if (num_qubits_local >= log_block_size_max) {
        num_blocks_init = num_states_local >> log_block_size_max;
        block_size_init = block_size_max;
    } else {
        num_blocks_init = 1;
        block_size_init = num_states_local;
    }

    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_flat_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device);
} /* initialize_flat */

void initialize_zero() {
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();
    if (proc_num == 0) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
        ATLC_CHECK_CUDA(cudaMemset, state_data_device + 1, 0, sizeof(qcs::complex_t) * (num_states_local -1));
    } else {
        ATLC_CHECK_CUDA(cudaMemset, state_data_device, 0, sizeof(qcs::complex_t) * num_states_local);
    }
}

void initialize_entangled() {
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();
    if (proc_num == 0) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
    } else if (proc_num == num_procs - 1) {
        qcs::complex_t const one = 1;
        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device + num_states_local - 1, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
    }
}

void initialize_use_curand() {

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] generating random state\n"); }
    curandGenerator_t rng_device;

    ATLC_CHECK_CUDA(cudaEventRecord, event_1, stream);

    {
        int const log_num_rand_areas = atlc::log2_int(num_rand_areas);
        uint64_t const num_states_rand_area = num_states_local >> log_num_rand_areas;
        for (int rand_area_num = 0; rand_area_num < num_rand_areas; rand_area_num++) {
            ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
            ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
            ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * num_rand_areas + rand_area_num);
            ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t*)(void*)(state_data_device + num_states_rand_area * ((uint64_t)rand_area_num)), num_states_rand_area * 2 /* complex */, 0.0, 1.0);
            ATLC_CHECK_CURAND(curandDestroyGenerator, rng_device);
        }
    }
} /* initialize_use_curand */

void initialize_laod_statevector() {
    measured_0_qubit_num_logical_list.clear();
    measured_1_qubit_num_logical_list.clear();

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_num == 0) { fprintf(stderr, "[info] load statevector\n"); }

    qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states_local * sizeof(qcs::complex_t));
    ATLC_DEFER_FUNC(free, state_data_host);

    for(int proc_num_active=0; proc_num_active<num_procs; proc_num_active++) {
        if (proc_num_active == proc_num) {
            FILE* const fp = fopen("statevector_input.bin", "rb");
            if (fp == NULL) {
                throw std::runtime_error("open failed");
            }
            fseek(fp, proc_num * num_states_local * sizeof(qcs::complex_t), SEEK_SET);
            size_t const ret = fread(state_data_host, sizeof(qcs::complex_t), num_states_local, fp);
            if (ret != num_states_local) {
                throw std::runtime_error("fread failed");
            }
            fclose(fp);

            ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host, num_states_local * sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);

        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
} /* initialize_laod_statevector */

void prepare_control_qubit_num_list() {

    std::vector<int>* const measured_X_qubit_num_logical_list_list[] = {
        &measured_0_qubit_num_logical_list,
        &measured_1_qubit_num_logical_list
    };

    std::vector<int>* const measured_X_qubit_num_logical_list_copy_list[] = {
        &measured_0_qubit_num_logical_list_copy,
        &measured_1_qubit_num_logical_list_copy
    };

    std::vector<int>* const X_control_qubit_num_logical_list_list[] = {
        &negative_control_qubit_num_logical_list,
        &positive_control_qubit_num_logical_list
    };

    measured_control_condition = true;

    #pragma unroll
    for(int measured_value = 0; measured_value < 2; measured_value++) {
        measured_X_qubit_num_logical_list_copy_list[measured_value]->clear();
        for (int mXqnl_idx = 0; mXqnl_idx < measured_X_qubit_num_logical_list_list[measured_value]->size(); mXqnl_idx++) {
            auto const mXqn = measured_X_qubit_num_logical_list_list[measured_value]->operator[](mXqnl_idx);
            bool const is_target = std::find(target_qubit_num_logical_list.begin(), target_qubit_num_logical_list.end(), mXqn) != target_qubit_num_logical_list.end();
            if (!is_target) {
                // the qubit is kept measured
                measured_X_qubit_num_logical_list_copy_list[measured_value]->push_back(mXqn);
                bool const is_control = std::find(X_control_qubit_num_logical_list_list[measured_value]->begin(), X_control_qubit_num_logical_list_list[measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[measured_value]->end();
                bool const is_control_other = std::find(X_control_qubit_num_logical_list_list[1-measured_value]->begin(), X_control_qubit_num_logical_list_list[1-measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[1-measured_value]->end();
                if ((!is_control)&&(!is_control_other)) {
                    X_control_qubit_num_logical_list_list[measured_value]->push_back(mXqn);
                } else if (is_control_other) {
                    measured_control_condition = false;
                }
            }
        }
    }

} /* prepare_control_qubit_num_list */

void ensure_local_qubits() {
    target_qubit_num_physical_list.resize(target_qubit_num_logical_list.size());
    for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++) {
        target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
    }

    swap_target_global_list.resize(0);
    swap_target_local_list.resize(0);
    for (int tqni = 0; tqni < target_qubit_num_physical_list.size(); tqni++) {
        auto const tqn_i = target_qubit_num_physical_list[tqni];
        if (tqn_i >= num_qubits_local) {
            swap_target_global_list.push_back(tqn_i);
            int const swap_target_local = num_qubits_local - swap_target_global_list.size();
            swap_target_local_list.push_back(swap_target_local);
            target_qubit_num_physical_list[tqni] = swap_target_local;
        }
    }
    int const num_swap_qubits = swap_target_global_list.size();

    /* target qubits is global */
    if (swap_target_global_list.size() > 0) {

        // b_min
        int const swap_target_local_min = *std::min_element(swap_target_local_list.data(), swap_target_local_list.data() + num_swap_qubits);

        uint64_t const local_buf_length = UINT64_C(1) << swap_target_local_min;
        uint64_t swap_buffer_length = swap_buffer_total_length;
        if (swap_buffer_length > local_buf_length) {
            swap_buffer_length = local_buf_length;
        }

        // generate a mask for generating global_nonswap_self
        uint64_t global_swap_self_mask = 0;
        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            // a_delta = a – n_local
            int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
            global_swap_self_mask |= (UINT64_C(1) << swap_target_global_delta);
        }

        // global_nonswap_self = make proc_num_self's a_delta_i-th digit zero
        uint64_t const global_nonswap_self = proc_num & ~global_swap_self_mask;

        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

        // 1<<(num_local_qubits - b_min) 
        uint64_t const num_local_areas = UINT64_C(1) << (num_qubits_local - swap_target_local_min);
        for (uint64_t local_num_self = 0; local_num_self < num_local_areas; local_num_self++) {

            // global_swap_peer = OR_i (local_num_selfのb_delta_i桁目)をa_delta_i桁目にする
            uint64_t global_swap_peer = 0;
            for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
                // a_delta_i
                int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                // b_delta_i
                int const swap_target_local_delta = swap_target_local_list[target_num] - swap_target_local_min;
                global_swap_peer |=
                    // local_num_selfのb_delta_i桁目
                    ((local_num_self >> swap_target_local_delta) & 1)
                    // をa_delta_i桁目にする
                    << swap_target_global_delta;
            }

            uint64_t const proc_num_peer = global_swap_peer | global_nonswap_self;

            // send & recv
            if (proc_num_peer == proc_num) { continue; }

            bool is_peer_greater = proc_num_peer > proc_num;
            for (uint64_t buffer_pos = 0; buffer_pos < local_buf_length; buffer_pos += swap_buffer_length) {
                ATLC_CHECK_NCCL(ncclGroupStart);
                for (int send_recv = 0; send_recv < 2; send_recv++) {
                    if (send_recv ^ is_peer_greater) {
                        ATLC_CHECK_NCCL(ncclSend, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                    } else {
                        ATLC_CHECK_NCCL(ncclRecv, swap_buffer, swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                    }
                }
                ATLC_CHECK_NCCL(ncclGroupEnd);
                ATLC_CHECK_CUDA(cudaMemcpyAsync, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer, swap_buffer_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, stream);
            }

        }

        // swap_target_global_logical_list[:] = perm_p2l[swap_target_global_list[:]]
        // swap_target_local_logical_list[:] = perm_p2l[swap_target_local_list[:]]
        swap_target_local_logical_list.resize(num_swap_qubits);
        swap_target_global_logical_list.resize(num_swap_qubits);
        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            swap_target_local_logical_list[target_num] = perm_p2l[swap_target_local_list[target_num]];
            swap_target_global_logical_list[target_num] = perm_p2l[swap_target_global_list[target_num]];
        }

        // update p2l & l2p
        // perm_p2l[swap_target_global_list[:]] = swap_target_local_logical_list[:]
        // perm_p2l[swap_target_local_list[:]] = swap_target_global_logical_list[:]
        // perm_l2p[swap_target_global_logical_list[:]] = swap_target_local_list[:]
        // perm_l2p[swap_target_local_logical_list[:]] = swap_target_global_list[:]

        for (int target_num = 0; target_num < num_swap_qubits; target_num++) {
            perm_p2l[swap_target_global_list[target_num]] = swap_target_local_logical_list[target_num];
            perm_p2l[swap_target_local_list[target_num]] = swap_target_global_logical_list[target_num];
            perm_l2p[swap_target_global_logical_list[target_num]] = swap_target_local_list[target_num];
            perm_l2p[swap_target_local_logical_list[target_num]] = swap_target_global_list[target_num];
        }

        // target_qubit_num_physical = swap_target_local;

    }
}; /* ensure_local_qubits */

void check_control_qubit_num_physical() {

    /* check whether proc_num is under control condition */
    proc_num_control_condition = true;

    std::vector<int>* x_control_qubit_num_logical_list_list[] = {&negative_control_qubit_num_logical_list, &positive_control_qubit_num_logical_list};
    std::vector<int>* x_control_qubit_num_physical_list_list[] = {&negative_control_qubit_num_physical_list, &positive_control_qubit_num_physical_list};
    std::vector<int>* x_control_qubit_num_physical_global_list_list[] = {&negative_control_qubit_num_physical_global_list, &positive_control_qubit_num_physical_global_list};
    std::vector<int>* x_control_qubit_num_physical_local_list_list[] = {&negative_control_qubit_num_physical_local_list, &positive_control_qubit_num_physical_local_list};

    #pragma unroll
    for (int control_np = 0; control_np < 2; control_np++) {
        x_control_qubit_num_physical_list_list[control_np]->resize(x_control_qubit_num_logical_list_list[control_np]->size());
        x_control_qubit_num_physical_global_list_list[control_np]->resize(0);
        x_control_qubit_num_physical_local_list_list[control_np]->resize(0);

        for (int cqni = 0; cqni < x_control_qubit_num_logical_list_list[control_np]->size(); cqni++) {
            auto const x_control_qubit_num_physical = perm_l2p[x_control_qubit_num_logical_list_list[control_np]->operator[](cqni)];
            x_control_qubit_num_physical_list_list[control_np]->operator[](cqni) = x_control_qubit_num_physical;
            if (x_control_qubit_num_physical >= num_qubits_local) {
                x_control_qubit_num_physical_global_list_list[control_np]->push_back(x_control_qubit_num_physical);
                if ((1 & (proc_num >> (x_control_qubit_num_physical - num_qubits_local))) != control_np) {
                    proc_num_control_condition = false;
                }
            } else {
                x_control_qubit_num_physical_local_list_list[control_np]->push_back(x_control_qubit_num_physical);
            }
        }
    }

}; /* check_control_qubit_num_physical */

void prepare_operating_gate() {

    if (!proc_num_control_condition) { return; }

    uint64_t const qkiqn_size = qcs::kernel_input_qnlist_struct::needed_size(
        positive_control_qubit_num_physical_list.size(),
        negative_control_qubit_num_physical_list.size(),
        target_qubit_num_physical_list.size()
    );
    if (qkiqn_size > qcs::kernel_input_max_size) {
        throw std::runtime_error(atlc::format("qkiqn_size(%" PRIu64 ") > qcs::kernel_input_max_size(%" PRIu64 ")", qkiqn_size, qcs::kernel_input_max_size));
    }
    qcs_kernel_input_host_buffer.resize(qkiqn_size);
    qcs::kernel_input_qnlist_struct* const qcs_kernel_input_host = (qcs::kernel_input_qnlist_struct*)qcs_kernel_input_host_buffer.data();

    qcs_kernel_input_host->num_positive_control_qubits = positive_control_qubit_num_physical_local_list.size();
    qcs_kernel_input_host->num_negative_control_qubits = negative_control_qubit_num_physical_local_list.size();
    qcs_kernel_input_host->num_target_qubits = target_qubit_num_physical_list.size();

    auto positive_control_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_positive_control_qubit_num_list();
    for (int pcqi = 0; pcqi < positive_control_qubit_num_physical_local_list.size(); pcqi++) {
        positive_control_qubit_num_list_kernel_arg[pcqi] = positive_control_qubit_num_physical_local_list[pcqi];
    }

    auto target_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_target_qubit_num_list();
    qcs_kernel_input_host->is_measured_bits = 0;
    qcs_kernel_input_host->measured_value_bits = 0;

    for (int tqi = 0; tqi < target_qubit_num_physical_list.size(); tqi++) {
        auto const tqn_phys = target_qubit_num_physical_list[tqi];
        target_qubit_num_list_kernel_arg[tqi] = tqn_phys;

        bool is_measured = false;
        int measured_value = 0;

        std::vector<int>* const measured_X_qubit_num_logical_list_list[] = {
            &measured_0_qubit_num_logical_list,
            &measured_1_qubit_num_logical_list
        };

        for (int loop_measured_value = 0; loop_measured_value < 2; loop_measured_value++) {
            for(int mXqnl_idx = 0; mXqnl_idx < measured_X_qubit_num_logical_list_list[loop_measured_value]->size(); mXqnl_idx++) {
                auto const mXqn_phys = perm_l2p[measured_X_qubit_num_logical_list_list[loop_measured_value]->operator[](mXqnl_idx)];
                if (mXqn_phys == tqn_phys) {
                    is_measured = true;
                    measured_value = loop_measured_value;
                    break;
                }
            }
            if (is_measured) { break; }
        }

        if (is_measured) {
            qcs_kernel_input_host->is_measured_bits |= UINT64_C(1) << tqi;
            if (measured_value /* == 1 */) {
                qcs_kernel_input_host->measured_value_bits |= UINT64_C(1) << tqi;
            }
        }

    }

    num_operand_qubits =
        positive_control_qubit_num_physical_local_list.size()
        + negative_control_qubit_num_physical_local_list.size()
        + target_qubit_num_physical_list.size();

    /* get sorted operand qubits */
    operand_qubit_num_list.clear();
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), positive_control_qubit_num_physical_local_list.begin(), positive_control_qubit_num_physical_local_list.end());
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), negative_control_qubit_num_physical_local_list.begin(), negative_control_qubit_num_physical_local_list.end());
    operand_qubit_num_list.insert(operand_qubit_num_list.end(), target_qubit_num_physical_list.begin(), target_qubit_num_physical_list.end());

    std::sort(operand_qubit_num_list.begin(), operand_qubit_num_list.end()); /* ascending order */

    auto qubit_num_list_sorted_kernel_arg = qcs_kernel_input_host->get_operand_qubit_num_list_sorted();
    for (int qni = 0; qni < operand_qubit_num_list.size(); qni++) {
        qubit_num_list_sorted_kernel_arg[qni] = operand_qubit_num_list[qni];
    }

    ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_input_constant_addr, qcs_kernel_input_host, qkiqn_size, cudaMemcpyHostToDevice, stream);

    uint64_t const log_num_threads = num_qubits_local - num_operand_qubits;

    uint64_t log_block_size_gateop;

    if (log_block_size_max > log_num_threads) {
        log_block_size_gateop = log_num_threads;
        num_blocks_gateop = 1;
    } else {
        log_block_size_gateop = log_block_size_max;
        num_blocks_gateop = UINT64_C(1) << (log_num_threads - log_block_size_max);
    }

    block_size_gateop = UINT64_C(1) << log_block_size_gateop;

} /* prepare_operating_gate */

void update_measured_list() {
    measured_0_qubit_num_logical_list = measured_0_qubit_num_logical_list_copy;
    measured_1_qubit_num_logical_list = measured_1_qubit_num_logical_list_copy;
}

void save_statevector() {

    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_num == 0) { fprintf(stderr, "[info] dump statevector\n"); }

    qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states_local * sizeof(qcs::complex_t));
    ATLC_DEFER_FUNC(free, state_data_host);

    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);

    for(int proc_num_active=0; proc_num_active<num_procs; proc_num_active++) {
        if (proc_num_active == proc_num) {
            FILE* const fp = fopen("statevector_output.bin", (proc_num==0)? "wb": "rb+");
            if (fp == NULL) {
                throw std::runtime_error("open failed");
            }

            for (uint64_t state_num_physical_local = 0; state_num_physical_local < num_states_local; state_num_physical_local++) {
                uint64_t const state_num_physical = state_num_physical_local | (((uint64_t)proc_num) << num_qubits_local);
                uint64_t state_num_logical = 0;
                for(int qubit_num_physical = 0; qubit_num_physical < num_qubits; qubit_num_physical++) {
                    int qubit_num_logical = perm_p2l[qubit_num_physical];
                    state_num_logical = state_num_logical | (((state_num_physical >> qubit_num_physical) & 1) << qubit_num_logical);
                }
                int const ret_fseek = fseek(fp, state_num_logical * sizeof(qcs::complex_t), SEEK_SET);
                if (ret_fseek!=0) {
                    auto const errno_saved = errno;
                    throw std::runtime_error(atlc::format("errorno=%d ret_fseek=%d", errno_saved, ret_fseek));
                }
                size_t const ret = fwrite(&state_data_host[state_num_physical_local], sizeof(qcs::complex_t), 1, fp);
                if (ret != 1) {
                    auto const errno_saved = errno;
                    throw std::runtime_error(atlc::format("fwrite failed ret=%zu errno=%d", ret, errno_saved));
                }
            }
            fflush(fp);
            fclose(fp);
            fsync(fileno(fp));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
} /* save_statevector */

void calculate_checksum() {

    if (proc_num==0) {
        fprintf(stderr, "[info] gathering state data\n");

        EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
        if (!mdctx) {
            throw std::runtime_error("EVP_MD_CTX_new failed");
        }
        ATLC_DEFER_FUNC(EVP_MD_CTX_free, mdctx);

        if (EVP_DigestInit_ex(mdctx, EVP_md5(), NULL) != 1) {
            throw std::runtime_error("EVP_DigestInit_ex failed");
        }

        qcs::complex_t* state_data_host = (qcs::complex_t*)malloc(num_states * sizeof(qcs::complex_t));
        ATLC_DEFER_FUNC(free, state_data_host);

        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);
        for(int peer_proc_num=1; peer_proc_num<num_procs; peer_proc_num++) {
            MPI_Status mpi_status;
            MPI_Recv(&state_data_host[peer_proc_num * num_states_local], num_states_local * 2, MPI_DOUBLE, peer_proc_num, 0, MPI_COMM_WORLD, &mpi_status);
        }
        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);

        for(int64_t state_num_logical = 0; state_num_logical < num_states; state_num_logical++) {
            int64_t state_num_physical = 0;
            for(int qubit_num_logical = 0; qubit_num_logical < num_qubits; qubit_num_logical++) {
                int qubit_num_physical = perm_l2p[qubit_num_logical];
                state_num_physical = state_num_physical | (((state_num_logical >> qubit_num_logical) & 1) << qubit_num_physical);
            }

            if (EVP_DigestUpdate(mdctx, &state_data_host[state_num_physical], sizeof(qcs::complex_t)) != 1) {
                throw std::runtime_error("EVP_DigestUpdate failed");
            }
        }

        std::vector<unsigned char> evp_hash(EVP_MAX_MD_SIZE); // [EVP_MAX_MD_SIZE];
        unsigned int evp_hash_len;
        if (EVP_DigestFinal_ex(mdctx, evp_hash.data(), &evp_hash_len) != 1) {
            throw std::runtime_error("EVP_DigestFinal_ex failed");
        }

        fprintf(stderr, "[info] checksum: ");
        for (unsigned int i = 0; i < evp_hash_len; i++) {
            fprintf(stderr, "%02x", evp_hash[i]);
        }
        fprintf(stderr, "\n");

    } else {
        MPI_Send(state_data_device, num_states_local * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

} /* calculate_checksum */

int measure_qubit(int const measure_qubit_num_logical) {

    for (auto const m0qn : measured_0_qubit_num_logical_list) {
        if (measure_qubit_num_logical == m0qn) {
            return 0;
        }
    }

    for (auto const m1qn : measured_1_qubit_num_logical_list) {
        if (measure_qubit_num_logical == m1qn) {
            return 1;
        }
    }

    target_qubit_num_logical_list = {measure_qubit_num_logical};
    positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
    negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;

    ensure_local_qubits();
    check_control_qubit_num_physical();
    prepare_operating_gate();

    float2_t measure_norm_host;

    if (proc_num_control_condition) {

        cubUtility::IndirectLoad loader;

        using CountingIter = thrust::counting_iterator<uint64_t>;
        using TransformIter = thrust::transform_iterator<decltype(loader), CountingIter>;


        CountingIter counting(0);
        TransformIter in_it(counting, loader);

        uint64_t temp_sz_required;

        cubUtility::float2Add float2AddObj;
        float2_t zero{};
        ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, NULL, temp_sz_required, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

        if (cub_temp_buffer_device_size < temp_sz_required) {
            ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
            ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, temp_sz_required, stream);
            cub_temp_buffer_device_size = temp_sz_required;
        }

        ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, cub_temp_buffer_device, cub_temp_buffer_device_size, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

        ATLC_CHECK_CUDA(cudaMemcpyAsync, &measure_norm_host, measure_norm_device, sizeof(float2_t), cudaMemcpyDeviceToHost, stream);

        ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
    } else {
        measure_norm_host = {0, 0};
    }

#if 1 /* parallel measurement */
    qcs::float_t measure_norm_global[2];
    MPI_Allreduce(measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];

    std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
    qcs::float_t const random_value = dist1(engine);
    int measure_result = measure_norm_global[0] < random_value;

    if (measure_result) { /* 1 */
        measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
    } else { /* 0 */
        measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
    }
#else /* measurement by master */
    qcs::float_t measure_norm_global[2];
    MPI_Reduce(measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    bool measure_result_0;
    if (proc_num == 0) {
        qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];

        std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
        qcs::float_t const random_value = dist1(engine);
        measure_result_0 = measure_norm_global[0] < random_value;
    }
    bool measure_result;
    MPI_Scatter(&measure_result_0, 1, MPI_C_BOOL, &measure_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (measure_result) { /* 1 */
        measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
        // measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
    } else { /* 0 */
        measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
    }
#endif

    // clear alternative state
    {
        uint64_t const log_num_threads = num_qubits_local - 1;
        uint64_t log_block_size_gateop;

        if (log_block_size_max > log_num_threads) {
            log_block_size_gateop = log_num_threads;
            num_blocks_gateop = 1;
        } else {
            log_block_size_gateop = log_block_size_max;
            num_blocks_gateop = UINT64_C(1) << (log_num_threads - log_block_size_max);
        }

        block_size_gateop = UINT64_C(1) << log_block_size_gateop;

        auto const measure_qubit_num_physical = perm_l2p[measure_qubit_num_logical];
        ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, clear_alternative_state, num_blocks_gateop, block_size_gateop, 0, stream, measure_qubit_num_physical, measure_result);
    }

    return measure_result;
}

template<typename GateType>
void operate_gate(GateType gateobj, std::vector<int> target_qubit_num_logical_list_input, std::vector<int> negative_control_qubit_num_logical_list_input, std::vector<int> positive_control_qubit_num_logical_list_input) {

    assert(target_qubit_num_logical_list_input.size() == GateType::num_target_qubits);

    target_qubit_num_logical_list = std::move(target_qubit_num_logical_list_input);
    negative_control_qubit_num_logical_list = std::move(negative_control_qubit_num_logical_list_input);
    positive_control_qubit_num_logical_list = std::move(positive_control_qubit_num_logical_list_input);

    prepare_control_qubit_num_list();
    if (!measured_control_condition) return;

    ensure_local_qubits();
    check_control_qubit_num_physical();
    prepare_operating_gate();

    if (proc_num_control_condition) {
        ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<GateType>, num_blocks_gateop, block_size_gateop, 0, stream, gateobj);
    }

    update_measured_list();

}

#if 0
void GHZ_circuit_sample() {

    allocate_memory(14);

    uint64_t measured_bit = 0;

    for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
        int const measured_value = measure_qubit(measure_qubit_num_logical);
        if (measured_value) {
            measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
        }
    }

    if (proc_num == 0) {
        fprintf(stdout, "%" PRIu64 "\n", measured_bit);
    }

    uint64_t const num_samples = UINT64_C(1) << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        /* begin gate operation */
        operate_gate(gate::hadamard(), {0}, {}, {});

        for(int target_qubit_num_logical = 1; target_qubit_num_logical < num_qubits; target_qubit_num_logical++)
        {
            if ((measured_bit>>target_qubit_num_logical)&1) {
                operate_gate(gate::x(), {target_qubit_num_logical}, {0}, {});
            } else {
                operate_gate(gate::x(), {target_qubit_num_logical}, {}, {0});
            }

        } /* target_qubit_num_logical loop */

        /* end gate operation */

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

}; /* GHZ_circuit_sample */

void measurement_sample() {

    allocate_memory(14);

    constexpr initstate_enum initstate_choice = initstate_enum::flat;
    switch (initstate_choice) {
        case initstate_enum::sequential:
            initialize_sequential();
            break;
        case initstate_enum::flat:
            initialize_flat();
            break;
        case initstate_enum::zero:
            initialize_zero();
            break;
        case initstate_enum::entangled:
            initialize_entangled();
            break;
        case initstate_enum::use_curand:
            initialize_use_curand();
            break;
        case initstate_enum::laod_statevector:
            initialize_laod_statevector();
            break;
        default:
            throw initstate_choice;
    }

    uint64_t measured_bit = 0;
    uint64_t const num_samples = UINT64_C(1) << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        // forget measurement
        // warn: do not use it, unless you fully understand the behavior of lazy view.
        measured_0_qubit_num_logical_list.clear();
        measured_1_qubit_num_logical_list.clear();

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

} /* measurement_sample */

int main() {

    constexpr bool flag_calculate_checksum = false;
    constexpr bool flag_save_statevector = false;

    setup();
    ATLC_DEFER_FUNC(dispose);

    ATLC_DEFER_FUNC(free_memory);

    // measurement_sample();
    GHZ_circuit_sample();

    if (flag_save_statevector) { save_statevector(); }
    if (flag_calculate_checksum) { calculate_checksum(); }

    return 0;

}; /* main */
#endif

void dispose() {
    free_memory();
    ATLC_CHECK_CUDA(cudaEventDestroy, event_1);
    ATLC_CHECK_CUDA(cudaEventDestroy, event_2);
    ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
    MPI_Finalize();
};

}; /* simulator_core */

simulator::simulator() {
    num_qubits = 0;
    core = NULL;
}

void simulator::setup() {
    core = new simulator_core();
    core->setup();
}

void simulator::dispose() {
    core->dispose();
    delete this->core;
}

int simulator::get_proc_num() {
    return core->proc_num;
}

int simulator::get_num_procs() {
    return core->num_procs;
}

void simulator::set_num_qubits(int num_qubits) {
    this->num_qubits = num_qubits;
}

void simulator::set_num_clbits(int num_clbits) {
    this->num_clbits = num_clbits;
    // this->clbits.resize(num_clbits);
}

int simulator::measure(int qubit_num) {
    ensure_qubits_allocated();
    int const result = core->measure_qubit(qubit_num);
    return result;
}

int simulator::measure(int qubit_num, int clbit_num) {
    int const result = measure(qubit_num);
    clbits[clbit_num] = result;
    return result;
}

int simulator::read(int clbit_num) {
    return clbits[clbit_num];
}

void simulator::ensure_qubits_allocated() {
    if (!core->is_qubit_allocated()) {
        core->allocate_memory(num_qubits);
    }
}

void simulator::reset(int qubit_num) {
    int measured_value = measure(qubit_num);
    if (measured_value) {
        gate_x({qubit_num}, {}, {});
    }
}

void simulator::set_zero_state() {
    ensure_qubits_allocated();
    core->initialize_zero();
}

void simulator::set_sequential_state() {
    ensure_qubits_allocated();
    core->initialize_sequential();
}

void simulator::set_flat_state() {
    ensure_qubits_allocated();
    core->initialize_flat();
}

void simulator::set_entangled_state() {
    ensure_qubits_allocated();
    core->initialize_entangled();
}

void simulator::set_random_state() {
    ensure_qubits_allocated();
    core->initialize_use_curand();
}

void simulator::gate_global_phase(double theta, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::global_phase(theta), {}, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_swap(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::swap(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_iswap(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::iswap(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_h(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::hadamard(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_x(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::x(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_y(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::y(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_z(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::z(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_s(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::s(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_sdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::sdg(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_t(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::t(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_tdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::tdg(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_sx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::sx(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_sxdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::sxdg(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rx(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_ry(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::ry(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rz(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rz(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_u4(double theta, double phi, double lambda, double gamma, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::u4(theta, phi, lambda, gamma), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}

void simulator::gate_u3(double theta, double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::u3(theta, phi, lambda), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}

void simulator::gate_u2(double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::u2(phi, lambda), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}

void simulator::gate_u1(double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::u1(lambda), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}

void simulator::gate_u(double theta, double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::u(theta, phi, lambda), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}

void simulator::gate_p(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::p(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_id(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::id(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_r(double theta, double phi, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::r(theta, phi), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rzz(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rzz(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rxx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rxx(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_ryy(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::ryy(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rzx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rzx(theta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_dcx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::dcx(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_ecr(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::ecr(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_xx_plus_yy(double theta, double beta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::xx_plus_yy(theta, beta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_xx_minus_yy(double theta, double beta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::xx_minus_yy(theta, beta), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rccx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rccx(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


void simulator::gate_rcccx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list) {
    ensure_qubits_allocated();
    core->operate_gate(gate::rcccx(), target_qubit_num_list, negctrl_qubit_num_list, ctrl_qubit_num_list);
}


} /* qcs */

int main(int argc, char** argv)
{
    qcs::simulator sim;
    sim.setup();
    ATLC_DEFER_FUNC(sim.dispose);

    if (argc <= 1) { throw std::runtime_error("argv[1] is not set."); }
    char const* const usercircuit_so_path = argv[1];
    int num_samples = 1;
    if (argc >= 3) { num_samples = atoi(argv[2]); }

    char const* const usercircuit_so_abspath = realpath(usercircuit_so_path, NULL);
    if (usercircuit_so_abspath == NULL) {
        throw std::runtime_error("realpath returned NULL");
    }
    ATLC_DEFER_FUNC(free, (void*)usercircuit_so_abspath);

    void* usercircuit_dl = dlopen(usercircuit_so_abspath, RTLD_LAZY);
    if (usercircuit_dl == NULL) { throw std::runtime_error("dlopen failed"); }
    ATLC_DEFER_FUNC(dlclose, usercircuit_dl);

    auto circuit_init = reinterpret_cast<void(*)(qcs::simulator*)>(dlsym(usercircuit_dl, "circuit_init"));
    auto circuit_run = reinterpret_cast<void(*)(qcs::simulator*)>(dlsym(usercircuit_dl, "circuit_run"));

    circuit_init(&sim);

    for (int i=0; i<num_samples; i++) {

        circuit_run(&sim);

        unsigned long long const clbit = sim.get_clbits().to_ullong();
        fprintf(stdout, "%llu\n", clbit);

        if (num_samples > 1) {
            sim.set_zero_state();
            sim.reset_clbits();
        }
    }

    return 0;
}
