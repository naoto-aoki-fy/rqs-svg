#include <qcs.hpp>
#include <cstdlib>

static unsigned int num_qubits;
static constexpr unsigned int num_clbits = 0;

extern "C"
void circuit_init(qcs::simulator* sim) {
    char const* const num_qubits_str = getenv("NUM_QUBITS");
    if (num_qubits_str == NULL || num_qubits_str[0] == '\0') {
        throw std::runtime_error("NUM_QUBITS is empty");
    }
    char* endptr;
    num_qubits = strtoul(num_qubits_str, &endptr, 10);
    if (endptr == num_qubits_str) {
        throw std::runtime_error("strtoul on NUM_QUBITS failed");
    }
    sim->set_num_qubits(num_qubits);
    sim->set_num_clbits(num_clbits);
}

extern "C"
void circuit_run(qcs::simulator* sim) {
    for(int qubit_num = 0; qubit_num < num_qubits; qubit_num++) {
        sim->gate_h({qubit_num}, {}, {});
    }
}
