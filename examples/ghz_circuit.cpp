#include <qcs.h>
#include <cstdlib>
#include <stdexcept>

static unsigned int num_qubits;
static unsigned int num_clbits;

extern "C"
void circuit_init(qcs_simulator* sim) {
    char const* const num_qubits_str = getenv("NUM_QUBITS");
    if (num_qubits_str == NULL || num_qubits_str[0] == '\0') {
        throw std::runtime_error("NUM_QUBITS is empty");
    }
    char* endptr;
    num_qubits = strtoul(num_qubits_str, &endptr, 10);
    if (endptr == num_qubits_str) {
        throw std::runtime_error("strtoul on NUM_QUBITS failed");
    }
    qcs_simulator_set_num_qubits(sim, num_qubits);
    num_clbits = num_qubits;
    qcs_simulator_set_num_clbits(sim, num_clbits);
}

extern "C"
void circuit_run(qcs_simulator* sim) {

    int target[] = {0};
    qcs_simulator_gate_h(sim, target, 1, NULL, 0, NULL, 0);

    for(int qubit_num = 1; qubit_num < num_qubits; qubit_num++)
    {
        int target[] = {qubit_num};
        int ctrl[] = {0};
        qcs_simulator_gate_x(sim, target, 1, NULL, 0, ctrl, 1);
    }

    for (int qubit_num = 0; qubit_num < num_qubits; qubit_num++) {
        qcs_simulator_measure_to_clbit(sim, qubit_num, qubit_num);
    }

}
