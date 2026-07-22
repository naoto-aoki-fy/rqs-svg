#include <qcs.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int num_qubits;
static int num_clbits;

void circuit_init(qcs_simulator* sim) {
    const char* const num_qubits_str = getenv("NUM_QUBITS");
    if (num_qubits_str == NULL || num_qubits_str[0] == '\0') {
        fprintf(stderr, "NUM_QUBITS is empty\n");
        exit(EXIT_FAILURE);
    }
    char* endptr;
    unsigned long parsed_num_qubits = strtoul(num_qubits_str, &endptr, 10);
    if (endptr == num_qubits_str || parsed_num_qubits > INT_MAX) {
        fprintf(stderr, "strtoul on NUM_QUBITS failed\n");
        exit(EXIT_FAILURE);
    }
    num_qubits = (int)parsed_num_qubits;
    qcs_simulator_set_num_qubits(sim, num_qubits);
    num_clbits = num_qubits;
    qcs_simulator_set_num_clbits(sim, num_clbits);
}

void circuit_run(qcs_simulator* sim) {

    int target[] = {0};
    qcs_simulator_gate_h(sim, target, 1, NULL, 0, NULL, 0);

    for (int qubit_num = 1; qubit_num < num_qubits; qubit_num++) {
        int target[] = {qubit_num};
        int ctrl[] = {0};
        qcs_simulator_gate_x(sim, target, 1, NULL, 0, ctrl, 1);
    }

    for (int qubit_num = 0; qubit_num < num_qubits; qubit_num++) {
        qcs_simulator_measure_to_clbit(sim, qubit_num, qubit_num);
    }

}
