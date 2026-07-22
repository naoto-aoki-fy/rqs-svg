#include <qcs.h>

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int num_qubits = 3;
    const int num_clbits = num_qubits;
    const int mapping[] = {0, 1, 2};

    qcs_simulator* sim = qcs_simulator_create();
    if (sim == NULL) {
        fprintf(stderr, "failed to create qcs simulator\n");
        return EXIT_FAILURE;
    }

    qcs_simulator_set_num_qubits(sim, num_qubits);
    qcs_simulator_set_num_clbits(sim, num_clbits);
    qcs_simulator_set_mapping(sim, mapping, num_qubits);
    qcs_simulator_allocate_memory(sim);

    int target[] = {0};
    qcs_simulator_gate_h(sim, target, 1, NULL, 0, NULL, 0);

    for (int qubit_num = 1; qubit_num < num_qubits; qubit_num++) {
        int x_target[] = {qubit_num};
        int ctrl[] = {0};
        qcs_simulator_gate_x(sim, x_target, 1, NULL, 0, ctrl, 1);
    }

    for (int qubit_num = 0; qubit_num < num_qubits; qubit_num++) {
        qcs_simulator_measure_to_clbit(sim, qubit_num, qubit_num);
    }

    char clbits[num_clbits + 1];
    qcs_simulator_get_clbits_string(sim, clbits);
    qcs_simulator_fprintf_master(sim, stdout, "measured clbits: %s\n", clbits);
    qcs_simulator_fflush_master(sim, stdout);

    qcs_simulator_destroy(sim);
    return EXIT_SUCCESS;
}
