// g++ -fPIC -shared -I.. -std=c++11 ghz_circuit.cpp -o ghz_circuit.so

#include <qcs.hpp>

static constexpr unsigned int num_qubits = 14;
static constexpr unsigned int num_clbits = num_qubits;

extern "C"
void circuit_init(qcs::simulator* sim) {
    sim->set_num_qubits(num_qubits);
    sim->set_num_clbits(num_clbits);
}

extern "C"
void circuit_run(qcs::simulator* sim) {

    sim->gate_h({0}, {}, {});

    for(int qubit_num = 1; qubit_num < num_qubits; qubit_num++)
    {
        sim->gate_x({qubit_num}, {}, {0});
    }

    for (int qubit_num = 0; qubit_num < num_qubits; qubit_num++) {
        sim->measure(qubit_num, qubit_num);
    }

}