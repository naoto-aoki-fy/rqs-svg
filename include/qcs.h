#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qcs_simulator qcs_simulator;

qcs_simulator* qcs_simulator_create(void);
void qcs_simulator_destroy(qcs_simulator* sim);
void qcs_simulator_init(qcs_simulator* sim);
void qcs_simulator_setup(qcs_simulator* sim);
void qcs_simulator_allocate_memory(qcs_simulator* sim);
void qcs_simulator_dispose(qcs_simulator* sim);
int qcs_simulator_get_num_procs(qcs_simulator* sim);
int qcs_simulator_get_proc_num(qcs_simulator* sim);
int qcs_simulator_get_num_qubits(const qcs_simulator* sim);
int qcs_simulator_get_num_clbits(const qcs_simulator* sim);
void qcs_simulator_set_num_qubits(qcs_simulator* sim, int num_qubits);
void qcs_simulator_set_mapping(qcs_simulator* sim, const int* perm_p2l, int perm_p2l_count);
void qcs_simulator_set_num_clbits(qcs_simulator* sim, int num_clbits);
void qcs_simulator_get_clbits(const qcs_simulator* sim, int* clbits);
void qcs_simulator_get_clbits_string(const qcs_simulator* sim, char* clbits_string);
void qcs_simulator_reset(qcs_simulator* sim, int qubit_num);
void qcs_simulator_set_zero_state(qcs_simulator* sim);
void qcs_simulator_set_sequential_state(qcs_simulator* sim);
void qcs_simulator_set_flat_state(qcs_simulator* sim);
void qcs_simulator_set_entangled_state(qcs_simulator* sim);
void qcs_simulator_set_random_state(qcs_simulator* sim);
void qcs_simulator_reset_clbits(qcs_simulator* sim);
void qcs_simulator_reset_measurement_state(qcs_simulator* sim);
void qcs_simulator_reinitialize_mapping(qcs_simulator* sim);

void qcs_simulator_gate_global_phase(qcs_simulator* sim, double theta, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_h(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_x(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_y(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_z(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_s(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_sdg(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_t(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_tdg(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_sx(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rx(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_ry(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rz(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_u4(qcs_simulator* sim, double theta, double phi, double lambda, double gamma, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_u3(qcs_simulator* sim, double theta, double phi, double lambda, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_u2(qcs_simulator* sim, double phi, double lambda, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_u1(qcs_simulator* sim, double lambda, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_u(qcs_simulator* sim, double theta, double phi, double lambda, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_p(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_swap(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_iswap(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_id(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_sxdg(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_r(qcs_simulator* sim, double theta, double phi, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rxx(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_ryy(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rzz(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rzx(qcs_simulator* sim, double theta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_dcx(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_ecr(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_xx_plus_yy(qcs_simulator* sim, double theta, double beta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_xx_minus_yy(qcs_simulator* sim, double theta, double beta, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rccx(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);
void qcs_simulator_gate_rcccx(qcs_simulator* sim, const int* target_qubit_num_list, int target_qubit_num_count, const int* negctrl_qubit_num_list, int negctrl_qubit_num_count, const int* ctrl_qubit_num_list, int ctrl_qubit_num_count);

int qcs_simulator_measure(qcs_simulator* sim, int qubit_num);
int qcs_simulator_measure_to_clbit(qcs_simulator* sim, int qubit_num, int clbit_num);
int qcs_simulator_read(qcs_simulator* sim, int clbit_num);
void qcs_simulator_save_statevector(qcs_simulator* sim, const char* outfn);
int qcs_simulator_event_create(qcs_simulator* sim);
void qcs_simulator_event_record(qcs_simulator* sim, int event_num);
double qcs_simulator_event_get_elapsed_time(qcs_simulator* sim, int start_event_num, int stop_event_num);
__attribute__((format(printf, 3, 4))) int qcs_simulator_fprintf_master(qcs_simulator* sim, FILE *fp, const char *format, ...);
__attribute__((format(printf, 3, 4))) int qcs_simulator_fprintf_all(qcs_simulator* sim, FILE *fp, const char *format, ...);
int qcs_simulator_fflush_master(qcs_simulator* sim, FILE* stream);
int qcs_simulator_fflush_all(qcs_simulator* sim, FILE* stream);

#ifdef __cplusplus
}
#endif
