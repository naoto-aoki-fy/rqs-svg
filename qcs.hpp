#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace qcs {
    struct simulator_core;
    class simulator {
    private:
        simulator_core* core;
        int num_qubits;
        void ensure_qubits_allocated();
    public:
        simulator();
        void setup();
        void dispose();

        int get_num_procs();
        int get_proc_num();

        void promise_qubits(int num_qubits);

        void reset(int qubit_num);
        void set_zero_state();
        void set_sequential_state();
        void set_flat_state();
        void set_entangled_state();
        void set_random_state();

        void global_phase(double theta, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void global_phase_pow(double exponent, double theta, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void hadamard(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void hadamard_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_x(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_x_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_y(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_y_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_z(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_z_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_s(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_s_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_sdg(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_sdg_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_t(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_t_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_tdg(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_tdg_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_sx(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_sx_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rx(double theta, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rx_pow(double theta, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_ry(double theta, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_ry_pow(double theta, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rz(double theta, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rz_pow(double theta, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_u4(double theta, double phi, double lambda, double gamma, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_u4_pow(double theta, double phi, double lambda, double gamma, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void swap(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void swap_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void iswap(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void iswap_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_id(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_id_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_sxdg(int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_sxdg_pow(double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_r(double theta, double phi, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_r_pow(double theta, double phi, double exponent, int target_qubit_num, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rxx(double theta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rxx_pow(double theta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_ryy(double theta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_ryy_pow(double theta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rzz(double theta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rzz_pow(double theta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rzx(double theta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rzx_pow(double theta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_dcx(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_dcx_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_ecr(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_ecr_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_xx_plus_yy(double theta, double beta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_xx_plus_yy_pow(double theta, double beta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_xx_minus_yy(double theta, double beta, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_xx_minus_yy_pow(double theta, double beta, double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rccx(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rccx_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        void gate_rcccx(std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);
        void gate_rcccx_pow(double exponent, std::vector<int>&& target_qubit_num_list, std::vector<int>&& negctrl_qubit_num_list, std::vector<int>&& ctrl_qubit_num_list);

        int measure(int qubit_num);
    };
}