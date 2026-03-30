#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace qcs {
    constexpr unsigned int max_num_clbits = 64;
    struct simulator_core;
    class simulator {
    private:
        simulator_core* core;
        int num_qubits;
        int num_clbits;
        std::vector<bool> clbits;
    public:
        simulator();
        void setup();
        void allocate_memory();
        void dispose();

        int get_num_procs();
        int get_proc_num();
        int get_num_qubits() const;

        void set_num_qubits(int num_qubits);
        void set_mapping(std::vector<int> const& perm_p2l);
        void set_num_clbits(int num_clbits);
        std::vector<bool> const& get_clbits() const;
        std::vector<bool>& get_clbits();
        std::string get_clbits_string() const;

        void reset(int qubit_num);
        void set_zero_state();
        void set_sequential_state();
        void set_flat_state();
        void set_entangled_state();
        void set_random_state();
        void reset_clbits();
        void reset_measurement_state();
        void reinitialize_mapping();

        void gate_global_phase(double theta, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_h(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_x(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_y(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_z(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_s(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_sdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_t(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_tdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_sx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_ry(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rz(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_u4(double theta, double phi, double lambda, double gamma, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_u3(double theta, double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_u2(double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_u1(double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_u(double theta, double phi, double lambda, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_p(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_swap(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_iswap(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_id(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_sxdg(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_r(double theta, double phi, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rxx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_ryy(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rzz(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rzx(double theta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_dcx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_ecr(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_xx_plus_yy(double theta, double beta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_xx_minus_yy(double theta, double beta, std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rccx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);
        void gate_rcccx(std::vector<int> target_qubit_num_list, std::vector<int> negctrl_qubit_num_list, std::vector<int> ctrl_qubit_num_list);

        int measure(int qubit_num);
        int measure(int qubit_num, int clbit_num);
        int read(int clbit_num);

        void save_statevector(char const* const outfn);

        int event_create();
        void event_record(int event_num);
        double event_get_elapsed_time(int const start_event_num, int const stop_event_num);

        __attribute__((format(printf, 3, 4)))
        int fprintf_master(FILE *fp, const char *format, ...);

        __attribute__((format(printf, 3, 4)))
        int fprintf_all(FILE *fp, const char *format, ...);

        int fflush_master(FILE* stream);

        int fflush_all(FILE* stream);
    };
}
