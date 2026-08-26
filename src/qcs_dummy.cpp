#include <qcs.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

struct qcs_simulator
{
    bit_num_t num_qubits = 0;
    bit_num_t num_clbits = 0;
    std::vector<bit_t> clbits;
    std::vector<bit_num_t> mapping;
    std::vector<std::chrono::steady_clock::time_point> events;
};

namespace
{
qcs_exception_callback exception_callback = nullptr;

template <typename Function>
int checked(Function function)
{
    try
    {
        function();
        return 1;
    }
    catch (const std::exception &error)
    {
        if (exception_callback != nullptr)
            exception_callback(error.what(), std::strlen(error.what()));
    }
    catch (...)
    {
        static const char message[] = "unknown exception";
        if (exception_callback != nullptr)
            exception_callback(message, sizeof(message) - 1);
    }
    return 0;
}

template <typename T>
void write_result(T *result, T value)
{
    if (result == nullptr)
        throw std::invalid_argument("result must not be null");
    *result = value;
}

void require_simulator(const qcs_simulator *sim)
{
    if (sim == nullptr)
        throw std::invalid_argument("simulator must not be null");
}
}

extern "C" int qcs_set_exception_callback(qcs_exception_callback callback)
{
    exception_callback = callback;
    return 1;
}

qcs_simulator *qcs_simulator_create_cxx() { return new qcs_simulator; }
void qcs_simulator_destroy_cxx(qcs_simulator *sim) { delete sim; }
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_dispose_cxx(qcs_simulator *sim) { require_simulator(sim); }
int qcs_simulator_get_num_procs_cxx(qcs_simulator *sim) { require_simulator(sim); return 1; }
int qcs_simulator_get_proc_num_cxx(qcs_simulator *sim) { require_simulator(sim); return 0; }
int qcs_simulator_get_num_qubits_cxx(const qcs_simulator *sim) { require_simulator(sim); return sim->num_qubits; }
int qcs_simulator_get_num_clbits_cxx(const qcs_simulator *sim) { require_simulator(sim); return sim->num_clbits; }
void qcs_simulator_set_num_qubits_cxx(qcs_simulator *sim, bit_num_t count) { require_simulator(sim); if (count < 0) throw std::invalid_argument("num_qubits must not be negative"); sim->num_qubits = count; }
void qcs_simulator_set_num_clbits_cxx(qcs_simulator *sim, bit_num_t count) { require_simulator(sim); if (count < 0) throw std::invalid_argument("num_clbits must not be negative"); sim->num_clbits = count; sim->clbits.assign(count, 0); }
void qcs_simulator_set_mapping_cxx(qcs_simulator *sim, const bit_num_t *mapping, bit_num_t count)
{
    require_simulator(sim);
    if (count != sim->num_qubits || (count != 0 && mapping == nullptr))
        throw std::invalid_argument("mapping must contain one entry per qubit");
    if (count == 0)
        sim->mapping.clear();
    else
        sim->mapping.assign(mapping, mapping + count);
}
void qcs_simulator_get_clbits_cxx(const qcs_simulator *sim, bit_t *output) { require_simulator(sim); if (sim->num_clbits && !output) throw std::invalid_argument("clbits must not be null"); std::copy(sim->clbits.begin(), sim->clbits.end(), output); }
void qcs_simulator_get_clbits_string_cxx(const qcs_simulator *sim, char *output) { require_simulator(sim); if (!output) throw std::invalid_argument("clbits_string must not be null"); for (auto it = sim->clbits.rbegin(); it != sim->clbits.rend(); ++it) *output++ = *it ? '1' : '0'; *output = '\0'; }
void qcs_simulator_reset_cxx(qcs_simulator *sim, bit_num_t) { require_simulator(sim); }
void qcs_simulator_set_zero_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_set_sequential_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_set_flat_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_set_entangled_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_set_random_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_reset_clbits_cxx(qcs_simulator *sim) { require_simulator(sim); std::fill(sim->clbits.begin(), sim->clbits.end(), 0); }
void qcs_simulator_reset_measurement_state_cxx(qcs_simulator *sim) { require_simulator(sim); }
void qcs_simulator_reinitialize_mapping_cxx(qcs_simulator *sim) { require_simulator(sim); }
int qcs_simulator_measure_cxx(qcs_simulator *sim, bit_num_t) { require_simulator(sim); return 0; }
int qcs_simulator_measure_to_clbit_cxx(qcs_simulator *sim, bit_num_t, bit_num_t clbit) { require_simulator(sim); if (clbit < 0 || clbit >= sim->num_clbits) throw std::out_of_range("clbit is out of range"); sim->clbits[clbit] = 0; return 0; }
int qcs_simulator_read_cxx(qcs_simulator *sim, bit_num_t clbit) { require_simulator(sim); if (clbit < 0 || clbit >= sim->num_clbits) throw std::out_of_range("clbit is out of range"); return sim->clbits[clbit]; }
void qcs_simulator_save_statevector_cxx(qcs_simulator *sim, const char *path) { require_simulator(sim); if (!path) throw std::invalid_argument("output path must not be null"); FILE *file = std::fopen(path, "wb"); if (!file) throw std::runtime_error("failed to create statevector placeholder"); std::fclose(file); }
int qcs_simulator_event_create_cxx(qcs_simulator *sim) { require_simulator(sim); sim->events.emplace_back(); return static_cast<int>(sim->events.size() - 1); }
void qcs_simulator_event_record_cxx(qcs_simulator *sim, int event) { require_simulator(sim); if (event < 0 || static_cast<size_t>(event) >= sim->events.size()) throw std::out_of_range("event is out of range"); sim->events[event] = std::chrono::steady_clock::now(); }
double qcs_simulator_event_get_elapsed_time_cxx(qcs_simulator *sim, event_num_t start, event_num_t stop) { require_simulator(sim); if (start < 0 || stop < 0 || static_cast<size_t>(start) >= sim->events.size() || static_cast<size_t>(stop) >= sim->events.size()) throw std::out_of_range("event is out of range"); return std::chrono::duration<double>(sim->events[stop] - sim->events[start]).count(); }

#define GATE_ARGS qcs_simulator *sim, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t
#define ANGLE_GATE_ARGS qcs_simulator *sim, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t
#define TWO_ANGLE_GATE_ARGS qcs_simulator *sim, double, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t
#define DUMMY_CXX(name, args) void qcs_simulator_gate_##name##_cxx(args) { require_simulator(sim); }
DUMMY_CXX(h, GATE_ARGS) DUMMY_CXX(x, GATE_ARGS) DUMMY_CXX(y, GATE_ARGS) DUMMY_CXX(z, GATE_ARGS)
DUMMY_CXX(s, GATE_ARGS) DUMMY_CXX(sdg, GATE_ARGS) DUMMY_CXX(t, GATE_ARGS) DUMMY_CXX(tdg, GATE_ARGS)
DUMMY_CXX(sx, GATE_ARGS) DUMMY_CXX(sxdg, GATE_ARGS) DUMMY_CXX(swap, GATE_ARGS) DUMMY_CXX(iswap, GATE_ARGS)
DUMMY_CXX(id, GATE_ARGS) DUMMY_CXX(dcx, GATE_ARGS) DUMMY_CXX(ecr, GATE_ARGS) DUMMY_CXX(rccx, GATE_ARGS) DUMMY_CXX(rcccx, GATE_ARGS)
DUMMY_CXX(rx, ANGLE_GATE_ARGS) DUMMY_CXX(ry, ANGLE_GATE_ARGS) DUMMY_CXX(rz, ANGLE_GATE_ARGS) DUMMY_CXX(u1, ANGLE_GATE_ARGS)
DUMMY_CXX(p, ANGLE_GATE_ARGS) DUMMY_CXX(rxx, ANGLE_GATE_ARGS) DUMMY_CXX(ryy, ANGLE_GATE_ARGS) DUMMY_CXX(rzz, ANGLE_GATE_ARGS) DUMMY_CXX(rzx, ANGLE_GATE_ARGS)
DUMMY_CXX(r, TWO_ANGLE_GATE_ARGS) DUMMY_CXX(u2, TWO_ANGLE_GATE_ARGS) DUMMY_CXX(xx_plus_yy, TWO_ANGLE_GATE_ARGS) DUMMY_CXX(xx_minus_yy, TWO_ANGLE_GATE_ARGS)
void qcs_simulator_gate_u3_cxx(qcs_simulator *sim, double, double, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t) { require_simulator(sim); }
void qcs_simulator_gate_u_cxx(qcs_simulator *sim, double, double, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t) { require_simulator(sim); }
void qcs_simulator_gate_u4_cxx(qcs_simulator *sim, double, double, double, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t) { require_simulator(sim); }
void qcs_simulator_gate_global_phase_cxx(qcs_simulator *sim, double, const bit_num_t *, bit_num_t, const bit_num_t *, bit_num_t) { require_simulator(sim); }

#define C_STATUS(name, params, call) extern "C" int qcs_simulator_##name params { return checked([&] { call; }); }
C_STATUS(create, (qcs_simulator **out), write_result(out, qcs_simulator_create_cxx()))
C_STATUS(destroy, (qcs_simulator *sim), qcs_simulator_destroy_cxx(sim))
C_STATUS(allocate_memory, (qcs_simulator *sim), qcs_simulator_allocate_memory_cxx(sim))
C_STATUS(dispose, (qcs_simulator *sim), qcs_simulator_dispose_cxx(sim))
#define C_GET(name, type) C_STATUS(name, (qcs_simulator *sim, type *out), write_result(out, static_cast<type>(qcs_simulator_##name##_cxx(sim))))
C_GET(get_num_procs, bit_t) C_GET(get_proc_num, bit_t)
extern "C" int qcs_simulator_get_num_qubits(const qcs_simulator *sim, bit_t *out) { return checked([&] { write_result(out, static_cast<bit_t>(qcs_simulator_get_num_qubits_cxx(sim))); }); }
extern "C" int qcs_simulator_get_num_clbits(const qcs_simulator *sim, bit_t *out) { return checked([&] { write_result(out, static_cast<bit_t>(qcs_simulator_get_num_clbits_cxx(sim))); }); }
C_STATUS(set_num_qubits, (qcs_simulator *sim, bit_num_t n), qcs_simulator_set_num_qubits_cxx(sim, n))
C_STATUS(set_num_clbits, (qcs_simulator *sim, bit_num_t n), qcs_simulator_set_num_clbits_cxx(sim, n))
C_STATUS(set_mapping, (qcs_simulator *sim, const bit_num_t *p, bit_num_t n), qcs_simulator_set_mapping_cxx(sim, p, n))
extern "C" int qcs_simulator_get_clbits(const qcs_simulator *sim, bit_t *out) { return checked([&] { qcs_simulator_get_clbits_cxx(sim, out); }); }
extern "C" int qcs_simulator_get_clbits_string(const qcs_simulator *sim, char *out) { return checked([&] { qcs_simulator_get_clbits_string_cxx(sim, out); }); }
#define C_SIMPLE(name) C_STATUS(name, (qcs_simulator *sim), qcs_simulator_##name##_cxx(sim))
C_SIMPLE(set_zero_state) C_SIMPLE(set_sequential_state) C_SIMPLE(set_flat_state) C_SIMPLE(set_entangled_state) C_SIMPLE(set_random_state)
C_SIMPLE(reset_clbits) C_SIMPLE(reset_measurement_state) C_SIMPLE(reinitialize_mapping)
C_STATUS(reset, (qcs_simulator *sim, bit_num_t q), qcs_simulator_reset_cxx(sim, q))
#define C_GATE(name) C_STATUS(gate_##name, (GATE_ARGS), qcs_simulator_gate_##name##_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count))
#undef GATE_ARGS
#define GATE_ARGS qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count
C_GATE(h) C_GATE(x) C_GATE(y) C_GATE(z) C_GATE(s) C_GATE(sdg) C_GATE(t) C_GATE(tdg) C_GATE(sx) C_GATE(sxdg) C_GATE(swap) C_GATE(iswap) C_GATE(id) C_GATE(dcx) C_GATE(ecr) C_GATE(rccx) C_GATE(rcccx)
#define C_ANGLE_GATE(name) C_STATUS(gate_##name, (qcs_simulator *sim, double angle, const bit_num_t *targets, bit_num_t target_count, const bit_num_t *negctrls, bit_num_t negctrl_count, const bit_num_t *ctrls, bit_num_t ctrl_count), qcs_simulator_gate_##name##_cxx(sim, angle, targets, target_count, negctrls, negctrl_count, ctrls, ctrl_count))
C_ANGLE_GATE(rx) C_ANGLE_GATE(ry) C_ANGLE_GATE(rz) C_ANGLE_GATE(u1) C_ANGLE_GATE(p) C_ANGLE_GATE(rxx) C_ANGLE_GATE(ryy) C_ANGLE_GATE(rzz) C_ANGLE_GATE(rzx)
#define C_TWO_ANGLE_GATE(name) C_STATUS(gate_##name, (qcs_simulator *sim, double a, double b, const bit_num_t *targets, bit_num_t target_count, const bit_num_t *negctrls, bit_num_t negctrl_count, const bit_num_t *ctrls, bit_num_t ctrl_count), qcs_simulator_gate_##name##_cxx(sim, a, b, targets, target_count, negctrls, negctrl_count, ctrls, ctrl_count))
C_TWO_ANGLE_GATE(r) C_TWO_ANGLE_GATE(u2) C_TWO_ANGLE_GATE(xx_plus_yy) C_TWO_ANGLE_GATE(xx_minus_yy)
C_STATUS(gate_u3, (qcs_simulator *sim, double a, double b, double c, const bit_num_t *t, bit_num_t tc, const bit_num_t *n, bit_num_t nc, const bit_num_t *x, bit_num_t xc), qcs_simulator_gate_u3_cxx(sim,a,b,c,t,tc,n,nc,x,xc))
C_STATUS(gate_u, (qcs_simulator *sim, double a, double b, double c, const bit_num_t *t, bit_num_t tc, const bit_num_t *n, bit_num_t nc, const bit_num_t *x, bit_num_t xc), qcs_simulator_gate_u_cxx(sim,a,b,c,t,tc,n,nc,x,xc))
C_STATUS(gate_u4, (qcs_simulator *sim, double a, double b, double c, double d, const bit_num_t *t, bit_num_t tc, const bit_num_t *n, bit_num_t nc, const bit_num_t *x, bit_num_t xc), qcs_simulator_gate_u4_cxx(sim,a,b,c,d,t,tc,n,nc,x,xc))
C_STATUS(gate_global_phase, (qcs_simulator *sim, double a, const bit_num_t *n, bit_num_t nc, const bit_num_t *x, bit_num_t xc), qcs_simulator_gate_global_phase_cxx(sim,a,n,nc,x,xc))
C_STATUS(measure, (qcs_simulator *sim, bit_num_t q, bit_t *out), write_result(out, static_cast<bit_t>(qcs_simulator_measure_cxx(sim,q))))
C_STATUS(measure_to_clbit, (qcs_simulator *sim, bit_num_t q, bit_num_t c, bit_t *out), write_result(out, static_cast<bit_t>(qcs_simulator_measure_to_clbit_cxx(sim,q,c))))
C_STATUS(read, (qcs_simulator *sim, bit_num_t c, bit_t *out), write_result(out, static_cast<bit_t>(qcs_simulator_read_cxx(sim,c))))
C_STATUS(save_statevector, (qcs_simulator *sim, const char *p), qcs_simulator_save_statevector_cxx(sim,p))
C_STATUS(event_create, (qcs_simulator *sim, bit_t *out), write_result(out, static_cast<bit_t>(qcs_simulator_event_create_cxx(sim))))
C_STATUS(event_record, (qcs_simulator *sim, int e), qcs_simulator_event_record_cxx(sim,e))
C_STATUS(event_get_elapsed_time, (qcs_simulator *sim, event_num_t a, event_num_t b, double *out), write_result(out, qcs_simulator_event_get_elapsed_time_cxx(sim,a,b)))
