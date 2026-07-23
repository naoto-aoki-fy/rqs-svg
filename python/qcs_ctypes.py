"""ctypes bindings for the RQS-SVG ``libqcs.so`` shared library."""

import ctypes
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

_C_INT_P = ctypes.POINTER(ctypes.c_int)
_NULL_INT_P = ctypes.cast(None, _C_INT_P)


class QcsError(RuntimeError):
    """Raised when the shared library cannot be loaded or initialized."""


def _default_library_candidates() -> List[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    return [
        repo_root / "libqcs.so",
        here.parent / "libqcs.so",
        Path.cwd() / "libqcs.so",
    ]


def find_library_path() -> Path:
    """Return the first usable libqcs.so path.

    Set ``QCS_LIBRARY_PATH`` to override the default lookup locations.
    """

    override = os.environ.get("QCS_LIBRARY_PATH")
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return path
        raise QcsError(f"QCS_LIBRARY_PATH does not exist: {path}")

    for candidate in _default_library_candidates():
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in _default_library_candidates())
    raise QcsError(
        "libqcs.so was not found. Build it with `make sharedlibrary` or set "
        f"QCS_LIBRARY_PATH. Searched: {searched}"
    )


def _int_array(
    values: Optional[Sequence[int]],
) -> Tuple[Optional[ctypes.Array], _C_INT_P, int]:
    if values is None:
        return None, _NULL_INT_P, 0
    array = (ctypes.c_int * len(values))(*values)
    return array, ctypes.cast(array, _C_INT_P), len(values)


def _configure_library(lib: ctypes.CDLL) -> None:
    sim = ctypes.c_void_p
    lib.qcs_simulator_create.argtypes = []
    lib.qcs_simulator_create.restype = sim
    lib.qcs_simulator_destroy.argtypes = [sim]
    lib.qcs_simulator_destroy.restype = None
    lib.qcs_simulator_init.argtypes = [sim]
    lib.qcs_simulator_init.restype = None
    lib.qcs_simulator_setup.argtypes = [sim]
    lib.qcs_simulator_setup.restype = None
    lib.qcs_simulator_allocate_memory.argtypes = [sim]
    lib.qcs_simulator_allocate_memory.restype = None
    lib.qcs_simulator_dispose.argtypes = [sim]
    lib.qcs_simulator_dispose.restype = None
    lib.qcs_simulator_set_num_qubits.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_set_num_qubits.restype = None
    lib.qcs_simulator_set_num_clbits.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_set_num_clbits.restype = None
    lib.qcs_simulator_set_mapping.argtypes = [sim, _C_INT_P, ctypes.c_int]
    lib.qcs_simulator_set_mapping.restype = None
    lib.qcs_simulator_get_proc_num.argtypes = [sim]
    lib.qcs_simulator_get_proc_num.restype = ctypes.c_int
    lib.qcs_simulator_get_num_procs.argtypes = [sim]
    lib.qcs_simulator_get_num_procs.restype = ctypes.c_int
    lib.qcs_simulator_get_num_qubits.argtypes = [sim]
    lib.qcs_simulator_get_num_qubits.restype = ctypes.c_int
    lib.qcs_simulator_get_clbits.argtypes = [sim, _C_INT_P]
    lib.qcs_simulator_get_clbits.restype = None
    lib.qcs_simulator_measure.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_measure.restype = ctypes.c_int
    lib.qcs_simulator_measure_to_clbit.argtypes = [sim, ctypes.c_int, ctypes.c_int]
    lib.qcs_simulator_measure_to_clbit.restype = ctypes.c_int
    lib.qcs_simulator_read.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_read.restype = ctypes.c_int
    lib.qcs_simulator_get_clbits_string.argtypes = [sim, ctypes.c_char_p]
    lib.qcs_simulator_get_clbits_string.restype = None
    lib.qcs_simulator_save_statevector.argtypes = [sim, ctypes.c_char_p]
    lib.qcs_simulator_save_statevector.restype = None
    lib.qcs_simulator_reset.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_reset.restype = None
    lib.qcs_simulator_set_zero_state.argtypes = [sim]
    lib.qcs_simulator_set_zero_state.restype = None
    lib.qcs_simulator_set_sequential_state.argtypes = [sim]
    lib.qcs_simulator_set_sequential_state.restype = None
    lib.qcs_simulator_set_flat_state.argtypes = [sim]
    lib.qcs_simulator_set_flat_state.restype = None
    lib.qcs_simulator_set_entangled_state.argtypes = [sim]
    lib.qcs_simulator_set_entangled_state.restype = None
    lib.qcs_simulator_set_random_state.argtypes = [sim]
    lib.qcs_simulator_set_random_state.restype = None
    lib.qcs_simulator_reset_clbits.argtypes = [sim]
    lib.qcs_simulator_reset_clbits.restype = None
    lib.qcs_simulator_reset_measurement_state.argtypes = [sim]
    lib.qcs_simulator_reset_measurement_state.restype = None
    lib.qcs_simulator_reinitialize_mapping.argtypes = [sim]
    lib.qcs_simulator_reinitialize_mapping.restype = None
    lib.qcs_simulator_event_create.argtypes = [sim]
    lib.qcs_simulator_event_create.restype = ctypes.c_int
    lib.qcs_simulator_event_record.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_event_record.restype = None
    lib.qcs_simulator_event_get_elapsed_time.argtypes = [
        sim,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.qcs_simulator_event_get_elapsed_time.restype = ctypes.c_double
    lib.qcs_simulator_fprintf_master.argtypes = [sim, ctypes.c_void_p, ctypes.c_char_p]
    lib.qcs_simulator_fprintf_master.restype = ctypes.c_int
    lib.qcs_simulator_fprintf_all.argtypes = [sim, ctypes.c_void_p, ctypes.c_char_p]
    lib.qcs_simulator_fprintf_all.restype = ctypes.c_int
    lib.qcs_simulator_fflush_master.argtypes = [sim, ctypes.c_void_p]
    lib.qcs_simulator_fflush_master.restype = ctypes.c_int
    lib.qcs_simulator_fflush_all.argtypes = [sim, ctypes.c_void_p]
    lib.qcs_simulator_fflush_all.restype = ctypes.c_int

    gate_args = [
        sim,
        _C_INT_P,
        ctypes.c_int,
        _C_INT_P,
        ctypes.c_int,
        _C_INT_P,
        ctypes.c_int,
    ]
    for name in (
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "swap",
        "iswap",
        "id",
        "dcx",
        "ecr",
        "rccx",
        "rcccx",
    ):
        fn = getattr(lib, f"qcs_simulator_gate_{name}")
        fn.argtypes = gate_args
        fn.restype = None

    lib.qcs_simulator_gate_global_phase.argtypes = [
        sim,
        ctypes.c_double,
        _C_INT_P,
        ctypes.c_int,
        _C_INT_P,
        ctypes.c_int,
    ]
    lib.qcs_simulator_gate_global_phase.restype = None

    gate_1_double_args = [sim, ctypes.c_double, *gate_args[1:]]
    for name in ("rx", "ry", "rz", "u1", "p", "rxx", "ryy", "rzz", "rzx"):
        fn = getattr(lib, f"qcs_simulator_gate_{name}")
        fn.argtypes = gate_1_double_args
        fn.restype = None

    gate_2_double_args = [sim, ctypes.c_double, ctypes.c_double, *gate_args[1:]]
    for name in ("r", "xx_plus_yy", "xx_minus_yy"):
        fn = getattr(lib, f"qcs_simulator_gate_{name}")
        fn.argtypes = gate_2_double_args
        fn.restype = None

    gate_3_double_args = [
        sim,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        *gate_args[1:],
    ]
    for name in ("u", "u3"):
        fn = getattr(lib, f"qcs_simulator_gate_{name}")
        fn.argtypes = gate_3_double_args
        fn.restype = None

    lib.qcs_simulator_gate_u2.argtypes = [
        sim,
        ctypes.c_double,
        ctypes.c_double,
        *gate_args[1:],
    ]
    lib.qcs_simulator_gate_u2.restype = None
    lib.qcs_simulator_gate_u4.argtypes = [
        sim,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        *gate_args[1:],
    ]
    lib.qcs_simulator_gate_u4.restype = None


class Simulator:
    """Small Python wrapper around a ``qcs_simulator*`` from ``libqcs.so``."""

    def __init__(
        self,
        num_qubits: int,
        num_clbits: Optional[int] = None,
        library_path: Optional[Union[str, os.PathLike]] = None,
    ):
        lib_path = Path(library_path) if library_path else find_library_path()
        self._lib = ctypes.CDLL(str(lib_path))
        _configure_library(self._lib)
        self._sim = self._lib.qcs_simulator_create()
        if not self._sim:
            raise QcsError("qcs_simulator_create returned NULL")
        self._closed = False
        self._num_clbits = num_qubits if num_clbits is None else num_clbits
        self._lib.qcs_simulator_set_num_qubits(self._sim, num_qubits)
        self._lib.qcs_simulator_set_num_clbits(self._sim, self._num_clbits)
        self._lib.qcs_simulator_allocate_memory(self._sim)

    @property
    def proc_num(self) -> int:
        return self._lib.qcs_simulator_get_proc_num(self._sim)

    @property
    def num_procs(self) -> int:
        return self._lib.qcs_simulator_get_num_procs(self._sim)

    @property
    def num_qubits(self) -> int:
        return self._lib.qcs_simulator_get_num_qubits(self._sim)

    def init(self) -> None:
        self._lib.qcs_simulator_init(self._sim)

    def setup(self) -> None:
        self._lib.qcs_simulator_setup(self._sim)

    def allocate_memory(self) -> None:
        self._lib.qcs_simulator_allocate_memory(self._sim)

    def dispose(self) -> None:
        self._lib.qcs_simulator_dispose(self._sim)

    def set_mapping(self, perm_p2l: Sequence[int]) -> None:
        keepalive, ptr, count = _int_array(perm_p2l)
        self._lib.qcs_simulator_set_mapping(self._sim, ptr, count)
        _ = keepalive

    def gate(
        self,
        name: str,
        targets: Iterable[int],
        controls: Iterable[int] = (),
        negative_controls: Iterable[int] = (),
        *parameters: float,
    ) -> None:
        fn = getattr(self._lib, f"qcs_simulator_gate_{name}")
        target_keepalive, target_ptr, target_count = _int_array(list(targets))
        neg_keepalive, neg_ptr, neg_count = _int_array(list(negative_controls))
        ctrl_keepalive, ctrl_ptr, ctrl_count = _int_array(list(controls))
        fn(
            self._sim,
            *parameters,
            target_ptr,
            target_count,
            neg_ptr,
            neg_count,
            ctrl_ptr,
            ctrl_count,
        )
        _ = (target_keepalive, neg_keepalive, ctrl_keepalive)

    def global_phase(
        self,
        theta: float,
        controls: Iterable[int] = (),
        negative_controls: Iterable[int] = (),
    ) -> None:
        fn = self._lib.qcs_simulator_gate_global_phase
        neg_keepalive, neg_ptr, neg_count = _int_array(list(negative_controls))
        ctrl_keepalive, ctrl_ptr, ctrl_count = _int_array(list(controls))
        fn(self._sim, theta, neg_ptr, neg_count, ctrl_ptr, ctrl_count)
        _ = (neg_keepalive, ctrl_keepalive)

    def h(self, qubit: int) -> None:
        self.gate("h", [qubit])

    def x(self, qubit: int, controls: Iterable[int] = ()) -> None:
        self.gate("x", [qubit], controls=controls)

    def measure_to_clbit(self, qubit: int, clbit: int) -> int:
        return self._lib.qcs_simulator_measure_to_clbit(self._sim, qubit, clbit)

    def measure(self, qubit: int) -> int:
        return self._lib.qcs_simulator_measure(self._sim, qubit)

    def read(self, clbit: int) -> int:
        return self._lib.qcs_simulator_read(self._sim, clbit)

    def reset(self, qubit: int) -> None:
        self._lib.qcs_simulator_reset(self._sim, qubit)

    def clbits(self) -> List[int]:
        buffer = (ctypes.c_int * self._num_clbits)()
        self._lib.qcs_simulator_get_clbits(self._sim, buffer)
        return list(buffer)

    def clbits_string(self) -> str:
        buffer = ctypes.create_string_buffer(self._num_clbits + 1)
        self._lib.qcs_simulator_get_clbits_string(self._sim, buffer)
        return buffer.value.decode("ascii")

    def save_statevector(self, filename: Union[str, os.PathLike]) -> None:
        self._lib.qcs_simulator_save_statevector(self._sim, os.fsencode(filename))

    def set_zero_state(self) -> None:
        self._lib.qcs_simulator_set_zero_state(self._sim)

    def set_sequential_state(self) -> None:
        self._lib.qcs_simulator_set_sequential_state(self._sim)

    def set_flat_state(self) -> None:
        self._lib.qcs_simulator_set_flat_state(self._sim)

    def set_entangled_state(self) -> None:
        self._lib.qcs_simulator_set_entangled_state(self._sim)

    def set_random_state(self) -> None:
        self._lib.qcs_simulator_set_random_state(self._sim)

    def reset_clbits(self) -> None:
        self._lib.qcs_simulator_reset_clbits(self._sim)

    def reset_measurement_state(self) -> None:
        self._lib.qcs_simulator_reset_measurement_state(self._sim)

    def reinitialize_mapping(self) -> None:
        self._lib.qcs_simulator_reinitialize_mapping(self._sim)

    def reset_for_next_sample(self) -> None:
        self.reinitialize_mapping()
        self.set_zero_state()
        self.reset_clbits()
        self.reset_measurement_state()

    def event_create(self) -> int:
        return self._lib.qcs_simulator_event_create(self._sim)

    def event_record(self, event_num: int) -> None:
        self._lib.qcs_simulator_event_record(self._sim, event_num)

    def event_get_elapsed_time(
        self, start_event_num: int, stop_event_num: int
    ) -> float:
        return self._lib.qcs_simulator_event_get_elapsed_time(
            self._sim, start_event_num, stop_event_num
        )

    def fflush_master(self, stream: Optional[int] = None) -> int:
        stream_ptr = (
            ctypes.c_void_p(stream) if stream is not None else ctypes.c_void_p()
        )
        return self._lib.qcs_simulator_fflush_master(self._sim, stream_ptr)

    def fflush_all(self, stream: Optional[int] = None) -> int:
        stream_ptr = (
            ctypes.c_void_p(stream) if stream is not None else ctypes.c_void_p()
        )
        return self._lib.qcs_simulator_fflush_all(self._sim, stream_ptr)

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            self._lib.qcs_simulator_destroy(self._sim)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
