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
    lib.qcs_simulator_set_num_qubits.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_set_num_qubits.restype = None
    lib.qcs_simulator_set_num_clbits.argtypes = [sim, ctypes.c_int]
    lib.qcs_simulator_set_num_clbits.restype = None
    lib.qcs_simulator_allocate_memory.argtypes = [sim]
    lib.qcs_simulator_allocate_memory.restype = None
    lib.qcs_simulator_warmup_nccl_communication.argtypes = [sim]
    lib.qcs_simulator_warmup_nccl_communication.restype = None
    lib.qcs_simulator_get_proc_num.argtypes = [sim]
    lib.qcs_simulator_get_proc_num.restype = ctypes.c_int
    lib.qcs_simulator_get_num_procs.argtypes = [sim]
    lib.qcs_simulator_get_num_procs.restype = ctypes.c_int
    lib.qcs_simulator_get_num_qubits.argtypes = [sim]
    lib.qcs_simulator_get_num_qubits.restype = ctypes.c_int
    lib.qcs_simulator_measure_to_clbit.argtypes = [sim, ctypes.c_int, ctypes.c_int]
    lib.qcs_simulator_measure_to_clbit.restype = ctypes.c_int
    lib.qcs_simulator_get_clbits_string.argtypes = [sim, ctypes.c_char_p]
    lib.qcs_simulator_get_clbits_string.restype = None
    lib.qcs_simulator_save_statevector.argtypes = [sim, ctypes.c_char_p]
    lib.qcs_simulator_save_statevector.restype = None
    lib.qcs_simulator_set_zero_state.argtypes = [sim]
    lib.qcs_simulator_set_zero_state.restype = None
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
    lib.qcs_simulator_event_get_elapsed_time.argtypes = [sim, ctypes.c_int, ctypes.c_int]
    lib.qcs_simulator_event_get_elapsed_time.restype = ctypes.c_double

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
    ):
        fn = getattr(lib, f"qcs_simulator_gate_{name}")
        fn.argtypes = gate_args
        fn.restype = None


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
        self._lib.qcs_simulator_warmup_nccl_communication(self._sim)

    @property
    def proc_num(self) -> int:
        return self._lib.qcs_simulator_get_proc_num(self._sim)

    @property
    def num_procs(self) -> int:
        return self._lib.qcs_simulator_get_num_procs(self._sim)

    def gate(
        self,
        name: str,
        targets: Iterable[int],
        controls: Iterable[int] = (),
        negative_controls: Iterable[int] = (),
    ) -> None:
        fn = getattr(self._lib, f"qcs_simulator_gate_{name}")
        target_keepalive, target_ptr, target_count = _int_array(list(targets))
        neg_keepalive, neg_ptr, neg_count = _int_array(list(negative_controls))
        ctrl_keepalive, ctrl_ptr, ctrl_count = _int_array(list(controls))
        fn(self._sim, target_ptr, target_count, neg_ptr, neg_count, ctrl_ptr, ctrl_count)
        _ = (target_keepalive, neg_keepalive, ctrl_keepalive)

    def h(self, qubit: int) -> None:
        self.gate("h", [qubit])

    def x(self, qubit: int, controls: Iterable[int] = ()) -> None:
        self.gate("x", [qubit], controls=controls)

    def measure_to_clbit(self, qubit: int, clbit: int) -> int:
        return self._lib.qcs_simulator_measure_to_clbit(self._sim, qubit, clbit)

    def clbits_string(self) -> str:
        buffer = ctypes.create_string_buffer(self._num_clbits + 1)
        self._lib.qcs_simulator_get_clbits_string(self._sim, buffer)
        return buffer.value.decode("ascii")

    def save_statevector(self, filename: Union[str, os.PathLike]) -> None:
        self._lib.qcs_simulator_save_statevector(self._sim, os.fsencode(filename))

    def set_zero_state(self) -> None:
        self._lib.qcs_simulator_set_zero_state(self._sim)

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

    def event_get_elapsed_time(self, start_event_num: int, stop_event_num: int) -> float:
        return self._lib.qcs_simulator_event_get_elapsed_time(
            self._sim, start_event_num, stop_event_num
        )

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
