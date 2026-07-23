#!/usr/bin/env python3
"""Execute Qiskit ``QuantumCircuit`` objects directly with RQS-SVG.

This standalone module includes its own Qiskit gate and input-file resolution
and calls ``qcs_ctypes.Simulator`` without generating or compiling C code.

``qcs_ctypes.py`` must be importable (for example, add the ``rqs-svg/python``
directory to ``PYTHONPATH``), and ``libqcs.so`` must be discoverable by that
module or supplied through ``library_path``.
"""

from __future__ import annotations

from qcs_ctypes import Simulator

import argparse
import json
import runpy
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

from qiskit import qasm2, qasm3, qpy
from qiskit.circuit import (
    AnnotatedOperation,
    Barrier,
    ClassicalRegister,
    ControlledGate,
    Gate,
    Measure,
    QuantumCircuit,
    Reset,
)
from qiskit.circuit.annotated_operation import ControlModifier
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.circuit.library.standard_gates.u import CUGate, UGate
from qiskit.circuit.controlflow import ForLoopOp, IfElseOp, WhileLoopOp


STANDARD_GATE_TYPE_NAME_PAIRS = [
    (type(gate), gate_name)
    for gate_name, gate in get_standard_gate_name_mapping().items()
]


def get_base_gate_name(operation) -> str:
    """Return the RQS-SVG gate name corresponding to a Qiskit operation."""
    if isinstance(operation, CUGate):
        return "u4"
    if isinstance(operation, UGate):
        return "u3"

    base_gate = operation
    while hasattr(base_gate, "base_gate"):
        base_gate = base_gate.base_gate

    for gate_type, gate_name in STANDARD_GATE_TYPE_NAME_PAIRS:
        if isinstance(base_gate, gate_type):
            return gate_name

    if isinstance(base_gate, Gate):
        raise ValueError(
            "Unsupported gate type for base-name resolution: "
            f"{type(base_gate)!r}"
        )

    raise ValueError(
        "Operation does not resolve to a Gate instance: "
        f"{type(base_gate)!r}"
    )


def get_num_ctrl_qubits(operation) -> int:
    """Return the number of control qubits represented by an operation."""
    if isinstance(operation, ControlledGate):
        return operation.num_ctrl_qubits

    if isinstance(operation, AnnotatedOperation):
        return sum(
            modifier.num_ctrl_qubits
            for modifier in operation.modifiers
            if isinstance(modifier, ControlModifier)
        )

    return 0


def load_python_circuit(path: Union[str, PathLike[str]]) -> QuantumCircuit:
    """Load the first QuantumCircuit exposed by a Python source file."""
    namespace = runpy.run_path(str(path))

    for preferred_name in ("qc", "circuit"):
        circuit = namespace.get(preferred_name)
        if isinstance(circuit, QuantumCircuit):
            return circuit

    for name, value in namespace.items():
        if not name.startswith("_") and isinstance(value, QuantumCircuit):
            return value

    raise ValueError(
        "Circuit file must define a QuantumCircuit, preferably as 'qc' or "
        "'circuit'."
    )


def load_qpy_circuit(path: Union[str, PathLike[str]]) -> QuantumCircuit:
    """Load the first circuit from a QPY file."""
    with open(path, "rb") as qpy_file:
        circuits = qpy.load(qpy_file)

    if not circuits:
        raise ValueError("QPY file does not contain any circuits.")
    return circuits[0]


def _rewrite_qasm_for_retry(qasm_text: str) -> str:
    return (
        qasm_text.replace("OPENQASM 2.0;", "OPENQASM 3.0;")
        .replace('include "qelib1.inc";', 'include "stdgates.inc";')
    )


def load_qasm_circuit(path: Union[str, PathLike[str]]) -> QuantumCircuit:
    """Load an OpenQASM 2 or OpenQASM 3 circuit."""
    qasm_text = Path(path).read_text(encoding="utf-8")

    for loader in (qasm3.loads, qasm2.loads):
        try:
            return loader(qasm_text)
        except Exception:
            pass

    rewritten_qasm = _rewrite_qasm_for_retry(qasm_text)
    if rewritten_qasm != qasm_text:
        for loader in (qasm3.loads, qasm2.loads):
            try:
                return loader(rewritten_qasm)
            except Exception:
                pass

    raise ValueError(
        "Failed to load QASM via qiskit.qasm3/qiskit.qasm2, including retry "
        "with OPENQASM/stdgates replacements."
    )


def load_circuit(path: Union[str, PathLike[str]]) -> QuantumCircuit:
    """Load a QuantumCircuit from Python, QPY, or QASM input."""
    suffix = Path(path).suffix.lower()
    if suffix == ".qpy":
        return load_qpy_circuit(path)
    if suffix in {".qasm", ".qasm2", ".qasm3"}:
        return load_qasm_circuit(path)
    return load_python_circuit(path)


@dataclass(frozen=True)
class CircuitRunResult:
    """Execution result including per-shot device timings in seconds."""

    counts: dict[str, int]
    shot_times_seconds: list[float]
    proc_num: int

    @property
    def total_time_seconds(self) -> float:
        return sum(self.shot_times_seconds)

    @property
    def average_time_seconds(self) -> float:
        if not self.shot_times_seconds:
            return 0.0
        return self.total_time_seconds / len(self.shot_times_seconds)


@dataclass(frozen=True)
class ShotRunResult:
    """Result produced immediately after one circuit shot completes."""

    sample_num: int
    clbits: str
    elapsed_time: float
    proc_num: int


try:
    from qcs_ctypes import Simulator
except ImportError as exc:  # pragma: no cover - depends on external repository
    raise ImportError(
        "qcs_ctypes.py could not be imported. Add the rqs-svg/python directory "
        "to PYTHONPATH, for example: "
        "PYTHONPATH=/path/to/rqs-svg/python:$PYTHONPATH"
    ) from exc


def _condition_value(condition, qc: QuantumCircuit, sim: Simulator) -> bool:
    """Evaluate a Qiskit legacy control-flow condition using simulator clbits."""
    bits, expected = condition
    bits_list = list(bits) if isinstance(bits, ClassicalRegister) else [bits]
    actual = 0
    for offset, bit in enumerate(bits_list):
        actual |= sim.read(qc.find_bit(bit).index) << offset
    return actual == int(expected)


def _float_parameters(parameters: Iterable[object]) -> list[float]:
    values: list[float] = []
    for parameter in parameters:
        try:
            values.append(float(parameter))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unbound or non-numeric gate parameter: {parameter!r}. "
                "Bind all circuit parameters before execution."
            ) from exc
    return values


def _split_controls(operation, qubit_numbers: Sequence[int]):
    num_controls = get_num_ctrl_qubits(operation)
    all_controls = qubit_numbers[:num_controls]
    targets = qubit_numbers[num_controls:]

    ctrl_state = getattr(operation, "ctrl_state", None)
    positive_controls: list[int] = []
    negative_controls: list[int] = []
    if ctrl_state is None:
        positive_controls.extend(all_controls)
    else:
        for offset, qubit in enumerate(all_controls):
            if (int(ctrl_state) >> offset) & 1:
                positive_controls.append(qubit)
            else:
                negative_controls.append(qubit)

    return targets, positive_controls, negative_controls


def _execute_instructions(
    instructions,
    qc: QuantumCircuit,
    sim: Simulator,
    *,
    max_while_iterations: int,
) -> None:
    for instruction in instructions:
        operation = instruction.operation
        qubits = tuple(qc.find_bit(bit).index for bit in instruction.qubits)
        clbits = tuple(qc.find_bit(bit).index for bit in instruction.clbits)

        if isinstance(operation, Measure):
            for qubit, clbit in zip(qubits, clbits):
                sim.measure_to_clbit(qubit, clbit)
            continue

        if isinstance(operation, Reset):
            for qubit in qubits:
                sim.reset(qubit)
            continue

        if isinstance(operation, Barrier):
            continue

        if isinstance(operation, IfElseOp):
            block_index = 0 if _condition_value(operation.condition, qc, sim) else 1
            if (
                block_index < len(operation.blocks)
                and operation.blocks[block_index] is not None
            ):
                _execute_instructions(
                    operation.blocks[block_index].data,
                    qc,
                    sim,
                    max_while_iterations=max_while_iterations,
                )
            continue

        if isinstance(operation, WhileLoopOp):
            iteration = 0
            while _condition_value(operation.condition, qc, sim):
                if iteration >= max_while_iterations:
                    raise RuntimeError(
                        "WhileLoopOp exceeded max_while_iterations="
                        f"{max_while_iterations}"
                    )
                _execute_instructions(
                    operation.blocks[0].data,
                    qc,
                    sim,
                    max_while_iterations=max_while_iterations,
                )
                iteration += 1
            continue

        if isinstance(operation, ForLoopOp):
            sequence, loop_parameter = operation.params[:2]
            for value in sequence:
                block = operation.blocks[0]
                if loop_parameter is not None:
                    block = block.assign_parameters(
                        {loop_parameter: value}, inplace=False
                    )
                _execute_instructions(
                    block.data,
                    qc,
                    sim,
                    max_while_iterations=max_while_iterations,
                )
            continue

        gate_name = get_base_gate_name(operation)
        targets, controls, negative_controls = _split_controls(operation, qubits)
        parameters = _float_parameters(operation.params)
        sim.gate(
            gate_name,
            targets,
            controls,
            negative_controls,
            *parameters,
        )


def _qiskit_bitstring(sim: Simulator, num_clbits: int) -> str:
    """Return a Qiskit-style count key (highest classical-bit index first)."""
    return "".join(str(bit) for bit in reversed(sim.clbits()[:num_clbits]))


def _validate_run_arguments(
    qc: QuantumCircuit,
    shots: int,
    max_while_iterations: int,
) -> None:
    if not isinstance(qc, QuantumCircuit):
        raise TypeError(f"qc must be QuantumCircuit, got {type(qc)!r}")
    if shots <= 0:
        raise ValueError("shots must be greater than zero")
    if max_while_iterations <= 0:
        raise ValueError("max_while_iterations must be greater than zero")
    if qc.num_clbits == 0:
        raise ValueError("The circuit has no classical bits; counts cannot be returned.")


def iter_circuit_shots(
    qc: QuantumCircuit,
    shots: int = 1,
    *,
    library_path: Optional[Union[str, PathLike[str]]] = None,
    max_while_iterations: int = 1_000_000,
) -> Iterator[ShotRunResult]:
    """Yield a result immediately after each shot of ``qc`` completes.

    ``elapsed_time`` is the value returned by
    ``Simulator.event_get_elapsed_time`` for that shot. The measured interval
    includes global-phase application, all circuit instructions, and
    measurements. Resetting for the next shot is outside the interval.

    The returned iterator owns the simulator. It must be fully consumed or
    explicitly closed so the simulator context is released promptly.
    """
    _validate_run_arguments(qc, shots, max_while_iterations)

    with Simulator(qc.num_qubits, qc.num_clbits, library_path=library_path) as sim:
        start_event = sim.event_create()
        stop_event = sim.event_create()

        for sample_num in range(shots):
            sim.event_record(start_event)

            if qc.global_phase:
                sim.global_phase(float(qc.global_phase))
            _execute_instructions(
                qc.data,
                qc,
                sim,
                max_while_iterations=max_while_iterations,
            )

            sim.event_record(stop_event)
            yield ShotRunResult(
                sample_num=sample_num,
                clbits=_qiskit_bitstring(sim, qc.num_clbits),
                elapsed_time=float(
                    sim.event_get_elapsed_time(start_event, stop_event)
                ),
                proc_num=sim.proc_num,
            )

            if sample_num + 1 < shots:
                sim.reset_for_next_sample()


def run_circuit(
    qc: QuantumCircuit,
    shots: int = 1,
    *,
    library_path: Optional[Union[str, PathLike[str]]] = None,
    max_while_iterations: int = 1_000_000,
) -> CircuitRunResult:
    """Run ``qc`` and aggregate results yielded by ``iter_circuit_shots``."""
    counts: Counter[str] = Counter()
    shot_times_seconds: list[float] = []
    proc_num = 0

    for shot_result in iter_circuit_shots(
        qc,
        shots=shots,
        library_path=library_path,
        max_while_iterations=max_while_iterations,
    ):
        counts[shot_result.clbits] += 1
        shot_times_seconds.append(shot_result.elapsed_time)
        proc_num = shot_result.proc_num

    return CircuitRunResult(
        counts=dict(counts),
        shot_times_seconds=shot_times_seconds,
        proc_num=proc_num,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a Qiskit circuit directly using rqs-svg qcs_ctypes"
    )
    parser.add_argument("circuit_file", help="Input Python, QPY, or QASM file")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--library", help="Explicit path to libqcs.so")
    parser.add_argument("--max-while-iterations", type=int, default=1_000_000)
    args = parser.parse_args()

    for shot_result in iter_circuit_shots(
        load_circuit(args.circuit_file),
        shots=args.shots,
        library_path=args.library,
        max_while_iterations=args.max_while_iterations,
    ):
        if shot_result.proc_num == 0:
            print(
                json.dumps(
                    {
                        "sample_num": shot_result.sample_num,
                        "clbits": shot_result.clbits,
                        "elapsed_time": shot_result.elapsed_time,
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
