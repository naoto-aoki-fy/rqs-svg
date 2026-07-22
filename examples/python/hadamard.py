#!/usr/bin/env python3
"""Create and measure a circuit applying Hadamard to every qubit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qcs_ctypes import Simulator  # noqa: E402


def run_hadamard_circuit(sim: Simulator, num_qubits: int, measure: bool = False) -> None:
    """Apply Hadamard to every qubit and measure into same-index classical bits."""
    for qubit in range(num_qubits):
        sim.h(qubit)

    if measure:
        for qubit in range(num_qubits):
            sim.measure_to_clbit(qubit, qubit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of times to run and measure the Hadamard circuit.",
    )
    parser.add_argument(
        "--library",
        help="Path to libqcs.so; defaults to QCS_LIBRARY_PATH or repo root",
    )
    args = parser.parse_args()

    if args.num_qubits <= 0:
        parser.error("--num-qubits must be greater than 0")

    if args.num_samples <= 0:
        parser.error("--num-samples must be greater than 0")

    with Simulator(args.num_qubits, library_path=args.library) as sim:
        event_start = sim.event_create()
        event_stop = sim.event_create()

        for sample_num in range(args.num_samples):
            sim.event_record(event_start)
            run_hadamard_circuit(sim, args.num_qubits)
            sim.event_record(event_stop)

            elapsed_time = sim.event_get_elapsed_time(event_start, event_stop)

            if sim.proc_num == 0:
                print(
                    json.dumps(
                        {
                            "sample_num": sample_num,
                            "clbits": sim.clbits_string(),
                            "elapsed_time": elapsed_time,
                        }
                    ),
                    flush=True,
                )

            if sample_num != args.num_samples - 1:
                sim.reset_for_next_sample()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
