#!/usr/bin/env python3
"""Create and measure a GHZ circuit through the Python ctypes binding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from qcs_ctypes import Simulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=3)
    parser.add_argument(
        "--library",
        help="Path to libqcs.so; defaults to QCS_LIBRARY_PATH or repo root",
    )
    args = parser.parse_args()

    with Simulator(args.num_qubits, library_path=args.library) as sim:
        sim.h(0)
        for qubit in range(1, args.num_qubits):
            sim.x(qubit, controls=[0])
        for qubit in range(args.num_qubits):
            sim.measure_to_clbit(qubit, qubit)
        if sim.proc_num == 0:
            print(f"measured clbits: {sim.clbits_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
