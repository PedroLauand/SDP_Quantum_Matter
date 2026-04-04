"""Shared helpers for the noisy-W GNME examples."""

from __future__ import annotations

import numpy as np

from state_example_common import (
    build_draft,
    partial_trace_qubits,
    print_result,
    solve_min_noise_level as solve_state_min_noise_level,
)


def w_density_matrix() -> np.ndarray:
    ket = np.zeros(8, dtype=complex)
    ket[1] = 1.0 / np.sqrt(3.0)
    ket[2] = 1.0 / np.sqrt(3.0)
    ket[4] = 1.0 / np.sqrt(3.0)
    return np.outer(ket, ket.conj())


def w_known_targets() -> dict[tuple[str, ...], np.ndarray]:
    rho_abc = w_density_matrix()
    return {
        ("A_11", "B_11"): partial_trace_qubits(rho_abc, keep_positions=(0, 1)),
        ("A_11", "C_11"): partial_trace_qubits(rho_abc, keep_positions=(0, 2)),
        ("B_11", "C_11"): partial_trace_qubits(rho_abc, keep_positions=(1, 2)),
        ("A_11", "B_11", "C_11"): rho_abc,
    }


def solve_min_noise_level(draft, Hermitian: bool, verbose: int):
    return solve_state_min_noise_level(
        draft,
        w_known_targets(),
        state_label="W",
        Hermitian=Hermitian,
        verbose=verbose,
    )
