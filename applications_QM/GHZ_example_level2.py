"""Level-2 noisy-GHZ optimization with the default real block formulation.

This example uses the default GNME package API with `Hermitian=False`.
It minimizes the white-noise parameter `p` in

    (1 - p) |GHZ><GHZ| + p I / 8.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import GNMEProblem, build_fusion_feasibility_model, build_sdp_draft  # noqa: E402

PROGRESS_VERBOSE = 1
HERMITIAN = False


def ghz_density_matrix() -> np.ndarray:
    ket = np.zeros(8, dtype=complex)
    ket[0] = 1.0 / np.sqrt(2.0)
    ket[7] = 1.0 / np.sqrt(2.0)
    return np.outer(ket, ket.conj())


def partial_trace_qubits(
    rho: np.ndarray,
    keep_positions: tuple[int, ...],
    n_qubits: int = 3,
) -> np.ndarray:
    dims = (2,) * n_qubits
    traced_positions = tuple(pos for pos in range(n_qubits) if pos not in keep_positions)
    tensor = rho.reshape(dims + dims)
    for pos in sorted(traced_positions, reverse=True):
        tensor = np.trace(tensor, axis1=pos, axis2=pos + tensor.ndim // 2)
    kept_dims = (2,) * len(keep_positions)
    return tensor.reshape((np.prod(kept_dims), np.prod(kept_dims)))


def matrix_coordinate_values(matrix: np.ndarray, Hermitian: bool) -> np.ndarray:
    real_values = []
    complex_dim = int(matrix.shape[0])
    for row in range(complex_dim):
        for col in range(row, complex_dim):
            real_values.append(float(np.real(matrix[row, col])))
    if not Hermitian:
        return np.asarray(real_values, dtype=np.float64)
    imag_values = []
    for row in range(complex_dim):
        for col in range(row + 1, complex_dim):
            imag_values.append(float(np.imag(matrix[row, col])))
    return np.asarray(real_values + imag_values, dtype=np.float64)


def ghz_known_targets() -> dict[tuple[str, ...], np.ndarray]:
    rho_abc = ghz_density_matrix()
    return {
        ("A_11", "B_11"): partial_trace_qubits(rho_abc, keep_positions=(0, 1)),
        ("A_11", "C_11"): partial_trace_qubits(rho_abc, keep_positions=(0, 2)),
        ("B_11", "C_11"): partial_trace_qubits(rho_abc, keep_positions=(1, 2)),
        ("A_11", "B_11", "C_11"): rho_abc,
    }


def maximally_mixed_targets() -> dict[tuple[str, ...], np.ndarray]:
    return {
        ("A_11", "B_11"): np.eye(4, dtype=float) / 4.0,
        ("A_11", "C_11"): np.eye(4, dtype=float) / 4.0,
        ("B_11", "C_11"): np.eye(4, dtype=float) / 4.0,
        ("A_11", "B_11", "C_11"): np.eye(8, dtype=float) / 8.0,
    }


def solve_min_noise_level(draft, Hermitian: bool, verbose: int = PROGRESS_VERBOSE):
    from mosek.fusion import AccSolutionStatus, Domain, Expr, ObjectiveSense

    ghz_targets = ghz_known_targets()
    mixed_targets = maximally_mixed_targets()
    known_reps_by_target = {
        tuple(representative.target_lexorder): representative
        for representative in draft.known_representatives
    }

    fusion_model = build_fusion_feasibility_model(
        draft,
        enforce_known_values=False,
        Hermitian=Hermitian,
        verbose=verbose,
    )
    M = fusion_model.model
    p_noise_var = M.variable("p_noise", 1, Domain.inRange(0.0, 1.0))
    p_noise = p_noise_var.index(0)

    constraint_index = 0
    for target_lexorder, ghz_matrix in ghz_targets.items():
        representative = known_reps_by_target[target_lexorder]
        rep_coord = fusion_model.known_representative_coordinate_vectors[representative.name]
        ghz_coord = matrix_coordinate_values(ghz_matrix, Hermitian)
        mixed_coord = matrix_coordinate_values(mixed_targets[target_lexorder], Hermitian)
        delta_coord = mixed_coord - ghz_coord
        for coord_index, (base_value, delta_value) in enumerate(zip(ghz_coord, delta_coord)):
            rhs = Expr.add(float(base_value), Expr.mul(float(delta_value), p_noise))
            M.constraint(
                f"known_affine_{constraint_index}",
                Expr.sub(rep_coord.index(coord_index), rhs),
                Domain.equalsTo(0.0),
            )
            constraint_index += 1

    M.objective(ObjectiveSense.Minimize, p_noise)
    M.acceptedSolutionStatus(AccSolutionStatus.Anything)
    if verbose > 0:
        M.setLogHandler(__import__("sys").stdout)

    solve_start = perf_counter()
    M.solve()
    solve_seconds = perf_counter() - solve_start

    return {
        "problem_status": str(M.getProblemStatus()),
        "primal_status": str(M.getPrimalSolutionStatus()),
        "dual_status": str(M.getDualSolutionStatus()),
        "p_noise": float(p_noise.level()[0]),
        "solve_seconds": solve_seconds,
        "affine_known_constraints": constraint_index,
        "base_constraint_counts": fusion_model.constraint_counts,
    }


if __name__ == "__main__":
    print("[GHZ level2 real] building GNME problem", flush=True)
    problem = GNMEProblem(
        n_parties=3,
        inflation_level=2,
        local_dims_per_party=2,
        verbose=PROGRESS_VERBOSE,
    )

    print("[GHZ level2 real] building SDP draft", flush=True)
    draft = build_sdp_draft(problem)

    print("[GHZ level2 real] solving optimization over p", flush=True)
    result = solve_min_noise_level(draft, Hermitian=HERMITIAN, verbose=PROGRESS_VERBOSE)

    print("GNME noisy-GHZ optimization (level 2)")
    print(f"  Hermitian: {HERMITIAN}")
    print(f"  problem status: {result['problem_status']}")
    print(f"  primal status: {result['primal_status']}")
    print(f"  dual status: {result['dual_status']}")
    print(f"  optimal p_noise: {result['p_noise']}")
    print(f"  solve time: {result['solve_seconds']:.2f}s")
    print(f"  affine known-value constraints: {result['affine_known_constraints']}")
    print(f"  base constraint counts: {result['base_constraint_counts']}")
