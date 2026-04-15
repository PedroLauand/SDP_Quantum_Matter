"""Readable level-2 complex robustness SDP.

This file is meant to read like the mathematical SDP itself.

Variables:

- ``sigma`` : two disconnected triangle copies
- ``tau``   : one connected 6-ring

Slot conventions:

- ``sigma`` lives on ``(A11, A22, B11, B22, C11, C22)``
- ``tau``   lives on ``(A11, A22, B11, B22, C12, C21)``

Optimization problem:

    minimize p

subject to

    Tr_rest(sigma) = (1-p) rho_ABC + p I/D
    0 <= p <= 1

plus the trace-one, symmetry, cross-inflation, and PPT constraints described
inline below.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp

try:
    from level3_scs_backend import (
        DEFAULT_SCS_ACCELERATION_INTERVAL,
        DEFAULT_SCS_ACCELERATION_LOOKBACK,
        DEFAULT_SCS_ALPHA,
        DEFAULT_SCS_RHO_X,
        DEFAULT_SCS_SCALE,
        MatrixVariableSpec,
        SCSAffineConstraintBuilder,
        SCSVectorSlice,
        _trace_coordinate_action,
        cached_coordinate_action_for_slot_permutation,
        cached_partial_trace_coordinate_action,
        cached_partial_transpose_coordinate_action,
        cached_scs_complex_psd_coordinate_action,
        ghz_density_matrix,
        identity_coordinate_action,
        pack_hermitian_coordinates,
        product_dim,
        solve_with_scs_compat,
        unpack_hermitian_coordinates,
    )
except ModuleNotFoundError:
    from Baroque_test_paper.level3_scs_backend import (
        DEFAULT_SCS_ACCELERATION_INTERVAL,
        DEFAULT_SCS_ACCELERATION_LOOKBACK,
        DEFAULT_SCS_ALPHA,
        DEFAULT_SCS_RHO_X,
        DEFAULT_SCS_SCALE,
        MatrixVariableSpec,
        SCSAffineConstraintBuilder,
        SCSVectorSlice,
        _trace_coordinate_action,
        cached_coordinate_action_for_slot_permutation,
        cached_partial_trace_coordinate_action,
        cached_partial_transpose_coordinate_action,
        cached_scs_complex_psd_coordinate_action,
        ghz_density_matrix,
        identity_coordinate_action,
        pack_hermitian_coordinates,
        product_dim,
        solve_with_scs_compat,
        unpack_hermitian_coordinates,
    )


@dataclass
class InflationRobustnessLevel2SCSModel:
    dims3: tuple[int, int, int]
    dims6: tuple[int, ...]
    D: int
    full_dim: int
    A: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    cone: dict[str, object]
    p_slice: SCSVectorSlice
    sigma: MatrixVariableSpec
    tau: MatrixVariableSpec
    ppt_auxiliaries: dict[str, MatrixVariableSpec]
    constraint_counts: dict[str, int]
    build_profile: dict[str, object]


# Slot order for sigma:
#   0   1   2   3   4   5
#   A11 A22 B11 B22 C11 C22
#
# Slot order for tau:
#   0   1   2   3   4   5
#   A11 A22 B11 B22 C12 C21

# Internal symmetries.
# sigma: swap the two disconnected triangles.
# tau:   half-turn of the connected 6-ring.
SIGMA_SWAP = (1, 0, 3, 2, 5, 4)
TAU_SWAP = (1, 0, 3, 2, 5, 4)

# Observed anchor on sigma:
#   Tr_{A22 B22 C22}(sigma) = (1-p) rho_ABC + p I_ABC / D
OBSERVED_ANCHOR_SIGMA_KEEP = (0, 2, 4)

# Cross-inflation equalities.
# 1) ABAB
EQ_SIGMA_TAU_ABAB_SIGMA = (0, 2, 1, 3)  # sigma(A11, B11, A22, B22)
EQ_SIGMA_TAU_ABAB_TAU = (0, 2, 1, 3)    # tau(A11, B11, A22, B22)

# 2) BCBC
EQ_SIGMA_TAU_BCBC_SIGMA = (2, 4, 3, 5)  # sigma(B11, C11, B22, C22)
EQ_SIGMA_TAU_BCBC_TAU = (2, 4, 3, 5)    # tau(B11, C12, B22, C21)

# 3) CACA
EQ_SIGMA_TAU_CACA_SIGMA = (4, 0, 5, 1)  # sigma(C11, A11, C22, A22)
EQ_SIGMA_TAU_CACA_TAU = (4, 1, 5, 0)    # tau(C12, A22, C21, A11)

# PPT on sigma across (A11 B11 C11) | (A22 B22 C22).
PPT_SIGMA_FULL_TRANSPOSE = (0, 2, 4)


def _validate_level2_inputs(
    rho: np.ndarray,
    dims3: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, ...], int, int, np.ndarray]:
    dims3 = tuple(int(dim) for dim in dims3)
    if len(dims3) != 3:
        raise ValueError("dims3 must contain exactly three local dimensions.")
    D = product_dim(dims3)
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (D, D):
        raise ValueError(f"rho must have shape {(D, D)}, got {rho.shape}.")
    if not np.allclose(rho, rho.conj().T, atol=1e-9):
        raise ValueError("rho must be Hermitian.")
    dims6 = (dims3[0], dims3[0], dims3[1], dims3[1], dims3[2], dims3[2])
    full_dim = product_dim(dims6)
    return dims3, dims6, D, full_dim, rho


def _extract_scs_variable_matrix(
    x_values: np.ndarray,
    variable_spec: MatrixVariableSpec,
) -> np.ndarray:
    coords = np.asarray(
        x_values[variable_spec.x_slice.offset:variable_spec.x_slice.stop],
        dtype=np.float64,
    )
    return unpack_hermitian_coordinates(coords, variable_spec.matrix_dim)


def _add_scalar_upper_bound(
    builder: SCSAffineConstraintBuilder,
    scalar_slice: SCSVectorSlice,
    upper_bound: float,
) -> None:
    row_offset = builder._reserve_rows(np.asarray([float(upper_bound)], dtype=np.float64))
    builder.row_blocks.append(np.asarray([row_offset], dtype=np.int32))
    builder.col_blocks.append(np.asarray([int(scalar_slice.offset)], dtype=np.int32))
    builder.val_blocks.append(np.asarray([1.0], dtype=np.float64))
    builder.constraint_counts["nonnegative"] += 1


def build_inflation_robustness_level2_scs_model(
    rho: np.ndarray,
    dims3: tuple[int, int, int],
    *,
    verbose: int = 1,
    optimizer_max_time: float | None = None,
) -> InflationRobustnessLevel2SCSModel:
    """Build the direct SCS cone program for the complex level-2 SDP."""
    del verbose, optimizer_max_time
    dims3, dims6, D, full_dim, rho = _validate_level2_inputs(rho, dims3)
    total_start = perf_counter()

    builder = SCSAffineConstraintBuilder()

    # Scalar optimization variable:
    #   Tr_rest(sigma) = (1-p) rho + p I/D
    p_slice = builder.add_scalar_variable("p")

    # Main level-2 inflated states.
    sigma = builder.add_matrix_variable("sigma", full_dim, full_dim * full_dim)
    tau = builder.add_matrix_variable("tau", full_dim, full_dim * full_dim)

    # Single full PPT auxiliary on sigma.
    ppt_sigma_full = builder.add_matrix_variable("ppt_sigma_full", full_dim, full_dim * full_dim)
    ppt_auxiliaries = {"ppt_sigma_full": ppt_sigma_full}

    # 1) Trace-one constraints:
    #   Tr(sigma) = Tr(tau) = 1
    full_trace_action = _trace_coordinate_action(full_dim)
    builder.add_trace_one(sigma, full_trace_action)
    builder.add_trace_one(tau, full_trace_action)

    # 2) Internal symmetries.
    # sigma(A11 B11 C11, A22 B22 C22) = sigma(A22 B22 C22, A11 B11 C11)
    builder.add_difference_equality(
        sigma,
        cached_coordinate_action_for_slot_permutation(dims6, SIGMA_SWAP),
        sigma,
        identity_coordinate_action(sigma.coord_dim),
        counter_key="symmetry",
    )
    # tau(A11 A22 B11 B22 C12 C21) = tau(A22 A11 B22 B11 C21 C12)
    builder.add_difference_equality(
        tau,
        cached_coordinate_action_for_slot_permutation(dims6, TAU_SWAP),
        tau,
        identity_coordinate_action(tau.coord_dim),
        counter_key="symmetry",
    )

    # 3) Observed ABC anchor on sigma:
    #   Tr_{A22 B22 C22}(sigma) = (1-p) rho_ABC + p I_ABC / D
    rho_coord = pack_hermitian_coordinates(rho)
    mixed_coord = pack_hermitian_coordinates(np.eye(D, dtype=np.complex128) / float(D))
    builder.add_observed_anchor(
        sigma,
        cached_partial_trace_coordinate_action(dims6, OBSERVED_ANCHOR_SIGMA_KEEP),
        rho_coord.astype(np.float64, copy=False),
        p_slice,
        (rho_coord - mixed_coord).astype(np.float64, copy=False),
    )

    # 4) Cross-inflation consistency equalities.
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_ABAB_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_ABAB_TAU),
        counter_key="equality",
    )
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_BCBC_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_BCBC_TAU),
        counter_key="equality",
    )
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_CACA_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims6, EQ_SIGMA_TAU_CACA_TAU),
        counter_key="equality",
    )

    # 5) PPT on sigma across (A11 B11 C11) | (A22 B22 C22).
    builder.add_auxiliary_link(
        ppt_sigma_full,
        sigma,
        cached_partial_transpose_coordinate_action(dims6, PPT_SIGMA_FULL_TRANSPOSE),
    )

    # 6) Linear bounds and PSD cone constraints.
    zero_rows = int(builder.next_row)
    builder.add_scalar_nonnegative(p_slice)     # p >= 0
    _add_scalar_upper_bound(builder, p_slice, 1.0)  # p <= 1

    cs_dims: list[int] = []
    for variable_spec in (sigma, tau, ppt_sigma_full):
        builder.add_psd_constraint(
            variable_spec,
            cached_scs_complex_psd_coordinate_action(variable_spec.matrix_dim),
        )
        cs_dims.append(int(variable_spec.matrix_dim))

    cone = {"z": int(zero_rows), "l": 2, "cs": cs_dims}
    A, b, c, constraint_counts = builder.finalize(objective_slice=p_slice, cone=cone)
    c = -c  # builder.finalize creates max-objective data; level 2 minimizes p.

    build_profile = {
        "total_build_time": perf_counter() - total_start,
        "scope": {
            "formulation": "level2_full_coordinate_scs",
            "full_dim": full_dim,
            "ppt_auxiliaries": len(ppt_auxiliaries),
            "x_dim": int(c.size),
            "cone_zero_rows": int(zero_rows),
            "cone_linear_rows": 2,
            "cone_cs_dims": tuple(int(dim) for dim in cs_dims),
        },
    }
    return InflationRobustnessLevel2SCSModel(
        dims3=dims3,
        dims6=dims6,
        D=D,
        full_dim=full_dim,
        A=A,
        b=b,
        c=c,
        cone=cone,
        p_slice=p_slice,
        sigma=sigma,
        tau=tau,
        ppt_auxiliaries=ppt_auxiliaries,
        constraint_counts=constraint_counts,
        build_profile=build_profile,
    )


def solve_inflation_robustness_level2_scs(
    rho: np.ndarray,
    dims3: tuple[int, int, int],
    *,
    verbose: int = 1,
    optimizer_max_time: float | None = None,
    scs_max_iters: int = int(1e5),
    scs_eps_abs: float = 1e-4,
    scs_eps_rel: float = 1e-4,
    scs_use_indirect: bool = False,
    scs_alpha: float = DEFAULT_SCS_ALPHA,
    scs_scale: float = DEFAULT_SCS_SCALE,
    scs_normalize: bool = True,
    scs_adaptive_scale: bool = True,
    scs_rho_x: float = DEFAULT_SCS_RHO_X,
    scs_acceleration_lookback: int = DEFAULT_SCS_ACCELERATION_LOOKBACK,
    scs_acceleration_interval: int = DEFAULT_SCS_ACCELERATION_INTERVAL,
) -> dict[str, object]:
    built = build_inflation_robustness_level2_scs_model(
        rho,
        dims3,
        verbose=verbose,
        optimizer_max_time=optimizer_max_time,
    )
    solve_start = perf_counter()
    solved = solve_with_scs_compat(
        {
            "P": sp.csc_matrix((built.c.size, built.c.size), dtype=np.float64),
            "A": built.A,
            "b": built.b,
            "c": built.c,
        },
        built.cone,
        verbose=bool(verbose),
        max_iters=int(scs_max_iters),
        eps_abs=float(scs_eps_abs),
        eps_rel=float(scs_eps_rel),
        alpha=float(scs_alpha),
        scale=float(scs_scale),
        normalize=bool(scs_normalize),
        adaptive_scale=bool(scs_adaptive_scale),
        rho_x=float(scs_rho_x),
        acceleration_lookback=int(scs_acceleration_lookback),
        acceleration_interval=int(scs_acceleration_interval),
        time_limit_secs=0.0 if optimizer_max_time is None else float(optimizer_max_time),
        use_indirect=bool(scs_use_indirect),
    )
    solve_seconds = perf_counter() - solve_start

    info = dict(solved.get("info", {}))
    x_values = solved.get("x")
    if x_values is None:
        p_value = float("nan")
        t_value = float("nan")
        sigma_matrix = None
        tau_matrix = None
    else:
        x_values = np.asarray(x_values, dtype=np.float64).reshape(-1)
        p_value = float(x_values[built.p_slice.offset])
        t_value = 1.0 - p_value
        sigma_matrix = _extract_scs_variable_matrix(x_values, built.sigma)
        tau_matrix = _extract_scs_variable_matrix(x_values, built.tau)

    status = str(info.get("status", "unknown"))
    return {
        "problem_status": status,
        "solution_status": status,
        "p_value": p_value,
        "t_value": t_value,
        "sigma_matrix": sigma_matrix,
        "tau_matrix": tau_matrix,
        "solve_seconds": solve_seconds,
        "constraint_counts": dict(built.constraint_counts),
        "build_profile": dict(built.build_profile),
        "solver_info": info,
    }


def inflation_robustness_level2(
    rho: np.ndarray,
    dims3: tuple[int, int, int],
    *,
    verbose: int = 1,
    optimizer_max_time: float | None = None,
    scs_max_iters: int = int(1e5),
    scs_eps_abs: float = 1e-4,
    scs_eps_rel: float = 1e-4,
    scs_use_indirect: bool = False,
    scs_alpha: float = DEFAULT_SCS_ALPHA,
    scs_scale: float = DEFAULT_SCS_SCALE,
    scs_normalize: bool = True,
    scs_adaptive_scale: bool = True,
    scs_rho_x: float = DEFAULT_SCS_RHO_X,
    scs_acceleration_lookback: int = DEFAULT_SCS_ACCELERATION_LOOKBACK,
    scs_acceleration_interval: int = DEFAULT_SCS_ACCELERATION_INTERVAL,
):
    solved = solve_inflation_robustness_level2_scs(
        rho,
        dims3,
        verbose=verbose,
        optimizer_max_time=optimizer_max_time,
        scs_max_iters=scs_max_iters,
        scs_eps_abs=scs_eps_abs,
        scs_eps_rel=scs_eps_rel,
        scs_use_indirect=scs_use_indirect,
        scs_alpha=scs_alpha,
        scs_scale=scs_scale,
        scs_normalize=scs_normalize,
        scs_adaptive_scale=scs_adaptive_scale,
        scs_rho_x=scs_rho_x,
        scs_acceleration_lookback=scs_acceleration_lookback,
        scs_acceleration_interval=scs_acceleration_interval,
    )
    return solved["p_value"], solved["sigma_matrix"], solved["tau_matrix"]
