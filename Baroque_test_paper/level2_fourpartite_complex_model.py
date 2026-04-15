"""Readable four-partite level-2 complex robustness SDP.

This file is meant to read like the mathematical SDP itself.

Variables:

- ``sigma`` : two disconnected tetrahedra
- ``tau``   : one connected 8-ring

Slot conventions:

- ``sigma`` lives on ``(A111, B111, C111, D111, A222, B222, C222, D222)``
- ``tau``   lives on ``(A211, B111, C111, D112, A122, B222, C222, D221)``

Optimization problem:

    minimize p

subject to

    Tr_rest(sigma) = (1-p) rho_ABCD + p I/Delta
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
        identity_coordinate_action,
        pack_hermitian_coordinates,
        product_dim,
        solve_with_scs_compat,
        unpack_hermitian_coordinates,
    )


@dataclass
class InflationRobustnessLevel2FourpartiteSCSModel:
    dims4: tuple[int, int, int, int]
    dims8: tuple[int, ...]
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
#   0    1    2    3    4    5    6    7
#   A111 B111 C111 D111 A222 B222 C222 D222
#
# Slot order for tau:
#   0    1    2    3    4    5    6    7
#   A211 B111 C111 D112 A122 B222 C222 D221

# Internal symmetries.
# sigma: swap the two disconnected tetrahedra.
# tau:   half-turn of the connected 8-ring.
SIGMA_SWAP = (4, 5, 6, 7, 0, 1, 2, 3)
TAU_HALFTURN = (4, 5, 6, 7, 0, 1, 2, 3)

# Observed anchor on sigma:
#   Tr_{A222 B222 C222 D222}(sigma) = (1-p) rho_ABCD + p I_ABCD / Delta
OBSERVED_ANCHOR_SIGMA_KEEP = (0, 1, 2, 3)

# Cross-inflation equalities between the disconnected tetrahedra and the 8-ring.
# 1) ABC consistency
EQ_SIGMA_TAU_ABC_SIGMA = (0, 1, 2, 4, 5, 6)  # sigma(A111, B111, C111, A222, B222, C222)
EQ_SIGMA_TAU_ABC_TAU = (0, 1, 2, 4, 5, 6)    # tau(A211, B111, C111, A122, B222, C222)

# 2) ABD consistency
EQ_SIGMA_TAU_ABD_SIGMA = (0, 1, 3, 4, 5, 7)  # sigma(A111, B111, D111, A222, B222, D222)
EQ_SIGMA_TAU_ABD_TAU = (0, 1, 3, 4, 5, 7)    # tau(A211, B111, D112, A122, B222, D221)

# 3) ACD consistency
EQ_SIGMA_TAU_ACD_SIGMA = (0, 2, 3, 4, 6, 7)  # sigma(A111, C111, D111, A222, C222, D222)
EQ_SIGMA_TAU_ACD_TAU = (0, 2, 3, 4, 6, 7)    # tau(A211, C111, D112, A122, C222, D221)

# 4) BCD consistency
EQ_SIGMA_TAU_BCD_SIGMA = (1, 2, 3, 5, 6, 7)  # sigma(B111, C111, D111, B222, C222, D222)
EQ_SIGMA_TAU_BCD_TAU = (1, 2, 3, 5, 6, 7)    # tau(B111, C111, D112, B222, C222, D221)

# PPT on sigma across (A111 B111 C111 D111) | (A222 B222 C222 D222).
PPT_SIGMA_FULL_TRANSPOSE = (0, 1, 2, 3)


def _validate_level2_fourpartite_inputs(
    rho: np.ndarray,
    dims4: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, ...], int, int, np.ndarray]:
    dims4 = tuple(int(dim) for dim in dims4)
    if len(dims4) != 4:
        raise ValueError("dims4 must contain exactly four local dimensions.")
    D = product_dim(dims4)
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (D, D):
        raise ValueError(f"rho must have shape {(D, D)}, got {rho.shape}.")
    if not np.allclose(rho, rho.conj().T, atol=1e-9):
        raise ValueError("rho must be Hermitian.")
    dims8 = (
        dims4[0], dims4[1], dims4[2], dims4[3],
        dims4[0], dims4[1], dims4[2], dims4[3],
    )
    full_dim = product_dim(dims8)
    return dims4, dims8, D, full_dim, rho


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


def build_inflation_robustness_level2_fourpartite_scs_model(
    rho: np.ndarray,
    dims4: tuple[int, int, int, int],
    *,
    verbose: int = 1,
    optimizer_max_time: float | None = None,
) -> InflationRobustnessLevel2FourpartiteSCSModel:
    """Build the direct SCS cone program for the complex four-partite level-2 SDP."""
    del verbose, optimizer_max_time
    dims4, dims8, D, full_dim, rho = _validate_level2_fourpartite_inputs(rho, dims4)
    total_start = perf_counter()

    builder = SCSAffineConstraintBuilder()

    # Scalar optimization variable:
    #   Tr_rest(sigma) = (1-p) rho + p I/Delta
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
    # sigma(111, 222) = sigma(222, 111)
    builder.add_difference_equality(
        sigma,
        cached_coordinate_action_for_slot_permutation(dims8, SIGMA_SWAP),
        sigma,
        identity_coordinate_action(sigma.coord_dim),
        counter_key="symmetry",
    )
    # tau( A211 B111 C111 D112 | A122 B222 C222 D221 )
    #   = tau( A122 B222 C222 D221 | A211 B111 C111 D112 )
    builder.add_difference_equality(
        tau,
        cached_coordinate_action_for_slot_permutation(dims8, TAU_HALFTURN),
        tau,
        identity_coordinate_action(tau.coord_dim),
        counter_key="symmetry",
    )

    # 3) Observed ABCD anchor on sigma:
    #   Tr_{A222 B222 C222 D222}(sigma) = (1-p) rho_ABCD + p I_ABCD / Delta
    rho_coord = pack_hermitian_coordinates(rho)
    mixed_coord = pack_hermitian_coordinates(np.eye(D, dtype=np.complex128) / float(D))
    builder.add_observed_anchor(
        sigma,
        cached_partial_trace_coordinate_action(dims8, OBSERVED_ANCHOR_SIGMA_KEEP),
        rho_coord.astype(np.float64, copy=False),
        p_slice,
        (rho_coord - mixed_coord).astype(np.float64, copy=False),
    )

    # 4) Cross-inflation consistency equalities.
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ABC_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ABC_TAU),
        counter_key="equality",
    )
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ABD_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ABD_TAU),
        counter_key="equality",
    )
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ACD_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_ACD_TAU),
        counter_key="equality",
    )
    builder.add_difference_equality(
        sigma,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_BCD_SIGMA),
        tau,
        cached_partial_trace_coordinate_action(dims8, EQ_SIGMA_TAU_BCD_TAU),
        counter_key="equality",
    )

    # 5) PPT on sigma across (A111 B111 C111 D111) | (A222 B222 C222 D222).
    builder.add_auxiliary_link(
        ppt_sigma_full,
        sigma,
        cached_partial_transpose_coordinate_action(dims8, PPT_SIGMA_FULL_TRANSPOSE),
    )

    # 6) Linear bounds and PSD cone constraints.
    zero_rows = int(builder.next_row)
    builder.add_scalar_nonnegative(p_slice)         # p >= 0
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
    c = -c  # builder.finalize creates max-objective data; this model minimizes p.

    build_profile = {
        "total_build_time": perf_counter() - total_start,
        "scope": {
            "formulation": "level2_fourpartite_full_coordinate_scs",
            "full_dim": full_dim,
            "ppt_auxiliaries": len(ppt_auxiliaries),
            "x_dim": int(c.size),
            "cone_zero_rows": int(zero_rows),
            "cone_linear_rows": 2,
            "cone_cs_dims": tuple(int(dim) for dim in cs_dims),
        },
    }
    return InflationRobustnessLevel2FourpartiteSCSModel(
        dims4=dims4,
        dims8=dims8,
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


def solve_inflation_robustness_level2_fourpartite_scs(
    rho: np.ndarray,
    dims4: tuple[int, int, int, int],
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
    built = build_inflation_robustness_level2_fourpartite_scs_model(
        rho,
        dims4,
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


def inflation_robustness_level2_fourpartite(
    rho: np.ndarray,
    dims4: tuple[int, int, int, int],
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
    solved = solve_inflation_robustness_level2_fourpartite_scs(
        rho,
        dims4,
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
