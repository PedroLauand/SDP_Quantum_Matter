"""Shared helpers for the GNME state examples."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import (  # noqa: E402
    DEFAULT_GNME_SDP_BACKEND,
    GNMEProblem,
    TopDownStateSDPDraft,
    build_default_feasibility_model,
    build_fusion_feasibility_model,
    build_smaller_sdp_draft,
    build_sdp_draft,
    build_top_down_block_task_feasibility_model,
    build_top_down_sdp_draft,
)
from gnme_inflation.sdp.GNMEStateSDP import validate_top_down_draft  # noqa: E402


DEFAULT_LOCAL_DIMS_PER_PARTY = (2, 2, 2)


def partial_trace_qubits(
    rho: np.ndarray,
    keep_positions: tuple[int, ...],
    n_qubits: int = 3,
) -> np.ndarray:
    """Partial trace over all qubits not listed in ``keep_positions``."""
    dims = (2,) * n_qubits
    traced_positions = tuple(pos for pos in range(n_qubits) if pos not in keep_positions)
    tensor = rho.reshape(dims + dims)
    for pos in sorted(traced_positions, reverse=True):
        tensor = np.trace(tensor, axis1=pos, axis2=pos + tensor.ndim // 2)
    kept_dims = (2,) * len(keep_positions)
    return tensor.reshape((np.prod(kept_dims), np.prod(kept_dims)))


def matrix_coordinate_values(matrix: np.ndarray, Hermitian: bool) -> np.ndarray:
    """Pack a matrix into the coordinate convention used by the backend."""
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


def maximally_mixed_targets() -> dict[tuple[str, ...], np.ndarray]:
    """Reference maximally mixed states used for the noise model."""
    return {
        ("A_11", "B_11"): np.eye(4, dtype=float) / 4.0,
        ("A_11", "C_11"): np.eye(4, dtype=float) / 4.0,
        ("B_11", "C_11"): np.eye(4, dtype=float) / 4.0,
        ("A_11", "B_11", "C_11"): np.eye(8, dtype=float) / 8.0,
    }


def _representatives_by_target(draft) -> dict[tuple[str, ...], object]:
    """Map objective anchors by their target lexorder."""
    if isinstance(draft, TopDownStateSDPDraft):
        return {
            tuple(representative.target_lexorder): representative
            for representative in draft.known_representatives
        }
    return {
        tuple(representative.target_lexorder): representative
        for representative in draft.known_representatives
    }


def _coordinate_to_bar_entries(
    matrix_dim: int,
    coord_index: int,
    Hermitian: bool,
) -> tuple[tuple[int, int, float], ...]:
    """Map one backend coordinate to lower-triangle entries of the Task bar variable."""
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            if cursor == coord_index:
                if not Hermitian:
                    return ((col, row, (1.0 if row == col else 0.5)),)
                if row == col:
                    return (
                        (row, row, 0.5),
                        (matrix_dim + row, matrix_dim + row, 0.5),
                    )
                return (
                    (col, row, 0.25),
                    (matrix_dim + col, matrix_dim + row, 0.25),
                )
            cursor += 1
    if Hermitian:
        for row in range(matrix_dim):
            for col in range(row + 1, matrix_dim):
                if cursor == coord_index:
                    return (
                        (matrix_dim + row, col, 0.25),
                        (matrix_dim + col, row, -0.25),
                    )
                cursor += 1
    raise IndexError(f"Coordinate index {coord_index} out of range for dim {matrix_dim}.")


def build_problem(
    level: int,
    verbose: int,
    local_dims_per_party: tuple[int, ...] = DEFAULT_LOCAL_DIMS_PER_PARTY,
) -> GNMEProblem:
    """Construct the default 3-party qubit GNME problem."""
    return GNMEProblem(
        n_parties=3,
        inflation_level=level,
        local_dims_per_party=local_dims_per_party,
        verbose=verbose,
    )


def build_draft(
    level: int,
    verbose: int,
    local_dims_per_party: tuple[int, ...] = DEFAULT_LOCAL_DIMS_PER_PARTY,
    formulation: str = "default",
):
    """Build the default GNME SDP draft for the example problem."""
    problem = build_problem(level, verbose, local_dims_per_party=local_dims_per_party)
    formulation = str(formulation).lower()
    if formulation in {"default", "bottom_up"}:
        return build_sdp_draft(problem)
    if formulation == "top_down":
        return build_top_down_sdp_draft(problem)
    if formulation in {"smaller", "minimal", "maximal"}:
        return build_smaller_sdp_draft(problem)
    raise ValueError(f"Unsupported formulation {formulation!r}.")


def build_top_down_draft(
    level: int,
    verbose: int,
    local_dims_per_party: tuple[int, ...] = DEFAULT_LOCAL_DIMS_PER_PARTY,
):
    """Convenience wrapper for the fresh top-down draft."""
    return build_draft(
        level=level,
        verbose=verbose,
        local_dims_per_party=local_dims_per_party,
        formulation="top_down",
    )


def summarize_draft(
    draft,
    *,
    validate: bool = False,
) -> dict[str, object]:
    """Return a compact structural summary for a GNME draft."""
    summary: dict[str, object] = {
        "draft_type": type(draft).__name__,
        "tau_variables": len(draft.psd_variables),
        "internal_symmetry_constraints": len(draft.internal_symmetry_constraints),
        "ppt_constraints": len(draft.ppt_constraints),
        "fixed_marginal_constraints": len(draft.fixed_marginal_constraints),
    }
    if isinstance(draft, TopDownStateSDPDraft):
        summary.update(
            {
                "formulation": "top_down",
                "maximal_families": len(draft.family_blueprint.maximal_families),
                "overlap_families": len(draft.family_blueprint.overlap_families),
                "known_families": len(draft.family_blueprint.known_families),
                "maximal_representatives": len(draft.maximal_representatives),
                "known_representatives": len(draft.known_representatives),
                "representative_links": len(draft.representative_links),
                "tau_family_equalities": len(draft.tau_representative_constraints),
                "cross_inflation_groups": len(draft.cross_inflation_groups),
                "tau_ppt_constraints": sum(
                    1
                    for constraint in draft.ppt_constraints
                    if constraint.source_variable_kind == "tau"
                ),
                "family_ppt_constraints": sum(
                    1
                    for constraint in draft.ppt_constraints
                    if constraint.source_variable_kind != "tau"
                ),
                "verified_at_draft_time": bool(draft.verified_at_draft_time),
            }
        )
        if validate:
            summary["validation"] = validate_top_down_draft(draft, verbose=0)
    else:
        summary.update(
            {
                "formulation": "bottom_up",
                "auxiliary_representatives": len(draft.auxiliary_representatives),
                "known_representatives": len(draft.known_representatives),
                "shared_representatives": len(draft.shared_representatives),
                "representative_links": len(draft.representative_links),
                "representative_constraints": len(draft.representative_constraints),
                "shared_marginal_constraints": len(draft.shared_marginal_constraints),
            }
        )
    return summary


def print_draft_summary(
    draft,
    *,
    validate: bool = False,
    prefix: str = "[draft]",
) -> dict[str, object]:
    """Print a compact structural summary for one draft."""
    summary = summarize_draft(draft, validate=validate)
    print(
        f"{prefix} formulation={summary['formulation']} type={summary['draft_type']}",
        flush=True,
    )
    if summary["formulation"] == "top_down":
        print(
            f"{prefix} tau={summary['tau_variables']}, maximal={summary['maximal_representatives']}, "
            f"known={summary['known_representatives']}",
            flush=True,
        )
        print(
            f"{prefix} families(max/overlap/known)="
            f"{summary['maximal_families']}/{summary['overlap_families']}/{summary['known_families']}, "
            f"links={summary['representative_links']}, "
            f"equalities(tau)={summary['tau_family_equalities']}",
            flush=True,
        )
        print(
            f"{prefix} ppt(total/tau/family)="
            f"{summary['ppt_constraints']}/{summary['tau_ppt_constraints']}/{summary['family_ppt_constraints']}, "
            f"cross-groups={summary['cross_inflation_groups']}, "
            f"verified={summary['verified_at_draft_time']}",
            flush=True,
        )
        validation = summary.get("validation")
        if isinstance(validation, dict):
            print(
                f"{prefix} validation={validation}",
                flush=True,
            )
    else:
        print(
            f"{prefix} tau={summary['tau_variables']}, aux={summary['auxiliary_representatives']}, "
            f"shared={summary['shared_representatives']}, known={summary['known_representatives']}",
            flush=True,
        )
        print(
            f"{prefix} representative constraints={summary['representative_constraints']}, "
            f"shared marginals={summary['shared_marginal_constraints']}, "
            f"ppt={summary['ppt_constraints']}",
            flush=True,
        )
    return summary


def print_build_profile(
    build_profile: dict[str, object],
    *,
    prefix: str = "[build]",
) -> None:
    """Print a compact build-profile report with timings and nested counters."""
    if not build_profile:
        print(f"{prefix} build profile unavailable", flush=True)
        return

    timing_keys = [
        key
        for key, value in build_profile.items()
        if key.endswith("_time") and isinstance(value, (int, float))
    ]
    if timing_keys:
        print(f"{prefix} timings:", flush=True)
        for key in sorted(timing_keys):
            print(f"{prefix}   {key}={float(build_profile[key]):.2f}s", flush=True)

    for section_key in (
        "scope",
        "tau_declaration_stats",
        "observed_stats",
        "equality_stats",
        "tau_map_cache",
        "tau_family_stats",
        "known_value_stats",
        "ppt_stats",
    ):
        section = build_profile.get(section_key)
        if isinstance(section, dict) and section:
            print(f"{prefix} {section_key}={section}", flush=True)


def build_noise_optimization_model(
    draft,
    known_targets: dict[tuple[str, ...], np.ndarray],
    state_label: str,
    Hermitian: bool,
    verbose: int,
    mosek_num_threads: int | None = None,
    optimizer_max_time: float | None = None,
    backend: str = DEFAULT_GNME_SDP_BACKEND,
):
    """Build the affine noise interpolation model without solving it."""
    backend = str(backend).lower()
    if backend not in {"fusion", "task"}:
        raise ValueError(f"Unsupported backend {backend!r}.")

    from mosek.fusion import AccSolutionStatus, Domain, Expr, ObjectiveSense
    import mosek

    mixed_targets = maximally_mixed_targets()
    top_down = isinstance(draft, TopDownStateSDPDraft)
    anchored = top_down
    known_targets = {
        tuple(target_lexorder): state_matrix
        for target_lexorder, state_matrix in known_targets.items()
    }
    known_reps_by_target = _representatives_by_target(draft)
    known_targets = {
        tuple(target_lexorder): state_matrix
        for target_lexorder, state_matrix in known_targets.items()
        if tuple(target_lexorder) in known_reps_by_target
    }
    if not known_targets:
        raise ValueError("No known targets matched representatives in the selected draft.")

    if backend == "fusion":
        if anchored:
            raise ValueError("Fusion backend is not implemented for anchored top-down drafts.")
        fusion_model = build_fusion_feasibility_model(
            draft,
            Hermitian=Hermitian,
            verbose=verbose,
        )
        M = fusion_model.model
        p_noise_var = M.variable("p_noise", 1, Domain.inRange(0.0, 1.0))
        p_noise = p_noise_var.index(0)

        affine_constraint_count = 0
        for target_lexorder, state_matrix in known_targets.items():
            representative = known_reps_by_target[target_lexorder]
            rep_coord = fusion_model.known_representative_coordinate_vectors[representative.name]
            state_coord = matrix_coordinate_values(state_matrix, Hermitian)
            mixed_coord = matrix_coordinate_values(mixed_targets[target_lexorder], Hermitian)
            delta_coord = mixed_coord - state_coord
            for coord_index, (base_value, delta_value) in enumerate(zip(state_coord, delta_coord)):
                rhs = Expr.add(float(base_value), Expr.mul(float(delta_value), p_noise))
                M.constraint(
                    f"known_affine_{affine_constraint_count}",
                    Expr.sub(rep_coord.index(coord_index), rhs),
                    Domain.equalsTo(0.0),
                )
                affine_constraint_count += 1

        M.objective(ObjectiveSense.Minimize, p_noise)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        if mosek_num_threads is not None:
            M.setSolverParam("numThreads", int(mosek_num_threads))
        if optimizer_max_time is not None:
            M.setSolverParam("optimizerMaxTime", float(optimizer_max_time))
        if verbose > 0:
            M.setLogHandler(sys.stdout)

        return {
            "state_label": state_label,
            "backend": backend,
            "fusion_model": fusion_model,
            "model": M,
            "p_noise": p_noise,
            "affine_known_constraints": affine_constraint_count,
            "base_constraint_counts": fusion_model.constraint_counts,
            "build_profile": getattr(fusion_model, "build_profile", {}),
        }

    if top_down:
        task_model = build_top_down_block_task_feasibility_model(
            draft,
            Hermitian=Hermitian,
            verbose=verbose,
        )
    else:
        task_model = build_default_feasibility_model(
            draft,
            Hermitian=Hermitian,
            verbose=verbose,
        )
    task = task_model.task
    p_noise_index = task.getnumvar()
    task.appendvars(1)
    task.putvarboundlist(
        np.asarray([p_noise_index], dtype=np.int32),
        [mosek.boundkey.ra],
        np.asarray([0.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
    )
    task.putcj(int(p_noise_index), 1.0)
    task.putobjsense(mosek.objsense.minimize)
    if mosek_num_threads is not None:
        task.putintparam(mosek.iparam.num_threads, int(mosek_num_threads))
    if optimizer_max_time is not None:
        task.putdouparam(mosek.dparam.optimizer_max_time, float(optimizer_max_time))

    affine_constraint_count = 0
    next_row = task.getnumcon()
    observed_iterable = (
        (known_reps_by_target[target_lexorder], state_matrix)
        for target_lexorder, state_matrix in known_targets.items()
    )

    for observed_item, state_matrix in observed_iterable:
        anchor_name = observed_item.name
        target_lexorder = tuple(observed_item.target_lexorder)
        state_coord = matrix_coordinate_values(state_matrix, Hermitian)
        mixed_coord = matrix_coordinate_values(mixed_targets[target_lexorder], Hermitian)
        delta_coord = mixed_coord - state_coord
        row_count = int(state_coord.size)
        anchor = task_model.representative_anchors[anchor_name]
        source_barvars = task_model.tau_bar_variables[anchor.source_variable_name]
        if row_count != int(anchor.coordinate_dim):
            raise ValueError(
                f"Known target coordinate mismatch for {anchor_name}: "
                f"{row_count} vs {anchor.coordinate_dim}."
            )
        task.appendcons(row_count)
        row_indices = np.arange(next_row, next_row + row_count, dtype=np.int32)
        next_row += row_count
        task.putconboundlist(
            row_indices,
            [mosek.boundkey.fx] * row_count,
            state_coord.astype(np.float64, copy=False),
            state_coord.astype(np.float64, copy=False),
        )
        task.putaijlist(
            row_indices,
            np.full(row_count, int(p_noise_index), dtype=np.int32),
            (-delta_coord).astype(np.float64, copy=False),
        )
        bar_rows = []
        bar_cols = []
        bar_vals = []
        bar_subi = []
        bar_subj = []
        for sector_map, sector_barvar in zip(anchor.sector_maps, source_barvars):
            map_rows, map_cols, map_vals = sector_map
            for coord_row, coord_col, coeff in zip(map_rows, map_cols, map_vals):
                for row, col, scale in _coordinate_to_bar_entries(
                    int(sector_barvar.complex_dim or sector_barvar.dim),
                    int(coord_col),
                    bool(sector_barvar.hermitian),
                ):
                    bar_subi.append(int(row_indices[int(coord_row)]))
                    bar_subj.append(int(sector_barvar.barvar_index))
                    bar_rows.append(int(row))
                    bar_cols.append(int(col))
                    bar_vals.append(float(coeff) * float(scale))
        task.putbarablocktriplet(
            np.asarray(bar_subi, dtype=np.int32),
            np.asarray(bar_subj, dtype=np.int32),
            np.asarray(bar_rows, dtype=np.int32),
            np.asarray(bar_cols, dtype=np.int32),
            np.asarray(bar_vals, dtype=np.float64),
        )
        affine_constraint_count += row_count

    return {
        "state_label": state_label,
        "backend": backend,
        "task_model": task_model,
        "model": task,
        "p_noise": p_noise_index,
        "affine_known_constraints": affine_constraint_count,
        "base_constraint_counts": task_model.constraint_counts,
        "build_profile": getattr(task_model, "build_profile", {}),
    }


def solve_min_noise_level(
    draft,
    known_targets: dict[tuple[str, ...], np.ndarray],
    state_label: str,
    Hermitian: bool,
    verbose: int,
    mosek_num_threads: int | None = None,
    optimizer_max_time: float | None = None,
    backend: str = DEFAULT_GNME_SDP_BACKEND,
):
    """Solve the affine noise interpolation against the provided target family."""
    build_data = build_noise_optimization_model(
        draft,
        known_targets,
        state_label=state_label,
        Hermitian=Hermitian,
        verbose=verbose,
        mosek_num_threads=mosek_num_threads,
        optimizer_max_time=optimizer_max_time,
        backend=backend,
    )
    M = build_data["model"]
    p_noise = build_data["p_noise"]

    solve_start = perf_counter()
    if build_data["backend"] == "fusion":
        M.solve()
        p_noise_value = float(p_noise.level()[0])
        problem_status = str(M.getProblemStatus())
        primal_status = str(M.getPrimalSolutionStatus())
        dual_status = str(M.getDualSolutionStatus())
    else:
        import mosek
        M.optimize()
        p_noise_level = M.getxx(mosek.soltype.itr)
        p_noise_value = float(p_noise_level[p_noise])
        problem_status = str(M.getprosta(mosek.soltype.itr))
        primal_status = str(M.getsolsta(mosek.soltype.itr))
        dual_status = str(M.getsolsta(mosek.soltype.itr))
    solve_seconds = perf_counter() - solve_start

    return {
        "state_label": build_data["state_label"],
        "problem_status": problem_status,
        "primal_status": primal_status,
        "dual_status": dual_status,
        "p_noise": p_noise_value,
        "solve_seconds": solve_seconds,
        "affine_known_constraints": build_data["affine_known_constraints"],
        "base_constraint_counts": build_data["base_constraint_counts"],
        "build_profile": build_data["build_profile"],
    }


def print_result(level: int, Hermitian: bool, result: dict) -> None:
    """Print a short optimization summary for one example run."""
    print(f"GNME noisy-{result['state_label']} optimization (level {level})")
    print(f"  Hermitian: {Hermitian}")
    print(f"  problem status: {result['problem_status']}")
    print(f"  primal status: {result['primal_status']}")
    print(f"  dual status: {result['dual_status']}")
    print(f"  optimal p_noise: {result['p_noise']}")
    print(f"  solve time: {result['solve_seconds']:.2f}s")
    print(f"  affine known-value constraints: {result['affine_known_constraints']}")
    print(f"  base constraint counts: {result['base_constraint_counts']}")


def run_noise_example(
    build_draft_fn,
    solve_fn,
    print_label: str,
    level: int,
    Hermitian: bool,
    verbose: int,
) -> None:
    """Small CLI runner shared by the level-2 example scripts."""
    print(f"[{print_label}] building SDP draft", flush=True)
    draft = build_draft_fn(level=level, verbose=verbose)

    print(f"[{print_label}] solving optimization over p", flush=True)
    result = solve_fn(draft, Hermitian=Hermitian, verbose=verbose)

    print_result(level=level, Hermitian=Hermitian, result=result)
