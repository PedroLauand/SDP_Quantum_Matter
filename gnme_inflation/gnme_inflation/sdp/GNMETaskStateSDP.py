"""MOSEK Task backend for GNME block state SDPs.

This module is the first low-level alternative to the Fusion-heavy block
backend. It keeps the GNME draft and symmetry reduction unchanged, but compiles
the real SDP directly into a MOSEK ``Task`` with semidefinite bar variables and
linear equality rows.

The current implementation intentionally targets the real backend only. It
reuses the existing cached symmetry-adapted maps from ``GNMEBlockStateSDP`` and
converts them into Task ``barA`` coefficients instead of symbolic Fusion
expression trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np

from .GNMEStateSDP import (
    AssignedStateSDPDraft,
    StateSDPDraft,
    TopDownStateSDPDraft,
    PaperStateSDPDraft,
    _canonical_keep_positions_orbit,
    _progress_log,
    _quotient_representative_constraint_sequence,
    _resolve_verbose,
    quotient_ppt_constraints,
    quotient_representative_constraints,
    symmetric_matrix_orbits,
)
from .GNMEBlockStateSDP import (
    _cached_block_partial_trace_coordinate_maps,
    _cached_exact_small_group_selected_coordinate_maps,
    _cached_tau_ppt_orbit_reduction,
    _cached_tau_ppt_stabilizer,
    _layout_signature,
    _load_persistent_object,
    _party_signature_from_lexorder,
    _restrict_sector_triplets_to_row_lookup,
    _save_persistent_object,
    _upper_triangle_coordinate_index,
    cached_block_partial_transpose_coordinate_maps,
    cached_coordinate_action_for_slot_permutation,
    cached_partial_trace_coordinate_action,
    cached_partial_transpose_coordinate_action,
    cached_tau_layout,
)


@dataclass(frozen=True)
class TaskBarVariableSpec:
    """One MOSEK bar variable representing a PSD block."""

    name: str
    kind: str
    parent_name: str
    barvar_index: int
    dim: int
    coordinate_dim: int
    sector_label: str | None = None
    irrep_dim: int = 1
    complex_dim: int | None = None
    hermitian: bool = False


@dataclass
class TaskBlockStateSDPModel:
    """Concrete MOSEK Task model for the real block backend."""

    env: object
    task: object
    Hermitian: bool
    tau_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...]]
    representative_anchors: Dict[str, "TaskRepresentativeAnchorSpec"]
    auxiliary_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec]
    known_representative_bar_variables: Dict[str, TaskBarVariableSpec]
    ppt_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec]
    constraint_counts: Dict[str, int]
    build_profile: Dict[str, object]


@dataclass(frozen=True)
class TaskRepresentativeAnchorSpec:
    """Anchor tau-marginal map used instead of a representative PSD variable."""

    representative_name: str
    representative_kind: str
    representative_lexorder: Tuple[str, ...]
    source_variable_name: str
    source_variable_kind: str
    keep_positions: Tuple[int, ...]
    traced_positions: Tuple[int, ...]
    coordinate_dim: int
    sector_maps: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class TaskBarTermTemplate:
    """Row-offset-free Task bar-term block reusable across equalities."""

    relative_rows: np.ndarray
    subj: np.ndarray
    ptrb: np.ndarray
    ptre: np.ndarray
    matidx: np.ndarray
    weights: np.ndarray


@lru_cache(maxsize=None)
def _cached_representative_row_lookup(
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> Tuple[int, np.ndarray]:
    """Orbit-reduce a real symmetric representative coordinate basis."""
    orbit_data = symmetric_matrix_orbits(slot_dims, local_symmetry_perms)
    row_lookup = np.full(int(orbit_data.upper_triangular_entries), -1, dtype=np.int32)
    for (row, col), orbit_index in orbit_data.pair_to_orbit.items():
        row_lookup[_upper_triangle_coordinate_index(int(row), int(col), int(orbit_data.matrix_dim))] = int(orbit_index)
    return len(orbit_data.orbit_representatives), row_lookup


@lru_cache(maxsize=None)
def _coordinate_to_lower_triangle_data(matrix_dim: int):
    """Map one upper-triangle coordinate index to one MOSEK lower-triangle entry."""
    rows = np.empty(matrix_dim * (matrix_dim + 1) // 2, dtype=np.int32)
    cols = np.empty_like(rows)
    scales = np.empty(rows.size, dtype=np.float64)
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            rows[cursor] = col
            cols[cursor] = row
            scales[cursor] = 1.0 if row == col else 0.5
            cursor += 1
    return rows, cols, scales


@lru_cache(maxsize=None)
def _hermitian_coordinate_to_lower_triangle_data(complex_dim: int):
    """Map Hermitian coordinates to lower-triangle entries of the realified PSD block."""
    coord_dim = complex_dim * complex_dim
    lower_rows_a = np.empty(coord_dim, dtype=np.int32)
    lower_cols_a = np.empty(coord_dim, dtype=np.int32)
    lower_scales_a = np.empty(coord_dim, dtype=np.float64)
    lower_rows_b = np.empty(coord_dim, dtype=np.int32)
    lower_cols_b = np.empty(coord_dim, dtype=np.int32)
    lower_scales_b = np.empty(coord_dim, dtype=np.float64)

    cursor = 0
    for row in range(complex_dim):
        for col in range(row, complex_dim):
            if row == col:
                lower_rows_a[cursor] = row
                lower_cols_a[cursor] = row
                lower_scales_a[cursor] = 0.5
                lower_rows_b[cursor] = complex_dim + row
                lower_cols_b[cursor] = complex_dim + row
                lower_scales_b[cursor] = 0.5
            else:
                lower_rows_a[cursor] = col
                lower_cols_a[cursor] = row
                lower_scales_a[cursor] = 0.25
                lower_rows_b[cursor] = complex_dim + col
                lower_cols_b[cursor] = complex_dim + row
                lower_scales_b[cursor] = 0.25
            cursor += 1

    for row in range(complex_dim):
        for col in range(row + 1, complex_dim):
            lower_rows_a[cursor] = complex_dim + row
            lower_cols_a[cursor] = col
            lower_scales_a[cursor] = 0.25
            lower_rows_b[cursor] = complex_dim + col
            lower_cols_b[cursor] = row
            lower_scales_b[cursor] = -0.25
            cursor += 1

    return (
        lower_rows_a,
        lower_cols_a,
        lower_scales_a,
        lower_rows_b,
        lower_cols_b,
        lower_scales_b,
    )


def _append_fx_rows(task, next_row: int, rhs_values: np.ndarray):
    """Append equality rows and set fixed bounds."""
    import mosek

    rhs_values = np.asarray(rhs_values, dtype=np.float64)
    row_count = int(rhs_values.size)
    task.appendcons(row_count)
    row_indices = np.arange(next_row, next_row + row_count, dtype=np.int32)
    task.putconboundlist(
        row_indices,
        [mosek.boundkey.fx] * row_count,
        rhs_values,
        rhs_values,
    )
    return row_indices, next_row + row_count


def _row_identity_triplets(coord_dim: int, scale: float = 1.0):
    rows = np.arange(coord_dim, dtype=np.int32)
    cols = rows.copy()
    vals = np.full(coord_dim, float(scale), dtype=np.float64)
    return rows, cols, vals


def _coordinate_triplets_to_bar_triplets(
    row_offset: int,
    barvar_index: int,
    matrix_dim: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
):
    """Convert raw symmetric-coordinate triplets into MOSEK barA triplets."""
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return empty_i32, empty_i32, empty_i32, empty_i32, empty_f64

    lower_rows, lower_cols, lower_scales = _coordinate_to_lower_triangle_data(matrix_dim)
    return (
        (row_offset + rows).astype(np.int32, copy=False),
        np.full(rows.size, int(barvar_index), dtype=np.int32),
        lower_rows[cols].astype(np.int32, copy=False),
        lower_cols[cols].astype(np.int32, copy=False),
        (vals * lower_scales[cols]).astype(np.float64, copy=False),
    )


def _hermitian_coordinate_triplets_to_bar_triplets(
    row_offset: int,
    barvar_index: int,
    complex_dim: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
):
    """Convert Hermitian coordinate triplets into barA triplets on the realified PSD block."""
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return empty_i32, empty_i32, empty_i32, empty_i32, empty_f64

    (
        lower_rows_a,
        lower_cols_a,
        lower_scales_a,
        lower_rows_b,
        lower_cols_b,
        lower_scales_b,
    ) = _hermitian_coordinate_to_lower_triangle_data(complex_dim)

    out_rows = np.repeat((row_offset + rows).astype(np.int32, copy=False), 2)
    out_bar = np.full(2 * rows.size, int(barvar_index), dtype=np.int32)
    out_subk = np.empty(2 * rows.size, dtype=np.int32)
    out_subl = np.empty(2 * rows.size, dtype=np.int32)
    out_vals = np.empty(2 * rows.size, dtype=np.float64)

    out_subk[0::2] = lower_rows_a[cols]
    out_subl[0::2] = lower_cols_a[cols]
    out_vals[0::2] = vals * lower_scales_a[cols]

    out_subk[1::2] = lower_rows_b[cols]
    out_subl[1::2] = lower_cols_b[cols]
    out_vals[1::2] = vals * lower_scales_b[cols]
    return out_rows, out_bar, out_subk, out_subl, out_vals


def _append_bar_triplets(task, triplet_blocks: Iterable[Tuple[np.ndarray, ...]]) -> None:
    """Bulk-load one set of barA triplets."""
    subi_blocks = []
    subj_blocks = []
    subk_blocks = []
    subl_blocks = []
    val_blocks = []
    for subi, subj, subk, subl, vals in triplet_blocks:
        if vals.size == 0:
            continue
        subi_blocks.append(np.asarray(subi, dtype=np.int32))
        subj_blocks.append(np.asarray(subj, dtype=np.int32))
        subk_blocks.append(np.asarray(subk, dtype=np.int32))
        subl_blocks.append(np.asarray(subl, dtype=np.int32))
        val_blocks.append(np.asarray(vals, dtype=np.float64))
    if not val_blocks:
        return
    task.putbarablocktriplet(
        np.concatenate(subi_blocks).astype(np.int32, copy=False),
        np.concatenate(subj_blocks).astype(np.int32, copy=False),
        np.concatenate(subk_blocks).astype(np.int32, copy=False),
        np.concatenate(subl_blocks).astype(np.int32, copy=False),
        np.concatenate(val_blocks).astype(np.float64, copy=False),
    )


def _append_coordinate_basis_matrices(task, matrix_dim: int, Hermitian: bool) -> np.ndarray:
    """Append one sparse symmetric basis matrix per backend coordinate."""
    if Hermitian:
        (
            lower_rows_a,
            lower_cols_a,
            lower_scales_a,
            lower_rows_b,
            lower_cols_b,
            lower_scales_b,
        ) = _hermitian_coordinate_to_lower_triangle_data(matrix_dim)
        coord_dim = int(lower_rows_a.size)
        dims = np.full(coord_dim, 2 * int(matrix_dim), dtype=np.int32)
        nz = np.full(coord_dim, 2, dtype=np.int64)
        subi = np.empty(2 * coord_dim, dtype=np.int32)
        subj = np.empty(2 * coord_dim, dtype=np.int32)
        valij = np.empty(2 * coord_dim, dtype=np.float64)
        subi[0::2] = lower_rows_a
        subi[1::2] = lower_rows_b
        subj[0::2] = lower_cols_a
        subj[1::2] = lower_cols_b
        valij[0::2] = lower_scales_a
        valij[1::2] = lower_scales_b
    else:
        lower_rows, lower_cols, lower_scales = _coordinate_to_lower_triangle_data(matrix_dim)
        coord_dim = int(lower_rows.size)
        dims = np.full(coord_dim, int(matrix_dim), dtype=np.int32)
        nz = np.ones(coord_dim, dtype=np.int64)
        subi = lower_rows.astype(np.int32, copy=False)
        subj = lower_cols.astype(np.int32, copy=False)
        valij = lower_scales.astype(np.float64, copy=False)

    return np.asarray(
        task.appendsparsesymmatlist(dims, nz, subi, subj, valij),
        dtype=np.int64,
    )


def _coalesce_coordinate_triplets(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum duplicate (row, coordinate) contributions before Task loading."""
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size <= 1:
        return rows, cols, vals

    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    starts = np.empty(rows.size, dtype=bool)
    starts[0] = True
    starts[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    unique_idx = np.flatnonzero(starts)
    summed_vals = np.add.reduceat(vals, unique_idx)
    keep = np.abs(summed_vals) > 0.0
    return (
        rows[unique_idx][keep].astype(np.int32, copy=False),
        cols[unique_idx][keep].astype(np.int32, copy=False),
        summed_vals[keep].astype(np.float64, copy=False),
    )


def _coordinate_triplets_to_bar_terms(
    row_offset: int,
    barvar_index: int,
    basis_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
):
    """Convert coordinate-map triplets into Task weighted sums of basis matrices."""
    rows, cols, vals = _coalesce_coordinate_triplets(rows, cols, vals)
    if vals.size == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_i64 = np.asarray([], dtype=np.int64)
        empty_f64 = np.asarray([], dtype=np.float64)
        return empty_i32, empty_i32, empty_i64, empty_i64, empty_i64, empty_f64

    if np.any(rows[1:] < rows[:-1]):
        order = np.argsort(rows, kind="mergesort")
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]

    starts = np.empty(rows.size, dtype=bool)
    starts[0] = True
    starts[1:] = rows[1:] != rows[:-1]
    ptrb = np.flatnonzero(starts).astype(np.int64, copy=False)
    ptre = np.empty(ptrb.size, dtype=np.int64)
    if ptrb.size:
        ptre[:-1] = ptrb[1:]
        ptre[-1] = rows.size

    return (
        (row_offset + rows[ptrb]).astype(np.int32, copy=False),
        np.full(ptrb.size, int(barvar_index), dtype=np.int32),
        ptrb,
        ptre,
        basis_indices[cols].astype(np.int64, copy=False),
        vals.astype(np.float64, copy=False),
    )


def _coordinate_triplets_to_bar_term_template(
    barvar_index: int,
    basis_indices: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
) -> TaskBarTermTemplate:
    """Compile coordinate-map triplets into a reusable Task bar-term template."""
    rows, cols, vals = _coalesce_coordinate_triplets(rows, cols, vals)
    if vals.size == 0:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_i64 = np.asarray([], dtype=np.int64)
        empty_f64 = np.asarray([], dtype=np.float64)
        return TaskBarTermTemplate(
            relative_rows=empty_i32,
            subj=empty_i32,
            ptrb=empty_i64,
            ptre=empty_i64,
            matidx=empty_i64,
            weights=empty_f64,
        )

    if np.any(rows[1:] < rows[:-1]):
        order = np.argsort(rows, kind="mergesort")
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]

    starts = np.empty(rows.size, dtype=bool)
    starts[0] = True
    starts[1:] = rows[1:] != rows[:-1]
    ptrb = np.flatnonzero(starts).astype(np.int64, copy=False)
    ptre = np.empty(ptrb.size, dtype=np.int64)
    if ptrb.size:
        ptre[:-1] = ptrb[1:]
        ptre[-1] = rows.size

    return TaskBarTermTemplate(
        relative_rows=rows[ptrb].astype(np.int32, copy=False),
        subj=np.full(ptrb.size, int(barvar_index), dtype=np.int32),
        ptrb=ptrb,
        ptre=ptre,
        matidx=basis_indices[cols].astype(np.int64, copy=False),
        weights=vals.astype(np.float64, copy=False),
    )


def _instantiate_bar_term_template(
    row_offset: int,
    template: TaskBarTermTemplate,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize one reusable Task bar-term template at a given row offset."""
    return (
        (row_offset + template.relative_rows).astype(np.int32, copy=False),
        template.subj,
        template.ptrb,
        template.ptre,
        template.matidx,
        template.weights,
    )


def _maybe_log_loop_progress(
    verbose: int,
    label: str,
    completed: int,
    total: int,
    start_time: float,
    last_log_time: float,
    *,
    level: int = 2,
    min_interval_s: float = 10.0,
    extra: str | None = None,
    force: bool = False,
) -> float:
    """Emit a periodic loop progress update and return the new last-log time."""
    now = perf_counter()
    should_log = force or completed <= 1 or completed >= total
    if not should_log and verbose >= level:
        if verbose >= 3:
            should_log = True
        elif now - last_log_time >= float(min_interval_s):
            should_log = True
    if not should_log:
        return last_log_time

    pct = 100.0 if total <= 0 else (100.0 * float(completed) / float(total))
    message = f"{label}: {completed}/{total} ({pct:.1f}%) in {now - start_time:.2f}s"
    if extra:
        message += f", {extra}"
    _progress_log(verbose, level, message)
    return now


def _append_bar_terms(task, term_blocks: Iterable[Tuple[np.ndarray, ...]]) -> None:
    """Bulk-load Task barA elements as weighted sums of stored symmetric matrices."""
    subi_blocks = []
    subj_blocks = []
    ptrb_blocks = []
    ptre_blocks = []
    matidx_blocks = []
    weight_blocks = []
    term_offset = 0
    for subi, subj, ptrb, ptre, matidx, weights in term_blocks:
        if weights.size == 0:
            continue
        subi_blocks.append(np.asarray(subi, dtype=np.int32))
        subj_blocks.append(np.asarray(subj, dtype=np.int32))
        ptrb_blocks.append(np.asarray(ptrb, dtype=np.int64) + term_offset)
        ptre_blocks.append(np.asarray(ptre, dtype=np.int64) + term_offset)
        matidx_blocks.append(np.asarray(matidx, dtype=np.int64))
        weight_blocks.append(np.asarray(weights, dtype=np.float64))
        term_offset += int(weights.size)
    if not weight_blocks:
        return
    task.putbaraijlist(
        np.concatenate(subi_blocks).astype(np.int32, copy=False),
        np.concatenate(subj_blocks).astype(np.int32, copy=False),
        np.concatenate(ptrb_blocks).astype(np.int64, copy=False),
        np.concatenate(ptre_blocks).astype(np.int64, copy=False),
        np.concatenate(matidx_blocks).astype(np.int64, copy=False),
        np.concatenate(weight_blocks).astype(np.float64, copy=False),
    )


def _compose_coordinate_action_with_sector_map(
    action_rows: np.ndarray,
    action_cols: np.ndarray,
    action_vals: np.ndarray,
    sector_map: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose a coordinate action with one tau-sector map."""
    src_rows, src_cols, src_vals = sector_map
    src_rows = np.asarray(src_rows, dtype=np.int32)
    src_cols = np.asarray(src_cols, dtype=np.int32)
    src_vals = np.asarray(src_vals, dtype=np.float64)
    if src_vals.size == 0 or action_vals.size == 0:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )

    order = np.argsort(src_rows, kind="mergesort")
    src_rows = src_rows[order]
    src_cols = src_cols[order]
    src_vals = src_vals[order]
    row_ptr = np.searchsorted(src_rows, np.arange(int(src_rows[-1]) + 2, dtype=np.int32))

    out_rows = []
    out_cols = []
    out_vals = []
    for target_row, source_row, scale in zip(
        np.asarray(action_rows, dtype=np.int32),
        np.asarray(action_cols, dtype=np.int32),
        np.asarray(action_vals, dtype=np.float64),
    ):
        if source_row + 1 >= row_ptr.size:
            continue
        start = int(row_ptr[source_row])
        end = int(row_ptr[source_row + 1])
        if start == end:
            continue
        out_rows.append(np.full(end - start, int(target_row), dtype=np.int32))
        out_cols.append(src_cols[start:end])
        out_vals.append((scale * src_vals[start:end]).astype(np.float64, copy=False))

    if not out_vals:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )
    rows = np.concatenate(out_rows).astype(np.int32, copy=False)
    cols = np.concatenate(out_cols).astype(np.int32, copy=False)
    vals = np.concatenate(out_vals).astype(np.float64, copy=False)
    return _coalesce_coordinate_triplets(rows, cols, vals)


def _compose_coordinate_action_with_sector_maps(
    action_rows: np.ndarray,
    action_cols: np.ndarray,
    action_vals: np.ndarray,
    sector_maps: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...],
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Compose a coordinate action with all tau-sector maps."""
    return tuple(
        _compose_coordinate_action_with_sector_map(action_rows, action_cols, action_vals, sector_map)
        for sector_map in sector_maps
    )


def _source_lexorder(source_spec) -> Tuple[str, ...]:
    lexorder = getattr(source_spec, "lexorder", None)
    if lexorder is None:
        lexorder = getattr(source_spec, "target_lexorder")
    return tuple(lexorder)


def _source_layout_signature(source_spec, source_sector_coordinate_dims: Tuple[int, ...]):
    lexorder = _source_lexorder(source_spec)
    return (
        *_layout_signature(
            lexorder,
            source_spec.slot_dims,
            source_spec.local_symmetry_perms,
        ),
        tuple(int(dim) for dim in source_sector_coordinate_dims),
    )


@lru_cache(maxsize=None)
def _structural_source_layout_signature(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    Hermitian: bool,
):
    """Structural cache key compatible with the persistent block-map store."""
    layout = cached_tau_layout(
        "__task_cache__",
        tuple(party_signature),
        tuple(slot_dims),
        tuple(local_symmetry_perms),
    )
    if Hermitian:
        sector_coordinate_dims = tuple(
            int(sector.multiplicity * sector.multiplicity)
            for sector in layout.sectors
        )
    else:
        sector_coordinate_dims = tuple(
            int(sector.multiplicity * (sector.multiplicity + 1) // 2)
            for sector in layout.sectors
        )
    return (
        tuple(party_signature),
        tuple(int(dim) for dim in slot_dims),
        tuple(tuple(int(position) for position in permutation) for permutation in local_symmetry_perms),
        sector_coordinate_dims,
    )


def _canonicalize_source_positions(source_spec, positions, map_kind: str, Hermitian: bool):
    positions = tuple(positions)
    canonical_positions = positions
    transport_rows = None
    transport_signs = None

    if source_spec.local_symmetry_perms:
        orbit_members = {
            tuple(sorted(permutation[position] for position in positions))
            for permutation in source_spec.local_symmetry_perms
        }
        canonical_positions = min(orbit_members)
        if canonical_positions != positions:
            selected_perm = None
            for permutation in source_spec.local_symmetry_perms:
                if tuple(sorted(permutation[position] for position in canonical_positions)) == positions:
                    selected_perm = permutation
                    break
            if selected_perm is not None:
                if map_kind == "partial_trace":
                    actual_index = {position: idx for idx, position in enumerate(positions)}
                    reduced_slot_permutation = tuple(
                        actual_index[selected_perm[position]]
                        for position in canonical_positions
                    )
                    kept_dims = tuple(source_spec.slot_dims[position] for position in canonical_positions)
                    (
                        _coord_dim,
                        _rows,
                        _cols,
                        _vals,
                        transport_rows,
                        transport_signs,
                    ) = cached_coordinate_action_for_slot_permutation(
                        kept_dims,
                        reduced_slot_permutation,
                        Hermitian,
                    )
                elif map_kind == "partial_transpose":
                    (
                        _coord_dim,
                        _rows,
                        _cols,
                        _vals,
                        transport_rows,
                        transport_signs,
                    ) = cached_coordinate_action_for_slot_permutation(
                        source_spec.slot_dims,
                        selected_perm,
                        Hermitian,
                    )
    return canonical_positions, transport_rows, transport_signs


@lru_cache(maxsize=None)
def _cached_canonical_tau_sector_triplets(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    canonical_positions: Tuple[int, ...],
    Hermitian: bool,
):
    persistent_key = (
        "sectorwise",
        "partial_trace",
        _structural_source_layout_signature(
            tuple(party_signature),
            tuple(slot_dims),
            tuple(local_symmetry_perms),
            bool(Hermitian),
        ),
        tuple(canonical_positions),
        bool(Hermitian),
    )
    persistent_payload = _load_persistent_object(persistent_key)
    if persistent_payload is not None:
        try:
            target_coord_dim, sector_payload = persistent_payload
            return (
                int(target_coord_dim),
                tuple(
                    (
                        np.asarray(rows, dtype=np.int32),
                        np.asarray(cols, dtype=np.int32),
                        np.asarray(vals, dtype=np.float64),
                    )
                    for rows, cols, vals in sector_payload
                ),
            )
        except Exception:
            pass

    target_coord_dim, sector_maps = _cached_block_partial_trace_coordinate_maps(
        party_signature,
        slot_dims,
        local_symmetry_perms,
        canonical_positions,
        Hermitian,
    )
    compiled = (int(target_coord_dim), tuple(
        (
            rows.astype(np.int32, copy=False),
            cols.astype(np.int32, copy=False),
            vals.astype(np.float64, copy=False),
        )
        for rows, cols, vals in sector_maps
    ))
    _save_persistent_object(persistent_key, compiled)
    return compiled


@lru_cache(maxsize=None)
def _cached_canonical_tau_pt_sector_triplets(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    canonical_positions: Tuple[int, ...],
    Hermitian: bool,
):
    persistent_key = (
        "sectorwise",
        "partial_transpose",
        _structural_source_layout_signature(
            tuple(party_signature),
            tuple(slot_dims),
            tuple(local_symmetry_perms),
            bool(Hermitian),
        ),
        tuple(canonical_positions),
        bool(Hermitian),
    )
    persistent_payload = _load_persistent_object(persistent_key)
    if persistent_payload is not None:
        try:
            target_coord_dim, sector_payload = persistent_payload
            return (
                int(target_coord_dim),
                tuple(
                    (
                        np.asarray(rows, dtype=np.int32),
                        np.asarray(cols, dtype=np.int32),
                        np.asarray(vals, dtype=np.float64),
                    )
                    for rows, cols, vals in sector_payload
                ),
            )
        except Exception:
            pass

    target_coord_dim, sector_maps = cached_block_partial_transpose_coordinate_maps(
        party_signature,
        slot_dims,
        local_symmetry_perms,
        canonical_positions,
        Hermitian,
    )
    compiled = (int(target_coord_dim), tuple(
        (
            rows.astype(np.int32, copy=False),
            cols.astype(np.int32, copy=False),
            vals.astype(np.float64, copy=False),
        )
        for rows, cols, vals in sector_maps
    ))
    _save_persistent_object(persistent_key, compiled)
    return compiled


def get_cached_tau_sector_linear_map_triplets(
    source_spec,
    positions,
    map_kind: str,
    source_sector_coordinate_dims: Tuple[int, ...],
    Hermitian: bool,
):
    """Return reduced tau maps with sectors kept separate."""
    canonical_positions, transport_rows, transport_signs = _canonicalize_source_positions(
        source_spec,
        positions,
        map_kind,
        Hermitian,
    )
    lexorder = _source_lexorder(source_spec)
    party_signature = _party_signature_from_lexorder(lexorder)
    if map_kind == "partial_trace":
        target_coord_dim, canonical_sector_maps = _cached_canonical_tau_sector_triplets(
            party_signature,
            tuple(source_spec.slot_dims),
            tuple(source_spec.local_symmetry_perms),
            tuple(canonical_positions),
            Hermitian,
        )
    elif map_kind == "partial_transpose":
        target_coord_dim, canonical_sector_maps = _cached_canonical_tau_pt_sector_triplets(
            party_signature,
            tuple(source_spec.slot_dims),
            tuple(source_spec.local_symmetry_perms),
            tuple(canonical_positions),
            Hermitian,
        )
    else:
        raise ValueError(f"Unsupported map kind {map_kind!r}.")

    if transport_rows is None:
        return target_coord_dim, canonical_sector_maps

    transported_sector_maps = []
    for rows, cols, vals in canonical_sector_maps:
        moved_rows = transport_rows[rows].astype(np.int32, copy=False)
        moved_vals = (transport_signs[rows] * vals).astype(np.float64, copy=False)
        if moved_rows.size:
            order = np.argsort(moved_rows, kind="mergesort")
            moved_rows = moved_rows[order]
            cols = cols[order]
            moved_vals = moved_vals[order]
        transported_sector_maps.append((moved_rows, cols, moved_vals))
    return target_coord_dim, tuple(transported_sector_maps)


def _transport_sector_maps_by_slot_permutation(
    sector_maps: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    slot_dims: Tuple[int, ...],
    slot_permutation: Tuple[int, ...] | None,
    Hermitian: bool,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Transport coordinate rows to a different target slot ordering."""
    if slot_permutation is None:
        return sector_maps
    (
        _coord_dim,
        _rows,
        _cols,
        _vals,
        transport_rows,
        transport_signs,
    ) = cached_coordinate_action_for_slot_permutation(
        tuple(slot_dims),
        tuple(int(pos) for pos in slot_permutation),
        Hermitian,
    )
    transported_sector_maps = []
    for rows, cols, vals in sector_maps:
        moved_rows = transport_rows[rows].astype(np.int32, copy=False)
        moved_vals = (transport_signs[rows] * vals).astype(np.float64, copy=False)
        if moved_rows.size:
            order = np.argsort(moved_rows, kind="mergesort")
            moved_rows = moved_rows[order]
            cols = cols[order]
            moved_vals = moved_vals[order]
        transported_sector_maps.append((moved_rows, cols, moved_vals))
    return tuple(transported_sector_maps)


def _append_trace_rows(
    task,
    next_row: int,
    variable_specs: Iterable[Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec],
) -> Tuple[int, int]:
    """Append trace-one constraints for PSD objects."""
    triplet_blocks = []
    row_rhs = []
    row_cursor = next_row
    count = 0
    for specs in variable_specs:
        if isinstance(specs, TaskBarVariableSpec):
            specs = (specs,)
        row_indices, row_cursor = _append_fx_rows(task, row_cursor, np.asarray([1.0], dtype=np.float64))
        row_index = int(row_indices[0])
        for spec in specs:
            diag = np.arange(spec.dim, dtype=np.int32)
            coeff = float(spec.irrep_dim)
            if spec.hermitian:
                coeff *= 0.5
            triplet_blocks.append(
                (
                    np.full(spec.dim, row_index, dtype=np.int32),
                    np.full(spec.dim, spec.barvar_index, dtype=np.int32),
                    diag,
                    diag,
                    np.full(spec.dim, coeff, dtype=np.float64),
                )
            )
        count += 1
    _append_bar_triplets(task, triplet_blocks)
    return row_cursor, count


def _append_full_psd_barvar(
    task,
    name: str,
    kind: str,
    parent_name: str,
    matrix_dim: int,
    next_barvar: int,
    Hermitian: bool,
) -> tuple[TaskBarVariableSpec, int, int]:
    """Append one full PSD bar variable and return its spec."""
    if Hermitian:
        task.appendbarvars([2 * int(matrix_dim)])
        coordinate_dim = int(matrix_dim * matrix_dim)
        bar_dim = 2 * int(matrix_dim)
    else:
        task.appendbarvars([int(matrix_dim)])
        coordinate_dim = int(matrix_dim * (matrix_dim + 1) // 2)
        bar_dim = int(matrix_dim)
    return (
        TaskBarVariableSpec(
            name=name,
            kind=kind,
            parent_name=parent_name,
            barvar_index=next_barvar,
            dim=bar_dim,
            coordinate_dim=coordinate_dim,
            complex_dim=int(matrix_dim),
            hermitian=bool(Hermitian),
        ),
        next_barvar + 1,
        coordinate_dim,
    )


def build_block_task_feasibility_model(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    model_name: str = "GNMETaskStateSDP",
    include_ppt: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = False,
    Hermitian: bool = True,
    verbose: int | None = None,
):
    """Instantiate the GNME draft using MOSEK Task semidefinite variables.

    It keeps the symmetry-reduced tau sectors and compiles representative/PPT
    relations as explicit Task ``barA`` coefficients.
    """
    import mosek

    if isinstance(assigned, AssignedStateSDPDraft):
        model = assigned.model
    else:
        model = assigned
    _ = enforce_known_values

    active_ppt_variables = model.ppt_variables
    active_ppt_constraints = model.ppt_constraints
    if include_ppt:
        active_ppt_variables, active_ppt_constraints = quotient_ppt_constraints(
            model,
            use_source_symmetry=True,
        )
    else:
        active_ppt_variables = tuple()
        active_ppt_constraints = tuple()

    representative_constraints = tuple()
    if include_representatives:
        representative_constraints = quotient_representative_constraints(
            model,
            use_source_symmetry=True,
        )

    real_verbose = _resolve_verbose(verbose, model.verbose)
    total_start = perf_counter()
    _progress_log(real_verbose, 1, f"Building MOSEK Task block feasibility model `{model_name}`...")

    env = mosek.Env()
    task = env.Task()
    task.putobjsense(mosek.objsense.minimize)

    build_profile = {
        "tau_declaration_time": 0.0,
        "auxiliary_declaration_time": 0.0,
        "trace_time": 0.0,
        "representative_time": 0.0,
        "ppt_time": 0.0,
        "total_build_time": 0.0,
    }
    constraint_counts = {
        "trace": 0,
        "internal_symmetry": 0,
        "representative": 0,
        "ppt": 0,
        "ppt_direct": 0,
        "block_parameters": 0,
    }

    tau_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...]] = {}
    tau_sector_coordinate_dims: Dict[str, Tuple[int, ...]] = {}
    tau_variable_specs = {variable.name: variable for variable in model.psd_variables}
    representative_specs = {representative.name: representative for representative in model.shared_representatives}

    representative_anchors: Dict[str, TaskRepresentativeAnchorSpec] = {}
    auxiliary_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    known_representative_bar_variables: Dict[str, TaskBarVariableSpec] = {}
    ppt_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    direct_tau_ppt_sector_maps: Dict[str, Tuple[Tuple[np.ndarray, ...], ...]] = {}
    direct_tau_ppt_row_lookups: Dict[str, np.ndarray] = {}
    basis_matrix_indices: Dict[Tuple[bool, int], np.ndarray] = {}

    next_barvar = 0

    step_start = perf_counter()
    for variable in model.psd_variables:
        layout = cached_tau_layout(
            variable.name,
            variable.lexorder,
            variable.slot_dims,
            variable.local_symmetry_perms,
        )
        sector_specs = []
        sector_coord_dims = []
        for sector in layout.sectors:
            if Hermitian:
                task.appendbarvars([2 * int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * sector.multiplicity)
                bar_dim = 2 * int(sector.multiplicity)
            else:
                task.appendbarvars([int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                bar_dim = int(sector.multiplicity)
            sector_specs.append(
                TaskBarVariableSpec(
                    name=f"{variable.name}_{sector.label}",
                    kind="tau_sector",
                    parent_name=variable.name,
                    barvar_index=next_barvar,
                    dim=bar_dim,
                    coordinate_dim=coordinate_dim,
                    sector_label=sector.label,
                    irrep_dim=int(sector.irrep_dim),
                    complex_dim=int(sector.multiplicity),
                    hermitian=bool(Hermitian),
                )
            )
            next_barvar += 1
            sector_coord_dims.append(coordinate_dim)
            constraint_counts["block_parameters"] += coordinate_dim
        tau_bar_variables[variable.name] = tuple(sector_specs)
        tau_sector_coordinate_dims[variable.name] = tuple(sector_coord_dims)
    build_profile["tau_declaration_time"] = perf_counter() - step_start

    step_start = perf_counter()
    if Hermitian:
        for ppt_variable in active_ppt_variables:
            if ppt_variable.name in ppt_bar_variables:
                continue
            task.appendbarvars([2 * int(ppt_variable.matrix_dim)])
            coordinate_dim = int(ppt_variable.matrix_dim * ppt_variable.matrix_dim)
            ppt_bar_variables[ppt_variable.name] = TaskBarVariableSpec(
                name=ppt_variable.name,
                kind="ppt_hermitian",
                parent_name=ppt_variable.name,
                barvar_index=next_barvar,
                dim=2 * int(ppt_variable.matrix_dim),
                coordinate_dim=coordinate_dim,
                complex_dim=int(ppt_variable.matrix_dim),
                hermitian=True,
            )
            next_barvar += 1
            constraint_counts["block_parameters"] += coordinate_dim
    else:
        for constraint in active_ppt_constraints:
            if constraint.ppt_variable_name in ppt_bar_variables:
                continue
            if constraint.source_variable_name in tau_variable_specs:
                source_spec = tau_variable_specs[constraint.source_variable_name]
                stabilizer_perms = _cached_tau_ppt_stabilizer(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                _stabilizer, orbit_data, row_lookup = _cached_tau_ppt_orbit_reduction(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                representative_rows = tuple(
                    _upper_triangle_coordinate_index(row, col, orbit_data.matrix_dim)
                    for row, col in orbit_data.orbit_representatives
                )
                selected_maps = _cached_exact_small_group_selected_coordinate_maps(
                    _party_signature_from_lexorder(source_spec.lexorder),
                    source_spec.slot_dims,
                    stabilizer_perms,
                    representative_rows,
                )
                stabilizer_layout = cached_tau_layout(
                    constraint.ppt_variable_name,
                    source_spec.lexorder,
                    source_spec.slot_dims,
                    stabilizer_perms,
                )
                sector_specs = []
                for sector in stabilizer_layout.sectors:
                    task.appendbarvars([int(sector.multiplicity)])
                    coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                    sector_specs.append(
                        TaskBarVariableSpec(
                            name=f"{constraint.ppt_variable_name}_{sector.label}",
                            kind="ppt_tau_sector",
                            parent_name=constraint.ppt_variable_name,
                            barvar_index=next_barvar,
                            dim=int(sector.multiplicity),
                            coordinate_dim=coordinate_dim,
                            sector_label=sector.label,
                            irrep_dim=int(sector.irrep_dim),
                            complex_dim=int(sector.multiplicity),
                            hermitian=False,
                        )
                    )
                    next_barvar += 1
                    constraint_counts["block_parameters"] += coordinate_dim
                ppt_bar_variables[constraint.ppt_variable_name] = tuple(sector_specs)
                direct_tau_ppt_row_lookups[constraint.ppt_variable_name] = row_lookup
                if selected_maps is not None:
                    _target_coord_dim, lhs_sector_maps = selected_maps
                else:
                    _target_coord_dim, lhs_sector_maps = _cached_block_partial_trace_coordinate_maps(
                        _party_signature_from_lexorder(source_spec.lexorder),
                        source_spec.slot_dims,
                        stabilizer_perms,
                        tuple(range(len(source_spec.slot_dims))),
                        False,
                    )
                    lhs_sector_maps = _restrict_sector_triplets_to_row_lookup(lhs_sector_maps, row_lookup)
                direct_tau_ppt_sector_maps[constraint.ppt_variable_name] = lhs_sector_maps
            else:
                task.appendbarvars([int(constraint.matrix_dim)])
                coordinate_dim = int(constraint.matrix_dim * (constraint.matrix_dim + 1) // 2)
                ppt_bar_variables[constraint.ppt_variable_name] = TaskBarVariableSpec(
                    name=constraint.ppt_variable_name,
                    kind="ppt_reduced",
                    parent_name=constraint.ppt_variable_name,
                    barvar_index=next_barvar,
                    dim=int(constraint.matrix_dim),
                    coordinate_dim=coordinate_dim,
                    complex_dim=int(constraint.matrix_dim),
                    hermitian=False,
                )
                next_barvar += 1
                constraint_counts["block_parameters"] += coordinate_dim
    build_profile["auxiliary_declaration_time"] = perf_counter() - step_start

    next_row = 0

    step_start = perf_counter()
    trace_specs: list[Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = []
    trace_specs.extend(tau_bar_variables.values())
    next_row, trace_count = _append_trace_rows(task, next_row, trace_specs)
    constraint_counts["trace"] = trace_count
    build_profile["trace_time"] = perf_counter() - step_start

    step_start = perf_counter()
    representative_groups: Dict[str, list] = {}
    for constraint in representative_constraints:
        representative_groups.setdefault(constraint.representative_name, []).append(constraint)

    source_spec_lookup = dict(tau_variable_specs)
    source_spec_lookup.update({name: representative for name, representative in representative_specs.items()})
    source_barvar_lookup: Dict[str, Tuple[TaskBarVariableSpec, ...]] = dict(tau_bar_variables)
    source_sector_coordinate_dims_lookup = dict(tau_sector_coordinate_dims)

    def _resolve_representative_source_maps(constraint):
        source_spec = source_spec_lookup[constraint.source_variable_name]
        source_barvars = source_barvar_lookup[constraint.source_variable_name]
        target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
            source_spec,
            constraint.keep_positions,
            "partial_trace",
            source_sector_coordinate_dims_lookup[constraint.source_variable_name],
            Hermitian=Hermitian,
        )
        sector_maps = _transport_sector_maps_by_slot_permutation(
            sector_maps,
            tuple(source_spec.slot_dims[position] for position in constraint.keep_positions),
            constraint.target_slot_permutation,
            Hermitian=Hermitian,
        )
        return (
            int(target_coord_dim),
            sector_maps,
            source_barvars,
            constraint.source_variable_name,
            constraint.source_variable_kind,
        )

    for representative_name, group in representative_groups.items():
        anchor_constraint = group[0]
        (
            anchor_sector_maps_dim,
            anchor_sector_maps,
            anchor_barvars,
            anchor_name,
            anchor_kind,
        ) = _resolve_representative_source_maps(anchor_constraint)
        anchor_lexorder = anchor_constraint.representative_lexorder
        anchor_keep_positions = anchor_constraint.keep_positions
        anchor_traced_positions = anchor_constraint.traced_positions
        representative_anchors[representative_name] = TaskRepresentativeAnchorSpec(
            representative_name=representative_name,
            representative_kind=anchor_kind,
            representative_lexorder=anchor_lexorder,
            source_variable_name=anchor_name,
            source_variable_kind=anchor_kind,
            keep_positions=anchor_keep_positions,
            traced_positions=anchor_traced_positions,
            coordinate_dim=int(anchor_sector_maps_dim),
            sector_maps=anchor_sector_maps,
        )
        for constraint in group[1:]:
            (
                target_coord_dim,
                sector_maps,
                source_barvars,
                _source_name,
                _source_kind,
            ) = _resolve_representative_source_maps(constraint)
            if int(target_coord_dim) != int(anchor_sector_maps_dim):
                raise ValueError(
                    f"Representative {constraint.representative_name} coordinate mismatch: "
                    f"{target_coord_dim} vs {anchor_sector_maps_dim}."
                )
            row_indices, next_row = _append_fx_rows(
                task,
                next_row,
                np.zeros(int(target_coord_dim), dtype=np.float64),
            )
            term_blocks = []
            for sector_map, sector_barvar in zip(sector_maps, source_barvars):
                basis_key = (bool(sector_barvar.hermitian), int(sector_barvar.complex_dim or sector_barvar.dim))
                basis_indices = basis_matrix_indices.get(basis_key)
                if basis_indices is None:
                    basis_indices = _append_coordinate_basis_matrices(
                        task,
                        int(sector_barvar.complex_dim or sector_barvar.dim),
                        bool(sector_barvar.hermitian),
                    )
                    basis_matrix_indices[basis_key] = basis_indices
                term_blocks.append(
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        sector_barvar.barvar_index,
                        basis_indices,
                        sector_map[0],
                        sector_map[1],
                        sector_map[2],
                    )
                )
            for sector_map, sector_barvar in zip(anchor_sector_maps, anchor_barvars):
                basis_key = (bool(sector_barvar.hermitian), int(sector_barvar.complex_dim or sector_barvar.dim))
                basis_indices = basis_matrix_indices.get(basis_key)
                if basis_indices is None:
                    basis_indices = _append_coordinate_basis_matrices(
                        task,
                        int(sector_barvar.complex_dim or sector_barvar.dim),
                        bool(sector_barvar.hermitian),
                    )
                    basis_matrix_indices[basis_key] = basis_indices
                term_blocks.append(
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        sector_barvar.barvar_index,
                        basis_indices,
                        sector_map[0],
                        sector_map[1],
                        -sector_map[2],
                    )
                )
            _append_bar_terms(task, term_blocks)
            constraint_counts["representative"] += int(target_coord_dim)
    build_profile["representative_time"] = perf_counter() - step_start

    step_start = perf_counter()
    for constraint in active_ppt_constraints:
        if constraint.source_variable_name in tau_variable_specs and not Hermitian:
            source_spec = tau_variable_specs[constraint.source_variable_name]
            source_barvars = tau_bar_variables[constraint.source_variable_name]
            target_barvars = ppt_bar_variables[constraint.ppt_variable_name]
            assert isinstance(target_barvars, tuple)
            row_lookup = direct_tau_ppt_row_lookups[constraint.ppt_variable_name]
            target_coord_dim, rhs_sector_maps = get_cached_tau_sector_linear_map_triplets(
                source_spec,
                constraint.transpose_positions,
                "partial_transpose",
                tau_sector_coordinate_dims[constraint.source_variable_name],
                Hermitian=False,
            )
            rhs_sector_maps = _restrict_sector_triplets_to_row_lookup(rhs_sector_maps, row_lookup)
            lhs_sector_maps = direct_tau_ppt_sector_maps[constraint.ppt_variable_name]
            reduced_coord_dim = int(max(row_lookup[row_lookup >= 0]) + 1) if np.any(row_lookup >= 0) else 0
            if int(target_coord_dim) < reduced_coord_dim:
                raise ValueError("Reduced PPT row count exceeds source coordinate dimension.")
            row_indices, next_row = _append_fx_rows(
                task,
                next_row,
                np.zeros(reduced_coord_dim, dtype=np.float64),
            )
            term_blocks = []
            for sector_map, sector_barvar in zip(lhs_sector_maps, target_barvars):
                basis_key = (False, int(sector_barvar.dim))
                basis_indices = basis_matrix_indices.get(basis_key)
                if basis_indices is None:
                    basis_indices = _append_coordinate_basis_matrices(task, int(sector_barvar.dim), False)
                    basis_matrix_indices[basis_key] = basis_indices
                term_blocks.append(
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        sector_barvar.barvar_index,
                        basis_indices,
                        sector_map[0],
                        sector_map[1],
                        sector_map[2],
                    )
                )
            for sector_map, sector_barvar in zip(rhs_sector_maps, source_barvars):
                basis_key = (False, int(sector_barvar.dim))
                basis_indices = basis_matrix_indices.get(basis_key)
                if basis_indices is None:
                    basis_indices = _append_coordinate_basis_matrices(task, int(sector_barvar.dim), False)
                    basis_matrix_indices[basis_key] = basis_indices
                term_blocks.append(
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        sector_barvar.barvar_index,
                        basis_indices,
                        sector_map[0],
                        sector_map[1],
                        -sector_map[2],
                    )
                )
            _append_bar_terms(task, term_blocks)
            constraint_counts["ppt_direct"] += 1
        else:
            target_spec = ppt_bar_variables[constraint.ppt_variable_name]
            assert isinstance(target_spec, TaskBarVariableSpec)
            if constraint.source_variable_name in tau_variable_specs:
                source_spec = tau_variable_specs[constraint.source_variable_name]
                source_barvars = tau_bar_variables[constraint.source_variable_name]
                target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                    source_spec,
                    constraint.transpose_positions,
                    "partial_transpose",
                    tau_sector_coordinate_dims[constraint.source_variable_name],
                    Hermitian=Hermitian,
                )
                if int(target_coord_dim) != int(target_spec.coordinate_dim):
                    raise ValueError(
                        f"Hermitian tau PPT coordinate mismatch: {target_coord_dim} vs {target_spec.coordinate_dim}."
                    )
                row_indices, next_row = _append_fx_rows(
                    task,
                    next_row,
                    np.zeros(int(target_coord_dim), dtype=np.float64),
                )
                id_rows, id_cols, id_vals = _row_identity_triplets(target_spec.coordinate_dim, scale=1.0)
                target_basis_key = (True, int(target_spec.complex_dim))
                target_basis_indices = basis_matrix_indices.get(target_basis_key)
                if target_basis_indices is None:
                    target_basis_indices = _append_coordinate_basis_matrices(task, int(target_spec.complex_dim), True)
                    basis_matrix_indices[target_basis_key] = target_basis_indices
                term_blocks = [
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        target_spec.barvar_index,
                        target_basis_indices,
                        id_rows,
                        id_cols,
                        id_vals,
                    )
                ]
                for sector_map, sector_barvar in zip(sector_maps, source_barvars):
                    source_basis_key = (True, int(sector_barvar.complex_dim))
                    source_basis_indices = basis_matrix_indices.get(source_basis_key)
                    if source_basis_indices is None:
                        source_basis_indices = _append_coordinate_basis_matrices(task, int(sector_barvar.complex_dim), True)
                        basis_matrix_indices[source_basis_key] = source_basis_indices
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            source_basis_indices,
                            sector_map[0],
                            sector_map[1],
                            -sector_map[2],
                        )
                    )
                _append_bar_terms(task, term_blocks)
            else:
                anchor = representative_anchors[constraint.source_variable_name]
                source_barvars = tau_bar_variables[anchor.source_variable_name]
                (
                    target_coord_dim,
                    rows,
                    cols,
                    vals,
                    *_,
                ) = cached_partial_transpose_coordinate_action(
                    constraint.slot_dims,
                    constraint.transpose_positions,
                    Hermitian=Hermitian,
                )
                if int(target_coord_dim) != int(target_spec.coordinate_dim):
                    raise ValueError(
                        f"Reduced PPT coordinate mismatch: {target_coord_dim} vs {target_spec.coordinate_dim}."
                    )
                source_sector_maps = _compose_coordinate_action_with_sector_maps(
                    rows,
                    cols,
                    vals,
                    anchor.sector_maps,
                )
                row_indices, next_row = _append_fx_rows(
                    task,
                    next_row,
                    np.zeros(int(target_coord_dim), dtype=np.float64),
                )
                id_rows, id_cols, id_vals = _row_identity_triplets(target_spec.coordinate_dim, scale=1.0)
                target_basis_key = (bool(target_spec.hermitian), int(target_spec.complex_dim or target_spec.dim))
                target_basis_indices = basis_matrix_indices.get(target_basis_key)
                if target_basis_indices is None:
                    target_basis_indices = _append_coordinate_basis_matrices(
                        task,
                        int(target_spec.complex_dim or target_spec.dim),
                        bool(target_spec.hermitian),
                    )
                    basis_matrix_indices[target_basis_key] = target_basis_indices
                term_blocks = [
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        target_spec.barvar_index,
                        target_basis_indices,
                        id_rows,
                        id_cols,
                        id_vals,
                    ),
                ]
                for sector_map, sector_barvar in zip(source_sector_maps, source_barvars):
                    source_basis_key = (bool(sector_barvar.hermitian), int(sector_barvar.complex_dim or sector_barvar.dim))
                    source_basis_indices = basis_matrix_indices.get(source_basis_key)
                    if source_basis_indices is None:
                        source_basis_indices = _append_coordinate_basis_matrices(
                            task,
                            int(sector_barvar.complex_dim or sector_barvar.dim),
                            bool(sector_barvar.hermitian),
                        )
                        basis_matrix_indices[source_basis_key] = source_basis_indices
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            source_basis_indices,
                            sector_map[0],
                            sector_map[1],
                            -sector_map[2],
                        )
                    )
                _append_bar_terms(task, term_blocks)
            constraint_counts["ppt_direct"] += 1
    build_profile["ppt_time"] = perf_counter() - step_start

    build_profile["total_build_time"] = perf_counter() - total_start
    _progress_log(
        real_verbose,
        1,
        "Task block build complete: "
        f"{constraint_counts['trace']} trace rows, "
        f"{constraint_counts['representative']} representative rows, "
        f"{constraint_counts['ppt_direct']} PPT blocks in "
        f"{build_profile['total_build_time']:.2f}s.",
    )

    return TaskBlockStateSDPModel(
        env=env,
        task=task,
        Hermitian=bool(Hermitian),
        tau_bar_variables=tau_bar_variables,
        representative_anchors=representative_anchors,
        auxiliary_bar_variables=auxiliary_bar_variables,
        known_representative_bar_variables=known_representative_bar_variables,
        ppt_bar_variables=ppt_bar_variables,
        constraint_counts=constraint_counts,
        build_profile=build_profile,
    )


def build_top_down_block_task_feasibility_model(
    assigned: AssignedStateSDPDraft | TopDownStateSDPDraft,
    model_name: str = "GNMETopDownTaskStateSDP",
    include_ppt: bool = True,
    include_family_ppt: bool = True,
    Hermitian: bool = True,
    verbose: int | None = None,
):
    """Instantiate the fresh top-down draft as an anchored MOSEK Task model.

    The family structure comes from the top-down metadata, but the supported
    anchored scope intentionally stays lean:
    - tau variables stay symmetry-reduced in sectors,
    - maximal/known families are anchored to one tau occurrence,
    - only non-anchor routes emit equality rows,
    - tau PPT uses the direct reduced path,
    - family PPT candidates are quotiented aggressively before build, so only
      irreducible, non-redundant family PPT sources survive when requested.
    """
    import mosek

    if isinstance(assigned, AssignedStateSDPDraft):
        model = assigned.model
        known_values = assigned.known_values
    else:
        model = assigned
        known_values = {}

    if not isinstance(model, TopDownStateSDPDraft):
        raise TypeError("build_top_down_block_task_feasibility_model requires a TopDownStateSDPDraft.")
    real_verbose = _resolve_verbose(verbose, model.verbose)
    total_start = perf_counter()
    _progress_log(real_verbose, 1, f"Building MOSEK Task top-down model `{model_name}`...")

    env = mosek.Env()
    task = env.Task()
    task.putobjsense(mosek.objsense.minimize)

    build_profile = {
        "tau_declaration_time": 0.0,
        "trace_time": 0.0,
        "tau_family_time": 0.0,
        "overlap_time": 0.0,
        "known_value_time": 0.0,
        "ppt_prepare_time": 0.0,
        "ppt_emit_time": 0.0,
        "ppt_time": 0.0,
        "total_build_time": 0.0,
        "scope": {
            "tau_variables": len(model.psd_variables),
            "maximal_representatives": len(model.maximal_representatives),
            "known_representatives": len(model.known_representatives),
            "cross_inflation_groups": len(model.cross_inflation_groups),
            "tau_family_constraints": len(model.tau_representative_constraints),
            "ppt_constraints_total": len(model.ppt_constraints),
        },
        "tau_declaration_stats": {},
        "tau_map_cache": {},
        "tau_family_stats": {},
        "known_value_stats": {},
        "ppt_stats": {},
    }
    constraint_counts = {
        "trace": 0,
        "internal_symmetry": 0,
        "representative": 0,
        "known_value": 0,
        "ppt_direct": 0,
        "block_parameters": 0,
    }

    tau_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...]] = {}
    tau_sector_coordinate_dims: Dict[str, Tuple[int, ...]] = {}
    auxiliary_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    known_representative_bar_variables: Dict[str, TaskBarVariableSpec] = {}
    representative_anchors: Dict[str, TaskRepresentativeAnchorSpec] = {}
    representative_anchor_source_entries: Dict[str, Dict[str, object]] = {}
    ppt_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    basis_matrix_indices: Dict[Tuple[bool, int], np.ndarray] = {}
    tau_map_cache_stats = {
        "hits": 0,
        "misses": 0,
        "entries": 0,
        "positive_templates": 0,
        "negative_templates": 0,
        "template_weights": 0,
        "template_rows": 0,
    }

    next_barvar = 0
    tau_variable_specs = {variable.name: variable for variable in model.psd_variables}
    representative_catalog = model.maximal_representatives + model.known_representatives
    known_representatives = model.known_representatives
    representative_specs = {
        representative.name: representative
        for representative in representative_catalog
    }

    def _basis_indices_for(spec: TaskBarVariableSpec) -> np.ndarray:
        key = (bool(spec.hermitian), int(spec.complex_dim or spec.dim))
        basis_indices = basis_matrix_indices.get(key)
        if basis_indices is None:
            basis_indices = _append_coordinate_basis_matrices(
                task,
                int(spec.complex_dim or spec.dim),
                bool(spec.hermitian),
            )
            basis_matrix_indices[key] = basis_indices
        return basis_indices

    resolved_tau_source_maps: Dict[
        Tuple[str, str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...] | None],
        Dict[str, object],
    ] = {}

    def _map_cache_key(constraint: object):
        target_slot_permutation = getattr(constraint, "target_slot_permutation", None)
        if target_slot_permutation is not None:
            target_slot_permutation = tuple(int(pos) for pos in target_slot_permutation)
        return (
            str(constraint.source_variable_kind),
            str(constraint.source_variable_name),
            tuple(int(pos) for pos in constraint.keep_positions),
            tuple(int(pos) for pos in constraint.traced_positions),
            target_slot_permutation,
        )

    def _build_sector_map_templates(
        sector_maps: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        source_barvars: Tuple[TaskBarVariableSpec, ...],
        scale: float,
    ) -> Tuple[TaskBarTermTemplate, ...]:
        templates = []
        for sector_map, sector_barvar in zip(sector_maps, source_barvars):
            vals = np.asarray(sector_map[2], dtype=np.float64)
            if scale != 1.0:
                vals = (float(scale) * vals).astype(np.float64, copy=False)
            templates.append(
                _coordinate_triplets_to_bar_term_template(
                    sector_barvar.barvar_index,
                    _basis_indices_for(sector_barvar),
                    sector_map[0],
                    sector_map[1],
                    vals,
                )
            )
        return tuple(templates)

    def _resolve_tau_source_maps(constraint: object):
        cache_key = _map_cache_key(constraint)
        cached = resolved_tau_source_maps.get(cache_key)
        if cached is not None:
            tau_map_cache_stats["hits"] += 1
            return cached

        source_spec = tau_variable_specs[constraint.source_variable_name]
        source_barvars = tau_bar_variables[constraint.source_variable_name]
        target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
            source_spec,
            constraint.keep_positions,
            "partial_trace",
            tau_sector_coordinate_dims[constraint.source_variable_name],
            Hermitian=Hermitian,
        )
        sector_maps = _transport_sector_maps_by_slot_permutation(
            sector_maps,
            tuple(source_spec.slot_dims[position] for position in constraint.keep_positions),
            constraint.target_slot_permutation,
            Hermitian=Hermitian,
        )
        cached = {
            "target_coord_dim": int(target_coord_dim),
            "sector_maps": sector_maps,
            "source_barvars": source_barvars,
            "positive_templates": _build_sector_map_templates(sector_maps, source_barvars, 1.0),
            "negative_templates": _build_sector_map_templates(sector_maps, source_barvars, -1.0),
        }
        resolved_tau_source_maps[cache_key] = cached
        tau_map_cache_stats["misses"] += 1
        tau_map_cache_stats["entries"] = len(resolved_tau_source_maps)
        tau_map_cache_stats["positive_templates"] += len(cached["positive_templates"])
        tau_map_cache_stats["negative_templates"] += len(cached["negative_templates"])
        tau_map_cache_stats["template_weights"] += sum(
            int(template.weights.size)
            for template in (
                tuple(cached["positive_templates"]) + tuple(cached["negative_templates"])
            )
        )
        tau_map_cache_stats["template_rows"] += sum(
            int(template.relative_rows.size)
            for template in (
                tuple(cached["positive_templates"]) + tuple(cached["negative_templates"])
            )
        )
        return cached

    step_start = perf_counter()
    _progress_log(
        real_verbose,
        1,
        f"Top-down Task step 1/5: declaring {len(model.psd_variables)} tau variables...",
    )
    tau_sector_total = 0
    for variable in model.psd_variables:
        layout = cached_tau_layout(
            variable.name,
            variable.lexorder,
            variable.slot_dims,
            variable.local_symmetry_perms,
        )
        sector_specs = []
        sector_coord_dims = []
        for sector in layout.sectors:
            if Hermitian:
                task.appendbarvars([2 * int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * sector.multiplicity)
                bar_dim = 2 * int(sector.multiplicity)
            else:
                task.appendbarvars([int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                bar_dim = int(sector.multiplicity)
            sector_specs.append(
                TaskBarVariableSpec(
                    name=f"{variable.name}_{sector.label}",
                    kind="tau_sector",
                    parent_name=variable.name,
                    barvar_index=next_barvar,
                    dim=bar_dim,
                    coordinate_dim=coordinate_dim,
                    sector_label=sector.label,
                    irrep_dim=int(sector.irrep_dim),
                    complex_dim=int(sector.multiplicity),
                    hermitian=bool(Hermitian),
                )
            )
            next_barvar += 1
            sector_coord_dims.append(coordinate_dim)
            constraint_counts["block_parameters"] += coordinate_dim
            tau_sector_total += 1
        tau_bar_variables[variable.name] = tuple(sector_specs)
        tau_sector_coordinate_dims[variable.name] = tuple(sector_coord_dims)
    build_profile["tau_declaration_time"] = perf_counter() - step_start
    build_profile["tau_declaration_stats"] = {
        "variables": len(model.psd_variables),
        "sectors": tau_sector_total,
    }
    _progress_log(
        real_verbose,
        1,
        f"Top-down Task step 1/5 complete: {len(model.psd_variables)} tau variables, "
        f"{tau_sector_total} sectors in {build_profile['tau_declaration_time']:.2f}s.",
    )

    next_row = 0

    step_start = perf_counter()
    _progress_log(real_verbose, 1, "Top-down Task step 2/5: emitting trace constraints...")
    trace_specs: list[Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = []
    trace_specs.extend(tau_bar_variables.values())
    next_row, trace_count = _append_trace_rows(task, next_row, trace_specs)
    constraint_counts["trace"] = trace_count
    build_profile["trace_time"] = perf_counter() - step_start
    _progress_log(
        real_verbose,
        1,
        f"Top-down Task step 2/5 complete: {trace_count} trace rows in "
        f"{build_profile['trace_time']:.2f}s.",
    )

    step_start = perf_counter()
    if not model.cross_inflation_groups and model.tau_representative_constraints:
        raise ValueError(
            "Anchored Task build requires precomputed cross-inflation "
            "groups in the draft."
        )
    total_cross_groups = len(model.cross_inflation_groups)
    total_non_anchor_routes = sum(
        len(group.non_anchor_constraints) for group in model.cross_inflation_groups
    )
    tau_family_stats = {
        "groups_total": total_cross_groups,
        "non_anchor_routes_total": total_non_anchor_routes,
        "anchors_resolved": 0,
        "non_anchor_routes_emitted": 0,
        "rows_emitted": 0,
    }
    _progress_log(
        real_verbose,
        1,
        "Top-down Task step 3/5: anchored cross-inflation equalities for "
        f"{total_cross_groups} groups and {total_non_anchor_routes} non-anchor routes...",
    )
    group_log_time = step_start
    for group in model.cross_inflation_groups:
        anchor_constraint = group.anchor_constraint
        representative_name = group.representative_name
        non_anchor_constraints = group.non_anchor_constraints

        anchor_entry = _resolve_tau_source_maps(anchor_constraint)
        anchor_coord_dim = int(anchor_entry["target_coord_dim"])
        anchor_sector_maps = anchor_entry["sector_maps"]
        representative_anchor_source_entries[representative_name] = anchor_entry
        tau_family_stats["anchors_resolved"] += 1
        representative_anchors[representative_name] = TaskRepresentativeAnchorSpec(
            representative_name=representative_name,
            representative_kind=anchor_constraint.representative_kind,
            representative_lexorder=anchor_constraint.representative_lexorder,
            source_variable_name=anchor_constraint.source_variable_name,
            source_variable_kind=anchor_constraint.source_variable_kind,
            keep_positions=anchor_constraint.keep_positions,
            traced_positions=anchor_constraint.traced_positions,
            coordinate_dim=int(anchor_coord_dim),
            sector_maps=anchor_sector_maps,
        )
        for constraint in non_anchor_constraints:
            source_entry = _resolve_tau_source_maps(constraint)
            if int(source_entry["target_coord_dim"]) != int(anchor_coord_dim):
                raise ValueError(
                    f"Top-down anchor coordinate mismatch for {representative_name}: "
                    f"{source_entry['target_coord_dim']} vs {anchor_coord_dim}."
                )
            row_indices, next_row = _append_fx_rows(
                task,
                next_row,
                np.zeros(int(source_entry["target_coord_dim"]), dtype=np.float64),
            )
            row_offset = int(row_indices[0])
            _append_bar_terms(
                task,
                tuple(
                    _instantiate_bar_term_template(row_offset, template)
                    for template in source_entry["positive_templates"]
                )
                + tuple(
                    _instantiate_bar_term_template(row_offset, template)
                    for template in anchor_entry["negative_templates"]
                ),
            )
            constraint_counts["representative"] += int(source_entry["target_coord_dim"])
            tau_family_stats["non_anchor_routes_emitted"] += 1
            tau_family_stats["rows_emitted"] += int(source_entry["target_coord_dim"])
        group_log_time = _maybe_log_loop_progress(
            real_verbose,
            "Top-down Task step 3/5 groups",
            tau_family_stats["anchors_resolved"],
            max(total_cross_groups, 1),
            step_start,
            group_log_time,
            extra=(
                f"routes={tau_family_stats['non_anchor_routes_emitted']}/"
                f"{total_non_anchor_routes}, rows={tau_family_stats['rows_emitted']}, "
                f"map-cache={tau_map_cache_stats['hits']}/{tau_map_cache_stats['misses']}"
            ),
        )
    build_profile["tau_family_time"] = perf_counter() - step_start
    build_profile["overlap_time"] = 0.0
    build_profile["tau_family_stats"] = tau_family_stats
    build_profile["tau_map_cache"] = dict(tau_map_cache_stats)
    _progress_log(
        real_verbose,
        1,
        "Top-down Task step 3/5 complete: "
        f"{tau_family_stats['non_anchor_routes_emitted']} routes, "
        f"{tau_family_stats['rows_emitted']} rows in "
        f"{build_profile['tau_family_time']:.2f}s.",
    )

    step_start = perf_counter()
    known_value_stats = {
        "representatives_total": len(known_values),
        "representatives_emitted": 0,
        "rows_emitted": 0,
    }
    if known_values:
        _progress_log(
            real_verbose,
            1,
            f"Top-down Task step 4/5: emitting {len(known_values)} known-value anchors...",
        )
    known_log_time = step_start
    if known_values:
        for representative in known_representatives:
            assignment = known_values[representative.name]
            anchor = representative_anchors[representative.name]
            anchor_entry = representative_anchor_source_entries[representative.name]
            rhs = np.asarray(assignment.matrix, dtype=np.complex128 if Hermitian else np.float64)
            if rhs.shape != (representative.matrix_dim, representative.matrix_dim):
                raise ValueError(
                    f"Known value for {representative.name} has shape {rhs.shape}, "
                    f"expected {(representative.matrix_dim, representative.matrix_dim)}."
                )
            rhs_coord = []
            for row in range(representative.matrix_dim):
                for col in range(row, representative.matrix_dim):
                    rhs_coord.append(float(np.real(rhs[row, col])))
            if Hermitian:
                for row in range(representative.matrix_dim):
                    for col in range(row + 1, representative.matrix_dim):
                        rhs_coord.append(float(np.imag(rhs[row, col])))
            rhs_coord = np.asarray(rhs_coord, dtype=np.float64)
            row_indices, next_row = _append_fx_rows(task, next_row, rhs_coord)
            row_offset = int(row_indices[0])
            _append_bar_terms(
                task,
                tuple(
                    _instantiate_bar_term_template(row_offset, template)
                    for template in anchor_entry["positive_templates"]
                ),
            )
            constraint_counts["known_value"] += int(anchor.coordinate_dim)
            known_value_stats["representatives_emitted"] += 1
            known_value_stats["rows_emitted"] += int(anchor.coordinate_dim)
            known_log_time = _maybe_log_loop_progress(
                real_verbose,
                "Top-down Task step 4/5 known values",
                known_value_stats["representatives_emitted"],
                max(known_value_stats["representatives_total"], 1),
                step_start,
                known_log_time,
                extra=f"rows={known_value_stats['rows_emitted']}",
            )
    build_profile["known_value_time"] = perf_counter() - step_start
    build_profile["known_value_stats"] = known_value_stats
    if known_values:
        _progress_log(
            real_verbose,
            1,
            "Top-down Task step 4/5 complete: "
            f"{known_value_stats['representatives_emitted']} known representatives, "
            f"{known_value_stats['rows_emitted']} rows in "
            f"{build_profile['known_value_time']:.2f}s.",
        )

    step_start = perf_counter()
    active_ppt_variables = tuple()
    active_ppt_constraints = tuple()
    if include_ppt:
        active_ppt_variables, active_ppt_constraints = quotient_ppt_constraints(
            model,
            use_source_symmetry=True,
            drop_representative_constraints_implied_by_tau=True,
            drop_factor_reducible_representative_ppts=True,
        )
        if not include_family_ppt:
            active_ppt_constraints = tuple(
                constraint
                for constraint in active_ppt_constraints
                if constraint.source_variable_kind == "tau"
            )
            active_ppt_variable_names = {
                constraint.ppt_variable_name for constraint in active_ppt_constraints
            }
            active_ppt_variables = tuple(
                variable
                for variable in active_ppt_variables
                if variable.name in active_ppt_variable_names
            )
    ppt_stats = {
        "raw_total": len(model.ppt_constraints),
        "raw_tau": sum(1 for constraint in model.ppt_constraints if constraint.source_variable_kind == "tau"),
        "raw_family": sum(1 for constraint in model.ppt_constraints if constraint.source_variable_kind != "tau"),
        "active_total": len(active_ppt_constraints),
        "active_tau": sum(1 for constraint in active_ppt_constraints if constraint.source_variable_kind == "tau"),
        "active_family": sum(1 for constraint in active_ppt_constraints if constraint.source_variable_kind != "tau"),
        "prepared_total": 0,
        "prepared_tau": 0,
        "prepared_family": 0,
        "emitted_total": 0,
        "emitted_tau": 0,
        "emitted_family": 0,
        "tau_rows_emitted": 0,
        "family_rows_emitted": 0,
    }
    if include_ppt:
        _progress_log(
            real_verbose,
            1,
            "Top-down Task step 5/5: preparing "
            f"{ppt_stats['active_total']} PPT constraints "
            f"({ppt_stats['active_tau']} tau, {ppt_stats['active_family']} family)...",
        )

    if include_ppt:
        ppt_prepare_start = perf_counter()
        direct_tau_ppt_sector_maps: Dict[str, Tuple[Tuple[np.ndarray, ...], ...]] = {}
        direct_tau_ppt_row_lookups: Dict[str, np.ndarray] = {}
        ppt_prepare_log_time = ppt_prepare_start
        for constraint in active_ppt_constraints:
            if constraint.source_variable_kind == "tau" and not Hermitian:
                source_spec = tau_variable_specs[constraint.source_variable_name]
                stabilizer_perms = _cached_tau_ppt_stabilizer(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                _stabilizer, orbit_data, row_lookup = _cached_tau_ppt_orbit_reduction(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                representative_rows = tuple(
                    _upper_triangle_coordinate_index(row, col, orbit_data.matrix_dim)
                    for row, col in orbit_data.orbit_representatives
                )
                selected_maps = _cached_exact_small_group_selected_coordinate_maps(
                    _party_signature_from_lexorder(source_spec.lexorder),
                    source_spec.slot_dims,
                    stabilizer_perms,
                    representative_rows,
                )
                stabilizer_layout = cached_tau_layout(
                    constraint.ppt_variable_name,
                    source_spec.lexorder,
                    source_spec.slot_dims,
                    stabilizer_perms,
                )
                sector_specs = []
                for sector in stabilizer_layout.sectors:
                    task.appendbarvars([int(sector.multiplicity)])
                    coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                    sector_specs.append(
                        TaskBarVariableSpec(
                            name=f"{constraint.ppt_variable_name}_{sector.label}",
                            kind="ppt_tau_sector",
                            parent_name=constraint.ppt_variable_name,
                            barvar_index=next_barvar,
                            dim=int(sector.multiplicity),
                            coordinate_dim=coordinate_dim,
                            sector_label=sector.label,
                            irrep_dim=int(sector.irrep_dim),
                            complex_dim=int(sector.multiplicity),
                            hermitian=False,
                        )
                    )
                    next_barvar += 1
                    constraint_counts["block_parameters"] += coordinate_dim
                ppt_bar_variables[constraint.ppt_variable_name] = tuple(sector_specs)
                direct_tau_ppt_row_lookups[constraint.ppt_variable_name] = row_lookup
                if selected_maps is not None:
                    _target_coord_dim, lhs_sector_maps = selected_maps
                else:
                    _target_coord_dim, lhs_sector_maps = _cached_block_partial_trace_coordinate_maps(
                        _party_signature_from_lexorder(source_spec.lexorder),
                        source_spec.slot_dims,
                        stabilizer_perms,
                        tuple(range(len(source_spec.slot_dims))),
                        False,
                    )
                    lhs_sector_maps = _restrict_sector_triplets_to_row_lookup(lhs_sector_maps, row_lookup)
                direct_tau_ppt_sector_maps[constraint.ppt_variable_name] = lhs_sector_maps
                ppt_stats["prepared_tau"] += 1
            else:
                target_spec, next_barvar, coordinate_dim = _append_full_psd_barvar(
                    task,
                    name=constraint.ppt_variable_name,
                    kind="ppt_reduced",
                    parent_name=constraint.ppt_variable_name,
                    matrix_dim=constraint.matrix_dim,
                    next_barvar=next_barvar,
                    Hermitian=Hermitian,
                )
                ppt_bar_variables[constraint.ppt_variable_name] = target_spec
                constraint_counts["block_parameters"] += coordinate_dim
                ppt_stats["prepared_family"] += 1
            ppt_stats["prepared_total"] += 1
            ppt_prepare_log_time = _maybe_log_loop_progress(
                real_verbose,
                "Top-down Task step 5/5 prepare",
                ppt_stats["prepared_total"],
                max(ppt_stats["active_total"], 1),
                ppt_prepare_start,
                ppt_prepare_log_time,
                extra=(
                    f"tau={ppt_stats['prepared_tau']}, "
                    f"family={ppt_stats['prepared_family']}"
                ),
            )
        build_profile["ppt_prepare_time"] = perf_counter() - ppt_prepare_start

    if include_ppt:
        ppt_emit_start = perf_counter()
        ppt_emit_log_time = ppt_emit_start
        for constraint in active_ppt_constraints:
            if constraint.source_variable_kind == "tau":
                if not Hermitian:
                    source_spec = tau_variable_specs[constraint.source_variable_name]
                    source_barvars = tau_bar_variables[constraint.source_variable_name]
                    target_barvars = ppt_bar_variables[constraint.ppt_variable_name]
                    assert isinstance(target_barvars, tuple)
                    row_lookup = direct_tau_ppt_row_lookups[constraint.ppt_variable_name]
                    target_coord_dim, rhs_sector_maps = get_cached_tau_sector_linear_map_triplets(
                        source_spec,
                        constraint.transpose_positions,
                        "partial_transpose",
                        tau_sector_coordinate_dims[constraint.source_variable_name],
                        Hermitian=False,
                    )
                    rhs_sector_maps = _restrict_sector_triplets_to_row_lookup(rhs_sector_maps, row_lookup)
                    lhs_sector_maps = direct_tau_ppt_sector_maps[constraint.ppt_variable_name]
                    reduced_coord_dim = int(max(row_lookup[row_lookup >= 0]) + 1) if np.any(row_lookup >= 0) else 0
                    if int(target_coord_dim) < reduced_coord_dim:
                        raise ValueError("Reduced top-down tau PPT row count exceeds source coordinate dimension.")
                    row_indices, next_row = _append_fx_rows(
                        task,
                        next_row,
                        np.zeros(reduced_coord_dim, dtype=np.float64),
                    )
                    term_blocks = []
                    for sector_map, sector_barvar in zip(lhs_sector_maps, target_barvars):
                        term_blocks.append(
                            _coordinate_triplets_to_bar_terms(
                                int(row_indices[0]),
                                sector_barvar.barvar_index,
                                _basis_indices_for(sector_barvar),
                                sector_map[0],
                                sector_map[1],
                                sector_map[2],
                            )
                        )
                    for sector_map, sector_barvar in zip(rhs_sector_maps, source_barvars):
                        term_blocks.append(
                            _coordinate_triplets_to_bar_terms(
                                int(row_indices[0]),
                                sector_barvar.barvar_index,
                                _basis_indices_for(sector_barvar),
                                sector_map[0],
                                sector_map[1],
                                -sector_map[2],
                            )
                        )
                    _append_bar_terms(task, term_blocks)
                    ppt_stats["tau_rows_emitted"] += int(reduced_coord_dim)
                else:
                    target_spec = ppt_bar_variables[constraint.ppt_variable_name]
                    assert isinstance(target_spec, TaskBarVariableSpec)
                    id_rows, id_cols, id_vals = _row_identity_triplets(target_spec.coordinate_dim, scale=1.0)
                    source_spec = tau_variable_specs[constraint.source_variable_name]
                    source_barvars = tau_bar_variables[constraint.source_variable_name]
                    target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                        source_spec,
                        constraint.transpose_positions,
                        "partial_transpose",
                        tau_sector_coordinate_dims[constraint.source_variable_name],
                        Hermitian=Hermitian,
                    )
                    row_indices, next_row = _append_fx_rows(
                        task,
                        next_row,
                        np.zeros(int(target_coord_dim), dtype=np.float64),
                    )
                    term_blocks = [
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            target_spec.barvar_index,
                            _basis_indices_for(target_spec),
                            id_rows,
                            id_cols,
                            id_vals,
                        )
                    ]
                    for sector_map, sector_barvar in zip(sector_maps, source_barvars):
                        term_blocks.append(
                            _coordinate_triplets_to_bar_terms(
                                int(row_indices[0]),
                                sector_barvar.barvar_index,
                                _basis_indices_for(sector_barvar),
                                sector_map[0],
                                sector_map[1],
                                -sector_map[2],
                            )
                        )
                    _append_bar_terms(task, term_blocks)
                    ppt_stats["tau_rows_emitted"] += int(target_coord_dim)
            else:
                target_spec = ppt_bar_variables[constraint.ppt_variable_name]
                assert isinstance(target_spec, TaskBarVariableSpec)
                id_rows, id_cols, id_vals = _row_identity_triplets(target_spec.coordinate_dim, scale=1.0)
                source_rep = representative_specs[constraint.source_variable_name]
                anchor = representative_anchors[constraint.source_variable_name]
                source_barvars = tau_bar_variables[anchor.source_variable_name]
                target_coord_dim, rows, cols, vals, *_ = cached_partial_transpose_coordinate_action(
                    tuple(source_rep.slot_dims),
                    tuple(int(pos) for pos in constraint.transpose_positions),
                    Hermitian=Hermitian,
                )
                source_sector_maps = _compose_coordinate_action_with_sector_maps(
                    rows,
                    cols,
                    vals,
                    anchor.sector_maps,
                )
                row_indices, next_row = _append_fx_rows(
                    task,
                    next_row,
                    np.zeros(int(target_coord_dim), dtype=np.float64),
                )
                term_blocks = [
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        target_spec.barvar_index,
                        _basis_indices_for(target_spec),
                        id_rows,
                        id_cols,
                        id_vals,
                    ),
                ]
                for sector_map, sector_barvar in zip(source_sector_maps, source_barvars):
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            _basis_indices_for(sector_barvar),
                            sector_map[0],
                            sector_map[1],
                            -sector_map[2],
                        )
                    )
                _append_bar_terms(task, term_blocks)
                ppt_stats["family_rows_emitted"] += int(target_coord_dim)
            constraint_counts["ppt_direct"] += 1
            ppt_stats["emitted_total"] += 1
            if constraint.source_variable_kind == "tau":
                ppt_stats["emitted_tau"] += 1
            else:
                ppt_stats["emitted_family"] += 1
            ppt_emit_log_time = _maybe_log_loop_progress(
                real_verbose,
                "Top-down Task step 5/5 emit",
                ppt_stats["emitted_total"],
                max(ppt_stats["active_total"], 1),
                ppt_emit_start,
                ppt_emit_log_time,
                extra=(
                    f"tau={ppt_stats['emitted_tau']}, family={ppt_stats['emitted_family']}, "
                    f"rows={ppt_stats['tau_rows_emitted'] + ppt_stats['family_rows_emitted']}"
                ),
            )
        build_profile["ppt_emit_time"] = perf_counter() - ppt_emit_start
    build_profile["ppt_time"] = perf_counter() - step_start
    build_profile["ppt_stats"] = ppt_stats
    if include_ppt:
        _progress_log(
            real_verbose,
            1,
            "Top-down Task step 5/5 complete: "
            f"{ppt_stats['emitted_total']} PPT constraints, "
            f"{ppt_stats['tau_rows_emitted'] + ppt_stats['family_rows_emitted']} rows in "
            f"{build_profile['ppt_time']:.2f}s.",
        )

    build_profile["total_build_time"] = perf_counter() - total_start
    _progress_log(
        real_verbose,
        1,
        "Top-down Task build complete: "
        f"{constraint_counts['trace']} trace rows, "
        f"{constraint_counts['representative']} representative rows, "
        f"{constraint_counts['ppt_direct']} PPT blocks in "
        f"{build_profile['total_build_time']:.2f}s.",
    )

    return TaskBlockStateSDPModel(
        env=env,
        task=task,
        Hermitian=bool(Hermitian),
        tau_bar_variables=tau_bar_variables,
        representative_anchors=representative_anchors,
        auxiliary_bar_variables=auxiliary_bar_variables,
        known_representative_bar_variables=known_representative_bar_variables,
        ppt_bar_variables=ppt_bar_variables,
        constraint_counts=constraint_counts,
        build_profile=build_profile,
    )


def build_paper_block_task_feasibility_model(
    assigned: AssignedStateSDPDraft | PaperStateSDPDraft,
    model_name: str = "GNMEPaperTaskStateSDP",
    include_ppt: bool = True,
    include_family_ppt: bool = True,
    Hermitian: bool = True,
    verbose: int | None = None,
):
    """Instantiate the direct paper-style draft as a MOSEK Task model."""
    import mosek

    if isinstance(assigned, AssignedStateSDPDraft):
        model = assigned.model
    else:
        model = assigned
    if not isinstance(model, PaperStateSDPDraft):
        raise TypeError("build_paper_block_task_feasibility_model requires a PaperStateSDPDraft.")

    real_verbose = _resolve_verbose(verbose, model.verbose)
    total_start = perf_counter()
    _progress_log(real_verbose, 1, f"Building MOSEK Task paper model `{model_name}`...")

    env = mosek.Env()
    task = env.Task()
    task.putobjsense(mosek.objsense.minimize)

    build_profile = {
        "tau_declaration_time": 0.0,
        "trace_time": 0.0,
        "observed_setup_time": 0.0,
        "equality_time": 0.0,
        "ppt_prepare_time": 0.0,
        "ppt_emit_time": 0.0,
        "ppt_time": 0.0,
        "total_build_time": 0.0,
        "scope": {
            "formulation": "paper",
            "tau_variables": len(model.psd_variables),
            "observed_constraints": len(model.observed_constraints),
            "equality_constraints": len(model.equality_constraints),
            "ppt_constraints_total": len(model.ppt_constraints),
        },
        "tau_declaration_stats": {},
        "tau_map_cache": {},
        "observed_stats": {},
        "equality_stats": {},
        "ppt_stats": {},
    }
    constraint_counts = {
        "trace": 0,
        "internal_symmetry": 0,
        "representative": 0,
        "known_value": 0,
        "ppt_direct": 0,
        "block_parameters": 0,
    }

    tau_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...]] = {}
    tau_sector_coordinate_dims: Dict[str, Tuple[int, ...]] = {}
    auxiliary_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    known_representative_bar_variables: Dict[str, TaskBarVariableSpec] = {}
    representative_anchors: Dict[str, TaskRepresentativeAnchorSpec] = {}
    ppt_bar_variables: Dict[str, Tuple[TaskBarVariableSpec, ...] | TaskBarVariableSpec] = {}
    basis_matrix_indices: Dict[Tuple[bool, int], np.ndarray] = {}
    tau_map_cache_stats = {
        "hits": 0,
        "misses": 0,
        "entries": 0,
        "positive_templates": 0,
        "negative_templates": 0,
        "template_weights": 0,
        "template_rows": 0,
    }

    next_barvar = 0
    tau_variable_specs = {variable.name: variable for variable in model.psd_variables}

    def _basis_indices_for(spec: TaskBarVariableSpec) -> np.ndarray:
        key = (bool(spec.hermitian), int(spec.complex_dim or spec.dim))
        basis_indices = basis_matrix_indices.get(key)
        if basis_indices is None:
            basis_indices = _append_coordinate_basis_matrices(
                task,
                int(spec.complex_dim or spec.dim),
                bool(spec.hermitian),
            )
            basis_matrix_indices[key] = basis_indices
        return basis_indices

    resolved_view_maps: Dict[
        Tuple[str, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...] | None],
        Dict[str, object],
    ] = {}

    def _build_sector_map_templates(
        sector_maps: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        source_barvars: Tuple[TaskBarVariableSpec, ...],
        scale: float,
    ) -> Tuple[TaskBarTermTemplate, ...]:
        templates = []
        for sector_map, sector_barvar in zip(sector_maps, source_barvars):
            vals = np.asarray(sector_map[2], dtype=np.float64)
            if scale != 1.0:
                vals = (float(scale) * vals).astype(np.float64, copy=False)
            templates.append(
                _coordinate_triplets_to_bar_term_template(
                    sector_barvar.barvar_index,
                    _basis_indices_for(sector_barvar),
                    sector_map[0],
                    sector_map[1],
                    vals,
                )
            )
        return tuple(templates)

    def _resolve_view_maps(view):
        target_slot_permutation = view.target_slot_permutation
        if target_slot_permutation is not None:
            target_slot_permutation = tuple(int(pos) for pos in target_slot_permutation)
        cache_key = (
            str(view.source_variable_name),
            tuple(int(pos) for pos in view.keep_positions),
            tuple(int(pos) for pos in view.traced_positions),
            target_slot_permutation,
        )
        cached = resolved_view_maps.get(cache_key)
        if cached is not None:
            tau_map_cache_stats["hits"] += 1
            return cached

        source_spec = tau_variable_specs[view.source_variable_name]
        source_barvars = tau_bar_variables[view.source_variable_name]
        target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
            source_spec,
            view.keep_positions,
            "partial_trace",
            tau_sector_coordinate_dims[view.source_variable_name],
            Hermitian=Hermitian,
        )
        sector_maps = _transport_sector_maps_by_slot_permutation(
            sector_maps,
            tuple(source_spec.slot_dims[position] for position in view.keep_positions),
            view.target_slot_permutation,
            Hermitian=Hermitian,
        )
        cached = {
            "target_coord_dim": int(target_coord_dim),
            "sector_maps": sector_maps,
            "source_barvars": source_barvars,
            "positive_templates": _build_sector_map_templates(sector_maps, source_barvars, 1.0),
            "negative_templates": _build_sector_map_templates(sector_maps, source_barvars, -1.0),
        }
        resolved_view_maps[cache_key] = cached
        tau_map_cache_stats["misses"] += 1
        tau_map_cache_stats["entries"] = len(resolved_view_maps)
        tau_map_cache_stats["positive_templates"] += len(cached["positive_templates"])
        tau_map_cache_stats["negative_templates"] += len(cached["negative_templates"])
        tau_map_cache_stats["template_weights"] += sum(
            int(template.weights.size)
            for template in (tuple(cached["positive_templates"]) + tuple(cached["negative_templates"]))
        )
        tau_map_cache_stats["template_rows"] += sum(
            int(template.relative_rows.size)
            for template in (tuple(cached["positive_templates"]) + tuple(cached["negative_templates"]))
        )
        return cached

    step_start = perf_counter()
    _progress_log(real_verbose, 1, f"Paper Task step 1/5: declaring {len(model.psd_variables)} tau variables...")
    tau_sector_total = 0
    for variable in model.psd_variables:
        layout = cached_tau_layout(
            variable.name,
            variable.lexorder,
            variable.slot_dims,
            variable.local_symmetry_perms,
        )
        sector_specs = []
        sector_coord_dims = []
        for sector in layout.sectors:
            if Hermitian:
                task.appendbarvars([2 * int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * sector.multiplicity)
                bar_dim = 2 * int(sector.multiplicity)
            else:
                task.appendbarvars([int(sector.multiplicity)])
                coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                bar_dim = int(sector.multiplicity)
            sector_specs.append(
                TaskBarVariableSpec(
                    name=f"{variable.name}_{sector.label}",
                    kind="tau_sector",
                    parent_name=variable.name,
                    barvar_index=next_barvar,
                    dim=bar_dim,
                    coordinate_dim=coordinate_dim,
                    sector_label=sector.label,
                    irrep_dim=int(sector.irrep_dim),
                    complex_dim=int(sector.multiplicity),
                    hermitian=bool(Hermitian),
                )
            )
            next_barvar += 1
            sector_coord_dims.append(coordinate_dim)
            constraint_counts["block_parameters"] += coordinate_dim
            tau_sector_total += 1
        tau_bar_variables[variable.name] = tuple(sector_specs)
        tau_sector_coordinate_dims[variable.name] = tuple(sector_coord_dims)
    build_profile["tau_declaration_time"] = perf_counter() - step_start
    build_profile["tau_declaration_stats"] = {"variables": len(model.psd_variables), "sectors": tau_sector_total}

    next_row = 0

    step_start = perf_counter()
    _progress_log(real_verbose, 1, "Paper Task step 2/5: emitting trace constraints...")
    next_row, trace_count = _append_trace_rows(task, next_row, list(tau_bar_variables.values()))
    constraint_counts["trace"] = trace_count
    build_profile["trace_time"] = perf_counter() - step_start

    step_start = perf_counter()
    observed_stats = {"total": len(model.observed_constraints), "resolved": 0}
    for constraint in model.observed_constraints:
        entry = _resolve_view_maps(constraint.marginal_view)
        representative_anchors[constraint.name] = TaskRepresentativeAnchorSpec(
            representative_name=constraint.name,
            representative_kind="observed",
            representative_lexorder=constraint.target_lexorder,
            source_variable_name=constraint.marginal_view.source_variable_name,
            source_variable_kind=constraint.marginal_view.source_variable_kind,
            keep_positions=constraint.marginal_view.keep_positions,
            traced_positions=constraint.marginal_view.traced_positions,
            coordinate_dim=int(entry["target_coord_dim"]),
            sector_maps=entry["sector_maps"],
        )
        observed_stats["resolved"] += 1
    build_profile["observed_setup_time"] = perf_counter() - step_start
    build_profile["observed_stats"] = observed_stats

    step_start = perf_counter()
    equality_stats = {
        "total": len(model.equality_constraints),
        "emitted": 0,
        "rows_emitted": 0,
    }
    equality_log_time = step_start
    _progress_log(
        real_verbose,
        1,
        f"Paper Task step 3/5: emitting {len(model.equality_constraints)} matching equalities...",
    )
    for constraint in model.equality_constraints:
        lhs_entry = _resolve_view_maps(constraint.lhs_view)
        rhs_entry = _resolve_view_maps(constraint.rhs_view)
        coord_dim = int(lhs_entry["target_coord_dim"])
        if coord_dim != int(rhs_entry["target_coord_dim"]):
            raise ValueError(f"Paper equality coordinate mismatch for {constraint.name}.")
        row_indices, next_row = _append_fx_rows(task, next_row, np.zeros(coord_dim, dtype=np.float64))
        row_offset = int(row_indices[0])
        _append_bar_terms(
            task,
            tuple(
                _instantiate_bar_term_template(row_offset, template)
                for template in lhs_entry["positive_templates"]
            ) + tuple(
                _instantiate_bar_term_template(row_offset, template)
                for template in rhs_entry["negative_templates"]
            ),
        )
        equality_stats["emitted"] += 1
        equality_stats["rows_emitted"] += coord_dim
        constraint_counts["representative"] += coord_dim
        equality_log_time = _maybe_log_loop_progress(
            real_verbose,
            "Paper Task step 3/5 equalities",
            equality_stats["emitted"],
            max(equality_stats["total"], 1),
            step_start,
            equality_log_time,
            extra=f"rows={equality_stats['rows_emitted']}, map-cache={tau_map_cache_stats['hits']}/{tau_map_cache_stats['misses']}",
        )
    build_profile["equality_time"] = perf_counter() - step_start
    build_profile["equality_stats"] = equality_stats
    build_profile["tau_map_cache"] = dict(tau_map_cache_stats)

    step_start = perf_counter()
    build_profile["known_value_time"] = 0.0
    build_profile["known_value_stats"] = {"representatives_total": 0, "representatives_emitted": 0, "rows_emitted": 0}

    ppt_stats = {
        "raw_total": len(model.ppt_constraints),
        "raw_full": sum(
            1 for constraint in model.ppt_constraints
            if len(constraint.marginal_view.keep_positions) == len(constraint.marginal_view.source_lexorder)
        ),
        "raw_marginal": sum(
            1 for constraint in model.ppt_constraints
            if len(constraint.marginal_view.keep_positions) != len(constraint.marginal_view.source_lexorder)
        ),
        "active_total": 0,
        "active_full": 0,
        "active_marginal": 0,
        "emitted_total": 0,
        "emitted_full": 0,
        "emitted_marginal": 0,
        "rows_emitted": 0,
    }
    active_ppt_constraints = tuple()
    if include_ppt:
        active_ppt_constraints = tuple(model.ppt_constraints)
        if not include_family_ppt:
            active_ppt_constraints = tuple(
                constraint
                for constraint in active_ppt_constraints
                if len(constraint.marginal_view.keep_positions) == len(constraint.marginal_view.source_lexorder)
            )
        ppt_stats["active_total"] = len(active_ppt_constraints)
        ppt_stats["active_full"] = sum(
            1 for constraint in active_ppt_constraints
            if len(constraint.marginal_view.keep_positions) == len(constraint.marginal_view.source_lexorder)
        )
        ppt_stats["active_marginal"] = ppt_stats["active_total"] - ppt_stats["active_full"]
        _progress_log(
            real_verbose,
            1,
            f"Paper Task step 5/5: preparing {ppt_stats['active_total']} PPT constraints...",
        )

        ppt_prepare_start = perf_counter()
        direct_tau_ppt_sector_maps: Dict[str, Tuple[Tuple[np.ndarray, ...], ...]] = {}
        direct_tau_ppt_row_lookups: Dict[str, np.ndarray] = {}
        ppt_direct_barvars: Dict[str, Tuple[TaskBarVariableSpec, ...]] = {}
        marginal_ppt_specs: Dict[str, TaskBarVariableSpec] = {}
        for constraint in active_ppt_constraints:
            view = constraint.marginal_view
            source_spec = tau_variable_specs[view.source_variable_name]
            full_source = tuple(view.keep_positions) == tuple(range(len(source_spec.slot_dims)))
            if full_source and not Hermitian:
                stabilizer_perms = _cached_tau_ppt_stabilizer(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                _stabilizer, orbit_data, row_lookup = _cached_tau_ppt_orbit_reduction(
                    source_spec.slot_dims,
                    source_spec.local_symmetry_perms,
                    constraint.transpose_positions,
                )
                representative_rows = tuple(
                    _upper_triangle_coordinate_index(row, col, orbit_data.matrix_dim)
                    for row, col in orbit_data.orbit_representatives
                )
                selected_maps = _cached_exact_small_group_selected_coordinate_maps(
                    _party_signature_from_lexorder(source_spec.lexorder),
                    source_spec.slot_dims,
                    stabilizer_perms,
                    representative_rows,
                )
                stabilizer_layout = cached_tau_layout(
                    constraint.name,
                    source_spec.lexorder,
                    source_spec.slot_dims,
                    stabilizer_perms,
                )
                sector_specs = []
                for sector in stabilizer_layout.sectors:
                    task.appendbarvars([int(sector.multiplicity)])
                    coordinate_dim = int(sector.multiplicity * (sector.multiplicity + 1) // 2)
                    sector_specs.append(
                        TaskBarVariableSpec(
                            name=f"{constraint.name}_{sector.label}",
                            kind="ppt_tau_sector",
                            parent_name=constraint.name,
                            barvar_index=next_barvar,
                            dim=int(sector.multiplicity),
                            coordinate_dim=coordinate_dim,
                            sector_label=sector.label,
                            irrep_dim=int(sector.irrep_dim),
                            complex_dim=int(sector.multiplicity),
                            hermitian=False,
                        )
                    )
                    next_barvar += 1
                    constraint_counts["block_parameters"] += coordinate_dim
                ppt_direct_barvars[constraint.name] = tuple(sector_specs)
                ppt_bar_variables[constraint.name] = tuple(sector_specs)
                direct_tau_ppt_row_lookups[constraint.name] = row_lookup
                if selected_maps is not None:
                    _target_coord_dim, lhs_sector_maps = selected_maps
                else:
                    _target_coord_dim, lhs_sector_maps = _cached_block_partial_trace_coordinate_maps(
                        _party_signature_from_lexorder(source_spec.lexorder),
                        source_spec.slot_dims,
                        stabilizer_perms,
                        tuple(range(len(source_spec.slot_dims))),
                        False,
                    )
                    lhs_sector_maps = _restrict_sector_triplets_to_row_lookup(lhs_sector_maps, row_lookup)
                direct_tau_ppt_sector_maps[constraint.name] = lhs_sector_maps
            else:
                target_spec, next_barvar, coordinate_dim = _append_full_psd_barvar(
                    task,
                    name=f"{constraint.name}_aux",
                    kind="ppt_paper",
                    parent_name=constraint.name,
                    matrix_dim=view.matrix_dim,
                    next_barvar=next_barvar,
                    Hermitian=Hermitian,
                )
                marginal_ppt_specs[constraint.name] = target_spec
                ppt_bar_variables[constraint.name] = target_spec
                constraint_counts["block_parameters"] += coordinate_dim
        build_profile["ppt_prepare_time"] = perf_counter() - ppt_prepare_start

        ppt_emit_start = perf_counter()
        ppt_log_time = ppt_emit_start
        for constraint in active_ppt_constraints:
            view = constraint.marginal_view
            source_spec = tau_variable_specs[view.source_variable_name]
            full_source = tuple(view.keep_positions) == tuple(range(len(source_spec.slot_dims)))
            if full_source and not Hermitian:
                source_barvars = tau_bar_variables[view.source_variable_name]
                target_barvars = ppt_direct_barvars[constraint.name]
                row_lookup = direct_tau_ppt_row_lookups[constraint.name]
                target_coord_dim, rhs_sector_maps = get_cached_tau_sector_linear_map_triplets(
                    source_spec,
                    constraint.transpose_positions,
                    "partial_transpose",
                    tau_sector_coordinate_dims[view.source_variable_name],
                    Hermitian=False,
                )
                rhs_sector_maps = _restrict_sector_triplets_to_row_lookup(rhs_sector_maps, row_lookup)
                lhs_sector_maps = direct_tau_ppt_sector_maps[constraint.name]
                reduced_coord_dim = int(max(row_lookup[row_lookup >= 0]) + 1) if np.any(row_lookup >= 0) else 0
                if int(target_coord_dim) < reduced_coord_dim:
                    raise ValueError("Reduced paper tau PPT row count exceeds source coordinate dimension.")
                row_indices, next_row = _append_fx_rows(task, next_row, np.zeros(reduced_coord_dim, dtype=np.float64))
                term_blocks = []
                for sector_map, sector_barvar in zip(lhs_sector_maps, target_barvars):
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            _basis_indices_for(sector_barvar),
                            sector_map[0],
                            sector_map[1],
                            sector_map[2],
                        )
                    )
                for sector_map, sector_barvar in zip(rhs_sector_maps, source_barvars):
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            _basis_indices_for(sector_barvar),
                            sector_map[0],
                            sector_map[1],
                            -sector_map[2],
                        )
                    )
                _append_bar_terms(task, term_blocks)
                rows_emitted = reduced_coord_dim
                ppt_stats["emitted_full"] += 1
            else:
                target_spec = marginal_ppt_specs[constraint.name]
                source_entry = _resolve_view_maps(view)
                target_coord_dim, rows, cols, vals, *_ = cached_partial_transpose_coordinate_action(
                    tuple(view.slot_dims),
                    tuple(int(pos) for pos in constraint.transpose_positions),
                    Hermitian=Hermitian,
                )
                source_sector_maps = _compose_coordinate_action_with_sector_maps(
                    rows,
                    cols,
                    vals,
                    source_entry["sector_maps"],
                )
                id_rows, id_cols, id_vals = _row_identity_triplets(target_spec.coordinate_dim, scale=1.0)
                row_indices, next_row = _append_fx_rows(task, next_row, np.zeros(int(target_coord_dim), dtype=np.float64))
                term_blocks = [
                    _coordinate_triplets_to_bar_terms(
                        int(row_indices[0]),
                        target_spec.barvar_index,
                        _basis_indices_for(target_spec),
                        id_rows,
                        id_cols,
                        id_vals,
                    )
                ]
                for sector_map, sector_barvar in zip(source_sector_maps, tau_bar_variables[view.source_variable_name]):
                    term_blocks.append(
                        _coordinate_triplets_to_bar_terms(
                            int(row_indices[0]),
                            sector_barvar.barvar_index,
                            _basis_indices_for(sector_barvar),
                            sector_map[0],
                            sector_map[1],
                            -sector_map[2],
                        )
                    )
                _append_bar_terms(task, term_blocks)
                rows_emitted = int(target_coord_dim)
                ppt_stats["emitted_marginal"] += 1

            constraint_counts["ppt_direct"] += 1
            ppt_stats["emitted_total"] += 1
            ppt_stats["rows_emitted"] += int(rows_emitted)
            ppt_log_time = _maybe_log_loop_progress(
                real_verbose,
                "Paper Task step 5/5 emit",
                ppt_stats["emitted_total"],
                max(ppt_stats["active_total"], 1),
                ppt_emit_start,
                ppt_log_time,
                extra=f"rows={ppt_stats['rows_emitted']}",
            )
        build_profile["ppt_emit_time"] = perf_counter() - ppt_emit_start
    build_profile["ppt_time"] = perf_counter() - step_start
    build_profile["ppt_stats"] = ppt_stats

    build_profile["total_build_time"] = perf_counter() - total_start
    _progress_log(
        real_verbose,
        1,
        "Paper Task build complete: "
        f"{constraint_counts['trace']} trace rows, "
        f"{constraint_counts['representative']} equality rows, "
        f"{constraint_counts['ppt_direct']} PPT blocks in "
        f"{build_profile['total_build_time']:.2f}s.",
    )

    return TaskBlockStateSDPModel(
        env=env,
        task=task,
        Hermitian=bool(Hermitian),
        tau_bar_variables=tau_bar_variables,
        representative_anchors=representative_anchors,
        auxiliary_bar_variables=auxiliary_bar_variables,
        known_representative_bar_variables=known_representative_bar_variables,
        ppt_bar_variables=ppt_bar_variables,
        constraint_counts=constraint_counts,
        build_profile=build_profile,
    )


__all__ = [
    "TaskBarVariableSpec",
    "TaskBlockStateSDPModel",
    "build_block_task_feasibility_model",
    "build_top_down_block_task_feasibility_model",
]
