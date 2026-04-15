"""Shared coordinate-action and SCS assembly backend for the level-3 models.

This file intentionally contains the low-level plumbing:

- coordinate pack/unpack maps
- slot-permutation, partial-trace, and partial-transpose actions
- direct SCS affine assembly helpers
- complex and real-restricted PSD cone encodings

The readable formulation lives in:

- ``level3_complex_model.py``
- ``level3_real_restricted_model.py``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import prod

import numpy as np
import scipy.sparse as sp

DEFAULT_SCS_ALPHA = 1.8
DEFAULT_SCS_SCALE = 0.5
DEFAULT_SCS_RHO_X = 1e-4
DEFAULT_SCS_ACCELERATION_LOOKBACK = 20
DEFAULT_SCS_ACCELERATION_INTERVAL = 5


@dataclass(frozen=True)
class SCSVectorSlice:
    name: str
    offset: int
    dim: int

    @property
    def stop(self) -> int:
        return int(self.offset + self.dim)


@dataclass(frozen=True)
class MatrixVariableSpec:
    name: str
    matrix_dim: int
    coord_dim: int
    x_slice: SCSVectorSlice


@dataclass
class SCSAffineConstraintBuilder:
    row_blocks: list[np.ndarray] = field(default_factory=list)
    col_blocks: list[np.ndarray] = field(default_factory=list)
    val_blocks: list[np.ndarray] = field(default_factory=list)
    b_parts: list[np.ndarray] = field(default_factory=list)
    next_var_offset: int = 0
    next_row: int = 0
    constraint_counts: dict[str, int] = field(
        default_factory=lambda: {
            "trace": 0,
            "symmetry": 0,
            "observed": 0,
            "equality": 0,
            "ppt": 0,
            "nonnegative": 0,
            "psd": 0,
            "rows_total": 0,
            "scalar_vars": 0,
            "vector_vars": 0,
        }
    )

    def add_scalar_variable(self, name: str) -> SCSVectorSlice:
        vector_slice = SCSVectorSlice(name, int(self.next_var_offset), 1)
        self.next_var_offset += 1
        self.constraint_counts["scalar_vars"] += 1
        self.constraint_counts["vector_vars"] = int(self.next_var_offset)
        return vector_slice

    def add_matrix_variable(self, name: str, matrix_dim: int, coord_dim: int) -> MatrixVariableSpec:
        vector_slice = SCSVectorSlice(name, int(self.next_var_offset), int(coord_dim))
        self.next_var_offset += int(coord_dim)
        self.constraint_counts["vector_vars"] = int(self.next_var_offset)
        return MatrixVariableSpec(name=name, matrix_dim=int(matrix_dim), coord_dim=int(coord_dim), x_slice=vector_slice)

    def _reserve_rows(self, rhs: np.ndarray) -> int:
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
        row_offset = int(self.next_row)
        self.next_row += int(rhs.size)
        self.b_parts.append(rhs)
        return row_offset

    def append_action(self, row_offset: int, vector_slice: SCSVectorSlice, action, *, scale: float = 1.0) -> None:
        _append_action_triplets(
            int(row_offset),
            int(vector_slice.offset),
            action,
            self.row_blocks,
            self.col_blocks,
            self.val_blocks,
            scale=float(scale),
        )

    def add_trace_one(self, variable: MatrixVariableSpec, trace_action) -> None:
        row_offset = self._reserve_rows(np.asarray([1.0], dtype=np.float64))
        self.append_action(row_offset, variable.x_slice, trace_action)
        self.constraint_counts["trace"] += 1

    def add_difference_equality(
        self,
        lhs_variable: MatrixVariableSpec,
        lhs_action,
        rhs_variable: MatrixVariableSpec,
        rhs_action,
        *,
        counter_key: str,
    ) -> None:
        target_dim = int(lhs_action[0])
        if target_dim != int(rhs_action[0]):
            raise ValueError("Action target dimensions do not match.")
        row_offset = self._reserve_rows(np.zeros(target_dim, dtype=np.float64))
        self.append_action(row_offset, lhs_variable.x_slice, lhs_action, scale=1.0)
        self.append_action(row_offset, rhs_variable.x_slice, rhs_action, scale=-1.0)
        self.constraint_counts[counter_key] += int(target_dim)

    def add_observed_anchor(
        self,
        variable: MatrixVariableSpec,
        anchor_action,
        rhs_vector: np.ndarray,
        scalar_slice: SCSVectorSlice,
        scalar_vector: np.ndarray,
    ) -> None:
        rhs_vector = np.asarray(rhs_vector, dtype=np.float64).reshape(-1)
        scalar_vector = np.asarray(scalar_vector, dtype=np.float64).reshape(-1)
        if rhs_vector.size != int(anchor_action[0]) or scalar_vector.size != rhs_vector.size:
            raise ValueError("Observed anchor dimensions do not match.")
        row_offset = self._reserve_rows(rhs_vector)
        self.append_action(row_offset, variable.x_slice, anchor_action, scale=1.0)
        self.row_blocks.append((row_offset + np.arange(rhs_vector.size, dtype=np.int32)).astype(np.int32, copy=False))
        self.col_blocks.append(np.full(rhs_vector.size, int(scalar_slice.offset), dtype=np.int32))
        self.val_blocks.append(scalar_vector.astype(np.float64, copy=False))
        self.constraint_counts["observed"] += int(rhs_vector.size)

    def add_auxiliary_link(
        self,
        auxiliary_variable: MatrixVariableSpec,
        source_variable: MatrixVariableSpec,
        source_action,
    ) -> None:
        row_offset = self._reserve_rows(np.zeros(auxiliary_variable.coord_dim, dtype=np.float64))
        self.append_action(row_offset, auxiliary_variable.x_slice, identity_coordinate_action(auxiliary_variable.coord_dim), scale=1.0)
        self.append_action(row_offset, source_variable.x_slice, source_action, scale=-1.0)
        self.constraint_counts["ppt"] += int(auxiliary_variable.coord_dim)

    def add_scalar_nonnegative(self, scalar_slice: SCSVectorSlice) -> None:
        row_offset = self._reserve_rows(np.zeros(1, dtype=np.float64))
        self.row_blocks.append(np.asarray([row_offset], dtype=np.int32))
        self.col_blocks.append(np.asarray([int(scalar_slice.offset)], dtype=np.int32))
        self.val_blocks.append(np.asarray([-1.0], dtype=np.float64))
        self.constraint_counts["nonnegative"] = 1

    def add_psd_constraint(self, variable: MatrixVariableSpec, cone_action) -> None:
        row_offset = self._reserve_rows(np.zeros(int(cone_action[0]), dtype=np.float64))
        self.append_action(row_offset, variable.x_slice, cone_action, scale=-1.0)
        self.constraint_counts["psd"] += int(cone_action[0])

    def finalize(
        self,
        *,
        objective_slice: SCSVectorSlice,
        cone: dict[str, object],
    ) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray, dict[str, int]]:
        A = sp.coo_matrix(
            (
                np.concatenate(self.val_blocks).astype(np.float64, copy=False) if self.val_blocks else np.asarray([], dtype=np.float64),
                (
                    np.concatenate(self.row_blocks).astype(np.int32, copy=False) if self.row_blocks else np.asarray([], dtype=np.int32),
                    np.concatenate(self.col_blocks).astype(np.int32, copy=False) if self.col_blocks else np.asarray([], dtype=np.int32),
                ),
            ),
            shape=(int(self.next_row), int(self.next_var_offset)),
        ).tocsc()
        b = np.concatenate(self.b_parts).astype(np.float64, copy=False) if self.b_parts else np.asarray([], dtype=np.float64)
        c = np.zeros(int(self.next_var_offset), dtype=np.float64)
        c[int(objective_slice.offset)] = -1.0
        self.constraint_counts["rows_total"] = int(self.next_row)
        self.constraint_counts["vector_vars"] = int(self.next_var_offset)
        return A, b, c, dict(self.constraint_counts)


def product_dim(dims) -> int:
    return int(prod(int(dim) for dim in dims))


def _flat_index(multi_index: tuple[int, ...], dims: tuple[int, ...]) -> int:
    value = 0
    for entry, dim in zip(multi_index, dims):
        value = value * int(dim) + int(entry)
    return value


def _unflatten_index(index: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    entries = [0] * len(dims)
    value = int(index)
    for pos in range(len(dims) - 1, -1, -1):
        entries[pos] = value % int(dims[pos])
        value //= int(dims[pos])
    return tuple(entries)


def _upper_triangle_coordinate_index(row: int, col: int, matrix_dim: int) -> int:
    if not (0 <= row <= col < matrix_dim):
        raise ValueError("Invalid upper-triangle coordinate.")
    return row * matrix_dim - row * (row - 1) // 2 + (col - row)


def _strict_upper_coordinate_index(row: int, col: int, matrix_dim: int) -> int:
    if not (0 <= row < col < matrix_dim):
        raise ValueError("Invalid strict upper-triangle coordinate.")
    return row * (matrix_dim - 1) - row * (row - 1) // 2 + (col - row - 1)


def _basis_permutation_map(
    slot_dims: tuple[int, ...],
    slot_permutation: tuple[int, ...],
) -> tuple[int, ...]:
    dim = product_dim(slot_dims)
    return tuple(
        _flat_index(
            tuple(_unflatten_index(index, slot_dims)[pos] for pos in slot_permutation),
            slot_dims,
        )
        for index in range(dim)
    )


def _partial_transpose_entry_map(
    row: int,
    col: int,
    slot_dims: tuple[int, ...],
    transpose_positions: tuple[int, ...],
) -> tuple[int, int]:
    bra = list(_unflatten_index(row, slot_dims))
    ket = list(_unflatten_index(col, slot_dims))
    for pos in transpose_positions:
        bra[pos], ket[pos] = ket[pos], bra[pos]
    return _flat_index(tuple(bra), slot_dims), _flat_index(tuple(ket), slot_dims)


def pack_hermitian_coordinates(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-9):
        raise ValueError("Expected a Hermitian matrix.")
    matrix_dim = int(matrix.shape[0])
    coords = np.empty(matrix_dim * matrix_dim, dtype=np.float64)
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            coords[cursor] = float(np.real(matrix[row, col]))
            cursor += 1
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            coords[cursor] = float(np.imag(matrix[row, col]))
            cursor += 1
    return coords


def unpack_hermitian_coordinates(coords: np.ndarray, matrix_dim: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    if coords.size != matrix_dim * matrix_dim:
        raise ValueError("Coordinate vector has incompatible length.")
    matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.complex128)
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            value = coords[cursor]
            cursor += 1
            matrix[row, col] = value
            matrix[col, row] = value
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            value = coords[cursor]
            cursor += 1
            matrix[row, col] += 1j * value
            matrix[col, row] -= 1j * value
    return matrix


@lru_cache(maxsize=None)
def cached_partial_trace_selector_triplets(
    slot_dims: tuple[int, ...],
    keep_positions: tuple[int, ...],
):
    complex_dim = product_dim(slot_dims)
    kept_dims = tuple(slot_dims[pos] for pos in keep_positions)
    traced_positions = tuple(pos for pos in range(len(slot_dims)) if pos not in keep_positions)
    kept_ranges = [range(dim) for dim in kept_dims]
    traced_ranges = [range(slot_dims[pos]) for pos in traced_positions]
    selectors = []
    for bra_keep in product(*kept_ranges):
        row = _flat_index(tuple(int(x) for x in bra_keep), kept_dims)
        for ket_keep in product(*kept_ranges):
            col = _flat_index(tuple(int(x) for x in ket_keep), kept_dims)
            if row > col:
                continue
            rows_re = []
            cols_re = []
            rows_im = []
            cols_im = []
            traced_iterator = product(*traced_ranges) if traced_ranges else [tuple()]
            for traced_values in traced_iterator:
                bra_full = [0] * len(slot_dims)
                ket_full = [0] * len(slot_dims)
                for pos, value in zip(keep_positions, bra_keep):
                    bra_full[pos] = int(value)
                for pos, value in zip(keep_positions, ket_keep):
                    ket_full[pos] = int(value)
                for pos, value in zip(traced_positions, traced_values):
                    bra_full[pos] = int(value)
                    ket_full[pos] = int(value)
                full_row = _flat_index(tuple(bra_full), slot_dims)
                full_col = _flat_index(tuple(ket_full), slot_dims)
                rows_re.append(full_row)
                cols_re.append(full_col)
                if row < col:
                    rows_im.append(complex_dim + full_row)
                    cols_im.append(full_col)
            selectors.append(
                (
                    row,
                    col,
                    np.asarray(rows_re, dtype=np.int32),
                    np.asarray(cols_re, dtype=np.int32),
                    np.asarray(rows_im, dtype=np.int32),
                    np.asarray(cols_im, dtype=np.int32),
                )
            )
    return tuple(selectors)


@lru_cache(maxsize=None)
def cached_partial_trace_coordinate_action(
    slot_dims: tuple[int, ...],
    keep_positions: tuple[int, ...],
):
    source_dim = product_dim(slot_dims)
    source_real_count = source_dim * (source_dim + 1) // 2
    kept_dims = tuple(slot_dims[pos] for pos in keep_positions)
    target_dim = product_dim(kept_dims) if kept_dims else 1
    target_real_count = target_dim * (target_dim + 1) // 2
    target_coord_dim = target_dim * target_dim

    row_blocks = []
    col_blocks = []
    val_blocks = []
    for row, col, rows_re, cols_re, _rows_im, _cols_im in cached_partial_trace_selector_triplets(slot_dims, keep_positions):
        target_row_re = _upper_triangle_coordinate_index(int(row), int(col), target_dim)
        source_row_min = np.minimum(rows_re, cols_re).astype(np.int64, copy=False)
        source_row_max = np.maximum(rows_re, cols_re).astype(np.int64, copy=False)
        source_real_coords = (
            source_row_min * source_dim
            - source_row_min * (source_row_min - 1) // 2
            + (source_row_max - source_row_min)
        ).astype(np.int32, copy=False)
        row_blocks.append(np.full(source_real_coords.size, target_row_re, dtype=np.int32))
        col_blocks.append(source_real_coords)
        val_blocks.append(np.ones(source_real_coords.size, dtype=np.float64))
        if row < col:
            target_row_im = target_real_count + _strict_upper_coordinate_index(int(row), int(col), target_dim)
            source_im_mask = rows_re != cols_re
            if np.any(source_im_mask):
                source_im_min = source_row_min[source_im_mask]
                source_im_max = source_row_max[source_im_mask]
                source_im_coords = (
                    source_real_count
                    + source_im_min * (source_dim - 1)
                    - source_im_min * (source_im_min - 1) // 2
                    + (source_im_max - source_im_min - 1)
                ).astype(np.int32, copy=False)
                source_im_signs = np.where(rows_re[source_im_mask] < cols_re[source_im_mask], 1.0, -1.0).astype(np.float64, copy=False)
                row_blocks.append(np.full(source_im_coords.size, target_row_im, dtype=np.int32))
                col_blocks.append(source_im_coords)
                val_blocks.append(source_im_signs)
    if not row_blocks:
        return (
            target_coord_dim,
            source_dim * source_dim,
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )
    return (
        target_coord_dim,
        source_dim * source_dim,
        np.concatenate(row_blocks).astype(np.int32, copy=False),
        np.concatenate(col_blocks).astype(np.int32, copy=False),
        np.concatenate(val_blocks).astype(np.float64, copy=False),
    )


@lru_cache(maxsize=None)
def cached_partial_transpose_coordinate_action(
    slot_dims: tuple[int, ...],
    transpose_positions: tuple[int, ...],
):
    matrix_dim = product_dim(slot_dims)
    real_count = matrix_dim * (matrix_dim + 1) // 2
    coord_dim = matrix_dim * matrix_dim
    source_to_target_row = np.empty(coord_dim, dtype=np.int32)
    source_to_target_sign = np.empty(coord_dim, dtype=np.float64)
    target_row = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            mapped_row, mapped_col = _partial_transpose_entry_map(row, col, slot_dims, transpose_positions)
            source_row = _upper_triangle_coordinate_index(min(mapped_row, mapped_col), max(mapped_row, mapped_col), matrix_dim)
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = 1.0
            target_row += 1
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            mapped_row, mapped_col = _partial_transpose_entry_map(row, col, slot_dims, transpose_positions)
            if mapped_row < mapped_col:
                source_row = real_count + _strict_upper_coordinate_index(mapped_row, mapped_col, matrix_dim)
                sign = 1.0
            else:
                source_row = real_count + _strict_upper_coordinate_index(mapped_col, mapped_row, matrix_dim)
                sign = -1.0
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = sign
            target_row += 1
    rows = np.arange(coord_dim, dtype=np.int32)
    cols = np.empty(coord_dim, dtype=np.int32)
    vals = np.empty(coord_dim, dtype=np.float64)
    for source_row in range(coord_dim):
        target = int(source_to_target_row[source_row])
        rows[target] = target
        cols[target] = source_row
        vals[target] = float(source_to_target_sign[source_row])
    return coord_dim, coord_dim, rows, cols, vals


@lru_cache(maxsize=None)
def cached_coordinate_action_for_slot_permutation(
    slot_dims: tuple[int, ...],
    slot_permutation: tuple[int, ...],
):
    matrix_dim = product_dim(slot_dims)
    real_count = matrix_dim * (matrix_dim + 1) // 2
    coord_dim = matrix_dim * matrix_dim
    basis_map = _basis_permutation_map(slot_dims, slot_permutation)
    source_to_target_row = np.empty(coord_dim, dtype=np.int32)
    source_to_target_sign = np.empty(coord_dim, dtype=np.float64)
    target_row = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            mapped_row = basis_map[row]
            mapped_col = basis_map[col]
            source_row = _upper_triangle_coordinate_index(min(mapped_row, mapped_col), max(mapped_row, mapped_col), matrix_dim)
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = 1.0
            target_row += 1
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            mapped_row = basis_map[row]
            mapped_col = basis_map[col]
            if mapped_row < mapped_col:
                source_row = real_count + _strict_upper_coordinate_index(mapped_row, mapped_col, matrix_dim)
                sign = 1.0
            else:
                source_row = real_count + _strict_upper_coordinate_index(mapped_col, mapped_row, matrix_dim)
                sign = -1.0
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = sign
            target_row += 1
    rows = np.arange(coord_dim, dtype=np.int32)
    cols = np.empty(coord_dim, dtype=np.int32)
    vals = np.empty(coord_dim, dtype=np.float64)
    for source_row in range(coord_dim):
        target = int(source_to_target_row[source_row])
        rows[target] = target
        cols[target] = source_row
        vals[target] = float(source_to_target_sign[source_row])
    return coord_dim, coord_dim, rows, cols, vals


@lru_cache(maxsize=None)
def _trace_coordinate_action(matrix_dim: int):
    diag_cols = np.asarray([_upper_triangle_coordinate_index(i, i, matrix_dim) for i in range(matrix_dim)], dtype=np.int32)
    diag_rows = np.zeros(matrix_dim, dtype=np.int32)
    diag_vals = np.ones(matrix_dim, dtype=np.float64)
    return 1, matrix_dim * matrix_dim, diag_rows, diag_cols, diag_vals


@lru_cache(maxsize=None)
def cached_scs_complex_psd_coordinate_action(complex_dim: int):
    real_count = complex_dim * (complex_dim + 1) // 2
    rows = []
    cols = []
    vals = []
    target = 0
    sqrt2 = float(np.sqrt(2.0))
    for col in range(complex_dim):
        for row in range(col, complex_dim):
            if row == col:
                rows.append(target)
                cols.append(_upper_triangle_coordinate_index(col, col, complex_dim))
                vals.append(1.0)
                target += 1
                continue
            rows.append(target)
            cols.append(_upper_triangle_coordinate_index(col, row, complex_dim))
            vals.append(sqrt2)
            target += 1
            rows.append(target)
            cols.append(real_count + _strict_upper_coordinate_index(col, row, complex_dim))
            vals.append(-sqrt2)
            target += 1
    return (
        int(complex_dim * complex_dim),
        int(complex_dim * complex_dim),
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )


def _append_action_triplets(
    row_offset: int,
    col_offset: int,
    action: tuple[int, int, np.ndarray, np.ndarray, np.ndarray],
    row_blocks: list[np.ndarray],
    col_blocks: list[np.ndarray],
    val_blocks: list[np.ndarray],
    *,
    scale: float = 1.0,
) -> None:
    rows = np.asarray(action[2], dtype=np.int32)
    cols = np.asarray(action[3], dtype=np.int32)
    vals = np.asarray(action[4], dtype=np.float64)
    if vals.size == 0:
        return
    row_blocks.append((int(row_offset) + rows).astype(np.int32, copy=False))
    col_blocks.append((int(col_offset) + cols).astype(np.int32, copy=False))
    if float(scale) == 1.0:
        val_blocks.append(vals.astype(np.float64, copy=False))
    else:
        val_blocks.append((float(scale) * vals).astype(np.float64, copy=False))


def identity_coordinate_action(coord_dim: int):
    rows = np.arange(int(coord_dim), dtype=np.int32)
    cols = rows.copy()
    vals = np.ones(int(coord_dim), dtype=np.float64)
    return int(coord_dim), int(coord_dim), rows, cols, vals


def compose_coordinate_actions(outer_action, inner_action):
    outer_target, outer_source, outer_rows, outer_cols, outer_vals = outer_action
    inner_target, inner_source, inner_rows, inner_cols, inner_vals = inner_action
    if int(outer_source) != int(inner_target):
        raise ValueError("Incompatible coordinate actions.")
    lookup: dict[int, list[tuple[int, float]]] = {}
    for row, col, val in zip(inner_rows, inner_cols, inner_vals):
        lookup.setdefault(int(row), []).append((int(col), float(val)))
    rows_out: list[int] = []
    cols_out: list[int] = []
    vals_out: list[float] = []
    for row, col, val in zip(outer_rows, outer_cols, outer_vals):
        for inner_col, inner_val in lookup.get(int(col), ()):
            rows_out.append(int(row))
            cols_out.append(int(inner_col))
            vals_out.append(float(val) * float(inner_val))
    if not vals_out:
        return (
            int(outer_target),
            int(inner_source),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )
    rows = np.asarray(rows_out, dtype=np.int32)
    cols = np.asarray(cols_out, dtype=np.int32)
    vals = np.asarray(vals_out, dtype=np.float64)
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
        int(outer_target),
        int(inner_source),
        rows[unique_idx][keep].astype(np.int32, copy=False),
        cols[unique_idx][keep].astype(np.int32, copy=False),
        summed_vals[keep].astype(np.float64, copy=False),
    )


def solve_with_scs_compat(
    data: dict[str, object],
    cone: dict[str, object],
    *,
    verbose: bool,
    max_iters: int,
    eps_abs: float,
    eps_rel: float,
    alpha: float,
    scale: float,
    normalize: bool,
    adaptive_scale: bool,
    rho_x: float,
    acceleration_lookback: int,
    acceleration_interval: int,
    time_limit_secs: float,
    use_indirect: bool,
):
    import scs

    settings = {
        "verbose": bool(verbose),
        "max_iters": int(max_iters),
        "eps_abs": float(eps_abs),
        "eps_rel": float(eps_rel),
        "alpha": float(alpha),
        "scale": float(scale),
        "normalize": bool(normalize),
        "adaptive_scale": bool(adaptive_scale),
        "rho_x": float(rho_x),
        "acceleration_lookback": int(acceleration_lookback),
        "acceleration_interval": int(acceleration_interval),
        "time_limit_secs": float(time_limit_secs),
        "use_indirect": bool(use_indirect),
    }
    if hasattr(scs, "SCS"):
        solver = scs.SCS(data, cone, **settings)
        return solver.solve()
    if cone.get("cs"):
        raise RuntimeError(
            "SCS < 3.0 does not support the 'cs' complex-PSD cone used by this model. "
            f"Found scs version {getattr(scs, '__version__', 'unknown')}. "
            "Use scs >= 3.x."
        )
    legacy_settings = {
        "verbose": bool(verbose),
        "max_iters": int(max_iters),
        "eps": float(min(eps_abs, eps_rel)),
        "alpha": float(alpha),
        "scale": float(scale),
        "normalize": bool(normalize),
        "rho_x": float(rho_x),
        "acceleration_lookback": int(acceleration_lookback),
        "use_indirect": bool(use_indirect),
    }
    return scs.solve(data, cone, **legacy_settings)


def _validate_level3_inputs(
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
    dims9 = dims3 + dims3 + dims3
    full_dim = product_dim(dims9)
    return dims3, dims9, D, full_dim, rho


def ghz_density_matrix() -> np.ndarray:
    ket = np.zeros(8, dtype=np.complex128)
    ket[0] = 1.0 / np.sqrt(2.0)
    ket[7] = 1.0 / np.sqrt(2.0)
    return np.outer(ket, ket.conj())


def _lower_triangle_columnwise_index(row: int, col: int, matrix_dim: int) -> int:
    if not (0 <= col <= row < matrix_dim):
        raise ValueError("Invalid lower-triangle coordinate.")
    return int(col * matrix_dim - col * (col - 1) // 2 + (row - col))


def _real_symmetric_coord_dim(matrix_dim: int) -> int:
    return int(matrix_dim * (matrix_dim + 1) // 2)


def pack_real_symmetric_coordinates(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-9):
        raise ValueError("Expected a Hermitian matrix.")
    if not np.allclose(np.imag(matrix), 0.0, atol=1e-9):
        raise ValueError("Real-restricted formulation requires a real symmetric matrix.")
    matrix_real = np.real(matrix)
    matrix_dim = int(matrix.shape[0])
    coords = np.empty(_real_symmetric_coord_dim(matrix_dim), dtype=np.float64)
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            coords[cursor] = float(matrix_real[row, col])
            cursor += 1
    return coords


def unpack_real_symmetric_coordinates(coords: np.ndarray, matrix_dim: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    expected = _real_symmetric_coord_dim(matrix_dim)
    if coords.size != expected:
        raise ValueError("Coordinate vector has incompatible length.")
    matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.float64)
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            value = float(coords[cursor])
            cursor += 1
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def _restrict_hermitian_action_to_real_symmetric(
    action,
    *,
    target_matrix_dim: int,
    source_matrix_dim: int,
):
    target_coord_dim, source_coord_dim, rows, cols, vals = action
    if int(target_coord_dim) != int(target_matrix_dim * target_matrix_dim):
        raise ValueError("Incompatible target Hermitian coordinate dimension.")
    if int(source_coord_dim) != int(source_matrix_dim * source_matrix_dim):
        raise ValueError("Incompatible source Hermitian coordinate dimension.")
    target_real_dim = _real_symmetric_coord_dim(target_matrix_dim)
    source_real_dim = _real_symmetric_coord_dim(source_matrix_dim)
    mask = (rows < target_real_dim) & (cols < source_real_dim)
    return (
        int(target_real_dim),
        int(source_real_dim),
        rows[mask].astype(np.int32, copy=False),
        cols[mask].astype(np.int32, copy=False),
        vals[mask].astype(np.float64, copy=False),
    )


@lru_cache(maxsize=None)
def cached_real_symmetric_trace_coordinate_action(matrix_dim: int):
    return _restrict_hermitian_action_to_real_symmetric(
        _trace_coordinate_action(int(matrix_dim)),
        target_matrix_dim=1,
        source_matrix_dim=int(matrix_dim),
    )


@lru_cache(maxsize=None)
def cached_real_symmetric_coordinate_action_for_slot_permutation(
    slot_dims: tuple[int, ...],
    slot_permutation: tuple[int, ...],
):
    matrix_dim = product_dim(slot_dims)
    return _restrict_hermitian_action_to_real_symmetric(
        cached_coordinate_action_for_slot_permutation(slot_dims, slot_permutation),
        target_matrix_dim=matrix_dim,
        source_matrix_dim=matrix_dim,
    )


@lru_cache(maxsize=None)
def cached_real_symmetric_partial_trace_coordinate_action(
    slot_dims: tuple[int, ...],
    keep_positions: tuple[int, ...],
):
    source_matrix_dim = product_dim(slot_dims)
    target_dims = tuple(slot_dims[pos] for pos in keep_positions)
    target_matrix_dim = product_dim(target_dims) if target_dims else 1
    return _restrict_hermitian_action_to_real_symmetric(
        cached_partial_trace_coordinate_action(slot_dims, keep_positions),
        target_matrix_dim=target_matrix_dim,
        source_matrix_dim=source_matrix_dim,
    )


@lru_cache(maxsize=None)
def cached_real_symmetric_partial_transpose_coordinate_action(
    slot_dims: tuple[int, ...],
    transpose_positions: tuple[int, ...],
):
    matrix_dim = product_dim(slot_dims)
    return _restrict_hermitian_action_to_real_symmetric(
        cached_partial_transpose_coordinate_action(slot_dims, transpose_positions),
        target_matrix_dim=matrix_dim,
        source_matrix_dim=matrix_dim,
    )


@lru_cache(maxsize=None)
def cached_scs_real_symmetric_psd_coordinate_action(matrix_dim: int):
    target_dim = _real_symmetric_coord_dim(matrix_dim)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    sqrt2 = float(np.sqrt(2.0))
    for row in range(int(matrix_dim)):
        for col in range(row, int(matrix_dim)):
            rows.append(_lower_triangle_columnwise_index(col, row, int(matrix_dim)))
            cols.append(_upper_triangle_coordinate_index(row, col, int(matrix_dim)))
            vals.append(1.0 if row == col else sqrt2)
    return (
        int(target_dim),
        int(target_dim),
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )
