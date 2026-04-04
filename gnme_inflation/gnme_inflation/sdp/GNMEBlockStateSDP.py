"""Block-symmetrized GNME state SDP backend.

This backend keeps the same GNME combinatorial layer and reduced-marginal
constraints as ``GNMEStateSDP.py`` but changes the encoding of the maximal full
variables ``tau_i``.

The working rule is:

1. build the local symmetry representation from ``GNMEProblem``,
2. parameterize ``tau_i`` by symmetry-adapted sector blocks,
3. apply reduced-marginal and PPT maps directly to those sector blocks.

The last point matters. If we reconstruct a full affine ``tau_i`` matrix and
then apply every constraint entrywise, we lose most of the benefit of the block
encoding. The representative and PPT layers below therefore read the sector
variables directly whenever the source is a block-symmetrized ``tau_i``.

Auxiliary reduced representatives and PPT auxiliaries are still treated as
Hermitian variables through the same realification helper, so the backend
remains consistent at the SDP-variable level.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numba import njit
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - purely optional UI dependency
    tqdm = None

from .GNMEStateSDP import (
    AssignedStateSDPDraft,
    FusionSolveResult,
    StateSDPDraft,
    _basis_permutation_map,
    _flat_index,
    _partial_transpose_entry_map,
    _progress_log,
    _progress_stride,
    _resolve_verbose,
    _sum_expr,
    _unflatten_index,
    product_dim,
    quotient_ppt_constraints,
    quotient_representative_constraints,
    symmetric_matrix_orbits,
)


@dataclass(frozen=True)
class BlockTauVariableData:
    """Symmetry-adapted block description of one full tau variable."""

    name: str
    Hermitian: bool
    complex_dim: int
    realified_dim: int
    symmetry_type: str
    sector_labels: Tuple[str, ...]
    sector_irrep_dims: Tuple[int, ...]
    sector_multiplicities: Tuple[int, ...]
    sector_complex_dims: Tuple[int, ...]
    parameter_count: int


@dataclass(frozen=True)
class TauSectorLayout:
    """Static symmetry-sector data for one maximal inflation variable.

    This object is purely structural. It contains the basis information coming
    from the local symmetry action, before any solver variables are declared.
    Later optimization steps should read this layout and build all reduced maps
    directly from it.
    """

    label: str
    irrep_dim: int
    multiplicity: int
    basis: np.ndarray


@dataclass(frozen=True)
class SymmetryAdaptedTauLayout:
    """Cached symmetry-adapted declaration of one full tau variable.

    The purpose of this object is to separate:

    1. symmetry analysis of the variable, and
    2. solver-specific declaration of block PSD variables.

    This is the first step toward a cleaner reduced-basis backend.
    """

    name: str
    lexorder: Tuple[str, ...]
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    symmetry_type: str
    sectors: Tuple[TauSectorLayout, ...]
    hermitian_parameter_count: int
    real_parameter_count: int


@dataclass(frozen=True)
class TauSectorExpressionData:
    """One Hermitian block variable inside a symmetry-adapted tau decomposition."""

    label: str
    Hermitian: bool
    irrep_dim: int
    multiplicity: int
    coordinate_dim: int
    coordinate_vector: object
    realified_expr: object


@dataclass
class FusionBlockStateSDPModel:
    """Concrete MOSEK Fusion model for the block-symmetrized backend."""

    model: object
    Hermitian: bool
    tau_variables: Dict[str, object]
    tau_block_variables: Dict[str, Tuple[TauSectorExpressionData, ...]]
    auxiliary_coordinate_vectors: Dict[str, object]
    auxiliary_variables: Dict[str, object]
    known_representative_coordinate_vectors: Dict[str, object]
    known_representative_variables: Dict[str, object]
    ppt_coordinate_vectors: Dict[str, object]
    ppt_variables: Dict[str, object]
    block_tau_data: Dict[str, BlockTauVariableData]
    constraint_counts: Dict[str, int]
    build_profile: Dict[str, object]


def _progress_bar(iterable, verbose: int, desc: str, total: int | None = None):
    """Optional tqdm wrapper used only for long model-building loops."""
    if verbose <= 0 or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


_MAP_CACHE_VERSION = "gnme_block_maps_v4"


def _persistent_map_cache_dir() -> Path:
    """Directory for persistent compiled reduced-map triplets."""
    custom_dir = os.environ.get("GNME_BLOCK_CACHE_DIR")
    if custom_dir:
        root = Path(custom_dir).expanduser()
    else:
        root = Path.home() / ".cache" / "gnme_inflation"
    cache_dir = root / _MAP_CACHE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _persistent_map_cache_path(cache_key: object) -> Path:
    payload = pickle.dumps((_MAP_CACHE_VERSION, cache_key), protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    return _persistent_map_cache_dir() / f"{digest}.pkl"


def _load_persistent_object(cache_key: object):
    """Load a raw cached object payload.

    This helper is shared by map-level and batch-level caches. The payload
    format is interpreted by the caller so the cache layer can stay generic.
    """
    path = _persistent_map_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_persistent_object(cache_key: object, payload) -> None:
    """Persist an arbitrary cache payload atomically when possible."""
    path = _persistent_map_cache_path(cache_key)
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        temp_path.replace(path)
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _load_persistent_triplet_map(cache_key: object):
    payload = _load_persistent_object(cache_key)
    if payload is None:
        return None
    try:
        target_coord_dim, rows, cols, vals = payload
        return (
            int(target_coord_dim),
            np.asarray(rows, dtype=np.int32),
            np.asarray(cols, dtype=np.int32),
            np.asarray(vals, dtype=np.float64),
        )
    except Exception:
        return None


def _save_persistent_triplet_map(
    cache_key: object,
    target_coord_dim: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
):
    payload = (
        int(target_coord_dim),
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )
    _save_persistent_object(cache_key, payload)


# ---------------------------------------------------------------------------
# Group / representation helpers
# ---------------------------------------------------------------------------


def _choose_reference_party_positions(lexorder: Tuple[str, ...]) -> Tuple[int, ...]:
    """Pick one party whose slots define the layer permutation.

    For the full GNME inflations considered here, every party appears exactly
    ``inflation_level`` times. Any party works; we pick the first one in
    lexorder order.
    """
    positions_by_party: Dict[str, List[int]] = {}
    for index, label in enumerate(lexorder):
        party = label.split("_", 1)[0]
        positions_by_party.setdefault(party, []).append(index)
    party = min(positions_by_party.keys())
    return tuple(positions_by_party[party])


def _party_signature_from_lexorder(lexorder: Tuple[str, ...]) -> Tuple[str, ...]:
    """Keep only the per-slot party pattern of a lexorder."""
    return tuple(label.split("_", 1)[0] for label in lexorder)


def _layout_signature(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> Tuple[object, ...]:
    """Structural signature for one symmetry-adapted tau layout."""
    return (
        _party_signature_from_lexorder(lexorder),
        tuple(int(dim) for dim in slot_dims),
        tuple(tuple(int(position) for position in permutation) for permutation in local_symmetry_perms),
    )



def _small_permutation_from_slots(
    lexorder: Tuple[str, ...],
    slot_permutation: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Extract the copy-layer permutation induced on one reference party."""
    reference_positions = _choose_reference_party_positions(lexorder)
    position_lookup = {position: idx for idx, position in enumerate(reference_positions)}
    return tuple(position_lookup[slot_permutation[position]] for position in reference_positions)



def _basis_maps_for_variable(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Unique basis maps paired with their reduced layer permutations."""
    seen = set()
    pairs = []
    for slot_permutation in local_symmetry_perms:
        basis_map = _basis_permutation_map(slot_dims, slot_permutation)
        if basis_map in seen:
            continue
        seen.add(basis_map)
        small_perm = _small_permutation_from_slots(lexorder, slot_permutation)
        pairs.append((small_perm, basis_map))
    return tuple(pairs)

def _dense_permutation_matrix(basis_map: Tuple[int, ...]) -> np.ndarray:
    dim = len(basis_map)
    matrix = np.zeros((dim, dim), dtype=complex)
    for column, row in enumerate(basis_map):
        matrix[row, column] = 1.0
    return matrix


def _permutation_sign(perm: Tuple[int, ...]) -> int:
    """Parity of a permutation."""
    inversions = 0
    for left in range(len(perm)):
        pivot = perm[left]
        for right in range(left + 1, len(perm)):
            if pivot > perm[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _orthonormal_basis_from_projector(projector: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Columns spanning the image of a Hermitian projector."""
    projector = 0.5 * (projector + projector.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    keep = eigenvalues > tol
    if not np.any(keep):
        return np.zeros((projector.shape[0], 0), dtype=np.complex128)
    return np.asarray(eigenvectors[:, keep], dtype=np.complex128)


def _exact_c2_sector_basis_from_basis_map(
    basis_map: Tuple[int, ...],
) -> Tuple[str, Tuple[Tuple[str, int, int, np.ndarray], ...]]:
    """Exact sparse sector basis for an involutive permutation action."""
    dim = len(basis_map)
    visited = np.zeros(dim, dtype=bool)
    plus_columns: List[np.ndarray] = []
    minus_columns: List[np.ndarray] = []
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    for index in range(dim):
        if visited[index]:
            continue
        partner = int(basis_map[index])
        if partner == index:
            column = np.zeros(dim, dtype=np.complex128)
            column[index] = 1.0
            plus_columns.append(column)
            visited[index] = True
            continue
        if int(basis_map[partner]) != index:
            raise ValueError("C2 basis construction requires an involution.")
        if partner < index:
            continue
        plus = np.zeros(dim, dtype=np.complex128)
        minus = np.zeros(dim, dtype=np.complex128)
        plus[index] = inv_sqrt2
        plus[partner] = inv_sqrt2
        minus[index] = inv_sqrt2
        minus[partner] = -inv_sqrt2
        plus_columns.append(plus)
        minus_columns.append(minus)
        visited[index] = True
        visited[partner] = True

    sectors = []
    if plus_columns:
        plus_basis = np.column_stack(plus_columns)
        sectors.append(("sector_even", 1, plus_basis.shape[1], plus_basis))
    if minus_columns:
        minus_basis = np.column_stack(minus_columns)
        sectors.append(("sector_odd", 1, minus_basis.shape[1], minus_basis))
    return "exact_c2", tuple(sectors)


def _exact_s3_sector_basis_from_pairs(
    basis_pairs: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...],
) -> Tuple[str, Tuple[Tuple[str, int, int, np.ndarray], ...]] | None:
    """Exact isotypic basis for the full S3 action on three layers."""
    reduced_permutations = tuple(pair[0] for pair in basis_pairs)
    if len(reduced_permutations) != 6 or {tuple(perm) for perm in reduced_permutations} != {
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    }:
        return None

    full_matrices = tuple(
        _dense_permutation_matrix(basis_map).astype(np.complex128, copy=False)
        for _small_perm, basis_map in basis_pairs
    )
    dim = full_matrices[0].shape[0]
    identity = np.eye(dim, dtype=np.complex128)

    trivial_projector = sum(full_matrices) / float(len(full_matrices))
    sign_projector = sum(
        _permutation_sign(small_perm) * full_matrix
        for (small_perm, _basis_map), full_matrix in zip(basis_pairs, full_matrices)
    ) / float(len(full_matrices))
    standard_projector = identity - trivial_projector - sign_projector

    trivial_basis = _orthonormal_basis_from_projector(trivial_projector)
    sign_basis = _orthonormal_basis_from_projector(sign_projector)

    q = np.asarray(
        [
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
            [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
            [0.0, -2.0 / np.sqrt(6.0)],
        ],
        dtype=np.float64,
    )
    small_rep_matrices = []
    for small_perm, _basis_map in basis_pairs:
        permutation_matrix = _dense_permutation_matrix(small_perm).real.astype(np.float64, copy=False)
        small_rep_matrices.append(q.T @ permutation_matrix @ q)

    e11 = sum(
        (2.0 / len(full_matrices)) * rep[0, 0] * full_matrix
        for rep, full_matrix in zip(small_rep_matrices, full_matrices)
    )
    e21 = sum(
        (2.0 / len(full_matrices)) * rep[1, 0] * full_matrix
        for rep, full_matrix in zip(small_rep_matrices, full_matrices)
    )

    standard_first = _orthonormal_basis_from_projector(e11)
    standard_second = e21 @ standard_first
    if standard_second.size:
        gram = standard_first.conj().T @ standard_second
        if np.linalg.norm(gram, ord="fro") > 1e-8:
            standard_second = standard_second - standard_first @ gram
        norms = np.linalg.norm(standard_second, axis=0)
        valid = norms > 1e-10
        standard_first = standard_first[:, valid]
        standard_second = standard_second[:, valid]
        if standard_second.size:
            standard_second = standard_second / norms[valid]

    sectors = []
    if trivial_basis.shape[1]:
        sectors.append(("sector_trivial", 1, trivial_basis.shape[1], trivial_basis))
    if sign_basis.shape[1]:
        sectors.append(("sector_sign", 1, sign_basis.shape[1], sign_basis))
    if standard_first.shape[1]:
        standard_basis = np.concatenate((standard_first, standard_second), axis=1)
        sectors.append(("sector_standard", 2, standard_first.shape[1], standard_basis))
    return "exact_s3", tuple(sectors)


def _exact_small_group_sector_basis_data(
    basis_pairs: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...],
) -> Tuple[str, Tuple[Tuple[str, int, int, np.ndarray], ...]] | None:
    """Exact reduced-basis path for the small groups that dominate level-3."""
    if len(basis_pairs) == 1:
        basis_map = basis_pairs[0][1]
        dimension = len(basis_map)
        return (
            "exact_order_1",
            (("sector_0", 1, dimension, np.eye(dimension, dtype=np.complex128)),),
        )

    if len(basis_pairs) == 2:
        basis_map = basis_pairs[1][1]
        if all(int(basis_map[int(basis_map[index])]) == index for index in range(len(basis_map))):
            return _exact_c2_sector_basis_from_basis_map(basis_map)

    return _exact_s3_sector_basis_from_pairs(basis_pairs)



def _stable_group_seed(reduced_permutations: Tuple[Tuple[int, ...], ...]) -> int:
    """Deterministic seed derived from the symmetry permutations."""
    seed = 2166136261
    for permutation in reduced_permutations:
        for value in permutation:
            seed ^= int(value) + 1
            seed = (seed * 16777619) & 0xFFFFFFFF
        seed ^= len(permutation) + 0x9E3779B9
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed or 1


def _random_group_algebra_hermitian(
    unitary_representation: Tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generic Hermitian element of the representation algebra."""
    matrix = np.zeros_like(unitary_representation[0], dtype=np.complex128)
    coefficients = rng.normal(size=len(unitary_representation)) + 1j * rng.normal(
        size=len(unitary_representation)
    )
    for coefficient, unitary in zip(coefficients, unitary_representation):
        matrix += coefficient * unitary
    return 0.5 * (matrix + matrix.conj().T)


def _cluster_hermitian_eigenspaces(
    hermitian: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[Tuple[float, np.ndarray], ...]:
    """Cluster degenerate eigenspaces of one Hermitian probe operator."""
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if eigenvalues.size == 0:
        return tuple()
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = tol * scale
    clusters = []
    start = 0
    while start < eigenvalues.size:
        end = start + 1
        while end < eigenvalues.size and abs(eigenvalues[end] - eigenvalues[end - 1]) <= threshold:
            end += 1
        clusters.append(
            (
                float(np.mean(eigenvalues[start:end])),
                np.asarray(eigenvectors[:, start:end], dtype=np.complex128),
            )
        )
        start = end
    return tuple(clusters)


def _group_eigenspaces_into_isotypic_components(
    eigenspaces: Tuple[Tuple[float, np.ndarray], ...],
    witness_matrices: Tuple[np.ndarray, ...],
    tol: float = 1e-8,
) -> Tuple[Tuple[int, ...], ...]:
    """Connect eigenspaces that belong to the same irreducible sector."""
    dimension_groups: Dict[int, List[int]] = {}
    for index, (_eigenvalue, basis) in enumerate(eigenspaces):
        dimension_groups.setdefault(int(basis.shape[1]), []).append(index)

    components = []
    witness_thresholds = tuple(
        tol * max(1.0, float(np.linalg.norm(matrix, ord="fro")))
        for matrix in witness_matrices
    )
    for indices in dimension_groups.values():
        if len(indices) == 1:
            components.append((indices[0],))
            continue
        visited = set()
        for start in indices:
            if start in visited:
                continue
            queue = [start]
            visited.add(start)
            component = []
            while queue:
                current = queue.pop()
                component.append(current)
                current_basis = eigenspaces[current][1]
                for candidate in indices:
                    if candidate in visited:
                        continue
                    candidate_basis = eigenspaces[candidate][1]
                    connected = False
                    for threshold, witness in zip(witness_thresholds, witness_matrices):
                        block = candidate_basis.conj().T @ witness @ current_basis
                        if float(np.linalg.norm(block, ord="fro")) > threshold:
                            connected = True
                            break
                    if connected:
                        visited.add(candidate)
                        queue.append(candidate)
            components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _align_isotypic_component_basis(
    eigenspaces: Tuple[Tuple[float, np.ndarray], ...],
    component: Tuple[int, ...],
    witness_matrices: Tuple[np.ndarray, ...],
    tol: float = 1e-8,
) -> Tuple[int, int, np.ndarray]:
    """Align multiplicity bases across one automatically detected sector."""
    sorted_component = tuple(sorted(component, key=lambda index: eigenspaces[index][0]))
    reference_index = sorted_component[0]
    reference_basis = eigenspaces[reference_index][1]
    multiplicity = int(reference_basis.shape[1])
    aligned_blocks = [reference_basis]

    witness_thresholds = tuple(
        tol * max(1.0, float(np.linalg.norm(matrix, ord="fro")))
        for matrix in witness_matrices
    )
    for index in sorted_component[1:]:
        basis = eigenspaces[index][1]
        aligned_basis = None
        for threshold, witness in zip(witness_thresholds, witness_matrices):
            block = basis.conj().T @ witness @ reference_basis
            if float(np.linalg.norm(block, ord="fro")) <= threshold:
                continue
            left, _singular_values, right_h = np.linalg.svd(block, full_matrices=False)
            aligned_basis = basis @ (left @ right_h)
            break
        if aligned_basis is None:
            raise RuntimeError(
                "Automatic symmetry decomposition failed to align multiplicity bases."
            )
        aligned_blocks.append(aligned_basis)

    irrep_dim = len(sorted_component)
    return irrep_dim, multiplicity, np.concatenate(aligned_blocks, axis=1)


def _automatic_sector_basis_data(
    unitary_representation: Tuple[np.ndarray, ...],
    reduced_permutations: Tuple[Tuple[int, ...], ...],
) -> Tuple[str, Tuple[Tuple[str, int, int, np.ndarray], ...]]:
    """Generic block decomposition from the symmetry group action alone."""
    if len(unitary_representation) == 1:
        dimension = int(unitary_representation[0].shape[0])
        return (
            "auto_order_1",
            (("sector_0", 1, dimension, np.eye(dimension, dtype=np.complex128)),),
        )

    rng = np.random.default_rng(_stable_group_seed(reduced_permutations))
    best_eigenspaces = None
    best_witnesses = None
    for _attempt in range(8):
        algebra_probe = _random_group_algebra_hermitian(unitary_representation, rng)
        eigenspaces = _cluster_hermitian_eigenspaces(algebra_probe)
        witness_matrices = tuple(
            _random_group_algebra_hermitian(unitary_representation, rng)
            for _ in range(3)
        )
        if best_eigenspaces is None or len(eigenspaces) > len(best_eigenspaces):
            best_eigenspaces = eigenspaces
            best_witnesses = witness_matrices
    if best_eigenspaces is None or best_witnesses is None:
        raise RuntimeError("Automatic symmetry decomposition failed to build probe operators.")

    components = _group_eigenspaces_into_isotypic_components(
        best_eigenspaces,
        best_witnesses,
    )
    sectors = []
    for sector_index, component in enumerate(components):
        irrep_dim, multiplicity, basis = _align_isotypic_component_basis(
            best_eigenspaces,
            component,
            best_witnesses,
        )
        sectors.append((f"sector_{sector_index}", irrep_dim, multiplicity, basis))
    return f"auto_order_{len(unitary_representation)}", tuple(sectors)



def _realify_constant(matrix: np.ndarray) -> np.ndarray:
    """Real block embedding of a complex matrix."""
    return np.block(
        [
            [np.real(matrix), -np.imag(matrix)],
            [np.imag(matrix), np.real(matrix)],
        ]
    )



def _entry(expr, row: int, col: int):
    return expr.index(np.array([int(row), int(col)], dtype=np.int32))



def _complex_entry_from_realified(realified_expr, complex_dim: int, row: int, col: int):
    """Real and imaginary parts of one complex entry from a realified matrix."""
    return (
        _entry(realified_expr, row, col),
        _entry(realified_expr, complex_dim + row, col),
    )



def _hermitian_trace_expr(realified_expr, complex_dim: int, Expr):
    """Trace of the underlying complex Hermitian matrix."""
    return _sum_expr((_entry(realified_expr, idx, idx) for idx in range(complex_dim)), Expr)


def _real_trace_expr(matrix_expr, matrix_dim: int, Expr):
    """Trace of a real symmetric matrix expression."""
    return _sum_expr((_entry(matrix_expr, idx, idx) for idx in range(matrix_dim)), Expr)


@lru_cache(maxsize=None)
def cached_partial_trace_selector_triplets(
    slot_dims: Tuple[int, ...],
    keep_positions: Tuple[int, ...],
):
    """Sparse selector data for reduced complex entries on a realified matrix."""
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
            vals_re = []
            rows_im = []
            cols_im = []
            vals_im = []
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
                vals_re.append(1.0)
                if row < col:
                    rows_im.append(complex_dim + full_row)
                    cols_im.append(full_col)
                    vals_im.append(1.0)
            selectors.append(
                (
                    row,
                    col,
                    np.asarray(rows_re, dtype=np.int32),
                    np.asarray(cols_re, dtype=np.int32),
                    tuple(vals_re),
                    np.asarray(rows_im, dtype=np.int32),
                    np.asarray(cols_im, dtype=np.int32),
                    tuple(vals_im),
                )
            )
    return tuple(selectors)


def _upper_triangle_coordinate_index(row: int, col: int, matrix_dim: int) -> int:
    """Index of a real upper-triangle coordinate in row-major order."""
    if not (0 <= row <= col < matrix_dim):
        raise ValueError("Invalid upper-triangle coordinate.")
    return row * matrix_dim - row * (row - 1) // 2 + (col - row)


def _strict_upper_coordinate_index(row: int, col: int, matrix_dim: int) -> int:
    """Index of a strict upper-triangle coordinate in row-major order."""
    if not (0 <= row < col < matrix_dim):
        raise ValueError("Invalid strict upper-triangle coordinate.")
    return row * (matrix_dim - 1) - row * (row - 1) // 2 + (col - row - 1)


@njit(cache=True)
def _real_selector_triplets_numba(
    selector_real: np.ndarray,
    target_row: int,
    tol: float,
):
    """Nonzero sparse triplets for one real-symmetric selector row."""
    multiplicity = selector_real.shape[0]
    count = 0
    for row in range(multiplicity):
        diag_value = selector_real[row, row]
        if abs(diag_value) > tol:
            count += 1
        for col in range(row + 1, multiplicity):
            value = selector_real[row, col] + selector_real[col, row]
            if abs(value) > tol:
                count += 1

    rows = np.empty(count, dtype=np.int32)
    cols = np.empty(count, dtype=np.int32)
    vals = np.empty(count, dtype=np.float64)

    cursor = 0
    coordinate = 0
    for row in range(multiplicity):
        diag_value = selector_real[row, row]
        if abs(diag_value) > tol:
            rows[cursor] = target_row
            cols[cursor] = coordinate
            vals[cursor] = diag_value
            cursor += 1
        coordinate += 1
        for col in range(row + 1, multiplicity):
            value = selector_real[row, col] + selector_real[col, row]
            if abs(value) > tol:
                rows[cursor] = target_row
                cols[cursor] = coordinate
                vals[cursor] = value
                cursor += 1
            coordinate += 1

    return rows, cols, vals


@njit(cache=True)
def _hermitian_selector_triplets_numba(
    selector_real: np.ndarray,
    selector_imag: np.ndarray,
    target_row_re: int,
    target_row_im: int,
    tol: float,
):
    """Nonzero sparse triplets for one Hermitian selector row pair."""
    multiplicity = selector_real.shape[0]
    real_count = multiplicity * (multiplicity + 1) // 2

    count = 0
    for row in range(multiplicity):
        diag_re = selector_real[row, row]
        diag_im = selector_imag[row, row]
        if abs(diag_re) > tol:
            count += 1
        if target_row_im >= 0 and abs(diag_im) > tol:
            count += 1
        for col in range(row + 1, multiplicity):
            u_re = selector_real[row, col]
            u_im = selector_imag[row, col]
            v_re = selector_real[col, row]
            v_im = selector_imag[col, row]

            real_coord_re = u_re + v_re
            imag_coord_re = u_im + v_im
            if abs(real_coord_re) > tol:
                count += 1
            if target_row_im >= 0 and abs(imag_coord_re) > tol:
                count += 1

            real_coord_im = u_im - v_im
            imag_coord_im = v_re - u_re
            if abs(real_coord_im) > tol:
                count += 1
            if target_row_im >= 0 and abs(imag_coord_im) > tol:
                count += 1

    rows = np.empty(count, dtype=np.int32)
    cols = np.empty(count, dtype=np.int32)
    vals = np.empty(count, dtype=np.float64)

    cursor = 0
    real_coordinate = 0
    imag_coordinate = real_count
    for row in range(multiplicity):
        diag_re = selector_real[row, row]
        diag_im = selector_imag[row, row]
        if abs(diag_re) > tol:
            rows[cursor] = target_row_re
            cols[cursor] = real_coordinate
            vals[cursor] = diag_re
            cursor += 1
        if target_row_im >= 0 and abs(diag_im) > tol:
            rows[cursor] = target_row_im
            cols[cursor] = real_coordinate
            vals[cursor] = diag_im
            cursor += 1
        real_coordinate += 1
        for col in range(row + 1, multiplicity):
            u_re = selector_real[row, col]
            u_im = selector_imag[row, col]
            v_re = selector_real[col, row]
            v_im = selector_imag[col, row]

            real_coord_re = u_re + v_re
            imag_coord_re = u_im + v_im
            if abs(real_coord_re) > tol:
                rows[cursor] = target_row_re
                cols[cursor] = real_coordinate
                vals[cursor] = real_coord_re
                cursor += 1
            if target_row_im >= 0 and abs(imag_coord_re) > tol:
                rows[cursor] = target_row_im
                cols[cursor] = real_coordinate
                vals[cursor] = imag_coord_re
                cursor += 1
            real_coordinate += 1

            real_coord_im = u_im - v_im
            imag_coord_im = v_re - u_re
            if abs(real_coord_im) > tol:
                rows[cursor] = target_row_re
                cols[cursor] = imag_coordinate
                vals[cursor] = real_coord_im
                cursor += 1
            if target_row_im >= 0 and abs(imag_coord_im) > tol:
                rows[cursor] = target_row_im
                cols[cursor] = imag_coordinate
                vals[cursor] = imag_coord_im
                cursor += 1
            imag_coordinate += 1

    return rows, cols, vals


@lru_cache(maxsize=1)
def _warm_numba_selector_kernels():
    """Compile the Numba selector kernels once before the heavy build loops."""
    tiny_real = np.zeros((1, 1), dtype=np.float64)
    tiny_imag = np.zeros((1, 1), dtype=np.float64)
    _real_selector_triplets_numba(tiny_real, 0, 1e-12)
    _hermitian_selector_triplets_numba(tiny_real, tiny_imag, 0, -1, 1e-12)


def _real_selector_triplets_from_sparse_supports(
    row_support_cols: np.ndarray,
    row_support_vals: np.ndarray,
    col_support_cols: np.ndarray,
    col_support_vals: np.ndarray,
    target_row: int,
    multiplicity: int,
    tol: float,
):
    """Sparse selector triplets for exact small-group real sectors.

    This is the real-symmetric analogue of `_real_selector_triplets_numba`
    specialized to the exact `order_1` / `C2` bases, where each full-space row
    only touches a tiny number of reduced coordinates.
    """
    if row_support_cols.size == 0 or col_support_cols.size == 0:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )

    coordinate_values: Dict[int, float] = {}
    for source_col, source_coeff in zip(col_support_cols, col_support_vals):
        for source_row, row_coeff in zip(row_support_cols, row_support_vals):
            value = float(source_coeff * row_coeff)
            if abs(value) <= tol:
                continue
            if source_row <= source_col:
                coordinate = _upper_triangle_coordinate_index(
                    int(source_row),
                    int(source_col),
                    multiplicity,
                )
            else:
                coordinate = _upper_triangle_coordinate_index(
                    int(source_col),
                    int(source_row),
                    multiplicity,
                )
            coordinate_values[coordinate] = coordinate_values.get(coordinate, 0.0) + value

    kept_items = [
        (coordinate, value)
        for coordinate, value in coordinate_values.items()
        if abs(value) > tol
    ]
    if not kept_items:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )
    kept_items.sort(key=lambda item: item[0])
    count = len(kept_items)
    rows = np.full(count, int(target_row), dtype=np.int32)
    cols = np.fromiter((coordinate for coordinate, _value in kept_items), dtype=np.int32, count=count)
    vals = np.fromiter((value for _coordinate, value in kept_items), dtype=np.float64, count=count)
    return rows, cols, vals


def _basis_row_supports(basis: np.ndarray, tol: float = 1e-12):
    """Reduced-coordinate supports of each full-space row of one exact basis."""
    basis = np.asarray(basis, dtype=np.complex128)
    if np.max(np.abs(np.imag(basis))) > tol:
        return None
    real_basis = np.ascontiguousarray(np.real(basis), dtype=np.float64)
    supports = []
    for row in range(real_basis.shape[0]):
        nz = np.flatnonzero(np.abs(real_basis[row]) > tol).astype(np.int32, copy=False)
        supports.append((nz, real_basis[row, nz].astype(np.float64, copy=False)))
    return tuple(supports)


def _exact_small_group_real_identity_coordinate_maps(
    slot_dims: Tuple[int, ...],
    symmetry_type: str,
    sectors,
):
    """Cheap identity-map compilation for exact `order_1` / `C2` real layouts."""
    if symmetry_type not in {"exact_order_1", "exact_c2"}:
        return None
    if any(int(irrep_dim) != 1 for _label, irrep_dim, _multiplicity, _basis in sectors):
        return None

    source_dim = product_dim(slot_dims)
    target_coord_dim = source_dim * (source_dim + 1) // 2
    tol = 1e-12

    sector_row_supports = []
    for _label, _irrep_dim, multiplicity, basis in sectors:
        row_supports = _basis_row_supports(np.asarray(basis, dtype=np.complex128), tol=tol)
        if row_supports is None:
            return None
        sector_row_supports.append((int(multiplicity), row_supports))

    per_sector_row_blocks = [[] for _ in sectors]
    per_sector_col_blocks = [[] for _ in sectors]
    per_sector_val_blocks = [[] for _ in sectors]

    for row in range(source_dim):
        for col in range(row, source_dim):
            target_row = _upper_triangle_coordinate_index(row, col, source_dim)
            for sector_index, (multiplicity, row_supports) in enumerate(sector_row_supports):
                rows, cols, vals = _real_selector_triplets_from_sparse_supports(
                    row_supports[row][0],
                    row_supports[row][1],
                    row_supports[col][0],
                    row_supports[col][1],
                    target_row,
                    multiplicity,
                    tol,
                )
                if rows.size:
                    per_sector_row_blocks[sector_index].append(rows)
                    per_sector_col_blocks[sector_index].append(cols)
                    per_sector_val_blocks[sector_index].append(vals)

    return (
        target_coord_dim,
        tuple(
            (
                np.concatenate(rows).astype(np.int32, copy=False)
                if rows
                else np.asarray([], dtype=np.int32),
                np.concatenate(cols).astype(np.int32, copy=False)
                if cols
                else np.asarray([], dtype=np.int32),
                np.concatenate(vals).astype(np.float64, copy=False)
                if vals
                else np.asarray([], dtype=np.float64),
            )
            for rows, cols, vals in zip(
                per_sector_row_blocks,
                per_sector_col_blocks,
                per_sector_val_blocks,
            )
        ),
    )


@lru_cache(maxsize=None)
def _cached_block_partial_trace_coordinate_maps(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    keep_positions: Tuple[int, ...],
    Hermitian: bool = True,
):
    """Sparse maps from tau sector coordinates to reduced matrix coordinates."""
    symmetry_type, sectors = _cached_sector_basis_data(
        party_signature,
        slot_dims,
        local_symmetry_perms,
    )
    if (
        not Hermitian
        and keep_positions == tuple(range(len(slot_dims)))
    ):
        exact_identity_maps = _exact_small_group_real_identity_coordinate_maps(
            slot_dims,
            symmetry_type,
            sectors,
        )
        if exact_identity_maps is not None:
            return exact_identity_maps
    triplets = cached_partial_trace_selector_triplets(slot_dims, keep_positions)
    source_dim = product_dim(slot_dims)
    kept_dims = tuple(slot_dims[pos] for pos in keep_positions)
    target_dim = product_dim(kept_dims) if kept_dims else 1
    target_real_count = target_dim * (target_dim + 1) // 2
    target_coord_dim = target_dim * target_dim if Hermitian else target_real_count

    per_sector_row_blocks = [[] for _ in sectors]
    per_sector_col_blocks = [[] for _ in sectors]
    per_sector_val_blocks = [[] for _ in sectors]
    tol = 1e-12

    sector_linear_data = []
    irrep_offsets = {}
    unique_irrep_dims = tuple(sorted({irrep_dim for _, irrep_dim, _, _ in sectors}))
    for irrep_dim in unique_irrep_dims:
        irrep_offsets[irrep_dim] = np.arange(irrep_dim, dtype=np.int64)

    for _label, irrep_dim, multiplicity, basis in sectors:
        basis_blocks = np.asarray(basis, dtype=np.complex128).reshape(
            source_dim,
            irrep_dim,
            multiplicity,
        )
        basis_flat = np.ascontiguousarray(basis_blocks.reshape(source_dim * irrep_dim, multiplicity))
        sector_linear_data.append(
            (
                irrep_dim,
                multiplicity,
                basis_flat,
                np.ascontiguousarray(np.conjugate(basis_flat)),
            )
        )

    for row, col, rows_re, cols_re, *_ in triplets:
        if row <= col:
            target_row_re = _upper_triangle_coordinate_index(row, col, target_dim)
        else:
            continue
        if row < col:
            target_row_im = target_real_count + _strict_upper_coordinate_index(row, col, target_dim)
        else:
            target_row_im = None

        rows_re_int = rows_re.astype(np.int64, copy=False)
        cols_re_int = cols_re.astype(np.int64, copy=False)
        expanded_indices_by_irrep = {}
        for irrep_dim in unique_irrep_dims:
            offsets = irrep_offsets[irrep_dim]
            expanded_indices_by_irrep[irrep_dim] = (
                np.ascontiguousarray((rows_re_int[:, None] * irrep_dim + offsets[None, :]).reshape(-1)),
                np.ascontiguousarray((cols_re_int[:, None] * irrep_dim + offsets[None, :]).reshape(-1)),
            )

        for sector_index, (irrep_dim, _multiplicity, basis_flat, basis_flat_conj) in enumerate(sector_linear_data):
            flat_rows, flat_cols = expanded_indices_by_irrep[irrep_dim]
            selector = np.take(basis_flat, flat_cols, axis=0).T @ np.take(
                basis_flat_conj,
                flat_rows,
                axis=0,
            )
            if Hermitian:
                rows, cols, vals = _hermitian_selector_triplets_numba(
                    np.ascontiguousarray(np.real(selector)),
                    np.ascontiguousarray(np.imag(selector)),
                    target_row_re,
                    -1 if target_row_im is None else target_row_im,
                    tol,
                )
            else:
                rows, cols, vals = _real_selector_triplets_numba(
                    np.ascontiguousarray(np.real(selector)),
                    target_row_re,
                    tol,
                )
            if rows.size:
                per_sector_row_blocks[sector_index].append(rows)
                per_sector_col_blocks[sector_index].append(cols)
                per_sector_val_blocks[sector_index].append(vals)

    return (
        target_coord_dim,
        tuple(
            (
                np.concatenate(rows).astype(np.int32, copy=False)
                if rows
                else np.asarray([], dtype=np.int32),
                np.concatenate(cols).astype(np.int32, copy=False)
                if cols
                else np.asarray([], dtype=np.int32),
                np.concatenate(vals).astype(np.float64, copy=False)
                if vals
                else np.asarray([], dtype=np.float64),
            )
            for rows, cols, vals in zip(
                per_sector_row_blocks,
                per_sector_col_blocks,
                per_sector_val_blocks,
            )
        ),
    )


def cached_block_partial_trace_coordinate_maps(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    keep_positions: Tuple[int, ...],
    Hermitian: bool = True,
):
    return _cached_block_partial_trace_coordinate_maps(
        _party_signature_from_lexorder(lexorder),
        slot_dims,
        local_symmetry_perms,
        keep_positions,
        Hermitian,
    )


@lru_cache(maxsize=None)
def cached_partial_transpose_coordinate_action(
    slot_dims: Tuple[int, ...],
    transpose_positions: Tuple[int, ...],
    Hermitian: bool = True,
):
    """Coordinate action induced by partial transpose."""
    matrix_dim = product_dim(slot_dims)
    real_count = matrix_dim * (matrix_dim + 1) // 2
    coord_dim = matrix_dim * matrix_dim if Hermitian else real_count

    source_to_target_row = np.empty(coord_dim, dtype=np.int32)
    source_to_target_sign = np.empty(coord_dim, dtype=np.float64)

    target_row = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            mapped_row, mapped_col = _partial_transpose_entry_map(
                row,
                col,
                slot_dims,
                transpose_positions,
            )
            source_row = _upper_triangle_coordinate_index(
                min(mapped_row, mapped_col),
                max(mapped_row, mapped_col),
                matrix_dim,
            )
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = 1.0
            target_row += 1

    if Hermitian:
        for row in range(matrix_dim):
            for col in range(row + 1, matrix_dim):
                mapped_row, mapped_col = _partial_transpose_entry_map(
                    row,
                    col,
                    slot_dims,
                    transpose_positions,
                )
                if mapped_row < mapped_col:
                    source_row = real_count + _strict_upper_coordinate_index(
                        mapped_row,
                        mapped_col,
                        matrix_dim,
                    )
                    sign = 1.0
                else:
                    source_row = real_count + _strict_upper_coordinate_index(
                        mapped_col,
                        mapped_row,
                        matrix_dim,
                    )
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

    return coord_dim, rows, cols, vals, source_to_target_row, source_to_target_sign


@lru_cache(maxsize=None)
def cached_coordinate_action_for_slot_permutation(
    slot_dims: Tuple[int, ...],
    slot_permutation: Tuple[int, ...],
    Hermitian: bool = True,
):
    """Coordinate action induced by permuting subsystem slots."""
    matrix_dim = product_dim(slot_dims)
    real_count = matrix_dim * (matrix_dim + 1) // 2
    coord_dim = matrix_dim * matrix_dim if Hermitian else real_count
    basis_map = _basis_permutation_map(slot_dims, slot_permutation)

    source_to_target_row = np.empty(coord_dim, dtype=np.int32)
    source_to_target_sign = np.empty(coord_dim, dtype=np.float64)

    target_row = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            mapped_row = basis_map[row]
            mapped_col = basis_map[col]
            source_row = _upper_triangle_coordinate_index(
                min(mapped_row, mapped_col),
                max(mapped_row, mapped_col),
                matrix_dim,
            )
            source_to_target_row[source_row] = target_row
            source_to_target_sign[source_row] = 1.0
            target_row += 1

    if Hermitian:
        for row in range(matrix_dim):
            for col in range(row + 1, matrix_dim):
                mapped_row = basis_map[row]
                mapped_col = basis_map[col]
                if mapped_row < mapped_col:
                    source_row = real_count + _strict_upper_coordinate_index(
                        mapped_row,
                        mapped_col,
                        matrix_dim,
                    )
                    sign = 1.0
                else:
                    source_row = real_count + _strict_upper_coordinate_index(
                        mapped_col,
                        mapped_row,
                        matrix_dim,
                    )
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

    return coord_dim, rows, cols, vals, source_to_target_row, source_to_target_sign


@lru_cache(maxsize=None)
def _cached_block_partial_transpose_coordinate_maps(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    transpose_positions: Tuple[int, ...],
    Hermitian: bool = True,
):
    """Sparse maps from tau sector coordinates to PPT output coordinates."""
    keep_positions = tuple(range(len(slot_dims)))
    target_coord_dim, sector_maps = _cached_block_partial_trace_coordinate_maps(
        party_signature,
        slot_dims,
        local_symmetry_perms,
        keep_positions,
        Hermitian,
    )
    (
        _coord_dim,
        _rows,
        _cols,
        _vals,
        source_to_target_row,
        source_to_target_sign,
    ) = cached_partial_transpose_coordinate_action(
        slot_dims,
        transpose_positions,
        Hermitian,
    )

    transformed_sector_maps = []
    for rows, cols, vals in sector_maps:
        if vals.size == 0:
            transformed_sector_maps.append(
                (
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.float64),
                )
            )
            continue
        transformed_sector_maps.append(
            (
                source_to_target_row[rows].astype(np.int32, copy=False),
                cols,
                (source_to_target_sign[rows] * vals).astype(np.float64, copy=False),
            )
        )

    return target_coord_dim, tuple(transformed_sector_maps)


def cached_block_partial_transpose_coordinate_maps(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    transpose_positions: Tuple[int, ...],
    Hermitian: bool = True,
):
    return _cached_block_partial_transpose_coordinate_maps(
        _party_signature_from_lexorder(lexorder),
        slot_dims,
        local_symmetry_perms,
        transpose_positions,
        Hermitian,
    )


@lru_cache(maxsize=None)
def _cached_tau_ppt_orbit_reduction(
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
    transpose_positions: Tuple[int, ...],
):
    """Orbit basis for the partial-transpose stabilizer of one tau source."""
    slot_count = len(slot_dims)
    transpose_key = tuple(sorted(transpose_positions))
    stabilizer = tuple(
        permutation
        for permutation in local_symmetry_perms
        if tuple(sorted(permutation[position] for position in transpose_positions)) == transpose_key
    )
    if not stabilizer:
        stabilizer = (tuple(range(slot_count)),)
    orbit_data = symmetric_matrix_orbits(slot_dims, stabilizer)
    representative_rows = np.asarray(
        [
            _upper_triangle_coordinate_index(row, col, orbit_data.matrix_dim)
            for row, col in orbit_data.orbit_representatives
        ],
        dtype=np.int32,
    )
    row_lookup = np.full(orbit_data.upper_triangular_entries, -1, dtype=np.int32)
    row_lookup[representative_rows] = np.arange(representative_rows.size, dtype=np.int32)
    return stabilizer, orbit_data, row_lookup


def _matrix_coordinate_vector_expr(real_scalars, imag_scalars, Expr, Hermitian: bool):
    """Coordinate vector for either Hermitian or real-symmetric blocks."""
    if not Hermitian:
        return Expr.flatten(real_scalars)
    if imag_scalars is None:
        return Expr.flatten(real_scalars)
    return Expr.vstack(Expr.flatten(real_scalars), Expr.flatten(imag_scalars))


@lru_cache(maxsize=None)
def _cached_real_symmetric_lifting_triplets(matrix_dim: int):
    """Sparse lifting triplets for a real symmetric upper-triangle parametrization."""
    rows = []
    cols = []
    vals = []
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row, matrix_dim):
            rows.append(row * matrix_dim + col)
            cols.append(cursor)
            vals.append(1.0)
            if row != col:
                rows.append(col * matrix_dim + row)
                cols.append(cursor)
                vals.append(1.0)
            cursor += 1
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )


@lru_cache(maxsize=None)
def _cached_real_skew_lifting_triplets(matrix_dim: int):
    """Sparse lifting triplets for a real skew-symmetric strict-upper parametrization."""
    rows = []
    cols = []
    vals = []
    cursor = 0
    for row in range(matrix_dim):
        for col in range(row + 1, matrix_dim):
            rows.append(row * matrix_dim + col)
            cols.append(cursor)
            vals.append(1.0)
            rows.append(col * matrix_dim + row)
            cols.append(cursor)
            vals.append(-1.0)
            cursor += 1
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(vals, dtype=np.float64),
    )


def _stack_coordinate_vectors(vectors, Expr):
    """One stacked coordinate vector from sector coordinate vectors."""
    vectors = tuple(vectors)
    if not vectors:
        raise ValueError("At least one coordinate vector is required.")
    if len(vectors) == 1:
        return vectors[0]
    return Expr.vstack(*vectors)


def _stack_rhs_blocks(blocks, Expr, Matrix):
    """Stack coordinate right-hand sides, allowing numeric and symbolic blocks."""
    expr_blocks = []
    for block in blocks:
        if isinstance(block, np.ndarray):
            expr_blocks.append(Matrix.dense(np.asarray(block, dtype=np.float64).reshape(-1, 1)))
        else:
            expr_blocks.append(block)
    if not expr_blocks:
        raise ValueError("At least one right-hand-side block is required.")
    if len(expr_blocks) == 1:
        return expr_blocks[0]
    return Expr.vstack(np.asarray(expr_blocks, dtype=object))


_REPRESENTATIVE_MAX_BLOCKS_PER_CHUNK = 24
_REPRESENTATIVE_MAX_ROWS_PER_CHUNK = 4096
_REPRESENTATIVE_MAX_NNZ_PER_CHUNK = 10_000_000
_REPRESENTATIVE_MAX_DENSE_BYTES_PER_CHUNK = 128_000_000

_PPT_MAX_BLOCKS_PER_CHUNK = 16
_PPT_MAX_ROWS_PER_CHUNK = 2048
_PPT_MAX_NNZ_PER_CHUNK = 5_000_000
_PPT_MAX_DENSE_BYTES_PER_CHUNK = 64_000_000
_PPT_MAX_ROWS_PER_EMIT = 65_536

_GLOBAL_PRECOMPILE_CANONICAL_MAP_LIMIT = 96


def _chunk_constraint_blocks(
    constraints,
    block_rows_fn,
    max_blocks_per_chunk: int = 48,
    max_rows_per_chunk: int = 4096,
    max_nnz_per_chunk: int = 25_000_000,
    max_dense_bytes_per_chunk: int = 256_000_000,
    constraint_nnz_fn=None,
    constraint_dense_bytes_fn=None,
):
    """Split one batch into smaller chunks using output rows and predicted nnz.

    The limits are backend heuristics, not mathematical parameters. The same
    logic is used for representative and PPT batches.
    """
    chunks = []
    current_chunk = []
    current_rows = 0
    current_nnz = 0
    current_dense_bytes = 0

    for constraint in constraints:
        block_rows = int(block_rows_fn(constraint))
        block_nnz = 0 if constraint_nnz_fn is None else int(constraint_nnz_fn(constraint))
        block_dense_bytes = (
            0
            if constraint_dense_bytes_fn is None
            else int(constraint_dense_bytes_fn(constraint))
        )
        would_overflow_rows = current_chunk and (current_rows + block_rows > max_rows_per_chunk)
        would_overflow_blocks = current_chunk and (len(current_chunk) >= max_blocks_per_chunk)
        would_overflow_nnz = (
            current_chunk
            and constraint_nnz_fn is not None
            and (current_nnz + block_nnz > max_nnz_per_chunk)
        )
        would_overflow_dense_bytes = (
            current_chunk
            and constraint_dense_bytes_fn is not None
            and (current_dense_bytes + block_dense_bytes > max_dense_bytes_per_chunk)
        )
        if (
            would_overflow_rows
            or would_overflow_blocks
            or would_overflow_nnz
            or would_overflow_dense_bytes
        ):
            chunks.append(tuple(current_chunk))
            current_chunk = []
            current_rows = 0
            current_nnz = 0
            current_dense_bytes = 0
        current_chunk.append(constraint)
        current_rows += block_rows
        current_nnz += block_nnz
        current_dense_bytes += block_dense_bytes

    if current_chunk:
        chunks.append(tuple(current_chunk))

    return tuple(chunks)


def _chunk_representative_constraints(
    constraints_for_source,
    representative_coord_dims: Dict[str, int],
    max_blocks_per_chunk: int = _REPRESENTATIVE_MAX_BLOCKS_PER_CHUNK,
    max_rows_per_chunk: int = _REPRESENTATIVE_MAX_ROWS_PER_CHUNK,
    max_nnz_per_chunk: int = _REPRESENTATIVE_MAX_NNZ_PER_CHUNK,
    max_dense_bytes_per_chunk: int = _REPRESENTATIVE_MAX_DENSE_BYTES_PER_CHUNK,
    constraint_nnz_fn=None,
    constraint_dense_bytes_fn=None,
):
    """Representative-specific wrapper around `_chunk_constraint_blocks`."""
    return _chunk_constraint_blocks(
        constraints_for_source,
        block_rows_fn=lambda constraint: representative_coord_dims[constraint.representative_name],
        max_blocks_per_chunk=max_blocks_per_chunk,
        max_rows_per_chunk=max_rows_per_chunk,
        max_nnz_per_chunk=max_nnz_per_chunk,
        max_dense_bytes_per_chunk=max_dense_bytes_per_chunk,
        constraint_nnz_fn=constraint_nnz_fn,
        constraint_dense_bytes_fn=constraint_dense_bytes_fn,
    )


def _slice_sorted_triplets(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    row_start: int,
    row_stop: int,
):
    """Slice one sorted sparse triplet array to a contiguous output-row range."""
    if rows.size == 0:
        return rows, cols, vals
    start = int(np.searchsorted(rows, row_start, side="left"))
    stop = int(np.searchsorted(rows, row_stop, side="left"))
    if start >= stop:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=cols.dtype),
            np.asarray([], dtype=vals.dtype),
        )
    return (
        (rows[start:stop] - row_start).astype(np.int32, copy=False),
        cols[start:stop],
        vals[start:stop],
    )


def _slice_sector_triplets(sector_triplets, row_start: int, row_stop: int):
    """Slice one sectorwise sparse payload to a contiguous output-row range."""
    return tuple(
        _slice_sorted_triplets(rows, cols, vals, row_start, row_stop)
        for rows, cols, vals in sector_triplets
    )


def _restrict_sector_triplets_to_row_lookup(sector_triplets, row_lookup: np.ndarray):
    """Keep only rows present in `row_lookup`, remapping them densely."""
    restricted_triplets = []
    for rows, cols, vals in sector_triplets:
        if rows.size == 0:
            restricted_triplets.append(
                (
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.float64),
                )
            )
            continue
        remapped_rows = row_lookup[rows]
        mask = remapped_rows >= 0
        if not np.any(mask):
            restricted_triplets.append(
                (
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.int32),
                    np.asarray([], dtype=np.float64),
                )
            )
            continue
        kept_rows = remapped_rows[mask].astype(np.int32, copy=False)
        kept_cols = cols[mask].astype(np.int32, copy=False)
        kept_vals = vals[mask].astype(np.float64, copy=False)
        if kept_rows.size > 1:
            order = np.argsort(kept_rows, kind="stable")
            kept_rows = kept_rows[order]
            kept_cols = kept_cols[order]
            kept_vals = kept_vals[order]
        restricted_triplets.append(
            (
                kept_rows,
                kept_cols,
                kept_vals,
            )
        )
    return tuple(restricted_triplets)


def _slice_stacked_expr_blocks(blocks, block_dims, row_start: int, row_stop: int, Expr):
    """Slice a stacked list of coordinate vectors without materializing the full stack."""
    sliced_blocks = []
    offset = 0
    for block, block_dim in zip(blocks, block_dims):
        block_dim = int(block_dim)
        next_offset = offset + block_dim
        if next_offset <= row_start:
            offset = next_offset
            continue
        if offset >= row_stop:
            break
        local_start = max(0, row_start - offset)
        local_stop = min(block_dim, row_stop - offset)
        if local_start < local_stop:
            if local_start == 0 and local_stop == block_dim:
                sliced_blocks.append(block)
            else:
                sliced_blocks.append(block.slice(int(local_start), int(local_stop)))
        offset = next_offset
    return _stack_coordinate_vectors(tuple(sliced_blocks), Expr)


def _iter_emit_row_ranges(total_rows: int, max_rows_per_emit: int):
    """Contiguous row ranges for splitting oversized Fusion equalities."""
    total_rows = int(total_rows)
    max_rows_per_emit = int(max_rows_per_emit)
    if total_rows <= 0:
        return tuple()
    if max_rows_per_emit <= 0 or total_rows <= max_rows_per_emit:
        return ((0, total_rows),)
    return tuple(
        (row_start, min(total_rows, row_start + max_rows_per_emit))
        for row_start in range(0, total_rows, max_rows_per_emit)
    )


def _combine_sector_triplets(
    target_coord_dim: int,
    sector_maps,
    sector_coordinate_dims: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine per-sector sparse triplets into one stacked-coordinate map."""
    row_blocks = []
    col_blocks = []
    val_blocks = []
    column_offset = 0
    for (rows, cols, vals), coordinate_dim in zip(sector_maps, sector_coordinate_dims):
        if vals.size:
            row_blocks.append(rows.astype(np.int32, copy=False))
            col_blocks.append((cols + column_offset).astype(np.int32, copy=False))
            val_blocks.append(vals.astype(np.float64, copy=False))
        column_offset += coordinate_dim
    if not row_blocks:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float64),
        )
    return (
        np.concatenate(row_blocks).astype(np.int32, copy=False),
        np.concatenate(col_blocks).astype(np.int32, copy=False),
        np.concatenate(val_blocks).astype(np.float64, copy=False),
    )


def _build_real_symmetric_matrix(
    M,
    name: str,
    matrix_dim: int,
    Domain,
    Expr,
    Matrix,
):
    """Affine expression for a real symmetric matrix from upper-triangle scalars."""
    variable_count = matrix_dim * (matrix_dim + 1) // 2
    scalars = M.variable(f"{name}_re", variable_count, Domain.unbounded())
    matrix_expr = real_symmetric_matrix_from_coordinate_expr(
        scalars,
        matrix_dim,
        Expr,
        Matrix,
    )
    return scalars, matrix_expr



def _build_real_skew_matrix(
    M,
    name: str,
    matrix_dim: int,
    Domain,
    Expr,
    Matrix,
):
    """Affine expression for a real skew-symmetric matrix."""
    variable_count = matrix_dim * (matrix_dim - 1) // 2
    if variable_count == 0:
        return None, Expr.zeros(matrix_dim, matrix_dim)
    scalars = M.variable(f"{name}_im", variable_count, Domain.unbounded())
    rows, cols, vals = _cached_real_skew_lifting_triplets(matrix_dim)
    lifting = Matrix.sparse(matrix_dim * matrix_dim, variable_count, rows, cols, vals)
    matrix_expr = Expr.reshape(Expr.mul(lifting, scalars), matrix_dim, matrix_dim)
    return scalars, matrix_expr



def real_symmetric_matrix_from_coordinate_expr(
    coordinate_expr,
    matrix_dim: int,
    Expr,
    Matrix,
):
    """Affine symmetric matrix expression from upper-triangle coordinates."""
    variable_count = matrix_dim * (matrix_dim + 1) // 2
    rows, cols, vals = _cached_real_symmetric_lifting_triplets(matrix_dim)
    lifting = Matrix.sparse(matrix_dim * matrix_dim, variable_count, rows, cols, vals)
    return Expr.reshape(Expr.mul(lifting, coordinate_expr), matrix_dim, matrix_dim)


@lru_cache(maxsize=None)
def _cached_real_svec_scaling_triplets(matrix_dim: int):
    """Diagonal scaling taking upper-triangle coordinates to MOSEK's sVec form."""
    coord_dim = matrix_dim * (matrix_dim + 1) // 2
    rows = np.arange(coord_dim, dtype=np.int32)
    cols = rows.copy()
    vals = np.empty(coord_dim, dtype=np.float64)
    cursor = 0
    sqrt2 = np.sqrt(2.0)
    for row in range(matrix_dim):
        vals[cursor] = 1.0
        cursor += 1
        for _col in range(row + 1, matrix_dim):
            vals[cursor] = sqrt2
            cursor += 1
    return rows, cols, vals


def real_svec_expr_from_coordinate_expr(
    coordinate_expr,
    matrix_dim: int,
    Expr,
    Matrix,
):
    """Scaled symmetric-vector expression compatible with `Domain.inSVecPSDCone`."""
    coord_dim = matrix_dim * (matrix_dim + 1) // 2
    rows, cols, vals = _cached_real_svec_scaling_triplets(matrix_dim)
    scaling = Matrix.sparse(coord_dim, coord_dim, rows, cols, vals)
    return Expr.mul(scaling, coordinate_expr)


def hermitian_psd_expression(
    M,
    name: str,
    complex_dim: int,
    Domain,
    Expr,
    Matrix,
):
    """Create a Hermitian PSD variable via realification.

    The returned matrix expression is the realified form
    ``[[Re(H), -Im(H)], [Im(H), Re(H)]]`` constrained to be real PSD.
    """
    real_scalars, real_part = _build_real_symmetric_matrix(M, name, complex_dim, Domain, Expr, Matrix)
    imag_scalars, imag_part = _build_real_skew_matrix(M, name, complex_dim, Domain, Expr, Matrix)
    realified = Expr.vstack(
        Expr.hstack(real_part, Expr.neg(imag_part)),
        Expr.hstack(imag_part, real_part),
    )
    M.constraint(f"psd_{name}", realified, Domain.inPSDCone(2 * complex_dim))
    return (real_scalars, imag_scalars), realified


def real_symmetric_psd_expression(
    M,
    name: str,
    matrix_dim: int,
    Domain,
    Expr,
    Matrix,
):
    """Create a real symmetric PSD variable."""
    real_scalars, real_matrix = _build_real_symmetric_matrix(
        M, name, matrix_dim, Domain, Expr, Matrix
    )
    M.constraint(f"psd_{name}", real_matrix, Domain.inPSDCone(matrix_dim))
    return (real_scalars, None), real_matrix



def _repeat_realified_block(block_expr, repeats: int, Expr):
    """Block diagonal matrix with repeated copies of the same realified block."""
    if repeats == 1:
        return block_expr
    if repeats < 1:
        raise ValueError("Number of repeats must be positive.")
    rows, cols = map(int, block_expr.getShape())
    zero = Expr.zeros(rows, cols)
    blocks = []
    for row in range(repeats):
        row_blocks = []
        for col in range(repeats):
            row_blocks.append(block_expr if row == col else zero)
        blocks.append(Expr.hstack(*row_blocks))
    return Expr.vstack(*blocks)



def _sector_basis_data(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
):
    """Automatic complex isotypic bases from the generated local symmetry action."""
    basis_pairs = _basis_maps_for_variable(lexorder, slot_dims, local_symmetry_perms)
    exact_sector_data = _exact_small_group_sector_basis_data(basis_pairs)
    if exact_sector_data is not None:
        return exact_sector_data
    reduced_permutations = tuple(pair[0] for pair in basis_pairs)
    basis_maps = tuple(pair[1] for pair in basis_pairs)
    unitary_representation = tuple(_dense_permutation_matrix(basis_map) for basis_map in basis_maps)
    return _automatic_sector_basis_data(unitary_representation, reduced_permutations)


@lru_cache(maxsize=None)
def _cached_sector_basis_data(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
):
    cache_key = (
        "sector_basis_v2",
        party_signature,
        slot_dims,
        local_symmetry_perms,
    )
    cached = _load_persistent_object(cache_key)
    if cached is not None:
        try:
            symmetry_type, sectors = cached
            restored_sectors = tuple(
                (
                    str(label),
                    int(irrep_dim),
                    int(multiplicity),
                    np.asarray(basis, dtype=np.complex128),
                )
                for label, irrep_dim, multiplicity, basis in sectors
            )
            return symmetry_type, restored_sectors
        except Exception:
            pass

    sector_data = _sector_basis_data(party_signature, slot_dims, local_symmetry_perms)
    try:
        symmetry_type, sectors = sector_data
        payload = (
            symmetry_type,
            tuple(
                (
                    label,
                    irrep_dim,
                    multiplicity,
                    np.asarray(basis, dtype=np.complex128),
                )
                for label, irrep_dim, multiplicity, basis in sectors
            ),
        )
        _save_persistent_object(cache_key, payload)
    except Exception:
        pass
    return sector_data


def cached_sector_basis_data(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
):
    return _cached_sector_basis_data(
        _party_signature_from_lexorder(lexorder),
        slot_dims,
        local_symmetry_perms,
    )


@lru_cache(maxsize=None)
def _cached_tau_layout_structure(
    party_signature: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
):
    symmetry_type, sectors = _cached_sector_basis_data(
        party_signature,
        slot_dims,
        local_symmetry_perms,
    )
    sector_layouts = tuple(
        TauSectorLayout(
            label=label,
            irrep_dim=irrep_dim,
            multiplicity=multiplicity,
            basis=basis,
        )
        for label, irrep_dim, multiplicity, basis in sectors
    )
    hermitian_parameter_count = sum(
        layout.multiplicity * layout.multiplicity
        for layout in sector_layouts
    )
    real_parameter_count = sum(
        layout.multiplicity * (layout.multiplicity + 1) // 2
        for layout in sector_layouts
    )
    return symmetry_type, sector_layouts, hermitian_parameter_count, real_parameter_count


def cached_tau_layout(
    name: str,
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> SymmetryAdaptedTauLayout:
    """Analyze one full inflation variable and cache its symmetry-adapted layout.

    This is the declaration-level object we want the backend to depend on:
    the variable is first reduced by symmetry, and only then translated into
    solver variables.
    """
    (
        symmetry_type,
        sector_layouts,
        hermitian_parameter_count,
        real_parameter_count,
    ) = _cached_tau_layout_structure(
        _party_signature_from_lexorder(lexorder),
        slot_dims,
        local_symmetry_perms,
    )
    return SymmetryAdaptedTauLayout(
        name=name,
        lexorder=lexorder,
        slot_dims=slot_dims,
        matrix_dim=product_dim(slot_dims),
        symmetry_type=symmetry_type,
        sectors=sector_layouts,
        hermitian_parameter_count=hermitian_parameter_count,
        real_parameter_count=real_parameter_count,
    )


def declare_symmetry_adapted_tau_variable(
    M,
    variable,
    layout: SymmetryAdaptedTauLayout,
    Domain,
    Expr,
    Matrix,
    Hermitian: bool = True,
) -> Tuple[Tuple[TauSectorExpressionData, ...], BlockTauVariableData]:
    """Declare one tau variable directly in the symmetry-adapted block basis.

    This is the core step of the block backend. It declares only the block
    variables coming from the symmetry analysis and returns the solver-facing
    sector expressions together with a compact summary of the declaration.
    """
    sector_block_data = []
    sector_labels = []
    sector_irrep_dims = []
    sector_multiplicities = []
    sector_complex_dims = []
    for sector in layout.sectors:
        sector_labels.append(sector.label)
        sector_irrep_dims.append(sector.irrep_dim)
        sector_multiplicities.append(sector.multiplicity)
        sector_complex_dims.append(sector.irrep_dim * sector.multiplicity)
        block_name = f"{variable.name}_{sector.label}"
        if Hermitian:
            scalar_vars, block_realified = hermitian_psd_expression(
                M,
                block_name,
                sector.multiplicity,
                Domain,
                Expr,
                Matrix,
            )
            coordinate_dim = sector.multiplicity * sector.multiplicity
        else:
            scalar_vars, block_realified = real_symmetric_psd_expression(
                M,
                block_name,
                sector.multiplicity,
                Domain,
                Expr,
                Matrix,
            )
            coordinate_dim = sector.multiplicity * (sector.multiplicity + 1) // 2
        real_scalars, imag_scalars = scalar_vars
        sector_block_data.append(
            TauSectorExpressionData(
                label=sector.label,
                Hermitian=Hermitian,
                irrep_dim=sector.irrep_dim,
                multiplicity=sector.multiplicity,
                coordinate_dim=coordinate_dim,
                coordinate_vector=_matrix_coordinate_vector_expr(
                    real_scalars,
                    imag_scalars,
                    Expr,
                    Hermitian,
                ),
                realified_expr=block_realified,
            )
        )

    block_data = BlockTauVariableData(
        name=variable.name,
        Hermitian=Hermitian,
        complex_dim=layout.matrix_dim,
        realified_dim=(2 * layout.matrix_dim if Hermitian else layout.matrix_dim),
        symmetry_type=layout.symmetry_type,
        sector_labels=tuple(sector_labels),
        sector_irrep_dims=tuple(sector_irrep_dims),
        sector_multiplicities=tuple(sector_multiplicities),
        sector_complex_dims=tuple(sector_complex_dims),
        parameter_count=(
            layout.hermitian_parameter_count if Hermitian else layout.real_parameter_count
        ),
    )
    return tuple(sector_block_data), block_data


# ---------------------------------------------------------------------------
# Block Fusion model construction
# ---------------------------------------------------------------------------


def build_block_fusion_feasibility_model(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    model_name: str = "GNMEBlockStateSDP",
    include_ppt: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = False,
    Hermitian: bool = True,
    verbose: int | None = None,
):
    """Instantiate the GNME draft using symmetry-adapted block variables.
    """
    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    _warm_numba_selector_kernels()

    if isinstance(assigned, AssignedStateSDPDraft):
        model = assigned.model
    else:
        model = assigned
    # Known representatives stay as free PSD variables. The public flag is
    # retained for API compatibility, but the active backend path does not
    # inject fixed numeric matrices directly into the model.
    _ = enforce_known_values

    active_ppt_variables = model.ppt_variables
    active_ppt_constraints = model.ppt_constraints
    if include_ppt:
        active_ppt_variables, active_ppt_constraints = quotient_ppt_constraints(
            model,
            use_source_symmetry=True,
        )
    direct_ppt_constraints = tuple()
    if include_ppt and not Hermitian:
        direct_ppt_constraints = tuple(active_ppt_constraints)
        active_ppt_constraints = tuple()
        active_ppt_variables = tuple()

    real_verbose = _resolve_verbose(verbose, model.verbose)
    total_start = perf_counter()
    _progress_log(real_verbose, 1, "Building MOSEK Fusion block feasibility model...")
    M = Model(model_name)
    build_profile = {
        "tau_declaration_time": 0.0,
        "auxiliary_declaration_time": 0.0,
        "trace_time": 0.0,
        "representative_quotient_time": 0.0,
        "representative_precompile_time": 0.0,
        "representative_batch_assembly_time": 0.0,
        "representative_batch_mul_time": 0.0,
        "representative_batch_sub_time": 0.0,
        "representative_batch_triplet_time": 0.0,
        "representative_batch_matrix_time": 0.0,
        "representative_batch_sparse_time": 0.0,
        "representative_batch_rhs_stack_time": 0.0,
        "representative_constraint_emit_time": 0.0,
        "representative_direct_count": 0,
        "representative_direct_expr_time": 0.0,
        "representative_direct_emit_time": 0.0,
        "representative_unique_canonical_maps": 0,
        "representative_source_batches": 0,
        "representative_chunk_count": 0,
        "representative_batch_memory_hits": 0,
        "representative_batch_disk_hits": 0,
        "representative_batch_misses": 0,
        "ppt_batch_assembly_time": 0.0,
        "ppt_batch_mul_time": 0.0,
        "ppt_batch_sub_time": 0.0,
        "ppt_batch_triplet_time": 0.0,
        "ppt_batch_matrix_time": 0.0,
        "ppt_batch_sparse_time": 0.0,
        "ppt_batch_rhs_stack_time": 0.0,
        "ppt_constraint_emit_time": 0.0,
        "ppt_source_batches": 0,
        "ppt_chunk_count": 0,
        "ppt_emit_slice_count": 0,
        "ppt_direct_count": 0,
        "ppt_direct_mul_time": 0.0,
        "ppt_direct_svec_time": 0.0,
        "ppt_direct_emit_time": 0.0,
        "ppt_direct_tau_count": 0,
        "ppt_direct_tau_rows_full": 0,
        "ppt_direct_tau_rows_reduced": 0,
        "ppt_direct_tau_select_time": 0.0,
        "ppt_direct_tau_mul_time": 0.0,
        "ppt_direct_tau_emit_time": 0.0,
        "ppt_direct_reduced_count": 0,
        "ppt_direct_reduced_mul_time": 0.0,
        "ppt_direct_reduced_svec_time": 0.0,
        "ppt_direct_reduced_emit_time": 0.0,
        "ppt_emit_time": 0.0,
        "tau_map_memory_hits": 0,
        "tau_map_disk_hits": 0,
        "tau_map_misses": 0,
        "tau_map_compile_time": 0.0,
        "tau_map_sparse_build_time": 0.0,
        "total_build_time": 0.0,
    }

    # Step 1: full tau variables in a symmetry-adapted block basis.
    step_start = perf_counter()
    tau_variables = {}
    tau_block_variables = {}
    tau_coordinate_vectors = {}
    tau_sector_coordinate_vectors = {}
    tau_sector_coordinate_dims = {}
    block_tau_data = {}
    total_block_parameters = 0
    tau_iter = _progress_bar(
        model.psd_variables,
        real_verbose,
        "block step 1: tau sectors",
        total=len(model.psd_variables),
    )
    for index, variable in enumerate(tau_iter, start=1):
        _progress_log(
            real_verbose,
            1,
            f"Fusion block step 1/5: tau {index}/{len(model.psd_variables)} = {variable.name}",
        )
        layout = cached_tau_layout(
            variable.name,
            variable.lexorder,
            variable.slot_dims,
            variable.local_symmetry_perms,
        )
        sector_block_data, block_data = declare_symmetry_adapted_tau_variable(
            M,
            variable,
            layout,
            Domain,
            Expr,
            Matrix,
            Hermitian,
        )
        # Keep tau in sector form. The expensive full affine lifted matrix is
        # intentionally not reconstructed here; representative and PPT layers
        # read the sector variables directly through cached selectors.
        tau_variables[variable.name] = None
        tau_block_variables[variable.name] = sector_block_data
        tau_coordinate_vectors[variable.name] = _stack_coordinate_vectors(
            tuple(sector.coordinate_vector for sector in sector_block_data),
            Expr,
        )
        tau_sector_coordinate_vectors[variable.name] = tuple(
            sector.coordinate_vector for sector in sector_block_data
        )
        tau_sector_coordinate_dims[variable.name] = tuple(
            sector.coordinate_dim for sector in sector_block_data
        )
        block_tau_data[variable.name] = block_data
        total_block_parameters += block_data.parameter_count
    _progress_log(
        real_verbose,
        1,
        "Fusion block step 1/5 complete: "
        f"{len(tau_variables)} tau variables, {total_block_parameters} block parameters "
        f"in {perf_counter() - step_start:.2f}s.",
    )
    build_profile["tau_declaration_time"] = perf_counter() - step_start
    tau_variable_specs = {variable.name: variable for variable in model.psd_variables}

    # Step 2: auxiliary reduced states, known free states, and PPT auxiliaries.
    step_start = perf_counter()
    auxiliary_variables = {}
    auxiliary_coordinate_vectors = {}
    representative_coord_dims = {}
    direct_real_representatives = not Hermitian
    aux_iter = _progress_bar(
        model.auxiliary_representatives,
        real_verbose,
        "block step 2: auxiliary reps",
        total=len(model.auxiliary_representatives),
    )
    expression_builder = hermitian_psd_expression if Hermitian else real_symmetric_psd_expression
    for representative in aux_iter:
        representative_coord_dims[representative.name] = (
            representative.matrix_dim * representative.matrix_dim
            if Hermitian
            else representative.matrix_dim * (representative.matrix_dim + 1) // 2
        )
        if direct_real_representatives:
            continue
        scalar_vars, expr = expression_builder(
            M,
            representative.name,
            representative.matrix_dim,
            Domain,
            Expr,
            Matrix,
        )
        auxiliary_variables[representative.name] = expr
        auxiliary_coordinate_vectors[representative.name] = _matrix_coordinate_vector_expr(
            scalar_vars[0],
            scalar_vars[1],
            Expr,
            Hermitian,
        )
    known_representative_variables = {}
    known_representative_coordinate_vectors = {}
    known_iter = _progress_bar(
        model.known_representatives,
        real_verbose,
        "block step 2: known reps",
        total=len(model.known_representatives),
    )
    for representative in known_iter:
        representative_coord_dims[representative.name] = (
            representative.matrix_dim * representative.matrix_dim
            if Hermitian
            else representative.matrix_dim * (representative.matrix_dim + 1) // 2
        )
        if direct_real_representatives:
            continue
        scalar_vars, expr = expression_builder(
            M,
            representative.name,
            representative.matrix_dim,
            Domain,
            Expr,
            Matrix,
        )
        known_representative_variables[representative.name] = expr
        known_representative_coordinate_vectors[representative.name] = _matrix_coordinate_vector_expr(
            scalar_vars[0],
            scalar_vars[1],
            Expr,
            Hermitian,
        )
    ppt_variables = {}
    ppt_coordinate_vectors = {}
    ppt_coordinate_dims = {}
    direct_ppt_coordinate_vectors = {}
    direct_tau_ppt_row_lookups = {}
    direct_tau_ppt_sector_coordinate_vectors = {}
    direct_tau_ppt_sector_coordinate_dims = {}
    direct_tau_ppt_sector_maps = {}
    if include_ppt:
        ppt_iter = _progress_bar(
            active_ppt_variables,
            real_verbose,
            "block step 2: PPT vars",
            total=len(active_ppt_variables),
        )
        for ppt_variable in ppt_iter:
            scalar_vars, expr = expression_builder(
                M,
                ppt_variable.name,
                ppt_variable.matrix_dim,
                Domain,
                Expr,
                Matrix,
            )
            ppt_variables[ppt_variable.name] = expr
            ppt_coordinate_vectors[ppt_variable.name] = _matrix_coordinate_vector_expr(
                scalar_vars[0],
                scalar_vars[1],
                Expr,
                Hermitian,
            )
            ppt_coordinate_dims[ppt_variable.name] = int(
                ppt_coordinate_vectors[ppt_variable.name].getShape()[0]
            )
        if direct_ppt_constraints:
            direct_iter = _progress_bar(
                direct_ppt_constraints,
                real_verbose,
                "block step 2: direct PPT vars",
                total=len(direct_ppt_constraints),
            )
            for constraint in direct_iter:
                if constraint.source_variable_name in tau_variable_specs:
                    source_spec = tau_variable_specs[constraint.source_variable_name]
                    stabilizer_perms, orbit_data, row_lookup = _cached_tau_ppt_orbit_reduction(
                        source_spec.slot_dims,
                        source_spec.local_symmetry_perms,
                        constraint.transpose_positions,
                    )
                    stabilizer_layout = cached_tau_layout(
                        constraint.ppt_variable_name,
                        source_spec.lexorder,
                        source_spec.slot_dims,
                        stabilizer_perms,
                    )
                    sector_block_data, block_data = declare_symmetry_adapted_tau_variable(
                        M,
                        SimpleNamespace(name=constraint.ppt_variable_name),
                        stabilizer_layout,
                        Domain,
                        Expr,
                        Matrix,
                        Hermitian=False,
                    )
                    direct_tau_ppt_row_lookups[constraint.ppt_variable_name] = row_lookup
                    direct_tau_ppt_sector_coordinate_vectors[constraint.ppt_variable_name] = tuple(
                        sector.coordinate_vector for sector in sector_block_data
                    )
                    direct_tau_ppt_sector_coordinate_dims[constraint.ppt_variable_name] = tuple(
                        sector.coordinate_dim for sector in sector_block_data
                    )
                    _target_coord_dim, lhs_sector_maps = _cached_block_partial_trace_coordinate_maps(
                        _party_signature_from_lexorder(source_spec.lexorder),
                        source_spec.slot_dims,
                        stabilizer_perms,
                        tuple(range(len(source_spec.slot_dims))),
                        False,
                    )
                    direct_tau_ppt_sector_maps[constraint.ppt_variable_name] = (
                        _restrict_sector_triplets_to_row_lookup(lhs_sector_maps, row_lookup)
                    )
                    ppt_variables[constraint.ppt_variable_name] = block_data
                    ppt_coordinate_dims[constraint.ppt_variable_name] = len(
                        orbit_data.orbit_representatives
                    )
                    build_profile["ppt_direct_tau_rows_full"] += (
                        constraint.matrix_dim * (constraint.matrix_dim + 1) // 2
                    )
                    build_profile["ppt_direct_tau_rows_reduced"] += len(
                        orbit_data.orbit_representatives
                    )
                else:
                    coord_dim = constraint.matrix_dim * (constraint.matrix_dim + 1) // 2
                    direct_ppt_coordinate_vectors[constraint.ppt_variable_name] = M.variable(
                        f"{constraint.ppt_variable_name}_svec",
                        Domain.inSVecPSDCone(coord_dim),
                    )
                    ppt_coordinate_dims[constraint.ppt_variable_name] = coord_dim
    _progress_log(
        real_verbose,
        1,
        "Fusion block step 2/5 complete: "
        f"{len(auxiliary_variables)} auxiliary vars, {len(known_representative_variables)} known vars, "
        f"{len(model.auxiliary_representatives) + len(model.known_representatives) if direct_real_representatives else 0} direct representatives, "
        f"{len(ppt_variables)} PPT variables, {len(direct_ppt_constraints)} direct real PPT cones "
        f"in {perf_counter() - step_start:.2f}s.",
    )
    constraint_counts = {
        "trace": 0,
        "internal_symmetry": 0,
        "representative": 0,
        "ppt": 0,
        "ppt_direct": 0,
        "block_parameters": total_block_parameters,
    }
    build_profile["auxiliary_declaration_time"] = perf_counter() - step_start
    compiled_linear_maps = {}
    compiled_triplet_maps = {}
    compiled_sector_triplet_maps = {}

    def _source_layout_signature(source_spec) -> Tuple[object, ...]:
        return (
            *_layout_signature(
                source_spec.lexorder,
                source_spec.slot_dims,
                source_spec.local_symmetry_perms,
            ),
            tuple(int(dim) for dim in tau_sector_coordinate_dims[source_spec.name]),
        )

    def _canonicalize_source_positions(source_spec, positions, map_kind: str):
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

    def _compile_canonical_tau_triplets(source_spec, canonical_positions, map_kind: str):
        cache_key = (
            "combined",
            map_kind,
            _source_layout_signature(source_spec),
            canonical_positions,
            Hermitian,
        )
        compiled = compiled_triplet_maps.get(cache_key)
        if compiled is None:
            persistent = _load_persistent_triplet_map(cache_key)
            if persistent is not None:
                build_profile["tau_map_disk_hits"] += 1
                compiled = persistent
            else:
                compile_start = perf_counter()
                if map_kind == "partial_trace":
                    target_coord_dim, sector_maps = cached_block_partial_trace_coordinate_maps(
                        source_spec.lexorder,
                        source_spec.slot_dims,
                        source_spec.local_symmetry_perms,
                        canonical_positions,
                        Hermitian,
                    )
                elif map_kind == "partial_transpose":
                    target_coord_dim, sector_maps = cached_block_partial_transpose_coordinate_maps(
                        source_spec.lexorder,
                        source_spec.slot_dims,
                        source_spec.local_symmetry_perms,
                        canonical_positions,
                        Hermitian,
                    )
                else:
                    raise ValueError(f"Unsupported cached map kind {map_kind!r}.")
                rows, cols, vals = _combine_sector_triplets(
                    target_coord_dim,
                    sector_maps,
                    tau_sector_coordinate_dims[source_spec.name],
                )
                build_profile["tau_map_compile_time"] += perf_counter() - compile_start
                build_profile["tau_map_misses"] += 1
                compiled = (target_coord_dim, rows, cols, vals)
                _save_persistent_triplet_map(cache_key, target_coord_dim, rows, cols, vals)
            compiled_triplet_maps[cache_key] = compiled
        else:
            build_profile["tau_map_memory_hits"] += 1
        return compiled

    def _compile_canonical_tau_sector_triplets(source_spec, canonical_positions, map_kind: str):
        """Compile one canonical reduced map while keeping sectors separate."""
        cache_key = (
            "sectorwise",
            map_kind,
            _source_layout_signature(source_spec),
            canonical_positions,
            Hermitian,
        )
        compiled = compiled_sector_triplet_maps.get(cache_key)
        if compiled is not None:
            build_profile["tau_map_memory_hits"] += 1
            return compiled

        persistent_payload = _load_persistent_object(cache_key)
        if persistent_payload is not None:
            try:
                target_coord_dim, sector_payload = persistent_payload
                compiled = (
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
                compiled_sector_triplet_maps[cache_key] = compiled
                build_profile["tau_map_disk_hits"] += 1
                return compiled
            except Exception:
                pass

        compile_start = perf_counter()
        if map_kind == "partial_trace":
            target_coord_dim, sector_maps = cached_block_partial_trace_coordinate_maps(
                source_spec.lexorder,
                source_spec.slot_dims,
                source_spec.local_symmetry_perms,
                canonical_positions,
                Hermitian,
            )
        elif map_kind == "partial_transpose":
            target_coord_dim, sector_maps = cached_block_partial_transpose_coordinate_maps(
                source_spec.lexorder,
                source_spec.slot_dims,
                source_spec.local_symmetry_perms,
                canonical_positions,
                Hermitian,
            )
        else:
            raise ValueError(f"Unsupported cached map kind {map_kind!r}.")

        build_profile["tau_map_compile_time"] += perf_counter() - compile_start
        build_profile["tau_map_misses"] += 1
        compiled = (
            int(target_coord_dim),
            tuple(
                (
                    rows.astype(np.int32, copy=False)[order],
                    cols.astype(np.int32, copy=False)[order],
                    vals.astype(np.float64, copy=False)[order],
                )
                if rows.size
                else (
                    rows.astype(np.int32, copy=False),
                    cols.astype(np.int32, copy=False),
                    vals.astype(np.float64, copy=False),
                )
                for rows, cols, vals in sector_maps
                for order in (
                    np.argsort(rows.astype(np.int32, copy=False), kind="mergesort"),
                )
            ),
        )
        compiled_sector_triplet_maps[cache_key] = compiled
        _save_persistent_object(cache_key, compiled)
        return compiled

    def get_cached_tau_linear_map_triplets(source_spec, positions, map_kind: str):
        canonical_positions, transport_rows, transport_signs = _canonicalize_source_positions(
            source_spec,
            positions,
            map_kind,
        )
        target_coord_dim, canonical_rows, canonical_cols, canonical_vals = _compile_canonical_tau_triplets(
            source_spec,
            canonical_positions,
            map_kind,
        )
        if transport_rows is None:
            return target_coord_dim, canonical_rows, canonical_cols, canonical_vals

        transport_key = (
            "transported",
            map_kind,
            _source_layout_signature(source_spec),
            canonical_positions,
            positions,
            Hermitian,
        )
        transported = compiled_triplet_maps.get(transport_key)
        if transported is not None:
            build_profile["tau_map_memory_hits"] += 1
            return transported

        moved_rows = transport_rows[canonical_rows].astype(np.int32, copy=False)
        moved_vals = (transport_signs[canonical_rows] * canonical_vals).astype(np.float64, copy=False)
        transported = (target_coord_dim, moved_rows, canonical_cols, moved_vals)
        compiled_triplet_maps[transport_key] = transported
        return transported

    def get_cached_tau_sector_linear_map_triplets(source_spec, positions, map_kind: str):
        """Return reduced tau maps without flattening sectors together."""
        canonical_positions, transport_rows, transport_signs = _canonicalize_source_positions(
            source_spec,
            positions,
            map_kind,
        )
        target_coord_dim, canonical_sector_maps = _compile_canonical_tau_sector_triplets(
            source_spec,
            canonical_positions,
            map_kind,
        )
        if transport_rows is None:
            return target_coord_dim, canonical_sector_maps

        transport_key = (
            "sectorwise_transported",
            map_kind,
            _source_layout_signature(source_spec),
            canonical_positions,
            positions,
            Hermitian,
        )
        transported = compiled_sector_triplet_maps.get(transport_key)
        if transported is not None:
            build_profile["tau_map_memory_hits"] += 1
            return transported

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
        transported = (target_coord_dim, tuple(transported_sector_maps))
        compiled_sector_triplet_maps[transport_key] = transported
        return transported

    def get_cached_tau_linear_map_nnz(source_spec, positions, map_kind: str) -> int:
        """Return the triplet count of a reduced tau linear map.

        The transport action used to move a canonical map to a requested source
        subset only permutes/sign-flips rows, so the nnz count is inherited
        from the canonical map. This makes nnz-aware chunking cheap after the
        canonical maps have been compiled.
        """
        canonical_positions, _transport_rows, _transport_signs = _canonicalize_source_positions(
            source_spec,
            positions,
            map_kind,
        )
        _target_coord_dim, canonical_rows, _canonical_cols, _canonical_vals = _compile_canonical_tau_triplets(
            source_spec,
            canonical_positions,
            map_kind,
        )
        return int(canonical_rows.size)

    def get_cached_tau_sector_linear_map_nnz(source_spec, positions, map_kind: str) -> int:
        """Triplet count of a reduced tau map with sectors kept separate."""
        _target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
            source_spec,
            positions,
            map_kind,
        )
        return int(sum(rows.size for rows, _cols, _vals in sector_maps))

    def get_cached_tau_linear_map(source_spec, positions, map_kind: str):
        sparse_key = (
            "matrix",
            _source_layout_signature(source_spec),
            map_kind,
            tuple(positions),
            Hermitian,
        )
        compiled = compiled_linear_maps.get(sparse_key)
        if compiled is not None:
            build_profile["tau_map_memory_hits"] += 1
            return compiled
        target_coord_dim, rows, cols, vals = get_cached_tau_linear_map_triplets(
            source_spec,
            positions,
            map_kind,
        )
        sparse_start = perf_counter()
        linear_map = Matrix.sparse(
            target_coord_dim,
            int(sum(tau_sector_coordinate_dims[source_spec.name])),
            rows,
            cols,
            vals,
        )
        build_profile["tau_map_sparse_build_time"] += perf_counter() - sparse_start
        compiled = (target_coord_dim, linear_map)
        compiled_linear_maps[sparse_key] = compiled
        return compiled

    def get_cached_free_partial_transpose_map(slot_dims, transpose_positions, coord_dim: int):
        cache_key = (
            "free_partial_transpose",
            tuple(slot_dims),
            tuple(transpose_positions),
            Hermitian,
            coord_dim,
        )
        compiled = compiled_linear_maps.get(cache_key)
        if compiled is not None:
            return compiled
        target_coord_dim, rows, cols, vals, *_ = cached_partial_transpose_coordinate_action(
            tuple(slot_dims),
            tuple(transpose_positions),
            Hermitian,
        )
        linear_map = Matrix.sparse(target_coord_dim, coord_dim, rows, cols, vals)
        compiled_linear_maps[cache_key] = (target_coord_dim, linear_map)
        return compiled_linear_maps[cache_key]

    def get_cached_representative_batch_triplets(source_spec, constraints_for_source):
        """Return one stacked sparse operator for a chunk of representative links.

        Each representative equality contributes a reduced partial-trace map
        from one source ``tau`` coordinate vector into one target coordinate
        block. The hot path in level-3 was repeatedly concatenating those
        blocks, so we cache the fully stacked triplets for each source/chunk
        signature and reuse them across runs.
        """
        compile_start = perf_counter()
        triplet_blocks = []
        target_dims = []
        row_offset = 0
        total_nnz = 0
        for constraint in constraints_for_source:
            target_coord_dim, rows, cols, vals = get_cached_tau_linear_map_triplets(
                source_spec,
                constraint.keep_positions,
                "partial_trace",
            )
            triplet_blocks.append(
                (
                    int(row_offset),
                    rows.astype(np.int32, copy=False),
                    cols.astype(np.int32, copy=False),
                    vals.astype(np.float64, copy=False),
                )
            )
            target_dims.append(int(target_coord_dim))
            total_nnz += int(rows.size)
            row_offset += target_coord_dim

        batch_rows = np.empty(total_nnz, dtype=np.int32)
        batch_cols = np.empty(total_nnz, dtype=np.int32)
        batch_vals = np.empty(total_nnz, dtype=np.float64)
        cursor = 0
        for block_row_offset, rows, cols, vals in triplet_blocks:
            nnz = int(rows.size)
            next_cursor = cursor + nnz
            batch_rows[cursor:next_cursor] = rows + block_row_offset
            batch_cols[cursor:next_cursor] = cols
            batch_vals[cursor:next_cursor] = vals
            cursor = next_cursor

        cached_payload = (
            int(row_offset),
            batch_rows,
            batch_cols,
            batch_vals,
            tuple(target_dims),
        )
        build_profile["representative_batch_misses"] += 1
        build_profile["representative_batch_triplet_time"] += perf_counter() - compile_start
        return cached_payload

    def get_cached_representative_sector_batch_triplets(source_spec, constraints_for_source):
        """Return one stacked sparse operator per tau sector for a chunk."""
        keep_positions_sequence = tuple(
            tuple(constraint.keep_positions) for constraint in constraints_for_source
        )
        cache_key = (
            "rep_sector_batch_triplets",
            _source_layout_signature(source_spec),
            keep_positions_sequence,
            Hermitian,
        )
        compile_start = perf_counter()
        if len(constraints_for_source) == 1:
            constraint = constraints_for_source[0]
            target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                source_spec,
                constraint.keep_positions,
                "partial_trace",
            )
            cached_payload = (
                int(target_coord_dim),
                tuple(
                    (
                        rows.astype(np.int32, copy=False),
                        cols.astype(np.int32, copy=False),
                        vals.astype(np.float64, copy=False),
                    )
                    for rows, cols, vals in sector_maps
                ),
                (int(target_coord_dim),),
            )
            build_profile["representative_batch_misses"] += 1
            build_profile["representative_batch_triplet_time"] += perf_counter() - compile_start
            return cached_payload
        sector_triplet_blocks = [[] for _ in tau_sector_coordinate_dims[source_spec.name]]
        sector_total_nnz = [0 for _ in tau_sector_coordinate_dims[source_spec.name]]
        target_dims = []
        row_offset = 0
        for constraint in constraints_for_source:
            target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                source_spec,
                constraint.keep_positions,
                "partial_trace",
            )
            for sector_index, (rows, cols, vals) in enumerate(sector_maps):
                sector_triplet_blocks[sector_index].append(
                    (
                        int(row_offset),
                        rows.astype(np.int32, copy=False),
                        cols.astype(np.int32, copy=False),
                        vals.astype(np.float64, copy=False),
                    )
                )
                sector_total_nnz[sector_index] += int(rows.size)
            target_dims.append(int(target_coord_dim))
            row_offset += target_coord_dim

        sector_payload = []
        for triplet_blocks, total_nnz in zip(sector_triplet_blocks, sector_total_nnz):
            batch_rows = np.empty(total_nnz, dtype=np.int32)
            batch_cols = np.empty(total_nnz, dtype=np.int32)
            batch_vals = np.empty(total_nnz, dtype=np.float64)
            cursor = 0
            for block_row_offset, rows, cols, vals in triplet_blocks:
                nnz = int(rows.size)
                next_cursor = cursor + nnz
                batch_rows[cursor:next_cursor] = rows + block_row_offset
                batch_cols[cursor:next_cursor] = cols
                batch_vals[cursor:next_cursor] = vals
                cursor = next_cursor
            sector_payload.append((batch_rows, batch_cols, batch_vals))

        cached_payload = (
            int(row_offset),
            tuple(sector_payload),
            tuple(target_dims),
        )
        build_profile["representative_batch_misses"] += 1
        build_profile["representative_batch_triplet_time"] += perf_counter() - compile_start
        return cached_payload

    def get_cached_ppt_sector_batch_triplets(source_spec, constraints_for_source):
        """Return one stacked partial-transpose operator per tau sector for a chunk."""
        transpose_positions_sequence = tuple(
            tuple(constraint.transpose_positions) for constraint in constraints_for_source
        )
        cache_key = (
            "ppt_sector_batch_triplets",
            _source_layout_signature(source_spec),
            transpose_positions_sequence,
            Hermitian,
        )
        compile_start = perf_counter()
        if len(constraints_for_source) == 1:
            constraint = constraints_for_source[0]
            target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                source_spec,
                constraint.transpose_positions,
                "partial_transpose",
            )
            cached_payload = (
                int(target_coord_dim),
                tuple(
                    (
                        rows.astype(np.int32, copy=False),
                        cols.astype(np.int32, copy=False),
                        vals.astype(np.float64, copy=False),
                    )
                    for rows, cols, vals in sector_maps
                ),
                (int(target_coord_dim),),
            )
            build_profile["ppt_batch_triplet_time"] += perf_counter() - compile_start
            return cached_payload
        sector_triplet_blocks = [[] for _ in tau_sector_coordinate_dims[source_spec.name]]
        sector_total_nnz = [0 for _ in tau_sector_coordinate_dims[source_spec.name]]
        target_dims = []
        row_offset = 0
        for constraint in constraints_for_source:
            target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                source_spec,
                constraint.transpose_positions,
                "partial_transpose",
            )
            for sector_index, (rows, cols, vals) in enumerate(sector_maps):
                sector_triplet_blocks[sector_index].append(
                    (
                        int(row_offset),
                        rows.astype(np.int32, copy=False),
                        cols.astype(np.int32, copy=False),
                        vals.astype(np.float64, copy=False),
                    )
                )
                sector_total_nnz[sector_index] += int(rows.size)
            target_dims.append(int(target_coord_dim))
            row_offset += target_coord_dim

        sector_payload = []
        for triplet_blocks, total_nnz in zip(sector_triplet_blocks, sector_total_nnz):
            batch_rows = np.empty(total_nnz, dtype=np.int32)
            batch_cols = np.empty(total_nnz, dtype=np.int32)
            batch_vals = np.empty(total_nnz, dtype=np.float64)
            cursor = 0
            for block_row_offset, rows, cols, vals in triplet_blocks:
                nnz = int(rows.size)
                next_cursor = cursor + nnz
                batch_rows[cursor:next_cursor] = rows + block_row_offset
                batch_cols[cursor:next_cursor] = cols
                batch_vals[cursor:next_cursor] = vals
                cursor = next_cursor
            sector_payload.append((batch_rows, batch_cols, batch_vals))

        cached_payload = (
            int(row_offset),
            tuple(sector_payload),
            tuple(target_dims),
        )
        build_profile["ppt_batch_triplet_time"] += perf_counter() - compile_start
        return cached_payload

    # Step 3: trace-one constraints.
    step_start = perf_counter()
    for variable in model.psd_variables:
        trace_terms = []
        for sector in tau_block_variables[variable.name]:
            if Hermitian:
                sector_trace = _hermitian_trace_expr(
                    sector.realified_expr,
                    sector.multiplicity,
                    Expr,
                )
            else:
                sector_trace = _real_trace_expr(
                    sector.realified_expr,
                    sector.multiplicity,
                    Expr,
                )
            trace_terms.extend([sector_trace] * sector.irrep_dim)
        M.constraint(
            f"trace_{variable.name}",
            _sum_expr(trace_terms, Expr),
            Domain.equalsTo(1.0),
        )
        constraint_counts["trace"] += 1
    for representative in model.auxiliary_representatives:
        if representative.name not in auxiliary_variables:
            continue
        M.constraint(
            f"trace_{representative.name}",
            _hermitian_trace_expr(
                auxiliary_variables[representative.name],
                representative.matrix_dim,
                Expr,
            ) if Hermitian else _real_trace_expr(
                auxiliary_variables[representative.name],
                representative.matrix_dim,
                Expr,
            ),
            Domain.equalsTo(1.0),
        )
        constraint_counts["trace"] += 1
    for representative in model.known_representatives:
        if representative.name not in known_representative_variables:
            continue
        M.constraint(
            f"trace_{representative.name}",
            _hermitian_trace_expr(
                known_representative_variables[representative.name],
                representative.matrix_dim,
                Expr,
            ) if Hermitian else _real_trace_expr(
                known_representative_variables[representative.name],
                representative.matrix_dim,
                Expr,
            ),
            Domain.equalsTo(1.0),
        )
        constraint_counts["trace"] += 1
    _progress_log(
        real_verbose,
        1,
        "Fusion block step 3/5 complete: "
        f"{constraint_counts['trace']} trace constraints in {perf_counter() - step_start:.2f}s.",
    )
    build_profile["trace_time"] = perf_counter() - step_start

    # Step 4: representative equalities via Hermitian partial traces.
    if include_representatives:
        step_start = perf_counter()
        quotient_start = perf_counter()
        representative_constraints = quotient_representative_constraints(
            model,
            use_source_symmetry=True,
        )
        build_profile["representative_quotient_time"] = perf_counter() - quotient_start
        _progress_log(
            real_verbose,
            1,
            f"Fusion block step 4/5: adding {len(representative_constraints)} representative blocks...",
        )
        tau_variable_specs = {variable.name: variable for variable in model.psd_variables}
        if direct_real_representatives:
            representative_lookup = {
                representative.name: representative
                for representative in (*model.auxiliary_representatives, *model.known_representatives)
            }
            grouped_representatives = defaultdict(list)
            for constraint in representative_constraints:
                grouped_representatives[constraint.representative_name].append(constraint)
            build_profile["representative_source_batches"] = len(grouped_representatives)
            batch_start = perf_counter()
            reduced_expr_cache = {}

            def get_tau_reduced_coordinate_expr(source_spec, keep_positions):
                cache_key = (source_spec.name, tuple(keep_positions))
                cached = reduced_expr_cache.get(cache_key)
                if cached is not None:
                    return cached
                expr_start = perf_counter()
                target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                    source_spec,
                    keep_positions,
                    "partial_trace",
                )
                terms = []
                for sector_index, ((rows, cols, vals), sector_dim) in enumerate(
                    zip(sector_maps, tau_sector_coordinate_dims[source_spec.name])
                ):
                    if vals.size == 0:
                        continue
                    sector_map = Matrix.sparse(int(target_coord_dim), int(sector_dim), rows, cols, vals)
                    terms.append(
                        Expr.mul(
                            sector_map,
                            tau_sector_coordinate_vectors[source_spec.name][sector_index],
                        )
                    )
                coord_expr = _sum_expr(terms, Expr)
                build_profile["representative_direct_expr_time"] += perf_counter() - expr_start
                cached = (int(target_coord_dim), coord_expr)
                reduced_expr_cache[cache_key] = cached
                return cached

            rep_items = list(grouped_representatives.items())
            rep_iter = _progress_bar(
                rep_items,
                real_verbose,
                "block step 4: direct representatives",
                total=len(rep_items),
            )
            for rep_index, (representative_name, constraints_for_rep) in enumerate(rep_iter, start=1):
                if real_verbose >= 2:
                    _progress_log(
                        real_verbose,
                        2,
                        f"  direct representative {rep_index}/{len(rep_items)} = {representative_name} "
                        f"with {len(constraints_for_rep)} occurrences",
                    )
                representative = representative_lookup[representative_name]
                anchor_constraint = constraints_for_rep[0]
                anchor_source_spec = tau_variable_specs[anchor_constraint.source_variable_name]
                anchor_coord_dim, anchor_coord = get_tau_reduced_coordinate_expr(
                    anchor_source_spec,
                    anchor_constraint.keep_positions,
                )
                expected_coord_dim = representative_coord_dims[representative_name]
                if int(anchor_coord_dim) != int(expected_coord_dim):
                    raise ValueError(
                        f"Representative {representative_name} expected coordinate dimension "
                        f"{expected_coord_dim}, got {anchor_coord_dim}."
                    )
                anchor_expr = real_symmetric_matrix_from_coordinate_expr(
                    anchor_coord,
                    representative.matrix_dim,
                    Expr,
                    Matrix,
                )
                if anchor_constraint.representative_kind == "known_matrix":
                    known_representative_coordinate_vectors[representative_name] = anchor_coord
                    known_representative_variables[representative_name] = anchor_expr
                else:
                    auxiliary_coordinate_vectors[representative_name] = anchor_coord
                    auxiliary_variables[representative_name] = anchor_expr
                build_profile["representative_direct_count"] += 1

                for occurrence_index, constraint in enumerate(constraints_for_rep[1:], start=1):
                    source_spec = tau_variable_specs[constraint.source_variable_name]
                    current_coord_dim, current_coord = get_tau_reduced_coordinate_expr(
                        source_spec,
                        constraint.keep_positions,
                    )
                    if int(current_coord_dim) != int(anchor_coord_dim):
                        raise ValueError(
                            f"Representative {representative_name} coordinate mismatch: "
                            f"{current_coord_dim} vs {anchor_coord_dim}."
                        )
                    sub_start = perf_counter()
                    difference = Expr.sub(current_coord, anchor_coord)
                    build_profile["representative_batch_sub_time"] += perf_counter() - sub_start
                    emit_start = perf_counter()
                    M.constraint(
                        f"rep_direct_{rep_index}_{occurrence_index}_{representative_name}",
                        difference,
                        Domain.equalsTo(0.0),
                    )
                    build_profile["representative_direct_emit_time"] += perf_counter() - emit_start
                    constraint_counts["representative"] += int(anchor_coord_dim)
            build_profile["representative_unique_canonical_maps"] = len(reduced_expr_cache)
            build_profile["representative_batch_assembly_time"] = perf_counter() - batch_start
        else:
            grouped_representatives = defaultdict(list)
            unique_canonical_maps = {}
            for constraint in representative_constraints:
                source_spec = tau_variable_specs[constraint.source_variable_name]
                grouped_representatives[constraint.source_variable_name].append(constraint)
                canonical_positions, _transport_rows, _transport_signs = _canonicalize_source_positions(
                    source_spec,
                    constraint.keep_positions,
                    "partial_trace",
                )
                unique_canonical_maps[
                    (
                        _source_layout_signature(source_spec),
                        canonical_positions,
                    )
                ] = source_spec

            build_profile["representative_unique_canonical_maps"] = len(unique_canonical_maps)
            build_profile["representative_source_batches"] = len(grouped_representatives)

            precompile_start = perf_counter()
            if len(unique_canonical_maps) <= _GLOBAL_PRECOMPILE_CANONICAL_MAP_LIMIT:
                precompile_iter = _progress_bar(
                    list(unique_canonical_maps.items()),
                    real_verbose,
                    "block step 4a: compile canonical maps",
                    total=len(unique_canonical_maps),
                )
                for (
                    _source_signature,
                    canonical_positions,
                ), source_spec in precompile_iter:
                    _compile_canonical_tau_sector_triplets(source_spec, canonical_positions, "partial_trace")
            build_profile["representative_precompile_time"] = perf_counter() - precompile_start

            batch_start = perf_counter()
            group_items = list(grouped_representatives.items())
            rep_iter = _progress_bar(
                group_items,
                real_verbose,
                "block step 4b: representative batches",
                total=len(group_items),
            )
            for batch_index, (source_name, constraints_for_source) in enumerate(rep_iter, start=1):
                if real_verbose >= 2:
                    _progress_log(
                        real_verbose,
                        2,
                            f"  representative batch {batch_index}/{len(group_items)} for {source_name} "
                            f"with {len(constraints_for_source)} blocks",
                    )
                source_spec = tau_variable_specs[source_name]
                constraint_chunks = _chunk_representative_constraints(
                    constraints_for_source,
                    representative_coord_dims,
                    constraint_nnz_fn=lambda constraint, _source_spec=source_spec: get_cached_tau_sector_linear_map_nnz(
                        _source_spec,
                        constraint.keep_positions,
                        "partial_trace",
                    ),
                )
                build_profile["representative_chunk_count"] += len(constraint_chunks)
                for chunk_index, constraint_chunk in enumerate(constraint_chunks, start=1):
                    compile_start = perf_counter()
                    row_offset, sector_triplets, target_dims = get_cached_representative_sector_batch_triplets(
                        source_spec,
                        constraint_chunk,
                    )
                    build_profile["representative_batch_triplet_time"] += perf_counter() - compile_start

                    rhs_blocks = []
                    for target_coord_dim, constraint in zip(target_dims, constraint_chunk):
                        constraint_counts["representative"] += int(target_coord_dim)
                        if constraint.representative_kind == "known_matrix":
                            rhs_blocks.append(
                                known_representative_coordinate_vectors[constraint.representative_name]
                            )
                        else:
                            rhs_blocks.append(auxiliary_coordinate_vectors[constraint.representative_name])

                    rhs_start = perf_counter()
                    rhs_coord = _stack_rhs_blocks(rhs_blocks, Expr, Matrix)
                    build_profile["representative_batch_rhs_stack_time"] += perf_counter() - rhs_start

                    lhs_terms = []
                    for sector_index, ((rows, cols, vals), sector_dim) in enumerate(
                        zip(sector_triplets, tau_sector_coordinate_dims[source_name])
                    ):
                        if vals.size == 0:
                            continue
                        matrix_start = perf_counter()
                        batch_map = Matrix.sparse(int(row_offset), int(sector_dim), rows, cols, vals)
                        build_profile["representative_batch_matrix_time"] += perf_counter() - matrix_start
                        mul_start = perf_counter()
                        lhs_terms.append(
                            Expr.mul(batch_map, tau_sector_coordinate_vectors[source_name][sector_index])
                        )
                        build_profile["representative_batch_mul_time"] += perf_counter() - mul_start

                    sub_start = perf_counter()
                    difference = Expr.sub(_sum_expr(lhs_terms, Expr), rhs_coord)
                    build_profile["representative_batch_sub_time"] += perf_counter() - sub_start

                    emit_start = perf_counter()
                    M.constraint(
                        f"rep_batch_{batch_index}_{chunk_index}_{source_name}",
                        difference,
                        Domain.equalsTo(0.0),
                    )
                    build_profile["representative_constraint_emit_time"] += perf_counter() - emit_start
                # Bound memory growth on large builds. Structural reuse still comes
                # from the persistent canonical-map cache.
                compiled_triplet_maps.clear()
                compiled_sector_triplet_maps.clear()
            build_profile["representative_batch_assembly_time"] = perf_counter() - batch_start
        _progress_log(
            real_verbose,
            1,
            "Fusion block step 4/5 complete: "
            f"{constraint_counts['representative']} scalar representative constraints "
            f"in {perf_counter() - step_start:.2f}s.",
        )

    # Step 5: PPT constraints on Hermitian variables.
    if include_ppt:
        step_start = perf_counter()
        _progress_log(
            real_verbose,
            1,
            f"Fusion block step 5/5: adding "
            f"{len(active_ppt_constraints) + len(direct_ppt_constraints)} PPT blocks...",
        )
        tau_variable_specs = {variable.name: variable for variable in model.psd_variables}
        source_free_variable_lookup = {}
        source_free_variable_lookup.update(auxiliary_coordinate_vectors)
        source_free_variable_lookup.update(known_representative_coordinate_vectors)
        if direct_ppt_constraints:
            direct_stride = _progress_stride(len(direct_ppt_constraints))
            for direct_index, constraint in enumerate(direct_ppt_constraints, start=1):
                if real_verbose >= 2 and (
                    direct_index == 1
                    or direct_index % direct_stride == 0
                    or direct_index == len(direct_ppt_constraints)
                ):
                    _progress_log(
                        real_verbose,
                        2,
                        f"  direct real PPT {direct_index}/{len(direct_ppt_constraints)} "
                        f"for {constraint.source_variable_name}",
                    )
                coord_dim = constraint.matrix_dim * (constraint.matrix_dim + 1) // 2
                if constraint.source_variable_name in tau_sector_coordinate_vectors:
                    source_spec = tau_variable_specs[constraint.source_variable_name]
                    target_coord_dim, sector_maps = get_cached_tau_sector_linear_map_triplets(
                        source_spec,
                        constraint.transpose_positions,
                        "partial_transpose",
                    )
                    if int(target_coord_dim) != coord_dim:
                        raise ValueError(
                            "Direct tau PPT expects real symmetric coordinates "
                            f"of size {coord_dim}, got {target_coord_dim}."
                        )
                    select_start = perf_counter()
                    restricted_sector_maps = _restrict_sector_triplets_to_row_lookup(
                        sector_maps,
                        direct_tau_ppt_row_lookups[constraint.ppt_variable_name],
                    )
                    build_profile["ppt_direct_tau_select_time"] += perf_counter() - select_start

                    orbit_coord_dim = int(ppt_coordinate_dims[constraint.ppt_variable_name])
                    emit_row_ranges = _iter_emit_row_ranges(orbit_coord_dim, _PPT_MAX_ROWS_PER_EMIT)
                    build_profile["ppt_emit_slice_count"] += len(emit_row_ranges)
                    for emit_index, (row_start, row_stop) in enumerate(emit_row_ranges, start=1):
                        lhs_sector_triplets = _slice_sector_triplets(
                            direct_tau_ppt_sector_maps[constraint.ppt_variable_name],
                            row_start,
                            row_stop,
                        )
                        lhs_terms = []
                        for (rows, cols, vals), sector_dim, sector_coord in zip(
                            lhs_sector_triplets,
                            direct_tau_ppt_sector_coordinate_dims[constraint.ppt_variable_name],
                            direct_tau_ppt_sector_coordinate_vectors[constraint.ppt_variable_name],
                        ):
                            if vals.size == 0:
                                continue
                            lhs_map = Matrix.sparse(
                                int(row_stop - row_start),
                                int(sector_dim),
                                rows,
                                cols,
                                vals,
                            )
                            lhs_terms.append(Expr.mul(lhs_map, sector_coord))
                        lhs_coord = _sum_expr(lhs_terms, Expr)
                        rhs_terms = []
                        sliced_sector_maps = _slice_sector_triplets(
                            restricted_sector_maps,
                            row_start,
                            row_stop,
                        )
                        for sector_index, ((rows, cols, vals), sector_dim) in enumerate(
                            zip(sliced_sector_maps, tau_sector_coordinate_dims[constraint.source_variable_name])
                        ):
                            if vals.size == 0:
                                continue
                            mul_start = perf_counter()
                            sector_map = Matrix.sparse(
                                int(row_stop - row_start),
                                int(sector_dim),
                                rows,
                                cols,
                                vals,
                            )
                            rhs_terms.append(
                                Expr.mul(
                                    sector_map,
                                    tau_sector_coordinate_vectors[constraint.source_variable_name][sector_index],
                                )
                            )
                            build_profile["ppt_direct_mul_time"] += perf_counter() - mul_start
                            build_profile["ppt_direct_tau_mul_time"] += perf_counter() - mul_start

                        emit_start = perf_counter()
                        M.constraint(
                            f"ppt_direct_{direct_index}_{emit_index}_{constraint.source_variable_name}",
                            Expr.sub(lhs_coord, _sum_expr(rhs_terms, Expr)),
                            Domain.equalsTo(0.0),
                        )
                        build_profile["ppt_direct_emit_time"] += perf_counter() - emit_start
                        build_profile["ppt_direct_tau_emit_time"] += perf_counter() - emit_start
                    build_profile["ppt_direct_count"] += 1
                    build_profile["ppt_direct_tau_count"] += 1
                    constraint_counts["ppt_direct"] += 1
                    continue
                else:
                    source_coord = source_free_variable_lookup[constraint.source_variable_name]
                    target_coord_dim, linear_map = get_cached_free_partial_transpose_map(
                        constraint.slot_dims,
                        constraint.transpose_positions,
                        int(source_coord.getShape()[0]),
                    )
                    if int(target_coord_dim) != coord_dim:
                        raise ValueError(
                            "Direct reduced PPT expects real symmetric coordinates "
                            f"of size {coord_dim}, got {target_coord_dim}."
                        )
                    mul_start = perf_counter()
                    ppt_coordinate_expr = Expr.mul(linear_map, source_coord)
                    build_profile["ppt_direct_mul_time"] += perf_counter() - mul_start
                    build_profile["ppt_direct_reduced_mul_time"] += perf_counter() - mul_start
                    svec_start = perf_counter()
                    ppt_svec_expr = real_svec_expr_from_coordinate_expr(
                        ppt_coordinate_expr,
                        constraint.matrix_dim,
                        Expr,
                        Matrix,
                    )
                    build_profile["ppt_direct_svec_time"] += perf_counter() - svec_start
                    build_profile["ppt_direct_reduced_svec_time"] += perf_counter() - svec_start

                emit_start = perf_counter()
                M.constraint(
                    f"ppt_direct_{direct_index}_{constraint.source_variable_name}",
                    Expr.sub(
                        direct_ppt_coordinate_vectors[constraint.ppt_variable_name],
                        ppt_svec_expr,
                    ),
                    Domain.equalsTo(0.0),
                )
                build_profile["ppt_direct_emit_time"] += perf_counter() - emit_start
                build_profile["ppt_direct_reduced_emit_time"] += perf_counter() - emit_start
                build_profile["ppt_direct_count"] += 1
                build_profile["ppt_direct_reduced_count"] += 1
                constraint_counts["ppt_direct"] += 1
        grouped_ppt_constraints = defaultdict(list)
        for constraint in active_ppt_constraints:
            grouped_ppt_constraints[constraint.source_variable_name].append(constraint)
        build_profile["ppt_source_batches"] = (
            len(grouped_ppt_constraints) + len({c.source_variable_name for c in direct_ppt_constraints})
        )

        ppt_group_items = list(grouped_ppt_constraints.items())
        ppt_iter = _progress_bar(
            ppt_group_items,
            real_verbose,
            "block step 5: PPT constraints",
            total=len(ppt_group_items),
        )
        for batch_index, (source_name, constraints_for_source) in enumerate(ppt_iter, start=1):
            source_spec = tau_variable_specs.get(source_name)
            source_coord = source_free_variable_lookup.get(source_name)
            if real_verbose >= 2:
                _progress_log(
                    real_verbose,
                    2,
                    f"  PPT batch {batch_index}/{len(ppt_group_items)} for {source_name} "
                    f"with {len(constraints_for_source)} blocks",
                )

            if source_spec is not None and source_name in tau_sector_coordinate_vectors:
                constraint_chunks = _chunk_constraint_blocks(
                    constraints_for_source,
                    block_rows_fn=lambda constraint: ppt_coordinate_dims[constraint.ppt_variable_name],
                    max_blocks_per_chunk=_PPT_MAX_BLOCKS_PER_CHUNK,
                    max_rows_per_chunk=_PPT_MAX_ROWS_PER_CHUNK,
                    max_nnz_per_chunk=_PPT_MAX_NNZ_PER_CHUNK,
                    max_dense_bytes_per_chunk=_PPT_MAX_DENSE_BYTES_PER_CHUNK,
                    constraint_nnz_fn=lambda constraint, _source_spec=source_spec: get_cached_tau_sector_linear_map_nnz(
                        _source_spec,
                        constraint.transpose_positions,
                        "partial_transpose",
                    ),
                )
                build_profile["ppt_chunk_count"] += len(constraint_chunks)
                for chunk_index, constraint_chunk in enumerate(constraint_chunks, start=1):
                    compile_start = perf_counter()
                    row_offset, sector_triplets, target_dims = get_cached_ppt_sector_batch_triplets(
                        source_spec,
                        constraint_chunk,
                    )
                    build_profile["ppt_batch_triplet_time"] += perf_counter() - compile_start

                    lhs_blocks = []
                    for target_coord_dim, constraint in zip(target_dims, constraint_chunk):
                        lhs_blocks.append(ppt_coordinate_vectors[constraint.ppt_variable_name])
                        constraint_counts["ppt"] += int(target_coord_dim)

                    emit_row_ranges = _iter_emit_row_ranges(row_offset, _PPT_MAX_ROWS_PER_EMIT)
                    build_profile["ppt_emit_slice_count"] += len(emit_row_ranges)
                    for emit_index, (row_start, row_stop) in enumerate(emit_row_ranges, start=1):
                        lhs_start = perf_counter()
                        lhs_coord = _slice_stacked_expr_blocks(
                            lhs_blocks,
                            target_dims,
                            row_start,
                            row_stop,
                            Expr,
                        )
                        build_profile["ppt_batch_rhs_stack_time"] += perf_counter() - lhs_start

                        rhs_terms = []
                        sliced_sector_triplets = _slice_sector_triplets(
                            sector_triplets,
                            row_start,
                            row_stop,
                        )
                        for sector_index, ((rows, cols, vals), sector_dim) in enumerate(
                            zip(sliced_sector_triplets, tau_sector_coordinate_dims[source_name])
                        ):
                            if vals.size == 0:
                                continue
                            matrix_start = perf_counter()
                            batch_map = Matrix.sparse(int(row_stop - row_start), int(sector_dim), rows, cols, vals)
                            build_profile["ppt_batch_matrix_time"] += perf_counter() - matrix_start
                            mul_start = perf_counter()
                            rhs_terms.append(
                                Expr.mul(batch_map, tau_sector_coordinate_vectors[source_name][sector_index])
                            )
                            build_profile["ppt_batch_mul_time"] += perf_counter() - mul_start

                        sub_start = perf_counter()
                        difference = Expr.sub(lhs_coord, _sum_expr(rhs_terms, Expr))
                        build_profile["ppt_batch_sub_time"] += perf_counter() - sub_start

                        emit_start = perf_counter()
                        M.constraint(
                            f"ppt_batch_{batch_index}_{chunk_index}_{emit_index}_{source_name}",
                            difference,
                            Domain.equalsTo(0.0),
                        )
                        build_profile["ppt_constraint_emit_time"] += perf_counter() - emit_start
                compiled_triplet_maps.clear()
                compiled_sector_triplet_maps.clear()
            else:
                constraint_chunks = _chunk_constraint_blocks(
                    constraints_for_source,
                    block_rows_fn=lambda constraint: ppt_coordinate_dims[constraint.ppt_variable_name],
                    max_blocks_per_chunk=_PPT_MAX_BLOCKS_PER_CHUNK,
                    max_rows_per_chunk=_PPT_MAX_ROWS_PER_CHUNK,
                    max_nnz_per_chunk=_PPT_MAX_NNZ_PER_CHUNK,
                    max_dense_bytes_per_chunk=_PPT_MAX_DENSE_BYTES_PER_CHUNK,
                )
                build_profile["ppt_chunk_count"] += len(constraint_chunks)
                for chunk_index, constraint_chunk in enumerate(constraint_chunks, start=1):
                    lhs_blocks = []
                    rhs_blocks = []
                    for constraint in constraint_chunk:
                        lhs_blocks.append(ppt_coordinate_vectors[constraint.ppt_variable_name])
                        target_coord_dim, linear_map = get_cached_free_partial_transpose_map(
                            constraint.slot_dims,
                            constraint.transpose_positions,
                            int(source_coord.getShape()[0]),
                        )
                        mul_start = perf_counter()
                        rhs_blocks.append(Expr.mul(linear_map, source_coord))
                        build_profile["ppt_batch_mul_time"] += perf_counter() - mul_start
                        constraint_counts["ppt"] += int(target_coord_dim)

                    stack_start = perf_counter()
                    lhs_coord = _stack_rhs_blocks(lhs_blocks, Expr, Matrix)
                    rhs_coord = _stack_rhs_blocks(rhs_blocks, Expr, Matrix)
                    build_profile["ppt_batch_rhs_stack_time"] += perf_counter() - stack_start

                    sub_start = perf_counter()
                    difference = Expr.sub(lhs_coord, rhs_coord)
                    build_profile["ppt_batch_sub_time"] += perf_counter() - sub_start

                    emit_start = perf_counter()
                    M.constraint(
                        f"ppt_batch_{batch_index}_{chunk_index}_{source_name}",
                        difference,
                        Domain.equalsTo(0.0),
                    )
                    build_profile["ppt_constraint_emit_time"] += perf_counter() - emit_start
        build_profile["ppt_batch_assembly_time"] = perf_counter() - step_start
        _progress_log(
            real_verbose,
            1,
            "Fusion block step 5/5 complete: "
            f"{constraint_counts['ppt']} scalar PPT constraints, "
            f"{constraint_counts['ppt_direct']} direct real PPT cones "
            f"in {perf_counter() - step_start:.2f}s.",
        )
        build_profile["ppt_emit_time"] = perf_counter() - step_start

    M.objective(ObjectiveSense.Minimize, 0.0)
    build_profile["total_build_time"] = perf_counter() - total_start
    _progress_log(
        real_verbose,
        1,
        "Block build profile: "
        f"rep quotient={build_profile['representative_quotient_time']:.2f}s, "
        f"rep precompile={build_profile['representative_precompile_time']:.2f}s, "
        f"rep batch={build_profile['representative_batch_assembly_time']:.2f}s, "
        f"rep batch-triplets={build_profile['representative_batch_triplet_time']:.2f}s, "
        f"rep batch-matrix={build_profile['representative_batch_matrix_time']:.2f}s, "
        f"rep batch-mul={build_profile['representative_batch_mul_time']:.2f}s, "
        f"rep rhs-stack={build_profile['representative_batch_rhs_stack_time']:.2f}s, "
        f"rep batch-sub={build_profile['representative_batch_sub_time']:.2f}s, "
        f"rep emit={build_profile['representative_constraint_emit_time']:.2f}s, "
        f"rep direct={build_profile['representative_direct_count']}, "
        f"rep direct-expr={build_profile['representative_direct_expr_time']:.2f}s, "
        f"rep direct-emit={build_profile['representative_direct_emit_time']:.2f}s, "
        f"rep chunks={build_profile['representative_chunk_count']}, "
        f"rep-batch hits(mem/disk/miss)="
        f"{build_profile['representative_batch_memory_hits']}/"
        f"{build_profile['representative_batch_disk_hits']}/"
        f"{build_profile['representative_batch_misses']}, "
        f"ppt batch={build_profile['ppt_batch_assembly_time']:.2f}s, "
        f"ppt batch-triplets={build_profile['ppt_batch_triplet_time']:.2f}s, "
        f"ppt batch-matrix={build_profile['ppt_batch_matrix_time']:.2f}s, "
        f"ppt batch-mul={build_profile['ppt_batch_mul_time']:.2f}s, "
        f"ppt rhs-stack={build_profile['ppt_batch_rhs_stack_time']:.2f}s, "
        f"ppt batch-sub={build_profile['ppt_batch_sub_time']:.2f}s, "
        f"ppt emit={build_profile['ppt_constraint_emit_time']:.2f}s, "
        f"ppt chunks={build_profile['ppt_chunk_count']}, "
        f"ppt emit-slices={build_profile['ppt_emit_slice_count']}, "
        f"ppt direct-cones={build_profile['ppt_direct_count']}, "
        f"ppt direct-mul={build_profile['ppt_direct_mul_time']:.2f}s, "
        f"ppt direct-svec={build_profile['ppt_direct_svec_time']:.2f}s, "
        f"ppt direct-emit={build_profile['ppt_direct_emit_time']:.2f}s, "
        f"ppt direct-tau={build_profile['ppt_direct_tau_count']}, "
        f"ppt tau-rows={build_profile['ppt_direct_tau_rows_full']}->"
        f"{build_profile['ppt_direct_tau_rows_reduced']}, "
        f"ppt tau-select={build_profile['ppt_direct_tau_select_time']:.2f}s, "
        f"ppt tau-mul={build_profile['ppt_direct_tau_mul_time']:.2f}s, "
        f"ppt tau-emit={build_profile['ppt_direct_tau_emit_time']:.2f}s, "
        f"ppt direct-reduced={build_profile['ppt_direct_reduced_count']}, "
        f"ppt reduced-mul={build_profile['ppt_direct_reduced_mul_time']:.2f}s, "
        f"ppt reduced-svec={build_profile['ppt_direct_reduced_svec_time']:.2f}s, "
        f"ppt reduced-emit={build_profile['ppt_direct_reduced_emit_time']:.2f}s, "
        f"tau-map hits(mem/disk/miss)="
        f"{build_profile['tau_map_memory_hits']}/"
        f"{build_profile['tau_map_disk_hits']}/"
        f"{build_profile['tau_map_misses']}",
    )
    _progress_log(
        real_verbose,
        1,
        "Block Fusion model ready in "
        f"{perf_counter() - total_start:.2f}s with constraint counts {constraint_counts}.",
    )
    return FusionBlockStateSDPModel(
        model=M,
        Hermitian=Hermitian,
        tau_variables=tau_variables,
        tau_block_variables=tau_block_variables,
        auxiliary_coordinate_vectors=auxiliary_coordinate_vectors,
        auxiliary_variables=auxiliary_variables,
        known_representative_coordinate_vectors=known_representative_coordinate_vectors,
        known_representative_variables=known_representative_variables,
        ppt_coordinate_vectors=ppt_coordinate_vectors,
        ppt_variables=ppt_variables,
        block_tau_data=block_tau_data,
        constraint_counts=constraint_counts,
        build_profile=build_profile,
    )



def solve_block_fusion_feasibility(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    solver_parameters: Dict[str, object] | None = None,
    verbose: int = -1,
    include_ppt: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = False,
    Hermitian: bool = True,
) -> FusionSolveResult:
    """Build and solve the block-symmetrized GNME feasibility model."""
    from mosek.fusion import AccSolutionStatus

    base_model = assigned.model if isinstance(assigned, AssignedStateSDPDraft) else assigned
    real_verbose = _resolve_verbose(verbose, base_model.verbose)
    total_start = perf_counter()
    fusion_model = build_block_fusion_feasibility_model(
        assigned,
        include_ppt=include_ppt,
        include_representatives=include_representatives,
        enforce_known_values=enforce_known_values,
        Hermitian=Hermitian,
        verbose=real_verbose,
    )
    M = fusion_model.model
    try:
        if real_verbose > 0:
            _progress_log(real_verbose, 1, "Starting MOSEK solve for block backend...")
            M.setLogHandler(__import__("sys").stdout)
        if solver_parameters:
            for param, value in solver_parameters.items():
                M.setSolverParam(param, value)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        solve_start = perf_counter()
        M.solve()
        _progress_log(
            real_verbose,
            1,
            "MOSEK block solve finished in "
            f"{perf_counter() - solve_start:.2f}s.",
        )
        objective_value = None
        try:
            objective_value = float(M.primalObjValue())
        except Exception:
            objective_value = None
        result = FusionSolveResult(
            problem_status=str(M.getProblemStatus()),
            primal_status=str(M.getPrimalSolutionStatus()),
            dual_status=str(M.getDualSolutionStatus()),
            objective_value=objective_value,
            constraint_counts=fusion_model.constraint_counts,
        )
        _progress_log(
            real_verbose,
            1,
            "Overall block feasibility pipeline finished in "
            f"{perf_counter() - total_start:.2f}s. Status={result.problem_status}.",
        )
        return result
    finally:
        M.dispose()


__all__ = [
    "BlockTauVariableData",
    "FusionBlockStateSDPModel",
    "build_block_fusion_feasibility_model",
    "solve_block_fusion_feasibility",
]
