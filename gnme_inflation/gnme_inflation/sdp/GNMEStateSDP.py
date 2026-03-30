"""GNME state-SDP builder.

This module is intentionally a thin layer on top of ``GNMEProblem``.
It does not recreate the combinatorics of the inflation package. Instead, it
turns the structural output of ``GNMEProblem`` into the objects we need for a
state-SDP:

1. full PSD variables ``tau_i`` for the maximal non-fanout inflations,
2. canonical reduced representatives ``mu_i`` / ``nu_i``,
3. partial-trace constraints tying full variables to those representatives,
4. internal copy-index symmetry constraints,
5. PPT auxiliary variables and constraints,
6. a simple MOSEK Fusion feasibility model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..GNMEProblem import GNMESDPBlueprint, GNMEProblem


# ---------------------------------------------------------------------------
# Structural draft objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PSDVariableDraft:
    """Structural description of one PSD state variable."""

    name: str
    lexorder: Tuple[str, ...]
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    local_symmetry_perms: Tuple[Tuple[int, ...], ...]
    factorization: Tuple[Tuple[str, ...], ...]
    fixed_known_marginals: Dict[Tuple[str, ...], Tuple]


@dataclass(frozen=True)
class MarginalConstraintDraft:
    """One marginalization rule written as linear sums over matrix entries."""

    variable_name: str
    keep_positions: Tuple[int, ...]
    traced_positions: Tuple[int, ...]
    target_lexorder: Tuple[str, ...]
    output_dim: int
    equation_count: int
    terms_per_equation: int
    einsum_spec: str
    named_einsum_spec: str
    input_tensor_shape: Tuple[int, ...]
    output_tensor_shape: Tuple[int, ...]


@dataclass(frozen=True)
class InternalSymmetryConstraintDraft:
    """One internal copy-index symmetry imposed on a PSD variable."""

    variable_name: str
    variable_kind: str
    lexorder: Tuple[str, ...]
    permutation: Tuple[int, ...]
    permuted_lexorder: Tuple[str, ...]
    named_action: str


@dataclass(frozen=True)
class RepresentativeConstraintDraft:
    """One reduced-marginal equality to a canonical representative."""

    source_variable_name: str
    source_variable_kind: str
    source_lexorder: Tuple[str, ...]
    representative_name: str
    representative_kind: str
    representative_lexorder: Tuple[str, ...]
    keep_positions: Tuple[int, ...]
    traced_positions: Tuple[int, ...]
    named_einsum_spec: str
    symbolic_einsum_spec: str
    occurrence_inflation_name: str
    occurrence_labels: Tuple[str, ...]


@dataclass(frozen=True)
class StateSDPDraft:
    """State-level SDP draft, before passing anything to MOSEK."""

    blueprint: GNMESDPBlueprint
    party_dims: Dict[str, int]
    verbose: int
    psd_variables: Tuple[PSDVariableDraft, ...]
    auxiliary_representatives: Tuple["SharedMarginalRepresentativeDraft", ...]
    known_representatives: Tuple["SharedMarginalRepresentativeDraft", ...]
    shared_representatives: Tuple["SharedMarginalRepresentativeDraft", ...]
    internal_symmetry_constraints: Tuple[InternalSymmetryConstraintDraft, ...]
    representative_links: Tuple["RepresentativeLinkDraft", ...]
    representative_constraints: Tuple[RepresentativeConstraintDraft, ...]
    ppt_variables: Tuple["PPTVariableDraft", ...]
    ppt_constraints: Tuple["PPTConstraintDraft", ...]
    fixed_marginal_constraints: Tuple[MarginalConstraintDraft, ...]
    shared_marginal_constraints: Tuple[MarginalConstraintDraft, ...]


@dataclass(frozen=True)
class AssignedStateSDPDraft:
    """State SDP draft with matrices attached to known representatives."""

    model: StateSDPDraft
    known_values: Dict[str, "KnownValueAssignmentDraft"]


@dataclass
class FusionStateSDPModel:
    """Concrete MOSEK Fusion feasibility model built from the draft."""

    model: object
    tau_scalar_variables: Dict[str, object]
    tau_variables: Dict[str, object]
    auxiliary_variables: Dict[str, object]
    known_representative_variables: Dict[str, object]
    ppt_variables: Dict[str, object]
    constraint_counts: Dict[str, int]


@dataclass(frozen=True)
class SymmetricMatrixOrbitData:
    """Orbit decomposition of matrix entries under a local symmetry group."""

    matrix_dim: int
    upper_triangular_entries: int
    orbit_representatives: Tuple[Tuple[int, int], ...]
    pair_to_orbit: Dict[Tuple[int, int], int]


def _canonical_keep_positions_orbit(
    keep_positions: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> Tuple[int, ...]:
    """Canonical representative of a kept-slot pattern under source symmetry.

    The source marginal map depends on the subset of slots we keep. When the
    source variable is already constrained to be invariant under its local copy
    symmetries, two kept-slot patterns in the same orbit yield redundant
    representative constraints as long as they target the same canonical
    representative.
    """
    if not local_symmetry_perms:
        return keep_positions
    orbit = {
        tuple(sorted(permutation[position] for position in keep_positions))
        for permutation in local_symmetry_perms
    }
    return min(orbit)


def quotient_representative_constraints(
    model: StateSDPDraft,
    use_source_symmetry: bool = True,
) -> Tuple[RepresentativeConstraintDraft, ...]:
    """Remove representative constraints duplicated by source symmetries.

    The quotient is safe only when the source variable is itself symmetric:
    either because the block backend builds the symmetry into the variable
    declaration, or because the legacy backend also imposes the internal
    symmetry constraints. We therefore keep the raw list when
    `use_source_symmetry=False`.
    """
    if not use_source_symmetry:
        return model.representative_constraints

    variable_lookup = {variable.name: variable for variable in model.psd_variables}
    kept_constraints = []
    seen = set()
    for constraint in model.representative_constraints:
        source_variable = variable_lookup.get(constraint.source_variable_name)
        if (
            source_variable is None
            or constraint.source_variable_kind != "tau"
            or not source_variable.local_symmetry_perms
        ):
            key = (
                constraint.source_variable_name,
                constraint.representative_name,
                constraint.keep_positions,
            )
        else:
            key = (
                constraint.source_variable_name,
                constraint.representative_name,
                _canonical_keep_positions_orbit(
                    constraint.keep_positions,
                    source_variable.local_symmetry_perms,
                ),
            )
        if key in seen:
            continue
        seen.add(key)
        kept_constraints.append(constraint)
    return tuple(kept_constraints)


@dataclass(frozen=True)
class FusionSolveResult:
    """Compact solver summary for the GNME feasibility model."""

    problem_status: str
    primal_status: str
    dual_status: str
    objective_value: float | None
    constraint_counts: Dict[str, int]


@dataclass(frozen=True)
class SharedMarginalRepresentativeDraft:
    """One auxiliary reduced-state variable for a shared subset class."""

    name: str
    signature: Tuple[Tuple[str, ...], Tuple[Tuple[str, Tuple[Tuple[str, ...], ...]], ...]]
    representative_occurrence: object
    target_lexorder: Tuple[str, ...]
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    factorization: Tuple[Tuple[str, ...], ...]
    kind: str
    is_fixed_known: bool
    occurrence_count: int


@dataclass(frozen=True)
class RepresentativeLinkDraft:
    """Assignment of one occurrence to its canonical representative."""

    representative_name: str
    representative_kind: str
    occurrence: object


@dataclass(frozen=True)
class KnownValueAssignmentDraft:
    """One numeric matrix assignment to a known representative."""

    representative_name: str
    target_lexorder: Tuple[str, ...]
    matrix: np.ndarray
    matrix_dim: int


@dataclass(frozen=True)
class PPTVariableDraft:
    """Auxiliary PSD variable used to enforce a PPT condition."""

    name: str
    source_variable_name: str
    source_variable_kind: str
    lexorder: Tuple[str, ...]
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    transpose_positions: Tuple[int, ...]
    complement_positions: Tuple[int, ...]
    transpose_lexorder: Tuple[str, ...]
    complement_lexorder: Tuple[str, ...]


@dataclass(frozen=True)
class PPTConstraintDraft:
    """Entrywise equality tying a PPT auxiliary variable to a partial transpose."""

    ppt_variable_name: str
    source_variable_name: str
    source_variable_kind: str
    lexorder: Tuple[str, ...]
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    factorization: Tuple[Tuple[str, ...], ...]
    transpose_positions: Tuple[int, ...]
    complement_positions: Tuple[int, ...]
    transpose_lexorder: Tuple[str, ...]
    complement_lexorder: Tuple[str, ...]
    named_einsum_spec: str
    symbolic_einsum_spec: str


# ---------------------------------------------------------------------------
# Tensor bookkeeping helpers
# ---------------------------------------------------------------------------

def product_dim(dims: Iterable[int]) -> int:
    """Small helper to avoid repeated numpy boilerplate."""
    value = 1
    for dim in dims:
        value *= int(dim)
    return value


def _resolve_verbose(explicit_verbose: int | None, fallback_verbose: int = 0) -> int:
    """Resolve a verbosity level while preserving inflation-style defaults."""
    if explicit_verbose is None:
        return int(fallback_verbose)
    if explicit_verbose == -1:
        return int(fallback_verbose)
    return int(explicit_verbose)


def _progress_log(verbose: int, level: int, message: str) -> None:
    """Emit a progress message when the requested verbosity allows it."""
    if verbose >= level:
        print(f"[GNMEStateSDP] {message}", flush=True)


def _progress_stride(total: int) -> int:
    """Choose a coarse logging stride so progress is visible but not noisy."""
    if total <= 10:
        return 1
    if total <= 100:
        return 10
    return 25


def partial_trace_einsum_spec(
    slot_dims: Tuple[int, ...],
    keep_positions: Tuple[int, ...],
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
    """Describe partial trace in tensor/einsum notation.

    The full density matrix is viewed as a 2N-index tensor with axes ordered as
    `(bra_0, ..., bra_{N-1}, ket_0, ..., ket_{N-1})`. Traced subsystems use the
    same einsum label on bra and ket, while kept subsystems use distinct labels.
    """
    symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    n_slots = len(slot_dims)
    needed = n_slots + len(keep_positions)
    if needed > len(symbols):
        raise ValueError("Not enough einsum symbols for this number of subsystems.")

    keep_set = set(keep_positions)
    bra_labels: List[str] = []
    ket_labels: List[str] = []
    output_labels: List[str] = []
    cursor = 0
    for pos in range(n_slots):
        if pos in keep_set:
            bra_symbol = symbols[cursor]
            ket_symbol = symbols[cursor + 1]
            cursor += 2
            bra_labels.append(bra_symbol)
            ket_labels.append(ket_symbol)
            output_labels.extend([bra_symbol, ket_symbol])
        else:
            trace_symbol = symbols[cursor]
            cursor += 1
            bra_labels.append(trace_symbol)
            ket_labels.append(trace_symbol)

    input_labels = "".join(bra_labels + ket_labels)
    output_subscript = "".join(output_labels)
    input_tensor_shape = tuple(slot_dims) + tuple(slot_dims)
    output_tensor_shape = tuple(slot_dims[pos] for pos in keep_positions) * 2
    return f"{input_labels}->{output_subscript}", input_tensor_shape, output_tensor_shape


def partial_trace_named_einsum_spec(
    lexorder: Tuple[str, ...],
    keep_positions: Tuple[int, ...],
) -> Tuple[str, Tuple[str, ...]]:
    """Human-readable einsum notation using lexorder labels directly.

    Kept subsystems are split into bra/ket labels, while traced subsystems use
    the same label on both sides to indicate contraction.
    """
    keep_set = set(keep_positions)
    bra_labels: List[str] = []
    ket_labels: List[str] = []
    output_labels: List[str] = []
    for pos, label in enumerate(lexorder):
        if pos in keep_set:
            bra_label = f"{label}_bra"
            ket_label = f"{label}_ket"
            bra_labels.append(bra_label)
            ket_labels.append(ket_label)
            output_labels.extend([bra_label, ket_label])
        else:
            bra_labels.append(label)
            ket_labels.append(label)
    named_spec = (
        f"{' '.join(bra_labels + ket_labels)}"
        f" -> {' '.join(output_labels)}"
    )
    target_lexorder = tuple(lexorder[pos] for pos in keep_positions)
    return named_spec, target_lexorder


def partial_trace_recipe(
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    keep_positions: Tuple[int, ...],
) -> MarginalConstraintDraft:
    """Describe the linear map for a reduced density matrix.

    For a state X on the full lexorder, each reduced entry is a sum of full
    matrix entries with identical traced indices on bra and ket. The number of
    summed terms per reduced entry is exactly the product of the traced-out
    dimensions.
    """
    all_positions = tuple(range(len(slot_dims)))
    traced_positions = tuple(pos for pos in all_positions if pos not in keep_positions)
    kept_dims = tuple(slot_dims[pos] for pos in keep_positions)
    traced_dims = tuple(slot_dims[pos] for pos in traced_positions)
    output_dim = product_dim(kept_dims)
    terms_per_equation = product_dim(traced_dims) if traced_dims else 1
    einsum_spec, input_tensor_shape, output_tensor_shape = partial_trace_einsum_spec(
        slot_dims,
        keep_positions,
    )
    named_einsum_spec, target_lexorder = partial_trace_named_einsum_spec(
        lexorder,
        keep_positions,
    )
    return MarginalConstraintDraft(
        variable_name="",
        keep_positions=keep_positions,
        traced_positions=traced_positions,
        target_lexorder=target_lexorder,
        output_dim=output_dim,
        equation_count=output_dim * output_dim,
        terms_per_equation=terms_per_equation,
        einsum_spec=einsum_spec,
        named_einsum_spec=named_einsum_spec,
        input_tensor_shape=input_tensor_shape,
        output_tensor_shape=output_tensor_shape,
    )


def partial_trace_terms(
    slot_dims: Tuple[int, ...],
    keep_positions: Tuple[int, ...],
    max_equations: int = 3,
) -> List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    """Give a small explicit sample of partial-trace summation terms.

    Each returned entry is one reduced matrix element written as a list of full
    matrix positions `(bra_multiindex, ket_multiindex)` that must be summed.
    This is the linear data that would later be translated into MOSEK Fusion
    equality constraints.
    """
    all_positions = tuple(range(len(slot_dims)))
    traced_positions = tuple(pos for pos in all_positions if pos not in keep_positions)
    kept_ranges = [range(slot_dims[pos]) for pos in keep_positions]
    traced_ranges = [range(slot_dims[pos]) for pos in traced_positions]
    examples: List[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = []
    eq_count = 0
    for bra_keep in product(*kept_ranges):
        for ket_keep in product(*kept_ranges):
            terms: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
            for traced_index in product(*traced_ranges) if traced_ranges else [tuple()]:
                bra = [0] * len(slot_dims)
                ket = [0] * len(slot_dims)
                for pos, value in zip(keep_positions, bra_keep):
                    bra[pos] = value
                for pos, value in zip(keep_positions, ket_keep):
                    ket[pos] = value
                for pos, value in zip(traced_positions, traced_index):
                    bra[pos] = value
                    ket[pos] = value
                terms.append((tuple(bra), tuple(ket)))
            examples.append(terms)
            eq_count += 1
            if eq_count >= max_equations:
                return examples
    return examples


def _flat_index(multi_index: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    """Row-major flattening of a tensor multi-index."""
    value = 0
    for entry, dim in zip(multi_index, dims):
        value = value * dim + entry
    return value


def _unflatten_index(index: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Inverse of `_flat_index` for small tensor-product spaces."""
    entries = [0] * len(dims)
    value = int(index)
    for pos in range(len(dims) - 1, -1, -1):
        entries[pos] = value % dims[pos]
        value //= dims[pos]
    return tuple(entries)


def _symmetric_entry(variable, row: int, col: int):
    """Access one entry of a real symmetric Fusion matrix object."""
    i, j = (row, col) if row <= col else (col, row)
    return variable.index(np.array([i, j], dtype=np.int32))


def _sum_expr(expressions: Iterable, Expr):
    """Robust summation helper for Fusion expressions."""
    expressions = list(expressions)
    if not expressions:
        return 0.0
    # Build a balanced addition tree. Long left-associated chains trigger
    # deep recursive evaluation in Fusion on large level-3 models.
    while len(expressions) > 1:
        next_level = []
        for index in range(0, len(expressions), 2):
            if index + 1 < len(expressions):
                next_level.append(Expr.add(expressions[index], expressions[index + 1]))
            else:
                next_level.append(expressions[index])
        expressions = next_level
    return expressions[0]


def _trace_expr(variable, matrix_dim: int, Expr):
    """Trace of a real symmetric Fusion PSD variable."""
    return _sum_expr((_symmetric_entry(variable, i, i) for i in range(matrix_dim)), Expr)


def symmetry_action_named_spec(
    lexorder: Tuple[str, ...],
    permutation: Tuple[int, ...],
) -> str:
    """Readable description of a slot permutation."""
    assignments = [
        f"{lexorder[idx]}->{lexorder[permutation[idx]]}"
        for idx in range(len(lexorder))
    ]
    return ", ".join(assignments)


def _basis_permutation_map(
    slot_dims: Tuple[int, ...],
    slot_permutation: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Map flat basis indices under a permutation of tensor slots."""
    dim = product_dim(slot_dims)
    return tuple(
        _flat_index(
            tuple(_unflatten_index(index, slot_dims)[pos] for pos in slot_permutation),
            slot_dims,
        )
        for index in range(dim)
    )


@lru_cache(maxsize=None)
def symmetric_matrix_orbits(
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> SymmetricMatrixOrbitData:
    """Entry orbits of a symmetric matrix under local copy-index symmetries.

    The action is induced on basis indices, then lifted to upper-triangular
    matrix entries `(i, j)` with `i <= j`. One scalar decision variable per
    orbit is enough to parameterize the invariant matrix.
    """
    matrix_dim = product_dim(slot_dims)
    upper_pairs = tuple(
        (row, col)
        for row in range(matrix_dim)
        for col in range(row, matrix_dim)
    )
    seen_basis_maps = set()
    basis_maps = []
    for perm in local_symmetry_perms:
        basis_map = _basis_permutation_map(slot_dims, perm)
        if basis_map not in seen_basis_maps:
            seen_basis_maps.add(basis_map)
            basis_maps.append(basis_map)

    pair_to_orbit: Dict[Tuple[int, int], int] = {}
    orbit_representatives: List[Tuple[int, int]] = []
    unvisited = set(upper_pairs)
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        orbit = {seed}
        unvisited.remove(seed)
        while stack:
            row, col = stack.pop()
            for basis_map in basis_maps:
                mapped_row = basis_map[row]
                mapped_col = basis_map[col]
                mapped_pair = (
                    (mapped_row, mapped_col)
                    if mapped_row <= mapped_col
                    else (mapped_col, mapped_row)
                )
                if mapped_pair not in orbit:
                    orbit.add(mapped_pair)
                    if mapped_pair in unvisited:
                        unvisited.remove(mapped_pair)
                    stack.append(mapped_pair)
        orbit_index = len(orbit_representatives)
        representative = min(orbit)
        orbit_representatives.append(representative)
        for pair in orbit:
            pair_to_orbit[pair] = orbit_index

    return SymmetricMatrixOrbitData(
        matrix_dim=matrix_dim,
        upper_triangular_entries=len(upper_pairs),
        orbit_representatives=tuple(orbit_representatives),
        pair_to_orbit=pair_to_orbit,
    )


def affine_symmetric_psd_matrix(
    M,
    name: str,
    matrix_dim: int,
    orbit_data: SymmetricMatrixOrbitData,
    Domain,
    Expr,
    Matrix,
):
    """Create one PSD matrix expression from one scalar variable per orbit."""
    orbit_count = len(orbit_data.orbit_representatives)
    scalar_variables = M.variable(f"{name}_orbits", orbit_count, Domain.unbounded())
    rows = []
    cols = []
    vals = []
    for row in range(matrix_dim):
        base = row * matrix_dim
        for col in range(matrix_dim):
            pair = (row, col) if row <= col else (col, row)
            rows.append(base + col)
            cols.append(orbit_data.pair_to_orbit[pair])
            vals.append(1.0)
    lifting = Matrix.sparse(
        matrix_dim * matrix_dim,
        orbit_count,
        rows,
        cols,
        vals,
    )
    matrix_expression = Expr.reshape(Expr.mul(lifting, scalar_variables), matrix_dim, matrix_dim)
    M.constraint(f"psd_{name}", matrix_expression, Domain.inPSDCone(matrix_dim))
    return scalar_variables, matrix_expression


def _partial_transpose_entry_map(
    row: int,
    col: int,
    slot_dims: Tuple[int, ...],
    transpose_positions: Tuple[int, ...],
) -> Tuple[int, int]:
    """Map one matrix entry under partial transposition on selected slots."""
    bra = list(_unflatten_index(row, slot_dims))
    ket = list(_unflatten_index(col, slot_dims))
    for pos in transpose_positions:
        bra[pos], ket[pos] = ket[pos], bra[pos]
    return _flat_index(tuple(bra), slot_dims), _flat_index(tuple(ket), slot_dims)


def factor_positions(
    lexorder: Tuple[str, ...],
    factorization: Tuple[Tuple[str, ...], ...],
) -> Tuple[Tuple[int, ...], ...]:
    """Convert factor labels into lexorder slot positions."""
    lookup = {label: idx for idx, label in enumerate(lexorder)}
    return tuple(
        tuple(lookup[label] for label in factor)
        for factor in factorization
    )


def canonical_factor_bipartitions(
    factorization: Tuple[Tuple[str, ...], ...],
    lexorder: Tuple[str, ...],
) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Generate bipartitions from factor blocks, modulo complement symmetry."""
    if len(factorization) <= 1:
        return tuple()

    factor_pos = factor_positions(lexorder, factorization)
    all_positions = tuple(range(len(lexorder)))
    canonical = {}
    factor_indices = range(len(factor_pos))
    for r in range(1, len(factor_pos)):
        for subset in combinations(factor_indices, r):
            transpose = tuple(sorted(
                pos for idx in subset for pos in factor_pos[idx]
            ))
            complement = tuple(pos for pos in all_positions if pos not in transpose)
            key = min((transpose, complement), (complement, transpose))
            canonical[key] = key
    return tuple(sorted(canonical.values()))


def partial_transpose_einsum_spec(
    slot_dims: Tuple[int, ...],
    transpose_positions: Tuple[int, ...],
) -> str:
    """Compact symbolic einsum notation for partial transposition."""
    symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    n_slots = len(slot_dims)
    if 2 * n_slots > len(symbols):
        raise ValueError("Not enough einsum symbols for this number of subsystems.")

    bra_labels = symbols[:n_slots]
    ket_labels = symbols[n_slots: 2 * n_slots]
    output_bra = list(bra_labels)
    output_ket = list(ket_labels)
    for pos in transpose_positions:
        output_bra[pos], output_ket[pos] = output_ket[pos], output_bra[pos]
    return (
        f"{''.join(bra_labels + ket_labels)}->"
        f"{''.join(output_bra + output_ket)}"
    )


def partial_transpose_named_einsum_spec(
    lexorder: Tuple[str, ...],
    transpose_positions: Tuple[int, ...],
) -> str:
    """Human-readable partial-transpose notation using lexorder labels."""
    bra_labels = [f"{label}_bra" for label in lexorder]
    ket_labels = [f"{label}_ket" for label in lexorder]
    output_bra = list(bra_labels)
    output_ket = list(ket_labels)
    for pos in transpose_positions:
        output_bra[pos], output_ket[pos] = output_ket[pos], output_bra[pos]
    return (
        f"{' '.join(bra_labels + ket_labels)}"
        f" -> {' '.join(output_bra + output_ket)}"
    )


def ppt_constraints_for_object(
    variable_name: str,
    variable_kind: str,
    lexorder: Tuple[str, ...],
    slot_dims: Tuple[int, ...],
    factorization: Tuple[Tuple[str, ...], ...],
    prefix: str,
) -> Tuple[Tuple[PPTVariableDraft, PPTConstraintDraft], ...]:
    """Build PPT auxiliary variables and equality constraints."""
    constraints = []
    for idx, (transpose_positions, complement_positions) in enumerate(
        canonical_factor_bipartitions(factorization, lexorder)
    ):
        ppt_name = f"{prefix}_{idx}"
        transpose_lexorder = tuple(lexorder[pos] for pos in transpose_positions)
        complement_lexorder = tuple(lexorder[pos] for pos in complement_positions)
        constraints.append(
            (
                PPTVariableDraft(
                    name=ppt_name,
                    source_variable_name=variable_name,
                    source_variable_kind=variable_kind,
                    lexorder=lexorder,
                    slot_dims=slot_dims,
                    matrix_dim=product_dim(slot_dims),
                    transpose_positions=transpose_positions,
                    complement_positions=complement_positions,
                    transpose_lexorder=transpose_lexorder,
                    complement_lexorder=complement_lexorder,
                ),
                PPTConstraintDraft(
                    ppt_variable_name=ppt_name,
                    source_variable_name=variable_name,
                    source_variable_kind=variable_kind,
                    lexorder=lexorder,
                    slot_dims=slot_dims,
                    matrix_dim=product_dim(slot_dims),
                    factorization=factorization,
                    transpose_positions=transpose_positions,
                    complement_positions=complement_positions,
                    transpose_lexorder=transpose_lexorder,
                    complement_lexorder=complement_lexorder,
                    named_einsum_spec=partial_transpose_named_einsum_spec(
                        lexorder,
                        transpose_positions,
                    ),
                    symbolic_einsum_spec=partial_transpose_einsum_spec(
                        slot_dims,
                        transpose_positions,
                    ),
                ),
            )
        )
    return tuple(constraints)


# ---------------------------------------------------------------------------
# Draft construction from GNMEProblem
# ---------------------------------------------------------------------------

def build_sdp_draft(
    problem: GNMEProblem,
    subset_sizes: Tuple[int, ...] = (2, 3, 4),
    local_dims_per_party: Dict[str, int] | Tuple[int, ...] | int | None = None,
    verbose: int | None = None,
):
    """Build the readable SDP draft from the package-level GNME blueprint.

    The function mirrors the way the inflation package first builds a purely
    combinatorial description and only later gives it numerical meaning.

    The output keeps four layers separate:
    - full PSD variables ``tau_i``,
    - canonical representatives for shared reduced marginals,
    - internal symmetries of each full variable,
    - PPT structure extracted from factorization.
    """
    real_verbose = _resolve_verbose(verbose, getattr(problem, "verbose", 0))
    t0 = perf_counter()
    _progress_log(real_verbose, 1, "Building GNME SDP blueprint...")
    blueprint = problem.sdp_blueprint(
        subset_sizes=subset_sizes,
        include_known_marginals=True,
        min_shared_occurrences=2,
        min_shared_inflations=2,
        local_dims_per_party=local_dims_per_party,
    )
    _progress_log(
        real_verbose,
        1,
        "Blueprint ready: "
        f"{len(blueprint.variables)} full inflations, "
        f"{len(blueprint.shared_subset_classes)} shared subset classes "
        f"in {perf_counter() - t0:.2f}s.",
    )

    # Step 1: create one full PSD variable per maximal GNME inflation and
    # collect the constraints that live purely inside that inflation.
    step_start = perf_counter()
    _progress_log(real_verbose, 1, "Draft step 1/3: building full PSD variables and internal structure...")
    psd_variables = []
    internal_symmetry_constraints = []
    ppt_variables = []
    ppt_constraints = []
    fixed_constraints = []
    for variable_index, variable in enumerate(blueprint.variables, start=1):
        _progress_log(
            real_verbose,
            2,
            f"  full variable {variable_index}/{len(blueprint.variables)}: {variable.name}",
        )
        psd_variables.append(
            PSDVariableDraft(
                name=variable.name,
                lexorder=variable.lexorder,
                slot_dims=variable.slot_dims,
                matrix_dim=product_dim(variable.slot_dims),
                local_symmetry_perms=variable.local_symmetry_perms,
                factorization=variable.factorization,
                fixed_known_marginals=variable.fixed_known_marginals,
            )
        )
        for permutation in variable.local_symmetry_perms:
            if permutation == tuple(range(len(variable.lexorder))):
                continue
            permuted_lexorder = tuple(variable.lexorder[pos] for pos in permutation)
            internal_symmetry_constraints.append(
                InternalSymmetryConstraintDraft(
                    variable_name=variable.name,
                    variable_kind="tau",
                    lexorder=variable.lexorder,
                    permutation=permutation,
                    permuted_lexorder=permuted_lexorder,
                    named_action=symmetry_action_named_spec(
                        variable.lexorder,
                        permutation,
                    ),
                )
            )
        for ppt_variable, ppt_constraint in ppt_constraints_for_object(
                variable_name=variable.name,
                variable_kind="tau",
                lexorder=variable.lexorder,
                slot_dims=variable.slot_dims,
                factorization=variable.factorization,
                prefix=f"{variable.name}_pt",
            ):
            ppt_variables.append(ppt_variable)
            ppt_constraints.append(ppt_constraint)
        for occurrences in variable.fixed_known_marginals.values():
            for occurrence in occurrences:
                recipe = partial_trace_recipe(
                    variable.lexorder,
                    variable.slot_dims,
                    occurrence.positions,
                )
                fixed_constraints.append(
                    MarginalConstraintDraft(
                        variable_name=variable.name,
                        keep_positions=recipe.keep_positions,
                        traced_positions=recipe.traced_positions,
                        target_lexorder=recipe.target_lexorder,
                        output_dim=recipe.output_dim,
                        equation_count=recipe.equation_count,
                        terms_per_equation=recipe.terms_per_equation,
                        einsum_spec=recipe.einsum_spec,
                        named_einsum_spec=recipe.named_einsum_spec,
                        input_tensor_shape=recipe.input_tensor_shape,
                        output_tensor_shape=recipe.output_tensor_shape,
                    )
                )
    _progress_log(
        real_verbose,
        1,
        "Draft step 1/3 complete: "
        f"{len(psd_variables)} full variables, "
        f"{len(internal_symmetry_constraints)} internal symmetries, "
        f"{len(ppt_variables)} PPT auxiliaries so far, "
        f"{len(fixed_constraints)} fixed marginal rules "
        f"in {perf_counter() - step_start:.2f}s.",
    )

    # Step 2: identify reduced subsystems that are shared across different full
    # inflations. Each such class gets one canonical representative.
    step_start = perf_counter()
    _progress_log(real_verbose, 1, "Draft step 2/3: building shared reduced representatives...")
    auxiliary_representatives = []
    known_representatives = []
    shared_representatives = []
    representative_links = []
    representative_constraints = []
    shared_constraints = []
    known_representatives_by_signature = {}
    shared_stride = _progress_stride(len(blueprint.shared_subset_classes))
    for class_index, subset_class in enumerate(blueprint.shared_subset_classes):
        if real_verbose >= 2 and (
            class_index == 0
            or (class_index + 1) % shared_stride == 0
            or class_index + 1 == len(blueprint.shared_subset_classes)
        ):
            _progress_log(
                real_verbose,
                2,
                f"  shared class {class_index + 1}/{len(blueprint.shared_subset_classes)}",
            )
        representative_occurrence = min(
            subset_class.occurrences,
            key=lambda occ: (occ.inflation_index, occ.positions, occ.labels),
        )
        representative_slot_dims = tuple(
            blueprint.party_dims[label.split("_", 1)[0]]
            for label in representative_occurrence.labels
        )
        representative_kind = "known_matrix" if representative_occurrence.is_known_marginal else "aux_psd"
        representative = SharedMarginalRepresentativeDraft(
            name=f"mu_{class_index}",
            signature=subset_class.signature,
            representative_occurrence=representative_occurrence,
            target_lexorder=representative_occurrence.labels,
            slot_dims=representative_slot_dims,
            matrix_dim=product_dim(representative_slot_dims),
            factorization=representative_occurrence.factorization,
            kind=representative_kind,
            is_fixed_known=representative_occurrence.is_known_marginal,
            occurrence_count=len(subset_class.occurrences),
        )
        shared_representatives.append(representative)
        if representative_kind == "aux_psd":
            auxiliary_representatives.append(representative)
            for ppt_variable, ppt_constraint in ppt_constraints_for_object(
                    variable_name=representative.name,
                    variable_kind="mu",
                    lexorder=representative.target_lexorder,
                    slot_dims=representative.slot_dims,
                    factorization=representative.factorization,
                    prefix=f"{representative.name}_pt",
                ):
                ppt_variables.append(ppt_variable)
                ppt_constraints.append(ppt_constraint)
        else:
            known_representatives.append(representative)
            known_representatives_by_signature[subset_class.signature] = representative
        for occurrence in subset_class.occurrences:
            representative_links.append(
                RepresentativeLinkDraft(
                    representative_name=f"mu_{class_index}",
                    representative_kind=representative_kind,
                    occurrence=occurrence,
                )
            )
            variable = blueprint.variables[occurrence.inflation_index]
            recipe = partial_trace_recipe(
                variable.lexorder,
                variable.slot_dims,
                occurrence.positions,
            )
            representative_constraints.append(
                RepresentativeConstraintDraft(
                    source_variable_name=variable.name,
                    source_variable_kind="tau",
                    source_lexorder=variable.lexorder,
                    representative_name=f"mu_{class_index}",
                    representative_kind=representative_kind,
                    representative_lexorder=representative.target_lexorder,
                    keep_positions=recipe.keep_positions,
                    traced_positions=recipe.traced_positions,
                    named_einsum_spec=recipe.named_einsum_spec,
                    symbolic_einsum_spec=recipe.einsum_spec,
                    occurrence_inflation_name=occurrence.inflation_name,
                    occurrence_labels=occurrence.labels,
                )
            )
            shared_constraints.append(
                MarginalConstraintDraft(
                    variable_name=variable.name,
                    keep_positions=recipe.keep_positions,
                    traced_positions=recipe.traced_positions,
                    target_lexorder=recipe.target_lexorder,
                    output_dim=recipe.output_dim,
                    equation_count=recipe.equation_count,
                    terms_per_equation=recipe.terms_per_equation,
                    einsum_spec=recipe.einsum_spec,
                    named_einsum_spec=recipe.named_einsum_spec,
                    input_tensor_shape=recipe.input_tensor_shape,
                        output_tensor_shape=recipe.output_tensor_shape,
                )
            )
    _progress_log(
        real_verbose,
        1,
        "Draft step 2/3 complete: "
        f"{len(shared_representatives)} shared representatives "
        f"({len(auxiliary_representatives)} auxiliary, {len(known_representatives)} known), "
        f"{len(representative_constraints)} representative equalities, "
        f"{len(ppt_variables)} total PPT auxiliaries "
        f"in {perf_counter() - step_start:.2f}s.",
    )

    # Step 3: some known marginals may not be cross-inflation shared. We still
    # want them as canonical known targets, because the solver-facing layer
    # should treat "known because fixed inside one inflation" and "known because
    # shared across inflations" in the same way.
    step_start = perf_counter()
    _progress_log(real_verbose, 1, "Draft step 3/3: registering fixed-only known marginals...")
    fixed_only_known_classes: Dict[Tuple, List] = {}
    for variable in blueprint.variables:
        for occurrences in variable.fixed_known_marginals.values():
            for occurrence in occurrences:
                if occurrence.signature in known_representatives_by_signature:
                    continue
                fixed_only_known_classes.setdefault(occurrence.signature, []).append(occurrence)

    for known_index, (signature, occurrences) in enumerate(sorted(
        fixed_only_known_classes.items(),
        key=lambda item: min(
            (occ.inflation_index, occ.positions, occ.labels)
            for occ in item[1]
        ),
    )):
        unique_occurrences = tuple(sorted(
            set(occurrences),
            key=lambda occ: (occ.inflation_index, occ.positions, occ.labels),
        ))
        representative_occurrence = unique_occurrences[0]
        representative_slot_dims = tuple(
            blueprint.party_dims[label.split("_", 1)[0]]
            for label in representative_occurrence.labels
        )
        representative = SharedMarginalRepresentativeDraft(
            name=f"nu_{known_index}",
            signature=signature,
            representative_occurrence=representative_occurrence,
            target_lexorder=representative_occurrence.labels,
            slot_dims=representative_slot_dims,
            matrix_dim=product_dim(representative_slot_dims),
            factorization=representative_occurrence.factorization,
            kind="known_matrix",
            is_fixed_known=True,
            occurrence_count=len(unique_occurrences),
        )
        known_representatives.append(representative)
        known_representatives_by_signature[signature] = representative
        for occurrence in unique_occurrences:
            representative_links.append(
                RepresentativeLinkDraft(
                    representative_name=representative.name,
                    representative_kind="known_matrix",
                    occurrence=occurrence,
                )
            )
            variable = blueprint.variables[occurrence.inflation_index]
            recipe = partial_trace_recipe(
                variable.lexorder,
                variable.slot_dims,
                occurrence.positions,
            )
            representative_constraints.append(
                RepresentativeConstraintDraft(
                    source_variable_name=variable.name,
                    source_variable_kind="tau",
                    source_lexorder=variable.lexorder,
                    representative_name=representative.name,
                    representative_kind="known_matrix",
                    representative_lexorder=representative.target_lexorder,
                    keep_positions=recipe.keep_positions,
                    traced_positions=recipe.traced_positions,
                    named_einsum_spec=recipe.named_einsum_spec,
                    symbolic_einsum_spec=recipe.einsum_spec,
                    occurrence_inflation_name=occurrence.inflation_name,
                    occurrence_labels=occurrence.labels,
                )
            )
    _progress_log(
        real_verbose,
        1,
        "Draft step 3/3 complete: "
        f"{len(known_representatives)} known representatives total, "
        f"{len(representative_constraints)} representative equalities total "
        f"in {perf_counter() - step_start:.2f}s.",
    )

    draft = StateSDPDraft(
        blueprint=blueprint,
        party_dims=blueprint.party_dims,
        verbose=real_verbose,
        psd_variables=tuple(psd_variables),
        auxiliary_representatives=tuple(auxiliary_representatives),
        known_representatives=tuple(known_representatives),
        shared_representatives=tuple(shared_representatives),
        internal_symmetry_constraints=tuple(internal_symmetry_constraints),
        representative_links=tuple(representative_links),
        representative_constraints=tuple(representative_constraints),
        ppt_variables=tuple(ppt_variables),
        ppt_constraints=tuple(ppt_constraints),
        fixed_marginal_constraints=tuple(fixed_constraints),
        shared_marginal_constraints=tuple(shared_constraints),
    )
    _progress_log(
        real_verbose,
        1,
        "GNME SDP draft ready: "
        f"{len(draft.psd_variables)} full variables, "
        f"{len(draft.auxiliary_representatives)} auxiliary representatives, "
        f"{len(draft.known_representatives)} known representatives, "
        f"{len(draft.ppt_variables)} PPT variables.",
    )
    return draft


# ---------------------------------------------------------------------------
# Known-value assignment
# ---------------------------------------------------------------------------

def _known_representative_lookup(
    model: StateSDPDraft,
) -> Tuple[Dict[str, SharedMarginalRepresentativeDraft], Dict[Tuple[str, ...], SharedMarginalRepresentativeDraft]]:
    """Index known representatives by name and lexorder target."""
    by_name = {rep.name: rep for rep in model.known_representatives}
    by_target = {rep.target_lexorder: rep for rep in model.known_representatives}
    return by_name, by_target


def _resolve_known_value_key(
    model: StateSDPDraft,
    key,
) -> SharedMarginalRepresentativeDraft:
    """Resolve a human-readable key to a known representative.

    Accepted keys:
    - representative name, e.g. ``mu_2``
    - tuple/list of labels, e.g. ``(\"A_11\", \"B_11\")``
    - space-separated string of labels, e.g. ``\"A_11 B_11\"``
    """
    by_name, by_target = _known_representative_lookup(model)
    if isinstance(key, str):
        if key in by_name:
            return by_name[key]
        parsed = tuple(token for token in key.split() if token)
        if parsed in by_target:
            return by_target[parsed]
    elif isinstance(key, (tuple, list)):
        parsed = tuple(map(str, key))
        if parsed in by_target:
            return by_target[parsed]
    raise KeyError(f"Unknown known representative key: {key!r}")


def set_values(
    model: StateSDPDraft,
    values: Dict,
) -> AssignedStateSDPDraft:
    """Assign numeric matrices to known representatives.

    This mirrors the role of ``set_values`` in the inflation package, but at the
    GNME reduced-state level. Only representatives tagged ``known_matrix`` can
    be assigned here.
    """
    assignments: Dict[str, KnownValueAssignmentDraft] = {}
    for key, value in values.items():
        representative = _resolve_known_value_key(model, key)
        matrix = np.asarray(value)
        expected_shape = (representative.matrix_dim, representative.matrix_dim)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Matrix for {representative.name} has shape {matrix.shape}, "
                f"expected {expected_shape}."
            )
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError(
                f"Matrix for {representative.name} must be Hermitian."
            )
        assignments[representative.name] = KnownValueAssignmentDraft(
            representative_name=representative.name,
            target_lexorder=representative.target_lexorder,
            matrix=matrix,
            matrix_dim=representative.matrix_dim,
        )

    missing = [
        representative.name
        for representative in model.known_representatives
        if representative.name not in assignments
    ]
    if missing:
        raise ValueError(
            "Missing values for known representatives: " + ", ".join(missing)
        )
    return AssignedStateSDPDraft(model=model, known_values=assignments)


# ---------------------------------------------------------------------------
# MOSEK Fusion model construction
# ---------------------------------------------------------------------------

def build_legacy_fusion_feasibility_model(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    model_name: str = "GNMEStateSDP",
    include_ppt: bool = True,
    include_internal_symmetry: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = True,
    verbose: int | None = None,
):
    """Instantiate the legacy entrywise GNME draft as a MOSEK Fusion model.

    The current implementation targets real symmetric state variables. This is
    enough for the GHZ sanity example and keeps the first solver layer close to
    the draft structure.
    """
    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

    if isinstance(assigned, AssignedStateSDPDraft):
        model = assigned.model
        known_values = assigned.known_values
    else:
        model = assigned
        known_values = {}
    real_verbose = _resolve_verbose(verbose, model.verbose)
    total_start = perf_counter()
    _progress_log(real_verbose, 1, "Building MOSEK Fusion feasibility model...")
    M = Model(model_name)

    # Step 1: create one free state object for every PSD component in the draft.
    # Full tau variables are symmetry-reduced when internal symmetry is enabled:
    # we keep one scalar variable per orbit of matrix entries and lift those
    # orbit variables to an affine PSD matrix expression.
    step_start = perf_counter()
    tau_scalar_variables = {}
    tau_variables = {}
    total_upper_entries = 0
    total_orbit_entries = 0
    for variable in model.psd_variables:
        upper_entries = variable.matrix_dim * (variable.matrix_dim + 1) // 2
        total_upper_entries += upper_entries
        if include_internal_symmetry:
            orbit_data = symmetric_matrix_orbits(
                variable.slot_dims,
                variable.local_symmetry_perms,
            )
            scalar_variables, matrix_expression = affine_symmetric_psd_matrix(
                M,
                variable.name,
                variable.matrix_dim,
                orbit_data,
                Domain,
                Expr,
                Matrix,
            )
            tau_scalar_variables[variable.name] = scalar_variables
            tau_variables[variable.name] = matrix_expression
            total_orbit_entries += len(orbit_data.orbit_representatives)
        else:
            tau_variable = M.variable(variable.name, Domain.inPSDCone(variable.matrix_dim))
            tau_scalar_variables[variable.name] = tau_variable
            tau_variables[variable.name] = tau_variable
            total_orbit_entries += upper_entries
    auxiliary_variables = {
        representative.name: M.variable(
            representative.name,
            Domain.inPSDCone(representative.matrix_dim),
        )
        for representative in model.auxiliary_representatives
    }
    known_representative_variables = (
        {
            representative.name: M.variable(
                representative.name,
                Domain.inPSDCone(representative.matrix_dim),
            )
            for representative in model.known_representatives
        }
        if not enforce_known_values
        else {}
    )
    ppt_variables = (
        {
            ppt_variable.name: M.variable(
                ppt_variable.name,
                Domain.inPSDCone(ppt_variable.matrix_dim),
            )
            for ppt_variable in model.ppt_variables
        }
        if include_ppt
        else {}
    )
    _progress_log(
        real_verbose,
        1,
        "Fusion step 1/5 complete: "
        f"{len(tau_variables)} tau, "
        f"{len(auxiliary_variables)} auxiliary, "
        f"{len(known_representative_variables)} free known-representative, "
        f"{len(ppt_variables)} PPT variables, "
        f"{total_orbit_entries}/{total_upper_entries} tau entry variables kept "
        f"in {perf_counter() - step_start:.2f}s.",
    )

    constraint_counts = {
        "trace": 0,
        "internal_symmetry": 0,
        "symmetry_reduction_orbits": total_orbit_entries,
        "representative": 0,
        "ppt": 0,
    }

    # Step 2: density matrices are normalized. Every free state variable,
    # whether full or reduced, gets `trace = 1`.
    step_start = perf_counter()
    for variable in model.psd_variables:
        M.constraint(
            f"trace_{variable.name}",
            _trace_expr(tau_variables[variable.name], variable.matrix_dim, Expr),
            Domain.equalsTo(1.0),
        )
        constraint_counts["trace"] += 1

    for representative in model.auxiliary_representatives:
        M.constraint(
            f"trace_{representative.name}",
            _trace_expr(auxiliary_variables[representative.name], representative.matrix_dim, Expr),
            Domain.equalsTo(1.0),
        )
        constraint_counts["trace"] += 1

    if not enforce_known_values:
        for representative in model.known_representatives:
            M.constraint(
                f"trace_{representative.name}",
                _trace_expr(
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
        "Fusion step 2/5 complete: "
        f"{constraint_counts['trace']} trace constraints "
        f"in {perf_counter() - step_start:.2f}s.",
    )

    # Step 3: if symmetry reduction is enabled, the tau variables already live
    # in the invariant subspace. Otherwise we fall back to explicit
    # equalities `X = P X P^T`.
    tau_variable_specs = {variable.name: variable for variable in model.psd_variables}
    if include_internal_symmetry:
        _progress_log(
            real_verbose,
            1,
            "Fusion step 3/5 complete: tau variables built directly in the symmetry-invariant subspace; no explicit symmetry equalities added.",
        )
    else:
        step_start = perf_counter()
        _progress_log(
            real_verbose,
            1,
            f"Fusion step 3/5: applying {len(model.internal_symmetry_constraints)} internal symmetry generators...",
        )
        symmetry_constraint_index = 0
        for constraint_index, constraint in enumerate(model.internal_symmetry_constraints, start=1):
            local_start = perf_counter()
            local_count = 0
            _progress_log(
                real_verbose,
                2,
                f"  symmetry {constraint_index}/{len(model.internal_symmetry_constraints)} on {constraint.variable_name}",
            )
            variable = tau_variables[constraint.variable_name]
            variable_spec = tau_variable_specs[constraint.variable_name]
            basis_map = _basis_permutation_map(variable_spec.slot_dims, constraint.permutation)
            for row in range(variable_spec.matrix_dim):
                mapped_row = basis_map[row]
                for col in range(row, variable_spec.matrix_dim):
                    mapped_col = basis_map[col]
                    lhs = _symmetric_entry(variable, row, col)
                    rhs = _symmetric_entry(variable, mapped_row, mapped_col)
                    if (row, col) == tuple(sorted((mapped_row, mapped_col))):
                        continue
                    M.constraint(
                        f"sym_{symmetry_constraint_index}",
                        Expr.sub(lhs, rhs),
                        Domain.equalsTo(0.0),
                    )
                    symmetry_constraint_index += 1
                    constraint_counts["internal_symmetry"] += 1
                    local_count += 1
            _progress_log(
                real_verbose,
                2,
                f"    emitted {local_count} scalar symmetry constraints in {perf_counter() - local_start:.2f}s.",
            )
        _progress_log(
            real_verbose,
            1,
            "Fusion step 3/5 complete: "
            f"{constraint_counts['internal_symmetry']} scalar symmetry constraints "
            f"in {perf_counter() - step_start:.2f}s.",
        )

    # Step 4: tie all reduced marginals to their canonical representatives.
    # This is the cross-inflation glue of the model.
    if include_representatives:
        step_start = perf_counter()
        representative_constraints = quotient_representative_constraints(
            model,
            use_source_symmetry=include_internal_symmetry,
        )
        _progress_log(
            real_verbose,
            1,
            f"Fusion step 4/5: adding {len(representative_constraints)} representative marginal blocks...",
        )
        representative_constraint_index = 0
        stride = _progress_stride(len(representative_constraints))
        for constraint_index, constraint in enumerate(representative_constraints, start=1):
            if real_verbose >= 2 and (
                constraint_index == 1
                or constraint_index % stride == 0
                or constraint_index == len(representative_constraints)
            ):
                _progress_log(
                    real_verbose,
                    2,
                    f"  representative block {constraint_index}/{len(representative_constraints)} "
                    f"from {constraint.source_variable_name} to {constraint.representative_name}",
                )
            source_spec = tau_variable_specs[constraint.source_variable_name]
            source_var = tau_variables[constraint.source_variable_name]
            kept_dims = tuple(source_spec.slot_dims[pos] for pos in constraint.keep_positions)

            if constraint.representative_kind == "known_matrix":
                if enforce_known_values:
                    known_assignment = known_values[constraint.representative_name]
                    if np.max(np.abs(np.imag(known_assignment.matrix))) > 1e-9:
                        raise ValueError(
                            f"Known matrix {constraint.representative_name} is not real; "
                            "the current Fusion draft only supports real symmetric data."
                        )
                    known_matrix = np.real_if_close(known_assignment.matrix, tol=1e5)
                    rhs_value = lambda r, c: float(known_matrix[r, c])
                    rhs_var = None
                else:
                    rhs_var = known_representative_variables[constraint.representative_name]
                    rhs_value = None
            else:
                rhs_var = auxiliary_variables[constraint.representative_name]
                rhs_value = None

            kept_ranges = [range(dim) for dim in kept_dims]
            traced_ranges = [
                range(source_spec.slot_dims[pos]) for pos in constraint.traced_positions
            ]
            for bra_keep in product(*kept_ranges):
                row = _flat_index(tuple(int(x) for x in bra_keep), kept_dims)
                for ket_keep in product(*kept_ranges):
                    col = _flat_index(tuple(int(x) for x in ket_keep), kept_dims)
                    if row > col:
                        continue
                    entry_terms = []
                    traced_iterator = product(*traced_ranges) if traced_ranges else [tuple()]
                    for traced_values in traced_iterator:
                        bra_full = [0] * len(source_spec.slot_dims)
                        ket_full = [0] * len(source_spec.slot_dims)
                        for pos, value in zip(constraint.keep_positions, bra_keep):
                            bra_full[pos] = int(value)
                        for pos, value in zip(constraint.keep_positions, ket_keep):
                            ket_full[pos] = int(value)
                        for pos, value in zip(constraint.traced_positions, traced_values):
                            bra_full[pos] = int(value)
                            ket_full[pos] = int(value)
                        full_row = _flat_index(tuple(bra_full), source_spec.slot_dims)
                        full_col = _flat_index(tuple(ket_full), source_spec.slot_dims)
                        entry_terms.append(_symmetric_entry(source_var, full_row, full_col))
                    lhs = _sum_expr(entry_terms, Expr)
                    rhs = rhs_value(row, col) if rhs_var is None else _symmetric_entry(rhs_var, row, col)
                    M.constraint(
                        f"rep_{representative_constraint_index}",
                        Expr.sub(lhs, rhs),
                        Domain.equalsTo(0.0),
                    )
                    representative_constraint_index += 1
                    constraint_counts["representative"] += 1
        _progress_log(
            real_verbose,
            1,
            "Fusion step 4/5 complete: "
            f"{constraint_counts['representative']} scalar representative constraints "
            f"in {perf_counter() - step_start:.2f}s.",
        )

    # Step 5: PPT constraints. For each requested partition we create a PSD
    # auxiliary variable and identify it with the partial transpose of the
    # source state.
    if include_ppt:
        step_start = perf_counter()
        _progress_log(
            real_verbose,
            1,
            f"Fusion step 5/5: adding {len(model.ppt_constraints)} PPT blocks...",
        )
        source_free_variable_lookup = {}
        source_free_variable_lookup.update(tau_variables)
        source_free_variable_lookup.update(auxiliary_variables)
        ppt_constraint_index = 0
        stride = _progress_stride(len(model.ppt_constraints))
        for constraint_index, constraint in enumerate(model.ppt_constraints, start=1):
            if real_verbose >= 2 and (
                constraint_index == 1
                or constraint_index % stride == 0
                or constraint_index == len(model.ppt_constraints)
            ):
                _progress_log(
                    real_verbose,
                    2,
                    f"  PPT block {constraint_index}/{len(model.ppt_constraints)} "
                    f"for {constraint.source_variable_name}",
                )
            source_var = source_free_variable_lookup[constraint.source_variable_name]
            ppt_var = ppt_variables[constraint.ppt_variable_name]
            for row in range(constraint.matrix_dim):
                for col in range(row, constraint.matrix_dim):
                    mapped_row, mapped_col = _partial_transpose_entry_map(
                        row,
                        col,
                        constraint.slot_dims,
                        constraint.transpose_positions,
                    )
                    lhs = _symmetric_entry(ppt_var, row, col)
                    rhs = _symmetric_entry(source_var, mapped_row, mapped_col)
                    M.constraint(
                        f"ppt_{ppt_constraint_index}",
                        Expr.sub(lhs, rhs),
                        Domain.equalsTo(0.0),
                    )
                    ppt_constraint_index += 1
                    constraint_counts["ppt"] += 1
        _progress_log(
            real_verbose,
            1,
            "Fusion step 5/5 complete: "
            f"{constraint_counts['ppt']} scalar PPT constraints "
            f"in {perf_counter() - step_start:.2f}s.",
        )

    # Step 6: pure feasibility model, zero objective.
    M.objective(ObjectiveSense.Minimize, 0.0)
    _progress_log(
        real_verbose,
        1,
        "Fusion model ready in "
        f"{perf_counter() - total_start:.2f}s with constraint counts {constraint_counts}.",
    )

    return FusionStateSDPModel(
        model=M,
        tau_scalar_variables=tau_scalar_variables,
        tau_variables=tau_variables,
        auxiliary_variables=auxiliary_variables,
        known_representative_variables=known_representative_variables,
        ppt_variables=ppt_variables,
        constraint_counts=constraint_counts,
    )


def solve_legacy_fusion_feasibility(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    solver_parameters: Dict[str, object] | None = None,
    verbose: int = -1,
    include_ppt: bool = True,
    include_internal_symmetry: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = True,
) -> FusionSolveResult:
    """Build and solve the legacy entrywise GNME feasibility model.

    The toggles are intentionally exposed because they are useful for debugging:
    we can switch off one logical block at a time and isolate the source of an
    infeasibility.
    """
    from mosek.fusion import AccSolutionStatus

    base_model = assigned.model if isinstance(assigned, AssignedStateSDPDraft) else assigned
    real_verbose = _resolve_verbose(verbose, base_model.verbose)
    total_start = perf_counter()
    fusion_model = build_legacy_fusion_feasibility_model(
        assigned,
        include_ppt=include_ppt,
        include_internal_symmetry=include_internal_symmetry,
        include_representatives=include_representatives,
        enforce_known_values=enforce_known_values,
        verbose=real_verbose,
    )
    M = fusion_model.model
    try:
        if real_verbose > 0:
            _progress_log(real_verbose, 1, "Starting MOSEK solve...")
            M.setLogHandler(sys.stdout)
        if solver_parameters:
            for param, value in solver_parameters.items():
                M.setSolverParam(param, value)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        solve_start = perf_counter()
        M.solve()
        _progress_log(
            real_verbose,
            1,
            "MOSEK solve finished in "
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
            "Overall feasibility pipeline finished in "
            f"{perf_counter() - total_start:.2f}s. "
            f"Status={result.problem_status}.",
        )
        return result
    finally:
        M.dispose()


def build_fusion_feasibility_model(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    model_name: str = "GNMEStateSDP",
    include_ppt: bool = True,
    include_internal_symmetry: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = True,
    Hermitian: bool = False,
    verbose: int | None = None,
):
    """Instantiate the default GNME Fusion model.

    The package default now uses the symmetry-adapted block backend. The
    legacy entrywise backend remains available explicitly through
    `build_legacy_fusion_feasibility_model(...)`.

    When `include_internal_symmetry=False`, we fall back to the legacy backend,
    since the block backend builds the symmetry into the variable declaration.
    """
    if not include_internal_symmetry:
        return build_legacy_fusion_feasibility_model(
            assigned,
            model_name=model_name,
            include_ppt=include_ppt,
            include_internal_symmetry=include_internal_symmetry,
            include_representatives=include_representatives,
            enforce_known_values=enforce_known_values,
            verbose=verbose,
        )

    from .GNMEBlockStateSDP import build_block_fusion_feasibility_model

    return build_block_fusion_feasibility_model(
        assigned,
        model_name=model_name,
        include_ppt=include_ppt,
        include_representatives=include_representatives,
        enforce_known_values=enforce_known_values,
        Hermitian=Hermitian,
        verbose=verbose,
    )


def solve_fusion_feasibility(
    assigned: AssignedStateSDPDraft | StateSDPDraft,
    solver_parameters: Dict[str, object] | None = None,
    verbose: int = -1,
    include_ppt: bool = True,
    include_internal_symmetry: bool = True,
    include_representatives: bool = True,
    enforce_known_values: bool = True,
    Hermitian: bool = False,
) -> FusionSolveResult:
    """Build and solve the default GNME Fusion model.

    The default path uses the block backend. The legacy entrywise solver path
    remains available through `solve_legacy_fusion_feasibility(...)`.
    """
    if not include_internal_symmetry:
        return solve_legacy_fusion_feasibility(
            assigned,
            solver_parameters=solver_parameters,
            verbose=verbose,
            include_ppt=include_ppt,
            include_internal_symmetry=include_internal_symmetry,
            include_representatives=include_representatives,
            enforce_known_values=enforce_known_values,
        )

    from .GNMEBlockStateSDP import solve_block_fusion_feasibility

    return solve_block_fusion_feasibility(
        assigned,
        solver_parameters=solver_parameters,
        verbose=verbose,
        include_ppt=include_ppt,
        include_representatives=include_representatives,
        enforce_known_values=enforce_known_values,
        Hermitian=Hermitian,
    )


# ---------------------------------------------------------------------------
# Human-readable inspection helpers
# ---------------------------------------------------------------------------

def print_assigned_values(assigned: AssignedStateSDPDraft) -> None:
    """Print the value assignments in a compact form."""
    print("Assigned Known Representative Values")
    for representative in assigned.model.known_representatives:
        assignment = assigned.known_values[representative.name]
        print(
            "  "
            f"{representative.name}: "
            f"target={' '.join(assignment.target_lexorder)} "
            f"shape={assignment.matrix.shape}"
        )


def print_draft(model: StateSDPDraft) -> None:
    """Print the SDP draft in a compact, inspection-oriented format."""
    print("Party dimensions")
    print(f"  {model.party_dims}")
    print()

    print("Draft SDP Variables")
    for variable in model.psd_variables:
        print(f"{variable.name}:")
        print(f"  lexorder: {' '.join(variable.lexorder)}")
        print(f"  slot dims: {variable.slot_dims}")
        print(f"  PSD shape: ({variable.matrix_dim}, {variable.matrix_dim})")
        print(f"  local symmetries: {variable.local_symmetry_perms}")
        print(f"  factors: {variable.factorization}")
        print("  fixed known marginals:")
        for parties, occurrences in variable.fixed_known_marginals.items():
            print(f"    {parties}:")
            for occurrence in occurrences:
                print(
                    "      "
                    f"positions={occurrence.positions} "
                    f"labels={' '.join(occurrence.labels)}"
                )
        print()

    print("Auxiliary Shared-Marginal Representatives")
    for representative in model.auxiliary_representatives:
        print(
            "  "
            f"{representative.name}: "
            f"target={' '.join(representative.target_lexorder)} "
            f"shape=({representative.matrix_dim}, {representative.matrix_dim}) "
            f"{representative.kind} "
            f"occurrences={representative.occurrence_count}"
        )
    print()

    print("Known Representative Marginals")
    for representative in model.known_representatives:
        print(
            "  "
            f"{representative.name}: "
            f"target={' '.join(representative.target_lexorder)} "
            f"shape=({representative.matrix_dim}, {representative.matrix_dim}) "
            f"{representative.kind} "
            f"occurrences={representative.occurrence_count}"
        )
    print()

    print("Internal Symmetry Constraints")
    for constraint in model.internal_symmetry_constraints:
        print(
            "  "
            f"{constraint.variable_name} = Perm({constraint.variable_name}) "
            f"[{constraint.variable_kind}] "
            f"perm={constraint.permutation}"
        )
        print(f"    action: {constraint.named_action}")
        print(f"    permuted lexorder: {' '.join(constraint.permuted_lexorder)}")
    print()

    print("Representative Assignments")
    for link in model.representative_links:
        print(
            "  "
            f"{link.representative_name} ({link.representative_kind}) <- "
            f"{link.occurrence.inflation_name} "
            f"{' '.join(link.occurrence.labels)}"
        )
    print()

    print("Representative Equality Constraints")
    for constraint in model.representative_constraints:
        print(
            "  "
            f"{constraint.source_variable_name} -> {constraint.representative_name} "
            f"[{constraint.representative_kind}] "
            f"keep={constraint.keep_positions} "
            f"target={' '.join(constraint.representative_lexorder)}"
        )
        print(
            "    "
            f"occurrence={constraint.occurrence_inflation_name} "
            f"{' '.join(constraint.occurrence_labels)}"
        )
        print(f"    named einsum: {constraint.named_einsum_spec}")
        print(f"    symbolic einsum: {constraint.symbolic_einsum_spec}")
    print()

    print("PPT Auxiliary Variables")
    for ppt_variable in model.ppt_variables:
        print(
            "  "
            f"{ppt_variable.name}: source={ppt_variable.source_variable_name} "
            f"transpose={' '.join(ppt_variable.transpose_lexorder)} | "
            f"complement={' '.join(ppt_variable.complement_lexorder)} "
            f"shape=({ppt_variable.matrix_dim}, {ppt_variable.matrix_dim})"
        )
    print()

    print("PPT Constraints")
    for candidate in model.ppt_constraints:
        print(
            "  "
            f"{candidate.ppt_variable_name} = PT({candidate.source_variable_name}) "
            f"[{candidate.source_variable_kind}] "
            f"transpose={' '.join(candidate.transpose_lexorder)} | "
            f"complement={' '.join(candidate.complement_lexorder)}"
        )
        print(f"    factorization: {candidate.factorization}")
        print(f"    named transpose: {candidate.named_einsum_spec}")
        print(f"    symbolic transpose: {candidate.symbolic_einsum_spec}")
    print()

    print("Fixed Marginalization Rules")
    for constraint in model.fixed_marginal_constraints:
        print(
            "  "
            f"{constraint.variable_name} keep={constraint.keep_positions} "
            f"trace={constraint.traced_positions} "
            f"target={' '.join(constraint.target_lexorder)} "
            f"reduced_dim={constraint.output_dim} "
            f"eqs={constraint.equation_count} "
            f"terms/eq={constraint.terms_per_equation}"
        )
        print(
            "    "
            f"named einsum: {constraint.named_einsum_spec}"
        )
        print(
            "    "
            f"symbolic einsum: {constraint.einsum_spec} "
            f"{constraint.input_tensor_shape} -> {constraint.output_tensor_shape}"
        )
    print()

    print("Candidate Cross-Inflation Equality Classes")
    for subset_class in model.blueprint.shared_subset_classes:
        print(f"signature={subset_class.signature}")
        for occurrence in subset_class.occurrences:
            print(
                "  "
                f"{occurrence.inflation_name} "
                f"positions={occurrence.positions} "
                f"labels={' '.join(occurrence.labels)}"
            )
    print()

    if model.psd_variables:
        variable = model.psd_variables[0]
        if variable.fixed_known_marginals:
            first_occurrence = next(iter(next(iter(variable.fixed_known_marginals.values()))))
            print("Sample Partial-Trace Terms")
            print(
                "  "
                f"{variable.name} keep={first_occurrence.positions} "
                f"labels={' '.join(first_occurrence.labels)}"
            )
            named_spec, target_lexorder = partial_trace_named_einsum_spec(
                variable.lexorder,
                first_occurrence.positions,
            )
            spec, input_shape, output_shape = partial_trace_einsum_spec(
                variable.slot_dims,
                first_occurrence.positions,
            )
            print(f"    target: {' '.join(target_lexorder)}")
            print(f"    named einsum: {named_spec}")
            print(f"    symbolic einsum: {spec} {input_shape} -> {output_shape}")
            examples = partial_trace_terms(variable.slot_dims, first_occurrence.positions)
            for equation_id, terms in enumerate(examples):
                print(f"    reduced entry {equation_id}:")
                for bra, ket in terms:
                    print(f"      X[{bra}, {ket}]")


__all__ = [
    "AssignedStateSDPDraft",
    "FusionSolveResult",
    "FusionStateSDPModel",
    "InternalSymmetryConstraintDraft",
    "KnownValueAssignmentDraft",
    "MarginalConstraintDraft",
    "PPTConstraintDraft",
    "PPTVariableDraft",
    "PSDVariableDraft",
    "RepresentativeConstraintDraft",
    "RepresentativeLinkDraft",
    "SharedMarginalRepresentativeDraft",
    "StateSDPDraft",
    "build_fusion_feasibility_model",
    "build_legacy_fusion_feasibility_model",
    "build_sdp_draft",
    "partial_trace_einsum_spec",
    "partial_trace_named_einsum_spec",
    "partial_trace_recipe",
    "print_assigned_values",
    "print_draft",
    "quotient_representative_constraints",
    "set_values",
    "solve_fusion_feasibility",
    "solve_legacy_fusion_feasibility",
]
