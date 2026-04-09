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

import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..GNMEProblem import GNMESDPBlueprint, GNMEProblem, GNMETopDownBlueprint, GNMETopDownFamily


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
    target_slot_permutation: Tuple[int, ...] | None
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


# ---------------------------------------------------------------------------
# Phase 1-3: Top-down draft enhancement dataclasses for anchor grouping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossInflationConstraintGroupDraft:
    """Canonical anchor + non-anchor tau routes for one representative."""
    
    representative_name: str
    anchor_constraint: RepresentativeConstraintDraft
    non_anchor_constraints: Tuple[RepresentativeConstraintDraft, ...]


@dataclass(frozen=True)
class TopDownStateSDPDraft:
    """Fresh top-down GNME draft driven by maximal family metadata.

    This object is solver-agnostic. The default supported anchored scope is:

    - tau PSD variables,
    - maximal shared families,
    - fixed known families,
    - tau-family anchor groups,
    - PPT candidates (tau always, family PPT optional at build time).

    Overlap-family instantiation is intentionally left out of the anchored
    default path until the hierarchy is implemented cleanly end-to-end.
    """

    family_blueprint: GNMETopDownBlueprint
    party_dims: Dict[str, int]
    verbose: int
    psd_variables: Tuple[PSDVariableDraft, ...]
    internal_symmetry_constraints: Tuple[InternalSymmetryConstraintDraft, ...]
    maximal_representatives: Tuple["SharedMarginalRepresentativeDraft", ...]
    known_representatives: Tuple["SharedMarginalRepresentativeDraft", ...]
    representative_links: Tuple["RepresentativeLinkDraft", ...]
    tau_representative_constraints: Tuple[RepresentativeConstraintDraft, ...]
    ppt_variables: Tuple["PPTVariableDraft", ...]
    ppt_constraints: Tuple["PPTConstraintDraft", ...]
    fixed_marginal_constraints: Tuple[MarginalConstraintDraft, ...]
    cross_inflation_groups: Tuple[CrossInflationConstraintGroupDraft, ...] = tuple()
    verified_at_draft_time: bool = False


@dataclass(frozen=True)
class PaperMarginalViewDraft:
    """One direct submarginal view of an inflated paper-style state."""

    source_variable_name: str
    source_variable_kind: str
    source_lexorder: Tuple[str, ...]
    keep_positions: Tuple[int, ...]
    traced_positions: Tuple[int, ...]
    target_lexorder: Tuple[str, ...]
    target_slot_permutation: Tuple[int, ...] | None
    slot_dims: Tuple[int, ...]
    matrix_dim: int
    factorization: Tuple[Tuple[str, ...], ...]
    occurrence_labels: Tuple[str, ...]


@dataclass(frozen=True)
class PaperObservedConstraintDraft:
    """One direct observed marginal constraint in the paper-style draft."""

    name: str
    target_lexorder: Tuple[str, ...]
    marginal_view: PaperMarginalViewDraft


@dataclass(frozen=True)
class PaperEqualityConstraintDraft:
    """One direct marginal equality between two inflated states."""

    name: str
    lhs_view: PaperMarginalViewDraft
    rhs_view: PaperMarginalViewDraft


@dataclass(frozen=True)
class PaperPPTConstraintDraft:
    """One direct PPT constraint on a full state or selected submarginal."""

    name: str
    marginal_view: PaperMarginalViewDraft
    transpose_positions: Tuple[int, ...]
    complement_positions: Tuple[int, ...]
    transpose_lexorder: Tuple[str, ...]
    complement_lexorder: Tuple[str, ...]


@dataclass(frozen=True)
class PaperStateSDPDraft:
    """Tripartite paper-style inflation draft for levels 2 and 3.

    This draft follows the appendix formulation directly:

    - a small number of inflated tau variables,
    - explicit named marginal anchors/equalities,
    - only the PPT conditions listed in the paper.
    """

    formulation_name: str
    inflation_level: int
    party_dims: Dict[str, int]
    verbose: int
    psd_variables: Tuple[PSDVariableDraft, ...]
    internal_symmetry_constraints: Tuple[InternalSymmetryConstraintDraft, ...]
    observed_constraints: Tuple["PaperObservedConstraintDraft", ...]
    equality_constraints: Tuple["PaperEqualityConstraintDraft", ...]
    ppt_constraints: Tuple["PaperPPTConstraintDraft", ...]
    fixed_marginal_constraints: Tuple[MarginalConstraintDraft, ...] = tuple()
    verified_at_draft_time: bool = False


@dataclass(frozen=True)
class AssignedStateSDPDraft:
    """State SDP draft with matrices attached to known representatives."""

    model: StateSDPDraft | TopDownStateSDPDraft | PaperStateSDPDraft
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


@lru_cache(maxsize=None)
def _cached_upper_triangle_pairs(matrix_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row-major upper-triangle `(row, col)` pairs for one matrix dimension."""
    rows, cols = np.triu_indices(matrix_dim)
    return (
        rows.astype(np.int32, copy=False),
        cols.astype(np.int32, copy=False),
    )


@lru_cache(maxsize=None)
def _cached_unique_basis_maps(
    slot_dims: Tuple[int, ...],
    local_symmetry_perms: Tuple[Tuple[int, ...], ...],
) -> Tuple[np.ndarray, ...]:
    """Unique basis-index permutation maps induced by the slot symmetries."""
    seen = set()
    basis_maps = []
    for perm in local_symmetry_perms:
        basis_map = _basis_permutation_map(slot_dims, perm)
        if basis_map in seen:
            continue
        seen.add(basis_map)
        basis_maps.append(np.asarray(basis_map, dtype=np.int32))
    if not basis_maps:
        dim = product_dim(slot_dims)
        basis_maps.append(np.arange(dim, dtype=np.int32))
    return tuple(basis_maps)


def _upper_triangle_coordinate_indices(
    rows: np.ndarray,
    cols: np.ndarray,
    matrix_dim: int,
) -> np.ndarray:
    """Vectorized row-major upper-triangle coordinate indices."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return (
        rows * int(matrix_dim)
        - rows * (rows - 1) // 2
        + (cols - rows)
    ).astype(np.int32, copy=False)


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


def _quotient_representative_constraint_sequence(
    constraints: Tuple[RepresentativeConstraintDraft, ...] | Iterable[RepresentativeConstraintDraft],
    source_lookup: Dict[str, object],
    use_source_symmetry: bool = True,
) -> Tuple[RepresentativeConstraintDraft, ...]:
    """Quotient a representative-constraint sequence by source symmetry orbits."""
    constraints = tuple(constraints)
    if not use_source_symmetry:
        return constraints

    kept_constraints = []
    seen = set()
    for constraint in constraints:
        source_variable = source_lookup.get(constraint.source_variable_name)
        local_symmetry_perms = getattr(source_variable, "local_symmetry_perms", tuple()) if source_variable is not None else tuple()
        if (
            source_variable is None
            or not local_symmetry_perms
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
                    local_symmetry_perms,
                ),
            )
        if key in seen:
            continue
        seen.add(key)
        kept_constraints.append(constraint)
    return tuple(kept_constraints)


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
    return _quotient_representative_constraint_sequence(
        model.representative_constraints,
        variable_lookup,
        use_source_symmetry=True,
    )


def quotient_ppt_constraints(
    model: StateSDPDraft | TopDownStateSDPDraft,
    use_source_symmetry: bool = True,
    drop_representative_constraints_implied_by_tau: bool = True,
    drop_factor_reducible_representative_ppts: bool = True,
) -> Tuple[Tuple["PPTVariableDraft", ...], Tuple["PPTConstraintDraft", ...]]:
    """Remove PPT constraints duplicated by source-variable symmetries.

    If a tau source already lives in the symmetry-invariant subspace, then for a
    local symmetry permutation ``g`` we have

    ``Gamma_{g(P)}(tau) = Pi_g Gamma_P(tau) Pi_g^T``.

    Positivity is preserved by this permutation congruence, so only one
    transpose pattern per source-symmetry orbit needs an explicit PPT auxiliary
    and equality constraint. We keep the raw list when
    ``use_source_symmetry=False`` because the quotient is only safe once the
    source symmetry has actually been enforced in the SDP variable.

    Optionally, reducible shared representatives (``mu_*`` with more than one
    factor block) can be dropped completely from the PPT layer. This keeps only
    the factor-irreducible representative operators as PPT sources.
    """
    if not use_source_symmetry:
        return model.ppt_variables, model.ppt_constraints

    source_symmetry_lookup = {
        variable.name: variable.local_symmetry_perms
        for variable in model.psd_variables
    }
    if isinstance(model, TopDownStateSDPDraft):
        representatives = model.maximal_representatives + model.known_representatives
        representative_constraints = model.tau_representative_constraints
        anchor_constraint_by_representative = {
            group.representative_name: group.anchor_constraint
            for group in model.cross_inflation_groups
        }
    else:
        representatives = model.auxiliary_representatives + model.known_representatives
        representative_constraints = model.representative_constraints
        anchor_constraint_by_representative = {}

    source_symmetry_lookup.update(
        {
            representative.name: representative.local_symmetry_perms
            for representative in representatives
        }
    )
    representative_lookup = {
        representative.name: representative
        for representative in representatives
    }
    for constraint in representative_constraints:
        anchor_constraint_by_representative.setdefault(
            constraint.representative_name,
            constraint,
        )

    variable_lookup = {variable.name: variable for variable in model.psd_variables}

    def _representative_ppt_implied_by_tau(constraint: PPTConstraintDraft) -> bool:
        representative = representative_lookup.get(constraint.source_variable_name)
        if representative is None:
            return False
        anchor_constraint = anchor_constraint_by_representative.get(representative.name)
        if anchor_constraint is None:
            return False
        source_variable = variable_lookup.get(anchor_constraint.source_variable_name)
        if source_variable is None or len(source_variable.factorization) <= 1:
            return False

        source_factor_lookup = {
            label: factor_index
            for factor_index, factor in enumerate(source_variable.factorization)
            for label in factor
        }
        representative_factor_positions = factor_positions(
            representative.target_lexorder,
            representative.factorization,
        )
        transpose_position_set = set(constraint.transpose_positions)

        source_factor_to_representative_factors: Dict[int, List[int]] = {}
        selected_representative_factors = set()
        for factor_index, (factor_labels, factor_positions_) in enumerate(
            zip(representative.factorization, representative_factor_positions)
        ):
            source_factor_ids = {source_factor_lookup[label] for label in factor_labels}
            if len(source_factor_ids) != 1:
                return False
            source_factor_id = next(iter(source_factor_ids))
            source_factor_to_representative_factors.setdefault(source_factor_id, []).append(factor_index)

            selected_positions = [pos in transpose_position_set for pos in factor_positions_]
            if any(selected_positions):
                if not all(selected_positions):
                    return False
                selected_representative_factors.add(factor_index)

        selected_source_factors = set()
        for source_factor_id, representative_factor_indices in source_factor_to_representative_factors.items():
            picked = [
                factor_index in selected_representative_factors
                for factor_index in representative_factor_indices
            ]
            if any(picked):
                if not all(picked):
                    return False
                selected_source_factors.add(source_factor_id)

        if not selected_source_factors:
            return False
        if len(selected_source_factors) == len(source_variable.factorization):
            return False
        return True

    def _representative_ppt_is_factor_reducible(constraint: PPTConstraintDraft) -> bool:
        representative = representative_lookup.get(constraint.source_variable_name)
        if representative is None:
            return False
        return len(representative.factorization) > 1

    kept_constraints = []
    kept_variable_names = set()
    seen = set()
    for constraint in model.ppt_constraints:
        if (
            drop_factor_reducible_representative_ppts
            and constraint.source_variable_kind == "mu"
            and _representative_ppt_is_factor_reducible(constraint)
        ):
            continue
        if (
            drop_representative_constraints_implied_by_tau
            and constraint.source_variable_kind == "mu"
            and _representative_ppt_implied_by_tau(constraint)
        ):
            continue
        local_symmetry_perms = source_symmetry_lookup.get(constraint.source_variable_name, tuple())
        if not local_symmetry_perms:
            key = (
                constraint.source_variable_name,
                constraint.transpose_positions,
            )
        else:
            key = (
                constraint.source_variable_name,
                _canonical_keep_positions_orbit(
                    constraint.transpose_positions,
                    local_symmetry_perms,
                ),
            )
        if key in seen:
            continue
        seen.add(key)
        kept_constraints.append(constraint)
        kept_variable_names.add(constraint.ppt_variable_name)

    kept_variables = tuple(
        ppt_variable
        for ppt_variable in model.ppt_variables
        if ppt_variable.name in kept_variable_names
    )
    return kept_variables, tuple(kept_constraints)

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
    local_symmetry_perms: Tuple[Tuple[int, ...], ...]
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


_BLUEPRINT_CACHE_VERSION = "gnme_sdp_blueprints_v1"


def _blueprint_cache_dir() -> Path:
    """Directory for persisted GNME blueprint objects.

    Level-3 runs spend substantial time in the inherited nonfanout blueprint
    generation. Caching the structural blueprint is safe because the object is
    purely combinatorial and independent of the later MOSEK model.
    """
    custom_dir = os.environ.get("GNME_BLUEPRINT_CACHE_DIR")
    if custom_dir:
        root = Path(custom_dir).expanduser()
    else:
        root = Path.home() / ".cache" / "gnme_inflation"
    cache_dir = root / _BLUEPRINT_CACHE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _blueprint_cache_path(cache_key: object) -> Path:
    payload = pickle.dumps((_BLUEPRINT_CACHE_VERSION, cache_key), protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    return _blueprint_cache_dir() / f"{digest}.pkl"


def _load_cached_blueprint(cache_key: object):
    """Load a persisted combinatorial blueprint, if present."""
    path = _blueprint_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _save_cached_blueprint(cache_key: object, blueprint) -> None:
    """Persist a combinatorial blueprint for reuse across identical runs."""
    path = _blueprint_cache_path(cache_key)
    temp_path = path.with_suffix(".tmp")
    try:
        with temp_path.open("wb") as handle:
            pickle.dump(blueprint, handle, protocol=5)
        temp_path.replace(path)
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


def _blueprint_cache_key(
    problem: GNMEProblem,
    subset_sizes: Tuple[int, ...],
    local_dims_per_party: Dict[str, int] | Tuple[int, ...] | int | None,
) -> Tuple[object, ...]:
    """Stable cache key for the GNME blueprint request."""
    if local_dims_per_party is None:
        normalized_dims = tuple(sorted(problem.local_dimensions_per_party.items()))
        local_dims_key = None
    else:
        normalized_dim_map = problem._normalize_local_dimensions(
            problem.party_names,
            local_dims_per_party,
        )
        normalized_dims = tuple(sorted(normalized_dim_map.items()))
        local_dims_key = normalized_dims
    return (
        problem.n_parties,
        problem.inflation_level,
        tuple(problem.party_names),
        normalized_dims,
        tuple(subset_sizes),
        True,   # include_known_marginals
        2,      # min_shared_occurrences
        2,      # min_shared_inflations
        local_dims_key,
    )


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
    pair_rows, pair_cols = _cached_upper_triangle_pairs(matrix_dim)
    basis_maps = _cached_unique_basis_maps(slot_dims, local_symmetry_perms)

    canonical_rows = None
    for basis_map in basis_maps:
        mapped_rows = basis_map[pair_rows]
        mapped_cols = basis_map[pair_cols]
        ordered_rows = np.minimum(mapped_rows, mapped_cols)
        ordered_cols = np.maximum(mapped_rows, mapped_cols)
        mapped_upper_rows = _upper_triangle_coordinate_indices(
            ordered_rows,
            ordered_cols,
            matrix_dim,
        )
        if canonical_rows is None:
            canonical_rows = mapped_upper_rows
        else:
            canonical_rows = np.minimum(canonical_rows, mapped_upper_rows)

    assert canonical_rows is not None
    representative_rows, orbit_lookup = np.unique(canonical_rows, return_inverse=True)
    orbit_representatives = tuple(
        (int(pair_rows[row]), int(pair_cols[row]))
        for row in representative_rows.astype(np.int64, copy=False)
    )
    pair_to_orbit = {
        (int(row), int(col)): int(orbit_index)
        for row, col, orbit_index in zip(
            pair_rows,
            pair_cols,
            orbit_lookup.astype(np.int32, copy=False),
        )
    }

    return SymmetricMatrixOrbitData(
        matrix_dim=matrix_dim,
        upper_triangular_entries=int(pair_rows.size),
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
    blueprint_cache_key = _blueprint_cache_key(
        problem,
        subset_sizes,
        local_dims_per_party,
    )
    blueprint = _load_cached_blueprint(blueprint_cache_key)
    blueprint_source = "cache"
    if blueprint is None:
        blueprint = problem.sdp_blueprint(
            subset_sizes=subset_sizes,
            include_known_marginals=True,
            min_shared_occurrences=2,
            min_shared_inflations=2,
            local_dims_per_party=local_dims_per_party,
        )
        _save_cached_blueprint(blueprint_cache_key, blueprint)
        blueprint_source = "fresh"
    _progress_log(
        real_verbose,
        1,
        "Blueprint ready: "
        f"{len(blueprint.variables)} full inflations, "
        f"{len(blueprint.shared_subset_classes)} shared subset classes "
        f"in {perf_counter() - t0:.2f}s "
        f"({blueprint_source}).",
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
    max_known_arity = max(
        (
            len(occurrence.labels)
            for variable in blueprint.variables
            for occurrences in variable.fixed_known_marginals.values()
            for occurrence in occurrences
        ),
        default=0,
    )
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
                if len(occurrence.labels) < max_known_arity:
                    continue
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
    shared_subset_classes = blueprint.shared_subset_classes

    shared_stride = _progress_stride(len(shared_subset_classes))
    for class_index, subset_class in enumerate(shared_subset_classes):
        if real_verbose >= 2 and (
            class_index == 0
            or (class_index + 1) % shared_stride == 0
            or class_index + 1 == len(shared_subset_classes)
        ):
            _progress_log(
                real_verbose,
                2,
                f"  shared class {class_index + 1}/{len(shared_subset_classes)}",
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
            local_symmetry_perms=problem.local_symmetry_permutations(
                representative_occurrence.labels
            ),
            factorization=representative_occurrence.factorization,
            kind=representative_kind,
            is_fixed_known=representative_occurrence.is_known_marginal,
            occurrence_count=len(subset_class.occurrences),
        )
        shared_representatives.append(representative)
        if representative_kind == "aux_psd":
            auxiliary_representatives.append(representative)
            if len(representative.factorization) == 1:
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
                    target_slot_permutation=None,
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
            local_symmetry_perms=problem.local_symmetry_permutations(
                representative_occurrence.labels
            ),
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
                    target_slot_permutation=None,
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

    (
        auxiliary_representatives,
        known_representatives,
        shared_representatives,
        representative_links,
        representative_constraints,
        ppt_variables,
        ppt_constraints,
    ) = _drop_dominated_known_representatives(
        auxiliary_representatives=tuple(auxiliary_representatives),
        known_representatives=tuple(known_representatives),
        shared_representatives=tuple(shared_representatives),
        representative_links=tuple(representative_links),
        representative_constraints=tuple(representative_constraints),
        ppt_variables=tuple(ppt_variables),
        ppt_constraints=tuple(ppt_constraints),
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


def _representative_from_top_down_family(
    family: GNMETopDownFamily,
    name: str,
    kind: str,
    is_fixed_known: bool,
) -> SharedMarginalRepresentativeDraft:
    return SharedMarginalRepresentativeDraft(
        name=name,
        signature=family.signature,
        representative_occurrence=family.representative_occurrence,
        target_lexorder=family.representative_occurrence.labels,
        slot_dims=family.slot_dims,
        matrix_dim=family.matrix_dim,
        local_symmetry_perms=family.local_symmetry_perms,
        factorization=family.factorization,
        kind=kind,
        is_fixed_known=is_fixed_known,
        occurrence_count=family.occurrence_count,
    )


def _compute_tau_representative_anchors(
    tau_representative_constraints: Tuple[RepresentativeConstraintDraft, ...],
) -> Dict[str, Tuple[int, ...]]:
    """Group tau representative routes by representative with a canonical anchor."""
    def _constraint_sort_key(constraint: RepresentativeConstraintDraft) -> Tuple:
        return (
            str(constraint.occurrence_inflation_name),
            tuple(constraint.occurrence_labels),
            tuple(int(pos) for pos in constraint.keep_positions),
            tuple(int(pos) for pos in constraint.traced_positions),
            tuple(int(pos) for pos in constraint.target_slot_permutation or ()),
        )

    groups_by_representative: Dict[str, List[int]] = {}
    for idx, constraint in enumerate(tau_representative_constraints):
        rep_name = constraint.representative_name
        if rep_name not in groups_by_representative:
            groups_by_representative[rep_name] = []
        groups_by_representative[rep_name].append(idx)

    return {
        rep: tuple(
            sorted(
                indices,
                key=lambda idx: _constraint_sort_key(tau_representative_constraints[idx]),
            )
        )
        for rep, indices in groups_by_representative.items()
    }


def _dominated_known_representative_names(
    known_representatives: Tuple[SharedMarginalRepresentativeDraft, ...] | Iterable[SharedMarginalRepresentativeDraft],
    representative_constraints: Tuple[RepresentativeConstraintDraft, ...] | Iterable[RepresentativeConstraintDraft],
) -> set[str]:
    """Known representatives implied by larger known anchors through one tau view.

    If a smaller known representative and a larger known representative both
    occur as marginals of the same source variable, then anchoring the larger
    one already fixes the smaller one by partial trace of that source state.
    """
    known_representatives = tuple(known_representatives)
    representative_constraints = tuple(representative_constraints)
    if not known_representatives:
        return set()

    constraints_by_rep: Dict[str, List[RepresentativeConstraintDraft]] = {}
    for constraint in representative_constraints:
        constraints_by_rep.setdefault(constraint.representative_name, []).append(constraint)

    dominated: set[str] = set()
    for representative in known_representatives:
        small_size = len(representative.target_lexorder)
        for larger in known_representatives:
            if len(larger.target_lexorder) <= small_size:
                continue
            if any(
                small_constraint.source_variable_name == large_constraint.source_variable_name
                and set(small_constraint.keep_positions).issubset(set(large_constraint.keep_positions))
                for small_constraint in constraints_by_rep.get(representative.name, ())
                for large_constraint in constraints_by_rep.get(larger.name, ())
            ):
                dominated.add(representative.name)
                break
    return dominated


def _drop_dominated_known_representatives(
    *,
    auxiliary_representatives: Tuple[SharedMarginalRepresentativeDraft, ...],
    known_representatives: Tuple[SharedMarginalRepresentativeDraft, ...],
    shared_representatives: Tuple[SharedMarginalRepresentativeDraft, ...] | None,
    representative_links: Tuple[RepresentativeLinkDraft, ...],
    representative_constraints: Tuple[RepresentativeConstraintDraft, ...],
    ppt_variables: Tuple[PPTVariableDraft, ...],
    ppt_constraints: Tuple[PPTConstraintDraft, ...],
) -> Tuple[
    Tuple[SharedMarginalRepresentativeDraft, ...],
    Tuple[SharedMarginalRepresentativeDraft, ...],
    Tuple[SharedMarginalRepresentativeDraft, ...] | None,
    Tuple[RepresentativeLinkDraft, ...],
    Tuple[RepresentativeConstraintDraft, ...],
    Tuple[PPTVariableDraft, ...],
    Tuple[PPTConstraintDraft, ...],
]:
    """Remove dominated known representatives and all derived constraints."""
    dominated = _dominated_known_representative_names(
        known_representatives,
        representative_constraints,
    )
    if not dominated:
        return (
            auxiliary_representatives,
            known_representatives,
            shared_representatives,
            representative_links,
            representative_constraints,
            ppt_variables,
            ppt_constraints,
        )

    new_auxiliary = tuple(auxiliary_representatives)
    new_known = tuple(
        representative
        for representative in known_representatives
        if representative.name not in dominated
    )
    if shared_representatives is None:
        new_shared = None
    else:
        new_shared = tuple(
            representative
            for representative in shared_representatives
            if representative.name not in dominated
        )

    new_links = tuple(
        link
        for link in representative_links
        if link.representative_name not in dominated
    )
    new_constraints = tuple(
        constraint
        for constraint in representative_constraints
        if constraint.representative_name not in dominated
    )

    return (
        new_auxiliary,
        new_known,
        new_shared,
        new_links,
        new_constraints,
        tuple(
            candidate
            for candidate in ppt_variables
            if candidate.source_variable_name not in dominated
        ),
        tuple(
            candidate
            for candidate in ppt_constraints
            if candidate.source_variable_name not in dominated
        ),
    )


def build_top_down_sdp_draft(
    problem: GNMEProblem,
    subset_sizes: Tuple[int, ...] = (2, 3, 4),
    local_dims_per_party: Dict[str, int] | Tuple[int, ...] | int | None = None,
    include_overlap_families: bool = False,
    verbose: int | None = None,
) -> TopDownStateSDPDraft:
    """Build the fresh top-down GNME draft from maximal-family metadata.

    This keeps the current stable bottom-up builder untouched. The returned
    object is the new structural target for the redesign:

    - tau PSD variables and their internal symmetries,
    - maximal shared families as first-class representatives,
    - no instantiated overlap descendants in the default anchored path,
    - fixed-only known families,
    - PPT candidates on tau and maximal families.
    """
    if include_overlap_families:
        raise NotImplementedError(
            "The anchored top-down draft currently supports maximal and known "
            "families only. Overlap-family instantiation is intentionally not "
            "enabled in this default path yet."
        )

    real_verbose = _resolve_verbose(verbose, getattr(problem, "verbose", 0))
    t0 = perf_counter()
    _progress_log(real_verbose, 1, "Building GNME top-down family blueprint...")
    family_blueprint = problem.top_down_family_blueprint(
        subset_sizes=None,
        min_occurrences=2,
        min_inflations=2,
        local_dims_per_party=local_dims_per_party,
    )
    variable_blueprint = problem.sdp_blueprint(
        subset_sizes=subset_sizes,
        include_known_marginals=True,
        min_shared_occurrences=2,
        min_shared_inflations=2,
        local_dims_per_party=local_dims_per_party,
    )
    _progress_log(
        real_verbose,
        1,
        "Top-down metadata ready: "
        f"{len(family_blueprint.maximal_families)} maximal families, "
        f"{len(family_blueprint.overlap_families)} overlap families, "
        f"{len(family_blueprint.known_families)} known families "
        f"in {perf_counter() - t0:.2f}s.",
    )

    psd_variables = []
    internal_symmetry_constraints = []
    fixed_constraints = []
    ppt_variables = []
    ppt_constraints = []
    max_known_arity = max(
        (
            len(occurrence.labels)
            for variable in variable_blueprint.variables
            for occurrences in variable.fixed_known_marginals.values()
            for occurrence in occurrences
        ),
        default=0,
    )
    for variable in variable_blueprint.variables:
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
                if len(occurrence.labels) < max_known_arity:
                    continue
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

    tau_lookup = {variable.name: variable for variable in variable_blueprint.variables}

    maximal_representatives = []
    known_representatives = []
    representative_links = []
    tau_representative_constraints = []

    for family_index, family in enumerate(family_blueprint.maximal_families):
        representative = _representative_from_top_down_family(
            family,
            name=f"mu_td_{family_index}",
            kind="aux_psd",
            is_fixed_known=False,
        )
        maximal_representatives.append(representative)
        for occurrence in family.occurrences:
            representative_links.append(
                RepresentativeLinkDraft(
                    representative_name=representative.name,
                    representative_kind=representative.kind,
                    occurrence=occurrence,
                )
            )
            source_variable = tau_lookup[f"tau_{occurrence.inflation_index}"]
            recipe = partial_trace_recipe(
                source_variable.lexorder,
                source_variable.slot_dims,
                occurrence.positions,
            )
            target_perm = problem.coarse_subset_alignment_permutation(
                occurrence.signature,
                family.representative_occurrence.signature,
            )
            if target_perm == tuple(range(len(target_perm))):
                target_perm = None
            tau_representative_constraints.append(
                RepresentativeConstraintDraft(
                    source_variable_name=source_variable.name,
                    source_variable_kind="tau",
                    source_lexorder=source_variable.lexorder,
                    representative_name=representative.name,
                    representative_kind=representative.kind,
                    representative_lexorder=representative.target_lexorder,
                    keep_positions=recipe.keep_positions,
                    traced_positions=recipe.traced_positions,
                    target_slot_permutation=target_perm,
                    named_einsum_spec=recipe.named_einsum_spec,
                    symbolic_einsum_spec=recipe.einsum_spec,
                    occurrence_inflation_name=occurrence.inflation_name,
                    occurrence_labels=occurrence.labels,
                )
            )
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

    for family_index, family in enumerate(family_blueprint.known_families):
        representative = _representative_from_top_down_family(
            family,
            name=f"nu_td_{family_index}",
            kind="known_matrix",
            is_fixed_known=True,
        )
        known_representatives.append(representative)
        for occurrence in family.occurrences:
            representative_links.append(
                RepresentativeLinkDraft(
                    representative_name=representative.name,
                    representative_kind=representative.kind,
                    occurrence=occurrence,
                )
            )
            source_variable = tau_lookup[f"tau_{occurrence.inflation_index}"]
            recipe = partial_trace_recipe(
                source_variable.lexorder,
                source_variable.slot_dims,
                occurrence.positions,
            )
            target_perm = problem.coarse_subset_alignment_permutation(
                occurrence.signature,
                family.representative_occurrence.signature,
            )
            if target_perm == tuple(range(len(target_perm))):
                target_perm = None
            tau_representative_constraints.append(
                RepresentativeConstraintDraft(
                    source_variable_name=source_variable.name,
                    source_variable_kind="tau",
                    source_lexorder=source_variable.lexorder,
                    representative_name=representative.name,
                    representative_kind=representative.kind,
                    representative_lexorder=representative.target_lexorder,
                    keep_positions=recipe.keep_positions,
                    traced_positions=recipe.traced_positions,
                    target_slot_permutation=target_perm,
                    named_einsum_spec=recipe.named_einsum_spec,
                    symbolic_einsum_spec=recipe.einsum_spec,
                    occurrence_inflation_name=occurrence.inflation_name,
                    occurrence_labels=occurrence.labels,
                )
            )

    tau_representative_constraints = list(
        _quotient_representative_constraint_sequence(
            tuple(tau_representative_constraints),
            tau_lookup,
            use_source_symmetry=True,
        )
    )

    _unused_auxiliary, known_representatives, _, representative_links, tau_representative_constraints, _ppt_vars_unused, _ppt_cons_unused = _drop_dominated_known_representatives(
        auxiliary_representatives=tuple(),
        known_representatives=tuple(known_representatives),
        shared_representatives=None,
        representative_links=tuple(representative_links),
        representative_constraints=tuple(tau_representative_constraints),
        ppt_variables=tuple(),
        ppt_constraints=tuple(),
    )
    maximal_representatives = tuple(maximal_representatives)

    # Compute cross-inflation groups from the quotiented tau constraints.
    tau_anchor_groups = _compute_tau_representative_anchors(tuple(tau_representative_constraints))
    cross_inflation_groups = []
    for rep_name, indices in sorted(tau_anchor_groups.items()):
        if not indices:
            continue
        # First constraint is the anchor, rest are non-anchors
        anchor_idx = indices[0]
        non_anchor_indices = indices[1:]
        anchor_constraint = tau_representative_constraints[anchor_idx]
        non_anchor_constraints = tuple(
            tau_representative_constraints[idx] for idx in non_anchor_indices
        )
        group = CrossInflationConstraintGroupDraft(
            representative_name=rep_name,
            anchor_constraint=anchor_constraint,
            non_anchor_constraints=non_anchor_constraints,
        )
        cross_inflation_groups.append(group)

    draft = TopDownStateSDPDraft(
        family_blueprint=family_blueprint,
        party_dims=family_blueprint.party_dims,
        verbose=real_verbose,
        psd_variables=tuple(psd_variables),
        internal_symmetry_constraints=tuple(internal_symmetry_constraints),
        maximal_representatives=tuple(maximal_representatives),
        known_representatives=tuple(known_representatives),
        representative_links=tuple(representative_links),
        tau_representative_constraints=tuple(tau_representative_constraints),
        ppt_variables=tuple(ppt_variables),
        ppt_constraints=tuple(ppt_constraints),
        fixed_marginal_constraints=tuple(fixed_constraints),
        cross_inflation_groups=tuple(cross_inflation_groups),
        verified_at_draft_time=False,
    )
    validation_stats = validate_top_down_draft(draft, verbose=0)
    draft = replace(draft, verified_at_draft_time=(validation_stats["errors"] == 0))
    _progress_log(
        real_verbose,
        1,
        "Top-down GNME draft ready: "
        f"{len(draft.psd_variables)} tau variables, "
        f"{len(draft.maximal_representatives)} maximal representatives, "
        f"{len(draft.known_representatives)} known representatives, "
        f"{len(draft.tau_representative_constraints)} tau-family equalities, "
        f"{len(draft.ppt_variables)} PPT candidates, "
        f"validated={draft.verified_at_draft_time}.",
    )
    return draft


def build_smaller_sdp_draft(
    problem: GNMEProblem,
    subset_sizes: Tuple[int, ...] = (2, 3, 4),
    local_dims_per_party: Dict[str, int] | Tuple[int, ...] | int | None = None,
    verbose: int | None = None,
) -> TopDownStateSDPDraft:
    """Build the maximal-family GNME draft used as the reduced package mode.

    This is the smallest solver-facing package draft currently supported:
    - maximal shared families only,
    - maximal known anchors only,
    - no non-maximal representative auxiliaries.
    """
    return build_top_down_sdp_draft(
        problem,
        subset_sizes=subset_sizes,
        local_dims_per_party=local_dims_per_party,
        include_overlap_families=False,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Paper-style inflation draft
# ---------------------------------------------------------------------------

def _paper_labels_to_positions(
    source_lexorder: Tuple[str, ...],
    labels: Tuple[str, ...],
) -> Tuple[int, ...]:
    position_by_label = {label: index for index, label in enumerate(source_lexorder)}
    try:
        return tuple(int(position_by_label[label]) for label in labels)
    except KeyError as exc:
        raise ValueError(f"Label {exc.args[0]!r} not present in source lexorder.") from exc


def _paper_marginal_view(
    problem: GNMEProblem,
    source_variable: PSDVariableDraft,
    target_lexorder: Tuple[str, ...],
    source_labels: Tuple[str, ...],
    party_dims: Dict[str, int],
) -> PaperMarginalViewDraft:
    keep_positions = _paper_labels_to_positions(source_variable.lexorder, source_labels)
    traced_positions = tuple(
        position
        for position in range(len(source_variable.lexorder))
        if position not in keep_positions
    )
    try:
        target_perm = problem.coarse_subset_alignment_permutation(
            source_labels,
            target_lexorder,
        )
    except ValueError:
        target_perm = _paper_party_alignment_permutation(
            tuple(source_labels),
            tuple(target_lexorder),
        )
    if target_perm == tuple(range(len(target_perm))):
        target_perm = None
    slot_dims = tuple(int(party_dims[label.split("_", 1)[0]]) for label in target_lexorder)
    return PaperMarginalViewDraft(
        source_variable_name=source_variable.name,
        source_variable_kind="tau",
        source_lexorder=tuple(source_variable.lexorder),
        keep_positions=keep_positions,
        traced_positions=traced_positions,
        target_lexorder=tuple(target_lexorder),
        target_slot_permutation=target_perm,
        slot_dims=slot_dims,
        matrix_dim=product_dim(slot_dims),
        factorization=tuple((label,) for label in target_lexorder),
        occurrence_labels=tuple(source_labels),
    )


def _paper_party_alignment_permutation(
    source_labels: Tuple[str, ...],
    target_labels: Tuple[str, ...],
) -> Tuple[int, ...]:
    """Align slots by party-occurrence order, ignoring copy indices."""
    source_tokens = []
    source_counts: Dict[str, int] = {}
    for label in source_labels:
        party = label.split("_", 1)[0]
        source_counts[party] = source_counts.get(party, 0) + 1
        source_tokens.append((party, source_counts[party]))

    target_tokens = []
    target_counts: Dict[str, int] = {}
    for label in target_labels:
        party = label.split("_", 1)[0]
        target_counts[party] = target_counts.get(party, 0) + 1
        target_tokens.append((party, target_counts[party]))

    if sorted(source_tokens) != sorted(target_tokens):
        raise ValueError("Paper marginal labels do not share the same party-occurrence pattern.")

    target_pos_by_token = {token: index for index, token in enumerate(target_tokens)}
    return tuple(int(target_pos_by_token[token]) for token in source_tokens)


def _paper_direct_ppt_constraint(
    name: str,
    marginal_view: PaperMarginalViewDraft,
    transpose_positions: Tuple[int, ...],
) -> PaperPPTConstraintDraft:
    transpose_positions = tuple(int(pos) for pos in transpose_positions)
    complement_positions = tuple(
        index for index in range(len(marginal_view.target_lexorder)) if index not in transpose_positions
    )
    return PaperPPTConstraintDraft(
        name=name,
        marginal_view=marginal_view,
        transpose_positions=transpose_positions,
        complement_positions=complement_positions,
        transpose_lexorder=tuple(marginal_view.target_lexorder[pos] for pos in transpose_positions),
        complement_lexorder=tuple(marginal_view.target_lexorder[pos] for pos in complement_positions),
    )


def _paper_tripartite_role_variables(
    variables: Tuple[PSDVariableDraft, ...],
    inflation_level: int,
) -> Dict[str, PSDVariableDraft]:
    by_name = {variable.name: variable for variable in variables}
    if inflation_level == 2:
        if "tau_0" not in by_name or "tau_1" not in by_name:
            raise ValueError("Paper level-2 draft expects tau_0 and tau_1.")
        return {
            "tau": by_name["tau_0"],
            "gamma": by_name["tau_1"],
        }
    if inflation_level == 3:
        by_symmetry = {len(variable.local_symmetry_perms): variable for variable in variables}
        try:
            return {
                "sigma": by_symmetry[6],
                "tau": by_symmetry[2],
                "gamma": by_symmetry[3],
            }
        except KeyError as exc:
            raise ValueError(
                "Paper level-3 draft expects tau symmetries of orders 6, 2 and 3."
            ) from exc
    raise NotImplementedError("Paper-style draft currently supports only levels 2 and 3.")


def _paper_ordered_occurrence_labels(
    view: PaperMarginalViewDraft,
) -> Tuple[str, ...]:
    """Occurrence labels ordered by the target-slot convention."""
    if view.target_slot_permutation is None:
        return tuple(view.occurrence_labels)
    ordered = [None] * len(view.occurrence_labels)
    for source_index, target_index in enumerate(view.target_slot_permutation):
        ordered[int(target_index)] = view.occurrence_labels[source_index]
    if any(label is None for label in ordered):
        raise ValueError("Paper marginal view permutation did not cover all target slots.")
    return tuple(ordered)  # type: ignore[return-value]


def _paper_view_key(
    view: PaperMarginalViewDraft,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Stable key for a paper marginal view in target-slot order."""
    return (
        str(view.source_variable_name),
        tuple(view.target_lexorder),
        _paper_ordered_occurrence_labels(view),
    )


def _paper_subview(
    problem: GNMEProblem,
    source_lookup: Dict[str, PSDVariableDraft],
    party_dims: Dict[str, int],
    view: PaperMarginalViewDraft,
    subset_positions: Tuple[int, ...],
) -> PaperMarginalViewDraft:
    ordered_labels = _paper_ordered_occurrence_labels(view)
    target_labels = tuple(view.target_lexorder[index] for index in subset_positions)
    target_parties = tuple(label.split("_", 1)[0] for label in target_labels)
    if len(set(target_parties)) != len(target_parties):
        raise ValueError("Paper subview is not a one-copy-per-party known marginal.")
    canonical_target = tuple(
        f"{party}_11"
        for party in problem.party_names
        if party in set(target_parties)
    )
    return _paper_marginal_view(
        problem,
        source_lookup[view.source_variable_name],
        canonical_target,
        tuple(ordered_labels[index] for index in subset_positions),
        party_dims,
    )


def _paper_symmetry_images(
    problem: GNMEProblem,
    source_lookup: Dict[str, PSDVariableDraft],
    party_dims: Dict[str, int],
    view: PaperMarginalViewDraft,
) -> Tuple[PaperMarginalViewDraft, ...]:
    source_variable = source_lookup[view.source_variable_name]
    ordered_labels = _paper_ordered_occurrence_labels(view)
    position_by_label = {label: index for index, label in enumerate(source_variable.lexorder)}
    images = []
    for permutation in source_variable.local_symmetry_perms:
        permuted_labels = tuple(
            source_variable.lexorder[permutation[position_by_label[label]]]
            for label in ordered_labels
        )
        images.append(
            _paper_marginal_view(
                problem,
                source_variable,
                tuple(view.target_lexorder),
                permuted_labels,
                party_dims,
            )
        )
    return tuple(images)


def _paper_prune_nonmaximal_observed_constraints(
    problem: GNMEProblem,
    psd_variables: Tuple[PSDVariableDraft, ...],
    observed_constraints: Tuple[PaperObservedConstraintDraft, ...] | Iterable[PaperObservedConstraintDraft],
    equality_constraints: Tuple[PaperEqualityConstraintDraft, ...] | Iterable[PaperEqualityConstraintDraft],
    party_dims: Dict[str, int],
) -> Tuple[PaperObservedConstraintDraft, ...]:
    """Keep only maximal observed anchors.

    In the tripartite hierarchy, pairwise observed targets are redundant once
    the corresponding tripartite anchors are generated. The paper-style draft
    should therefore emit only maximal observed families and let the rest
    follow from trace/equality/symmetry structure.
    """
    observed_constraints = tuple(observed_constraints)
    if not observed_constraints:
        return observed_constraints

    observed_sizes = {len(constraint.target_lexorder) for constraint in observed_constraints}
    if len(observed_sizes) <= 1:
        return observed_constraints
    max_size = max(observed_sizes)
    return tuple(
        constraint
        for constraint in observed_constraints
        if len(constraint.target_lexorder) == max_size
    )


def build_paper_sdp_draft(
    problem: GNMEProblem,
    subset_sizes: Tuple[int, ...] = (2, 3, 4),
    local_dims_per_party: Dict[str, int] | Tuple[int, ...] | int | None = None,
    verbose: int | None = None,
) -> PaperStateSDPDraft:
    """Build the appendix-style tripartite inflation draft for levels 2 and 3."""
    if int(problem.n_parties) != 3:
        raise NotImplementedError("Paper-style draft currently supports only 3 parties.")
    if int(problem.inflation_level) not in {2, 3}:
        raise NotImplementedError("Paper-style draft currently supports only levels 2 and 3.")

    real_verbose = _resolve_verbose(verbose, getattr(problem, "verbose", 0))
    variable_blueprint = problem.sdp_blueprint(
        subset_sizes=subset_sizes,
        include_known_marginals=True,
        min_shared_occurrences=2,
        min_shared_inflations=2,
        local_dims_per_party=local_dims_per_party,
    )
    party_dims = dict(variable_blueprint.party_dims)

    psd_variables = []
    internal_symmetry_constraints = []
    for variable in variable_blueprint.variables:
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
            internal_symmetry_constraints.append(
                InternalSymmetryConstraintDraft(
                    variable_name=variable.name,
                    variable_kind="tau",
                    lexorder=variable.lexorder,
                    permutation=permutation,
                    permuted_lexorder=tuple(variable.lexorder[pos] for pos in permutation),
                    named_action=symmetry_action_named_spec(variable.lexorder, permutation),
                )
            )
    psd_variables = tuple(psd_variables)
    role_variables = _paper_tripartite_role_variables(psd_variables, int(problem.inflation_level))

    observed_constraints: List[PaperObservedConstraintDraft] = []
    equality_constraints: List[PaperEqualityConstraintDraft] = []
    ppt_constraints: List[PaperPPTConstraintDraft] = []

    def add_observed_view(
        name: str,
        source_variable: PSDVariableDraft,
        source_labels: Tuple[str, ...],
        target_lexorder: Tuple[str, ...],
    ) -> None:
        observed_constraints.append(
            PaperObservedConstraintDraft(
                name=name,
                target_lexorder=tuple(target_lexorder),
                marginal_view=_paper_marginal_view(
                    problem,
                    source_variable,
                    tuple(target_lexorder),
                    tuple(source_labels),
                    party_dims,
                ),
            )
        )

    def add_known_occurrences(source_variable: PSDVariableDraft) -> None:
        for occurrences in source_variable.fixed_known_marginals.values():
            for occurrence in sorted(
                occurrences,
                key=lambda occ: (occ.positions, occ.labels),
            ):
                target_lexorder = tuple(
                    f"{label.split('_', 1)[0]}_11"
                    for label in occurrence.labels
                )
                name = f"rho_{source_variable.name}_{'__'.join(occurrence.labels)}"
                add_observed_view(
                    name,
                    source_variable,
                    tuple(occurrence.labels),
                    target_lexorder,
                )

    def add_equality(
        name: str,
        lhs_role: str,
        lhs_labels: Tuple[str, ...],
        rhs_role: str,
        rhs_labels: Tuple[str, ...],
        target_lexorder: Tuple[str, ...] | None = None,
    ) -> None:
        target_lexorder = tuple(target_lexorder or lhs_labels)
        equality_constraints.append(
            PaperEqualityConstraintDraft(
                name=name,
                lhs_view=_paper_marginal_view(
                    problem,
                    role_variables[lhs_role],
                    target_lexorder,
                    tuple(lhs_labels),
                    party_dims,
                ),
                rhs_view=_paper_marginal_view(
                    problem,
                    role_variables[rhs_role],
                    target_lexorder,
                    tuple(rhs_labels),
                    party_dims,
                ),
            )
        )

    def add_ppt(
        name: str,
        role: str,
        source_labels: Tuple[str, ...],
        transpose_positions: Tuple[int, ...],
        target_lexorder: Tuple[str, ...] | None = None,
    ) -> None:
        target_lexorder = tuple(target_lexorder or source_labels)
        view = _paper_marginal_view(
            problem,
            role_variables[role],
            target_lexorder,
            tuple(source_labels),
            party_dims,
        )
        ppt_constraints.append(
            _paper_direct_ppt_constraint(name, view, tuple(transpose_positions))
        )

    for source_variable in psd_variables:
        add_known_occurrences(source_variable)

    if int(problem.inflation_level) == 2:

        add_equality(
            "eq_abab",
            "gamma",
            ("A_11", "B_11", "A_22", "B_22"),
            "tau",
            ("A_11", "B_11", "A_22", "B_22"),
        )
        add_equality(
            "eq_bcbc",
            "gamma",
            ("B_11", "C_12", "B_22", "C_21"),
            "tau",
            ("B_11", "C_11", "B_22", "C_22"),
        )
        add_equality(
            "eq_caca",
            "gamma",
            ("C_12", "A_22", "C_21", "A_11"),
            "tau",
            ("C_11", "A_11", "C_22", "A_22"),
        )

        add_ppt("ppt_gamma_b2", "gamma", ("A_11", "B_11", "C_12", "B_22"), (3,))
        add_ppt("ppt_gamma_c2", "gamma", ("B_11", "C_12", "A_22", "C_21"), (3,))
        add_ppt("ppt_gamma_a1", "gamma", ("C_12", "A_22", "B_22", "A_11"), (3,))
        add_ppt("ppt_tau_full", "tau", tuple(role_variables["tau"].lexorder), (0, 2, 4))
    else:
        add_equality(
            "eq_gamma_tau_0",
            "gamma",
            ("B_33", "C_31", "A_11", "B_11", "C_12", "B_22", "C_23"),
            "tau",
            ("B_22", "C_23", "A_11", "B_11", "C_11", "B_33", "C_32"),
        )
        add_equality(
            "eq_gamma_tau_1",
            "gamma",
            ("C_31", "A_11", "B_11", "C_12", "A_22", "C_23", "A_33"),
            "tau",
            ("C_23", "A_11", "B_11", "C_11", "A_22", "C_32", "A_33"),
        )
        add_equality(
            "eq_gamma_tau_2",
            "gamma",
            ("A_11", "B_11", "C_12", "A_22", "B_22", "A_33", "B_33"),
            "tau",
            ("A_11", "B_11", "C_11", "A_22", "B_22", "A_33", "B_33"),
        )

        add_equality(
            "eq_tau_sigma_0",
            "tau",
            ("A_11", "B_11", "A_22", "B_22", "A_33", "B_33", "C_32"),
            "sigma",
            ("A_11", "B_11", "A_22", "B_22", "A_33", "B_33", "C_33"),
        )
        add_equality(
            "eq_tau_sigma_1",
            "tau",
            ("B_11", "C_11", "B_22", "C_23", "A_33", "B_33", "C_32"),
            "sigma",
            ("B_11", "C_11", "B_22", "C_22", "A_33", "B_33", "C_33"),
        )
        add_equality(
            "eq_tau_sigma_2",
            "tau",
            ("C_11", "A_22", "C_23", "A_11", "A_33", "B_33", "C_32"),
            "sigma",
            ("C_11", "A_11", "C_22", "A_22", "A_33", "B_33", "C_33"),
        )

        for name, labels, transpose_positions in (
            ("ppt_gamma_a1", ("A_11", "C_12", "A_22", "B_22", "C_23", "A_33", "B_33"), (0,)),
            ("ppt_gamma_b1", ("B_11", "A_22", "B_22", "C_23", "A_33", "B_33", "C_31"), (0,)),
            ("ppt_gamma_c1", ("C_12", "B_22", "C_23", "A_33", "B_33", "C_31", "A_11"), (0,)),
            ("ppt_gamma_a1b1c1", ("A_11", "B_11", "C_12", "B_22", "C_23", "A_33", "B_33"), (0, 1, 2)),
            ("ppt_gamma_b1c1a2", ("B_11", "C_12", "A_22", "C_23", "A_33", "B_33", "C_31"), (0, 1, 2)),
            ("ppt_gamma_c1a2b2", ("C_12", "A_22", "B_22", "A_33", "B_33", "C_31", "A_11"), (0, 1, 2)),
        ):
            add_ppt(name, "gamma", labels, transpose_positions)

        for name, labels, transpose_positions in (
            ("ppt_tau_a1", ("A_11", "C_11", "A_22", "B_22", "A_33", "B_33", "C_32"), (0,)),
            ("ppt_tau_b1", ("B_11", "A_22", "B_22", "C_23", "A_33", "B_33", "C_32"), (0,)),
            ("ppt_tau_c1", ("C_11", "B_22", "C_23", "A_11", "A_33", "B_33", "C_32"), (0,)),
            ("ppt_tau_c1a2b2", ("A_11", "C_11", "A_22", "B_22", "A_33", "B_33", "C_32"), (1, 2, 3)),
            ("ppt_tau_a2b2c2", ("B_11", "A_22", "B_22", "C_23", "A_33", "B_33", "C_32"), (1, 2, 3)),
            ("ppt_tau_b2c2a1", ("C_11", "B_22", "C_23", "A_11", "A_33", "B_33", "C_32"), (1, 2, 3)),
        ):
            add_ppt(name, "tau", labels, transpose_positions)

        add_ppt("ppt_tau_full", "tau", tuple(role_variables["tau"].lexorder), (2, 5, 8))
        add_ppt("ppt_sigma_full", "sigma", tuple(role_variables["sigma"].lexorder), (0, 3, 6))

    observed_constraints = list(
        _paper_prune_nonmaximal_observed_constraints(
            problem,
            psd_variables,
            tuple(observed_constraints),
            tuple(equality_constraints),
            party_dims,
        )
    )

    draft = PaperStateSDPDraft(
        formulation_name="paper",
        inflation_level=int(problem.inflation_level),
        party_dims=party_dims,
        verbose=real_verbose,
        psd_variables=psd_variables,
        internal_symmetry_constraints=tuple(internal_symmetry_constraints),
        observed_constraints=tuple(observed_constraints),
        equality_constraints=tuple(equality_constraints),
        ppt_constraints=tuple(ppt_constraints),
        verified_at_draft_time=True,
    )
    _progress_log(
        real_verbose,
        1,
        "Paper-style GNME draft ready: "
        f"{len(draft.psd_variables)} tau variables, "
        f"{len(draft.observed_constraints)} observed, "
        f"{len(draft.equality_constraints)} equalities, "
        f"{len(draft.ppt_constraints)} PPT constraints.",
    )
    return draft


def validate_paper_draft(
    draft: PaperStateSDPDraft,
    verbose: int = 0,
) -> Dict[str, int]:
    """Validate the integrity of the paper-style draft."""
    stats = {
        "errors": 0,
        "warnings": 0,
        "constraints_checked": len(draft.equality_constraints),
        "observed_constraints_checked": len(draft.observed_constraints),
        "ppt_constraints_checked": len(draft.ppt_constraints),
    }
    observed_names = [constraint.name for constraint in draft.observed_constraints]
    if len(set(observed_names)) != len(observed_names):
        stats["errors"] += len(observed_names) - len(set(observed_names))
    for constraint in draft.equality_constraints:
        if constraint.lhs_view.matrix_dim != constraint.rhs_view.matrix_dim:
            stats["errors"] += 1
    for constraint in draft.ppt_constraints:
        if len(constraint.transpose_positions) == 0:
            stats["warnings"] += 1
    return stats


# ---------------------------------------------------------------------------
# Phase 3: Draft validation
# ---------------------------------------------------------------------------

def validate_top_down_draft(
    draft: TopDownStateSDPDraft,
    verbose: int = 0,
) -> Dict[str, int]:
    """Validate the integrity of a top-down SDP draft.
    
    Checks that:
    - all tau representative constraints are covered by anchor groups,
    - every anchor group references a valid maximal or known representative,
    - anchor/non-anchor representative names are consistent,
    - the default anchored path does not silently instantiate overlaps.
    
    Returns a dictionary with validation statistics.
    """
    stats = {
        'errors': 0,
        'warnings': 0,
        'constraints_checked': 0,
        'cross_inflation_groups_checked': 0,
    }
    
    expected_anchor_names = {
        rep.name for rep in (draft.maximal_representatives + draft.known_representatives)
    }
    group_names = {group.representative_name for group in draft.cross_inflation_groups}

    if group_names != expected_anchor_names:
        missing = expected_anchor_names - group_names
        extra = group_names - expected_anchor_names
        if verbose >= 1 and missing:
            _progress_log(verbose, 1, f"Error: missing anchor groups for {sorted(missing)}")
        if verbose >= 1 and extra:
            _progress_log(verbose, 1, f"Error: unexpected anchor groups for {sorted(extra)}")
        stats['errors'] += len(missing) + len(extra)

    total_constraints = len(draft.tau_representative_constraints)
    stats['constraints_checked'] = total_constraints
    
    all_constraint_indices = set()
    for group_idx, group in enumerate(draft.cross_inflation_groups):
        stats['cross_inflation_groups_checked'] += 1
        # Check anchor constraint is valid
        anchor_idx = None
        for idx, constraint in enumerate(draft.tau_representative_constraints):
            if constraint is group.anchor_constraint:
                anchor_idx = idx
                break
        
        if anchor_idx is None:
            if verbose >= 1:
                _progress_log(
                    verbose, 1,
                    f"Error: Anchor constraint not found in group {group_idx}"
                )
            stats['errors'] += 1
        else:
            if group.anchor_constraint.representative_name != group.representative_name:
                if verbose >= 1:
                    _progress_log(
                        verbose,
                        1,
                        f"Error: group {group_idx} anchor representative mismatch.",
                    )
                stats['errors'] += 1
            all_constraint_indices.add(anchor_idx)
        
        # Check non-anchor constraints are valid
        for non_anchor_constraint in group.non_anchor_constraints:
            if non_anchor_constraint.representative_name != group.representative_name:
                if verbose >= 1:
                    _progress_log(
                        verbose,
                        1,
                        f"Error: non-anchor representative mismatch in group {group_idx}.",
                    )
                stats['errors'] += 1
            found = False
            for idx, constraint in enumerate(draft.tau_representative_constraints):
                if constraint is non_anchor_constraint:
                    found = True
                    all_constraint_indices.add(idx)
                    break
            if not found:
                if verbose >= 1:
                    _progress_log(
                        verbose, 1,
                        f"Error: Non-anchor constraint not found in group {group_idx}"
                    )
                stats['errors'] += 1
    
    # Check all constraints are referenced
    if len(all_constraint_indices) != total_constraints:
        missing = set(range(total_constraints)) - all_constraint_indices
        if verbose >= 1:
            _progress_log(
                verbose, 1,
                f"Error: {len(missing)} tau constraints not in any group: {missing}"
            )
        stats['errors'] += len(missing)
    
    return stats


# ---------------------------------------------------------------------------
# Known-value assignment
# ---------------------------------------------------------------------------

def _known_representative_lookup(
    model: StateSDPDraft | TopDownStateSDPDraft,
) -> Tuple[Dict[str, SharedMarginalRepresentativeDraft], Dict[Tuple[str, ...], SharedMarginalRepresentativeDraft]]:
    """Index known representatives by name and lexorder target."""
    by_name = {rep.name: rep for rep in model.known_representatives}
    by_target = {rep.target_lexorder: rep for rep in model.known_representatives}
    return by_name, by_target


def _resolve_known_value_key(
    model: StateSDPDraft | TopDownStateSDPDraft,
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
    model: StateSDPDraft | TopDownStateSDPDraft,
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
    enforce_known_values: bool = False,
    verbose: int | None = None,
):
    """Instantiate the legacy entrywise GNME draft as a MOSEK Fusion model.

    The current implementation targets real symmetric state variables. This is
    enough for the GHZ sanity example and keeps the first solver layer close to
    the draft structure. Known representatives always remain free PSD
    variables; callers can pin them later with affine constraints.
    """
    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense

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
            use_source_symmetry=include_internal_symmetry,
        )
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
    known_representative_variables = {
        representative.name: M.variable(
            representative.name,
            Domain.inPSDCone(representative.matrix_dim),
        )
        for representative in model.known_representatives
    }
    ppt_variables = (
        {
            ppt_variable.name: M.variable(
                ppt_variable.name,
                Domain.inPSDCone(ppt_variable.matrix_dim),
            )
            for ppt_variable in active_ppt_variables
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
                rhs_var = known_representative_variables[constraint.representative_name]
            else:
                rhs_var = auxiliary_variables[constraint.representative_name]

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
                    rhs = _symmetric_entry(rhs_var, row, col)
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
            f"Fusion step 5/5: adding {len(active_ppt_constraints)} PPT blocks...",
        )
        source_free_variable_lookup = {}
        source_free_variable_lookup.update(tau_variables)
        source_free_variable_lookup.update(auxiliary_variables)
        ppt_constraint_index = 0
        stride = _progress_stride(len(active_ppt_constraints))
        for constraint_index, constraint in enumerate(active_ppt_constraints, start=1):
            if real_verbose >= 2 and (
                constraint_index == 1
                or constraint_index % stride == 0
                or constraint_index == len(active_ppt_constraints)
            ):
                _progress_log(
                    real_verbose,
                    2,
                    f"  PPT block {constraint_index}/{len(active_ppt_constraints)} "
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
    enforce_known_values: bool = False,
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
    enforce_known_values: bool = False,
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
    enforce_known_values: bool = False,
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


def print_draft(model: StateSDPDraft | TopDownStateSDPDraft | PaperStateSDPDraft) -> None:
    """Print the SDP draft in a compact, inspection-oriented format."""
    if isinstance(model, PaperStateSDPDraft):
        print("Paper-Style Draft")
        print(f"  inflation level: {model.inflation_level}")
        print(f"  party dimensions: {model.party_dims}")
        print()
        print("Tau Variables")
        for variable in model.psd_variables:
            print(f"  {variable.name}: {' '.join(variable.lexorder)} shape=({variable.matrix_dim}, {variable.matrix_dim})")
        print()
        print("Observed Constraints")
        for constraint in model.observed_constraints:
            print(
                "  "
                f"{constraint.name}: "
                f"{constraint.marginal_view.source_variable_name} -> "
                f"{' '.join(constraint.target_lexorder)}"
            )
        print()
        print("Matching Equalities")
        for constraint in model.equality_constraints:
            print(
                "  "
                f"{constraint.name}: "
                f"{constraint.lhs_view.source_variable_name}({' '.join(constraint.lhs_view.occurrence_labels)})"
                f" = "
                f"{constraint.rhs_view.source_variable_name}({' '.join(constraint.rhs_view.occurrence_labels)})"
            )
        print()
        print("PPT Constraints")
        for candidate in model.ppt_constraints:
            print(
                "  "
                f"{candidate.name}: PT({candidate.marginal_view.source_variable_name}"
                f"({' '.join(candidate.marginal_view.occurrence_labels)})) "
                f"transpose={' '.join(candidate.transpose_lexorder)} | "
                f"complement={' '.join(candidate.complement_lexorder)}"
            )
        return

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
    "CrossInflationConstraintGroupDraft",
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
    "TopDownStateSDPDraft",
    "build_fusion_feasibility_model",
    "build_legacy_fusion_feasibility_model",
    "build_smaller_sdp_draft",
    "build_sdp_draft",
    "build_top_down_sdp_draft",
    "partial_trace_einsum_spec",
    "partial_trace_named_einsum_spec",
    "partial_trace_recipe",
    "print_assigned_values",
    "print_draft",
    "quotient_ppt_constraints",
    "quotient_representative_constraints",
    "set_values",
    "solve_fusion_feasibility",
    "solve_legacy_fusion_feasibility",
    "validate_top_down_draft",
]
