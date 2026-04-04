"""GNME-specific problem builder.

This module starts from the native ``InflationProblem`` implementation and only
changes the input model. For the GNME hierarchy we specify:

- the number of parties ``k``
- the inflation level ``n``

The underlying inflation scenario is then fixed to the network with ``k``
sources, where each source is connected to exactly ``k - 1`` parties. Outcome
and setting cardinalities are always taken to be trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, permutations, product
from string import ascii_uppercase
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .InflationProblem import InflationProblem
from .utils import ndarray_bytes_key


SubsetSignature = Tuple[
    Tuple[str, ...],
    Tuple[Tuple[str, Tuple[Tuple[str, ...], ...]], ...],
]


@dataclass(frozen=True)
class GNMEInflation:
    """Container for one maximal nonfanout inflation class.

    Each class is represented by a single lexorder-sized sequence together with
    the native copy-index orbit it belongs to. This is the right level for
    attaching an SDP variable later on.
    """

    representative: Tuple[str, ...]
    orbit: Tuple[Tuple[str, ...], ...]
    factorization: Tuple[Tuple[str, ...], ...]
    atomic_known_subsets: Tuple[Tuple[str, ...], ...]
    known_marginals: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]
    lexorder_lookup: Dict[str, int]
    local_symmetry_perms: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class GNMESubsetOccurrence:
    """One subset occurrence inside one maximal GNME inflation.

    This is the GNME-side analogue of how the LP code keeps monomial metadata
    together while classifying variables. The difference is that here we store
    reduced subsystems of one SDP variable rather than LP probability
    monomials.
    """

    inflation_index: int
    inflation_name: str
    labels: Tuple[str, ...]
    positions: Tuple[int, ...]
    parties: Tuple[str, ...]
    signature: SubsetSignature
    factorization: Tuple[Tuple[str, ...], ...]
    is_known_marginal: bool
    is_atomic_known: bool


@dataclass(frozen=True)
class GNMESharedSubsetClass:
    """Class of subset occurrences shared across GNME inflations."""

    signature: SubsetSignature
    occurrences: Tuple[GNMESubsetOccurrence, ...]


@dataclass(frozen=True)
class GNMESDPVariable:
    """SDP-facing view of one maximal GNME inflation."""

    inflation_index: int
    name: str
    lexorder: Tuple[str, ...]
    lexorder_lookup: Dict[str, int]
    slot_dims: Tuple[int, ...]
    local_symmetry_perms: Tuple[Tuple[int, ...], ...]
    factorization: Tuple[Tuple[str, ...], ...]
    fixed_known_marginals: Dict[Tuple[str, ...], Tuple[GNMESubsetOccurrence, ...]]


@dataclass(frozen=True)
class GNMESDPBlueprint:
    """Package-level SDP generation blueprint.

    This is intentionally structural only: it packages the maximal GNME
    variables, their internal fixed marginals, and the shared subset classes
    that can later become inter-variable equality constraints.
    """

    party_dims: Dict[str, int]
    variables: Tuple[GNMESDPVariable, ...]
    shared_subset_classes: Tuple[GNMESharedSubsetClass, ...]


class GNMEProblem(InflationProblem):
    """Inflation-style problem object with GNME-specific inputs.

    The class intentionally subclasses ``InflationProblem`` so that all the
    native inflation data structures remain available:

    - ``hypergraph``
    - ``inflation_indices_per_party``
    - ``_lexorder``
    - ``symmetries``

    The only change at this stage is how the scenario is defined.
    """

    def __init__(
        self,
        n_parties: int,
        inflation_level: int,
        local_dims_per_party: int | Tuple[int, ...] | List[int] | Dict[str, int] = 2,
        order: Tuple[str, ...] | List[str] = tuple(),
        verbose: int = 0,
    ) -> None:
        if n_parties < 2:
            raise ValueError("GNMEProblem requires at least two parties.")
        if inflation_level < 1:
            raise ValueError("The inflation level must be at least one.")

        party_names = tuple(order) if order else self._default_party_names(n_parties)
        local_dim_map = self._normalize_local_dimensions(
            party_names,
            local_dims_per_party,
        )
        dag = self.build_gnme_dag(party_names)

        # GNME starts from the same inflation engine, but with trivial local
        # measurement structure. Every lexorder entry is then just a party copy.
        super().__init__(
            dag=dag,
            outcomes_per_party=(1,) * n_parties,
            settings_per_party=(1,) * n_parties,
            inflation_level_per_source=(inflation_level,) * len(dag),
            order=party_names,
            verbose=verbose,
        )

        self.n_parties = n_parties
        self.inflation_level = inflation_level
        self.party_names = tuple(self.names)
        self.source_names = tuple(self._actual_sources.tolist())
        self.local_dimensions_per_party = {
            party: int(local_dim_map[party]) for party in self.party_names
        }
        self.local_dims_per_party = tuple(
            self.local_dimensions_per_party[party] for party in self.party_names
        )

    @staticmethod
    def _default_party_names(n_parties: int) -> Tuple[str, ...]:
        """Generate readable party labels without requiring user input."""
        if n_parties <= len(ascii_uppercase):
            return tuple(ascii_uppercase[i] for i in range(n_parties))
        return tuple(f"X{i + 1}" for i in range(n_parties))

    @staticmethod
    def build_gnme_dag(party_names: Tuple[str, ...] | List[str]) -> Dict[str, List[str]]:
        """Build the k-source network where each source misses one party.

        For example:

        - ``(A, B, C)`` gives ``rho_AB, rho_AC, rho_BC``
        - ``(A, B, C, D)`` gives ``rho_ABC, rho_ABD, rho_ACD, rho_BCD``
        """
        ordered_names = tuple(map(str, party_names))
        sources = []
        for excluded_party in ordered_names:
            children = tuple(name for name in ordered_names if name != excluded_party)
            source_name = "rho_" + "".join(children)
            sources.append((source_name, list(children)))
        return dict(sorted(sources, key=lambda item: item[0]))

    @staticmethod
    def _normalize_local_dimensions(
        party_names: Tuple[str, ...],
        local_dims_per_party: int | Tuple[int, ...] | List[int] | Dict[str, int],
    ) -> Dict[str, int]:
        """Normalize local dimensions to a per-party dictionary.

        GNME uses trivial outcome/setting cardinalities internally, so the user
        must instead provide local Hilbert-space dimensions for the later
        state-SDP generation.
        """
        if isinstance(local_dims_per_party, int):
            if local_dims_per_party < 1:
                raise ValueError("Local dimensions must be positive.")
            return {party: int(local_dims_per_party) for party in party_names}

        if isinstance(local_dims_per_party, dict):
            missing = [party for party in party_names if party not in local_dims_per_party]
            if missing:
                raise ValueError(
                    "Missing local dimensions for parties: "
                    + ", ".join(map(str, missing))
                )
            normalized = {
                str(party): int(local_dims_per_party[party]) for party in party_names
            }
        else:
            if len(local_dims_per_party) != len(party_names):
                raise ValueError(
                    "The number of local dimensions must match the number of parties."
                )
            normalized = {
                party: int(dim) for party, dim in zip(party_names, local_dims_per_party)
            }

        if any(dim < 1 for dim in normalized.values()):
            raise ValueError("Local dimensions must be positive.")
        return normalized

    @cached_property
    def copy_labels_by_party(self) -> Dict[str, List[str]]:
        """Compact party-copy labels derived from the inflation indices.

        These labels are easier to read than the full native operator names
        because the setting/outcome data is trivial in the GNME setup.
        """
        labels: Dict[str, List[str]] = {}
        for party_name, index_rows in zip(self.names, self.inflation_indices_per_party):
            party_labels = []
            for row in index_rows:
                relevant_entries = [str(entry) for entry in row.tolist() if entry > 0]
                party_labels.append(f"{party_name}_{''.join(relevant_entries)}")
            labels[str(party_name)] = party_labels
        return labels

    @cached_property
    def operator_lexorder(self) -> Tuple[str, ...]:
        """Readable lexorder adapted to the trivial GNME operator structure."""
        labels = []
        for op_dict in self._lexrepr_to_dicts:
            party = str(op_dict["Party"])
            relevant_entries = [str(entry) for entry in op_dict["Relevant Copy Indices"]]
            labels.append(f"{party}_{''.join(relevant_entries)}")
        return tuple(labels)

    @cached_property
    def full_sequence_size(self) -> int:
        """Number of party-copy nodes in one full GNME inflation."""
        return self.n_parties * self.inflation_level

    @cached_property
    def party_full_sequence_supports(self) -> Dict[str, Tuple[Tuple[int, ...], ...]]:
        """All maximal nonfanout support patterns for each party.

        In GNME the compatibility graph is block-diagonal across parties, so a
        full inflation is obtained by choosing one maximal party pattern per
        party and taking their union. For one party, maximality means that each
        connected source copy appears exactly once across the chosen operators.
        """
        copy_range = tuple(range(1, self.inflation_level + 1))
        base_permutations = tuple(permutations(copy_range))
        label_to_index = {
            label: index for index, label in enumerate(self.operator_lexorder)
        }
        supports_by_party: Dict[str, Tuple[Tuple[int, ...], ...]] = {}

        for party in self.party_names:
            labels = tuple(self.copy_labels_by_party[party])
            if not labels:
                supports_by_party[party] = tuple()
                continue

            connected_sources = tuple(
                source for source in self.source_names
                if source in self.label_source_copies[labels[0]]
            )
            label_lookup = {
                tuple(self.label_source_copies[label][source] for source in connected_sources):
                label_to_index[label]
                for label in labels
            }

            party_supports = []
            for other_permutations in product(
                base_permutations,
                repeat=max(0, len(connected_sources) - 1),
            ):
                support = []
                for ref_copy in copy_range:
                    key = [ref_copy]
                    for permutation in other_permutations:
                        key.append(permutation[ref_copy - 1])
                    support.append(label_lookup[tuple(key)])
                party_supports.append(tuple(sorted(support)))
            supports_by_party[party] = tuple(party_supports)

        return supports_by_party

    @cached_property
    def full_nonfanout_sequences_as_supports(self) -> np.ndarray:
        """Full GNME inflations as fixed-width sorted supports."""
        per_party_supports = [
            self.party_full_sequence_supports[party]
            for party in self.party_names
        ]
        if not per_party_supports or any(len(supports) == 0 for supports in per_party_supports):
            return np.empty((0, self.full_sequence_size), dtype=np.int32)

        sequence_count = int(np.prod([len(supports) for supports in per_party_supports], dtype=np.int64))
        full_supports = np.empty((sequence_count, self.full_sequence_size), dtype=np.int32)
        for row, combination in enumerate(product(*per_party_supports)):
            merged_support = np.fromiter(
                (index for party_support in combination for index in party_support),
                dtype=np.int32,
                count=self.full_sequence_size,
            )
            merged_support.sort()
            full_supports[row] = merged_support
        return full_supports

    @cached_property
    def _lp_nonfanout(self):
        """Native LP helper used only to enumerate allowed nonfanout sequences.

        The LP code already knows how to:

        - generate the compatible nonfanout operator sequences,
        - apply the native inflation symmetries,
        - factorize sequences through the underlying InflationProblem.

        We keep the outcome/setting structure trivial, so ``include_all_outcomes``
        must be enabled or the LP collapses to the constant term only.
        """
        from .lp.InflationLP import InflationLP

        return InflationLP(
            self,
            nonfanout=True,
            include_all_outcomes=True,
            verbose=self.verbose,
        )

    @cached_property
    def raw_nonfanout_sequences_as_boolvecs(self) -> np.ndarray:
        """Legacy all-length nonfanout sequence generator.

        The GNME SDP pipeline now uses the maximal-template support path
        directly. This property is kept for compatibility with older probes.
        """
        return self._lp_nonfanout._raw_monomials_as_lexboolvecs.copy()

    @cached_property
    def full_nonfanout_sequences_as_boolvecs(self) -> np.ndarray:
        """Full GNME inflations as dense boolvecs.

        Dense boolvecs are preserved for compatibility, but the internal GNME
        combinatorial path now works from fixed-width support rows.
        """
        supports = self.full_nonfanout_sequences_as_supports
        bitvecs = np.zeros((len(supports), self._nr_operators), dtype=bool)
        if supports.size:
            rows = np.repeat(np.arange(len(supports), dtype=np.int32), supports.shape[1])
            bitvecs[rows, supports.reshape(-1)] = True
        return bitvecs

    def sequence_support_to_labels(self, support: np.ndarray) -> Tuple[str, ...]:
        """Convert a support-index representation into readable labels."""
        return tuple(self.operator_lexorder[i] for i in np.asarray(support, dtype=np.int32))

    def sequence_boolvec_to_labels(self, bitvec: np.ndarray) -> Tuple[str, ...]:
        """Convert a boolean support vector over the operator lexorder into labels."""
        return self.sequence_support_to_labels(np.flatnonzero(bitvec))

    def factorize_sequence_labels(self, labels: Tuple[str, ...]) -> Tuple[Tuple[str, ...], ...]:
        """Factorize a sequence into connected components using native inflation logic."""
        op_to_index = {label: i for i, label in enumerate(self.operator_lexorder)}
        lexmon = np.array([op_to_index[label] for label in labels], dtype=np.intc)
        factors = self.factorize_monomial_1d(lexmon, canonical_order=True)
        return tuple(
            tuple(self.operator_lexorder[i] for i in factor)
            for factor in factors
        )

    @cached_property
    def copy_only_symmetry_generators(self) -> np.ndarray:
        """Native copy-index symmetries on the operator lexorder.

        At this stage GNME keeps party labels fixed and quotients only by the
        same source-copy relabellings used by the original inflation package.
        """
        return np.array(self.symmetries, copy=True)

    def local_symmetry_permutations(
        self,
        sequence: Tuple[str, ...],
    ) -> Tuple[Tuple[int, ...], ...]:
        """Stabilizer of one maximal sequence as slot permutations.

        The native inflation package stores symmetries on the ambient operator
        lexorder. For the SDP we need the induced permutations on the slots of
        one maximal GNME variable.
        """
        lex_index = {label: i for i, label in enumerate(self.operator_lexorder)}
        representative_indices = [lex_index[label] for label in sequence]
        stabilizer = set()
        for perm in self.copy_only_symmetry_generators:
            image = tuple(self.operator_lexorder[perm[idx]] for idx in representative_indices)
            if sorted(image) != sorted(sequence):
                continue
            stabilizer.add(tuple(sequence.index(label) for label in image))
        return tuple(sorted(stabilizer))

    @cached_property
    def label_source_copies(self) -> Dict[str, Dict[str, int]]:
        """Map each compact GNME operator label to its source-copy indices."""
        mapping: Dict[str, Dict[str, int]] = {}
        for party_idx, party_name in enumerate(self.party_names):
            rows = self.inflation_indices_per_party[party_idx]
            for label, row in zip(self.copy_labels_by_party[party_name], rows):
                mapping[label] = {
                    self.source_names[source_idx]: int(copy_idx)
                    for source_idx, copy_idx in enumerate(row)
                    if int(copy_idx) > 0
                }
        return mapping

    @staticmethod
    def _labels_by_party(sequence: Tuple[str, ...]) -> Dict[str, Tuple[str, ...]]:
        """Group a sequence by party in lexicographic order."""
        grouped: Dict[str, List[str]] = {}
        for label in sequence:
            party_name = label.split("_", 1)[0]
            grouped.setdefault(party_name, []).append(label)
        return {
            party_name: tuple(sorted(labels))
            for party_name, labels in grouped.items()
        }

    def _is_source_consistent_subset(self, candidate: Tuple[str, ...]) -> bool:
        """Check if a one-copy-per-party subset is compatible with one marginal."""
        for source_name in self.source_names:
            selected_indices = [
                self.label_source_copies[label][source_name]
                for label in candidate
                if source_name in self.label_source_copies[label]
            ]
            if len(set(selected_indices)) > 1:
                return False
        return True

    def known_marginals(
        self,
        sequence: Tuple[str, ...],
        min_size: int = 2,
    ) -> Dict[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
        """Enumerate known one-copy-per-party marginals inside a GNME sequence.

        The keys are tuples of party names, e.g. ``('A', 'C')`` or
        ``('A', 'B', 'C')``. The values are all occurrences in the sequence
        that realize that same reduced marginal.
        """
        grouped = self._labels_by_party(sequence)
        marginal_map: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], ...]] = {}
        for size in range(min_size, len(self.party_names) + 1):
            for parties in combinations(self.party_names, size):
                occurrences = []
                for candidate in product(*(grouped[party] for party in parties)):
                    if self._is_source_consistent_subset(tuple(candidate)):
                        occurrences.append(tuple(candidate))
                if occurrences:
                    marginal_map[tuple(parties)] = tuple(sorted(occurrences))
        return marginal_map

    def subset_signature(self, subset: Iterable[str]) -> SubsetSignature:
        """Canonical signature for a reduced subsystem.

        The signature keeps:
        - the selected party-copy multiplicities,
        - for each source touched by at least two selected nodes, the partition
          of the selected nodes according to the source copy they carry.

        This matches the GNME relation we need later for cross-inflation SDP
        equalities: party labels are kept fixed, while copy labels are seen only
        through the native source-copy structure.
        """
        selected = tuple(sorted(subset))
        party_counts: Dict[str, int] = {}
        label_to_token: Dict[str, str] = {}
        node_tokens = []
        for label in selected:
            party = label.split("_", 1)[0]
            count = party_counts.get(party, 0) + 1
            party_counts[party] = count
            token = f"{party}{count}"
            label_to_token[label] = token
            node_tokens.append(token)

        source_blocks = []
        for source_name in self.source_names:
            groups: Dict[int, List[str]] = {}
            for label in selected:
                copy_map = self.label_source_copies[label]
                if source_name in copy_map:
                    groups.setdefault(copy_map[source_name], []).append(label_to_token[label])
            if sum(len(group) for group in groups.values()) < 2:
                continue
            canonical_groups = tuple(
                sorted(tuple(sorted(group)) for group in groups.values())
            )
            source_blocks.append((source_name, canonical_groups))
        return tuple(node_tokens), tuple(sorted(source_blocks))

    def subset_occurrence(
        self,
        inflation_index: int,
        labels: Iterable[str],
    ) -> GNMESubsetOccurrence:
        """Build one subset occurrence with SDP-relevant metadata."""
        inflation = self.inflations[inflation_index]
        canonical_labels = tuple(
            sorted(labels, key=lambda label: inflation.lexorder_lookup[label])
        )
        positions = tuple(inflation.lexorder_lookup[label] for label in canonical_labels)
        parties = tuple(label.split("_", 1)[0] for label in canonical_labels)
        is_known_marginal = canonical_labels in inflation.known_marginals.get(parties, tuple())
        is_atomic_known = canonical_labels in inflation.atomic_known_subsets
        return GNMESubsetOccurrence(
            inflation_index=inflation_index,
            inflation_name=f"inflation_{inflation_index}",
            labels=canonical_labels,
            positions=positions,
            parties=parties,
            signature=self.subset_signature(canonical_labels),
            factorization=self.factorize_sequence_labels(canonical_labels),
            is_known_marginal=is_known_marginal,
            is_atomic_known=is_atomic_known,
        )

    def subset_occurrences(
        self,
        inflation_index: int,
        subset_sizes: Tuple[int, ...] = (2, 3, 4),
        include_known_marginals: bool = True,
    ) -> Tuple[GNMESubsetOccurrence, ...]:
        """Enumerate subset occurrences inside one maximal inflation.

        This is the GNME analogue of collecting internal monomial information:
        one place that keeps together subset positions, factorization, and
        whether the subset is already a known marginal.
        """
        inflation = self.inflations[inflation_index]
        occurrences: Dict[Tuple[int, ...], GNMESubsetOccurrence] = {}
        if include_known_marginals:
            for known_occurrences in inflation.known_marginals.values():
                for labels in known_occurrences:
                    occurrence = self.subset_occurrence(inflation_index, labels)
                    occurrences[occurrence.positions] = occurrence

        for subset_size in subset_sizes:
            if subset_size > len(inflation.representative):
                continue
            for subset in combinations(inflation.representative, subset_size):
                occurrence = self.subset_occurrence(inflation_index, subset)
                occurrences[occurrence.positions] = occurrence

        return tuple(
            occurrence
            for _, occurrence in sorted(occurrences.items(), key=lambda item: item[0])
        )

    def shared_subset_classes(
        self,
        subset_sizes: Tuple[int, ...] = (2, 3, 4),
        include_known_marginals: bool = True,
        min_occurrences: int = 2,
        min_inflations: int = 1,
    ) -> Tuple[GNMESharedSubsetClass, ...]:
        """Group subset occurrences shared across maximal GNME inflations.

        This packages the information needed later to relate different SDP
        variables. The grouping is independent of whether a subset is known or
        unknown; known subsets are simply marked inside each occurrence.
        """
        grouped: Dict[SubsetSignature, List[GNMESubsetOccurrence]] = {}
        for inflation_index in range(len(self.inflations)):
            for occurrence in self.subset_occurrences(
                inflation_index,
                subset_sizes=subset_sizes,
                include_known_marginals=include_known_marginals,
            ):
                grouped.setdefault(occurrence.signature, []).append(occurrence)

        classes = []
        for signature, occurrences in grouped.items():
            unique_occurrences = tuple(sorted(
                set(occurrences),
                key=lambda occ: (
                    occ.inflation_index,
                    occ.positions,
                    occ.labels,
                ),
            ))
            if len(unique_occurrences) < min_occurrences:
                continue
            if len({occ.inflation_index for occ in unique_occurrences}) < min_inflations:
                continue
            classes.append(
                GNMESharedSubsetClass(
                    signature=signature,
                    occurrences=unique_occurrences,
                )
            )
        return tuple(sorted(
            classes,
            key=lambda subset_class: (
                len(subset_class.signature[0]),
                subset_class.signature,
                len(subset_class.occurrences),
            ),
        ))

    def sdp_blueprint(
        self,
        subset_sizes: Tuple[int, ...] = (2, 3, 4),
        include_known_marginals: bool = True,
        min_shared_occurrences: int = 2,
        min_shared_inflations: int = 2,
        variable_prefix: str = "tau",
        local_dims_per_party: Dict[str, int] | Tuple[int, ...] | List[int] | int | None = None,
    ) -> GNMESDPBlueprint:
        """Package SDP-facing structural data for the GNME hierarchy.

        This mirrors the role that the inflation LP code plays for probability
        variables: it collects the combinatorial data needed before assigning
        numerical values. The output is generic and agnostic about whether a
        shared subset is known, unknown, or semi-known.
        """
        if local_dims_per_party is None:
            local_dim_map = self.local_dimensions_per_party
        else:
            local_dim_map = self._normalize_local_dimensions(
                self.party_names,
                local_dims_per_party,
            )

        variables = []
        for inflation_index, inflation in enumerate(self.inflations):
            fixed_knowns = {
                parties: tuple(
                    self.subset_occurrence(inflation_index, labels)
                    for labels in occurrences
                )
                for parties, occurrences in inflation.known_marginals.items()
            }
            variables.append(
                GNMESDPVariable(
                    inflation_index=inflation_index,
                    name=f"{variable_prefix}_{inflation_index}",
                    lexorder=inflation.representative,
                    lexorder_lookup=inflation.lexorder_lookup,
                    local_symmetry_perms=inflation.local_symmetry_perms,
                    factorization=inflation.factorization,
                    fixed_known_marginals=fixed_knowns,
                    slot_dims=tuple(
                        local_dim_map[label.split("_", 1)[0]]
                        for label in inflation.representative
                    ),
                )
            )

        shared_classes = self.shared_subset_classes(
            subset_sizes=subset_sizes,
            include_known_marginals=include_known_marginals,
            min_occurrences=min_shared_occurrences,
            min_inflations=min_shared_inflations,
        )
        return GNMESDPBlueprint(
            party_dims={party: int(dim) for party, dim in local_dim_map.items()},
            variables=tuple(variables),
            shared_subset_classes=shared_classes,
        )

    def atomic_known_subsets(self, sequence: Tuple[str, ...]) -> Tuple[Tuple[str, ...], ...]:
        """Top-level known marginals, one subsystem per original party."""
        grouped = self._labels_by_party(sequence)
        ordered_party_labels = [grouped[party] for party in self.party_names]
        atomic_knowns = []
        for candidate in product(*ordered_party_labels):
            if self._is_source_consistent_subset(tuple(candidate)):
                atomic_knowns.append(tuple(candidate))
        return tuple(sorted(atomic_knowns))

    @staticmethod
    def _orbit_of_support(support: np.ndarray,
                          symmetries: np.ndarray,
                          lookup: Dict[bytes, int]) -> Tuple[int, ...]:
        """Return the symmetry orbit of one fixed-width support row."""
        moved_supports = np.sort(symmetries[:, support], axis=1).astype(np.int32, copy=False)
        orbit = {
            lookup[support_hash]
            for support_hash in (
                ndarray_bytes_key(moved, dtype=np.int32) for moved in moved_supports
            )
            if support_hash in lookup
        }
        return tuple(sorted(orbit))

    @cached_property
    def full_nonfanout_sequence_orbits(self) -> Tuple[Tuple[Tuple[str, ...], ...], ...]:
        """Group the full nonfanout sequences into copy-index symmetry orbits."""
        supports = self.full_nonfanout_sequences_as_supports
        if len(supports) == 0:
            return tuple()

        lookup = {
            ndarray_bytes_key(support, dtype=np.int32): i
            for i, support in enumerate(supports)
        }
        unseen = set(range(len(supports)))
        orbits = []
        while unseen:
            seed = min(unseen)
            orbit_indices = self._orbit_of_support(
                supports[seed],
                self.copy_only_symmetry_generators,
                lookup,
            )
            unseen.difference_update(orbit_indices)
            orbit_labels = tuple(
                self.sequence_support_to_labels(supports[i])
                for i in orbit_indices
            )
            orbit_labels = tuple(sorted(orbit_labels))
            orbits.append(orbit_labels)
        return tuple(sorted(orbits, key=lambda orb: orb[0]))

    @cached_property
    def minimal_nonfanout_sequences(self) -> Tuple[Tuple[str, ...], ...]:
        """One canonical representative for each full nonfanout symmetry class."""
        return tuple(orbit[0] for orbit in self.full_nonfanout_sequence_orbits)

    @cached_property
    def inflations(self) -> Tuple[GNMEInflation, ...]:
        """Structured maximal nonfanout inflations for later SDP construction."""
        inflations = []
        for orbit in self.full_nonfanout_sequence_orbits:
            representative = orbit[0]
            inflations.append(
                GNMEInflation(
                    representative=representative,
                    orbit=orbit,
                    factorization=self.factorize_sequence_labels(representative),
                    atomic_known_subsets=self.atomic_known_subsets(representative),
                    known_marginals=self.known_marginals(representative),
                    lexorder_lookup={
                        label: idx for idx, label in enumerate(representative)
                    },
                    local_symmetry_perms=self.local_symmetry_permutations(representative),
                )
            )
        return tuple(inflations)
