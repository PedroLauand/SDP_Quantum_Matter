"""Exploration file for the first GNMEProblem wrapper.

This file is meant for quick test runs while the package code is still being
adapted. It should stay lightweight and avoid modifying the package internals.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import GNMEProblem  # noqa: E402


def describe_problem(problem: GNMEProblem) -> None:
    """Print the GNME data built from the native nonfanout inflation."""
    print("GNMEProblem snapshot")
    print(f"  n_parties: {problem.n_parties}")
    print(f"  inflation_level: {problem.inflation_level}")
    print(f"  local_dims_per_party: {problem.local_dimensions_per_party}")
    print(f"  parties: {list(problem.party_names)}")
    print(f"  sources: {list(problem.source_names)}")
    print("  dag:")
    for source_name, children in problem.dag.items():
        print(f"    {source_name}: {sorted(children)}")
    print("  hypergraph:")
    print(problem.hypergraph.astype(int))
    print()

    print("Party-copy labels")
    for party_name, labels in problem.copy_labels_by_party.items():
        print(f"  {party_name}: {labels}")
    print()

    print("Operator lexorder")
    print(f"  length: {len(problem.operator_lexorder)}")
    print(f"  labels: {list(problem.operator_lexorder)}")
    print()

    print("Full nonfanout sequence orbits")
    print(f"  full sequence size: {problem.full_sequence_size}")
    print(f"  raw full sequences: {len(problem.full_nonfanout_sequences_as_boolvecs)}")
    print(f"  orbit count: {len(problem.full_nonfanout_sequence_orbits)}")
    for orbit_id, inflation in enumerate(problem.inflations):
        print(f"  orbit {orbit_id}:")
        print(f"    representative: {' '.join(inflation.representative)}")
        print(f"    orbit size: {len(inflation.orbit)}")
        print(f"    local symmetries: {inflation.local_symmetry_perms}")
        print(f"    factors: {inflation.factorization}")
        print("    known marginals:")
        for parties, occurrences in inflation.known_marginals.items():
            print(f"      {parties}:")
            for occurrence in occurrences:
                print(f"        {' '.join(occurrence)}")
    print()

    print("Shared subset classes")
    for subset_class in problem.shared_subset_classes(subset_sizes=(2, 3)):
        inflations = sorted({occ.inflation_name for occ in subset_class.occurrences})
        if len(inflations) < 2:
            continue
        print(f"  signature: {subset_class.signature}")
        for occurrence in subset_class.occurrences:
            flags = []
            if occurrence.is_known_marginal:
                flags.append("known")
            if occurrence.is_atomic_known:
                flags.append("atomic")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            print(
                "    "
                f"{occurrence.inflation_name} "
                f"positions={occurrence.positions} "
                f"labels={' '.join(occurrence.labels)}"
                f"{suffix}"
            )
    print()


def main(n_parties: int, inflation_level: int) -> None:
    problem = GNMEProblem(n_parties=n_parties, inflation_level=inflation_level)
    describe_problem(problem)


if __name__ == "__main__":
    n_parties = 3
    inflation_level = 2
    main(n_parties, inflation_level)
