"""Explore the native InflationLP monomial layer for GNME-style inputs.

This file is intentionally external to the package so we can inspect the LP
machinery without committing to package-level design changes yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from string import ascii_uppercase


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import InflationLP, InflationProblem  # noqa: E402


def default_party_names(n_parties: int) -> tuple[str, ...]:
    if n_parties <= len(ascii_uppercase):
        return tuple(ascii_uppercase[i] for i in range(n_parties))
    return tuple(f"X{i + 1}" for i in range(n_parties))


def build_missing_one_party_dag(n_parties: int) -> dict[str, list[str]]:
    """k sources, each connected to exactly k-1 parties."""
    party_names = default_party_names(n_parties)
    sources = []
    for excluded in party_names:
        children = [party for party in party_names if party != excluded]
        sources.append(("rho_" + "".join(children), children))
    return dict(sorted(sources, key=lambda item: item[0]))


def build_lp_problem(n_parties: int, inflation_level: int) -> tuple[InflationProblem, InflationLP]:
    dag = build_missing_one_party_dag(n_parties)
    problem = InflationProblem(
        dag=dag,
        outcomes_per_party=(1,) * n_parties,
        settings_per_party=(1,) * n_parties,
        inflation_level_per_source=(inflation_level,) * len(dag),
        verbose=0,
    )
    # With fully trivial local structure, the LP collapses to the constant term
    # unless all outcomes are explicitly kept.
    lp = InflationLP(
        problem,
        nonfanout=True,
        include_all_outcomes=True,
        verbose=0,
    )
    return problem, lp


def print_snapshot(problem: InflationProblem, lp: InflationLP, max_monomials: int = 20) -> None:
    print("InflationProblem")
    print(f"  parties: {list(problem.names)}")
    print(f"  sources: {problem._actual_sources.tolist()}")
    print("  hypergraph:")
    print(problem.hypergraph.astype(int))
    print(f"  operator lexorder size: {problem._nr_operators}")
    print(f"  operator lexorder names: {problem._lexrepr_to_names.tolist()}")
    print()

    print("InflationLP")
    print(f"  raw_n_columns: {lp.raw_n_columns}")
    print(f"  n_columns: {lp.n_columns}")
    print(f"  knowable/semi/unknowable: {lp.n_knowable}/{lp.n_something_knowable}/{lp.n_unknowable}")
    print()

    print("First monomials")
    for mon in lp.monomials[:max_monomials]:
        factors = tuple(factor.name for factor in mon.factors)
        print(f"  {mon.name}")
        print(f"    status={mon.knowability_status}, atomic={mon.is_atomic}, factors={factors}")
    print()

    print("Sample factorization conditions")
    for i, (name, factor_names) in enumerate(lp.factorization_conditions_by_name.items()):
        if i >= max_monomials:
            break
        print(f"  {name} -> {factor_names}")


def main(n_parties: int, inflation_level: int) -> None:
    problem, lp = build_lp_problem(n_parties, inflation_level)
    print_snapshot(problem, lp)


if __name__ == "__main__":
    n_parties = 3
    inflation_level = 2
    main(n_parties, inflation_level)
