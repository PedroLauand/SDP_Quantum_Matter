"""Benchmark the GNME combinatorial front-end without solving the SDP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import GNMEProblem, build_sdp_draft  # noqa: E402


def benchmark_level(level: int, local_dim: int, verbose: int) -> None:
    print(f"[benchmark] level={level} local_dim={local_dim}", flush=True)
    problem = GNMEProblem(
        n_parties=3,
        inflation_level=level,
        local_dims_per_party=local_dim,
        verbose=verbose,
    )

    t0 = perf_counter()
    supports = problem.full_nonfanout_sequences_as_supports
    print(
        "[benchmark] full supports: "
        f"count={len(supports)} shape={supports.shape} "
        f"dt={perf_counter() - t0:.2f}s"
    , flush=True)

    t1 = perf_counter()
    orbits = problem.full_nonfanout_sequence_orbits
    print(
        "[benchmark] full orbits: "
        f"count={len(orbits)} dt={perf_counter() - t1:.2f}s"
    , flush=True)

    t2 = perf_counter()
    draft = build_sdp_draft(problem, verbose=verbose)
    print(
        "[benchmark] draft: "
        f"psd={len(draft.psd_variables)} "
        f"aux={len(draft.auxiliary_representatives)} "
        f"known={len(draft.known_representatives)} "
        f"ppt={len(draft.ppt_variables)} "
        f"dt={perf_counter() - t2:.2f}s"
    , flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Inflation levels to benchmark.",
    )
    parser.add_argument(
        "--local-dim",
        type=int,
        default=2,
        help="Per-party local Hilbert-space dimension.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="GNME verbosity level.",
    )
    args = parser.parse_args()

    for level in args.levels:
        benchmark_level(level, args.local_dim, args.verbose)


if __name__ == "__main__":
    main()
