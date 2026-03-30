"""Thin application-level wrapper for the package GNME SDP builder.

The implementation now lives in `gnme_inflation.gnme_inflation.sdp.GNMEStateSDP`.
This file remains only as a convenient entry point for local inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gnme_inflation"))

from gnme_inflation import (  # noqa: E402,F401
    AssignedStateSDPDraft,
    FusionSolveResult,
    FusionStateSDPModel,
    StateSDPDraft,
    build_fusion_feasibility_model,
    build_sdp_draft,
    print_assigned_values,
    print_draft,
    set_values,
    solve_fusion_feasibility,
    GNMEProblem,
)


def main(n_parties: int, inflation_level: int) -> None:
    problem = GNMEProblem(n_parties=n_parties, inflation_level=inflation_level)
    model = build_sdp_draft(problem)
    print_draft(model)


if __name__ == "__main__":
    n_parties = 3
    inflation_level = 2
    main(n_parties, inflation_level)
