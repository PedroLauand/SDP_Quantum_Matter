"""Public package surface for the GNME inflation fork.

The package exposes three layers:

1. combinatorial GNME problem generation via :class:`GNMEProblem`,
2. readable state-SDP drafts via :func:`build_sdp_draft`,
3. solver-facing MOSEK Fusion builders, with the block backend as default.

The upstream inflation LP/SDP modules are still vendored here because GNME
reuses part of their infrastructure, but the package-level defaults are GNME
state-SDP oriented.
"""

from .InflationProblem import InflationProblem
from .GNMEProblem import (
    GNMEInflation,
    GNMEProblem,
    GNMESDPBlueprint,
    GNMESDPVariable,
    GNMESharedSubsetClass,
    GNMESubsetOccurrence,
)
from .sdp.InflationSDP import InflationSDP
from .sdp.GNMEStateSDP import (
    AssignedStateSDPDraft,
    FusionSolveResult,
    FusionStateSDPModel,
    StateSDPDraft,
    build_fusion_feasibility_model,
    build_legacy_fusion_feasibility_model,
    build_sdp_draft,
    print_assigned_values,
    print_draft,
    quotient_ppt_constraints,
    quotient_representative_constraints,
    set_values,
    solve_fusion_feasibility,
    solve_legacy_fusion_feasibility,
)
from .sdp.GNMEBlockSDPDraft import (
    BlockSectorDraft,
    BlockVariableDraft,
    GNMEBlockSDPDraft,
    build_block_sdp_draft,
    print_block_draft,
)
from .sdp.GNMEBlockStateSDP import (
    BlockTauVariableData,
    FusionBlockStateSDPModel,
    build_block_fusion_feasibility_model,
    solve_block_fusion_feasibility,
)
from .lp.InflationLP import InflationLP
from .optimization_utils import max_within_feasible
from ._about import about
from ._version import __version__

__all__ = [
    "InflationProblem",
    "GNMEProblem",
    "GNMEInflation",
    "GNMESDPVariable",
    "GNMESDPBlueprint",
    "GNMESubsetOccurrence",
    "GNMESharedSubsetClass",
    "StateSDPDraft",
    "AssignedStateSDPDraft",
    "FusionStateSDPModel",
    "FusionSolveResult",
    "BlockSectorDraft",
    "BlockVariableDraft",
    "GNMEBlockSDPDraft",
    "BlockTauVariableData",
    "FusionBlockStateSDPModel",
    "build_sdp_draft",
    "build_block_sdp_draft",
    "build_fusion_feasibility_model",
    "build_legacy_fusion_feasibility_model",
    "build_block_fusion_feasibility_model",
    "set_values",
    "solve_fusion_feasibility",
    "solve_legacy_fusion_feasibility",
    "solve_block_fusion_feasibility",
    "print_draft",
    "print_assigned_values",
    "quotient_ppt_constraints",
    "quotient_representative_constraints",
    "print_block_draft",
    "InflationSDP",
    "InflationLP",
    "max_within_feasible",
]
