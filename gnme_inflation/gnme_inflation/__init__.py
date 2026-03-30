"""Inflation
==================
Provides
 1. A tool for describing causal scenarios and reducing them to network form
    (see the definitions of network and non-network causal scenarios in
    arXiv:1707.06476 and arXiv:1909.10519).
 2. A tool for setting up and solving feasibility and optimization problems
    over probability distributions compatible with quantum causal scenarios.
 3. A tool for setting up and solving feasibility and optimization problems
    over probability distributions compatible with theory-independent and
    classical causal scenarios
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

__all__ = ["InflationProblem",
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
           "quotient_representative_constraints",
           "print_block_draft",
           "InflationSDP",
           "InflationLP",
           "max_within_feasible"]
