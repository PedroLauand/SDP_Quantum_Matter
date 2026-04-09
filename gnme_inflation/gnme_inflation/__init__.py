"""Public package surface for the GNME inflation fork.

The package exposes three layers:

1. combinatorial GNME problem generation via :class:`GNMEProblem`,
2. readable state-SDP drafts via :func:`build_sdp_draft`,
3. solver-facing builders, with the minimized MOSEK Task backend as the
   default GNME path.

The upstream inflation LP/SDP modules are still vendored here because GNME
reuses part of their infrastructure, but the package-level defaults are GNME
state-SDP oriented.
"""

import importlib
import sys
from threading import RLock

def _configure_tqdm_lock() -> None:
    """Force tqdm to use a thread-only lock.

    The default POSIX tqdm lock is a multiprocessing ``RLock``. On Python 3.8
    that can trigger a leaked-semaphore warning at interpreter shutdown even in
    single-process CLI runs.
    """
    lock = RLock()
    try:
        from tqdm import tqdm as _std_tqdm
        _std_tqdm.set_lock(lock)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        from tqdm.auto import tqdm as _auto_tqdm
        _auto_tqdm.set_lock(lock)
    except Exception:  # pragma: no cover - optional dependency
        pass

_configure_tqdm_lock()

from .InflationProblem import InflationProblem
from .GNMEProblem import (
    GNMEInflation,
    GNMEOverlapRoute,
    GNMEProblem,
    GNMESDPBlueprint,
    GNMESDPVariable,
    GNMESharedSubsetClass,
    GNMESubsetOccurrence,
    GNMETopDownBlueprint,
    GNMETopDownFamily,
)
from .sdp.InflationSDP import InflationSDP
from .sdp.GNMEStateSDP import (
    AssignedStateSDPDraft,
    CrossInflationConstraintGroupDraft,
    FusionSolveResult,
    FusionStateSDPModel,
    StateSDPDraft,
    TopDownStateSDPDraft,
    build_fusion_feasibility_model,
    build_legacy_fusion_feasibility_model,
    build_smaller_sdp_draft,
    build_sdp_draft,
    build_top_down_sdp_draft,
    print_assigned_values,
    print_draft,
    quotient_ppt_constraints,
    quotient_representative_constraints,
    set_values,
    solve_fusion_feasibility,
    solve_legacy_fusion_feasibility,
    validate_top_down_draft,
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
from .sdp.GNMETaskStateSDP import (
    TaskBarVariableSpec,
    TaskBlockStateSDPModel,
    build_block_task_feasibility_model,
    build_top_down_block_task_feasibility_model,
)
from .lp.InflationLP import InflationLP
from .optimization_utils import max_within_feasible
from ._about import about
from ._version import __version__

# Compatibility alias for older numba caches that still import ``gnme_inflation.sdp``.
sys.modules.setdefault("gnme_inflation.sdp", sys.modules[f"{__name__}.sdp"])
sys.modules.setdefault(
    "gnme_inflation.GNMEProblem",
    importlib.import_module(f"{__name__}.GNMEProblem"),
)
sys.modules.setdefault(
    "gnme_inflation.InflationProblem",
    importlib.import_module(f"{__name__}.InflationProblem"),
)

DEFAULT_GNME_SDP_BACKEND = "task"
build_default_feasibility_model = build_block_task_feasibility_model

__all__ = [
    "DEFAULT_GNME_SDP_BACKEND",
    "InflationProblem",
    "GNMEProblem",
    "GNMEInflation",
    "GNMESDPVariable",
    "GNMESDPBlueprint",
    "GNMESubsetOccurrence",
    "GNMESharedSubsetClass",
    "GNMEOverlapRoute",
    "GNMETopDownFamily",
    "GNMETopDownBlueprint",
    "StateSDPDraft",
    "TopDownStateSDPDraft",
    "AssignedStateSDPDraft",
    "CrossInflationConstraintGroupDraft",
    "FusionStateSDPModel",
    "FusionSolveResult",
    "BlockSectorDraft",
    "BlockVariableDraft",
    "GNMEBlockSDPDraft",
    "BlockTauVariableData",
    "FusionBlockStateSDPModel",
    "TaskBarVariableSpec",
    "TaskBlockStateSDPModel",
    "build_sdp_draft",
    "build_top_down_sdp_draft",
    "build_smaller_sdp_draft",
    "build_block_sdp_draft",
    "build_default_feasibility_model",
    "build_fusion_feasibility_model",
    "build_legacy_fusion_feasibility_model",
    "build_block_fusion_feasibility_model",
    "build_block_task_feasibility_model",
    "build_top_down_block_task_feasibility_model",
    "set_values",
    "solve_fusion_feasibility",
    "solve_legacy_fusion_feasibility",
    "solve_block_fusion_feasibility",
    "print_draft",
    "print_assigned_values",
    "quotient_ppt_constraints",
    "quotient_representative_constraints",
    "print_block_draft",
    "validate_top_down_draft",
    "InflationSDP",
    "InflationLP",
    "max_within_feasible",
]
