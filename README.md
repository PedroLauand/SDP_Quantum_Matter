# SDP_Quantum_Matter

Research code for GNME state-SDP generation on top of a vendored inflation fork.

## Repository layout

- `gnme_inflation/`: the installable GNME package.
- `applications_QM/`: runnable GHZ/W examples and lightweight benchmarks.
- `inflation/`: upstream inflation package, tracked as a submodule for reference.

## Package structure

The current GNME stack is split into three layers:

1. `gnme_inflation/GNMEProblem.py`
   Builds maximal nonfanout inflations, shared subset classes, local symmetry data, and the structural SDP blueprint.
2. `gnme_inflation/sdp/GNMEStateSDP.py`
   Builds readable state-SDP drafts and exposes the default package API. The package-level `build_fusion_feasibility_model(...)` entry point dispatches to the block backend by default.
3. `gnme_inflation/sdp/GNMEBlockStateSDP.py`
   Solver-facing block backend. This is the main optimized path for real and Hermitian state-SDP builds.

## Current examples

- `applications_QM/GHZ_level2_real.py`: level-2 noisy-GHZ optimization with the default real formulation.
- `applications_QM/GHZ_level2_hermitian.py`: level-2 noisy-GHZ optimization with the Hermitian formulation.
- `applications_QM/W_level2_real.py`: level-2 noisy-W optimization with the default real formulation.
- `applications_QM/W_level2_hermitian.py`: level-2 noisy-W optimization with the Hermitian formulation.
- `applications_QM/benchmark_gnme_frontend.py`: front-end benchmark for maximal-sequence generation and draft construction.

## Quick start

From the repository root:

```bash
python3 applications_QM/GHZ_level2_real.py
python3 applications_QM/GHZ_level2_hermitian.py
python3 applications_QM/W_level2_real.py
python3 applications_QM/W_level2_hermitian.py
python3 applications_QM/benchmark_gnme_frontend.py --levels 2 3
```

The examples import the local package directly from `gnme_inflation/`.

## Current backend status

- The default public API uses the block-symmetrized backend.
- `Hermitian=False` is the default formulation in the package wrappers and currently gets the most optimization work.
- The GNME front-end no longer goes through the inherited all-monomial LP path to generate maximal nonfanout inflations.
- Real-mode representative constraints are emitted directly from tau reduced maps instead of through separate auxiliary PSD representative variables.
- Real-mode PPT constraints are emitted as direct `svec` cone conditions, with shared-representative symmetry quotienting applied before model build.

## Level-3 status

- Level 3 front-end generation and draft construction are now fast enough to use routinely.
- The remaining bottleneck is the level-3 real PPT stage in `GNMEBlockStateSDP.py`, especially the large `tau_*` partial-transpose constraints after representative emission.
