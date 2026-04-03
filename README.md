# SDP_Quantum_Matter

Research code for GNME state-SDP generation and examples based on the inflation framework.

## Repository Layout

- `gnme_inflation/`: GNME package and package metadata.
- `applications_QM/`: runnable GHZ and package examples.
- `inflation/`: upstream inflation package, tracked as a submodule.

## Current Examples

- `applications_QM/GHZ_level2_real.py`: level-2 noisy-GHZ optimization with the default real formulation.
- `applications_QM/GHZ_level2_hermitian.py`: level-2 noisy-GHZ optimization with the Hermitian formulation.
- `applications_QM/W_level2_real.py`: level-2 noisy-W optimization with the default real formulation.
- `applications_QM/W_level2_hermitian.py`: level-2 noisy-W optimization with the Hermitian formulation.

## Quick Start

From the repository root:

```bash
python3 applications_QM/GHZ_level2_real.py
python3 applications_QM/GHZ_level2_hermitian.py
python3 applications_QM/W_level2_real.py
python3 applications_QM/W_level2_hermitian.py
```

The examples import the local package directly from `gnme_inflation/`.

## Notes

- The default public API in `gnme_inflation` uses the block-symmetrized backend.
- `Hermitian=False` is the default formulation in the package wrappers.
- The level-2 GHZ and W scripts are the current runnable examples in `applications_QM/`.
