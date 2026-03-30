# SDP_Quantum_Matter

Research code for GNME state-SDP generation and examples based on the inflation framework.

## Repository Layout

- `gnme_inflation/`: GNME package and package metadata.
- `applications_QM/`: runnable GHZ and package examples.
- `inflation/`: upstream inflation package, tracked as a submodule.

## Current Examples

- `applications_QM/GHZ_example_level2.py`: level-2 noisy-GHZ optimization with the default real block backend.
- `applications_QM/GHZ_block_example_level2.py`: level-2 noisy-GHZ optimization with the Hermitian block backend.
- `applications_QM/GHZ_block_example_level3.py`: level-3 noisy-GHZ optimization with the default real block backend.

## Quick Start

From the repository root:

```bash
python3 applications_QM/GHZ_example_level2.py
python3 applications_QM/GHZ_block_example_level2.py
python3 applications_QM/GHZ_block_example_level3.py
```

The examples import the local package directly from `gnme_inflation/`.

## Notes

- The default public API in `gnme_inflation` uses the block-symmetrized backend.
- `Hermitian=False` is the default formulation in the package wrappers.
- The level-3 build path is still under active optimization.
