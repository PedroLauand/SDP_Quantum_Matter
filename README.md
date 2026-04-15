# Baroque Test Paper

This repository now contains a single self-contained sandbox for explicit SCS builds of small GNME inflation SDPs:

- [Baroque_test_paper](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper)

The code is organized to keep the **mathematical SDP declaration** readable and to keep the **coordinate-action / sparse SCS backend** separate.

## What Is In The Folder

Core model files:

- [level3_complex_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level3_complex_model.py)
- [level3_real_restricted_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level3_real_restricted_model.py)
- [level2_complex_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_complex_model.py)
- [level2_real_restricted_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_real_restricted_model.py)
- [level2_fourpartite_complex_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_fourpartite_complex_model.py)
- [level2_fourpartite_real_restricted_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_fourpartite_real_restricted_model.py)
- [level3_scs_backend.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level3_scs_backend.py)

Runnable example scripts:

- [inflation_robustness_level2.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2.py)
- [inflation_robustness_level2_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_real_restricted.py)
- [inflation_robustness_level2_w.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w.py)
- [inflation_robustness_level2_w_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w_real_restricted.py)
- [inflation_robustness_level3.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3.py)
- [inflation_robustness_level3_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_real_restricted.py)
- [inflation_robustness_level3_w.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w.py)
- [inflation_robustness_level3_w_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w_real_restricted.py)

Reference SCS logs:

- [inflation_robustness_level2_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_run_reference.txt)
- [inflation_robustness_level2_real_restricted_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_real_restricted_run_reference.txt)
- [inflation_robustness_level2_w_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w_run_reference.txt)
- [inflation_robustness_level2_w_real_restricted_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w_real_restricted_run_reference.txt)
- [inflation_robustness_level3_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_run_reference.txt)
- [inflation_robustness_level3_real_restricted_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_real_restricted_run_reference.txt)
- [inflation_robustness_level3_w_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w_run_reference.txt)
- [inflation_robustness_level3_w_real_restricted_run_reference.txt](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w_real_restricted_run_reference.txt)

## Mathematical Formulation

### Level 2, Tripartite

The level-2 tripartite model uses two PSD variables:

- `sigma`: two disconnected triangle copies
- `tau`: one connected 6-ring

The optimization variable is `p`, with objective

```text
min p
```

and observed anchor

```text
Tr_{A22 B22 C22}(sigma) = (1-p) rho_ABC + p I / D
```

Additional constraints:

- `Tr(sigma) = 1`
- `Tr(tau) = 1`
- internal symmetry of `sigma` under swapping the two disconnected triangles
- internal symmetry of `tau` under the half-turn of the 6-ring
- three cross-inflation consistency equalities:
  - `ABAB`
  - `BCBC`
  - `CACA`
- one full PPT constraint on `sigma` across `(A11 B11 C11) | (A22 B22 C22)`

This is the formulation that reproduces the expected level-2 values for GHZ and W.

### Level 3, Tripartite

The level-3 tripartite model uses three PSD variables:

- `tau`: three disconnected triangle copies
- `gamma`: one triangle copy plus one disconnected 6-ring
- `sigma`: one connected 9-ring

The optimization variable is `t`, with observed anchor

```text
Tr_{A2 B2 C2 A3 B3 C3}(sigma) = t rho_ABC + (1-t) I / D
```

Additional constraints:

- `Tr(tau) = Tr(gamma) = Tr(sigma) = 1`
- internal symmetries from the older MATLAB-style formulation
- three `gamma <-> tau` seven-party marginal equalities
- three `tau <-> sigma` seven-party marginal equalities

Current PPT choice in the active code:

- only the two full PPT constraints are active
  - one on `tau`
  - one on `sigma`
- the reduced PPT auxiliaries from the older 14-PPT family are still documented in the model files, but they are disabled in the active build

This is deliberate. The reference logs in this repository correspond to the **current simplified 2-PPT level-3 build**, not to the older 14-PPT variant.

### Level 2, Four-Partite

There is also a four-partite level-2 builder:

- `sigma`: two disconnected tetrahedra on 8 slots
- `tau`: one connected 8-ring on 8 slots

with:

- one observed four-party anchor on `sigma`
- four six-party cross-inflation equalities (`ABC`, `ABD`, `ACD`, `BCD`)
- one full PPT on `sigma`

These files are:

- [level2_fourpartite_complex_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_fourpartite_complex_model.py)
- [level2_fourpartite_real_restricted_model.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level2_fourpartite_real_restricted_model.py)

They are mathematically defined and smoke-tested for small `dims4`, but they are not practical for large grouped targets such as the 8-qubit Dicke state grouped into four 2-qubit parties. That target would require much stronger symmetry reduction than the present full-coordinate Baroque backend provides.

## Implementation Choices

### 1. Readable formulation first

The model files are written so that the top-level build reads like the SDP:

- declare variables
- add trace constraints
- add symmetry constraints
- add observed anchor
- add cross-inflation equalities
- add PPT links
- add PSD constraints

The sparse coordinate machinery is moved out to:

- [level3_scs_backend.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/level3_scs_backend.py)

### 2. Two representations

For both level 2 and level 3 there are two backends:

- `complex`: Hermitian coordinates with SCS `cs` cones
- `real-restricted`: real symmetric coordinates with SCS `s` cones

The real-restricted version is not a real-lift of the complex problem. It declares the matrices directly in the real symmetric basis and is intended for targets and maps that are real in the computational basis.

### 3. Direct SCS assembly

The repository does not use CVXPY or MOSEK wrappers here. It builds the SCS data directly:

- sparse `A`
- dense `b`
- dense `c`
- cone dictionary

and solves through the SCS Python interface.

This keeps the model explicit and makes the SCS logs easy to compare against the saved reference runs.

## Which Scripts Are Which

Level-2 examples:

- [inflation_robustness_level2.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2.py): GHZ, complex
- [inflation_robustness_level2_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_real_restricted.py): GHZ, real-restricted
- [inflation_robustness_level2_w.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w.py): W, complex
- [inflation_robustness_level2_w_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level2_w_real_restricted.py): W, real-restricted

Level-3 examples:

- [inflation_robustness_level3.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3.py): GHZ, complex
- [inflation_robustness_level3_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_real_restricted.py): GHZ, real-restricted
- [inflation_robustness_level3_w.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w.py): W, complex
- [inflation_robustness_level3_w_real_restricted.py](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper/inflation_robustness_level3_w_real_restricted.py): W, real-restricted

## Tests And Validation

There is no separate unit-test suite in this repository anymore. The current validation is example-driven:

- the example scripts are the primary runnable tests
- the `--build-only` mode is the primary compile/build smoke test
- the saved `.txt` files are the reference outputs for SCS runs

Practical checks:

```bash
python3 Baroque_test_paper/inflation_robustness_level2.py --build-only --verbose 0
python3 Baroque_test_paper/inflation_robustness_level2.py --verbose 0
python3 Baroque_test_paper/inflation_robustness_level3.py --build-only --verbose 0
```

## Reference Logs

The `.txt` files in [Baroque_test_paper](/Users/pedrolauand/SDP_Quantum_Matter/Baroque_test_paper) are intended to serve as stable reference runs for SCS:

- model size
- cone structure
- iteration trace
- final objective / `pOpt` / `tMax`

Use them to compare:

- complex vs real-restricted
- GHZ vs W
- level 2 vs level 3

They are especially useful after changing:

- the symmetry constraints
- the PPT family
- the observed anchor
- the sparse backend

## Requirements

The current code expects:

- Python 3.11
- `numpy`
- `scipy`
- `scs >= 3`

The example scripts call SCS directly and expose the main SCS tuning flags on the command line.
