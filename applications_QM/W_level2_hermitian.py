"""Level-2 noisy-W optimization in the Hermitian formulation."""

from __future__ import annotations

from w_example_common import build_draft, print_result, solve_min_noise_level

PROGRESS_VERBOSE = 1
HERMITIAN = True
LEVEL = 2


if __name__ == "__main__":
    print("[W level2 Hermitian] building SDP draft", flush=True)
    draft = build_draft(level=LEVEL, verbose=PROGRESS_VERBOSE)

    print("[W level2 Hermitian] solving optimization over p", flush=True)
    result = solve_min_noise_level(draft, Hermitian=HERMITIAN, verbose=PROGRESS_VERBOSE)

    print_result(level=LEVEL, Hermitian=HERMITIAN, result=result)
