"""Level-2 noisy-W optimization in the real formulation."""

from __future__ import annotations

from w_example_common import build_draft, print_result, solve_min_noise_level

PROGRESS_VERBOSE = 1
HERMITIAN = False
LEVEL = 2


if __name__ == "__main__":
    print("[W level2 real] building SDP draft", flush=True)
    draft = build_draft(level=LEVEL, verbose=PROGRESS_VERBOSE)

    print("[W level2 real] solving optimization over p", flush=True)
    result = solve_min_noise_level(draft, Hermitian=HERMITIAN, verbose=PROGRESS_VERBOSE)

    print_result(level=LEVEL, Hermitian=HERMITIAN, result=result)
