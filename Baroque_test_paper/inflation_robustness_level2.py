"""Readable example script for the full-coordinate level-2 GHZ robustness SDP."""

from __future__ import annotations

import argparse

try:
    from level2_complex_model import (
        DEFAULT_SCS_ACCELERATION_INTERVAL,
        DEFAULT_SCS_ACCELERATION_LOOKBACK,
        DEFAULT_SCS_ALPHA,
        DEFAULT_SCS_RHO_X,
        DEFAULT_SCS_SCALE,
        build_inflation_robustness_level2_scs_model,
        ghz_density_matrix,
        solve_inflation_robustness_level2_scs,
    )
except ModuleNotFoundError:
    from Baroque_test_paper.level2_complex_model import (
        DEFAULT_SCS_ACCELERATION_INTERVAL,
        DEFAULT_SCS_ACCELERATION_LOOKBACK,
        DEFAULT_SCS_ALPHA,
        DEFAULT_SCS_RHO_X,
        DEFAULT_SCS_SCALE,
        build_inflation_robustness_level2_scs_model,
        ghz_density_matrix,
        solve_inflation_robustness_level2_scs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or solve the full-coordinate SCS level-2 inflation robustness SDP for GHZ.",
    )
    parser.add_argument("--build-only", action="store_true", help="Build the model and stop before solve().")
    parser.add_argument("--verbose", type=int, default=1, help="SCS verbosity level.")
    parser.add_argument("--optimizer-max-time", type=float, default=None, help="Optional SCS time limit in seconds.")
    parser.add_argument("--scs-max-iters", type=int, default=int(1e5), help="Maximum SCS iterations.")
    parser.add_argument("--scs-eps-abs", type=float, default=1e-4, help="Absolute SCS tolerance.")
    parser.add_argument("--scs-eps-rel", type=float, default=1e-4, help="Relative SCS tolerance.")
    parser.add_argument("--scs-alpha", type=float, default=DEFAULT_SCS_ALPHA, help="Douglas-Rachford relaxation parameter.")
    parser.add_argument("--scs-scale", type=float, default=DEFAULT_SCS_SCALE, help="Initial SCS dual scaling.")
    parser.add_argument("--scs-rho-x", type=float, default=DEFAULT_SCS_RHO_X, help="Primal scale parameter rho_x.")
    parser.add_argument(
        "--scs-acceleration-lookback",
        type=int,
        default=DEFAULT_SCS_ACCELERATION_LOOKBACK,
        help="Anderson acceleration lookback.",
    )
    parser.add_argument(
        "--scs-acceleration-interval",
        type=int,
        default=DEFAULT_SCS_ACCELERATION_INTERVAL,
        help="Anderson acceleration interval.",
    )
    parser.add_argument("--scs-no-normalize", action="store_true", help="Disable SCS data normalization.")
    parser.add_argument("--scs-no-adaptive-scale", action="store_true", help="Disable SCS adaptive scaling.")
    parser.add_argument("--scs-use-indirect", action="store_true", help="Use SCS indirect linear solver.")
    args = parser.parse_args()

    rho = ghz_density_matrix()
    dims3 = (2, 2, 2)

    built = build_inflation_robustness_level2_scs_model(
        rho,
        dims3,
        verbose=args.verbose,
        optimizer_max_time=args.optimizer_max_time,
    )
    print(
        "Model built: "
        f"backend=scs, full_dim={built.full_dim}, "
        f"ppt_auxiliaries={len(built.ppt_auxiliaries)}, "
        f"rows_total={built.constraint_counts['rows_total']}, "
        f"x_dim={built.c.size}, "
        f"cs_blocks={len(built.cone['cs'])}",
        flush=True,
    )
    if args.build_only:
        raise SystemExit(0)

    solved = solve_inflation_robustness_level2_scs(
        rho,
        dims3,
        verbose=args.verbose,
        optimizer_max_time=args.optimizer_max_time,
        scs_max_iters=args.scs_max_iters,
        scs_eps_abs=args.scs_eps_abs,
        scs_eps_rel=args.scs_eps_rel,
        scs_use_indirect=args.scs_use_indirect,
        scs_alpha=args.scs_alpha,
        scs_scale=args.scs_scale,
        scs_normalize=not args.scs_no_normalize,
        scs_adaptive_scale=not args.scs_no_adaptive_scale,
        scs_rho_x=args.scs_rho_x,
        scs_acceleration_lookback=args.scs_acceleration_lookback,
        scs_acceleration_interval=args.scs_acceleration_interval,
    )
    print(
        f"problem_status={solved['problem_status']}, solution_status={solved['solution_status']}",
        flush=True,
    )
    p_value = float(solved["p_value"])
    t_value = float(solved["t_value"])
    print(f"pOpt = {p_value:.8f}", flush=True)
    print(f"tOpt = {t_value:.8f}", flush=True)
    if p_value < 1.0:
        print("The state is non-network.", flush=True)
