"""Solve the paper's 5-node example and check the solution.

Usage:
    python examples/sequential_5node.py [--backend highs|gurobi|auto]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from due_lcp.networks import wakui_5node
from due_lcp.one_to_many import check_sequential_result, fw_solver, solve_sequential
from due_lcp.one_to_many.lcp_builder import build_lcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--no-bulk", action="store_true", help="skip the bulk reference"
    )
    args = parser.parse_args()

    net = wakui_5node()
    t0 = time.perf_counter()
    res = solve_sequential(net, eps=1e-8, backend=args.backend)
    t_seq = time.perf_counter() - t0
    rep = check_sequential_result(net, res, tol=1e-6)
    print(
        f"[sequential/{res.backend}] t={t_seq:.3f}s"
        f" PD={len(res.pd_steps)} ZD={len(res.zd_steps)}"
    )
    print(rep.summary())

    print("\nlink volumes (vehicles):")
    for ij, link in enumerate(net.links):
        print(f"  {link.key}: {np.sum(res.y[ij]) * net.dt:8.3f}")
    print("\npi at kappa = 15:")
    for d in net.destinations:
        print(f"  pi[{d}] = {res.pi[res.pi_idx_of[d], 14]:.4f}")

    if not args.no_bulk:
        lcp = build_lcp(net)
        t0 = time.perf_counter()
        bulk = fw_solver.solve(lcp, eps=1e-8, max_iter=500, backend=args.backend)
        t_bulk = time.perf_counter() - t0
        K, L = net.T, net.n_links
        pi_b = bulk.x[lcp.var_index["pi"]].reshape(-1, K)
        y_b = bulk.x[lcp.var_index["y"]].reshape(L, K)
        print(
            f"\n[bulk reference] t={t_bulk:.3f}s iters={bulk.n_iter}"
            f" max|pi-seq|={np.max(np.abs(pi_b - res.pi)):.2e}"
            f" max|y-seq|={np.max(np.abs(y_b - res.y)):.2e}"
        )


if __name__ == "__main__":
    main()
