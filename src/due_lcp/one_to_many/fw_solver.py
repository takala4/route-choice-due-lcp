"""Bulk Frank-Wolfe solver for the all-time [DUE-LCP] (reference solver).

This solves the LCP over all departure steps at once, which is NOT the
algorithm of Wakui, Sakai and Akamatsu (2023) (that one is time-decomposed;
see `sequential.py`). It serves as an independent reference solution on
small networks.  It satisfies the equilibrium conditions but, lacking the
paper's Step 5, does not fix the physically undetermined (w, pi) on links
and nodes without flow; use `verify.check_solution` accordingly.

Steps:
  Step 0: phase-I LP for an initial feasible x^(1)
  Step 1: at iter n, solve the linearized LP
            min  grad_phi(x^(n))^T x_hat
            s.t. M x_hat + q >= 0,  Aineq x_hat <= bineq,  x_hat >= 0
  Step 2: exact line search on the quadratic (paper eqs (24)-(25))
  Step 3: x^(n+1) = (1 - alpha_n) x^(n) + alpha_n x_hat
  Step 4: phi(x^(n+1)) < eps  ->  return
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..lp_backend import resolve_backend, solve_lp
from .lcp_builder import LCPSystem


def line_search(z0: float, z1: float, num: float, denom: float) -> float:
    """Exact minimizer of the quadratic z(alpha) on [0, 1] (paper Step 2).

    Args:
        z0: z at alpha = 0.
        z1: z at alpha = 1.
        num: dz/dalpha at alpha = 0, i.e. grad(x)^T (x_hat - x).
        denom: d^2 z / dalpha^2 = (x_hat - x)^T (M + M^T) (x_hat - x).

    Returns:
        alpha in [0, 1].  Concave case (denom <= 0): endpoint comparison,
        eq (24).  Convex case: 0 if increasing at 0, 1 if decreasing at 1,
        otherwise the stationary point -num/denom, eq (25).
    """
    if denom <= 0.0:
        return 0.0 if z0 < z1 else 1.0
    if num > 0.0:
        return 0.0
    if num + denom < 0.0:
        return 1.0
    return -num / denom


@dataclass
class FWResult:
    x: np.ndarray
    phi: float
    n_iter: int
    converged: bool
    history: list[dict] = field(default_factory=list)


def _feasibility_rows(lcp: LCPSystem) -> tuple[sp.csr_matrix, np.ndarray]:
    """Rows of  A x <= b  encoding  M x + q >= 0  and  Aineq x <= bineq."""
    A = sp.vstack([-lcp.M, lcp.Aineq], format="csr")
    b = np.concatenate([lcp.q, lcp.bineq])
    return A, b


def _solve_init_lp(lcp: LCPSystem, backend: str) -> np.ndarray:
    """Phase-I LP: feasible point of {M x + q >= 0, Aineq x <= bineq, x >= 0}.

    Slacks s with  M x + q + s >= 0;  minimize sum(s).
    """
    n = lcp.n
    m = lcp.M.shape[0]
    A = sp.vstack(
        [
            sp.hstack([-lcp.M, -sp.identity(m, format="csr")]),
            sp.hstack([lcp.Aineq, sp.csr_matrix((lcp.Aineq.shape[0], m))]),
        ],
        format="csr",
    )
    b = np.concatenate([lcp.q, lcp.bineq])
    c = np.concatenate([np.zeros(n), np.ones(m)])
    z = solve_lp(c, A, b, np.zeros(n + m), backend=backend)
    slack_total = float(z[n:].sum())
    if slack_total > 1e-6:
        warnings.warn(
            f"phase-I residual sum = {slack_total:.3e}; LCP feasibility violated",
            stacklevel=2,
        )
    return z[:n]


def solve(
    lcp: LCPSystem,
    *,
    eps: float = 1e-6,
    max_iter: int = 500,
    backend: str = "auto",
    verbose: bool = False,
) -> FWResult:
    """Frank-Wolfe on the bulk LCP.  See module docstring."""
    backend = resolve_backend(backend)
    x = _solve_init_lp(lcp, backend)
    A, b = _feasibility_rows(lcp)
    lb = np.zeros(lcp.n)
    history: list[dict] = []
    converged = False
    n_iter = 0
    sym = (lcp.M + lcp.M.T).tocsr()

    for n_iter in range(1, max_iter + 1):
        phi_n = lcp.phi(x)
        grad = sym @ x + lcp.q
        if phi_n < eps:
            converged = True
            history.append({"iter": n_iter - 1, "phi": phi_n, "alpha": None})
            break
        x_hat = solve_lp(grad, A, b, lb, backend=backend)
        d = x_hat - x
        denom = float(d @ (sym @ d))
        num = float(grad @ d)
        alpha = line_search(phi_n, lcp.phi(x_hat), num, denom)
        x = x + alpha * d
        phi_new = lcp.phi(x)
        history.append({"iter": n_iter, "phi": phi_new, "alpha": alpha})
        if verbose and (n_iter % 10 == 1 or n_iter <= 5):
            print(
                f"[FW] iter={n_iter:4d}  phi={phi_new:.6e}"
                f"  alpha={alpha:.4f}  d_norm={np.linalg.norm(d):.3e}"
            )
        if phi_new < eps:
            converged = True
            break

    return FWResult(
        x=x, phi=lcp.phi(x), n_iter=n_iter, converged=converged, history=history
    )
