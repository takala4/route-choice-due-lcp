"""Linear-programming backends.

Two backends are supported:

* ``"highs"``: HiGHS through ``scipy.optimize.linprog``.  Always available,
  used by default.
* ``"gurobi"``: Gurobi through ``gurobipy`` (optional dependency, needs a
  licence).  Keeps persistent models across departure steps and is
  noticeably faster on large networks.

`solve_lp` is a one-off LP solve used by the bulk reference solver.
`StepLPs` wraps the two LPs of the paper's PD algorithm ([DUE-FW-init]
and [DUE-FW-LP]) whose coefficient matrices are identical for every
departure step; only right-hand sides, bounds and objectives change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

BACKENDS = ("highs", "gurobi")


def gurobi_available() -> bool:
    try:
        import gurobipy as gp

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
        return True
    except Exception:  # noqa: BLE001 - any import / licence failure
        return False


def resolve_backend(name: str = "auto") -> str:
    """Map ``"auto"`` to gurobi if usable, else highs; validate explicit names."""
    if name == "auto":
        return "gurobi" if gurobi_available() else "highs"
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    return name


# ---------------------------------------------------------------------------
# one-off LP
# ---------------------------------------------------------------------------
def solve_lp(
    c: np.ndarray,
    A_ub: sp.spmatrix,
    b_ub: np.ndarray,
    lb: np.ndarray,
    *,
    backend: str = "highs",
) -> np.ndarray:
    """Solve  min c^T x  s.t.  A_ub x <= b_ub,  x >= lb."""
    backend = resolve_backend(backend)
    if backend == "highs":
        bounds = np.column_stack([lb, np.full(len(lb), np.inf)])
        res = linprog(
            c, A_ub=sp.csr_matrix(A_ub), b_ub=b_ub, bounds=bounds, method="highs"
        )
        if res.status != 0:
            raise RuntimeError(f"HiGHS LP failed: {res.message}")
        return np.asarray(res.x)
    import gurobipy as gp
    from gurobipy import GRB

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(len(c), lb=lb, name="x")
            if A_ub.shape[0] > 0:
                model.addMConstr(
                    sp.csr_matrix(A_ub), x, GRB.LESS_EQUAL, b_ub, name="ub"
                )
            model.setObjective(c @ x, GRB.MINIMIZE)
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi LP status={model.Status}")
            return np.asarray(x.X)


# ---------------------------------------------------------------------------
# per-step LPs of the PD algorithm
# ---------------------------------------------------------------------------
@dataclass
class StepLPData:
    """Step-invariant data of [DUE-QP(kappa)].

    Variables are stacked as x = (w[L], y[L], pi[n_pi]).
    ``M`` is the LCP coefficient matrix, ``alphas = mu / ds``.
    """

    M: sp.csr_matrix
    alphas: np.ndarray
    pi0: np.ndarray

    @property
    def L(self) -> int:
        return len(self.alphas)

    @property
    def n_pi(self) -> int:
        return len(self.pi0)

    @property
    def n_var(self) -> int:
        return 2 * self.L + self.n_pi


class StepLPs:
    """Interface: the two LPs of the PD algorithm for one departure step."""

    def __init__(self, data: StepLPData):
        self.data = data

    def solve_init(
        self,
        b: np.ndarray,
        w_prev: np.ndarray,
        pi_prev: np.ndarray,
        pi_lb: np.ndarray,
    ) -> np.ndarray:
        """[DUE-FW-init]: feasible x closest (L1) to (w_prev, pi_prev).

        min  1^T t_w + 1^T t_pi
        s.t. M x + b >= 0,  |w - w_prev| <= t_w,  |pi - pi_prev| <= t_pi,
             w, y >= 0,  pi >= pi_lb = max[pi_prev - ds, pi0].
        """
        raise NotImplementedError

    def solve_fw(
        self, grad: np.ndarray, b: np.ndarray, w_prev: np.ndarray
    ) -> np.ndarray:
        """[DUE-FW-LP]: min grad^T x s.t. (19), (20), w, y >= 0, pi >= pi0.

        (19)  M x + b >= 0
        (20)  -w + alpha^{-1} y + w_prev >= 0
        """
        raise NotImplementedError

    def close(self) -> None:
        return None


class HighsStepLPs(StepLPs):
    """scipy / HiGHS implementation; constant matrices built once."""

    def __init__(self, data: StepLPData):
        super().__init__(data)
        L, n_pi, n = data.L, data.n_pi, data.n_var
        I_w = sp.csr_matrix((np.ones(L), (np.arange(L), np.arange(L))), shape=(L, n))
        I_y = sp.csr_matrix(
            (np.ones(L), (np.arange(L), L + np.arange(L))), shape=(L, n)
        )
        I_pi = sp.csr_matrix(
            (np.ones(n_pi), (np.arange(n_pi), 2 * L + np.arange(n_pi))), shape=(n_pi, n)
        )
        eye_L = sp.identity(L, format="csr")
        eye_p = sp.identity(n_pi, format="csr")
        zero_Lp = sp.csr_matrix((L, n_pi))
        zero_pL = sp.csr_matrix((n_pi, L))
        zero_nL = sp.csr_matrix((n, L))
        zero_np = sp.csr_matrix((n, n_pi))
        # init LP over z = (x, t_w, t_pi)
        self.A_init = sp.vstack(
            [
                sp.hstack([-data.M, zero_nL, zero_np]),
                sp.hstack([I_w, -eye_L, zero_Lp]),
                sp.hstack([-I_w, -eye_L, zero_Lp]),
                sp.hstack([I_pi, zero_pL, -eye_p]),
                sp.hstack([-I_pi, zero_pL, -eye_p]),
            ],
            format="csr",
        )
        self.c_init = np.concatenate([np.zeros(n), np.ones(L), np.ones(n_pi)])
        # FW LP over x
        inv_alpha = sp.diags(1.0 / data.alphas)
        self.A_fw = sp.vstack([-data.M, I_w - inv_alpha @ I_y], format="csr")
        self.lb_fw = np.concatenate([np.zeros(2 * L), data.pi0])

    def solve_init(self, b, w_prev, pi_prev, pi_lb):
        L, n_pi, n = self.data.L, self.data.n_pi, self.data.n_var
        b_ub = np.concatenate([b, w_prev, -w_prev, pi_prev, -pi_prev])
        lb = np.concatenate([np.zeros(2 * L), pi_lb, np.zeros(L + n_pi)])
        z = solve_lp(self.c_init, self.A_init, b_ub, lb, backend="highs")
        return z[:n]

    def solve_fw(self, grad, b, w_prev):
        b_ub = np.concatenate([b, w_prev])
        return solve_lp(grad, self.A_fw, b_ub, self.lb_fw, backend="highs")


class GurobiStepLPs(StepLPs):
    """Gurobi implementation; persistent models updated via RHS / bounds / objective."""

    def __init__(self, data: StepLPData):
        super().__init__(data)
        import gurobipy as gp
        from gurobipy import GRB

        self._GRB = GRB
        L, n_pi, n = data.L, data.n_pi, data.n_var
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

        m = gp.Model(env=self.env)
        x = m.addMVar(n, lb=0.0, name="x")
        t_w = m.addMVar(L, lb=0.0, name="t_w")
        t_pi = m.addMVar(n_pi, lb=0.0, name="t_pi")
        w = x[:L]
        pi = x[2 * L :]
        self.init = {
            "model": m,
            "x": x,
            "pi": pi,
            "lcp": m.addMConstr(data.M, x, GRB.GREATER_EQUAL, np.zeros(n), name="lcp"),
            "w_up": m.addConstr(w - t_w <= np.zeros(L), name="w_up"),
            "w_lo": m.addConstr(w + t_w >= np.zeros(L), name="w_lo"),
            "pi_up": m.addConstr(pi - t_pi <= np.zeros(n_pi), name="pi_up"),
            "pi_lo": m.addConstr(pi + t_pi >= np.zeros(n_pi), name="pi_lo"),
        }
        m.setObjective(t_w.sum() + t_pi.sum(), GRB.MINIMIZE)
        m.update()

        m2 = gp.Model(env=self.env)
        x2 = m2.addMVar(n, lb=0.0, name="x")
        x2[2 * L :].LB = data.pi0
        self.fw = {
            "model": m2,
            "x": x2,
            "lcp": m2.addMConstr(
                data.M, x2, GRB.GREATER_EQUAL, np.zeros(n), name="lcp"
            ),
            "c20": m2.addConstr(
                -x2[:L] + (1.0 / data.alphas) * x2[L : 2 * L] >= np.zeros(L), name="c20"
            ),
        }
        m2.setObjective(0.0, GRB.MINIMIZE)
        m2.update()

    def solve_init(self, b, w_prev, pi_prev, pi_lb):
        m = self.init
        m["lcp"].RHS = -b
        m["w_up"].RHS = w_prev
        m["w_lo"].RHS = w_prev
        m["pi_up"].RHS = pi_prev
        m["pi_lo"].RHS = pi_prev
        m["pi"].LB = pi_lb
        m["model"].optimize()
        if m["model"].Status != self._GRB.OPTIMAL:
            raise RuntimeError(f"[DUE-FW-init] status={m['model'].Status}")
        return np.asarray(m["x"].X)

    def solve_fw(self, grad, b, w_prev):
        m = self.fw
        m["lcp"].RHS = -b
        m["c20"].RHS = -w_prev
        m["model"].setObjective(grad @ m["x"], self._GRB.MINIMIZE)
        m["model"].optimize()
        if m["model"].Status != self._GRB.OPTIMAL:
            raise RuntimeError(f"[DUE-FW-LP] status={m['model'].Status}")
        return np.asarray(m["x"].X)

    def close(self) -> None:
        self.init["model"].dispose()
        self.fw["model"].dispose()
        self.env.dispose()


def make_step_lps(data: StepLPData, backend: str = "auto") -> StepLPs:
    backend = resolve_backend(backend)
    if backend == "gurobi":
        return GurobiStepLPs(data)
    return HighsStepLPs(data)
