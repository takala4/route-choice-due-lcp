"""Independent checker for DUE-LCP solutions.

Reference: Wakui, Sakai, Akamatsu (2023), JSCE Vol.79(4), 22-00301.

Two groups of conditions are evaluated for a solution (w, y, pi) of shape
(L, K), (L, K), (N - 1, K) on the discrete time grid of `NetworkData`.

1. Equilibrium conditions (paper Section 3, eqs (11)-(15) with the FIFO
   and free-flow bounds).  For every step kappa with previous state
   (w_prev, pi_prev), (w_0, pi_0) = (0, pi0):

       R_w  = alpha (w - w_prev) - y + alpha (pi_tail - pi_tail,prev) + mu
       R_y  = c0 + w + pi_tail - pi_head
       R_pi = inflow(y) - outflow(y) - q

       w, y, pi >= 0,   R_w >= 0,   R_y >= 0,   R_pi = 0,
       w . R_w = 0,     y . R_y = 0,
       pi >= max[pi_prev - ds, pi0].

2. Physical consistency (paper Appendix III).  The equilibrium conditions
   alone leave w and pi undetermined on links and nodes that carry no
   flow.  PD Step 5 and the ZD algorithm pin them down so that the state
   handed to the next step is physically meaningful:

   * pi_n equals the origin-to-n shortest travel time under link costs
     c0 + w, for every node n (not only nodes on used paths);
   * on steps without demand, y = 0 and no queue grows (w <= w_prev).

This module shares no code with `sequential.py`: the residuals and the
shortest-path search are re-implemented here so that the check is
independent of the solver.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..network import NetworkData


@dataclass
class EquilibriumReport:
    """Worst-case violation of each equilibrium condition over all steps."""

    x_negative: float = 0.0  # max(0, -x) over w, y, pi
    r_w_negative: float = 0.0  # max(0, -R_w)
    r_y_negative: float = 0.0  # max(0, -R_y)
    conservation: float = 0.0  # max |R_pi|
    w_complementarity: float = 0.0  # max |w * R_w|
    y_complementarity: float = 0.0  # max |y * R_y|
    fifo: float = 0.0  # max(0, pi_prev - ds - pi)
    free_flow: float = 0.0  # max(0, pi0 - pi)

    def max_violation(self) -> float:
        return max(vars(self).values())


@dataclass
class PhysicalReport:
    """Worst-case violation of each physical-consistency condition."""

    pi_shortest_path: float = 0.0  # max |pi_n - dist_n(c0 + w)| over n, kappa
    zd_flow: float = 0.0  # max |y| on steps with q = 0
    zd_queue_growth: float = 0.0  # max(0, w - w_prev) on steps with q = 0

    def max_violation(self) -> float:
        return max(vars(self).values())


@dataclass
class CheckReport:
    """Combined result of `check_solution`."""

    equilibrium: EquilibriumReport
    physical: PhysicalReport
    tol: float
    per_step_equilibrium: np.ndarray  # max violation per step, length K
    per_step_physical: np.ndarray  # max violation per step, length K
    zd_steps: list[int] = field(default_factory=list)  # 1-based

    @property
    def equilibrium_ok(self) -> bool:
        return self.equilibrium.max_violation() <= self.tol

    @property
    def physical_ok(self) -> bool:
        return self.physical.max_violation() <= self.tol

    @property
    def passed(self) -> bool:
        return self.equilibrium_ok and self.physical_ok

    def first_failing_step(self) -> int | None:
        """1-based first step violating any condition, or None."""
        bad = np.maximum(self.per_step_equilibrium, self.per_step_physical) > self.tol
        return int(np.argmax(bad)) + 1 if bad.any() else None

    def summary(self) -> str:
        lines = [f"[equilibrium] {'OK' if self.equilibrium_ok else 'FAIL'}"]
        for name, val in vars(self.equilibrium).items():
            lines.append(f"  {name:18s} {val:.3e}")
        lines.append(f"[physical]    {'OK' if self.physical_ok else 'FAIL'}")
        for name, val in vars(self.physical).items():
            lines.append(f"  {name:18s} {val:.3e}")
        step = self.first_failing_step()
        if step is not None:
            lines.append(f"first failing step: {step}")
        lines.append(f"tol = {self.tol:.1e}  ->  {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _shortest_path(
    net: NetworkData, link_cost: np.ndarray, origin: int
) -> dict[int, float]:
    """Dijkstra from `origin` with the given per-link costs."""
    inf = float("inf")
    dist = {n: inf for n in net.nodes}
    dist[origin] = 0.0
    adj: dict[int, list[tuple[int, float]]] = {n: [] for n in net.nodes}
    for ij, link in enumerate(net.links):
        adj[link.tail].append((link.head, float(link_cost[ij])))
    heap: list[tuple[float, int]] = [(0.0, origin)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, c in adj[u]:
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def check_solution(
    net: NetworkData,
    w: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    pi_nodes: list[int],
    *,
    tol: float = 1e-8,
) -> CheckReport:
    """Check a solution against the equilibrium and physical conditions.

    Args:
        net: Network and demand.
        w: Queue delays, shape (L, K).
        y: Link inflow rates, shape (L, K).
        pi: Origin-to-node shortest travel times for non-origin nodes,
            shape (N - 1, K), rows ordered as `pi_nodes`.
        pi_nodes: Node ids corresponding to the rows of `pi`.
        tol: Tolerance used for the pass / fail verdict.

    Returns:
        CheckReport with worst-case violations, per-step diagnostics and
        the verdict.
    """
    L = net.n_links
    K = net.T
    dt = net.dt
    origin = net.origin
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    pi = np.asarray(pi, dtype=float)
    if w.shape != (L, K) or y.shape != (L, K) or pi.shape != (len(pi_nodes), K):
        raise ValueError("w, y, pi must have shapes (L, K), (L, K), (N - 1, K)")
    if origin in pi_nodes or set(pi_nodes) != set(net.nodes) - {origin}:
        raise ValueError("pi_nodes must be exactly the non-origin nodes")

    idx_of = {n: i for i, n in enumerate(pi_nodes)}
    alphas = np.array([link.mu_hat / dt for link in net.links])
    mu = np.array([link.mu_hat for link in net.links])
    c0 = np.array([link.tau_hat for link in net.links])
    tail = np.array(
        [idx_of[ln.tail] if ln.tail != origin else -1 for ln in net.links], dtype=int
    )
    head = np.array(
        [idx_of[ln.head] if ln.head != origin else -1 for ln in net.links], dtype=int
    )

    pi0_dist = _shortest_path(net, c0, origin)
    pi0 = np.array([pi0_dist[n] for n in pi_nodes])
    if not np.all(np.isfinite(pi0)):
        raise ValueError("some node is unreachable from the origin")

    q_node = np.zeros((len(pi_nodes), K))
    for d in net.destinations:
        q_node[idx_of[d]] += net.demand[d]
    zd_steps = [k + 1 for k in range(K) if float(q_node[:, k].sum()) == 0.0]

    def pi_at(pi_vec: np.ndarray, node_idx: np.ndarray) -> np.ndarray:
        out = np.zeros(L)
        mask = node_idx >= 0
        out[mask] = pi_vec[node_idx[mask]]
        return out

    eq = EquilibriumReport()
    ph = PhysicalReport()
    per_eq = np.zeros(K)
    per_ph = np.zeros(K)

    w_prev = np.zeros(L)
    pi_prev = pi0.copy()
    for k in range(K):
        w_k, y_k, pi_k = w[:, k], y[:, k], pi[:, k]

        # ---- equilibrium conditions ----
        r_w = (
            alphas * (w_k - w_prev)
            - y_k
            + alphas * (pi_at(pi_k, tail) - pi_at(pi_prev, tail))
            + mu
        )
        r_y = c0 + w_k + pi_at(pi_k, tail) - pi_at(pi_k, head)
        r_pi = -q_node[:, k].copy()
        for ij in range(L):
            if head[ij] >= 0:
                r_pi[head[ij]] += y_k[ij]
            if tail[ij] >= 0:
                r_pi[tail[ij]] -= y_k[ij]

        step_eq = {
            "x_negative": float(
                np.max(np.maximum(-np.concatenate([w_k, y_k, pi_k]), 0.0))
            ),
            "r_w_negative": float(np.max(np.maximum(-r_w, 0.0))),
            "r_y_negative": float(np.max(np.maximum(-r_y, 0.0))),
            "conservation": float(np.max(np.abs(r_pi))),
            "w_complementarity": float(np.max(np.abs(w_k * r_w))),
            "y_complementarity": float(np.max(np.abs(y_k * r_y))),
            "fifo": float(np.max(np.maximum(pi_prev - dt - pi_k, 0.0))),
            "free_flow": float(np.max(np.maximum(pi0 - pi_k, 0.0))),
        }
        for name, val in step_eq.items():
            setattr(eq, name, max(getattr(eq, name), val))
        per_eq[k] = max(step_eq.values())

        # ---- physical consistency ----
        dist = _shortest_path(net, c0 + w_k, origin)
        dist_vec = np.array([dist[n] for n in pi_nodes])
        step_ph = {"pi_shortest_path": float(np.max(np.abs(pi_k - dist_vec)))}
        if (k + 1) in zd_steps:
            step_ph["zd_flow"] = float(np.max(np.abs(y_k)))
            step_ph["zd_queue_growth"] = float(np.max(np.maximum(w_k - w_prev, 0.0)))
        for name, val in step_ph.items():
            setattr(ph, name, max(getattr(ph, name), val))
        per_ph[k] = max(step_ph.values())

        w_prev, pi_prev = w_k, pi_k

    return CheckReport(
        equilibrium=eq,
        physical=ph,
        tol=tol,
        per_step_equilibrium=per_eq,
        per_step_physical=per_ph,
        zd_steps=zd_steps,
    )


def excess_travel_time(
    net: NetworkData,
    w: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    pi_nodes: list[int],
) -> float:
    """Mean excess travel time per vehicle over the shortest path (DUE gap).

    Summing y_ij R_y_ij along any path telescopes to (path travel time -
    pi_destination), so sum_k sum_ij y_ij[k] ds R_y_ij[k] is the total
    travel time in excess of the shortest-path time, summed over all
    users.  Dividing by the number of users gives a scalar DUE gap that is
    comparable across solvers (0 at an exact DUE).
    """
    L = net.n_links
    origin = net.origin
    idx_of = {n: i for i, n in enumerate(pi_nodes)}
    c0 = np.array([link.tau_hat for link in net.links])
    tail = np.array(
        [idx_of[ln.tail] if ln.tail != origin else -1 for ln in net.links], dtype=int
    )
    head = np.array(
        [idx_of[ln.head] if ln.head != origin else -1 for ln in net.links], dtype=int
    )

    def pi_at(pi_vec: np.ndarray, node_idx: np.ndarray) -> np.ndarray:
        out = np.zeros(L)
        mask = node_idx >= 0
        out[mask] = pi_vec[node_idx[mask]]
        return out

    total = 0.0
    for k in range(net.T):
        r_y = c0 + w[:, k] + pi_at(pi[:, k], tail) - pi_at(pi[:, k], head)
        total += float(np.sum(y[:, k] * net.dt * r_y))
    n_users = net.total_demand()
    return total / n_users if n_users > 0 else 0.0


def check_sequential_result(net: NetworkData, res, *, tol: float = 1e-8) -> CheckReport:
    """Convenience wrapper for a `sequential.SequentialResult`."""
    return check_solution(net, res.w, res.y, res.pi, res.pi_nodes, tol=tol)


def check_bulk_result(net: NetworkData, lcp, x: np.ndarray, *, tol: float = 1e-8):
    """Convenience wrapper for a bulk solution vector x of `lcp_builder.LCPSystem`."""
    L = net.n_links
    K = net.T
    pi_nodes = lcp.var_meta["pi_nodes"]
    w = x[lcp.var_index["w"]].reshape(L, K)
    y = x[lcp.var_index["y"]].reshape(L, K)
    pi = x[lcp.var_index["pi"]].reshape(len(pi_nodes), K)
    return check_solution(net, w, y, pi, pi_nodes, tol=tol)
