"""Time-decomposed DUE solver of Wakui, Sakai and Akamatsu (2023), Section 4.

The algorithm marches forward in departure time. Given the equilibrium
state (w_{k-1}, pi_{k-1}) at step k-1, step k is solved by one of two
modules chosen by the presence of demand:

    state := (w_0, pi_0) = (0, free-flow shortest path)
    for kappa = 1, 2, ..., K:
        if q_kappa != 0:   PD algorithm  (Section 4 (2), Steps 0-5)
        else:              ZD algorithm  (Section 4 (3))
        record (w_kappa, y_kappa, pi_kappa)

PD algorithm (Positive Demand), Frank-Wolfe on [DUE-QP(kappa)]:
  Step 0: initial feasible point from [DUE-FW-init], the LP that keeps
          (w, pi) as close as possible (L1) to the previous step.
  Step 1: linearized LP [DUE-FW-LP] with constraints (19), (20),
          w, y >= 0, pi >= pi0.  Constraint (20) replaces the FIFO
          bound (21); see Appendix II of the paper.
  Step 2: exact line search on the quadratic z(alpha), eqs (24)-(25).
  Step 3: convex combination.
  Step 4: stop when z(x) < eps  (the QP optimum is known to be 0).
  Step 5: reconciliation (Appendix III): Dijkstra from the origin that
          fixes pi to the true shortest travel time under (c0 + w) and
          rewrites w on the out-links of every settled node so that the
          queue-dynamics complementarity (III.2) is preserved.

ZD algorithm (Zero Demand), y = 0: a single Dijkstra pass that decays
each queue by (ds - d_v), where d_v is the decrease of pi at the link's
tail between steps kappa-1 and kappa (paper Section 4 (3)).

Time convention: step kappa (0-based index k) refers to users departing
the origin at clock time s_kappa = kappa * ds = (k + 1) * dt, i.e. the end
of demand bin [k dt, (k + 1) dt).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..lp_backend import StepLPData, make_step_lps
from ..network import NetworkData
from .fw_solver import line_search
from .lcp_builder import free_flow_shortest_path


@dataclass
class SequentialResult:
    """Full DUE solution from the sequential solver.

    Attributes:
        w: Queue delay by link and departure step, shape (L, K).
        y: Link inflow rate by link and departure step, shape (L, K).
        pi: Origin-to-node travel time for non-origin nodes, shape (N - 1, K).
        pi0: Free-flow shortest travel times, length N - 1.
        pi_idx_of: Map node id -> row of ``pi``.
        pi_nodes: Node ids of the rows of ``pi``.
        iters_per_step: Frank-Wolfe iterations per step (0 on ZD steps).
        phi_per_step: Final z(x) per step.
        pd_steps, zd_steps: 1-based step numbers handled by each algorithm.
        backend: LP backend that was used.
    """

    w: np.ndarray
    y: np.ndarray
    pi: np.ndarray
    pi0: np.ndarray
    pi_idx_of: dict[int, int]
    pi_nodes: list[int]
    iters_per_step: list[int] = field(default_factory=list)
    phi_per_step: list[float] = field(default_factory=list)
    pd_steps: list[int] = field(default_factory=list)
    zd_steps: list[int] = field(default_factory=list)
    backend: str = ""


# ---------------------------------------------------------------------------
# Dijkstra kernel shared by PD Step 5 and the ZD algorithm
# ---------------------------------------------------------------------------
def _dijkstra_rewrite(
    net: NetworkData,
    w_ref: np.ndarray,
    pi_ref: np.ndarray,
    base_shift: float,
    pi_idx_of: dict[int, int],
    outlinks: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Shortest-path search that rewrites link queues as nodes are settled.

    When node v is settled with shortest travel time pi_v, every out-link
    (v, k) gets

        w_vk := max[ w_ref_vk - (base_shift + (pi_v - pi_ref_v)), 0 ]

    and is then used with cost c0_vk + w_vk to relax k.  With
    ``base_shift = 0`` and ``(w_ref, pi_ref) = (w', pi')`` from Frank-Wolfe
    this is PD Step 5 (shift = d_v = pi_v - pi'_v).  With
    ``base_shift = ds`` and ``(w_ref, pi_ref) = (w_{k-1}, pi_{k-1})`` it is
    the ZD algorithm (shift = ds - d_v, d_v = pi_{k-1,v} - pi_v).

    pi_ref of the origin is taken as 0.  Returns (w_new, pi_new) with pi
    indexed like pi_ref (non-origin nodes only).
    """
    origin = net.origin
    inf = float("inf")
    dist = {n: inf for n in net.nodes}
    dist[origin] = 0.0
    settled: set[int] = set()
    w_new = w_ref.copy()
    heap: list[tuple[float, int]] = [(0.0, origin)]
    while heap:
        d, v = heapq.heappop(heap)
        if v in settled:
            continue
        settled.add(v)
        ref = 0.0 if v == origin else float(pi_ref[pi_idx_of[v]])
        shift = base_shift + (d - ref)
        for ij in outlinks[v]:
            w_new[ij] = max(w_ref[ij] - shift, 0.0)
            link = net.links[ij]
            nd = d + link.tau_hat + w_new[ij]
            if nd < dist[link.head]:
                dist[link.head] = nd
                heapq.heappush(heap, (nd, link.head))
    if len(settled) != len(net.nodes):
        missing = sorted(set(net.nodes) - settled)
        raise ValueError(f"nodes unreachable from origin: {missing}")
    pi_new = np.empty(len(pi_idx_of))
    for n_node, n_local in pi_idx_of.items():
        pi_new[n_local] = dist[n_node]
    return w_new, pi_new


# ---------------------------------------------------------------------------
# PD algorithm
# ---------------------------------------------------------------------------
def build_step_data(
    net: NetworkData, pi_idx_of: dict[int, int], pi0: np.ndarray
) -> tuple[StepLPData, np.ndarray, np.ndarray]:
    """Step-invariant coefficient matrix M and constant part of b.

    Returns (data, b_const, tail_pi_idx) where ``tail_pi_idx[ij]`` is the pi
    index of link ij's tail, or -1 if the tail is the origin.
    """
    L = net.n_links
    n_pi = len(pi0)
    origin = net.origin
    w_off, y_off, pi_off = 0, L, 2 * L
    n_var = 2 * L + n_pi
    alphas = np.array([link.mu_hat / net.dt for link in net.links])
    tail_pi_idx = np.array(
        [pi_idx_of[ln.tail] if ln.tail != origin else -1 for ln in net.links],
        dtype=int,
    )
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def put(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # Block W: alpha w - y + alpha A_+^T pi
    for ij in range(L):
        put(w_off + ij, w_off + ij, alphas[ij])
        put(w_off + ij, y_off + ij, -1.0)
        if tail_pi_idx[ij] >= 0:
            put(w_off + ij, pi_off + tail_pi_idx[ij], alphas[ij])
    # Block Y: w + A^T pi
    for ij, link in enumerate(net.links):
        put(y_off + ij, w_off + ij, 1.0)
        if tail_pi_idx[ij] >= 0:
            put(y_off + ij, pi_off + tail_pi_idx[ij], 1.0)
        if link.head != origin:
            put(y_off + ij, pi_off + pi_idx_of[link.head], -1.0)
    # Block Pi: -A y (conservation)
    for ij, link in enumerate(net.links):
        if link.head != origin:
            put(pi_off + pi_idx_of[link.head], y_off + ij, 1.0)
        if link.tail != origin:
            put(pi_off + pi_idx_of[link.tail], y_off + ij, -1.0)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n_var, n_var))

    b_const = np.zeros(n_var)
    for ij, link in enumerate(net.links):
        b_const[w_off + ij] = link.mu_hat
        b_const[y_off + ij] = link.tau_hat
    return (
        StepLPData(M=M, alphas=alphas, pi0=np.asarray(pi0, float)),
        b_const,
        tail_pi_idx,
    )


def b_for_step(
    data: StepLPData,
    b_const: np.ndarray,
    tail_pi_idx: np.ndarray,
    w_prev: np.ndarray,
    pi_prev: np.ndarray,
    q_step: np.ndarray,
) -> np.ndarray:
    """b_kappa = (-alpha beta, c0, -q) with beta = w_prev + A_+^T pi_prev - ds."""
    L = data.L
    b = b_const.copy()
    b[:L] -= data.alphas * w_prev
    nonorigin = tail_pi_idx >= 0
    b[:L][nonorigin] -= data.alphas[nonorigin] * pi_prev[tail_pi_idx[nonorigin]]
    b[2 * L :] -= q_step
    return b


def _frank_wolfe(
    lps,
    data: StepLPData,
    sym_M: sp.csr_matrix,
    b: np.ndarray,
    w_prev: np.ndarray,
    pi_prev: np.ndarray,
    pi_lb: np.ndarray,
    eps: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float, bool]:
    """PD Steps 0-4 for one departure step. Returns (x', n_iter, z(x'), converged)."""

    def z(v: np.ndarray) -> float:
        return float(v @ (data.M @ v + b))

    x = lps.solve_init(b, w_prev, pi_prev, pi_lb)  # Step 0
    n_iter = 0
    z_x = z(x)
    if z_x < eps:
        return x, 0, z_x, True
    converged = False
    while n_iter < max_iter:
        n_iter += 1
        grad = sym_M @ x + b  # eq (22)
        x_hat = lps.solve_fw(grad, b, w_prev)  # Step 1
        d = x_hat - x
        num = float(grad @ d)
        denom = float(d @ (sym_M @ d))
        alpha = line_search(z_x, z(x_hat), num, denom)  # Step 2
        x = x + alpha * d  # Step 3
        z_x = z(x)
        if z_x < eps:  # Step 4
            converged = True
            break
    return x, n_iter, z_x, converged


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def solve_sequential(
    net: NetworkData,
    *,
    eps: float = 1e-6,
    max_iter_per_step: int = 50,
    zd_method: str = "dijkstra",
    backend: str = "auto",
    verbose: bool = False,
) -> SequentialResult:
    """Solve the one-to-many DUE step by step with the PD / ZD algorithms.

    Args:
        net: Network and demand definition.
        eps: Step 4 convergence threshold on z(x) for each PD step.
        max_iter_per_step: Frank-Wolfe iteration cap per PD step.
        zd_method: ``"dijkstra"`` runs the paper's ZD algorithm on steps
            with q_kappa = 0 (default).  ``"pd"`` runs the PD algorithm on
            those steps as well; the paper notes this is valid but slower.
        backend: LP backend, ``"auto"`` (gurobi if available, else highs),
            ``"highs"`` or ``"gurobi"``.
        verbose: Print progress every 10 steps.
    """
    if zd_method not in ("dijkstra", "pd"):
        raise ValueError(f"zd_method must be 'dijkstra' or 'pd', got {zd_method!r}")

    L = net.n_links
    K = net.T
    dt = net.dt
    pi_nodes = net.pi_nodes
    pi_idx_of = {n: i for i, n in enumerate(pi_nodes)}
    pi0_dist = free_flow_shortest_path(net)
    pi0 = np.array([pi0_dist[n] for n in pi_nodes])
    if not np.all(np.isfinite(pi0)):
        raise ValueError("some node is unreachable from the origin")

    outlinks: dict[int, list[int]] = {n: [] for n in net.nodes}
    for ij, link in enumerate(net.links):
        outlinks[link.tail].append(ij)

    n_pi = len(pi_nodes)
    w = np.zeros((L, K))
    y = np.zeros((L, K))
    pi = np.zeros((n_pi, K))
    q_node = np.zeros((n_pi, K))
    for d in net.destinations:
        q_node[pi_idx_of[d]] += net.demand[d]

    iters: list[int] = []
    phis: list[float] = []
    pd_steps: list[int] = []
    zd_steps: list[int] = []

    data, b_const, tail_pi_idx = build_step_data(net, pi_idx_of, pi0)
    sym_M = (data.M + data.M.T).tocsr()
    lps = make_step_lps(data, backend)
    w_prev = np.zeros(L)
    pi_prev = pi0.copy()
    try:
        for k in range(1, K + 1):
            idx = k - 1
            q_k = q_node[:, idx]
            is_pd = float(q_k.sum()) > 0.0

            if is_pd or zd_method == "pd":
                b = b_for_step(data, b_const, tail_pi_idx, w_prev, pi_prev, q_k)
                pi_lb = np.maximum(pi_prev - dt, pi0)
                x_k, n_iter, phi_k, _ = _frank_wolfe(
                    lps, data, sym_M, b, w_prev, pi_prev, pi_lb, eps, max_iter_per_step
                )
                w_fw = x_k[:L]
                y_k = x_k[L : 2 * L]
                pi_fw = x_k[2 * L :]
                w_k, pi_k = _dijkstra_rewrite(
                    net, w_fw, pi_fw, 0.0, pi_idx_of, outlinks
                )
                iters.append(n_iter)
                phis.append(phi_k)
                (pd_steps if is_pd else zd_steps).append(k)
            else:
                w_k, pi_k = _dijkstra_rewrite(
                    net, w_prev, pi_prev, dt, pi_idx_of, outlinks
                )
                y_k = np.zeros(L)
                iters.append(0)
                phis.append(0.0)
                zd_steps.append(k)

            w[:, idx] = w_k
            y[:, idx] = y_k
            pi[:, idx] = pi_k
            if verbose and (k % 10 == 1 or k <= 5):
                tag = "PD" if is_pd else "ZD"
                print(
                    f"[seq] k={k:3d} {tag} iters={iters[-1]:2d}"
                    f" z={phis[-1]:.2e} max_w={w_k.max():.3f}"
                )
            w_prev = w_k
            pi_prev = pi_k
    finally:
        lps.close()

    return SequentialResult(
        w=w,
        y=y,
        pi=pi,
        pi0=pi0,
        pi_idx_of=pi_idx_of,
        pi_nodes=pi_nodes,
        iters_per_step=iters,
        phi_per_step=phis,
        pd_steps=pd_steps,
        zd_steps=zd_steps,
        backend=type(lps).__name__.replace("StepLPs", "").lower(),
    )
